"""
Graphormer implementation for graph-level prediction.
"""

import torch
import torch._dynamo as dynamo
from pydantic import BaseModel
from torch import nn
from torch_geometric.utils import degree, to_dense_batch


class GraphormerConfig(BaseModel):
    hidden_channels: int = 128
    num_hidden_layers: int = 4
    num_node_features: int = 1
    out_channels: int = 5
    num_heads: int = 8
    dropout: float = 0.1
    max_degree: int = 32
    max_distance: int = 16  # max shortest-path distance bucket for spatial encoding
    use_spatial_encoding: bool = True  # implemented as a gate (no control-flow in forward)
    ffn_expansion_ratio: int = 4  # FFN hidden size = hidden_channels * ratio
    spd_undirected: bool = True  # treat edges as undirected when computing SPD


class GraphormerLayer(nn.Module):
    """
    Single Graphormer transformer layer with multi-head self-attention and FFN.

    Architecture:
        1. Layer Norm -> Multi-Head Attention -> Residual
        2. Layer Norm -> FFN -> Residual
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_expansion_ratio: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads for Graphormer."
            )
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        ffn_hidden_dim = hidden_dim * ffn_expansion_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, _ = x.shape
        x = x.view(bsz, n_nodes, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.norm_attn(x)

        q = self._reshape_heads(self.q_proj(x_norm))
        k = self._reshape_heads(self.k_proj(x_norm))
        v = self._reshape_heads(self.v_proj(x_norm))

        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / (self.head_dim**0.5)
        attn_scores = attn_scores + attn_bias
        attn_scores = attn_scores.masked_fill(
            pad_mask[:, None, None, :], float("-inf")
        )
        attn = torch.softmax(attn_scores, dim=-1)
        # Handle NaN from all-masked rows (though should be rare with proper input)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(x.size(0), x.size(1), self.hidden_dim)
        out = self.out_proj(out)
        out = self.dropout(out)
        x = residual + out

        residual = x
        x = self.norm_ffn(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x
        return x


class Graphormer(nn.Module):
    """
    Graphormer: Transformer architecture for graph-level prediction.

    This implementation follows the original Graphormer paper with:
    - Multi-head self-attention with structural encodings
    - Degree-based centrality encodings (in-degree and out-degree)
    - Optional graph token for global representation

    The model processes graphs by:
    1. Converting sparse graphs to dense batched representations
    2. Adding structural encodings (degree embeddings)
    3. Processing through transformer layers
    4. Pooling to graph-level representation (via graph token or mean pooling)

    Args:
        config: GraphormerConfig object with model hyperparameters

    Input:
        batch: PyTorch Geometric Batch object with:
            - x: Node features [num_nodes, num_node_features]
            - edge_index: Edge connectivity [2, num_edges]
            - batch: Batch assignment [num_nodes]

    Output:
        Graph-level predictions [batch_size, out_channels]

    Complexity:
        - Time: O(batch_size * max_nodes^2 * hidden_dim)
        - Space: O(batch_size * max_nodes^2 * num_heads)

    Note:
        For large graphs, the dense representation can be memory-intensive.
        Consider using sparse attention variants for graphs with >1000 nodes.
    """
    def __init__(self, config: GraphormerConfig):
        super().__init__()
        if config.hidden_channels % config.num_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads for Graphormer."
            )
        self.hidden_dim = config.hidden_channels
        self.num_heads = config.num_heads
        self.use_spatial_encoding = config.use_spatial_encoding
        self.spd_undirected = config.spd_undirected

        self.input_proj = nn.Linear(
            config.num_node_features, config.hidden_channels
        )
        self.in_degree_emb = nn.Embedding(
            config.max_degree + 1, config.hidden_channels
        )
        self.out_degree_emb = nn.Embedding(
            config.max_degree + 1, config.hidden_channels
        )
        self.max_distance = config.max_distance
        # Spatial encoding is an attention bias indexed by shortest-path distance.
        # We allocate max_distance+2 buckets: [0..max_distance] plus one bucket for "unreachable/overflow".
        self.spatial_pos_emb = nn.Embedding(
            config.max_distance + 2, config.num_heads
        )

        self.layers = nn.ModuleList(
            [
                GraphormerLayer(
                    hidden_dim=config.hidden_channels,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_expansion_ratio=config.ffn_expansion_ratio,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.dropout = nn.Dropout(config.dropout)

        self.final_linear = nn.Linear(config.hidden_channels, config.out_channels)

    @staticmethod
    def _compute_spd_single(
        edge_index: torch.Tensor,
        num_nodes: int,
        max_dist: int,
        device: torch.device,
        undirected: bool,
    ) -> torch.Tensor:
        """Compute clipped shortest-path distances for one graph.

        Returns a [num_nodes, num_nodes] LongTensor with values in:
          - 0..max_dist for reachable pairs
          - max_dist+1 for unreachable or > max_dist
        """
        # Build adjacency list on CPU for fast Python BFS; move result back to device.
        # edge_index is expected to be [2, E] with nodes in [0, num_nodes).
        ei = edge_index.detach().to("cpu")
        adj = [[] for _ in range(num_nodes)]
        if ei.numel() > 0:
            src = ei[0].tolist()
            dst = ei[1].tolist()
            for u, v in zip(src, dst):
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    adj[u].append(v)
                    if undirected:
                        adj[v].append(u)

        unreachable = max_dist + 1
        dist = torch.full((num_nodes, num_nodes), unreachable, dtype=torch.long)

        from collections import deque

        for s in range(num_nodes):
            dist[s, s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = int(dist[s, u].item())
                if du >= max_dist:
                    continue
                for v in adj[u]:
                    if dist[s, v] == unreachable:
                        dist[s, v] = du + 1
                        q.append(v)

        return dist.to(device)

    def _pad_spd_from_precomputed(
        self,
        spatial_pos,
        batch_size: int,
        max_nodes: int,
        device: torch.device,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        unreachable = self.max_distance + 1
        spd = torch.full(
            (batch_size, max_nodes, max_nodes),
            unreachable,
            dtype=torch.long,
            device=device,
        )

        if isinstance(spatial_pos, torch.Tensor):
            spd_tensor = spatial_pos.to(device=device, dtype=torch.long)
            if spd_tensor.dim() == 2:
                spd_tensor = spd_tensor.unsqueeze(0)
            spd[:, : spd_tensor.size(1), : spd_tensor.size(2)] = spd_tensor
        else:
            for idx, spd_item in enumerate(spatial_pos):
                if idx >= batch_size:
                    break
                spd_tensor = torch.as_tensor(spd_item, dtype=torch.long, device=device)
                if spd_tensor.dim() < 2:
                    continue
                n = spd_tensor.size(0)
                spd[idx, :n, :n] = spd_tensor

        spd = spd.clamp(max=unreachable)
        spd = spd.masked_fill(node_mask[:, :, None], unreachable)
        spd = spd.masked_fill(node_mask[:, None, :], unreachable)
        return spd

    @dynamo.disable
    def _compute_spd_batch(
        self,
        edge_index: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_size: int,
        max_nodes: int,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute [B, max_nodes, max_nodes] SPD matrices aligned with to_dense_batch order.

        node_mask is [B, max_nodes] with True for padding positions.
        """
        device = batch_idx.device
        unreachable = self.max_distance + 1
        spd = torch.full(
            (batch_size, max_nodes, max_nodes),
            unreachable,
            dtype=torch.long,
            device=device,
        )

        # For each graph, extract its nodes, remap to local indices, compute SPD, then place into spd.
        for g in range(batch_size):
            node_ids = (batch_idx == g).nonzero(as_tuple=False).view(-1)
            n = int(node_ids.numel())
            if n == 0:
                continue

            # Map global node id -> local index
            mapping = {int(node_ids[i].item()): i for i in range(n)}

            # Select edges fully inside this graph and remap
            if edge_index.numel() == 0:
                ei_local = torch.empty((2, 0), dtype=torch.long, device=device)
            else:
                src = edge_index[0]
                dst = edge_index[1]
                mask = (batch_idx[src] == g) & (batch_idx[dst] == g)
                src_g = src[mask].tolist()
                dst_g = dst[mask].tolist()
                if len(src_g) == 0:
                    ei_local = torch.empty((2, 0), dtype=torch.long, device=device)
                else:
                    src_l = [mapping[int(u)] for u in src_g]
                    dst_l = [mapping[int(v)] for v in dst_g]
                    ei_local = torch.tensor([src_l, dst_l], dtype=torch.long, device=device)

            dist_g = self._compute_spd_single(
                ei_local, n, self.max_distance, device, self.spd_undirected
            )

            spd[g, :n, :n] = dist_g

        # Make sure padded rows/cols are set to unreachable
        spd = spd.masked_fill(node_mask[:, :, None], unreachable)
        spd = spd.masked_fill(node_mask[:, None, :], unreachable)
        return spd

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch

        x = self.input_proj(x)

        out_deg = degree(
            edge_index[0], num_nodes=x.size(0), dtype=torch.long
        ).clamp(max=self.out_degree_emb.num_embeddings - 1)
        in_deg = degree(
            edge_index[1], num_nodes=x.size(0), dtype=torch.long
        ).clamp(max=self.in_degree_emb.num_embeddings - 1)
        x = x + self.in_degree_emb(in_deg) + self.out_degree_emb(out_deg)

        x_dense, node_mask = to_dense_batch(x, batch_idx)
        batch_size, max_nodes, _ = x_dense.shape

        node_mask = ~node_mask

        if self.use_spatial_encoding:
            if hasattr(batch, "spatial_pos") and batch.spatial_pos is not None:
                spd = self._pad_spd_from_precomputed(
                    batch.spatial_pos, batch_size, max_nodes, x_dense.device, node_mask
                )
            else:
                spd = self._compute_spd_batch(
                    edge_index=edge_index,
                    batch_idx=batch_idx,
                    batch_size=batch_size,
                    max_nodes=max_nodes,
                    node_mask=node_mask,
                )
            # spatial_pos_emb returns [B, N, N, H]; permute to [B, H, N, N]
            attn_bias = self.spatial_pos_emb(spd).permute(0, 3, 1, 2).contiguous()
        else:
            attn_bias = x_dense.new_zeros(
                (batch_size, self.num_heads, max_nodes, max_nodes)
            )

        x_dense = self.dropout(x_dense)

        for layer in self.layers:
            x_dense = layer(x_dense, node_mask, attn_bias)

        valid_nodes = ~node_mask
        summed = (x_dense * valid_nodes.unsqueeze(-1)).sum(dim=1)
        denom = valid_nodes.sum(dim=1).clamp(min=1).unsqueeze(-1)
        out = summed / denom

        out = self.dropout(out)
        out = self.final_linear(out)
        return out
