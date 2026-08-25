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
    n_layers: int = 4
    in_channels: int = 1
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
        hidden_channels: Hidden feature size for each layer.
        n_layers: Number of transformer layers.
        in_channels: Input node feature size.
        out_channels: Output feature size.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_degree: Max degree for degree embeddings.
        max_distance: Max shortest-path distance bucket for spatial encoding.
        use_spatial_encoding: Whether to add spatial attention bias.
        ffn_expansion_ratio: Expansion ratio for FFN hidden size.
        spd_undirected: Treat edges as undirected when computing SPD.

    Input:
        batch: PyTorch Geometric Batch object with:
            - x: Node features [num_nodes, in_channels]
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

    def __init__(
            self,
            hidden_channels: int = 128,
            n_layers: int = 4,
            in_channels: int = 1,
            out_channels: int = 5,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_degree: int = 32,
            max_distance: int = 16,
            use_spatial_encoding: bool = True,
            ffn_expansion_ratio: int = 4,
            spd_undirected: bool = True,
            pool_method: str = 'mean',
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError(
                "hidden_channels must be divisible by num_heads for Graphormer."
            )
        assert pool_method in ('mean', 'sum'), " Unknwon pooling methods"
        self.pool_method = pool_method
        self.hidden_dim = hidden_channels
        self.num_heads = num_heads
        self.use_spatial_encoding = use_spatial_encoding
        self.spd_undirected = spd_undirected
        self.max_distance = max_distance

        self.input_proj = nn.Linear(
            in_channels, hidden_channels
        )
        self.in_degree_emb = nn.Embedding(
            max_degree + 1, hidden_channels
        )
        self.out_degree_emb = nn.Embedding(
            max_degree + 1, hidden_channels
        )
        # Spatial encoding is an attention bias indexed by shortest-path distance.
        # We allocate max_distance+2 buckets: [0..max_distance] plus one bucket for "unreachable/overflow".
        self.spatial_pos_emb = nn.Embedding(
            max_distance + 2, num_heads
        )

        self.layers = nn.ModuleList(
            [
                GraphormerLayer(
                    hidden_dim=hidden_channels,
                    num_heads=num_heads,
                    dropout=dropout,
                    ffn_expansion_ratio=ffn_expansion_ratio,
                )
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.final_linear = nn.Linear(hidden_channels, out_channels)

    def _unpack_dense_spd(
        self,
        spd_flat: torch.Tensor,
        spd_n: torch.Tensor,
        batch_size: int,
        max_nodes: int,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct padded (B, max_nodes, max_nodes) long SPD from flat uint8.

        spd_flat: 1D uint8 tensor of length Σ n_i² (per-graph dense SPD blocks
            concatenated row-major). Encoding: 0 on the diagonal, 1..max_distance
            for actual hop distances, max_distance+1 for unreachable/overflow.
        spd_n: 1D LongTensor of shape (B,) — node count per graph.

        Padding cells are filled with the unreachable bucket so the embedding
        lookup is well-defined; ``node_mask`` is then re-applied to be safe.
        """
        device = spd_flat.device
        unreachable = self.max_distance + 1
        out = torch.full(
            (batch_size, max_nodes, max_nodes),
            unreachable,
            dtype=torch.long,
            device=device,
        )

        sizes_sq = (spd_n * spd_n).tolist()
        splits = torch.split(spd_flat, sizes_sq)
        ns = spd_n.tolist()
        for idx, (flat, n) in enumerate(zip(splits, ns)):
            if n > 0:
                out[idx, :n, :n] = flat.view(n, n).to(torch.long)

        out = out.clamp(max=unreachable)
        out = out.masked_fill(node_mask[:, :, None], unreachable)
        out = out.masked_fill(node_mask[:, None, :], unreachable)
        return out

    @dynamo.disable
    def _prepare_dense_inputs(self, batch):
        """Prepare dense features and attention bias outside Dynamo tracing."""
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
            if not (hasattr(batch, "spd") and hasattr(batch, "spd_n")):
                raise ValueError(
                    "use_spatial_encoding=True but batch is missing 'spd' / 'spd_n'. "
                    "Add GraphormerSPDPreTransform to your dataset pre_transform; "
                    "it stores SPD as a dense uint8 flat block per graph."
                )
            spd = self._unpack_dense_spd(
                batch.spd,
                batch.spd_n,
                batch_size,
                max_nodes,
                node_mask,
            )
            attn_bias = self.spatial_pos_emb(spd).permute(0, 3, 1, 2).contiguous()
        else:
            attn_bias = x_dense.new_zeros(
                (batch_size, self.num_heads, max_nodes, max_nodes)
            )

        return x_dense, node_mask, attn_bias

    def forward(self, batch):
        x_dense, node_mask, attn_bias = self._prepare_dense_inputs(batch)

        x_dense = self.dropout(x_dense)

        for layer in self.layers:
            x_dense = layer(x_dense, node_mask, attn_bias)

        valid_nodes = ~node_mask
        out = (x_dense * valid_nodes.unsqueeze(-1)).sum(dim=1)
        if self.pool_method == 'mean':
            denom = valid_nodes.sum(dim=1).clamp(min=1).unsqueeze(-1)
            out = out / denom

        out = self.dropout(out)
        out = self.final_linear(out)
        return out
