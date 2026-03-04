import torch


class GraphormerSPDPreTransform:
    def __init__(self, max_distance: int = 16, undirected: bool = True, attr_name: str = "spatial_pos"):
        self.max_distance = max_distance
        self.undirected = undirected
        self.attr_name = attr_name

    def _compute_spd(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        ei = edge_index.detach().to("cpu")
        adj = [[] for _ in range(num_nodes)]
        if ei.numel() > 0:
            src = ei[0].tolist()
            dst = ei[1].tolist()
            for u, v in zip(src, dst):
                if 0 <= u < num_nodes and 0 <= v < num_nodes:
                    adj[u].append(v)
                    if self.undirected:
                        adj[v].append(u)

        unreachable = self.max_distance + 1
        dist = torch.full((num_nodes, num_nodes), unreachable, dtype=torch.long)

        from collections import deque

        for s in range(num_nodes):
            dist[s, s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = int(dist[s, u].item())
                if du >= self.max_distance:
                    continue
                for v in adj[u]:
                    if dist[s, v] == unreachable:
                        dist[s, v] = du + 1
                        q.append(v)

        return dist

    def __call__(self, data):
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        if getattr(data, "num_nodes", None) is not None:
            num_nodes = int(data.num_nodes)
        elif hasattr(data, "x") and data.x is not None:
            num_nodes = int(data.x.size(0))
        else:
            num_nodes = int(data.edge_index.max().item()) + 1 if data.edge_index.numel() > 0 else 0

        if num_nodes == 0:
            setattr(data, self.attr_name, [])
            return data

        dist = self._compute_spd(data.edge_index, num_nodes)
        # Store as a list to avoid PyG concatenation for variable-sized graphs.
        setattr(data, self.attr_name, dist.tolist())
        return data

