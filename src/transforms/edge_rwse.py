"""Edge-level random walk structural encoding (EdgeRWSE).

 "Facilitating Graph Neural Networks with Random Walk
on Simplicial Complexes" (NeurIPS 2023)"
"""
import torch
from torch_geometric.transforms import BaseTransform


class EdgeRWSE(BaseTransform):
    def __init__(self, walk_length: int = 8, attr_name: str = "edge_rwse"):
        super().__init__()
        assert walk_length >= 1
        self.walk_length = walk_length
        self.attr_name = attr_name

    def forward(self, data):
        assert "edge_index" in data, "No edge index in data"
        edge_index = data.edge_index

        lo = torch.minimum(edge_index[0], edge_index[1])
        hi = torch.maximum(edge_index[0], edge_index[1])
        uniq, inverse = torch.unique(
            torch.stack([lo, hi]), dim=1, return_inverse=True
        )
        m = uniq.shape[1]
        n = int(edge_index.max().item()) + 1

        C = torch.zeros(n, m)
        cols = torch.arange(m)
        C[uniq[0], cols] = 1.0
        C[uniq[1], cols] = 1.0
        A = C.T @ C - 2.0 * torch.eye(m)

        deg = A.sum(dim=1)
        P = torch.zeros(m, m)
        has_nb = deg > 0
        P[has_nb] = A[has_nb] / deg[has_nb].unsqueeze(1)
        P[~has_nb, ~has_nb] = 1.0

        feats = torch.empty(m, self.walk_length)
        M = P
        feats[:, 0] = M.diagonal()
        for k in range(1, self.walk_length):
            M = M @ P
            feats[:, k] = M.diagonal()

        data[self.attr_name] = feats[inverse]
        return data
