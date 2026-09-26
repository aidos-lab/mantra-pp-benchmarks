from itertools import combinations

import torch
from torch_geometric.transforms import BaseTransform


class VertexDegree(BaseTransform):
    """The degree of every vertex of the triangulation, as ``[n, 1]`` features.

    Computed from the top simplices, so it needs no ``edge_index`` and
    serves the simplicial-complex pipeline (where ``PropagateConvexComb``
    spreads it to the higher ranks) as a label-free vertex feature.
    Rows follow the vertex labels ``1..n``.
    """

    def __init__(self, dst: str = "degree"):
        super().__init__()
        self.dst = dst

    def forward(self, data):
        edges = {frozenset(e) for s in data.triangulation for e in combinations(s, 2)}
        ends = torch.tensor([v for e in edges for v in e]) - 1
        data[self.dst] = torch.bincount(ends).float().unsqueeze(1)
        return data
