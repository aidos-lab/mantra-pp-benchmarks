from collections import defaultdict

import torch
from torch_geometric.transforms import BaseTransform


class VertexDegree(BaseTransform):
    """The degree of every vertex of the triangulation, as ``[n, 1]`` features.

    Computed from the top simplices, so it does not need an
    ``edge_index`` and serves the simplicial-complex pipeline (where
    ``PropagateConvexComb`` then spreads it to the higher ranks) as a
    label-free vertex feature. Rows follow the vertex labels ``1..n``.
    """

    def __init__(self, dst: str = "degree"):
        super().__init__()
        self.dst = dst

    def forward(self, data):
        neighbours = defaultdict(set)
        for simplex in data.triangulation:
            for v in simplex:
                neighbours[v].update(simplex)
        n = max(neighbours)
        degree = torch.zeros(n, 1)
        for v, adjacent in neighbours.items():
            degree[v - 1, 0] = len(adjacent) - 1
        data[self.dst] = degree
        return data
