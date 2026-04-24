import torch
import networkx as nx
from torch_geometric.utils import to_networkx


class GraphormerSPDPreTransform:
    def __init__(self, max_distance: int = 16, undirected: bool = True, attr_prefix: str = "spd"):
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        self.max_distance = max_distance
        self.undirected = undirected
        self.attr_prefix = attr_prefix

    def _compute_sparse_spd(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        """All-pairs shortest paths via V-times BFS with depth cutoff, COO output.

        Stores only pairs with 0 < d <= max_distance; d == 0 (diagonal) and
        d > max_distance (overflow) are implicit.
        """
        if data.edge_index.numel() == 0:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        g = to_networkx(data, to_undirected=self.undirected)

        srcs, dsts, dists = [], [], []
        for source, lengths in nx.all_pairs_shortest_path_length(g, cutoff=self.max_distance):
            for target, d in lengths.items():
                if d > 0:
                    srcs.append(source)
                    dsts.append(target)
                    dists.append(d)

        if not srcs:
            return (
                torch.empty(2, 0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )

        edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
        edge_attr = torch.tensor(dists, dtype=torch.long)
        return edge_index, edge_attr

    def __call__(self, data):
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        ei_key = f"{self.attr_prefix}_edge_index"
        ea_key = f"{self.attr_prefix}_edge_attr"

        if data.num_nodes == 0:
            setattr(data, ei_key, torch.empty(2, 0, dtype=torch.long))
            setattr(data, ea_key, torch.empty(0, dtype=torch.long))
            return data

        edge_index, edge_attr = self._compute_sparse_spd(data)
        setattr(data, ei_key, edge_index)
        setattr(data, ea_key, edge_attr)
        return data
