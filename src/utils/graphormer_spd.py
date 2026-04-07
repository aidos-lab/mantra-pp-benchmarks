import torch
import numpy as np
import networkx as nx
from torch_geometric.utils import to_networkx


class GraphormerSPDPreTransform:
    def __init__(self, max_distance: int = 16, undirected: bool = True, attr_name: str = "spatial_pos"):
        self.max_distance = max_distance
        self.undirected = undirected
        self.attr_name = attr_name

    def _compute_spd(self, data) -> torch.Tensor:
        """All-pairs shortest paths via V-times BFS with depth cutoff."""
        num_nodes = data.num_nodes
        unreachable = self.max_distance + 1

        if data.edge_index.numel() == 0:
            # empty edge-list guard rail
            dist = torch.full((num_nodes, num_nodes), unreachable, dtype=torch.long)
            dist.fill_diagonal_(0)
            return dist

        g = to_networkx(data, to_undirected=self.undirected)

        dist = np.full((num_nodes, num_nodes), unreachable, dtype=np.int64)
        for source, lengths in nx.all_pairs_shortest_path_length(g, cutoff=self.max_distance):
            targets, dists = zip(*lengths.items()) if lengths else ((), ())
            dist[source, list(targets)] = list(dists)

        return torch.from_numpy(dist)

    def __call__(self, data):
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        if data.num_nodes == 0:
            # empty node list guard rail
            setattr(data, self.attr_name, torch.zeros(0, dtype=torch.long))
            data.spatial_pos_n = torch.zeros(1, dtype=torch.long)
            return data

        dist = self._compute_spd(data)
        # Store as flattened 1D tensor so PyG can batch via concatenation.
        # spatial_pos_n stores the node count to reshape back to (n, n).
        setattr(data, self.attr_name, dist.reshape(-1))
        data.spatial_pos_n = torch.tensor([data.num_nodes], dtype=torch.long)
        return data
