import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_sparse import SparseTensor


class SPDData(Data):
    """Data subclass that block-diagonally batches a square SPD SparseTensor.

    PyG's default SparseTensor batching row-concatenates only, which leaves
    column indices un-offset. For a graph-local square matrix we need both
    axes offset by num_nodes.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "spd":
            return torch.tensor([[self.num_nodes], [self.num_nodes]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "spd":
            return (0, 1)
        return super().__cat_dim__(key, value, *args, **kwargs)


class GraphormerSPDPreTransform:
    def __init__(self, max_distance: int = 16, undirected: bool = True, attr_name: str = "spd"):
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        self.max_distance = max_distance
        self.undirected = undirected
        self.attr_name = attr_name

    def _compute_sparse_spd(self, data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-pairs shortest paths via V-times BFS with depth cutoff.

        Returns (row, col, val) LongTensors with only pairs where
        0 < d <= max_distance. Diagonal (d == 0) and overflow (d > max_distance)
        are implicit.
        """
        if data.edge_index.numel() == 0:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, empty

        g = to_networkx(data, to_undirected=self.undirected)

        rows, cols, vals = [], [], []
        for source, lengths in nx.all_pairs_shortest_path_length(g, cutoff=self.max_distance):
            for target, d in lengths.items():
                if d > 0:
                    rows.append(source)
                    cols.append(target)
                    vals.append(d)

        if not rows:
            empty = torch.empty(0, dtype=torch.long)
            return empty, empty, empty

        return (
            torch.tensor(rows, dtype=torch.long),
            torch.tensor(cols, dtype=torch.long),
            torch.tensor(vals, dtype=torch.long),
        )

    def __call__(self, data):
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        n = data.num_nodes
        row, col, val = self._compute_sparse_spd(data) if n > 0 else (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )

        spd = SparseTensor(
            row=row, col=col, value=val, sparse_sizes=(n, n)
        )
        setattr(data, self.attr_name, spd)
        object.__setattr__(data, "__class__", SPDData)
        return data
