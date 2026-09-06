"""Graphormer shortest-path-distance pre-transform.

Storage strategy: dense per-graph uint8 SPD blocks, flattened to 1D for PyG
batching. This is intentionally NOT a sparse representation:

* On Hasse / barycentric-subdivided manifold graphs, BFS from any source
  reaches almost every other node within typical max_distance values, so the
  "sparse" SPD has nnz ≈ V². A coo SparseTensor with int64 row/col indices
  pays 24 bytes per nonzero, i.e. ~24× the cost of a dense uint8 block.
* Per-sample retained cost here is V² bytes (uint8). For V=5664 that is
  ~32 MB per sample versus ~770 MB with the previous SparseTensor int64
  layout. For V on the order of 30k+ this would still scale quadratically;
  in that regime, lower max_distance to make the SPD genuinely sparse and
  switch back to a sparse layout.

Encoding stored in ``data.spd`` (uint8):
  - 0                  → diagonal (self) OR unreachable / overflow
  - 1..max_distance    → actual hop distance
  - max_distance + 1   → unreachable / overflow

We use ``max_distance + 1`` (not 0) for unreachable so the model can route
the "no path within cutoff" case to its own embedding bucket. The uint8
range bounds max_distance ≤ 254.

Compute: scipy.sparse.csgraph.shortest_path with unweighted=True. It runs
in C and is dramatically faster than networkx's per-source BFS loop.
"""


import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class SPDData(Data):
    """Data subclass that batches flat dense SPD blocks correctly.

    ``spd`` is concatenated (no offset) along dim 0, yielding a 1D uint8
    tensor of length Σ n_i² in the batch.

    ``spd_n`` is concatenated (no offset) along dim 0, yielding a 1D long
    tensor of length B.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ("spd", "spd_n"):
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key in ("spd", "spd_n"):
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class GraphormerSPDPreTransform(BaseTransform):
    """Compute all-pairs shortest paths and store them as dense uint8 blocks.

    Parameters
    ----------
    max_distance : int
        Distances strictly larger than this map to the unreachable bucket.
        Must satisfy 1 <= max_distance <= 254 (uint8 range).
    undirected : bool
        If True, treat ``edge_index`` as defining an undirected graph for
        BFS purposes.
    attr_name : str
        Name of the SPD attribute on the output ``Data``. The companion
        size attribute is ``f"{attr_name}_n"``.
    """

    def __init__(
        self,
        max_distance: int = 16,
        undirected: bool = True,
        attr_name: str = "spd",
    ):
        super().__init__()
        if max_distance <= 0:
            raise ValueError(f"max_distance must be positive, got {max_distance}")
        if max_distance + 1 > 255:
            raise ValueError(
                f"max_distance + 1 must fit in uint8 (<= 255); "
                f"got max_distance={max_distance}"
            )
        self.max_distance = max_distance
        self.undirected = undirected
        self.attr_name = attr_name
        self.size_attr_name = f"{attr_name}_n"

    def _resolve_num_nodes(self, data: Data) -> int:
        """Best-effort num_nodes resolution that doesn't depend on PyG warnings."""
        n = getattr(data, "num_nodes", None)
        if n is not None:
            return int(n)
        if hasattr(data, "x") and data.x is not None:
            return int(data.x.size(0))
        if hasattr(data, "edge_index") and data.edge_index is not None and data.edge_index.numel() > 0:
            return int(data.edge_index.max().item()) + 1
        return 0

    def _compute_dense_spd(self, edge_index: torch.Tensor, n: int) -> np.ndarray:
        """Return an (n, n) uint8 SPD matrix with the encoding documented above."""
        unreachable = self.max_distance + 1

        if n == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        if edge_index.numel() == 0:
            out = np.full((n, n), unreachable, dtype=np.uint8)
            np.fill_diagonal(out, 0)
            return out

        row = edge_index[0].detach().cpu().numpy()
        col = edge_index[1].detach().cpu().numpy()
        data_arr = np.ones(row.shape[0], dtype=np.int8)
        adj = csr_matrix((data_arr, (row, col)), shape=(n, n))

        D = shortest_path(adj, directed=not self.undirected, unweighted=True)
        # D is float64; np.inf for unreachable, integer-valued otherwise.
        D = np.where(np.isfinite(D), D, unreachable)
        D = np.where(D > self.max_distance, unreachable, D)
        # Diagonal is already 0 from scipy. uint8 cast preserves that.
        return D.astype(np.uint8)

    def forward(self, data: Data) -> Data:
        if not hasattr(data, "edge_index") or data.edge_index is None:
            return data

        n = self._resolve_num_nodes(data)
        edge_index = data.edge_index

        D = self._compute_dense_spd(edge_index, n)

        spd_flat = torch.from_numpy(np.ascontiguousarray(D).reshape(-1).copy())
        # torch.from_numpy on uint8 yields a uint8 tensor; explicit for clarity.
        spd_flat = spd_flat.to(torch.uint8)
        size_tensor = torch.tensor([n], dtype=torch.long)

        setattr(data, self.attr_name, spd_flat)
        setattr(data, self.size_attr_name, size_tensor)

        # Promote to SPDData so PyG batching uses the right __cat_dim__/__inc__.
        object.__setattr__(data, "__class__", SPDData)
        return data

