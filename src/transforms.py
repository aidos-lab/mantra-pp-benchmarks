"""Benchmark-side transforms for the Calabi-Yau label columns.

These consume the extra columns of the CY parquet files (second Chern
class per prime toric divisor, triple intersection numbers; generated
by ``create_dataset.py`` in the aidos-lab/calabi-yau repository),
which ``mantra.datasets.CY`` attaches to each ``Data`` object verbatim
as numpy arrays.
"""

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform


class CreateNodeRegressionLabels(BaseTransform):
    """Node-level regression targets from a per-vertex attribute.

    Turns an attribute holding one value per triangulation vertex
    (e.g. ``c2``, the second Chern class paired with each prime toric
    divisor) into ``data.y`` of shape ``(n_vertices, 1)``, plus a
    boolean ``data.node_mask`` selecting the supervised nodes.

    In the CYTools point ordering, vertex 0 is the origin of the
    polytope: it corresponds to no divisor, and its ``c2`` entry is
    ``-sum`` of the others. It is therefore masked out by default.
    """

    def __init__(self, source="c2", mask_first=True):
        super().__init__()
        self.source = source
        self.mask_first = mask_first
        # Interface compatibility with `CreateLabels`; no label
        # indexing happens for regression targets.
        self.label_to_index = {}

    def forward(self, data):
        if self.source not in data:
            # Cached sample: the pre-transform already consumed the
            # source attribute and baked `y` and `node_mask`.
            assert "y" in data and "node_mask" in data
            return data

        values = np.asarray(data[self.source], dtype=np.float32)
        data.y = torch.from_numpy(values).view(-1, 1)

        mask = torch.ones(data.y.shape[0], dtype=torch.bool)
        if self.mask_first:
            mask[0] = False
        data.node_mask = mask

        # The ragged numpy attribute would break collation when the
        # processed cache is saved; `y` carries the values from here.
        del data[self.source]
        return data


class TensorizeAttributes(BaseTransform):
    """Convert numpy/list attributes into tensors.

    ``mantra.datasets.CY`` attaches parquet list columns (``c2``,
    ``kappa_i`` ...) as numpy arrays, which the PyG collate used for
    cache saving and batching does not handle. Converting them to
    (variable-length, 1-D or 2-D) tensors makes them collate like any
    other per-sample tensor attribute. Missing sources are skipped.
    """

    def __init__(self, sources):
        super().__init__()
        self.sources = [sources] if isinstance(sources, str) else list(sources)

    @staticmethod
    def _to_array(value):
        """Convert a (possibly nested/ragged) parquet list column.

        Parquet ``list<list<...>>`` columns arrive as an object array of
        arrays (e.g. the ``(nnz, 4)`` COO block of
        ``intersection_numbers``). ``np.asarray`` leaves those as dtype
        ``object``, which ``torch.as_tensor`` rejects, so stack the rows
        explicitly. Genuinely ragged rows have no tensor form and are
        reported rather than silently mangled.
        """
        array = np.asarray(value)
        if array.dtype != object:
            return array

        rows = [np.asarray(row) for row in value]
        if not rows:
            return np.zeros((0,), dtype=np.int64)

        widths = {row.shape for row in rows}
        if len(widths) != 1:
            raise ValueError(
                f"cannot tensorize ragged rows with shapes {sorted(widths)}"
            )
        return np.stack(rows)

    def forward(self, data):
        for source in self.sources:
            if source in data and not isinstance(data[source], torch.Tensor):
                value = torch.as_tensor(self._to_array(data[source]))
                if value.is_floating_point():
                    value = value.to(torch.float32)
                data[source] = value
        return data
