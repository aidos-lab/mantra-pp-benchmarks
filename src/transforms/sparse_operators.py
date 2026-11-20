"""Store simplicial operators as index and value tensors in the cache."""

import re
from typing import Sequence

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

from src.utils.data_loading import SimplicialData

OPERATOR = re.compile(r"^(incidence|\w*adjacency|\w*laplacian)_(\d+)$")
RANK_FEATURES = re.compile(r"^x_(\d+)$")


class SparseOperatorsToCOO(BaseTransform):
    """Replace each kept operator `name` by `name_idx` and `name_val`.

    The representations leave one sparse tensor per sample and operator,
    which `InMemoryDataset` cannot merge: the cache then holds a Python
    object per sample and operator, every loader worker copies that heap
    on first touch, and `collate_simplicial` converts sample by sample.
    The index and value tensors of a `SimplicialData` are concatenated
    into shared storage instead and batch with `Batch.from_data_list`.

    Parameters
    ----------
    keep : sequence of str
        Operators to convert; every other operator is dropped. A sample
        without one of them raises.
    max_rank : int
        Highest rank whose features `x_k` stay in the sample. Batching
        offsets an operator's indices by the feature counts of its
        ranks, so every kept operator must act within `max_rank`.
    """

    def __init__(self, keep: Sequence[str], max_rank: int = 2) -> None:
        super().__init__()
        self.keep = tuple(keep)
        self.max_rank = max_rank
        for name in self.keep:
            match = OPERATOR.match(name)
            if match is None or int(match.group(2)) > max_rank:
                raise ValueError(
                    f"{name!r} is not an operator of rank <= {max_rank}"
                )

    def forward(self, data: Data) -> SimplicialData:
        out = SimplicialData()
        for key, value in data.items():
            rank = RANK_FEATURES.match(key)
            if OPERATOR.match(key) or (
                rank and int(rank.group(1)) > self.max_rank
            ):
                continue
            out[key] = value
        for name in self.keep:
            operator = data[name].coalesce()
            out[f"{name}_idx"] = operator.indices()
            out[f"{name}_val"] = operator.values().float()
        return out
