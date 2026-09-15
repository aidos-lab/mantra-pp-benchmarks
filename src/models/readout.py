"""The readout every model shares: pool per complex, then one linear head.

Graph models pool their node features; simplicial models pool every
rank and combine the ranks with ``rank_readout``, either ``concat`` or
``alternating``, i.e. ``sum_r (-1)^r pooled_r``, which for constant
features equals the Euler characteristic times that constant.
``forward`` returns ``(logits, embedding)``, ``embedding`` being the
vector the head consumes; the Pachner objectives act on it.
"""

from typing import List, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

POOL_METHODS = ("sum", "mean")
RANK_READOUTS = ("concat", "alternating")


def pool(x: Tensor, batch: Tensor, reduce: str = "sum", dim_size=None) -> Tensor:
    """Pool the rows of ``x`` per complex given by ``batch``."""
    if reduce not in POOL_METHODS:
        raise ValueError(
            f"pool_method must be one of {POOL_METHODS}, got '{reduce}'"
        )
    return scatter(x, batch, dim=-2, dim_size=dim_size, reduce=reduce)


class Readout(nn.Module):
    """Pool one rank of features and apply the linear head."""

    def __init__(self, in_features: int, out_features: int, pool_method: str = "sum"):
        super().__init__()
        self.pool_method = pool_method
        self.head = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor, batch: Tensor):
        embedding = pool(x, batch, self.pool_method)
        return self.head(embedding), embedding


class RankReadout(nn.Module):
    """Pool every rank, combine the ranks and apply the linear head.

    Parameters
    ----------
    in_features : sequence of int
        Width of the features of each rank, ``r = 0, 1, ...``.
    out_features : int
        Number of outputs (classes).
    pool_method : str
        Pooling applied per rank, see :func:`pool`.
    rank_readout : str
        ``concat`` or ``alternating``; the latter needs every rank to
        have the same width.
    """

    def __init__(
        self,
        in_features: Sequence[int],
        out_features: int,
        pool_method: str = "sum",
        rank_readout: str = "concat",
    ):
        super().__init__()
        if rank_readout not in RANK_READOUTS:
            raise ValueError(
                f"rank_readout must be one of {RANK_READOUTS}, got '{rank_readout}'"
            )
        in_features = list(in_features)
        if rank_readout == "alternating" and len(set(in_features)) != 1:
            raise ValueError(
                "the alternating readout needs the same width on every "
                f"rank, got {in_features}"
            )
        self.pool_method = pool_method
        self.rank_readout = rank_readout
        width = sum(in_features) if rank_readout == "concat" else in_features[0]
        self.head = nn.Linear(width, out_features)

    def forward(self, xs: List[Tensor], batches: List[Tensor]):
        # One row per complex on every rank, even if a complex had no
        # simplex of some rank.
        n_complexes = int(max(b.max() for b in batches)) + 1
        pooled = [
            pool(x, b, self.pool_method, dim_size=n_complexes)
            for x, b in zip(xs, batches)
        ]
        if self.rank_readout == "concat":
            embedding = torch.cat(pooled, dim=-1)
        else:
            embedding = sum((-1) ** r * p for r, p in enumerate(pooled))
        return self.head(embedding), embedding
