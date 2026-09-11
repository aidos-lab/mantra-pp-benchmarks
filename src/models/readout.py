"""The one readout every model shares.

A model turns its input into features on nodes (graph models) or on the
simplices of every rank (simplicial models). Everything after that is
the same for all of them, so that architecture and readout can be
varied independently:

1. pool the features of each rank over the complex with ``pool_method``
   (``sum``, ``mean`` or ``max``),
2. for several ranks, combine the pooled vectors with ``rank_readout``:
   ``concat`` keeps them side by side, ``alternating`` forms

   .. math::

       e = \\sum_{r} (-1)^r \\sum_{\\sigma \\in K_r} h_\\sigma ,

   which with constant features per rank equals the Euler
   characteristic times that constant (and, unlike ``concat``, has the
   same width whatever the number of ranks),
3. apply one linear layer to the resulting embedding.

``forward`` returns ``(logits, embedding)`` where ``embedding`` is
exactly the vector the linear head consumes; the Pachner objectives act
on it.
"""

from typing import List, Sequence

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

POOL_METHODS = ("sum", "mean", "max")
RANK_READOUTS = ("concat", "alternating")


def pool(x: Tensor, batch: Tensor, reduce: str = "sum", dim_size=None) -> Tensor:
    """Pool the rows of ``x`` per complex given by ``batch``."""
    if reduce not in POOL_METHODS:
        raise ValueError(
            f"pool_method must be one of {POOL_METHODS}, got '{reduce}'"
        )
    return scatter(x, batch, dim=-2, dim_size=dim_size, reduce=reduce)


class Readout(nn.Module):
    """Pool one rank of features and apply the linear head.

    Parameters
    ----------
    in_features : int
        Width of the per-node features.
    out_features : int
        Number of outputs (classes).
    pool_method : str
        One of ``sum``, ``mean``, ``max``.
    """

    def __init__(self, in_features: int, out_features: int, pool_method: str = "sum"):
        super().__init__()
        if pool_method not in POOL_METHODS:
            raise ValueError(
                f"pool_method must be one of {POOL_METHODS}, got '{pool_method}'"
            )
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
        One of ``sum``, ``mean``, ``max``, applied per rank.
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
        if pool_method not in POOL_METHODS:
            raise ValueError(
                f"pool_method must be one of {POOL_METHODS}, got '{pool_method}'"
            )
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
            embedding = sum(
                (-1) ** r * p for r, p in enumerate(pooled)
            )
        return self.head(embedding), embedding
