"""Per-batch randomisation of the positional node identities.

The moment-curve features are a positional ID tied to the vertex order
of the release, which is not topological, so a model may fit that
labelling. Re-drawing the assignment for every batch removes the
shortcut. On graphs the ID columns of ``x`` are permuted among the
nodes of each graph; on complexes the ID columns of ``x_0`` are
permuted among the vertices of each complex and the barycentric
higher-rank features are recomputed with the unsigned boundary
matrices, ``x_1 = |B_1|^T x_0 / 2`` and ``x_2 = |B_2|^T x_1 / 3``.
"""

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import scatter


def _within_group_permutation(group: Tensor, generator=None) -> Tensor:
    """Random permutation that keeps every row inside its (sorted) group.

    Sorting ``group + U(0, 1)`` orders rows by group with a random order
    inside each group.
    """
    noise = torch.rand(group.shape[0], generator=generator, device=group.device)
    return torch.argsort(group.double() + noise.double())


def _propagate(x_lower: Tensor, idx: Tensor, val: Tensor, n_upper: int, k: int) -> Tensor:
    """``|B|^T x_lower / k`` from the COO boundary matrix ``(idx, val)``."""
    row, col = idx
    return scatter(
        x_lower[row] * val.abs().unsqueeze(-1), col, dim=0, dim_size=n_upper, reduce="sum"
    ) / k


def permute_ids(batch, n_columns: Optional[int] = None, generator=None):
    """Randomise the positional IDs of a batch in place; returns it.

    Parameters
    ----------
    batch : Batch
        Graph batch (``x``, ``batch``) or collated complex batch
        (``x_0``, ``batch_0``, ``incidence_1_idx/_val``,
        ``incidence_2_idx/_val``).
    n_columns : int or None
        Number of leading feature columns that are IDs; ``None`` means
        all columns. Graphs only: on the Levi graph the one-hot node
        type follows the moment curve.
    generator : torch.Generator, optional
        Source of the permutations.
    """
    if "x_0" in batch:
        if n_columns is not None:
            raise ValueError(
                "n_columns is only defined for graphs; complex features "
                "are recomputed from x_0 as a whole"
            )
        perm = _within_group_permutation(batch["batch_0"], generator)
        x_0 = batch["x_0"][perm]
        batch["x_0"] = x_0
        x_1 = _propagate(
            x_0, batch["incidence_1_idx"], batch["incidence_1_val"], batch["x_1"].shape[0], 2
        )
        batch["x_1"] = x_1
        if "x_2" in batch:
            batch["x_2"] = _propagate(
                x_1, batch["incidence_2_idx"], batch["incidence_2_val"], batch["x_2"].shape[0], 3
            )
        return batch
    perm = _within_group_permutation(batch.batch, generator)
    if n_columns is None:
        batch.x = batch.x[perm]
    else:
        batch.x = torch.cat([batch.x[perm, :n_columns], batch.x[:, n_columns:]], dim=-1)
    return batch
