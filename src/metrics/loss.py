"""Training objectives: the task loss, optionally with a Pachner term.

Every objective maps ``(logits, embedding, batch)`` to the total loss
and a dictionary of detached component tensors for logging. Same-walk
pairs are recovered from the ``walk_base`` and ``walk_step``
attributes that :class:`PachnerWalkDataset` attaches to its entries;
see :func:`same_walk_mask`.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

WALK_ATTRIBUTES = ("walk_base", "walk_step")

NO_WALK_MESSAGE = (
    "{name} needs same-walk pairs, but the batch has no "
    "`walk_base`/`walk_step` attributes. Set dataset.walk_length > 0 "
    "(and trainer.sampler=walks to keep the snapshots of a walk in one "
    "batch)."
)


def same_walk_mask(batch, name: str) -> Tensor:
    """Boolean ``[B, B]`` mask of the same-walk pairs of a batch.

    Two entries form a pair when they share ``walk_base`` and differ in
    ``walk_step``. Only the strict upper triangle is set, so every
    unordered pair appears once.

    Raises
    ------
    ValueError
        If the batch has no walk attributes, i.e. the dataset was
        built with ``walk_length == 0``.
    """
    if not all(attribute in batch for attribute in WALK_ATTRIBUTES):
        raise ValueError(NO_WALK_MESSAGE.format(name=name))
    walk_base = batch.walk_base.reshape(-1)
    walk_step = batch.walk_step.reshape(-1)
    upper = torch.triu(
        torch.ones(
            len(walk_base), len(walk_base), dtype=torch.bool, device=walk_base.device
        ),
        diagonal=1,
    )
    same_base = walk_base[:, None] == walk_base[None, :]
    same_step = walk_step[:, None] == walk_step[None, :]
    return upper & same_base & ~same_step


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Mean of ``values`` over ``mask``; zero (not NaN) if it is empty."""
    count = mask.sum()
    if count == 0:
        return values.sum() * 0.0
    return values[mask].sum() / count


class Objective(nn.Module):
    """Base class of the training objectives.

    Parameters
    ----------
    base_loss : nn.Module
        Per-task loss on ``(logits, y)``, e.g. cross-entropy.
    """

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(
        self, logits: Tensor, embedding: Tensor, batch
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError


class CrossEntropyObjective(Objective):
    """The plain task loss; the baseline objective."""

    def forward(self, logits, embedding, batch):
        ce = self.base_loss(logits, batch.y)
        return ce, {"ce": ce.detach()}


class PachnerConsistency(Objective):
    """Task loss plus a consistency term over same-walk pairs.

    Parameters
    ----------
    base_loss : nn.Module
        Per-task loss.
    weight : float
        Weight of the consistency term.
    target : str
        ``"embedding"`` penalises the squared L2 distance between the
        embeddings of two snapshots of the same walk;
        ``"relative"`` the scale-free ratio
        ``||g_a - g_b||^2 / (||g_a||^2 + ||g_b||^2 + eps)``, which lies
        in [0, 2] and cannot be minimised by shrinking the embedding,
        so it does not collapse a sum readout the way the absolute
        distance can; ``"logits"`` the symmetric KL divergence between
        the softmax outputs.
    eps : float
        Stabiliser of the ``"relative"`` denominator.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weight: float = 1.0,
        target: str = "embedding",
        eps: float = 1e-8,
    ):
        super().__init__(base_loss)
        if target not in ("embedding", "relative", "logits"):
            raise ValueError(
                "target must be 'embedding', 'relative' or 'logits', "
                f"got '{target}'"
            )
        self.weight = weight
        self.target = target
        self.eps = eps

    def forward(self, logits, embedding, batch):
        ce = self.base_loss(logits, batch.y)
        mask = same_walk_mask(batch, type(self).__name__)
        if self.target in ("embedding", "relative"):
            diff = embedding[:, None, :] - embedding[None, :, :]
            values = (diff * diff).sum(-1)
            if self.target == "relative":
                norms = (embedding * embedding).sum(-1)
                values = values / (norms[:, None] + norms[None, :] + self.eps)
        else:
            log_p = F.log_softmax(logits, dim=-1)
            p = log_p.exp()
            # KL(a || b) + KL(b || a) written with the log-probabilities.
            cross = (p[:, None, :] * log_p[None, :, :]).sum(-1)
            entropy = (p * log_p).sum(-1)
            values = entropy[:, None] - cross + entropy[None, :] - cross.t()
        consistency = _masked_mean(values, mask)
        total = ce + self.weight * consistency
        return total, {"ce": ce.detach(), "consistency": consistency.detach()}
