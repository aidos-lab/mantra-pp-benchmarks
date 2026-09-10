"""Training objectives combining the task loss with Pachner-walk terms.

Every objective maps ``(logits, embedding, batch)`` to the total loss
and a dictionary of detached component tensors for logging. The pair
structure of a batch is recovered from the ``walk_base``,
``walk_step`` and ``y`` attributes that :class:`PachnerWalkDataset`
attaches to its entries; see :func:`pair_masks`.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

WALK_ATTRIBUTES = ("walk_base", "walk_step")

NO_WALK_MESSAGE = (
    "{name} needs same-walk pairs, but the batch has no "
    "`walk_base`/`walk_step` attributes. Set dataset.walk_length > 0 "
    "(and trainer.sampler=pachner_pairs to get such pairs into a batch)."
)


def walk_attributes(batch, name: str) -> Tuple[Tensor, Tensor]:
    """Return ``(walk_base, walk_step)`` as ``[B]`` long tensors.

    Raises
    ------
    ValueError
        If the batch has no walk attributes, i.e. the dataset was
        built with ``walk_length == 0``.
    """
    if not all(attribute in batch for attribute in WALK_ATTRIBUTES):
        raise ValueError(NO_WALK_MESSAGE.format(name=name))
    return (
        batch.walk_base.reshape(-1).long(),
        batch.walk_step.reshape(-1).long(),
    )


def pair_masks(batch, y: Tensor, name: str) -> Dict[str, Tensor]:
    """Compute the upper-triangular pair masks of a batch.

    Parameters
    ----------
    batch : Batch
        Collated batch with ``walk_base`` and ``walk_step``.
    y : Tensor
        Class labels of shape ``[B]``.
    name : str
        Objective name used in the error message.

    Returns
    -------
    dict of str to Tensor
        Boolean ``[B, B]`` masks ``same_walk`` (same base, different
        step), ``same_class_other_base`` and ``diff_class``. Only the
        strict upper triangle is set, so every unordered pair appears
        once and self-pairs are excluded.
    """
    walk_base, walk_step = walk_attributes(batch, name)
    y = y.reshape(-1)
    upper = torch.triu(
        torch.ones(len(y), len(y), dtype=torch.bool, device=y.device),
        diagonal=1,
    )
    same_base = walk_base[:, None] == walk_base[None, :]
    same_step = walk_step[:, None] == walk_step[None, :]
    same_class = y[:, None] == y[None, :]
    return {
        "same_walk": upper & same_base & ~same_step,
        "same_class_other_base": upper & same_class & ~same_base,
        "diff_class": upper & ~same_class,
    }


def pairwise_distances(embedding: Tensor, eps: float = 1e-8) -> Tensor:
    """Return the ``[B, B]`` Euclidean distances with a safe gradient."""
    diff = embedding[:, None, :] - embedding[None, :, :]
    return torch.sqrt((diff * diff).sum(-1) + eps)


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
        embeddings of two snapshots of the same walk; ``"logits"`` the
        symmetric KL divergence between their softmax outputs.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weight: float = 1.0,
        target: str = "embedding",
    ):
        super().__init__(base_loss)
        if target not in ("embedding", "logits"):
            raise ValueError(
                f"target must be 'embedding' or 'logits', got '{target}'"
            )
        self.weight = weight
        self.target = target

    def forward(self, logits, embedding, batch):
        ce = self.base_loss(logits, batch.y)
        mask = pair_masks(batch, batch.y, type(self).__name__)["same_walk"]
        if self.target == "embedding":
            diff = embedding[:, None, :] - embedding[None, :, :]
            values = (diff * diff).sum(-1)
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


class PachnerPairwise(Objective):
    """Metric-learning objective on the pair structure of a batch.

    Same-walk pairs are pulled to the distance ``alpha * d_ab`` where
    ``d_ab`` is the number of random moves between the snapshots;
    same-class pairs of different walks are pulled inside ``radius``;
    pairs of different classes are pushed beyond ``margin``.

    Parameters
    ----------
    base_loss : nn.Module
        Per-task loss, added with weight ``ce_weight``.
    alpha : float
        Embedding distance per Pachner move on the same walk.
    radius : float
        Target radius of a class.
    margin : float
        Minimum distance between classes.
    ce_weight : float
        Weight of the task loss.
    kind_weights : list of float
        Weights of the same-walk, same-class and different-class terms.
    moves_per_step : int
        Random moves between consecutive walk snapshots
        (``dataset.moves_per_step``).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        alpha: float = 1.0,
        radius: float = 1.0,
        margin: float = 2.0,
        ce_weight: float = 0.0,
        kind_weights: List[float] = (1.0, 1.0, 1.0),
        moves_per_step: int = 1,
    ):
        super().__init__(base_loss)
        if len(kind_weights) != 3:
            raise ValueError(
                f"kind_weights needs three entries, got {list(kind_weights)}"
            )
        self.alpha = alpha
        self.radius = radius
        self.margin = margin
        self.ce_weight = ce_weight
        self.kind_weights = [float(w) for w in kind_weights]
        self.moves_per_step = moves_per_step

    def forward(self, logits, embedding, batch):
        name = type(self).__name__
        masks = pair_masks(batch, batch.y, name)
        _, walk_step = walk_attributes(batch, name)
        distance = pairwise_distances(embedding)

        moves = (walk_step[:, None] - walk_step[None, :]).abs().to(
            distance.dtype
        ) * self.moves_per_step
        walk = _masked_mean(
            (distance - self.alpha * moves) ** 2, masks["same_walk"]
        )
        same_class = _masked_mean(
            F.relu(distance - self.radius) ** 2,
            masks["same_class_other_base"],
        )
        diff_class = _masked_mean(
            F.relu(self.margin - distance) ** 2, masks["diff_class"]
        )
        ce = self.base_loss(logits, batch.y)

        w_walk, w_class, w_diff = self.kind_weights
        total = (
            w_walk * walk
            + w_class * same_class
            + w_diff * diff_class
            + self.ce_weight * ce
        )
        return total, {
            "walk": walk.detach(),
            "same_class": same_class.detach(),
            "diff_class": diff_class.detach(),
            "ce": ce.detach(),
        }


def supervised_contrastive(
    embedding: Tensor, y: Tensor, temperature: float
) -> Tensor:
    """Supervised contrastive loss of Khosla et al. (2020).

    Embeddings are L2-normalised; the positives of an anchor are all
    other entries with the same label, the denominator runs over all
    other entries of the batch. Anchors without positives are skipped;
    the loss is zero if no anchor has one.
    """
    z = F.normalize(embedding, dim=-1)
    similarity = z @ z.t() / temperature
    n = len(y)
    not_self = ~torch.eye(n, dtype=torch.bool, device=y.device)
    positives = (y[:, None] == y[None, :]) & not_self

    log_denominator = torch.logsumexp(
        similarity.masked_fill(~not_self, float("-inf")), dim=-1
    )
    log_prob = similarity - log_denominator[:, None]
    n_positives = positives.sum(-1)
    anchors = n_positives > 0
    if not anchors.any():
        return embedding.sum() * 0.0
    per_anchor = (
        -(log_prob * positives).sum(-1)[anchors] / n_positives[anchors]
    )
    return per_anchor.mean()


class PachnerInfoNCE(Objective):
    """Supervised contrastive objective with same-class positives.

    Snapshots of the same walk share the class of their base entry, so
    they are positives of each other without any walk attributes being
    required.

    Parameters
    ----------
    base_loss : nn.Module
        Per-task loss, added with weight ``ce_weight``.
    temperature : float
        Softmax temperature of the similarities.
    ce_weight : float
        Weight of the task loss.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        temperature: float = 0.1,
        ce_weight: float = 0.0,
    ):
        super().__init__(base_loss)
        self.temperature = temperature
        self.ce_weight = ce_weight

    def forward(self, logits, embedding, batch):
        infonce = supervised_contrastive(
            embedding, batch.y.reshape(-1), self.temperature
        )
        ce = self.base_loss(logits, batch.y)
        total = infonce + self.ce_weight * ce
        return total, {"infonce": infonce.detach(), "ce": ce.detach()}
