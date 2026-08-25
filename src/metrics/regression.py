import torch
from torchmetrics import Metric


class RoundedAccuracy(Metric):
    """Accuracy of regression predictions rounded to the nearest integer.

    This is the accuracy notion used for Hodge-number regression in
    arXiv:2112.09117: a prediction counts as correct iff it rounds to
    the true (integer) target. For multi-target regression, every
    component is counted individually.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape(target.shape)
        self.correct += (preds.round() == target.round()).float().sum()
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total


class LatticeRoundedAccuracy(Metric):
    """Accuracy of predictions rounded to the label lattice.

    Some integer targets only attain multiples of a fixed spacing: the
    Chern targets (c2 . D_i and c2_anticanonical) are even on every
    sample of the Kreuzer-Skarke data. Rounding to the nearest plain
    integer then wastes accuracy on predictions (odd numbers) that no
    sample ever attains -- a prediction of 2.9 is scored against 3
    instead of the nearest attainable label 2. Here a prediction counts
    as correct iff its nearest multiple of ``spacing`` equals the
    target's.

    With ``spacing=1`` this is exactly ``RoundedAccuracy``.
    """

    def __init__(self, spacing: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.spacing = spacing
        self.add_state(
            "correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds = preds.reshape(target.shape)
        self.correct += (
            ((preds / self.spacing).round() == (target / self.spacing).round())
            .float().sum()
        )
        self.total += target.numel()

    def compute(self):
        return self.correct / self.total
