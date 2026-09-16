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
