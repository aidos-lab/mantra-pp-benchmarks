import torch
import torch.nn as nn


class CrossEntropy:
    def __call__(y_hat, y, weight: torch.Tensor | None):
        if weight is None:
            return nn.functional.cross_entropy(y, y_hat)
        return nn.functional.cross_entropy(y, y_hat, weight)

class Contrastive:
    def __call__(y1_pred, y2_pred, gamma: float = 0):
        return max(0, (y1_pred @ y2_pred)/(torch.norm(y1_pred)*torch.norm(y2_pred)) - gamma)