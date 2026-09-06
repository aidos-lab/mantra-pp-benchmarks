from enum import Enum

import torch
import torch.nn as nn


class LossType(Enum):
    BINARY = (1,)
    MULTICLASS = (2,)
    REGRESSION = 3


class BaseLoss(nn.Module):
    def __init__(self, task_type: LossType):
        super().__init__()
        self.task_type = task_type

    def turn_to_probs(self, y_hat):
        if self.task_type == LossType.BINARY:
            return nn.functional.sigmoid(y_hat)
        elif self.task_type == LossType.MULTICLASS:
            return nn.functional.log_softmax(y_hat, dim=-1)
        return y_hat


class Contrastive(BaseLoss):
    def __call__(self, y1_pred, y2_pred, gamma: float = 0):
        return max(
            0,
            (y1_pred @ y2_pred) / (torch.norm(y1_pred) * torch.norm(y2_pred))
            - gamma,
        )
