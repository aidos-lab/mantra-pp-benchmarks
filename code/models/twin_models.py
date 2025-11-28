"""
Base pytorch lightning model.
"""

from typing import Literal

import lightning as L
import numpy as np
import torch
import torch.nn as nn


class TwinModel(L.LightningModule):
    def __init__(
        self,
        model,
        training_accuracy,
        test_accuracy,
        validation_accuracy,
        accuracies_fn,
        loss_fn,
        contrastive_loss_activated,
        learning_rate,
        imbalance,
        out_channels=None,
        mlp=None,
    ):
        super().__init__()
        #self.training_accuracy = training_accuracy
        #self.validation_accuracy = validation_accuracy
        #self.test_accuracy = test_accuracy
        self.loss_fn = loss_fn
        #self.accuracies_fn = accuracies_fn
        self.model = model
        self.contrastive_loss_activated = contrastive_loss_activated
        if mlp is not None:
            self.mlp = nn.Sequential(nn.Linear(out_channels, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
        else:
            self.mlp = None
        self.learning_rate = learning_rate
        self.imbalance = imbalance
        if imbalance is not None:
            self.imbalance = np.array(list(self.imbalance))
            self.imbalance = self.imbalance / np.sum(self.imbalance)

    def forward(self, batch):
        x = self.model(batch) #[2*B, f]
        if self.mlp is not None:
            x = x[::2] + x[1::2] #[B, f]
            x = self.mlp(x)
        else:
            x1 = x[::2]
            x2 = x[1::2]
            x = (x1, x2)
        return x

    def get_log_name(self, log_type: str, step: str):
        loss_log_name = f"{step}_{log_type}"
        return loss_log_name

    def general_step(self, batch, batch_idx, step: str):

        # This is rather ugly, open to better solutions,
        # but torch_geometric and the toponetx dl have rather different
        # signatures.
        if hasattr(batch, "batch_size"):
            batch_len = batch.batch_size
        # We batch based on number of nodes
        elif hasattr(batch, "x_belonging"):
            batch_len = batch.x_belonging[0].max() + 1
        elif hasattr(batch, "batch"):
            batch_len = batch.batch.max() + 1
        else:
            raise ValueError(
                "Batch object does not have a known way to compute batch size."
            )

        imbalance = (
            torch.tensor(
                self.imbalance, device=self.device, dtype=torch.float32
            )
            if self.imbalance is not None
            else None
        )
        if self.contrastive_loss_activated:
            x1_hat, x2_hat = self(batch)
            # Squeeze x_hat to match the shape of y
            x1_hat = x1_hat.squeeze()
            x2_hat = x2_hat.squeeze()
            loss = self.loss_fn(
                x1_hat,
                x2_hat,
            )
        else:
            x = self(batch)
            x_hat = x.squeeze()
            loss = self.loss_fn(
                x_hat,
                batch.y,
                weight=imbalance,
            )

        self.log(
            self.get_log_name("loss", step),
            loss,
            prog_bar=True,
            batch_size=batch_len,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        if not self.contrastive_loss_activated:
            # todo: accuracies can be calculated for binary classification
            self.log_accuracies(x_hat, batch.y, batch_len, step)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        return optimizer

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
        assert self.contrastive_loss_activated != False, "Accuracies can only be logged for non-contrastive tasks."
        if step == "train":
            acc_fun = self.training_accuracy
        elif step == "test":
            acc_fun = self.test_accuracy
        elif step == "validation":
            acc_fun = self.validation_accuracy
        else:
            ValueError("Unknown step Literal")

        accuracies = self.accuracies_fn(acc_fun, x_hat, y, step)
        for accuracy in accuracies:
            self.log(
                **accuracy,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=batch_len,
                sync_dist=True,
            )

    def test_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "test")

    def validation_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "validation")

    def training_step(self, batch, batch_idx):
        return self.general_step(batch, batch_idx, "train")
