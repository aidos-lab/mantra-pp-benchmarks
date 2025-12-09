"""
Base pytorch lightning model.
"""

from typing import Literal

import lightning as L
import numpy as np
import torch
import torch_geometric


class BaseModel(L.LightningModule):
    def __init__(
        self,
        model,
        training_accuracy,
        test_accuracy,
        validation_accuracy,
        accuracies_fn,
        loss_fn,
        learning_rate,
        imbalance,
        factor: float = 0.5,
        patience: int = 10,
        weight_decay: float = 0.0
    ):
        super().__init__()
        self.training_accuracy = training_accuracy
        self.validation_accuracy = validation_accuracy
        self.test_accuracy = test_accuracy
        self.loss_fn = loss_fn
        self.accuracies_fn = accuracies_fn
        self.model = model
        self.learning_rate = learning_rate
        self.imbalance = imbalance
        self.factor = factor
        self.patience = patience
        self.weight_decay = weight_decay
        if imbalance is not None:
            self.imbalance = np.array(list(self.imbalance))
            self.imbalance = self.imbalance / np.sum(self.imbalance)

    def forward(self, batch):
        x = self.model(batch)
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
        elif hasattr(batch, "batch"):
            batch_len = batch.batch.max() + 1
        else:
            raise ValueError(
                "Batch object does not have a known way to compute batch size."
            )

    
        # Generalizing to accomodate for the different signatures.
        x_hat = self(batch)
        # Squeeze x_hat to match the shape of y
        x_hat = x_hat.squeeze()
        imbalance = (
            torch.tensor(
                self.imbalance, device=self.device, dtype=torch.float32
            )
            if self.imbalance is not None
            else None
        )
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
        self.log_accuracies(x_hat, batch.y, batch_len, step)
        return loss

    def configure_optimizers(self):
        """
        Adapted from: https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch_geometric.nn.GCNConv)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, )
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith('layer_scale'):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.Adam(optim_groups, lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.factor, patience=self.patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "validation_loss"}

    def log_accuracies(
        self, x_hat, y, batch_len, step: Literal["train", "test", "validation"]
    ):
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
