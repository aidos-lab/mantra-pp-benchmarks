
import wandb
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
from collections import defaultdict

import torch
from torch_geometric.transforms import BaseTransform, Compose
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import train_test_split

from mantra.datasets import ManifoldTriangulations
from mantra.transforms import MomentCurveEmbedding, SelectAttributes, SelectFeatures

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metric

cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    dataset_config = cfg.dataset
    transform_dict = dataset_config.transform
    transform_order = dataset_config.transform_order

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    transforms = []
    label_transform = None
    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == 'create_labels':
            label_transform = trans
        transforms.append(trans)
        #TODO: Add the ordering here

    assert label_transform is not None, "Missing the `create_labels` transform"

    transforms = Compose(transforms)

    dataset = ManifoldTriangulations(
        root=dataset_config.root,
        dimension=dataset_config.dimension,
        version=dataset_config.version,
        pre_transform=transforms
    )

    label_to_index = label_transform.label_to_index

    model = hydra.utils.instantiate(cfg.model)

    train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset, cfg)

    imbalance = None
    if cfg.task.imbalanced:
        imbalance = sorted_imbalance_weights(
            dataset
        )
        imbalance = np.array(list(imbalance))
        imbalance = imbalance / np.sum(imbalance)

    if cfg.train:
        training_loop(train_dataloader, val_dataloader, imbalance, model, cfg)

    model.eval()
    test(test_dataloader, model, cfg)

def split_dataset(dataset, cfg):
    split_params = cfg.dataset.train_parameters.split

    train_val_index, test_index = train_test_split(
        np.arange(len(dataset)),
        test_size=split_params.test,
        shuffle=True,
        stratify=(
            dataset.y
            if cfg.use_stratified
            else None
        ),
        random_state=cfg.seed,
    )

    # train val split
    train_index, val_index = train_test_split(
        train_val_index,
        test_size=split_params.val
        / (split_params.train + split_params.val),
        shuffle=True,
        stratify=(
            dataset[train_val_index]
            if cfg.use_stratified
            else None
        ),
        random_state=cfg.seed,
    )

    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    test_dataset = Subset(dataset, test_index)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train(model, optimizer, train_dataset, loss_fn, cfg):

    model.train()
    metrics = defaultdict(float)
    # Go over each batch
    for batch in train_dataset:
        optimizer.zero_grad()
        y_hat = model(batch)
        y_hat = y_hat.squeeze()

        imbalance = (
            torch.tensor(
                imbalance, device=cfg.device, dtype=torch.float32
            )
            if imbalance is not None
            else None
        )

        loss = loss_fn(
            y_hat,
            batch.y,
            weight=imbalance,
        )

        metrics['loss'] += loss

        for acc in accuracies:
            metrics['acc'] = acc(y_hat, batch.y, imbalance)


        loss.backward()
        optimizer.step()


    return metrics

def validate():
    pass

def test():
    pass

def training_loop(train_dataloader, val_dataloader, imbalance, model, cfg):
    MAX_EPOCH = cfg.trainer.max_epoch
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    #TODO: Maybe this should change ?
    loss_fn = hydra.utils.instantiate(cfg.task.loss)

    # Go over each epoch
    for epoch in range(MAX_EPOCH):
        # Train and logging
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, epoch, cfg)
        for name, value in train_metrics.items():
            log_metric(name, value, job_type="train", epoch=epoch, cfg=cfg)

        # Validation and loggin
        val_metrics = validate(model, optimizer, val_dataloader, loss_fn, epoch, cfg)
        for name, value in train_metrics.items():
            log_metric(name, value, job_type="validation", epoch=epoch, cfg=cfg)

        scheduler.step(val_metrics['loss'])






def configure_optimizers():
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
    return  optimizer, scheduler

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


if __name__ == "__main__":
    main()
