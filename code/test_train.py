
import wandb
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
from collections import defaultdict

import torch
from torch_geometric.transforms import BaseTransform, Compose


from mantra.datasets import ManifoldTriangulations
from mantra.transforms import MomentCurveEmbedding, SelectAttributes, SelectFeatures

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_handling import split_dataset, filter_low_ocurrence_labels

from tqdm import tqdm
from torch_geometric.profile import count_parameters

cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)

torch._dynamo.config.capture_scalar_outputs = True

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    dataset_config = cfg.dataset
    transform_dict = dataset_config.transform
    transform_order = dataset_config.transform_order


    transforms = [None for _ in range(len(list(transform_dict.items())))]
    print(transform_dict)

    label_transform = None
    select_transform = None
    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == 'create_labels':
            label_transform = trans
        elif name == 'select_features':
            select_transform = trans
        transforms[transform_order.index(name)] = trans

    print(transforms)
    assert label_transform is not None, "Missing the `create_labels` transform"
    assert select_transform is not None, "Missing the `select_features` transform"

    transforms = Compose(transforms)

    dataset = ManifoldTriangulations(
        root=dataset_config.root,
        dimension=dataset_config.dimension,
        version=dataset_config.version,
        pre_transform=transforms,
        force_reload=dataset_config.reload,
        transform=label_transform
    )


     
    for data in dataset:
        pass

    dataset = filter_low_ocurrence_labels(dataset)
    label_to_index = label_transform.label_to_index


    # Change the outputs based on classification task
    if cfg.task.task_type == 'classification':
        num_classes = len(label_to_index)
        cfg.model.out_channels = num_classes

    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.trainer.device)
    model = torch.compile(model, dynamic=True)

    print(f"Model parameters: {count_parameters(model)}")

    #NOTE: Dataset initialization
    train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset, cfg)

    imbalance = None
    if cfg.task.imbalanced:
        imbalance = sorted_imbalance_weights(
            dataset
        )
        imbalance = np.array(list(imbalance))
        imbalance = imbalance / np.sum(imbalance)

    with wandb.init(project=cfg.logger.project) as run:
        if cfg.train:
            run_id = training_loop(train_dataloader, val_dataloader, imbalance, model, run, cfg)

        test(test_dataloader, model, label_to_index, run, cfg)

def test(test_dataset, model, label_to_index, run, cfg):

    is_classification = cfg.task.task_type == 'classification'

    if not is_classification:
        raise NotImplementedError("No test for regression yet")

    conf_mat = torch.zeros(cfg.model.out_channels, cfg.model.out_channels) if is_classification else None
    index_to_label = {v: k for k,v in label_to_index.items()}
    y_all = []
    y_hat_all = []

    model.eval()
    for batch in test_dataset:

        y_hat = model(batch).argmax(dim=1)
        y = batch.y.long()

        y_all.append(y)
        y_hat_all.append(y_hat)

        # for t,p in zip(y.view(-1), y_hat.view(-1)):
        #     conf_mat[t.long(), p.long()] += 1


    y_total = torch.cat(y_all)
    y_hat_total = torch.cat(y_hat_all)


    def remap_to_consecutive(x):
        mapping = {}
        next_id = 0

        remapped = torch.empty_like(x, dtype=torch.long)
    
        for i, val in enumerate(x.tolist()):
            if val not in mapping:
                mapping[val] = next_id
                next_id += 1
            remapped[i] = mapping[val]
        
        return remapped, mapping

    y_total, remaping = remap_to_consecutive(y_total)
    inv_mapping = {v: k for k,v in remaping.items()}

    class_names = ["" for _ in y_total.unique().tolist()]

    # Make a class_names sorted by this ordering
    for idx, true_idx in inv_mapping.items():
        class_names[idx] = index_to_label[true_idx]

    # Remap tensor
    y_hat_total.apply_(lambda x: remaping[x])

    # class_names = [index_to_label[l] for l in y_total.unique().tolist()]
    # class_names = list(index_to_label.values())

    run.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_total.tolist(), preds=y_hat_total.tolist(), class_names=class_names)
        }
    )


def train(model, optimizer, train_dataset, loss_fn, accuracies, cfg):

    model.train()
    metrics = defaultdict(float)
    # Go over each batch
    for batch in train_dataset:
        optimizer.zero_grad()
        y_hat = model(batch)

        y_hat = y_hat.squeeze()
        y = batch.y.long()

        loss = loss_fn(
            y_hat,
            y,
        )

        metrics['train/loss'] += loss

        for acc in accuracies:
            metrics['train/acc'] = acc(y_hat, y)

        loss.backward()
        optimizer.step()

    return metrics

def validate(model, optimizer, val_dataset, loss_fn, epoch, cfg):
    model.eval()
    metrics = defaultdict(float)

    for batch in val_dataset:

        y_hat = model(batch).squeeze()
        y = batch.y.long()

        loss = loss_fn(y_hat, y)

        metrics['val/loss'] += loss

    return metrics


def init_metrics(cfg):
    accs = []
    for name, acc in cfg.task.metrics.items():
        if 'accuracy' in acc._target_.lower():
            assert cfg.task.task_type == 'classification', "Cannot assign accuracy for regression tasks"
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels)
            accs.append(metric)
    return accs


def training_loop(train_dataloader, val_dataloader, imbalance, model, run, cfg):
    MAX_EPOCH = cfg.trainer.max_epoch
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.trainer.learning_rate, weight_decay=cfg.trainer.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # TODO: Should this be like this ?

    imbalance = (
        torch.tensor(
            imbalance, device=cfg.trainer.device, dtype=torch.float32
        )
        if imbalance is not None
        else None
    )

    #TODO: Maybe this should change ?
    if imbalance is not None:
        loss_fn = hydra.utils.instantiate(cfg.task.loss, weight=imbalance)
    else:
        loss_fn = hydra.utils.instantiate(cfg.task.loss)

    accuracies = init_metrics(cfg) 
        
    # Go over each epoch
    for epoch in tqdm(range(MAX_EPOCH)):

        # Train
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg)

        # Validation
        val_metrics = validate(model, optimizer, val_dataloader, loss_fn, epoch, cfg)

        log_metrics({**train_metrics, **val_metrics}, epoch=epoch, step=None, run=run, cfg=cfg)

        scheduler.step(val_metrics['val/loss'])

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
