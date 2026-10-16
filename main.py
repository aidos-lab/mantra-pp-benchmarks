import hashlib
import json
import time

import wandb
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import torch
import os
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import Compose


from mantra.datasets import MantraDataset, CalabiYauDataset
from mantra.representations.internal import SimplexTrie
from mantra.representations.internal.simplex_trie import SimplexNode
from mantra.transforms.task_transforms import (
    AttributeToClassTransform,
    AttributeToNodeClassTransform,
    AttributeToNodeRegressionTransform,
    AttributeToRegressionTransform,
    BettiToClassTransform,
    NameToClass2MTransform,
    NameToClass3MTransform,
    OrientableToClassTransform,
)

from src.schema import RootConfig
from src.transforms import DropCacheAttributes, RemapLabels
from src.utils.baselines import regression_target_stats, trivial_baselines
from src.utils.logging import log_metrics
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import train, validate, test, init_metrics
from src.utils.gpu_utils import is_blackwell_gpu, set_gpu_configs

from tqdm import tqdm
from torch_geometric.profile import count_parameters

torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])


cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)


torch._dynamo.config.capture_scalar_outputs = True

def task_to_transform(cfg):
    """Label transform of the task, i.e. the transform writing `data.y`.

    The MANTRA tasks are keyed by name. Every other task reads the raw
    attribute `task.target`; classification stores the raw value, which
    `create_dataset` remaps over the train split. `task.task_level=node`
    selects the per-vertex variants.
    """
    task_level = cfg.task.task_level

    if cfg.task.name == "BettiNumbers":
        return BettiToClassTransform(cfg.dataset.dimension)
    elif cfg.task.name == "Orientability":
        return OrientableToClassTransform()
    elif cfg.task.name == "Name":
        if cfg.dataset.dimension == 2:
            return NameToClass2MTransform()
        elif cfg.dataset.dimension == 3:
            return NameToClass3MTransform()
        else:
            raise ValueError("Invalid dimension")
    elif cfg.task.task_type == "regression":
        if task_level == "node":
            return AttributeToNodeRegressionTransform(cfg.task.target)
        return AttributeToRegressionTransform(cfg.task.target)
    elif cfg.task.task_type == "classification":
        if task_level == "node":
            return AttributeToNodeClassTransform(cfg.task.target, None)
        return AttributeToClassTransform(cfg.task.target, None)
    else:
        raise ValueError("Invalid task")


def cache_name(dataset_cfg):
    """Name of the cached variant: a hash of its transform config.

    `MantraDataset` and `CalabiYau` key their processed directory on this
    name, so a root can hold several transform stacks side by side and a
    changed stack never reads a stale cache.
    """
    chain = OmegaConf.to_container(dataset_cfg.transform, resolve=True)
    order = list(dataset_cfg.transform_order)
    digest = hashlib.sha1(repr((chain, order)).encode()).hexdigest()
    return f"t{digest[:10]}"


def _target_values(dataset):
    """Set of the raw class labels in `dataset`, scalar or per vertex."""
    return {
        int(value)
        for data in dataset
        for value in data.y.reshape(-1).tolist()
    }


def create_dataset(cfg):
    """Build the train/val/test datasets and one OOD dataset per table entry.

    The representation and encoding transforms run once as the
    `pre_transform` of the cache named by `cache_name`; only the task
    transform (and the label remap for classification) runs on access.
    `dataset.ood_evaluations` maps a name to the subdivision arguments of
    `MantraDataset` (`division_type` plus `graded_vertex_number`,
    `fraction` or `round`); each entry becomes an `ood` split built from
    the same cached test entries and logged under `ood_<name>/`. The
    `ood` config group fills the table, and it is empty by default.
    `dataset.reload` rebuilds the splits once and every OOD split of the
    run.
    The `cy` dataset uses the same cache layout through `CalabiYauDataset`
    and has no OOD splits.

    Returns
    -------
    tuple
        Train, val and test datasets, the dict of OOD datasets keyed
        `ood_<name>`, and the label-to-index map (empty for regression).
    """
    torch.manual_seed(cfg.torch_seed)

    dataset_cfg = cfg.dataset
    transform_cfg = dataset_cfg.transform
    transform_order = dataset_cfg.transform_order
    is_cy = dataset_cfg.type == "cy"


    missing = [name for name in transform_cfg if name not in transform_order]
    if missing:
        raise ValueError(f"Transforms {missing} are not in dataset.transform_order")
    transforms = [
        hydra.utils.instantiate(transform_cfg[name])
        for name in transform_order
        if name in transform_cfg
    ]

    task_transform = task_to_transform(cfg)

    split = dataset_cfg.train_parameters.split

    if is_cy:
        # The representation chain is cached in the dataset root as
        # pre_transform; only the cheap label transform runs on access.
        # The cache name hashes the transform config, so differently
        # transformed variants never read each other's files.
        select = transform_cfg.get("select_features")
        drop = DropCacheAttributes(
            [select.src] if select is not None else [],
            graph="representation" in transform_cfg,
        )
        pre_transform = Compose([*transforms, drop])
        dataset_parameters = {
            "root": dataset_cfg.root,
            "name": cache_name(dataset_cfg),
            "version": dataset_cfg.version,
            "local_path": dataset_cfg.local_path,
            "limit": dataset_cfg.limit,
            "force_reload": dataset_cfg.reload,
            "seed": dataset_cfg.seed,
            "split_proportions": [split.train, split.val, split.test],
            "stratified": dataset_cfg.use_stratified,
            "label_source": dataset_cfg.label_source,
            "min_sample_per_class": dataset_cfg.min_sample_per_class,
            "pre_transform": pre_transform,
            "transform": task_transform,
        }

        train_dataset = CalabiYauDataset(split_type="train", **dataset_parameters)
        val_dataset = CalabiYauDataset(split_type="val", **dataset_parameters)
        test_dataset = CalabiYauDataset(split_type="test", **dataset_parameters)
        ood_datasets = {}
    else:
        dataset_parameters = {
            "root": dataset_cfg.root,
            "name": cache_name(dataset_cfg),
            "dimension": dataset_cfg.dimension,
            "version": dataset_cfg.version,
            "balanced": dataset_cfg.balanced,
            "local_path": dataset_cfg.local_path,
            "seed": dataset_cfg.seed,
            "split_proportions": [split.train, split.val, split.test],
            "min_sample_per_class": dataset_cfg.min_sample_per_class,
            "stratified": dataset_cfg.use_stratified,
            "max_vertices": dataset_cfg.max_vertices,
            "n_moves": dataset_cfg.n_moves,
            "target_count": dataset_cfg.target_count,
            "use_surgery": dataset_cfg.use_surgery,
            "max_ood_size_per_class": dataset_cfg.max_ood_size_per_class,
            "pre_transform": Compose(transforms),
            "transform": task_transform,
        }

        # force_reload removes every cached file, so only the train split sets it.
        train_dataset = MantraDataset(
            split_type="train", force_reload=dataset_cfg.reload, **dataset_parameters
        )
        val_dataset = MantraDataset(split_type="val", **dataset_parameters)
        test_dataset = MantraDataset(split_type="test", **dataset_parameters)

        ood_evaluations = OmegaConf.to_container(dataset_cfg.ood_evaluations, resolve=True)
        ood_datasets = {
            f"ood_{name}": MantraDataset(split_type="ood", **dataset_parameters, **kwargs)
            for name, kwargs in ood_evaluations.items()
        }

    label_to_index = {}

    if cfg.task.task_type == "classification":
        labels = sorted(_target_values(train_dataset))
        label_to_index = {label: index for index, label in enumerate(labels)}
        unseen = (
            _target_values(val_dataset) | _target_values(test_dataset)
        ) - set(labels)
        if unseen:
            raise ValueError(
                f"Classes {sorted(unseen)} occur in val/test but not in "
                "train; change dataset.seed or raise "
                "dataset.min_sample_per_class"
            )
        remap_labels = RemapLabels(label_to_index)

        for dataset in (train_dataset, val_dataset, test_dataset, *ood_datasets.values()):
            dataset.transform = Compose([dataset.transform, remap_labels])

    return train_dataset, val_dataset, test_dataset, ood_datasets, label_to_index


def create_loaders(cfg, train_dataset, val_dataset, test_dataset, ood_datasets=None):
    trainer_cfg = cfg.trainer
    persistent_workers = trainer_cfg.num_workers > 0

    is_graph_representation = "x_0" not in train_dataset[0]

    loader_kwargs = dict(
        batch_size=trainer_cfg.batch_size,
        num_workers=trainer_cfg.num_workers,
        pin_memory=trainer_cfg.pin_memory,
    )
    if is_graph_representation:
        loader_cls = PyGDataLoader
    else:
        loader_cls = DataLoader
        loader_kwargs["collate_fn"] = collate_simplicial

    train_loader = loader_cls(
        train_dataset, shuffle=True, persistent_workers=persistent_workers, **loader_kwargs
    )
    val_loader = loader_cls(
        val_dataset, shuffle=False, persistent_workers=persistent_workers, **loader_kwargs
    )
    test_loader = loader_cls(test_dataset, shuffle=False, **loader_kwargs)
    ood_loaders = {
        name: loader_cls(dataset, shuffle=False, **loader_kwargs)
        for name, dataset in (ood_datasets or {}).items()
    }

    return train_loader, val_loader, test_loader, ood_loaders

class TargetAffine(torch.nn.Module):
    """Predict in standardised units and map back: `mean + std * f(x)`.

    Loss, metrics and checkpoints stay in the target's original scale,
    while the wrapped model's output starts at unit scale instead of
    having to learn an offset of tens or hundreds.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))

    def forward(self, batch):
        return self.mean + self.std * self.model(batch)


def create_model(cfg, num_features, label_to_index, target_stats=None):
    """Instantiate the configured model; regression heads get `TargetAffine`
    with the train-split `target_stats` (mean, std) when given."""
    cfg.model.out_channels = (
        len(label_to_index) if cfg.task.task_type == "classification" else 1
    )

    # Models branch on the string, so a typo must not pick a head silently.
    task_level = cfg.task.task_level
    if task_level not in ("graph", "node"):
        raise ValueError(
            f"task.task_level must be 'graph' or 'node', got {task_level!r}"
        )
    cfg.model.task_level = task_level

    in_channels = num_features

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, in_channels=in_channels)
    if target_stats is not None and cfg.task.task_type == "regression":
        model = TargetAffine(model, *target_stats)
    model = model.to(cfg.trainer.device)

    # Only compile on GPU
    if is_blackwell_gpu(cfg) and cfg.compile:
        model = torch.compile(model, dynamic=True)

    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    OmegaConf.resolve(cfg)

    set_gpu_configs(cfg)

    # ---- Dataset -----

    train_ds, val_ds, test_ds, ood_ds, label_to_index = create_dataset(cfg)
    #draws from the global RNG; reseed so model init and
    # shuffling do not depend on whether the cache already existed.
    torch.manual_seed(cfg.torch_seed)
    train_dl, val_dl, test_dl, ood_dls = create_loaders(cfg, train_ds, val_ds, test_ds, ood_ds)

    sample = train_ds[0]
    is_graph_representation = "x_0" not in sample

    if is_graph_representation:
        num_features = sample.x.shape[1]
    else:
        num_features = [
            sample[f"x_{dim}"].shape[1]
            for dim in range(int(sample.dimension.item()) + 1)
        ]

    # ----- Model ----
    target_stats = (
        regression_target_stats(cfg, train_ds)
        if cfg.task.task_type == "regression"
        else None
    )
    model = create_model(cfg, num_features, label_to_index, target_stats)

    print(f"Model parameters: {count_parameters(model)}")

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        run.summary.update(trivial_baselines(cfg, train_ds, test_ds))
        accuracies = init_metrics(cfg)
        last_epoch = 0
        if cfg.train:
            model, last_epoch = training_loop(train_dl, val_dl, model, accuracies, run, cfg, label_to_index)

        test(model, test_dl, accuracies, label_to_index, run, cfg, epoch=last_epoch)

        for name, loader in ood_dls.items():
            test(model, loader, accuracies, label_to_index, run, cfg, epoch=last_epoch, split=name)


def training_loop(train_dataloader, val_dataloader, model, accuracies, run, cfg, label_to_index):
    MAX_EPOCH = cfg.trainer.max_epoch
    if MAX_EPOCH <= 0:
        raise ValueError("trainer.max_epoch must be greater than zero")
    if cfg.trainer.early_stopping <= 0:
        raise ValueError("trainer.early_stopping must be greater than zero")

    if is_blackwell_gpu(cfg):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg.trainer.learning_rate), weight_decay=cfg.trainer.weight_decay,
            fused=True
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(cfg.trainer.learning_rate), weight_decay=cfg.trainer.weight_decay,
        )


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg.trainer.scheduler_patience
    )
    loss_fn = hydra.utils.instantiate(cfg.task.loss)

    early_stopping_cnt = cfg.trainer.early_stopping
    last_val_loss = None
    last_epoch = 0

    checkpoint_dir = os.path.join(cfg.checkpoint_path, run.name or run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "label_to_index.json"), "w") as f:
        json.dump(
            [
                {"label": label, "index": index}
                for label, index in label_to_index.items()
            ],
            f,
        )
    with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        if early_stopping_cnt == 0:
            print("Early stopping...")
            break

        last_epoch = epoch

        # Train
        epoch_start = time.perf_counter()
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg)
        train_metrics['train/epoch_time_s'] = time.perf_counter() - epoch_start

        # Validation
        val_metrics = validate(model, val_dataloader, loss_fn, accuracies, cfg)

        # Construct the progress bar
        pbar_str = ""
        for k,v in {**train_metrics, **val_metrics}.items():
            if 'loss' in k:
                pbar_str += f"{k}: {v:.5f},"
            else:
                pbar_str += f"{k}: {v:.2f},"

        pbar.set_description(pbar_str)

        log_metrics({**train_metrics, **val_metrics}, epoch=epoch, step=None, run=run, cfg=cfg)

        scheduler.step(val_metrics['val/loss'])

        if last_val_loss is None or val_metrics['val/loss'] < last_val_loss:
            last_val_loss = val_metrics['val/loss']
            early_stopping_cnt = cfg.trainer.early_stopping

            # Save weights and config of the best model
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_val.pt"))
            best_val_metrics = val_metrics
        else:
            early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(checkpoint_dir, "best_val.pt"), weights_only=True)

    model.load_state_dict(state_dict)
    run.summary.update(best_val_metrics)
    return model, last_epoch

if __name__ == "__main__":
    main()
