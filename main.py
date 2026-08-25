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


from mantra.datasets import MantraDataset
from mantra.representations.internal import SimplexTrie
from mantra.representations.internal.simplex_trie import SimplexNode
from mantra.transforms.task_transforms import BettiToClassTransform, OrientableToClassTransform, NameToClass2MTransform, NameToClass3MTransform

from src.schema import RootConfig
from src.transforms import RemapLabels
from src.utils.logging import log_metrics
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import (
    train,
    validate,
    test,
    init_metrics,
    compute_target_scaler,
)
from src.utils.gpu_utils import is_blackwell_gpu, set_gpu_configs

from tqdm import tqdm
from torch_geometric.profile import count_parameters

torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])


cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)


torch._dynamo.config.capture_scalar_outputs = True

def task_to_transform(cfg):
    if cfg.task.name == "BettiNumbers":
        return BettiToClassTransform()
    elif cfg.task.name == "Orientability":
        return OrientableToClassTransform()
    elif cfg.task.name == "Name":
        if cfg.dataset.dimension == 2:
            return NameToClass2MTransform()
        elif cfg.dataset.dimension == 3:
            return NameToClass3MTransform()
        else:
            raise ValueError("Invalid dimension")
    else:
        raise ValueError("Invalid task")

        
def _cy_class_mapping(cfg):
    """Class mapping {attribute value: output neuron} for CY classification.

    Built from the raw label attribute over the *full* dataset (sorted
    values), so the neuron assignment is independent of the split and
    the seed, and values that only occur in val or test still get an
    output neuron.
    """
    from mantra.datasets import CalabiYau

    dataset_cfg = cfg.dataset
    source = dataset_cfg.transform.create_labels.source
    dataset = CalabiYau(
        root=dataset_cfg.root,
        version=dataset_cfg.version,
        local_path=dataset_cfg.local_path,
        limit=dataset_cfg.limit,
        force_reload=dataset_cfg.reload,
    )

    values = set()
    for data in dataset:
        value = data[source]
        if isinstance(value, torch.Tensor):
            value = value.item()
        values.add(int(value))

    return {value: index for index, value in enumerate(sorted(values))}


def create_dataset(cfg):
    torch.manual_seed(cfg.torch_seed)

    dataset_cfg = cfg.dataset
    transform_cfg = dataset_cfg.transform
    transform_order = dataset_cfg.transform_order

    is_cy = dataset_cfg.type == "cy"

    # For CY classification the label transform (an
    # AttributeToClassTransform) needs its mapping at instantiation.
    transform_overrides = {}
    cy_label_to_index = {}
    if is_cy and cfg.task.task_type == "classification":
        mapping = transform_cfg.create_labels.get("mapping")
        if mapping is None:
            cy_label_to_index = _cy_class_mapping(cfg)
            transform_overrides["create_labels"] = {
                "mapping": cy_label_to_index
            }
        else:
            cy_label_to_index = dict(mapping)

    transforms = [None for _ in range(len(list(transform_cfg.items())))]
    task_transform = task_to_transform(cfg)

    for name, transform in transform_cfg.items():
        trans = hydra.utils.instantiate(transform)
        assert name in transform_order, f"Transform {name} not in transforms"
        transforms[transform_order.index(name)] = trans

    transforms.insert(0, task_transform)

    transforms = Compose(transforms)

    split = dataset_cfg.train_parameters.split

    if is_cy:
        # CalabiYau only exists on the mantra `cy` branch; import
        # lazily so the MANTRA tasks keep working against mantra main.
        from mantra.datasets import CalabiYau

        dataset_parameters = {
            "root": dataset_cfg.root,
            "version": dataset_cfg.version,
            "local_path": dataset_cfg.local_path,
            "limit": dataset_cfg.limit,
            "force_reload": dataset_cfg.reload,
            "seed": cfg.seed,
            "split_proportions": [split.train, split.val, split.test],
            "transform": transforms,
        }

        train_dataset = CalabiYau(split_type="train", **dataset_parameters)
        val_dataset = CalabiYau(split_type="val", **dataset_parameters)
        test_dataset = CalabiYau(split_type="test", **dataset_parameters)

        # The class mapping already spans the full dataset, so `y` is
        # contiguous by construction; the train-split remapping below
        # is only needed for the MANTRA datasets.
        return train_dataset, val_dataset, test_dataset, cy_label_to_index

    dataset_parameters = {
            "root": dataset_cfg.root,
            "dimension": dataset_cfg.dimension,
            "version": dataset_cfg.version,
            "balanced": dataset_cfg.balanced,
            "local_path": dataset_cfg.local_path,
            "force_reload": dataset_cfg.reload,
            "seed": dataset_cfg.seed,
            "split_proportions": [split.train, split.val, split.test],
            "min_sample_per_class": dataset_cfg.min_sample_per_class,
            "division_type": dataset_cfg.division_type,
            "stratified": dataset_cfg.use_stratified,
            "max_vertices": dataset_cfg.max_vertices,
            "n_moves": dataset_cfg.n_moves,
            "target_count": dataset_cfg.target_count,
            "use_surgery": dataset_cfg.use_surgery,
            "max_ood_size_per_class": dataset_cfg.max_ood_size_per_class,
            "transform": transforms,

        }

    train_dataset = MantraDataset(
        split_type="train",
        **dataset_parameters
    )
    val_dataset = MantraDataset(
            split_type="val",
            **dataset_parameters
    )
    test_dataset = MantraDataset(
            split_type="test",
            **dataset_parameters
    )

    label_to_index = {}

    # If we are doing classification perform a relabaling that
    # enumerates correctly
    if cfg.task.task_type == "classification":
        labels = sorted({int(data.y.item()) for data in train_dataset})
        label_to_index = {label: index for index, label in enumerate(labels)}
        remap_labels = RemapLabels(label_to_index)

        for dataset in (train_dataset, val_dataset, test_dataset):
            dataset.transform = Compose([dataset.transform, remap_labels])

    return train_dataset, val_dataset, test_dataset, label_to_index

def create_loaders(cfg, train_dataset, val_dataset, test_dataset):
    trainer_cfg = cfg.trainer
    persistent_workers = trainer_cfg.num_workers > 0

    is_graph_representation = "x_0" not in train_dataset[0]

    if not is_graph_representation:
        train_loader = DataLoader(train_dataset, batch_size=trainer_cfg.batch_size, shuffle=True, collate_fn=collate_simplicial, num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)
        val_loader = DataLoader(val_dataset, batch_size=trainer_cfg.batch_size, shuffle=False, collate_fn=collate_simplicial, num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)
        test_loader = DataLoader(test_dataset, batch_size=trainer_cfg.batch_size, shuffle=False, collate_fn=collate_simplicial, num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)
    else: # Graph
        train_loader = PyGDataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)
        val_loader = PyGDataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False,  num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)
        test_loader = PyGDataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False,  num_workers=trainer_cfg.num_workers, pin_memory=trainer_cfg.pin_memory, persistent_workers=persistent_workers)

    return train_loader, val_loader, test_loader

def create_model(cfg, num_features, label_to_index):
    cfg.model.out_channels = (
        len(label_to_index) if cfg.task.task_type == "classification" else 1
    )

    # A ReLU on the output head clamps predictions to >= 0, which
    # cannot fit regression targets; force the plain linear head on
    # models that expose the switch (MLP).
    if cfg.task.task_type != "classification" and "readout_activation" in cfg.model:
        cfg.model.readout_activation = False

    in_channels = num_features

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, in_channels=in_channels)
    model = model.to(cfg.trainer.device)

    # Only compile on GPU
    if is_blackwell_gpu(cfg) and cfg.compile:
        model = torch.compile(model, dynamic=True)

    return model


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    OmegaConf.resolve(cfg)

    if cfg.task.task_type == "regression" and "CrossEntropy" in cfg.task.loss._target_:
        raise RuntimeError(
            "task_type=regression combined with CrossEntropyLoss: the "
            "default loss config is still active and would silently "
            "produce a meaningless loss. Add `loss=mse` to the overrides."
        )

    set_gpu_configs(cfg)

    # ---- Dataset -----

    train_ds, val_ds, test_ds, label_to_index = create_dataset(cfg)
    train_dl, val_dl, test_dl = create_loaders(cfg, train_ds, val_ds, test_ds)

    sample = train_ds[0]
    if "x_0" in sample:
        num_features = [
            sample[f"x_{dim}"].shape[1]
            for dim in range(int(sample.dimension.item()) + 1)
        ]
    else:
        num_features = sample.x.shape[1]

    # ----- Model ----
    model = create_model(cfg, num_features, label_to_index)

    print(f"Model parameters: {count_parameters(model)}")

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    # z-score regression targets with train-split statistics; metrics
    # are computed in original units (see compute_target_scaler).
    target_scaler = None
    if cfg.task.task_type == "regression" and cfg.task.standardize_targets:
        target_scaler = compute_target_scaler(train_dl)
        print(f"Target scaler: mean={target_scaler[0]:.4f} std={target_scaler[1]:.4f}")

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        last_epoch = 0
        if cfg.train:
            model, last_epoch = training_loop(train_dl, val_dl, model, accuracies, run, cfg, label_to_index, target_scaler)

        test(model, test_dl, accuracies, label_to_index, run, cfg, epoch=last_epoch, target_scaler=target_scaler)


def training_loop(train_dataloader, val_dataloader, model, accuracies, run, cfg, label_to_index, target_scaler=None):
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


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
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
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg, target_scaler)
        train_metrics['train/epoch_time_s'] = time.perf_counter() - epoch_start

        # Validation
        val_metrics = validate(model, val_dataloader, loss_fn, accuracies, cfg, target_scaler)

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
        else:
            early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(checkpoint_dir, "best_val.pt"), weights_only=True)

    model.load_state_dict(state_dict)
    return model, last_epoch

if __name__ == "__main__":
    main()
