import json

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

from src.schema import RootConfig
from src.transforms import RemapLabels
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import train, validate, test, init_metrics
from src.utils.gpu_utils import is_gpu, is_blackwell_gpu, set_gpu_configs

from tqdm import tqdm
from torch_geometric.profile import count_parameters

torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])


cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)


torch._dynamo.config.capture_scalar_outputs = True

def create_dataset(cfg):
    dataset_cfg = cfg.dataset
    transform_cfg = dataset_cfg.transform
    transform_order = dataset_cfg.transform_order

    transforms = [None for _ in range(len(list(transform_cfg.items())))]
    assert "SelectFeatures" in [v._target_.split('.')[-1] for k,v in transform_cfg.items()], "Missing `SelectFeatures` transform"

    for name, transform in transform_cfg.items():
        trans = hydra.utils.instantiate(transform)
        assert name in transform_order, f"Transform {name} not in transforms"
        transforms[transform_order.index(name)] = trans

    transforms = Compose(transforms)

    dataset_parameters = {
            "root": dataset_cfg.root,
            "dimension": dataset_cfg.dimension,
            "version": dataset_cfg.version,
            "balanced": dataset_cfg.balanced,
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
    transform_cfg = cfg.dataset.transform
    persistent_workers = trainer_cfg.num_workers > 0

    is_graph_representation = transform_cfg.select_features.representation == 'graph'

    # TODO: Fix this assignment
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

    is_graph_representation = (
        cfg.dataset.transform.select_features.representation == "graph"
    )
    if not is_graph_representation:
        num_feature_channels = cfg.model.in_channels
        in_channels = [num_features] * num_feature_channels # One for each i-simplex
    else:
        in_channels = num_features

    # #TODO: Ugly fix, do something about this
    # # TODO Fix, this is only for the case of SCCNN
    # if is_graph_representation or ("SCCNN" not in cfg.model._target_ and 'CWN' not in cfg.model._target_):
    #     if cfg.dataset.num_features != 'auto':
    #         in_channels = int(cfg.dataset.num_features)
    #     else: #NOTE: Handles the case for SAN
    #         in_channels = dataset[0]['x_1'].shape[1] if hasattr(dataset[0], 'x_1') else dataset[0].x.shape[1]
    # else:
    #     feature_keys = [feat_key for feat_key in dataset[0].keys() if feat_key.split('_')[0] == "x" ]
    #     in_channels = [dataset[0][feat_key].shape[1] for feat_key in feature_keys] if cfg.dataset.num_features == 'auto' else [int(cfg.dataset.num_features)] * len(list(dataset[0].x.keys()))
    #

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

    dataset_cfg = cfg.dataset
    trainer_cfg = cfg.trainer
    transform_cfg = dataset_cfg.transform

    assert not ((dataset_cfg.local_path is not None) ^ dataset_cfg.balanced), "For the balanced dataset, only a local version is supported, while the original dataset always needs to be downloaded."


    # ---- Dataset -----

    train_ds, val_ds, test_ds, label_to_index = create_dataset(cfg)
    train_dl, val_dl, test_dl = create_loaders(cfg, train_ds, val_ds, test_ds)

    #NOTE: Is this the right way ?
    num_features = train_ds[0]['x'].shape[1]

    # ----- Model ----
    model = create_model(cfg, num_features, label_to_index)

    print(f"Model parameters: {count_parameters(model)}")

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        last_epoch = 0
        if cfg.train:
            model, last_epoch = training_loop(train_dl, val_dl, model, accuracies, run, cfg, label_to_index)

        test(model, test_dl, accuracies, label_to_index, run, cfg, epoch=last_epoch)


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

    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        if early_stopping_cnt == 0:
            print("Early stopping...")
            break

        last_epoch = epoch

        # Train
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg)

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
            with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))
        else:
            early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(checkpoint_dir, "best_val.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    return model, last_epoch


if __name__ == "__main__":
    main()
