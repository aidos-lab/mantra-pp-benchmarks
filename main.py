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


from mantra.datasets import PachnerWalkDataset
from mantra.representations.internal import SimplexTrie
from mantra.representations.internal.simplex_trie import SimplexNode
from mantra.transforms.task_transforms import BettiToClassTransform, OrientableToClassTransform, NameToClass2MTransform, NameToClass3MTransform

from src.metrics.embedding import evaluate_embeddings
from src.schema import RootConfig
from src.transforms import RemapLabels
from src.utils.logging import log_metrics
from src.utils.data_loading import collate_simplicial
from src.utils.sampling import WalkBatchSampler
from src.utils.train_utils import train, validate, test, init_metrics
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

        

def create_dataset(cfg):
    torch.manual_seed(cfg.torch_seed)

    dataset_cfg = cfg.dataset
    transform_cfg = dataset_cfg.transform
    transform_order = dataset_cfg.transform_order

    transforms = [None for _ in range(len(list(transform_cfg.items())))]
    task_transform = task_to_transform(cfg)

    for name, transform in transform_cfg.items():
        trans = hydra.utils.instantiate(transform)
        assert name in transform_order, f"Transform {name} not in transforms"
        transforms[transform_order.index(name)] = trans

    transforms.insert(0, task_transform)

    transforms = Compose(transforms)

    split = dataset_cfg.train_parameters.split
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
            "walk_length": dataset_cfg.walk_length,
            "moves_per_step": dataset_cfg.moves_per_step,
            "move_weights": dataset_cfg.move_weights,
            "transform": transforms,
            # MantraDataset names its processed files after the
            # subdivision, so the in-distribution splits take it too.
            **OmegaConf.to_container(dataset_cfg.subdivision, resolve=True),

        }

    train_dataset = PachnerWalkDataset(
        split_type="train",
        **dataset_parameters
    )
    val_dataset = PachnerWalkDataset(
            split_type="val",
            **dataset_parameters
    )
    test_dataset = PachnerWalkDataset(
            split_type="test",
            **dataset_parameters
    )

    # One OOD split per requested subdivision. Every variant re-runs
    # process(), which rewrites the identical train/val/test files and
    # adds its own ood file, so they coexist in one cache root.
    ood_datasets = {
        name: PachnerWalkDataset(
            split_type="ood", **{**dataset_parameters, **kwargs}
        )
        for name, kwargs in ood_evaluations(cfg)
    }

    datasets = [train_dataset, val_dataset, test_dataset, *ood_datasets.values()]

    label_to_index = {}

    # If we are doing classification perform a relabaling that
    # enumerates correctly
    if cfg.task.task_type == "classification":
        labels = sorted({int(data.y.item()) for data in train_dataset})
        label_to_index = {label: index for index, label in enumerate(labels)}
        remap_labels = RemapLabels(label_to_index)

        for dataset in datasets:
            dataset.transform = Compose([dataset.transform, remap_labels])

    return train_dataset, val_dataset, test_dataset, ood_datasets, label_to_index


def ood_evaluations(cfg):
    """Return the (metric prefix, dataset override) pairs of the OOD splits.

    `dataset.ood_evaluations` holds the subdivisions of the OOD table,
    keyed by the name their metrics are logged under. When it is empty
    the run falls back to the single split described by
    `dataset.division_type` and `dataset.subdivision`, and to no OOD
    split at all for `division_type=none`.
    """
    table = OmegaConf.to_container(cfg.dataset.ood_evaluations, resolve=True)
    if table:
        return [
            (
                f"ood_{name}",
                {
                    "division_type": spec["division_type"],
                    **(spec.get("subdivision") or {}),
                },
            )
            for name, spec in table.items()
        ]
    if cfg.dataset.division_type == "none":
        return []
    return [("ood", {})]


def create_loaders(cfg, train_dataset, val_dataset, test_dataset, ood_datasets=None):
    trainer_cfg = cfg.trainer
    num_workers = trainer_cfg.num_workers

    is_graph_representation = "x_0" not in train_dataset[0]

    if trainer_cfg.materialize:
        # Run the (deterministic) transforms once instead of on every
        # access, and collate in-process: forked workers would each copy
        # the materialized lists.
        num_workers = 0
        ood_datasets = {name: materialize(dataset, name) for name, dataset in (ood_datasets or {}).items()}
        train_dataset = materialize(train_dataset, "train")
        val_dataset = materialize(val_dataset, "val")
        test_dataset = materialize(test_dataset, "test")

    loader_cls = PyGDataLoader if is_graph_representation else DataLoader
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": trainer_cfg.pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if not is_graph_representation:
        loader_kwargs["collate_fn"] = collate_simplicial

    if trainer_cfg.sampler == "random":
        train_loader = loader_cls(train_dataset, batch_size=trainer_cfg.batch_size, shuffle=True, **loader_kwargs)
    elif trainer_cfg.sampler == "walks":
        # batch_sampler is mutually exclusive with batch_size and shuffle.
        train_sampler = WalkBatchSampler(train_dataset, batch_size=trainer_cfg.batch_size, seed=cfg.seed)
        train_loader = loader_cls(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    else:
        raise ValueError(f"Unknown trainer.sampler '{trainer_cfg.sampler}', expected 'random' or 'walks'")

    val_loader = loader_cls(val_dataset, batch_size=trainer_cfg.batch_size, shuffle=False, **loader_kwargs)
    test_loader = loader_cls(test_dataset, batch_size=trainer_cfg.batch_size, shuffle=False, **loader_kwargs)
    ood_loaders = {
        name: loader_cls(dataset, batch_size=trainer_cfg.batch_size, shuffle=False, **loader_kwargs)
        for name, dataset in (ood_datasets or {}).items()
    }

    return train_loader, val_loader, test_loader, ood_loaders


def materialize(dataset, name):
    """Apply the dataset's transforms once and return the entries as a list.

    The raw payload the transforms consume (`triangulation`, `simplex_trie`)
    is dropped; nothing downstream reads it and it is about half of an
    entry's memory.
    """
    entries = []
    for i in tqdm(range(len(dataset)), desc=f"Materializing {name}"):
        data = dataset[i]
        for key in ("triangulation", "simplex_trie"):
            if key in data:
                del data[key]
        entries.append(data)
    return entries

def create_model(cfg, num_features, label_to_index):
    cfg.model.out_channels = (
        len(label_to_index) if cfg.task.task_type == "classification" else 1
    )

    in_channels = num_features

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, in_channels=in_channels)
    model = model.to(cfg.trainer.device)

    # Only compile on GPU
    if is_blackwell_gpu(cfg) and cfg.compile:
        model = torch.compile(model, dynamic=True)

    return model


def load_checkpoint(model, cfg, label_to_index):
    """Load the best-validation weights of an earlier run into ``model``.

    ``cfg.checkpoint`` is the run's checkpoint directory (``best_val.pt``,
    ``config.yaml`` and ``label_to_index.json`` as ``training_loop``
    writes them). The run must be evaluation-only (``train=False``): the
    point is to score a trained model on splits it was not scored on,
    and the dataset, model and readout overrides of the evaluation must
    match the training run. The label mapping is checked against the
    checkpoint's, so a different split or class filter fails here
    rather than as silently permuted classes.
    """
    if cfg.train:
        raise ValueError(
            "checkpoint= loads trained weights and needs train=False"
        )
    ckpt_dir = cfg.checkpoint
    with open(os.path.join(ckpt_dir, "label_to_index.json")) as f:
        saved = {int(e["label"]): int(e["index"]) for e in json.load(f)}
    if saved != {int(k): int(v) for k, v in label_to_index.items()}:
        raise ValueError(
            f"label_to_index of {ckpt_dir} ({saved}) differs from the one "
            f"of this run ({label_to_index}); the splits do not match"
        )
    state_dict = torch.load(
        os.path.join(ckpt_dir, "best_val.pt"),
        map_location=cfg.trainer.device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    print(f"Loaded {ckpt_dir}/best_val.pt")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    OmegaConf.resolve(cfg)

    set_gpu_configs(cfg)

    # ---- Dataset -----

    train_ds, val_ds, test_ds, ood_ds, label_to_index = create_dataset(cfg)
    train_dl, val_dl, test_dl, ood_dls = create_loaders(cfg, train_ds, val_ds, test_ds, ood_ds)

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

    if cfg.checkpoint:
        load_checkpoint(model, cfg, label_to_index)

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

        for name, loader in ood_dls.items():
            test(model, loader, accuracies, label_to_index, run, cfg, epoch=last_epoch, prefix=name)

        if cfg.trainer.eval_embeddings:
            metrics = evaluate_embeddings(model, train_dl, {"test": test_dl, **ood_dls}, cfg)
            log_metrics(metrics, epoch=last_epoch, step=None, run=run, cfg=cfg)


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
    objective = hydra.utils.instantiate(
        cfg.loss, base_loss=hydra.utils.instantiate(cfg.task.loss)
    )

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
        train_metrics = train(model, optimizer, train_dataloader, objective, accuracies, cfg)

        # Validation
        val_metrics = validate(model, val_dataloader, objective, accuracies, cfg)

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
