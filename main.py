import json
import time

import wandb
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np

import torch
import os
from torch_geometric.transforms import Compose


from mantra.datasets import CY, ManifoldTriangulations
from mantra.representations.internal.simplex_trie import SimplexNode, SimplexTrie

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_handling import split_dataset, filter_low_ocurrence_labels
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import (
    train, validate, test, init_metrics, compute_target_scaler,
)

from tqdm import tqdm
from torch_geometric.profile import count_parameters

torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])

# With many dataloader workers the default file_descriptor sharing
# strategy can exhaust `ulimit -n` ("received 0 items of ancdata"),
# while file_system proved fragile on shared nodes (/dev/shm segments
# unlinked underneath running jobs -> "could not unlink the shared
# memory file"). Data loads in the main process by default; this is the
# escape hatch for runs that do use workers:
if os.environ.get("TORCH_SHARING_STRATEGY"):
    torch.multiprocessing.set_sharing_strategy(os.environ["TORCH_SHARING_STRATEGY"])


cs = ConfigStore().instance()
cs.store(name="main_conf", node=RootConfig)

def is_blackwell_gpu(cfg):
    if cfg.trainer.device == 'cpu':
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability()
    return cap[0] >= 10

def is_gpu(cfg):
    if cfg.trainer.device == 'cpu':
        return False
    if not torch.cuda.is_available():
        return False
    return True

torch._dynamo.config.capture_scalar_outputs = True


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    dataset_config = cfg.dataset
    transform_dict = dataset_config.transform
    transform_order = dataset_config.transform_order

    is_cy = dataset_config.get("type", "mantra") == "cy"

    if not is_cy:
        assert not ((dataset_config.local_path is not None) ^ dataset_config.balanced), "For the balanced dataset, only a local version is supported, while the original dataset always needs to be downloaded."

    if cfg.task.task_type == 'regression' and 'CrossEntropy' in cfg.task.loss._target_:
        raise RuntimeError(
            "task_type=regression combined with CrossEntropyLoss: the "
            "default loss config is still active and would silently "
            "produce a zero loss. Add `loss=mse` to the overrides."
        )

    if cfg.task.imbalanced:
        raise RuntimeError(
            "cfg.task.imbalanced=True is a legacy option and is not supported: "
            "filter_low_ocurrence_labels drops classes after label_to_index is "
            "built, so the class-weight tensor and model.out_channels disagree "
            "and cross-entropy crashes. All published models were trained with "
            "imbalanced=False; set it to False (the new default for task=name)."
        )

    # Only set for CPU. os.cpu_count() reports all node cores; under a
    # cgroup (SLURM) size the thread pool to the actual allocation.
    if not is_gpu(cfg):
        if hasattr(os, "sched_getaffinity"):
            torch.set_num_threads(len(os.sched_getaffinity(0)))
        else:
            torch.set_num_threads(os.cpu_count())

    # Set config GPU
    if is_gpu(cfg):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if is_blackwell_gpu(cfg):
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


    transforms = [None for _ in range(len(list(transform_dict.items())))]

    label_transform = None
    select_transform = None
    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == 'create_labels':
            label_transform = trans
        elif name == 'select_features':
            select_transform = trans

        assert name in transform_order, f"Transform {name} not in transforms"
        transforms[transform_order.index(name)] = trans

    assert label_transform is not None, "Missing the `create_labels` transform"
    assert select_transform is not None, "Missing the `select_features` transform"

    transforms = Compose(transforms)

    if is_cy:
        dataset = CY(
            root=dataset_config.root,
            version=dataset_config.version,
            local_path=dataset_config.local_path,
            limit=dataset_config.get("limit", None),
            pre_transform=transforms,
            force_reload=dataset_config.reload,
            transform=label_transform
        )
    else:
        dataset = ManifoldTriangulations(
            balanced=dataset_config.balanced,
            root=dataset_config.root,
            dimension=dataset_config.dimension,
            version=dataset_config.version,
            local_path=dataset_config.local_path,
            pre_transform=transforms,
            force_reload=dataset_config.reload,
            transform=label_transform
        )

    label_to_index = label_transform.label_to_index

    # Change the outputs based on the task type
    if cfg.task.task_type == 'classification':
        dataset = filter_low_ocurrence_labels(dataset)

        # Populate `label_to_index` by touching every sample once.
        # Clear it first: on a cache-building run the pre-transform has
        # already registered every pre-filter label, which would size
        # the model head differently than a rerun on the same cache.
        label_to_index.clear()
        for data in dataset:
            pass

        num_classes = len(label_to_index)
        cfg.model.out_channels = num_classes
    else:
        cfg.model.out_channels = int(dataset[0].y.shape[-1])

        # A ReLU on the output head clamps predictions to >= 0, which can
        # never fit regression targets; force the plain linear head on
        # models that expose the switch (MLP).
        if 'readout_activation' in cfg.model:
            cfg.model.readout_activation = False


    OmegaConf.resolve(cfg)

    #TODO: Ugly fix, do something about this

    is_graph_representation =  cfg.dataset.transform.select_features.representation == 'graph'

    # TODO Fix, this is only for the case of SCCNN
    if is_graph_representation or ("SCCNN" not in cfg.model._target_ and 'CWN' not in cfg.model._target_):
        if cfg.dataset.num_features != 'auto':
            in_channels = int(cfg.dataset.num_features)
        else: #NOTE: Handles the case for SAN
            in_channels = dataset[0]['x_1'].shape[1] if hasattr(dataset[0], 'x_1') else dataset[0].x.shape[1]
    else:
        feature_keys = [feat_key for feat_key in dataset[0].keys() if feat_key.split('_')[0] == "x" ]
        in_channels = [dataset[0][feat_key].shape[1] for feat_key in feature_keys] if cfg.dataset.num_features == 'auto' else [int(cfg.dataset.num_features)] * len(list(dataset[0].x.keys()))
        

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, in_channels=in_channels)
    model = model.to(cfg.trainer.device)

    # Only compile on GPU
    if is_blackwell_gpu(cfg) and cfg.compile:
        model = torch.compile(model, dynamic=True)


    print(f"Model parameters: {count_parameters(model)}")

    collate_fn = None if is_graph_representation else collate_simplicial

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    #NOTE: Dataset initialization
    train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset, cfg, pin_memory= (cfg.trainer.device != 'cpu'), collate_fn=collate_fn)

    imbalance = None
    if cfg.task.imbalanced:
        imbalance = sorted_imbalance_weights(
            dataset
        )
        imbalance = np.array(list(imbalance))
        imbalance = imbalance / np.sum(imbalance)

    # z-score regression targets with train-split statistics; metrics
    # are computed in original units (see compute_target_scaler).
    target_scaler = None
    if cfg.task.task_type == 'regression' and cfg.task.standardize_targets:
        target_scaler = compute_target_scaler(train_dataloader)
        print(f"Target scaler: mean={target_scaler[0]:.4f} std={target_scaler[1]:.4f}")

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        if cfg.train:
            model = training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg, label_to_index, target_scaler)

        test(model, test_dataloader, accuracies, label_to_index, run, cfg, target_scaler)




def training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg, label_to_index, target_scaler=None):
    MAX_EPOCH = cfg.trainer.max_epoch

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

    early_stopping_cnt = cfg.trainer.early_stopping
    last_val_loss = None

    checkpoint_dir = os.path.join(cfg.checkpoint_path, run.name or run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "label_to_index.json"), "w") as f:
        # Multi-target labels are tuples; JSON keys must be strings.
        json.dump({str(k): v for k, v in label_to_index.items()}, f)

    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        if early_stopping_cnt == 0:
            print("Early stopping...")
            break

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
            with open(os.path.join(checkpoint_dir, "config.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg))

        early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(checkpoint_dir, "best_val.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    main()
