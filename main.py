import json

import wandb
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np

import torch
import os
from torch_geometric.transforms import Compose


from mantra.datasets import MantraDataset

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import train, validate, test, init_metrics

from tqdm import tqdm
from torch_geometric.profile import count_parameters

torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])


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


def set_gpu_configs(cfg):
    # Only set for CPU
    if not is_gpu(cfg):
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

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):

    dataset_config = cfg.dataset
    transform_dict = dataset_config.transform
    transform_order = dataset_config.transform_order

    assert not ((dataset_config.local_path is not None) ^ dataset_config.balanced), "For the balanced dataset, only a local version is supported, while the original dataset always needs to be downloaded."

    transforms = [None for _ in range(len(list(transform_dict.items())))]

    select_transform = None
    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == 'select_features':
            select_transform = trans

        assert name in transform_order, f"Transform {name} not in transforms"
        transforms[transform_order.index(name)] = trans

    assert select_transform is not None, "Missing the `select_features` transform"

    transforms = Compose(transforms)

    dataset_parameters = {
            "root": dataset_config.root,
            "dimension": dataset_config.dimension,
            "version": dataset_config.version,
            "balanced": dataset_config.balanced,
            "min_sample_per_class": 100,
            "division_type": "none",
            "stratified": dataset_config.use_stratified,
            "max_vertices": 10,
            "n_moves": 5,
            "target_count": 100,
            "use_surgery": True,
            "max_ood_size_per_class": 100,

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


    #TODO: Re-asign labels 


    OmegaConf.resolve(cfg)

    #TODO: Ugly fix, do something about this

    is_graph_representation =  cfg.dataset.transform.select_features.representation == 'graph'


    # TODO: Fix this assignment
    if is not is_graph_representation:
        train_loader = DataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, collate_fn=collate_simplicial, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, collate_fn=collate_simplicial, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False, collate_fn=collate_simplicial, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    else: # Graph
        train_loader = PyGDataLoader(train_dataset, batch_size=cfg.trainer.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        val_loader = PyGDataLoader(val_dataset, batch_size=cfg.trainer.batch_size, shuffle=False,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
        test_loader = PyGDataLoader(test_dataset, batch_size=cfg.trainer.batch_size, shuffle=False,  num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    #TODO: Output classes for the final layer
    # Change the outputs based on classification task
    if cfg.task.task_type == 'classification':
        num_classes = len(label_to_index) # This has to be the train classes
        cfg.model.out_channels = num_classes


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

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        last_epoch = 0
        if cfg.train:
            model, last_epoch = training_loop(train_dataloader, val_dataloader, model, accuracies, run, cfg, label_to_index)

        test(model, test_dataloader, accuracies, label_to_index, run, cfg, epoch=last_epoch)




def training_loop(train_dataloader, val_dataloader, model, accuracies, run, cfg, label_to_index):
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
    loss_fn = hydra.utils.instantiate(cfg.task.loss)

    early_stopping_cnt = cfg.trainer.early_stopping
    last_val_loss = None
    last_epoch = 0

    checkpoint_dir = os.path.join(cfg.checkpoint_path, run.name or run.id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    with open(os.path.join(checkpoint_dir, "label_to_index.json"), "w") as f:
        json.dump(label_to_index, f)

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
