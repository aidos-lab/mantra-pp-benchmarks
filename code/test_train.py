
import wandb
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np
from collections import defaultdict

import torch
import os
from torch_geometric.transforms import BaseTransform, Compose


from mantra.datasets import ManifoldTriangulations
from mantra.transforms import MomentCurveEmbedding, SelectAttributes, SelectFeatures

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_handling import split_dataset, filter_low_ocurrence_labels
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import train, validate, test, init_metrics

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

    dataset = ManifoldTriangulations(
        root=dataset_config.root,
        dimension=dataset_config.dimension,
        version=dataset_config.version,
        pre_transform=transforms,
        force_reload=dataset_config.reload,
        transform=label_transform
    )

    dataset = filter_low_ocurrence_labels(dataset)

    #TODO: Fix this, shouldn't need to do a pass over the dataset to calculate the labels
    for data in dataset:
        pass

    label_to_index = label_transform.label_to_index

    # Change the outputs based on classification task
    if cfg.task.task_type == 'classification':
        num_classes = len(label_to_index)
        cfg.model.out_channels = num_classes


    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    print(dataset[0])

    #TODO: Ugly fix, do something about this

    is_graph_representation =  cfg.dataset.transform.select_features.representation == 'graph'

    if is_graph_representation:
        in_channels = dataset[0].x.shape[1]  if cfg.dataset.num_features == 'auto' else int(cfg.dataset.num_features)
    else:
        feature_keys = [feat_key for feat_key in dataset[0].keys() if feat_key.split('_')[0] == "x" ]
        in_channels = [dataset[0][feat_key].shape[1] for feat_key in feature_keys] if cfg.dataset.num_features == 'auto' else [int(cfg.dataset.num_features)] * len(list(dataset[0].x.keys()))
        

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, in_channels=in_channels)
    model = model.to(cfg.trainer.device)

    # Only compile on GPU
    if cfg.trainer.device != 'cpu' and cfg.compile:
        model = torch.compile(model, dynamic=True)


    print(f"Model parameters: {count_parameters(model)}")

    collate_fn = None if is_graph_representation else collate_simplicial

    #NOTE: Create checkpoint directory
    if not os.path.exists(cfg.checkpoint_path):
        os.makedirs(cfg.checkpoint_path)

    #NOTE: Dataset initialization
    train_dataloader, val_dataloader, test_dataloader = split_dataset(dataset, cfg, collate_fn=collate_fn)

    imbalance = None
    if cfg.task.imbalanced:
        imbalance = sorted_imbalance_weights(
            dataset
        )
        imbalance = np.array(list(imbalance))
        imbalance = imbalance / np.sum(imbalance)

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        if cfg.train:
            model = training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg)

        test(model, test_dataloader, accuracies, label_to_index, run, cfg)




def training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg):
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

    early_stopping_cnt = cfg.trainer.early_stopping
    last_val_loss = None

    #NOTE: Make directory for checkpoints
    os.makedirs(os.path.join(cfg.checkpoint_path, run.name))

    model = model.to(cfg.trainer.device)
    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        if early_stopping_cnt == 0:
            print("Early stopping...")
            break

        # Train
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg)

        # Validation
        with torch.no_grad():
            val_metrics = validate(model, optimizer, val_dataloader, loss_fn, accuracies, cfg)

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

            # Save weights of the best model
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_path, run.name, "best_val.pt"))

        early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(cfg.checkpoint_path, run.name, "best_val.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    main()
