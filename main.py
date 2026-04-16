import wandb
from omegaconf import OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

import numpy as np

import torch
import os
from torch_geometric.transforms import Compose


from mantra.datasets import ManifoldTriangulations
from mantra.representations.internal.simplex_trie import SimplexNode, SimplexTrie

from src.schema import RootConfig
from src.utils.imbalance_handling import sorted_imbalance_weights
from src.utils.logging import log_metrics
from src.utils.data_handling import split_dataset, filter_low_ocurrence_labels
from src.utils.data_loading import collate_simplicial
from src.utils.train_utils import train, validate, test, init_metrics

from tqdm import tqdm
from torch_geometric.profile import count_parameters

#torch.serialization.add_safe_globals([SimplexNode, SimplexTrie])


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

    assert not ((dataset_config.local_path is not None) ^ dataset_config.balanced), "For the balanced dataset, only a local version is supported, while the original dataset always needs to be downloaded."

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
        balanced=dataset_config.balanced,
        root=dataset_config.root,
        dimension=dataset_config.dimension,
        version=dataset_config.version,
        local_path=dataset_config.local_path,
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

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        accuracies = init_metrics(cfg)
        if cfg.train:
            model = training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg)

        test(model, test_dataloader, accuracies, label_to_index, run, cfg)




def training_loop(train_dataloader, val_dataloader, imbalance, model, accuracies, run, cfg):
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

    checkpoint_dir = os.path.join(cfg.checkpoint_path, run.name)
    os.makedirs(checkpoint_dir)

    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        if early_stopping_cnt == 0:
            print("Early stopping...")
            break

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

        early_stopping_cnt -= 1

    # Save model with best validation
    state_dict = torch.load(os.path.join(checkpoint_dir, "best_val.pt"), weights_only=True)
    model.load_state_dict(state_dict)
    return model


if __name__ == "__main__":
    main()
