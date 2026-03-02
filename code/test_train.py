
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

    label_transform = None
    select_transform = None
    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == 'create_labels':
            label_transform = trans
        elif name == 'select_features':
            select_transform = trans
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

    #TODO: Ugly fix, do something about this
    in_channels = dataset[0].x.shape[1]  if cfg.dataset.num_features == 'auto' else int(cfg.dataset.num_features)

    #NOTE: Model initialization
    model = hydra.utils.instantiate(cfg.model, num_node_features=in_channels)
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

    conf_dict = OmegaConf.to_container(cfg, resolve=True)

    with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=conf_dict) as run:
        run.log({"model_parameters": count_parameters(model)})
        if cfg.train:
            run_id = training_loop(train_dataloader, val_dataloader, imbalance, model, run, cfg)

        test(test_dataloader, model, label_to_index, run, cfg)


def train(model, optimizer, train_dataset, loss_fn, accuracies, cfg):

    model.train()
    metrics = defaultdict(float)
    losses = []
    # Go over each batch
    for batch in train_dataset:
        batch = batch.to(cfg.trainer.device)
        optimizer.zero_grad()

        y_hat = model(batch)

        y_hat = y_hat.squeeze()
        y = batch.y


        loss = loss_fn(
            y_hat,
            y,
        )

        losses.append(loss.detach().cpu().item())

        for _, acc in accuracies:
            acc(y_hat, y)

        loss.backward()
        optimizer.step()

    metrics['train/loss'] = torch.tensor(losses).mean().item()

    for name, acc in accuracies:
        metrics[f'train/{name}'] = acc.compute().item()

    return metrics

def validate(model, optimizer, val_dataset, loss_fn, accuracies, cfg):
    model.eval()
    metrics = defaultdict(float)
    losses = []

    for batch in val_dataset:
        batch = batch.to(cfg.trainer.device)

        y_hat = model(batch).squeeze()
        y = batch.y.long()

        loss = loss_fn(y_hat, y)
        losses.append(loss.detach().cpu().item())

        for _, acc in accuracies:
            acc(y_hat, y)

    
    metrics['val/loss'] = torch.tensor(losses).mean().item()

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().item()

    return metrics

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
        batch = batch.to(cfg.trainer.device)

        y_hat = model(batch).argmax(dim=1)
        y = batch.y.long()

        y_all.append(y)
        y_hat_all.append(y_hat)

        # for t,p in zip(y.view(-1), y_hat.view(-1)):
        #     conf_mat[t.long(), p.long()] += 1


    y_total = torch.cat(y_all)
    y_hat_total = torch.cat(y_hat_all)


    class_names = list(index_to_label.values())

    run.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_total.tolist(), preds=y_hat_total.tolist(), class_names=class_names)
        }
    )


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
        
    model = model.to(cfg.trainer.device)
    # Go over each epoch
    for epoch in (pbar := tqdm(range(MAX_EPOCH))):

        # Train
        train_metrics = train(model, optimizer, train_dataloader, loss_fn, accuracies, cfg)

        # Validation
        with torch.no_grad():
            val_metrics = validate(model, optimizer, val_dataloader, loss_fn, accuracies, cfg)

        pbar_str = ""
        for k,v in {**train_metrics, **val_metrics}.items():
            if 'loss' in k:
                pbar_str += f"{k}: {v:.5f},"
            else:
                pbar_str += f"{k}: {v:.2f},"

        pbar.set_description(pbar_str)

        log_metrics({**train_metrics, **val_metrics}, epoch=epoch, step=None, run=run, cfg=cfg)

        scheduler.step(val_metrics['val/loss'])

def init_metrics(cfg):
    accs = []
    for name, acc in cfg.task.metrics.items():
        if 'accuracy' in acc._target_.lower():
            assert cfg.task.task_type == 'classification', "Cannot assign accuracy for regression tasks"
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels).to(cfg.trainer.device)
            accs.append((name, metric))
    return accs

if __name__ == "__main__":
    main()
