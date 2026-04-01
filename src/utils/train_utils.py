import torch
import wandb
import hydra

from collections import defaultdict

from src.utils.logging import log_metrics
from src.schema import RootConfig

def train(model, optimizer, train_dataset, loss_fn, accuracies, cfg):
    """ Run the training 


    """

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

    metrics['train/loss'] = torch.tensor(losses).mean().detach().cpu().item()

    for name, acc in accuracies:
        metrics[f'train/{name}'] = acc.compute().detach().cpu().item()

    return metrics

def validate(model, optimizer, val_dataset, loss_fn, accuracies, cfg):
    """ Run the validation evaluation

    """
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

    
    metrics['val/loss'] = torch.tensor(losses).mean().detach().cpu().item()

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().detach().cpu().item()

    return metrics

def test(model, test_dataset, accuracies, label_to_index, run, cfg):
    """ Run the test evaluation


    """

    is_classification = cfg.task.task_type == 'classification'

    if not is_classification:
        raise NotImplementedError("No test for regression yet")

    conf_mat = torch.zeros(cfg.model.out_channels, cfg.model.out_channels) if is_classification else None
    index_to_label = {v: k for k,v in label_to_index.items()}
    y_all = []
    y_hat_all = []
    metrics = dict()

    model.eval()
    for batch in test_dataset:
        batch = batch.to(cfg.trainer.device)

        y_hat = model(batch).argmax(dim=1)
        y = batch.y.long()

        for _, acc in accuracies:
            acc(y_hat, y)

        y_all.append(y)
        y_hat_all.append(y_hat)

        # for t,p in zip(y.view(-1), y_hat.view(-1)):
        #     conf_mat[t.long(), p.long()] += 1


    y_total = torch.cat(y_all)
    y_hat_total = torch.cat(y_hat_all)

    for name, acc in accuracies:
        metrics[f"test/{name}"] = acc.compute().item()

    log_metrics(metrics, epoch=cfg.trainer.max_epoch, step=None, run=run, cfg=cfg)

    class_names = list(index_to_label.values())

    run.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_total.tolist(), preds=y_hat_total.tolist(), class_names=class_names)
        }
    )

def init_metrics(cfg: RootConfig):
    """ Initialize the metric classes


    """
    accs = []
    for name, acc in cfg.task.metrics.items():
        if 'accuracy' in acc._target_.lower():
            assert cfg.task.task_type == 'classification', "Cannot assign accuracy for regression tasks"
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels).to(cfg.trainer.device)
            accs.append((name, metric))
    return accs

