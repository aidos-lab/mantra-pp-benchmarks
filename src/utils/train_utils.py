import torch
import wandb
import hydra

from collections import defaultdict

from src.utils.logging import log_metrics
from src.schema import RootConfig



def select_supervised(y_hat, batch):
    """ Restrict node-level predictions to the supervised nodes.

    Node-level tasks carry a boolean `node_mask`; graph-level batches
    have no mask and pass through unchanged.
    """
    mask = getattr(batch, 'node_mask', None)
    if mask is not None and y_hat.shape[0] == mask.shape[0]:
        return y_hat[mask], batch.y[mask]
    return y_hat, batch.y


def compute_target_scaler(train_dataloader):
    """ (mean, std) of the supervised regression targets, train split only.

    The model is trained against `(y - mean) / std` and its predictions
    are mapped back with `y_hat * std + mean` before any metric, so the
    reported metrics stay in original label units.
    """
    ys = []
    for data in train_dataloader.dataset:
        y = data.y
        mask = getattr(data, 'node_mask', None)
        if mask is not None and y.shape[0] == mask.shape[0]:
            y = y[mask]
        ys.append(y.detach().float().reshape(-1))
    y = torch.cat(ys)
    std = y.std().item()
    if not (std > 1e-8):  # constant targets: fall back to a no-op scale
        std = 1.0
    return y.mean().item(), std


def scale_targets(y, scaler):
    if scaler is None:
        return y
    mean, std = scaler
    return (y.float() - mean) / std


def unscale_predictions(y_hat, scaler):
    if scaler is None:
        return y_hat
    mean, std = scaler
    return y_hat * std + mean


def train(model, optimizer, train_dataset, loss_fn, accuracies, cfg, target_scaler=None):
    """ Run the training


    """

    model.train()
    metrics = defaultdict(float)
    losses = []


    # Go over each batch
    for batch in train_dataset:
        batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))
        optimizer.zero_grad(set_to_none=True)

        y_hat, y = select_supervised(model(batch), batch)

        loss = loss_fn(y_hat, scale_targets(y, target_scaler))

        losses.append(loss.detach())

        for _, acc in accuracies:
            acc(unscale_predictions(y_hat.detach(), target_scaler), y)

        loss.backward()
        optimizer.step()

    metrics['train/loss'] = torch.stack(losses).mean().item()

    for name, acc in accuracies:
        metrics[f'train/{name}'] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def validate(model,  val_dataset, loss_fn, accuracies, cfg, target_scaler=None):
    """ Run the validation evaluation

    """
    model.eval()
    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        metrics = defaultdict(float)
        losses = []
        for batch in val_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            y_hat, y = select_supervised(model(batch), batch)

            loss = loss_fn(y_hat, scale_targets(y, target_scaler))

            for _, acc in accuracies:
                acc(unscale_predictions(y_hat.detach(), target_scaler), y)

            losses.append(loss.detach())
        
        metrics['val/loss'] = torch.stack(losses).mean().item()

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def test(model, test_dataset, accuracies, label_to_index, run, cfg, target_scaler=None):
    """ Run the test evaluation
    """

    is_classification = cfg.task.task_type == 'classification'

    if not is_classification:
        return test_regression(model, test_dataset, accuracies, run, cfg,
                               target_scaler=target_scaler)

    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        index_to_label = {v: k for k,v in label_to_index.items()}
        y_all = []
        y_hat_all = []
        metrics = dict()

        model.eval()
        for batch in test_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

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
            acc.reset()

    log_metrics(metrics, epoch=cfg.trainer.max_epoch, step=None, run=run, cfg=cfg)

    class_names = list(index_to_label.values())

    run.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_total.cpu().tolist(), preds=y_hat_total.cpu().tolist(), class_names=class_names)
        }
    )

def test_regression(model, test_dataset, accuracies, run, cfg, target_scaler=None):
    """ Run the test evaluation for regression tasks
    """

    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        metrics = dict()

        model.eval()
        for batch in test_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            y_hat, y = select_supervised(model(batch), batch)

            for _, acc in accuracies:
                acc(unscale_predictions(y_hat.detach(), target_scaler), y)

        for name, acc in accuracies:
            metrics[f"test/{name}"] = acc.compute().item()
            acc.reset()

    log_metrics(metrics, epoch=cfg.trainer.max_epoch, step=None, run=run, cfg=cfg)
    # The experiment drivers grep this final-metrics line from the run
    # logs to detect completed runs; keep it stable.
    print({k: round(v, 4) for k, v in metrics.items()})

def init_metrics(cfg: RootConfig):
    """ Initialize the metric classes


    """
    accs = []
    for name, acc in cfg.task.metrics.items():
        if cfg.task.task_type == 'regression':
            metric = hydra.utils.instantiate(acc).to(cfg.trainer.device)
            accs.append((name, metric))
        elif 'accuracy' in acc._target_.lower():
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels).to(cfg.trainer.device)
            accs.append((name, metric))
    return accs

