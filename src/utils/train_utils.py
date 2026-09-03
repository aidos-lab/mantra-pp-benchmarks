import torch
import wandb
import hydra

from collections import defaultdict

from src.utils.logging import log_metrics
from src.schema import RootConfig



def select_supervised(y_hat, batch):
    """ Restrict node-level predictions and targets to the supervised nodes.

    The node-level task transforms store a boolean `node_mask` next to
    the per-vertex `y`; graph-level batches carry no mask and pass
    through unchanged.
    """
    mask = getattr(batch, "node_mask", None)
    if mask is None:
        return y_hat, batch.y
    if y_hat.shape[0] != mask.shape[0]:
        raise ValueError(
            f"node_mask covers {mask.shape[0]} vertices but the model "
            f"returned {y_hat.shape[0]} predictions; node-level tasks need "
            "a model with task_level=node and a representation whose nodes "
            "are the vertices"
        )
    return y_hat[mask], batch.y[mask]


def supervised_targets(data):
    """ `y` of one sample restricted to its supervised nodes.
    """
    mask = getattr(data, "node_mask", None)
    return data.y if mask is None else data.y[mask]


def compute_target_scaler(train_dataloader):
    """ (mean, std) of the supervised regression targets, train split only.

    The model is trained against `(y - mean) / std` and its predictions
    are mapped back with `y_hat * std + mean` before any metric, so the
    reported metrics stay in original label units.
    """
    ys = [
        supervised_targets(data).detach().float().reshape(-1)
        for data in train_dataloader.dataset
    ]
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
    batch_sizes = []

    # Go over each batch
    for batch in train_dataset:
        batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))
        optimizer.zero_grad(set_to_none=True)

        y_hat, y = select_supervised(model(batch), batch)

        loss = loss_fn(
            y_hat,
            scale_targets(y, target_scaler),
        )

        losses.append(loss.detach())
        batch_sizes.append(y.shape[0])

        for _, acc in accuracies:
            acc(unscale_predictions(y_hat.detach(), target_scaler), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    losses = torch.stack(losses)
    sizes = torch.tensor(batch_sizes, device=losses.device, dtype=losses.dtype)
    metrics['train/loss'] = (losses * sizes).sum().item() / sizes.sum().item()

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
        batch_sizes = []
        for batch in val_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            y_hat, y = select_supervised(model(batch), batch)

            loss = loss_fn(y_hat, scale_targets(y, target_scaler))

            for _, acc in accuracies:
                acc(unscale_predictions(y_hat.detach(), target_scaler), y)

            losses.append(loss.detach())
            batch_sizes.append(y.shape[0])

        losses = torch.stack(losses)
        sizes = torch.tensor(batch_sizes, device=losses.device, dtype=losses.dtype)
        metrics['val/loss'] = (losses * sizes).sum().item() / sizes.sum().item()

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def test(model, test_dataset, accuracies, label_to_index, run, cfg, epoch, target_scaler=None):
    """ Run the test evaluation
    """

    is_classification = cfg.task.task_type == 'classification'

    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        index_to_label = {}
        y_all = []
        y_hat_all = []
        metrics = dict()

        model.eval()
        for batch in test_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            predictions, y = select_supervised(model(batch), batch)
            predictions = unscale_predictions(predictions, target_scaler)
            y = y.long() if is_classification else y

            if is_classification:
                # Per-sample original labels exist for graph-level tasks
                # only; node-level classes are named by label_to_index.
                if getattr(batch, "node_mask", None) is None:
                    make_labels(index_to_label, y, batch.label)
                y_all.append(y)
                y_hat_all.append(predictions.argmax(dim=1))

            for _, acc in accuracies:
                acc(predictions.detach(), y)

        for name, acc in accuracies:
            metrics[f"test/{name}"] = acc.compute().item()
            acc.reset()

    log_metrics(metrics, epoch=epoch, step=None, run=run, cfg=cfg)

    if not is_classification:
        return

    y_total = torch.cat(y_all)
    y_hat_total = torch.cat(y_hat_all)
    original_label_by_index = {
        index: label for label, index in label_to_index.items()
    }
    class_names = [
        str(index_to_label.get(index, original_label_by_index[index]))
        for index in range(len(label_to_index))
    ]

    run.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_total.cpu().tolist(), preds=y_hat_total.cpu().tolist(), class_names=class_names)
        }
    )


def make_labels(index_to_label, y, labels):
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().reshape(-1).tolist()
    elif not isinstance(labels, (list, tuple)):
        labels = [labels]

    indices = y.detach().cpu().reshape(-1).tolist()
    if len(indices) != len(labels):
        raise ValueError("Each test target must have one corresponding data.label")

    for index, label in zip(indices, labels):
        previous_label = index_to_label.setdefault(index, label)
        assert previous_label == label, f"There's a label mismatch in {label}"

def init_metrics(cfg: RootConfig):
    """ Initialize the metric classes
    """
    accs = []
    is_classification = cfg.task.task_type == 'classification'
    for name, acc in cfg.task.metrics.items():
        # torchmetrics' Accuracy needs the task wiring; the regression
        # accuracies of src.metrics.regression take none.
        if acc._target_.startswith('torchmetrics.') and 'accuracy' in acc._target_.lower():
            assert cfg.task.task_type == 'classification', "Cannot assign accuracy for regression tasks"
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels)
        else:
            metric = hydra.utils.instantiate(acc)
        accs.append((name, metric.to(cfg.trainer.device)))
    return accs
