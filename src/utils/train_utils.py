import torch
import wandb
import hydra

from collections import defaultdict

from src.utils.augment import permute_ids
from src.utils.logging import log_metrics
from src.schema import RootConfig



def _weighted_mean(values, sizes):
    """Sample-weighted mean of per-batch tensors, synced once."""
    values = torch.stack(values)
    weights = torch.as_tensor(
        sizes, device=values.device, dtype=values.dtype
    )
    return ((values * weights).sum() / weights.sum()).item()


def train(model, optimizer, train_dataset, objective, accuracies, cfg):
    """Run one training epoch.

    Parameters
    ----------
    objective : nn.Module
        Training objective mapping ``(logits, embedding, batch)`` to the
        loss and its components; see :mod:`src.metrics.loss`.
    """

    model.train()
    metrics = defaultdict(float)
    losses = []
    components = defaultdict(list)
    batch_sizes = []

    # Go over each batch
    for batch in train_dataset:
        batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))
        if getattr(cfg.trainer, "permute_ids", False):
            # Fresh random vertex labelling for every batch; evaluation
            # keeps the canonical one.
            batch = permute_ids(batch, getattr(cfg.trainer, "id_columns", None))
        optimizer.zero_grad(set_to_none=True)

        y_hat, embedding = model(batch, return_embedding=True)
        y = batch.y

        loss, parts = objective(y_hat, embedding, batch)

        losses.append(loss.detach())
        batch_sizes.append(y.shape[0])
        for name, value in parts.items():
            components[name].append(value)

        for _, acc in accuracies:
            acc(y_hat.detach(), y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    metrics['train/loss'] = _weighted_mean(losses, batch_sizes)
    for name, values in components.items():
        metrics[f'train/loss_{name}'] = _weighted_mean(values, batch_sizes)

    for name, acc in accuracies:
        metrics[f'train/{name}'] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def validate(model, val_dataset, objective, accuracies, cfg):
    """Run the validation evaluation.

    The validation batches are sequential, and the entries of a walk are
    stored consecutively, so a batch still contains the same-walk pairs
    the Pachner objectives need.
    """
    model.eval()
    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        metrics = defaultdict(float)
        losses = []
        components = defaultdict(list)
        batch_sizes = []
        for batch in val_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            y_hat, embedding = model(batch, return_embedding=True)
            y = batch.y

            loss, parts = objective(y_hat, embedding, batch)

            for _, acc in accuracies:
                acc(y_hat.detach(), y)

            losses.append(loss.detach())
            batch_sizes.append(y.shape[0])
            for name, value in parts.items():
                components[name].append(value)

        metrics['val/loss'] = _weighted_mean(losses, batch_sizes)
        for name, values in components.items():
            metrics[f'val/loss_{name}'] = _weighted_mean(values, batch_sizes)

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def test(model, test_dataset, accuracies, label_to_index, run, cfg, epoch, prefix='test'):
    """Run the evaluation of one split.

    Parameters
    ----------
    prefix : str
        Metric prefix; ``"test"`` for the held-out split and
        ``"ood"`` for the subdivided one. The confusion matrix is only
        logged for the test split.
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

            predictions = model(batch)
            y = batch.y.long() if is_classification else batch.y

            if is_classification:
                make_labels(index_to_label, y, batch.label)
                y_all.append(y)
                y_hat_all.append(predictions.argmax(dim=1))

            for _, acc in accuracies:
                acc(predictions.detach(), y)

        for name, acc in accuracies:
            metrics[f"{prefix}/{name}"] = acc.compute().item()
            acc.reset()

    log_metrics(metrics, epoch=epoch, step=None, run=run, cfg=cfg)

    if not is_classification or prefix != 'test':
        return metrics

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

    return metrics


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
    for name, acc in cfg.task.metrics.items():
        if 'accuracy' in acc._target_.lower():
            assert cfg.task.task_type == 'classification', "Cannot assign accuracy for regression tasks"
            metric = hydra.utils.instantiate(acc, task='multiclass' if cfg.model.out_channels > 1 else 'binary', num_classes = cfg.model.out_channels)
        else:
            metric = hydra.utils.instantiate(acc)
        accs.append((name, metric.to(cfg.trainer.device)))
    return accs
