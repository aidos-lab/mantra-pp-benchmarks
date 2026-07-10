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
        batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))
        optimizer.zero_grad(set_to_none=True)

        y_hat = model(batch)
        y = batch.y

        loss = loss_fn(
            y_hat,
            y,
        )

        losses.append(loss.detach())

        for _, acc in accuracies:
            acc(y_hat.detach(), y)

        loss.backward()
        optimizer.step()

    metrics['train/loss'] = torch.stack(losses).mean().item()

    for name, acc in accuracies:
        metrics[f'train/{name}'] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def validate(model,  val_dataset, loss_fn, accuracies, cfg):
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

            y_hat = model(batch)
            y = batch.y

            loss = loss_fn(y_hat, y)

            for _, acc in accuracies:
                acc(y_hat.detach(), y)

            losses.append(loss.detach())
        
        metrics['val/loss'] = torch.stack(losses).mean().item()

    for name, acc in accuracies:
        metrics[f"val/{name}"] = acc.compute().detach().cpu().item()
        acc.reset()

    return metrics

def test(model, test_dataset, accuracies, label_to_index, run, cfg):
    """ Run the test evaluation
    """

    is_classification = cfg.task.task_type == 'classification'

    if not is_classification:
        return test_regression(model, test_dataset, accuracies, run, cfg)

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

def test_regression(model, test_dataset, accuracies, run, cfg):
    """ Run the test evaluation for regression tasks
    """

    for _, acc in accuracies:
        acc.reset()

    with torch.no_grad():
        metrics = dict()

        model.eval()
        for batch in test_dataset:
            batch = batch.to(cfg.trainer.device, non_blocking=(cfg.trainer.device != 'cpu'))

            y_hat = model(batch)
            y = batch.y

            for _, acc in accuracies:
                acc(y_hat.detach(), y)

        for name, acc in accuracies:
            metrics[f"test/{name}"] = acc.compute().item()
            acc.reset()

    log_metrics(metrics, epoch=cfg.trainer.max_epoch, step=None, run=run, cfg=cfg)
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

