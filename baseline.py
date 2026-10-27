"""Trivial baselines of a task, logged like model runs.

Takes the overrides of `main.py` (the `model` group is ignored) and logs
one W&B run per baseline with the task's metrics on the test split: the
majority class or the train median, and for `h11` the vertex count
minus five, which is exact for favorable polytopes.
"""

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from torch import nn

from main import create_dataset, create_loaders
from src.schema import RootConfig
from src.utils.train_utils import init_metrics, test


class Constant(nn.Module):
    """Predict `value` for every row: a class index or a regression value."""

    def __init__(self, value, out_channels):
        super().__init__()
        self.value = value
        self.out_channels = out_channels

    def forward(self, batch):
        rows, device = batch.y.shape[0], batch.y.device
        if self.out_channels > 1:
            index = torch.full((rows,), int(self.value), device=device)
            return F.one_hot(index, self.out_channels).float()
        return torch.full((rows, 1), float(self.value), device=device)


class VertexCount(nn.Module):
    """Predict `h11 = n_vertices - 5`, which holds for favorable polytopes."""

    def __init__(self, label_to_index):
        super().__init__()
        self.label_to_index = label_to_index

    def forward(self, batch):
        if batch.coords.shape[0] != int(batch.n_vertices.sum()):
            raise ValueError(
                "`n_vertices` is not the vertex count under this representation; "
                "use representation@dataset.transform.representation=one_skeleton"
            )
        h11 = batch.n_vertices - 5
        if self.label_to_index:
            index = torch.tensor([self.label_to_index.get(int(v), 0) for v in h11])
            return F.one_hot(index.to(h11.device), len(self.label_to_index)).float()
        return h11.float().unsqueeze(-1)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):
    OmegaConf.resolve(cfg)
    train_ds, val_ds, test_ds, ood_ds, label_to_index = create_dataset(cfg)
    _, _, test_dl, ood_dls = create_loaders(cfg, train_ds, val_ds, test_ds, ood_ds)
    y = torch.cat([data.y.reshape(-1) for data in train_ds])

    is_classification = cfg.task.task_type == "classification"
    cfg.model.out_channels = len(label_to_index) if is_classification else 1
    if is_classification:
        baselines = {"majority": Constant(y.long().mode().values, cfg.model.out_channels)}
    else:
        baselines = {"median": Constant(y.median(), 1)}
    if cfg.task.target == "h11" and cfg.task.task_level == "graph":
        baselines["count"] = VertexCount(label_to_index)

    config = OmegaConf.to_container(cfg, resolve=True)
    for name, baseline in baselines.items():
        config["model"] = {"name": "baseline", "kind": name}
        with wandb.init(entity=cfg.logger.entity, project=cfg.logger.project, config=config) as run:
            accuracies = init_metrics(cfg)
            test(baseline, test_dl, accuracies, label_to_index, run, cfg, epoch=0)
            for split, loader in ood_dls.items():
                test(baseline, loader, accuracies, label_to_index, run, cfg, epoch=0, split=split)


if __name__ == "__main__":
    main()
