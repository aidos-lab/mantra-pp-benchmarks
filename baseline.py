"""Trivial baselines of a task, logged like model runs.

Takes the overrides of `main.py` (the `model` group is ignored) and logs
one W&B run per baseline with the task's metrics on the test split: the
majority class or the train median, for `h11` the vertex count minus
five, which is exact for favorable polytopes, and on a dataset with
several phases per polytope (`dataset.group_by`) the phase-blind oracle
of the per-vertex and per-triangle tasks.
"""

import statistics
from collections import defaultdict

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


class PhaseOracle(nn.Module):
    """Phase-blind ceiling of a per-simplex task on several phases per polytope.

    Predicts for a vertex, boundary triangle or candidate triangle the
    median of its target over the phases of its polytope: the best MAE any
    predictor that sees the point set and the simplex but not the phase
    can reach (the mode would be the optimum of the rounded accuracy).
    Reads the targets of the split it is evaluated on, so it is a
    ceiling, not a model.
    """

    def __init__(self, dataset, task_level):
        super().__init__()
        self.task_level = task_level
        self.index = "candidate_index_2" if task_level == "candidate" else "simplex_index_2"
        values = defaultdict(list)
        for data in dataset:
            for key, y in zip(self.keys(data), data.y.reshape(-1).tolist()):
                values[(int(data.polytope_id), key)].append(y)
        self.medians = {k: statistics.median(v) for k, v in values.items()}

    def keys(self, data):
        """One key per target row: the vertex index or the triangle."""
        if self.task_level in ("triangle", "candidate"):
            return map(tuple, data[self.index].t().tolist())
        return range(data.y.shape[0])

    def forward(self, batch):
        # Simplicial batches carry the vertex-to-sample map as batch_0.
        node_sample = batch.batch if batch.batch is not None else batch.batch_0
        ptr = torch.cat([node_sample.new_zeros(1), torch.bincount(node_sample).cumsum(0)])
        if self.task_level in ("triangle", "candidate"):
            # The triangle index is offset by the node counts when collated.
            tri = batch[self.index]
            sample = node_sample[tri[0]]
            keys = map(tuple, (tri - ptr[sample]).t().tolist())
        else:
            sample = node_sample
            local = torch.arange(sample.shape[0], device=sample.device)
            keys = (local - ptr[sample]).tolist()
        pids = batch.polytope_id[sample].tolist()
        medians = [self.medians[(pid, key)] for pid, key in zip(pids, keys)]
        return torch.tensor(medians, device=batch.y.device).unsqueeze(-1)


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
    if cfg.task.task_level in ("node", "triangle", "candidate") and cfg.dataset.group_by:
        baselines["phase_median_oracle"] = PhaseOracle(test_ds, cfg.task.task_level)

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
