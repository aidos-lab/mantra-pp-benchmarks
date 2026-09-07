"""PointNet++ over the lattice points of a polytope.

Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
in a Metric Space", NeurIPS 2017 (https://arxiv.org/abs/1706.02413), in
the form of PyTorch Geometric's `pointnet2_classification` and
`pointnet2_segmentation` examples. Set-abstraction (SA) levels sample
centroids by farthest point sampling (FPS), group the points within a
radius of each centroid and encode every group with a shared PointNet;
a global level pools the surviving points into one feature per set. At
node level, feature-propagation (FP) levels interpolate the features
back to the finer levels by inverse-distance weighted k-NN and encode
them with the skip features of that level (Section 3.4 of the paper).

Like `DeepSets`, the model reads only `batch.x` (the coordinates) and
`batch.batch`, so it sees the polytope and nothing of the triangulation.
FPS starts at each set's first point, the origin for the lattice
coordinates, so a forward pass is deterministic; the hierarchy is
anchored at the origin and the output is not invariant to reordering
the input rows. Deviations from the paper matching the other models of
this repository: LayerNorm instead of BatchNorm (sets have variable
size), a configurable `sum`/`mean`/`max` readout (`sum` keeps the
prediction extensive) and single-scale grouping.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, knn_interpolate, radius

from src.models.utils import pool


def _mlp(channels, dropout):
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Linear(c_in, c_out), nn.LayerNorm(c_out), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """One SA level: FPS-sample `ratio` of the points, group the points
    within `r` of each sample and encode the groups with a shared
    PointNet on the features and the coordinates relative to the
    centroid."""

    def __init__(
        self, ratio: float, r: float, local_nn: nn.Module, max_num_neighbors: int
    ):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.max_num_neighbors = max_num_neighbors
        # radius() already pairs every centroid with itself; PyG's self
        # loops would pair flat indices across the sets of a batch.
        self.conv = PointNetConv(local_nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio, random_start=False)
        row, col = radius(
            pos,
            pos[idx],
            self.r,
            batch,
            batch[idx],
            max_num_neighbors=self.max_num_neighbors,
        )
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class FeaturePropagation(nn.Module):
    """One FP level: interpolate the coarse features onto the finer
    points by inverse-distance weighted k-NN and encode them together
    with the skip features of the finer level."""

    def __init__(self, k: int, mlp: nn.Module):
        super().__init__()
        self.k = k
        self.mlp = mlp

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        return self.mlp(torch.cat([x, x_skip], dim=1))


class PointNetPP(nn.Module):
    """PointNet++ over a batch of point sets.

    Parameters
    ----------
    in_channels : int
        Dimension of the point coordinates (4 for the lattice points);
        the points enter the hierarchy as positions only.
    hidden_channels : int
        Width of the shared MLPs.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    dropout : float
        Dropout after every hidden layer.
    sa_ratios, sa_radii : list of float
        FPS sampling ratio and grouping radius per SA level.
    max_num_neighbors : int
        Points kept per group; large enough never to truncate one.
    k : int
        Neighbours of the k-NN interpolation at node level.
    pool_method : str
        Symmetric aggregation of the global level.
    task_level : str
        `'graph'` for one prediction per set, `'node'` for one per point
        through the FP levels.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.0,
        sa_ratios=(0.5, 0.5),
        sa_radii=(2.0, 4.0),
        max_num_neighbors: int = 512,
        k: int = 3,
        pool_method: str = "sum",
        task_level: str = "graph",
    ):
        super().__init__()
        if not sa_ratios or len(sa_ratios) != len(sa_radii):
            raise ValueError(
                "sa_ratios and sa_radii must be non-empty lists of equal length"
            )
        self.pool_method = pool_method
        self.task_level = task_level

        h = hidden_channels
        self.sa_levels = nn.ModuleList()
        feat = 0
        for ratio, r in zip(sa_ratios, sa_radii):
            # local_nn sees the features of a grouped point and its
            # coordinates relative to the centroid.
            self.sa_levels.append(
                SetAbstraction(
                    ratio, r, _mlp([feat + in_channels, h, h], dropout), max_num_neighbors
                )
            )
            feat = h
        self.global_nn = _mlp([h + in_channels, h, h], dropout)

        # FP levels run from the global level back to the input points,
        # whose skip features are the coordinates.
        self.fp_levels = nn.ModuleList()
        if task_level == "node":
            for skip in [h] * len(sa_ratios) + [in_channels]:
                self.fp_levels.append(
                    FeaturePropagation(k, _mlp([h + skip, h, h], dropout))
                )

        self.final_linear = nn.Linear(h, out_channels)

    def forward(self, batch) -> torch.Tensor:
        pos, batch_idx = batch.x, batch.batch

        levels = [(pos, pos, batch_idx)]
        x, p, b = None, pos, batch_idx
        for sa in self.sa_levels:
            x, p, b = sa(x, p, b)
            levels.append((x, p, b))

        x = pool(self.global_nn(torch.cat([x, p], dim=1)), b, self.pool_method)
        if self.task_level == "graph":
            return self.final_linear(x)

        # The global level is one point per set at the origin, so the
        # first FP level copies the set feature to every coarse point.
        p = pos.new_zeros(x.shape[0], pos.shape[1])
        b = torch.arange(x.shape[0], device=pos.device)
        for fp, (x_skip, p_skip, b_skip) in zip(self.fp_levels, reversed(levels)):
            x = fp(x, p, b, x_skip, p_skip, b_skip)
            p, b = p_skip, b_skip
        return self.final_linear(x)
