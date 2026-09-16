"""PointNet++ over the lattice points of a polytope.

Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point Sets
in a Metric Space", NeurIPS 2017 (arXiv:1706.02413), in the form of
PyTorch Geometric's `pointnet2_classification` / `_segmentation`
examples. Each set-abstraction (SA) level samples centroids by farthest
point sampling, groups the points within a radius of each centroid
(ball query) and encodes every group with a shared PointNet on the
features and the coordinates relative to the centroid; a global level
encodes and pools the surviving points into one feature per set. At
node level, feature-propagation (FP) levels interpolate the features
back to each finer level by inverse squared-distance weighted k-NN
(k = 3), concatenate the skip features of that level and apply a shared
MLP, the "unit PointNet" (Section 3.4; the part-segmentation network of
Appendix B). FPS starts at the first point of each set as in the
original implementation; for the lattice coordinates that is the
origin, so forward passes are deterministic and the hierarchy is
anchored there, but not invariant to reordering the input rows.

Deviations, matching the rest of this repository: LayerNorm instead of
BatchNorm, a configurable `sum`/`mean`/`max` global readout (the paper
pools with max; `sum` keeps the prediction extensive), single-scale
grouping, the coordinates as the input-level skip features, and no
neighbour cap that could truncate a group. Like `DeepSets`, the model
reads only `batch.x` and `batch.batch` (README, "Point-set models").
"""

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, fps, knn_interpolate, radius

from src.models.utils import mlp, pool


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
        if len(sa_ratios) != len(sa_radii):
            raise ValueError(
                "sa_ratios and sa_radii must have the same length"
            )
        self.pool_method = pool_method
        self.task_level = task_level
        self.sa_ratios = list(sa_ratios)
        self.sa_radii = list(sa_radii)
        self.max_num_neighbors = max_num_neighbors
        self.k = k

        h = hidden_channels
        # A group's PointNet sees a point's features and its coordinates
        # relative to the centroid. radius() already pairs each centroid
        # with itself; PyG's self loops would cross the sets of a batch.
        self.sa_convs = nn.ModuleList(
            [
                PointNetConv(
                    mlp([feat + in_channels, h, h], dropout),
                    add_self_loops=False,
                )
                for feat in [0] + [h] * (len(sa_ratios) - 1)
            ]
        )
        self.global_nn = mlp([h + in_channels, h, h], dropout)
        # FP levels run from the global level back to the input points,
        # whose skip features are the coordinates.
        skips = [h] * len(sa_ratios) + [in_channels]
        self.fp_mlps = nn.ModuleList(
            [mlp([h + skip, h, h], dropout) for skip in skips]
            if task_level == "node"
            else []
        )
        self.final_linear = nn.Linear(h, out_channels)

    def forward(self, batch) -> torch.Tensor:
        x, p, b = None, batch.x, batch.batch
        levels = [(p, p, b)]
        sa_levels = zip(self.sa_ratios, self.sa_radii, self.sa_convs)
        for ratio, r, conv in sa_levels:
            idx = fps(p, b, ratio=ratio, random_start=False)
            row, col = radius(
                p,
                p[idx],
                r,
                b,
                b[idx],
                max_num_neighbors=self.max_num_neighbors,
            )
            x_dst = None if x is None else x[idx]
            x = conv((x, x_dst), (p, p[idx]), torch.stack([col, row]))
            p, b = p[idx], b[idx]
            levels.append((x, p, b))

        x = pool(self.global_nn(torch.cat([x, p], dim=1)), b, self.pool_method)
        if self.task_level == "graph":
            return self.final_linear(x)

        # The global level is one point per set at the origin, so the
        # first interpolation copies the set feature to every coarse point.
        p = p.new_zeros(x.shape[0], p.shape[1])
        b = torch.arange(x.shape[0], device=p.device)
        fp_levels = zip(self.fp_mlps, reversed(levels))
        for fp, (x_skip, p_skip, b_skip) in fp_levels:
            x = knn_interpolate(x, p, p_skip, b, b_skip, k=self.k)
            x = fp(torch.cat([x, x_skip], dim=1))
            p, b = p_skip, b_skip
        return self.final_linear(x)
