"""PointNet++ for Calabi-Yau triangulations.

Qi et al., "PointNet++: Deep Hierarchical Feature Learning on Point
Sets in a Metric Space", NeurIPS 2017 (https://arxiv.org/abs/1706.02413).
Set-abstraction (SA) levels sample centroids with farthest point
sampling (FPS), group the points within a radius around each centroid,
and encode every group with a shared PointNet; a final global stage
pools the surviving points into one feature per complex.

FPS and radius grouping are implemented here in plain torch (dense
per-batch distance matrices) instead of pulling in the compiled
`torch-cluster` package: the lattice point sets are tiny (tens of
points), where the brute-force versions are exact and fast. FPS starts
deterministically at each complex's first point, so a forward pass is
reproducible without seeding.

Further deviations from the paper, matching the conventions of this
repository: LayerNorm instead of BatchNorm (batches hold a variable
number of points per complex), a configurable `sum`/`mean` readout
instead of max pooling (`sum` keeps the prediction extensive, which
matters for the counting-like CY targets), and single-scale grouping.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv
from torch_geometric.utils import to_dense_batch

from src.models.utils import pool


def _dense(pos, batch):
    """Dense view (B, N, D) with validity mask (B, N) and flat offsets."""
    dense, mask = to_dense_batch(pos, batch)
    counts = mask.sum(dim=1)
    ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    return dense, mask, counts, ptr


def farthest_point_indices(pos, batch, ratio):
    """Indices of `ceil(ratio * n)` FPS-sampled points per graph.

    Deterministic: sampling starts at each graph's first point.
    """
    dense, mask, counts, ptr = _dense(pos, batch)
    n_batch = mask.shape[0]
    k = torch.ceil(ratio * counts.float()).long().clamp(min=1)

    dist = torch.cdist(dense, dense)
    dist.masked_fill_(~mask.unsqueeze(2), float('inf'))  # invalid sources

    rows = torch.arange(n_batch, device=pos.device)
    picked = [torch.zeros(n_batch, dtype=torch.long, device=pos.device)]
    min_dist = dist[:, :, 0].clone()  # distance of every point to the pick set
    for _ in range(1, int(k.max())):
        candidates = min_dist.masked_fill(~mask, -1.0)
        nxt = candidates.argmax(dim=1)
        picked.append(nxt)
        min_dist = torch.minimum(min_dist, dist[rows, nxt])
    picked = torch.stack(picked, dim=1)  # (B, k_max)

    keep = torch.arange(picked.shape[1], device=pos.device) < k.unsqueeze(1)
    return (picked + ptr.unsqueeze(1))[keep]


def radius_edges(pos_src, batch_src, pos_dst, batch_dst, r):
    """Bipartite edges (src -> dst) between points within distance `r`."""
    src, src_mask, _, src_ptr = _dense(pos_src, batch_src)
    dst, dst_mask, _, dst_ptr = _dense(pos_dst, batch_dst)

    dist = torch.cdist(dst, src)  # (B, M, N)
    within = (dist <= r) & dst_mask.unsqueeze(2) & src_mask.unsqueeze(1)
    b, m, n = within.nonzero(as_tuple=True)
    return torch.stack([src_ptr[b] + n, dst_ptr[b] + m])


def _mlp(channels, dropout):
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Linear(c_in, c_out), nn.LayerNorm(c_out), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class SetAbstraction(nn.Module):
    """One SA level: FPS-sample `ratio` of the points, group all points
    within `r` of each sample, encode groups with a shared PointNet
    (relative coordinates concatenated to the features)."""

    def __init__(self, ratio, r, local_nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(local_nn=local_nn)

    def forward(self, x, pos, batch):
        idx = farthest_point_indices(pos, batch, self.ratio)
        edge_index = radius_edges(pos, batch, pos[idx], batch[idx], self.r)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        return x, pos[idx], batch[idx]


class PointNetPP(nn.Module):
    """PointNet++ over the points of a triangulation.

    Parameters
    ----------
    in_channels : int
        Dimension of the point coordinates (4 for the lattice points);
        the points enter the hierarchy as positions only, so the first
        SA level sees relative coordinates.
    hidden_channels : int
        Width of the shared MLPs.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    sa_ratios, sa_radii : list of float
        FPS sampling ratio and grouping radius per SA level.
    pool_method : str
        Symmetric aggregation of the global stage.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        dropout: float = 0.0,
        sa_ratios=(0.5, 0.5),
        sa_radii=(2.0, 4.0),
        pool_method: str = 'sum',
    ):
        super().__init__()
        assert pool_method in ('mean', 'sum'), " Unknwon pooling methods"
        assert len(sa_ratios) == len(sa_radii)
        self.pool_method = pool_method

        h = hidden_channels
        self.sa_levels = nn.ModuleList()
        feat = 0  # points enter as positions only
        for ratio, r in zip(sa_ratios, sa_radii):
            # local_nn input: features of a grouped point plus its
            # coordinates relative to the centroid.
            self.sa_levels.append(
                SetAbstraction(ratio, r, _mlp([feat + in_channels, h, h], dropout))
            )
            feat = h

        # Global stage: encode every surviving point with its absolute
        # coordinates attached, then pool per complex.
        self.global_nn = _mlp([h + in_channels, h, h], dropout)

        self.final_linear = nn.Linear(h, out_channels)

    def forward(self, batch):
        pos, batch_idx = batch.x, batch.batch

        x, p, b = None, pos, batch_idx
        for sa in self.sa_levels:
            x, p, b = sa(x, p, b)

        x = self.global_nn(torch.cat([x, p], dim=1))

        return self.final_linear(pool(x, b, self.pool_method))
