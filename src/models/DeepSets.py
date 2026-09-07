"""DeepSets and PointNet over the lattice points of a polytope.

Zaheer et al., "Deep Sets", NeurIPS 2017 (https://arxiv.org/abs/1703.06114):
a permutation-invariant function of a set is `rho(sum_i phi(x_i))` with
`phi` a shared point-wise network and `rho` a network on the pooled
feature. With max pooling this is the vanilla PointNet of Qi et al.,
"PointNet: Deep Learning on Point Sets for 3D Classification and
Segmentation", CVPR 2017 (https://arxiv.org/abs/1612.00593), without the
input and feature transforms, which have no role for integer lattice
coordinates; `config/model/pointnet.yaml` is that configuration.

The model reads only `batch.x` and `batch.batch`, never the connectivity,
so with `x` set to the lattice coordinates it sees the polytope and
nothing of the triangulation: the geometry-only control for the
Calabi-Yau benchmarks. The Hodge numbers are properties of the polytope
(for a favorable polytope `h11_cy = n_points - 5`, so a sum-pooled model
can count its way to it), while the second Chern class and the
intersection numbers depend on the triangulation, where a point-set
model is expected to fall behind the models that use the simplicial
structure. The gap between the two is the quantity of interest.

At node level every point is predicted from its own embedding
concatenated with the pooled set feature, the segmentation head of
PointNet.
"""

import torch
import torch.nn as nn

from src.models.utils import pool


class PointLayer(nn.Module):
    """Shared point-wise transformation: Linear -> LayerNorm -> ReLU -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_norm: bool,
        dropout: float,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        # LayerNorm rather than BatchNorm: sets have variable size, and
        # per-point statistics keep the layer independent of the batch.
        self.norm = nn.LayerNorm(out_channels) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(torch.relu(self.norm(self.linear(x))))


class DeepSets(nn.Module):
    """DeepSets over a batch of point sets.

    Parameters
    ----------
    in_channels : int
        Point feature size (4 for the lattice coordinates).
    hidden_channels : int
        Width of the shared point-wise network and of the head.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    n_layers : int
        Number of shared point-wise layers (`phi`).
    dropout : float
        Dropout after every hidden layer.
    layer_norm : bool
        LayerNorm after every hidden linear layer.
    pool_method : str
        Symmetric aggregation over the points: `'sum'` (DeepSets, keeps
        the prediction extensive), `'mean'`, or `'max'` (PointNet).
    task_level : str
        `'graph'` for one prediction per set, `'node'` for one per point.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        n_layers: int = 3,
        dropout: float = 0.0,
        layer_norm: bool = True,
        pool_method: str = "sum",
        task_level: str = "graph",
    ):
        super().__init__()
        self.pool_method = pool_method
        self.task_level = task_level

        widths = [in_channels] + [hidden_channels] * n_layers
        self.phi = nn.Sequential(
            *[
                PointLayer(a, b, layer_norm, dropout)
                for a, b in zip(widths[:-1], widths[1:])
            ]
        )
        rho_in = 2 * hidden_channels if task_level == "node" else hidden_channels
        self.rho = nn.Sequential(
            PointLayer(rho_in, hidden_channels, layer_norm, dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, batch) -> torch.Tensor:
        h = self.phi(batch.x)
        g = pool(h, batch.batch, self.pool_method)
        if self.task_level == "node":
            # Each point sees its own embedding and the pooled set feature
            return self.rho(torch.cat([h, g[batch.batch]], dim=1))
        return self.rho(g)
