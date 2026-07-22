"""PointNet for Calabi-Yau triangulations.

PointNet treats a sample as an unordered *set* of points and never looks
at the connectivity: each point is embedded by a shared MLP, and the
graph-level prediction is a symmetric aggregation over the points. That
makes it the natural geometry-only control for the Calabi-Yau
benchmarks.

The Hodge numbers are properties of the polytope, so a model that only
sees the lattice coordinates can do well on them. The second Chern class
and the intersection numbers, in contrast, depend on the triangulation,
so PointNet is expected to fall behind the models that use the
simplicial structure. The gap between the two is the quantity of
interest.

For `task_level='node'` the per-point features are returned directly
(optionally concatenated with the pooled global feature, as in the
segmentation head of the original architecture), which is what the
per-vertex `c2 . D_i` task needs.
"""

import torch
import torch.nn as nn

from src.models.utils import READOUTS, pool


class PointNetLayer(nn.Module):
    """Shared point-wise transformation: Linear -> Norm -> ReLU -> Dropout."""

    def __init__(self, in_channels, out_channels, batch_norm, dropout):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        # LayerNorm rather than BatchNorm1d: batches hold a variable
        # number of points per complex, and LayerNorm is independent of
        # that, which keeps the model usable at any batch size.
        self.norm = nn.LayerNorm(out_channels) if batch_norm else nn.Identity()
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.linear(x))))


class PointNet(nn.Module):
    """PointNet over the points of a triangulation.

    Parameters
    ----------
    in_channels : int
        Point feature size (the lattice coordinates, plus whatever the
        embedding transform appends).
    hidden_channels : int
        Width of the shared point-wise MLP.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    n_layers : int
        Number of shared point-wise layers.
    task_level : str
        `'graph'` pools to one prediction per complex; `'node'` emits one
        prediction per point.
    readout : str
        Symmetric aggregation used for the graph-level readout. `'sum'`
        keeps it extensive, which matters for the counting-like targets.
    global_feature : bool
        For `task_level='node'`, concatenate the pooled global feature
        onto every point feature before the head (the PointNet
        segmentation head). Ignored for graph-level tasks.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        n_layers: int = 3,
        dropout: float = 0.0,
        batch_norm: bool = True,
        task_level: str = 'graph',
        readout: str = 'sum',
        global_feature: bool = True,
    ):
        super().__init__()
        assert task_level in ('graph', 'node')
        assert readout in READOUTS
        self.task_level = task_level
        self.readout = readout
        self.global_feature = global_feature

        layers = [PointNetLayer(in_channels, hidden_channels, batch_norm, dropout)]
        layers += [
            PointNetLayer(hidden_channels, hidden_channels, batch_norm, dropout)
            for _ in range(n_layers - 1)
        ]
        self.point_layers = nn.ModuleList(layers)

        # The segmentation head sees the point feature concatenated with
        # the pooled global feature.
        head_in = hidden_channels
        if self.task_level == 'node' and self.global_feature:
            head_in = 2 * hidden_channels
        self.final_linear = nn.Linear(head_in, out_channels)

    def forward(self, batch):
        x, batch_idx = batch.x, batch.batch

        # 1. Shared point-wise embedding (no connectivity is used).
        for layer in self.point_layers:
            x = layer(x)

        # 2. Symmetric aggregation.
        if self.task_level == 'graph':
            x = pool(x, batch_idx, self.readout)
        elif self.global_feature:
            global_x = pool(x, batch_idx, self.readout)
            # Broadcast each complex's global feature back to its points.
            x = torch.cat([x, global_x[batch_idx]], dim=-1)

        return self.final_linear(x)
