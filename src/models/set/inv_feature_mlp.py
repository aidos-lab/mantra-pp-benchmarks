"""The "MLP with invariant features" of Ewert, Magruder, Maiboroda, Shen,
Singh and Platt, "Group-invariant machine learning on the Kreuzer-Skarke
dataset", Phys. Lett. B 858 (2024) 138996, Section 3.3.

The polytope is the `(n_points, 4)` coordinate matrix. A column encoder
applies one MLP to every point and max-pools over the points, which is
invariant to reordering the points; a row encoder applies one MLP to
every coordinate axis, i.e. to the vector of all points' values along
that axis, and max-pools over the four axes, which is invariant to
permuting the axes. The sum of the two features goes through a decoder
MLP. The paper trained on the 4 x 26 vertex matrices of the polytopes
with 26 vertices and reached 96.9 % accuracy on the first Hodge number.

The row encoder needs a fixed width, so point sets are zero-padded to
`max_points`; with `point_mask` only the marked points (e.g. the hull
vertices of `src.transforms.HullVertices`) enter either encoder.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch


def _relu_mlp(channels):
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers += [nn.Linear(c_in, c_out), nn.ReLU()]
    return nn.Sequential(*layers)


class InvariantFeatureMLP(nn.Module):
    """Column- and row-encoder MLP over the coordinate matrix of a point set.

    Parameters
    ----------
    in_channels : int
        Coordinates per point (4 for the lattice points).
    out_channels : int
        Prediction width; set from the task by `main.py`.
    max_points : int
        Width of the row encoder: point sets are zero-padded to it and
        may not exceed it.
    hidden_channels : int
        Width of both encoders and of the decoder.
    n_layers : int
        Layers per encoder.
    decoder_layers : int
        Hidden layers of the decoder.
    point_mask : str or None
        Boolean batch attribute selecting the points that enter the
        encoders; `None` uses every point.
    task_level : str
        Only `'graph'`: the features are pooled over the points.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        max_points: int,
        hidden_channels: int = 256,
        n_layers: int = 3,
        decoder_layers: int = 3,
        point_mask: str | None = None,
        task_level: str = "graph",
    ):
        super().__init__()
        if task_level != "graph":
            raise ValueError("InvariantFeatureMLP predicts one value per point set")
        self.max_points = max_points
        self.point_mask = point_mask
        h = hidden_channels
        self.column_encoder = _relu_mlp([in_channels] + [h] * n_layers)
        self.row_encoder = _relu_mlp([max_points] + [h] * n_layers)
        decoder = []
        for _ in range(decoder_layers):
            decoder += [nn.Linear(h, h), nn.BatchNorm1d(h), nn.ReLU()]
        self.decoder = nn.Sequential(*decoder, nn.Linear(h, out_channels))

    def forward(self, batch) -> torch.Tensor:
        x, index = batch.x, batch.batch
        if self.point_mask is not None:
            keep = batch[self.point_mask]
            x, index = x[keep], index[keep]
        dense, mask = to_dense_batch(x, index, max_num_nodes=self.max_points)
        columns = self.column_encoder(dense).masked_fill(~mask[..., None], float("-inf"))
        rows = self.row_encoder(dense.transpose(1, 2))
        return self.decoder(columns.amax(dim=1) + rows.amax(dim=1))
