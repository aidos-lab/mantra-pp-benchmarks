"""Differentiable Euler characteristic transform with learnable directions.

Röell and Rieck, "Differentiable Euler Characteristic Transforms for
Shape Classification", ICLR 2024 (arXiv:2310.07630). For a direction `v`
and a threshold `t` the Euler characteristic of the sublevel complex is

    ECT(v, t) = sum_k (-1)^k #{simplices of rank k with height <= t},

where the height of a simplex is the largest `<x, v>` of its vertices.
Replacing the indicator by a sigmoid makes the count differentiable in
the coordinates and in the directions, which are parameters here. The
sum runs over the ranks up to `max_rank`: 0 is the ECT of the point
set, 1 of the one-skeleton, and so on up to the full triangulation.

The `(num_directions, resolution)` image is standardised per entry with
running statistics, because curves of all directions share a level that
carries the size of the sample (see `src.models.ect_vit`). The paper
encodes the image with a 2D CNN. Learned directions have no order, so
the encoder here convolves along the thresholds only and takes the
directions as its input channels.

At node level a point is predicted from an embedding of its coordinates
concatenated with the embedding of the image, as in the node head of
`DeepSets`: the image is the only place the triangulation enters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import head, mlp, target_rows


class DECT(nn.Module):
    """ECT layer with learnable directions, a 1D CNN and a readout.

    Reads the coordinates in `batch.x` and, for `max_rank >= 1`, the
    simplices in `batch.faces_index_<rank>` (`src.transforms.FaceIndex`).

    Parameters
    ----------
    in_channels : int
        Dimension of the coordinates.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    num_directions : int
        Learnable directions, initialised uniformly on the sphere.
    resolution : int
        Thresholds per direction, equally spaced in `[-radius, radius]`.
    radius : float
        Half-width of the threshold interval, in coordinate units.
    scale : float
        Slope of the sigmoid in units of the radius; `resolution` spreads
        a step over about two thresholds.
    max_rank : int
        Highest rank of the simplices that enter the transform.
    hidden_channels : int
        Width of the encoder and of the readout.
    n_layers : int
        Convolution blocks; each halves the threshold axis.
    dropout : float
        Dropout in the readout.
    task_level : str
        `'graph'` for one prediction per sample; `'node'`, `'triangle'`
        or `'candidate'` for one per row of `target_rows`.
    head_layers : int
        Hidden layers of the readout (`head`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_directions: int = 64,
        resolution: int = 64,
        radius: float = 8.0,
        scale: float = 64.0,
        max_rank: int = 4,
        hidden_channels: int = 128,
        n_layers: int = 3,
        dropout: float = 0.0,
        task_level: str = "graph",
        head_layers: int = 0,
    ):
        super().__init__()
        if task_level not in ("graph", "node", "triangle", "candidate"):
            raise ValueError(f"task_level must be 'graph', 'node', 'triangle' or 'candidate', got {task_level!r}")
        if resolution % 2**n_layers:
            raise ValueError(f"resolution must be divisible by 2**n_layers, got {resolution} and {n_layers}")
        self.radius = radius
        self.scale = scale
        self.max_rank = max_rank
        self.task_level = task_level

        self.directions = nn.Parameter(F.normalize(torch.randn(num_directions, in_channels), dim=1))
        self.register_buffer("thresholds", torch.linspace(-1, 1, resolution))
        self.norm = nn.BatchNorm1d(num_directions * resolution, affine=False)

        h = hidden_channels
        blocks = []
        for channels in [num_directions] + [h] * (n_layers - 1):
            blocks += [nn.Conv1d(channels, h, 5, padding=2), nn.BatchNorm1d(h), nn.ReLU(), nn.MaxPool1d(2)]
        self.encoder = nn.Sequential(
            *blocks, nn.Flatten(), mlp([h * resolution // 2**n_layers, h], dropout)
        )
        if task_level != "graph":
            self.point = mlp([in_channels, h, h], dropout)
        readout_in = h if task_level == "graph" else 2 * h
        self.readout = nn.Sequential(mlp([readout_in, h], dropout), head(h, out_channels, head_layers))

    def ect(self, batch) -> torch.Tensor:
        """Image of shape `(num_graphs, num_directions, resolution)`."""
        directions = F.normalize(self.directions, dim=1)
        heights = batch.x.float() @ directions.t() / self.radius
        image = heights.new_zeros(batch.num_graphs, *heights.shape[1:], self.thresholds.shape[0])
        for rank in range(self.max_rank + 1):
            height, sample = heights, batch.batch
            if rank > 0:
                index = batch[f"faces_index_{rank}"]
                height, sample = heights[index].amax(0), batch.batch[index[0]]
            below = torch.sigmoid(self.scale * (self.thresholds - height.unsqueeze(-1)))
            image.index_add_(0, sample, below, alpha=(-1) ** rank)
        return image

    def forward(self, batch) -> torch.Tensor:
        image = self.ect(batch)
        g = self.encoder(self.norm(image.flatten(1)).view_as(image))
        if self.task_level == "graph":
            return self.readout(g)
        z = torch.cat([self.point(batch.x.float()), g[batch.batch]], dim=1)
        return self.readout(target_rows(z, batch, self.task_level))
