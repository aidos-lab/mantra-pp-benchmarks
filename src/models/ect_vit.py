"""Transformer over the direction curves of a precomputed Euler
characteristic transform.

The ECT of a sample is a `(num_directions, resolution)` image; each
direction's curve is one token, embedded linearly like a ViT patch
(Dosovitskiy et al., ICLR 2021, arXiv:2010.11929). A learnable
positional embedding per direction slot identifies the direction, which
is fixed across samples by the transform's seed. A class token read out
after a pre-norm encoder gives one prediction per sample.

The ECT values are face counts: curves of similar shape in every
direction, offset by a level that carries the sample's size. Fed raw,
that common level dominates every token and the encoder does not
generalise. `main.py` therefore sets per-feature train-split statistics
with `set_input_stats`, and the model standardises every (direction,
threshold) entry before embedding it.

With `num_ranks > 1` the transform's `per_rank` output `(num_ranks,
num_directions, resolution)` is read and the ranks of a direction are
concatenated into its token. At node level the lattice points join the
sequence as tokens of their coordinates (`batch.x`), so every point
attends to the ECT of its polytope and is read out on its own.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from src.models.utils import head, target_rows


class ECTViT(nn.Module):
    """Vision transformer whose tokens are the direction curves of an ECT.

    The model reads `batch.ect`, written by
    `mantra.transforms.EulerCharacteristicTransform`.

    Parameters
    ----------
    in_channels : int
        Width of `batch.x`, the point coordinates; read at node level only.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    num_directions : int
        Directions of the ECT, the token count.
    resolution : int
        Thresholds per direction.
    num_ranks : int
        Rank channels of the ECT: 1 for the Euler characteristic, the
        number of ranks of a `per_rank` transform otherwise. A token is
        the `num_ranks * resolution` values of one direction.
    hidden_channels : int
        Token width.
    n_layers : int
        Encoder layers.
    num_heads : int
        Attention heads per layer.
    mlp_ratio : float
        Feed-forward width as a multiple of `hidden_channels`.
    dropout : float
        Dropout inside the encoder and on the embedded tokens.
    task_level : str
        `'graph'` reads the class token; `'node'` appends one token per
        lattice point and predicts from those.
    head_layers : int
        Hidden layers of the readout (`head`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_directions: int,
        resolution: int,
        num_ranks: int = 1,
        hidden_channels: int = 128,
        n_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        task_level: str = "graph",
        head_layers: int = 0,
    ):
        super().__init__()
        if task_level not in ("graph", "node", "triangle"):
            raise ValueError(f"task_level must be 'graph', 'node' or 'triangle', got {task_level!r}")
        self.num_ranks = num_ranks
        self.task_level = task_level
        shape = (num_ranks, num_directions, resolution) if num_ranks > 1 else (num_directions, resolution)
        self.register_buffer("input_mean", torch.zeros(shape))
        self.register_buffer("input_std", torch.ones(shape))

        self.embedding = nn.Linear(num_ranks * resolution, hidden_channels)
        if task_level != "graph":
            self.point_embedding = nn.Linear(in_channels, hidden_channels)
            self.point_type = nn.Parameter(0.02 * torch.randn(1, 1, hidden_channels))
        self.pos_embedding = nn.Parameter(
            0.02 * torch.randn(1, num_directions, hidden_channels)
        )
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, hidden_channels))
        self.embed_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=num_heads,
            dim_feedforward=int(hidden_channels * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden_channels)
        self.readout = head(hidden_channels, out_channels, head_layers)

    def set_input_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Standardise inputs with per-feature statistics in the shape of
        one sample's ECT; both are saved with the model's state."""
        if mean.shape != self.input_mean.shape or std.shape != self.input_std.shape:
            raise ValueError(
                f"statistics of shape {tuple(mean.shape)} do not match the model's "
                f"{tuple(self.input_mean.shape)}"
            )
        self.input_mean.copy_(mean)
        self.input_std.copy_(std)

    def forward(self, batch):
        ect = batch.ect.float()
        if ect.shape[1:] != self.input_mean.shape:
            raise ValueError(
                f"ECT of shape {tuple(ect.shape[1:])} does not match the model's "
                f"{tuple(self.input_mean.shape)}"
            )
        ect = (ect - self.input_mean) / self.input_std
        if self.num_ranks > 1:
            ect = ect.movedim(1, 2).flatten(2)
        tokens = self.embedding(ect) + self.pos_embedding
        cls = self.cls_token.expand(ect.shape[0], -1, -1)
        h = torch.cat([cls, tokens], dim=1)
        padding = None
        if self.task_level != "graph":
            points, present = to_dense_batch(batch.x.float(), batch.batch)
            h = torch.cat([h, self.point_embedding(points) + self.point_type], dim=1)
            padding = torch.cat([present.new_zeros(h.shape[0], tokens.shape[1] + 1), ~present], dim=1)
        h = self.encoder(self.embed_dropout(h), src_key_padding_mask=padding)
        if self.task_level != "graph":
            points = h[:, tokens.shape[1] + 1 :][present]
            return self.readout(self.norm(target_rows(points, batch, self.task_level)))
        return self.readout(self.norm(h[:, 0]))
