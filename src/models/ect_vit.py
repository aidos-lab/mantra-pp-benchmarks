"""Transformer over the direction curves of a precomputed Euler
characteristic transform.

The ECT of a sample is a `(num_directions, resolution)` image; each
direction's curve is one token, embedded linearly like a ViT patch
(Dosovitskiy et al., ICLR 2021, arXiv:2010.11929). A learnable
positional embedding per direction slot identifies the direction, which
is fixed across samples by the transform's seed. A class token read out
after a pre-norm encoder gives one prediction per sample.

The model reads only `batch.ect`, written by
`mantra.transforms.EulerCharacteristicTransform`; `in_channels` is
accepted for the `create_model` call and ignored.
"""

import torch
import torch.nn as nn


class ECTViT(nn.Module):
    """Vision transformer whose tokens are the direction curves of an ECT.

    Parameters
    ----------
    in_channels : int
        Ignored; the model reads `batch.ect`, not `batch.x`.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    num_directions : int
        Directions of the ECT, the token count.
    resolution : int
        Thresholds per direction, the token input width.
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
        Only `'graph'`: the ECT is a descriptor of the whole sample.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_directions: int,
        resolution: int,
        hidden_channels: int = 128,
        n_layers: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        task_level: str = "graph",
    ):
        super().__init__()
        if task_level != "graph":
            raise ValueError("ECTViT predicts one value per sample, not per node")
        self.num_directions = num_directions
        self.resolution = resolution

        self.embedding = nn.Linear(resolution, hidden_channels)
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
        self.readout = nn.Linear(hidden_channels, out_channels)

    def forward(self, batch):
        ect = batch.ect.float()
        if ect.shape[1:] != (self.num_directions, self.resolution):
            raise ValueError(
                f"ECT of shape {tuple(ect.shape[1:])} does not match the model's "
                f"({self.num_directions}, {self.resolution})"
            )
        tokens = self.embedding(ect) + self.pos_embedding
        cls = self.cls_token.expand(ect.shape[0], -1, -1)
        h = self.embed_dropout(torch.cat([cls, tokens], dim=1))
        h = self.encoder(h)
        return self.readout(self.norm(h[:, 0]))
