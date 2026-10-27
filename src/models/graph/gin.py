"""Graph Isomorphism Network (GIN).

Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019
(arXiv:1810.00826), Eq. 4.1: `h_v = MLP((1 + eps) h_v + sum_u h_u)` with
a learned eps (GIN-eps), a 2-layer MLP and BatchNorm on every hidden
layer. Sum aggregation with an MLP update makes GIN as expressive as the
1-WL test, an upper-bound control for the message-passing models on the
triangulation benchmarks.

Deviations, for consistency with the other models of this repository:
the graph readout pools only the final layer (Eq. 4.2 concatenates the
readouts of every layer) with `pool_method`, sum by default as in the
paper's bioinformatics setting, dropout follows every layer, and
residual connections are available. The data carries no
`edge_attr`, so this is the plain GINConv.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MLP, GINConv

from src.models.utils import pool, target_nodes


class GINLayer(nn.Module):
    """GINConv -> BatchNorm -> ReLU -> Dropout, with an optional residual."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        batch_norm: bool,
        residual: bool,
        dropout: float,
    ):
        super().__init__()
        self.residual = residual
        self.dropout = dropout

        # The 2-layer MLP update of the paper; train_eps learns the
        # self-loop weight (1 + eps).
        mlp = MLP(
            [in_dim, out_dim, out_dim],
            norm="batch_norm" if batch_norm else None,
            plain_last=True,
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.batchnorm_h = (
            nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        h = F.relu(self.batchnorm_h(self.conv(x=x, edge_index=edge_index)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.residual:
            h = x + h
        return h


class GIN(nn.Module):
    """GIN with a graph-level or node-level head.

    Parameters
    ----------
    in_channels, hidden_channels, out_channels : int
        Input, hidden and prediction widths; `out_channels` is set from
        the task by `main.py`.
    n_layers : int
        Hidden GIN layers after the input layer.
    residuals, batch_norm : bool
        Residual connections and BatchNorm in the hidden layers.
    dropout : float
        Dropout after every layer.
    pool_method : str
        Graph readout: `'mean'`, `'sum'` or `'max'`.
    task_level : str
        `'graph'` pools before the readout, `'node'` emits one row per
        target node (see `target_nodes`).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_layers: int,
        residuals: bool,
        batch_norm: bool,
        dropout: float = 0.5,
        pool_method: str = "sum",
        task_level: str = "graph",
    ):
        super().__init__()
        self.pool_method = pool_method
        self.task_level = task_level

        h = hidden_channels
        self.hidden_layers = nn.ModuleList(
            [GINLayer(in_channels, h, batch_norm, False, dropout)]
            + [
                GINLayer(h, h, batch_norm, residuals, dropout)
                for _ in range(n_layers)
            ]
        )

        self.final_linear = nn.Linear(h, out_channels)

    def forward(self, batch) -> torch.Tensor:
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        if self.task_level == "graph":
            x = pool(x, batch_idx, self.pool_method)
        else:
            x = target_nodes(x, batch)
        return self.final_linear(x)
