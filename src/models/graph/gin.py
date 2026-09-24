
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import MLP, GINConv

from src.models.utils import pool


class GINLayer(nn.Module):

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
        return self.final_linear(x)
