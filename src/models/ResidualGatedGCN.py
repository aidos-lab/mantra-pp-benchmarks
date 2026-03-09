"""
Residual Gated Graph ConvNets for graph-level prediction.
"""

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch_geometric.nn import global_mean_pool

from .model_types import ModelType


class ResidualGatedGCNConfig(BaseModel):
    hidden_channels: int = 128
    num_hidden_layers: int = 5
    num_node_features: int = 1
    out_channels: int = 5
    batch_norm: bool = True
    residuals: bool = True
    dropout: float = 0.5


class ResidualGatedGCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        batch_norm: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.batch_norm = batch_norm
        self.residual = residual and (in_dim == out_dim)

        self.self_lin = nn.Linear(in_dim, out_dim, bias=True)
        self.msg_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.gate_src = nn.Linear(in_dim, out_dim, bias=True)
        self.gate_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h_in = x

        src_feat = x[src]
        dst_feat = x[dst]
        gates = torch.sigmoid(self.gate_src(src_feat) + self.gate_dst(dst_feat))
        messages = gates * self.msg_lin(src_feat)

        aggregated = x.new_zeros((x.size(0), messages.size(1)))
        aggregated.index_add_(0, dst, messages)

        out = self.self_lin(x) + aggregated
        if self.batch_norm:
            out = self.bn(out)
        out = F.relu(out)
        if self.residual:
            out = out + h_in
        return out


class ResidualGatedGCN(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 128,
        num_hidden_layers: int = 5,
        num_node_features: int = 1,
        out_channels: int = 5,
        batch_norm: bool = True,
        residuals: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.input_proj = nn.Linear(
            num_node_features, hidden_channels
        )
        self.hidden_layers = nn.ModuleList(
            [
                ResidualGatedGCNLayer(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    batch_norm=batch_norm,
                    residual=residuals,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(
            hidden_channels, out_channels
        )
        self.dropout = dropout

    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        x = self.input_proj(x)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)
        x = global_mean_pool(x, batch_idx)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)
        return x
