"""
Residual Gated Graph ConvNets for graph-level prediction.
"""

import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch_geometric.utils import scatter


class ResidualGatedGCNConfig(BaseModel):
    hidden_channels: int = 128
    n_layers: int = 5
    in_channels: int = 1
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
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

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
        n_layers: int = 5,
        in_channels: int = 1,
        out_channels: int = 5,
        batch_norm: bool = True,
        residuals: bool = True,
        dropout: float = 0.5,
        readout: str = 'mean',
    ):
        super().__init__()
        assert readout in ('mean', 'sum')
        self.readout = readout
        self.input_proj = nn.Linear(
            in_channels, hidden_channels
        )
        self.hidden_layers = nn.ModuleList(
            [
                ResidualGatedGCNLayer(
                    in_dim=hidden_channels,
                    out_dim=hidden_channels,
                    batch_norm=batch_norm,
                    residual=residuals,
                )
                for _ in range(n_layers)
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
        x = scatter(x, batch_idx, dim=-2, reduce=self.readout)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)
        return x
