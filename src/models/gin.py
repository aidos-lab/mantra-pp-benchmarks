"""
Implementation for GIN
"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv
from torch.nn import Sequential
from src.models.utils import mean_pool, sum_pool




class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual

        #assert in_dim == out_dim, "Cannot use residuals here"

        self.mlp = Sequential(
            nn.Linear(self.in_channels, 2 * out_dim),
            nn.BatchNorm1d(2 * out_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * out_dim, out_dim)
        )
        self.conv = GINConv(self.mlp, train_eps=True)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        h_in = x # Saving the residual

        # Actual convolution
        h = self.conv(x = x, edge_index = edge_index)

        # Batch normalization before activation
        h = self.batchnorm_h(h)

        h = F.leaky_relu(h)

        # Applying residual after activation
        if self.residual:
            h = h_in + h

        return h


class GIN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,  n_layers: int, residuals: bool, batch_norm: bool, dropout: float = 0.5):
        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [
                GINLayer(hidden_channels, hidden_channels, batch_norm, residuals, dropout)
                for _ in range(n_layers)
            ]
        )
        self.hidden_layers.insert(0, GINLayer(in_channels, hidden_channels, batch_norm, residual=False, dropout=dropout))

        self.dropout = dropout
        self.lin_1 = nn.Linear(hidden_channels, hidden_channels)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.final_linear = nn.Linear(
            hidden_channels, out_channels
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        # 1. Obtain node embeddings
        #x = self.conv_input(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # 2. Readout layer
        x = sum_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.leaky_relu(self.batch_norm(self.lin_1(x)))
        if self.dropout is not None:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_linear(x)
        return x
