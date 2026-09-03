"""
Implementation for Graph Convolutional Neural Network.
"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv
from src.models.utils import pool




class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual, dropout):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        #assert in_dim == out_dim, "Cannot use residuals here"

        self.batchnorm_h = nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)

    def forward(self, x, edge_index):
        h_in = x # Saving the residual

        # Actual convolution
        h = self.conv(x = x, edge_index = edge_index)

        # Batch normalization before activation
        h = self.batchnorm_h(h)

        h = F.relu(h)

        if self.dropout is not None:
            h = F.dropout(h, p=self.dropout, training=self.training)
        # Applying residual after activation
        if self.residual:
            h = h_in + h

        return h


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,  n_layers: int, residuals: bool, batch_norm: bool, dropout: float = 0.5, pool_method: str = "mean", task_level: str = "graph"):
        super().__init__()
        assert task_level in ("graph", "node"), f"Unknown task_level {task_level!r}"
        self.pool_method = pool_method
        self.task_level = task_level

        self.hidden_layers = nn.ModuleList(
            [
                GCNLayer(hidden_channels, hidden_channels, batch_norm, residuals, dropout)
                for _ in range(n_layers)
            ]
        )
        self.hidden_layers.insert(0, GCNLayer(in_channels, hidden_channels, batch_norm, residual=False, dropout=dropout))

        self.final_linear = nn.Linear(
            hidden_channels, out_channels
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        # 1. Obtain node embeddings
        #x = self.conv_input(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # 2. Readout layer; node-level tasks keep one embedding per node
        if self.task_level == "graph":
            x = pool(x, batch, self.pool_method)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.final_linear(x)
        return x
