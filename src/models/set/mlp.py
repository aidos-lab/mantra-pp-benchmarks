"""
Multi-Layer Perceptron implementation.
"""

import torch.nn as nn

from src.models.utils import head, pool, target_rows


class MLPLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, batch_norm, dropout, residuals):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_channels)
        self.batch_norm = nn.Identity()
        if batch_norm:
            self.batch_norm = nn.LayerNorm(hidden_channels)

        self.activation =  nn.ReLU()
        self.dropout = nn.Identity()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.residuals = residuals

        if in_channels != hidden_channels:
            self.residuals = False

    def forward(self, x):
        h = x

        h = self.linear(h)
        h = self.batch_norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        if self.residuals:
            h = h + x

        return h

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, batch_norm, residuals, pool_method="mean", task_level="graph", head_layers=0):
        super().__init__()
        self.pool_method = pool_method
        self.task_level = task_level

        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for _ in range(n_layers):
            self.hidden_layers.append(MLPLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, residuals=residuals, dropout=dropout))

        # Input layer
        self.hidden_layers.insert(0, MLPLayer(in_channels, hidden_channels, batch_norm=batch_norm, residuals=False, dropout=dropout))

        self.readout = head(hidden_channels, out_channels, head_layers)



    def forward(self, batch):
        x = batch.x
        for layer in self.hidden_layers:
            x = layer(x)
        if self.task_level == "graph":
            x = pool(x, batch.batch, self.pool_method)
        else:
            x = target_rows(x, batch, self.task_level)
        x = self.readout(x)
        return x
