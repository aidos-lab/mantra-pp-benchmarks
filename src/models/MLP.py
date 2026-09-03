"""
Multi-Layer Perceptron implementation.
"""

import torch.nn as nn

from src.models.utils import pool


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
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, batch_norm, residuals, pool_method="mean", readout_activation=True):
        super().__init__()
        self.pool_method = pool_method

        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for _ in range(n_layers):
            self.hidden_layers.append(MLPLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, residuals=residuals, dropout=dropout))

        # Input layer
        self.hidden_layers.insert(0, MLPLayer(in_channels, hidden_channels, batch_norm=batch_norm, residuals=False, dropout=dropout))

        # Readout layer. The MLPLayer head ends in a ReLU, which clamps
        # the output to >= 0; regression tasks need the plain linear
        # head (create_model forces readout_activation=False for them).
        if readout_activation:
            self.readout = MLPLayer(hidden_channels, out_channels, batch_norm=False, residuals=False, dropout=None)
        else:
            self.readout = nn.Linear(hidden_channels, out_channels)



    def forward(self, batch):
        x = batch.x
        for layer in self.hidden_layers:
            x = layer(x)
        x = pool(x, batch.batch, self.pool_method)
        x = self.readout(x)
        return x
