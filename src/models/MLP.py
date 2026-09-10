"""
Multi-Layer Perceptron implementation.
"""

import torch.nn as nn
from torch_geometric.utils import scatter


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
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dropout, batch_norm, residuals):
        super().__init__()

        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for _ in range(n_layers):
            self.hidden_layers.append(MLPLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, residuals=residuals, dropout=dropout))

        # Input layer
        self.hidden_layers.insert(0, MLPLayer(in_channels, hidden_channels, batch_norm=batch_norm, residuals=False, dropout=dropout))

        # Readout layer
        self.readout = MLPLayer(hidden_channels, out_channels, batch_norm=False, residuals=False, dropout=None)



    def forward(self, batch, return_embedding: bool = False):
        """Return the logits, and with `return_embedding` the pooled
        graph embedding alongside them."""
        x = batch.x
        for layer in self.hidden_layers:
            x = layer(x)
        embedding = scatter(x, batch.batch, reduce='mean')
        logits = scatter(self.readout(x), batch.batch, reduce='mean')
        return (logits, embedding) if return_embedding else logits
