"""
Multi-Layer Perceptron implementation.
"""

import torch.nn as nn
from pydantic import BaseModel
from torch_geometric.nn import pool

from .model_types import ModelType


class MLPConfig(BaseModel):
    type: ModelType = ModelType.MLP
    num_hidden_neurons: int = 64
    num_hidden_layers: int = 3
    num_node_features: int = 1
    out_channels: int = 5


class MLP(nn.Module):
    def __init__(
        self,
        config: MLPConfig,
    ):
        super().__init__()
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(config.num_hidden_neurons, config.num_hidden_neurons)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(
            config.num_hidden_neurons, config.out_channels
        )

    def forward(self, batch):
        x, _, batch = batch.x, batch.edge_index, batch.batch
        x = self.input_layer(x)
        x = nn.functional.relu(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return pool.global_mean_pool(x, batch)


class MLPLayer(nn.Module):
    def __init__(self, input_channels, hidden_channels, batch_norm, dropout, residuals):
        self.linear = nn.Linear(input_channels, hidden_channels)
        self.batch_norm = nn.Identity()
        if batch_norm:
            self.batch_norm = nn.LayerNorm(hidden_channels)

        self.activation =  nn.ReLU()
        self.dropout = nn.Identity()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        self.residuals = residuals

        if input_channels != hidden_channels:
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

class MLPHydra(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, n_layers, dropout, batch_norm, residuals):
        super().__init__()

        self.hidden_layers = nn.ModuleList()

        # Hidden layers
        for _ in n_layers:
            self.hidden_layers.append(MLPLayer(hidden_channels, hidden_channels, batch_norm=batch_norm, residuals=residuals, dropout=dropout))

        # Input layer
        self.hidden_layers.insert(0, MLPLayer(input_channels, hidden_channels, batch_norm=batch_norm, residuals=False, dropout=dropout))

        # Readout layer
        self.readout = MLPLayer(hidden_channels, out_channels, batch_norm=False, residuals=False, dropout=None)



    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.readout(x)
        return x
