"""
Implementation for Graph Convolutional Neural Network.
"""

import torch.nn.functional as F
from pydantic import BaseModel
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool, GCN2Conv
from torch_geometric.utils import scatter

from .model_types import ModelType


class GCNConfig(BaseModel):
    type: ModelType = ModelType.GCN
    hidden_channels: int = 256
    num_hidden_layers: int = 5
    num_node_features: int = 1
    out_channels: int = 5
    batch_norm: bool = True
    residuals: bool = True


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual, dropout):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        #assert in_dim == out_dim, "Cannot use residuals here"

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
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

def mean_pool(x, batch):
    return scatter(x, batch, dim=-2, reduce='mean')

class GCN(nn.Module):
    def __init__(self, config: GCNConfig):
        super().__init__()

        self.conv_input = GCNConv(
            config.num_node_features, config.hidden_channels
        )
        self.hidden_layers = nn.ModuleList(
            [
                GCNLayer(config.hidden_channels, config.hidden_channels, config.batch_norm, config.residuals)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.final_linear = nn.Linear(
            config.hidden_channels, config.out_channels
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        # 1. Obtain node embeddings
        x = self.conv_input(x, edge_index)
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.final_linear(x)
        return x

class GCNHydra(nn.Module):
    def __init__(self, hidden_channels, out_channels, num_node_features, layers: int, residuals: bool, batch_norm: bool, dropout: float = 0.5):
        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [
                GCNLayer(hidden_channels, hidden_channels, batch_norm, residuals, dropout)
                for _ in range(layers)
            ]
        )
        self.hidden_layers.insert(0, GCNLayer(num_node_features, hidden_channels, batch_norm, residual=False, dropout=dropout))

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
        x = mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.final_linear(x)
        return x
