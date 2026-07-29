"""
Implementation of the Graph Isomorphism Network (GIN).

Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019
(https://arxiv.org/abs/1810.10826). Sum aggregation with an MLP update
makes GIN as expressive as the 1-WL test -- a natural upper-bound
control for the message-passing models on the triangulation benchmarks.

The current data chains carry no `edge_attr`, so this is the plain
GINConv; switch to GINEConv if edge features are ever added (only
lattice-geometry features would be safe for the Chern tasks -- the
intersection-number columns are what the targets are computed from).
"""

import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv
from src.models.utils import READOUTS, pool


class GINLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, residual, dropout):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.batch_norm = batch_norm
        self.residual = residual
        self.dropout = dropout

        # The 2-layer MLP update of the original paper; train_eps learns
        # the self-loop weight (1 + eps).
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv = GINConv(mlp, train_eps=True)
        self.batchnorm_h = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        h_in = x  # Saving the residual

        h = self.conv(x=x, edge_index=edge_index)

        # Batch normalization before activation
        h = self.batchnorm_h(h)

        h = F.relu(h)

        if self.dropout is not None:
            h = F.dropout(h, p=self.dropout, training=self.training)
        # Applying residual after activation
        if self.residual:
            h = h_in + h

        return h


class GIN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int, residuals: bool, batch_norm: bool, dropout: float = 0.5, task_level: str = 'graph', readout: str = 'mean'):
        super().__init__()
        assert task_level in ('graph', 'node')
        assert readout in READOUTS
        self.task_level = task_level
        self.readout = readout

        self.hidden_layers = nn.ModuleList(
            [
                GINLayer(hidden_channels, hidden_channels, batch_norm, residuals, dropout)
                for _ in range(n_layers)
            ]
        )
        self.hidden_layers.insert(0, GINLayer(in_channels, hidden_channels, batch_norm, residual=False, dropout=dropout))

        self.final_linear = nn.Linear(
            hidden_channels, out_channels
        )

    def forward(self, batch):
        x, edge_index, batch = batch.x, batch.edge_index, batch.batch
        for layer in self.hidden_layers:
            x = layer(x, edge_index)

        # Readout layer (node-level tasks keep per-node embeddings)
        if self.task_level == 'graph':
            x = pool(x, batch, self.readout)  # [batch_size, hidden_channels]
        x = self.final_linear(x)
        return x
