"""Tests for GCN and MLP models."""

import torch
from torch_geometric.data import Batch, Data

from src.models.GCN import GCN, GCNLayer
from src.models.MLP import MLP, MLPLayer


def _make_batch():
    g1 = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long),
    )
    g2 = Data(
        x=torch.randn(2, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    return Batch.from_data_list([g1, g2])


class TestGCNLayer:
    def test_output_shape(self):
        layer = GCNLayer(in_dim=3, out_dim=8, batch_norm=True, residual=False, dropout=0.0)
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (4, 8)

    def test_residual_requires_matching_dims(self):
        layer = GCNLayer(in_dim=8, out_dim=8, batch_norm=True, residual=True, dropout=0.0)
        x = torch.randn(3, 8)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_dropout_applied_in_training(self):
        layer = GCNLayer(in_dim=3, out_dim=8, batch_norm=True, residual=False, dropout=0.5)
        layer.train()
        x = torch.randn(4, 3)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        out = layer(x, edge_index)
        assert out.shape == (4, 8)


class TestGCN:
    def test_forward_shape(self):
        model = GCN(in_channels=3, hidden_channels=16, out_channels=5, n_layers=2,
                     residuals=True, batch_norm=True, dropout=0.0)
        model.eval()
        batch = _make_batch()
        out = model(batch)
        assert out.shape == (2, 5)

    def test_output_is_finite(self):
        model = GCN(in_channels=3, hidden_channels=8, out_channels=3, n_layers=1,
                     residuals=False, batch_norm=True, dropout=0.0)
        model.eval()
        out = model(_make_batch())
        assert torch.isfinite(out).all()

    def test_single_graph(self):
        g = Data(
            x=torch.randn(3, 2),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        batch = Batch.from_data_list([g])
        model = GCN(in_channels=2, hidden_channels=8, out_channels=4, n_layers=1,
                     residuals=False, batch_norm=True, dropout=0.0)
        model.eval()
        out = model(batch)
        assert out.shape == (1, 4)


class TestMLPLayer:
    def test_output_shape(self):
        layer = MLPLayer(in_channels=3, hidden_channels=8, batch_norm=True, dropout=0.1, residuals=False)
        x = torch.randn(5, 3)
        out = layer(x)
        assert out.shape == (5, 8)

    def test_residual_connection(self):
        layer = MLPLayer(in_channels=8, hidden_channels=8, batch_norm=True, dropout=None, residuals=True)
        x = torch.randn(3, 8)
        out = layer(x)
        assert out.shape == (3, 8)

    def test_residual_disabled_when_dims_differ(self):
        layer = MLPLayer(in_channels=3, hidden_channels=8, batch_norm=True, dropout=None, residuals=True)
        assert layer.residuals is False


class TestMLP:
    def test_forward_shape(self):
        model = MLP(in_channels=3, hidden_channels=16, out_channels=5, n_layers=2,
                     dropout=0.0, batch_norm=True, residuals=True)
        model.eval()
        batch = _make_batch()
        out = model(batch)
        assert out.shape == (2, 5)

    def test_output_is_finite(self):
        model = MLP(in_channels=3, hidden_channels=8, out_channels=3, n_layers=1,
                     dropout=0.0, batch_norm=False, residuals=False)
        model.eval()
        out = model(_make_batch())
        assert torch.isfinite(out).all()
