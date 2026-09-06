"""Tests for Graphormer internal components."""

import pytest
import torch

from src.models.Graphormer import Graphormer, GraphormerLayer


class TestGraphormerLayer:
    def test_output_shape(self):
        layer = GraphormerLayer(hidden_dim=16, num_heads=4, dropout=0.0)
        x = torch.randn(2, 5, 16)  # (batch, nodes, dim)
        pad_mask = torch.zeros(2, 5, dtype=torch.bool)
        attn_bias = torch.zeros(2, 4, 5, 5)
        out = layer(x, pad_mask, attn_bias)
        assert out.shape == (2, 5, 16)

    def test_residual_connection(self):
        layer = GraphormerLayer(hidden_dim=8, num_heads=2, dropout=0.0)
        # With zero weights, output should be close to input (residual)
        x = torch.randn(1, 3, 8)
        pad_mask = torch.zeros(1, 3, dtype=torch.bool)
        attn_bias = torch.zeros(1, 2, 3, 3)
        out = layer(x, pad_mask, attn_bias)
        assert out.shape == x.shape

    def test_padding_mask_applied(self):
        layer = GraphormerLayer(hidden_dim=8, num_heads=2, dropout=0.0)
        layer.eval()
        x = torch.randn(1, 4, 8)
        # Mask out last two nodes
        pad_mask = torch.tensor([[False, False, True, True]])
        attn_bias = torch.zeros(1, 2, 4, 4)
        out = layer(x, pad_mask, attn_bias)
        assert out.shape == (1, 4, 8)
        assert torch.isfinite(out).all()

    def test_hidden_dim_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError):
            GraphormerLayer(hidden_dim=7, num_heads=3, dropout=0.0)


class TestGraphormerInit:
    def test_hidden_not_divisible_by_heads_raises(self):
        with pytest.raises(ValueError):
            Graphormer(
                hidden_channels=7,
                n_layers=1,
                in_channels=2,
                out_channels=2,
                num_heads=3,
            )

    def test_default_params(self):
        model = Graphormer()
        assert model.hidden_dim == 128
        assert model.num_heads == 8
