"""Tests for src/models/GIN.py: the readout relation, the eps self-loop
of an isolated node, and the node-level head."""

import torch
from torch_geometric.data import Batch, Data

from src.models.GIN import GIN, GINLayer


def _triangle_graph():
    return Data(
        x=torch.tensor([[1.0, 2.0], [3.0, -4.0], [0.5, 0.5]]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        ),
    )


def _edge_graph():
    return Data(
        x=torch.tensor([[-1.0, -1.0], [2.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )


def _graphs():
    return [_triangle_graph(), _edge_graph()]


def _batch(graphs=None):
    return Batch.from_data_list(_graphs() if graphs is None else graphs)


def _gin(task_level="graph", **overrides):
    kwargs = dict(in_channels=2, hidden_channels=8, out_channels=3,
                  n_layers=2, residuals=True, batch_norm=True, dropout=0.0,
                  task_level=task_level)
    kwargs.update(overrides)
    torch.manual_seed(0)
    return GIN(**kwargs)


class TestGIN:
    def test_sum_readout_is_n_times_mean_readout(self):
        """With identical weights, pooled_sum = n * pooled_mean per graph,
        so out_sum - b = n * (out_mean - b) after the final linear."""
        model_mean = _gin(pool_method="mean")
        model_sum = _gin(pool_method="sum")
        model_sum.load_state_dict(model_mean.state_dict())
        model_mean.eval()
        model_sum.eval()

        out_mean = model_mean(_batch())
        out_sum = model_sum(_batch())

        bias = model_mean.final_linear.bias
        n = torch.tensor([[3.0], [2.0]])
        assert torch.allclose(out_sum - bias, n * (out_mean - bias), atol=1e-5)

    def test_gin_layer_isolated_node_uses_eps_self_loop(self):
        """With no edges, GINConv reduces to mlp((1 + eps) * x)."""
        torch.manual_seed(0)
        layer = GINLayer(2, 2, batch_norm=True, residual=False, dropout=0.0)
        layer.eval()
        x = torch.tensor([[1.0, -2.0]])
        empty = torch.empty((2, 0), dtype=torch.long)
        out = layer(x, empty)
        eps = layer.conv.eps
        expected = torch.relu(layer.batchnorm_h(layer.conv.nn((1 + eps) * x)))
        assert torch.allclose(out, expected, atol=1e-6)


class TestGINNodeHead:
    def test_node_outputs_equal_per_graph_outputs(self):
        """Batching must neither mix graphs nor reorder nodes."""
        model = _gin("node").eval()
        batched = model(_batch())
        single = torch.cat([model(_batch([g])) for g in _graphs()])
        assert batched.shape == (5, 3)
        assert torch.allclose(batched, single, atol=1e-5)
