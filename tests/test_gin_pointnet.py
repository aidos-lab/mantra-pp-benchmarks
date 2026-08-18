"""Tests for src/models/GIN.py and src/models/PointNet.py.

GIN: readout relations, gradient flow (graph and node level), invalid
arguments. PointNet: exact hand-computed forward value, permutation
invariance/equivariance, connectivity-blindness, node-level heads with
and without the global feature, gradient flow.

Style mirrors tests/test_values.py (plain pytest, small synthetic
graphs, exact-value asserts where feasible).
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models.GIN import GIN, GINLayer
from src.models.PointNet import PointNet, PointNetLayer


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


def _make_batch():
    return Batch.from_data_list([_triangle_graph(), _edge_graph()])


# ===========================================================================
# 1. GIN
# ===========================================================================

class TestGIN:
    def _make_model(self, **overrides):
        kwargs = dict(
            in_channels=2,
            hidden_channels=8,
            out_channels=3,
            n_layers=2,
            residuals=True,
            batch_norm=True,
            dropout=0.0,
        )
        kwargs.update(overrides)
        torch.manual_seed(0)
        return GIN(**kwargs)

    def test_sum_readout_is_n_times_mean_readout(self):
        """With identical weights, pooled_sum = n * pooled_mean per graph,
        so out_sum - b = n * (out_mean - b) after the final linear."""
        model_mean = self._make_model(readout='mean')
        model_sum = self._make_model(readout='sum')
        model_sum.load_state_dict(model_mean.state_dict())
        model_mean.eval()
        model_sum.eval()

        out_mean = model_mean(_make_batch())
        out_sum = model_sum(_make_batch())

        bias = model_mean.final_linear.bias
        n = torch.tensor([[3.0], [2.0]])  # nodes per graph
        assert torch.allclose(out_sum - bias, n * (out_mean - bias), atol=1e-5)

    def test_graph_level_gradients_flow_to_all_parameters(self):
        model = self._make_model()
        out = model(_make_batch())
        loss = out.sum()
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_node_level_output_and_gradients(self):
        model = self._make_model(task_level='node', out_channels=1)
        batch = _make_batch()
        out = model(batch)
        assert out.shape == (5, 1)  # 3 + 2 nodes, per-node predictions
        target = torch.zeros(5, 1)
        loss = nn.MSELoss()(out, target)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_node_and_graph_level_share_backbone(self):
        """Graph-level 'sum' output equals final_linear(sum of node embeddings):
        pooling before vs. after the (linear) head commutes up to the bias."""
        model_node = self._make_model(task_level='node', out_channels=3)
        model_graph = self._make_model(task_level='graph', out_channels=3,
                                       readout='sum')
        model_graph.load_state_dict(model_node.state_dict())
        model_node.eval()
        model_graph.eval()

        node_out = model_node(_make_batch())     # (5, 3) per-node predictions
        graph_out = model_graph(_make_batch())   # (2, 3) pooled predictions

        bias = model_node.final_linear.bias
        summed = torch.stack([
            node_out[:3].sum(dim=0) - 2 * bias,  # 3 nodes: bias added 3x, keep 1
            node_out[3:].sum(dim=0) - 1 * bias,
        ])
        assert torch.allclose(graph_out, summed, atol=1e-5)

    def test_invalid_task_level_rejected(self):
        with pytest.raises(AssertionError):
            self._make_model(task_level='edge')

    def test_invalid_readout_rejected(self):
        with pytest.raises(AssertionError):
            self._make_model(readout='max')

    def test_gin_layer_isolated_node_uses_eps_self_loop(self):
        """With no edges, GINConv reduces to mlp((1 + eps) * x)."""
        torch.manual_seed(0)
        layer = GINLayer(2, 2, batch_norm=True, residual=False, dropout=None)
        layer.eval()
        x = torch.tensor([[1.0, -2.0]])
        empty = torch.empty((2, 0), dtype=torch.long)
        out = layer(x, empty)
        eps = layer.conv.eps
        expected = torch.relu(layer.batchnorm_h(layer.conv.nn((1 + eps) * x)))
        assert torch.allclose(out, expected, atol=1e-6)


# ===========================================================================
# 2. PointNet
# ===========================================================================

class TestPointNet:
    def _make_model(self, **overrides):
        kwargs = dict(
            in_channels=2,
            hidden_channels=8,
            out_channels=1,
            n_layers=2,
            dropout=0.0,
            batch_norm=True,
        )
        kwargs.update(overrides)
        torch.manual_seed(0)
        return PointNet(**kwargs)

    def test_exact_forward_value_with_hand_set_weights(self):
        """1 layer, no norm, identity weights, sum readout:
        graph A: relu([[1,2],[3,-4]]) = [[1,2],[3,0]] -> sum [4,2] -> 4+2 = 6
        graph B: relu([[-1,-1],[2,0]]) = [[0,0],[2,0]] -> sum [2,0] -> 2
        """
        model = PointNet(
            in_channels=2, hidden_channels=2, out_channels=1, n_layers=1,
            dropout=0.0, batch_norm=False, task_level='graph', readout='sum',
        )
        with torch.no_grad():
            model.point_layers[0].linear.weight.copy_(torch.eye(2))
            model.point_layers[0].linear.bias.zero_()
            model.final_linear.weight.copy_(torch.tensor([[1.0, 1.0]]))
            model.final_linear.bias.zero_()
        model.eval()

        out = model(_make_batch())
        # graph A has a third point [0.5, 0.5] -> relu sum adds [0.5, 0.5]
        expected = torch.tensor([[6.0 + 1.0], [2.0]])
        assert torch.allclose(out, expected, atol=1e-6)

    def test_permutation_invariance_graph_level(self):
        model = self._make_model(task_level='graph', readout='sum')
        model.eval()

        graph = _triangle_graph()
        perm = torch.tensor([2, 0, 1])
        permuted = Data(x=graph.x[perm], edge_index=graph.edge_index)

        out_a = model(Batch.from_data_list([graph]))
        out_b = model(Batch.from_data_list([permuted]))
        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_connectivity_blindness(self):
        """PointNet never reads edge_index: rewiring changes nothing."""
        model = self._make_model(task_level='graph', readout='mean')
        model.eval()

        graph = _triangle_graph()
        rewired = Data(
            x=graph.x,
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        out_a = model(Batch.from_data_list([graph]))
        out_b = model(Batch.from_data_list([rewired]))
        assert torch.equal(out_a, out_b)

    def test_node_level_shape_and_equivariance_with_global_feature(self):
        model = self._make_model(task_level='node', global_feature=True)
        model.eval()
        assert model.final_linear.in_features == 16  # 2 * hidden_channels

        batch = _make_batch()
        out = model(batch)
        assert out.shape == (5, 1)

        # Permuting the points of graph A permutes its outputs; the pooled
        # global feature is permutation invariant, so rows just reorder.
        perm = torch.tensor([1, 2, 0])
        graph = _triangle_graph()
        permuted = Data(x=graph.x[perm], edge_index=graph.edge_index)
        out_single = model(Batch.from_data_list([graph]))
        out_permuted = model(Batch.from_data_list([permuted]))
        assert torch.allclose(out_permuted, out_single[perm], atol=1e-5)

    def test_node_level_without_global_feature_is_pointwise(self):
        """global_feature=False: each point's prediction depends only on
        its own coordinates, so shared points give identical rows."""
        model = self._make_model(task_level='node', global_feature=False)
        model.eval()
        assert model.final_linear.in_features == 8  # hidden_channels only

        x = torch.tensor([[1.0, 2.0], [1.0, 2.0], [0.0, 1.0]])
        batch = Batch.from_data_list([
            Data(x=x, edge_index=torch.empty((2, 0), dtype=torch.long))
        ])
        out = model(batch)
        assert out.shape == (3, 1)
        assert torch.allclose(out[0], out[1], atol=1e-6)

    def test_gradients_flow_node_level(self):
        model = self._make_model(task_level='node', global_feature=True)
        out = model(_make_batch())
        loss = nn.MSELoss()(out, torch.zeros_like(out))
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_invalid_arguments_rejected(self):
        with pytest.raises(AssertionError):
            self._make_model(task_level='edge')
        with pytest.raises(AssertionError):
            self._make_model(readout='max')

    def test_pointnet_layer_identity_when_norm_and_dropout_off(self):
        torch.manual_seed(0)
        layer = PointNetLayer(2, 2, batch_norm=False, dropout=0.0)
        assert isinstance(layer.norm, nn.Identity)
        assert isinstance(layer.dropout, nn.Identity)
        x = torch.tensor([[1.0, -1.0]])
        expected = torch.relu(layer.linear(x))
        assert torch.equal(layer(x), expected)
