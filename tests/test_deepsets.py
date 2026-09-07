"""Tests for src/models/DeepSets.py: exact forward values with hand-set
weights, permutation invariance, connectivity blindness, the node-level
head, and gradient flow."""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.models.DeepSets import DeepSets


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


def _model(**overrides):
    kwargs = dict(in_channels=2, hidden_channels=8, out_channels=3,
                  n_layers=2, dropout=0.0, layer_norm=True, pool_method="sum")
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepSets(**kwargs)


def _identity_model(pool_method):
    """Width-2 model: phi and the rho hidden layer are ReLU(identity),
    the readout sums the two features."""
    model = DeepSets(in_channels=2, hidden_channels=2, out_channels=1,
                     n_layers=1, dropout=0.0, layer_norm=False,
                     pool_method=pool_method)
    with torch.no_grad():
        for layer in (model.phi[0], model.rho[0]):
            layer.linear.weight.copy_(torch.eye(2))
            layer.linear.bias.zero_()
        model.rho[1].weight.fill_(1.0)
        model.rho[1].bias.zero_()
    return model.eval()


class TestDeepSets:
    def test_exact_forward_value_sum(self):
        """Graph A: relu of the points sums to [4.5, 2.5] -> 7; graph B:
        relu([[-1,-1],[2,0]]) sums to [2, 0] -> 2."""
        out = _identity_model("sum")(_batch())
        assert torch.allclose(out, torch.tensor([[7.0], [2.0]]))

    def test_exact_forward_value_max(self):
        """Same weights with max pooling: graph A max is [3, 2] -> 5,
        graph B [2, 0] -> 2."""
        out = _identity_model("max")(_batch())
        assert torch.allclose(out, torch.tensor([[5.0], [2.0]]))

    @pytest.mark.parametrize("pool_method", ["sum", "mean", "max"])
    def test_permutation_invariance(self, pool_method):
        model = _model(pool_method=pool_method).eval()
        graph = _triangle_graph()
        perm = torch.tensor([2, 0, 1])
        permuted = Data(x=graph.x[perm], edge_index=graph.edge_index)
        out = model(_batch([graph]))
        out_perm = model(_batch([permuted]))
        assert torch.allclose(out, out_perm, atol=1e-6)

    def test_connectivity_blindness(self):
        model = _model().eval()
        rewired = [
            Data(x=g.x, edge_index=torch.empty((2, 0), dtype=torch.long))
            for g in _graphs()
        ]
        assert torch.equal(model(_batch()), model(_batch(rewired)))

    def test_readout_is_linear(self):
        assert isinstance(_model().rho[-1], nn.Linear)

    def test_gradients_flow_to_all_parameters(self):
        model = _model()
        model(_batch()).sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_invalid_pool_method_rejected(self):
        with pytest.raises(ValueError):
            _model(pool_method="median")(_batch())


class TestDeepSetsNodeHead:
    def test_node_level_emits_one_row_per_point(self):
        model = _model(task_level="node").eval()
        out = model(_batch())
        assert out.shape == (5, 3)
        assert torch.isfinite(out).all()

    def test_node_outputs_equal_per_graph_outputs(self):
        model = _model(task_level="node").eval()
        batched = model(_batch())
        single = torch.cat([model(_batch([g])) for g in _graphs()])
        assert torch.allclose(batched, single, atol=1e-6)

    def test_node_output_is_permutation_equivariant(self):
        model = _model(task_level="node").eval()
        graph = _triangle_graph()
        perm = torch.tensor([2, 0, 1])
        permuted = Data(x=graph.x[perm], edge_index=graph.edge_index)
        out = model(_batch([graph]))
        out_perm = model(_batch([permuted]))
        assert torch.allclose(out[perm], out_perm, atol=1e-6)

    def test_node_output_sees_the_whole_set(self):
        """The pooled set feature enters every point's prediction, so
        moving one point changes the output of the others."""
        model = _model(task_level="node").eval()
        graph = _triangle_graph()
        moved = graph.clone()
        moved.x[2] += 1.0
        out = model(_batch([graph]))
        out_moved = model(_batch([moved]))
        assert not torch.allclose(out[0], out_moved[0])

    def test_node_head_width(self):
        model = _model(task_level="node")
        assert model.rho[0].linear.in_features == 2 * 8
