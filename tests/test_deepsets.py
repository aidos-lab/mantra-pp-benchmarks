"""Tests for src/models/DeepSets.py: permutation invariance, connectivity
blindness and the node-level head."""

import torch
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
                  n_layers=2, dropout=0.0, pool_method="sum")
    kwargs.update(overrides)
    torch.manual_seed(0)
    return DeepSets(**kwargs).eval()


def _permuted(graph, perm):
    return Data(x=graph.x[perm], edge_index=graph.edge_index)


class TestDeepSets:
    def test_permutation_invariance(self):
        model = _model(pool_method="max")
        graph = _triangle_graph()
        perm = torch.tensor([2, 0, 1])
        out = model(_batch([graph]))
        out_perm = model(_batch([_permuted(graph, perm)]))
        assert torch.allclose(out, out_perm, atol=1e-6)

    def test_connectivity_blindness(self):
        model = _model()
        rewired = [
            Data(x=g.x, edge_index=torch.empty((2, 0), dtype=torch.long))
            for g in _graphs()
        ]
        assert torch.equal(model(_batch()), model(_batch(rewired)))


class TestDeepSetsNodeHead:
    def test_node_outputs_equal_per_graph_outputs(self):
        model = _model(task_level="node")
        batched = model(_batch())
        single = torch.cat([model(_batch([g])) for g in _graphs()])
        assert batched.shape == (5, 3)
        assert torch.allclose(batched, single, atol=1e-6)

    def test_node_output_is_permutation_equivariant(self):
        model = _model(task_level="node")
        graph = _triangle_graph()
        perm = torch.tensor([2, 0, 1])
        out = model(_batch([graph]))
        out_perm = model(_batch([_permuted(graph, perm)]))
        assert torch.allclose(out[perm], out_perm, atol=1e-6)

    def test_node_output_sees_the_whole_set(self):
        """The pooled set feature enters every point's prediction, so
        moving one point changes the output of the others."""
        model = _model(task_level="node")
        graph = _triangle_graph()
        moved = graph.clone()
        moved.x[2] += 1.0
        out = model(_batch([graph]))
        out_moved = model(_batch([moved]))
        assert not torch.allclose(out[0], out_moved[0])
