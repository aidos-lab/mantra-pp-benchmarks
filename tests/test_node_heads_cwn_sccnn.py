"""Tests for the node-level heads of CWN and SCCNN.

Existing tests (tests/test_values.py) already check the node-level
output *shapes*; these drafts add value-level consistency (CWN graph
output == scatter_add of node output), gradient flow through the node
heads, multi-channel node outputs, and head-module structure.

Batch construction mirrors the helpers in tests/test_values.py, made
deterministic with explicit seeds so a batch can be rebuilt identically.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import scatter


def _make_cwn_batch(seed=0):
    """Two simplicial complexes, mirroring tests/test_values.py."""
    torch.manual_seed(seed)
    batch = Data()
    batch.x_0 = torch.randn(7, 3)
    batch.x_1 = torch.randn(8, 3)
    batch.x_2 = torch.randn(3, 3)
    batch.batch_0 = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch.batch_1 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    batch.batch_2 = torch.tensor([0, 0, 1])
    batch.adjacency_1_idx = torch.tensor([[0, 1], [1, 0]])
    batch.adjacency_1_val = torch.tensor([1.0, 1.0])
    batch.incidence_1_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_1_val = torch.tensor([1.0, -1.0])
    batch.incidence_2_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_2_val = torch.tensor([1.0, -1.0])
    return batch


def _make_sccnn_batch(seed=0):
    """Two simplicial complexes, mirroring tests/test_values.py."""
    torch.manual_seed(seed)
    batch = Data()
    batch.x_0 = torch.randn(7, 3)
    batch.x_1 = torch.randn(7, 3)
    batch.x_2 = torch.randn(3, 3)
    batch.batch_0 = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch.batch_1 = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    batch.batch_2 = torch.tensor([0, 0, 1])
    batch.hodge_laplacian_0_idx = torch.tensor([[0, 1], [1, 0]])
    batch.hodge_laplacian_0_val = torch.tensor([1.0, 1.0])
    batch.down_laplacian_1_idx = torch.tensor([[0, 1], [1, 0]])
    batch.down_laplacian_1_val = torch.tensor([1.0, 1.0])
    batch.up_laplacian_1_idx = torch.tensor([[0, 1], [1, 0]])
    batch.up_laplacian_1_val = torch.tensor([1.0, 1.0])
    batch.down_laplacian_2_idx = torch.tensor([[0, 1], [1, 0]])
    batch.down_laplacian_2_val = torch.tensor([1.0, 1.0])
    batch.incidence_1_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_1_val = torch.tensor([1.0, -1.0])
    batch.incidence_2_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_2_val = torch.tensor([1.0, -1.0])
    return batch


# ===========================================================================
# 1. CWN node-level head
# ===========================================================================

class TestCWNNodeHead:
    def _make_model(self, task_level, out_channels=1):
        from src.models.cwn import CWN

        torch.manual_seed(7)
        return CWN(
            in_channels=[3, 3, 3],
            hidden_channels=[4, 4, 4],
            n_layers=1,
            out_channels=out_channels,
            task_level=task_level,
        )

    def test_graph_output_is_scatter_add_of_node_output(self):
        """The graph path is exactly scatter_add over the node path, so
        with identical weights the two must agree to numerical identity."""
        model = self._make_model(task_level='node', out_channels=2)
        model.eval()

        node_out = model(_make_cwn_batch(seed=3))
        assert node_out.shape == (7, 2)

        model.task_level = 'graph'
        graph_out = model(_make_cwn_batch(seed=3))
        assert graph_out.shape == (2, 2)

        pooled = scatter(
            node_out, torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            dim=-2, reduce='sum',
        )
        assert torch.allclose(graph_out, pooled, atol=1e-6)

    def test_node_head_gradient_flow(self):
        model = self._make_model(task_level='node', out_channels=1)
        out = model(_make_cwn_batch(seed=1))
        loss = nn.MSELoss()(out, torch.zeros_like(out))
        loss.backward()
        # The final head and the rank-0 projection must receive gradients.
        assert model.graph_readout.lins[0].weight.grad is not None
        assert model.proj_0.weight.grad is not None
        assert model.proj_0.weight.grad.abs().sum() > 0

    def test_node_output_depends_on_node_features(self):
        model = self._make_model(task_level='node', out_channels=1)
        model.eval()

        batch_a = _make_cwn_batch(seed=3)
        out_a = model(batch_a)
        batch_b = _make_cwn_batch(seed=3)
        batch_b.x_0 = batch_b.x_0 + 1.0
        out_b = model(batch_b)
        assert not torch.allclose(out_a, out_b)

    def test_invalid_task_level_rejected(self):
        with pytest.raises(AssertionError):
            self._make_model(task_level='edge')


# ===========================================================================
# 2. SCCNN node-level head
# ===========================================================================

class TestSCCNNNodeHead:
    def _make_model(self, task_level, out_channels=1):
        from src.models.sccnn import SCCNN

        torch.manual_seed(7)
        return SCCNN(
            in_channels=[3, 3, 3],
            hidden_channels=[4, 4, 4],
            out_channels=out_channels,
            conv_order=1,
            sc_order=2,
            n_layers=2,
            task_level=task_level,
        )

    def test_multi_channel_node_output_shape(self):
        model = self._make_model(task_level='node', out_channels=3)
        model.eval()
        out = model(_make_sccnn_batch(seed=3))
        assert out.shape == (7, 3)
        assert torch.isfinite(out).all()

    def test_node_head_is_linear_on_rank0(self):
        """Node mode swaps the three SumReadouts for a single Linear head."""
        model = self._make_model(task_level='node', out_channels=3)
        assert isinstance(model.node_readout, nn.Linear)
        assert model.node_readout.in_features == 4
        assert model.node_readout.out_features == 3
        assert not hasattr(model, 'readout_0')

        graph_model = self._make_model(task_level='graph', out_channels=3)
        assert not hasattr(graph_model, 'node_readout')
        assert hasattr(graph_model, 'readout_0')

    def test_node_head_gradient_flow(self):
        model = self._make_model(task_level='node', out_channels=1)
        out = model(_make_sccnn_batch(seed=1))
        assert out.shape == (7, 1)
        loss = nn.MSELoss()(out, torch.zeros_like(out))
        loss.backward()
        assert model.node_readout.weight.grad is not None
        assert model.node_readout.weight.grad.abs().sum() > 0

    def test_node_output_depends_on_node_features(self):
        model = self._make_model(task_level='node', out_channels=1)
        model.eval()

        out_a = model(_make_sccnn_batch(seed=3))
        batch_b = _make_sccnn_batch(seed=3)
        batch_b.x_0 = batch_b.x_0 + 1.0
        out_b = model(batch_b)
        assert not torch.allclose(out_a, out_b)

    def test_invalid_task_level_rejected(self):
        with pytest.raises(AssertionError):
            self._make_model(task_level='edge')
