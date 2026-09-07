"""Value-level tests for the pool_method knob of MLP and Graphormer
(GCN and ResidualGatedGCN already route through src.models.utils.pool)."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.Graphormer import Graphormer
from src.models.MLP import MLP
from src.utils.graphormer_spd import GraphormerSPDPreTransform


def _mlp(pool_method="mean", out_channels=3):
    torch.manual_seed(0)
    return MLP(2, 8, out_channels, n_layers=2, dropout=0.0,
               batch_norm=True, residuals=False, pool_method=pool_method)


def _graphs():
    return [
        Data(x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
             edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]],
                                     dtype=torch.long)),
        Data(x=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
             edge_index=torch.empty((2, 0), dtype=torch.long)),
    ]


class TestMLPPooling:
    def test_sum_readout_is_n_times_mean(self):
        """The scatter runs before the readout, so the pooled features
        (not the final output) scale by n; zero the shared readout
        bias to isolate that relationship from the one-time offset it
        adds after pooling."""
        model_mean = _mlp("mean")
        model_sum = _mlp("sum")
        model_sum.load_state_dict(model_mean.state_dict())
        model_mean.readout.bias.data.zero_()
        model_sum.readout.bias.data.zero_()
        model_mean.eval()
        model_sum.eval()

        batch = Batch.from_data_list(_graphs())
        out_mean = model_mean(batch)
        out_sum = model_sum(batch)
        n = torch.tensor([[3.0], [2.0]])
        assert torch.allclose(out_sum, n * out_mean, atol=1e-5)

    def test_invalid_pool_method_rejected(self):
        model = _mlp("median")
        batch = Batch.from_data_list(_graphs())
        with pytest.raises(ValueError):
            model(batch)

    def test_readout_is_linear(self):
        """The readout is a plain Linear layer, so regression outputs
        are not clamped to >= 0 (c2_anticanonical can be negative)."""
        model = _mlp(out_channels=1)
        assert isinstance(model.readout, torch.nn.Linear)
        model.eval()
        with torch.no_grad():
            model.readout.weight.fill_(-1.0)
            model.readout.bias.fill_(-1.0)
        graphs = [
            Data(x=torch.randn(4, 2),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
            for _ in range(8)
        ]
        assert (model(Batch.from_data_list(graphs)) < 0).all()


class TestGraphormerPooling:
    def _build(self, pool_method):
        torch.manual_seed(0)
        return Graphormer(
            hidden_channels=8, n_layers=1, in_channels=2,
            out_channels=2, num_heads=2, dropout=0.0,
            max_degree=4, max_distance=4, pool_method=pool_method,
        )

    def test_sum_readout_relation_and_validation(self):
        """out_sum - b = n_valid * (out_mean - b) per graph; padding is
        excluded from both readouts."""
        model_mean = self._build('mean')
        model_sum = self._build('sum')
        model_sum.load_state_dict(model_mean.state_dict())
        model_mean.eval()
        model_sum.eval()

        pre = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        batch = Batch.from_data_list([pre(g) for g in _graphs()])

        out_mean = model_mean(batch)
        out_sum = model_sum(batch)
        bias = model_mean.final_linear.bias
        n = torch.tensor([[3.0], [2.0]])
        assert torch.allclose(out_sum - bias, n * (out_mean - bias),
                              atol=1e-5)

        with pytest.raises(AssertionError):
            self._build('max')
