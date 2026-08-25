"""Tests for the node-level supervision path: select_supervised, the
node_mask handling of compute_target_scaler, and masked batches in
test_regression."""

import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Data

from src.utils.train_utils import (
    compute_target_scaler,
    scale_targets,
    select_supervised,
    test_regression as _test_regression,
)


class TestSelectSupervised:
    def test_masked_node_level_batch(self):
        y = torch.tensor([[10.0], [1.0], [2.0], [3.0]])
        mask = torch.tensor([False, True, True, False])
        batch = Data(y=y, node_mask=mask)
        y_hat = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        out_hat, out_y = select_supervised(y_hat, batch)
        assert torch.equal(out_hat, torch.tensor([[0.2], [0.3]]))
        assert torch.equal(out_y, torch.tensor([[1.0], [2.0]]))

    def test_graph_level_batch_passes_through(self):
        y = torch.tensor([[1.0], [2.0]])
        batch = Data(y=y)
        y_hat = torch.tensor([[0.5], [0.6]])
        out_hat, out_y = select_supervised(y_hat, batch)
        assert out_hat is y_hat
        assert out_y is y

    def test_shape_mismatch_disables_masking(self):
        """Graph-level predictions with a stale node_mask: the row-count
        check keeps the mask from being applied."""
        y = torch.tensor([[1.0], [2.0]])
        mask = torch.tensor([False, True, True, False, True])
        batch = Data(y=y, node_mask=mask)
        y_hat = torch.tensor([[0.5], [0.6]])
        out_hat, out_y = select_supervised(y_hat, batch)
        assert out_hat is y_hat
        assert out_y is y


class TestTargetScalerWithMasks:
    def test_masked_values_are_excluded(self):
        """Masked node values are excluded; std is the unbiased (ddof=1)
        torch default: values [1,2,3,4] -> mean 2.5, std sqrt(5/3)."""
        loader = SimpleNamespace(dataset=[
            Data(
                y=torch.tensor([[10.0], [1.0], [2.0]]),
                node_mask=torch.tensor([False, True, True]),
            ),
            Data(y=torch.tensor([[3.0], [4.0]])),
        ])
        mean, std = compute_target_scaler(loader)
        assert mean == pytest.approx(2.5)
        assert std == pytest.approx((5.0 / 3.0) ** 0.5)

    def test_mask_with_mismatched_shape_is_ignored(self):
        loader = SimpleNamespace(dataset=[
            Data(y=torch.tensor([7.0, 9.0]),
                 node_mask=torch.tensor([True, False, True])),
        ])
        mean, _ = compute_target_scaler(loader)
        assert mean == pytest.approx(8.0)


class _FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


class _EchoScaledModel(torch.nn.Module):
    """Predicts the z-scored ground truth exactly: after unscaling and
    masking, every metric must report a perfect score."""

    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, batch):
        return scale_targets(batch.y, self.scaler)


class TestTestRegressionMasked:
    def test_masked_rows_are_excluded_from_metrics(self):
        """The unmasked origin row (99.0) is echoed wrong by design; it
        must not count because the mask drops it."""
        from src.metrics.regression import RoundedAccuracy

        scaler = (5.0, 2.0)
        model = _EchoScaledModel(scaler)
        run = _FakeRun()
        cfg = SimpleNamespace(
            task=SimpleNamespace(task_type="regression"),
            trainer=SimpleNamespace(device="cpu", max_epoch=5),
        )
        batches = [
            Data(y=torch.tensor([[2.0], [4.0]])),
            Data(
                y=torch.tensor([[99.0], [6.0], [8.0]]),
                node_mask=torch.tensor([False, True, True]),
            ),
        ]
        accuracies = [("rounded_accuracy", RoundedAccuracy())]
        _test_regression(
            model, batches, accuracies, run, cfg, target_scaler=scaler,
        )
        assert run.logged[0]["test/rounded_accuracy"] == pytest.approx(1.0)
