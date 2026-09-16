"""Tests for the pure helpers in src/utils/train_utils.py: init_metrics
and the regression branches of validate and test.
"""

import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Data

from src.utils.train_utils import (
    init_metrics,
    test as _test_entrypoint,
    validate,
)


class TestInitMetrics:
    def test_regression_metrics_instantiate_as_configured(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "task": {
                "task_type": "regression",
                "metrics": {
                    "mae": {"_target_": "torchmetrics.MeanAbsoluteError"},
                },
            },
            "trainer": {"device": "cpu"},
            "model": {"out_channels": 1},
        })
        accs = init_metrics(cfg)
        assert [name for name, _ in accs] == ["mae"]
        import torchmetrics
        assert isinstance(accs[0][1], torchmetrics.MeanAbsoluteError)

    def test_regression_rounded_accuracy_instantiates(self):
        """RoundedAccuracy is a regression metric; it must not receive
        the classification wiring of torchmetrics.Accuracy."""
        from omegaconf import OmegaConf
        from src.metrics.regression import RoundedAccuracy

        cfg = OmegaConf.create({
            "task": {
                "task_type": "regression",
                "metrics": {
                    "rounded_accuracy": {
                        "_target_": "src.metrics.regression.RoundedAccuracy"
                    },
                },
            },
            "trainer": {"device": "cpu"},
            "model": {"out_channels": 1},
        })
        (name, metric), = init_metrics(cfg)
        assert name == "rounded_accuracy"
        assert isinstance(metric, RoundedAccuracy)

    def test_classification_accuracy_gets_multiclass_wiring(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "task": {
                "task_type": "classification",
                "metrics": {
                    "accuracy": {"_target_": "torchmetrics.Accuracy"},
                },
            },
            "trainer": {"device": "cpu"},
            "model": {"out_channels": 3},
        })
        (name, metric), = init_metrics(cfg)
        assert name == "accuracy"
        assert metric.num_classes == 3


class _FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


class _EchoModel(torch.nn.Module):
    """Predicts the ground truth exactly: every metric must report a
    perfect score."""

    def forward(self, batch):
        return batch.y


class _OffByOne(torch.nn.Module):
    def forward(self, batch):
        return batch.y + 1.0


def _regression_cfg():
    return SimpleNamespace(
        task=SimpleNamespace(task_type="regression"),
        trainer=SimpleNamespace(device="cpu"),
        logger=SimpleNamespace(),
    )


def _batches():
    return [
        Data(y=torch.tensor([[2.0], [4.0]])),
        Data(y=torch.tensor([[6.0], [8.0]])),
    ]


class TestRegressionEvaluation:
    def test_validate_reports_loss_and_metrics(self):
        import torchmetrics

        metrics = validate(
            _EchoModel(), _batches(), torch.nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            _regression_cfg(),
        )
        assert metrics["val/loss"] == pytest.approx(0.0)
        assert metrics["val/mae"] == pytest.approx(0.0)

    def test_test_logs_epoch_and_metrics(self):
        import torchmetrics

        run = _FakeRun()
        _test_entrypoint(
            _EchoModel(), _batches(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            label_to_index={}, run=run, cfg=_regression_cfg(),
            epoch=3,
        )
        assert len(run.logged) == 1
        payload = run.logged[0]
        assert payload["epoch"] == 3
        assert payload["test/mae"] == pytest.approx(0.0)

    def test_off_by_one_predictions_score_mae_one(self):
        import torchmetrics

        run = _FakeRun()
        _test_entrypoint(
            _OffByOne(), _batches(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            label_to_index={}, run=run, cfg=_regression_cfg(),
            epoch=0,
        )
        assert run.logged[0]["test/mae"] == pytest.approx(1.0)
