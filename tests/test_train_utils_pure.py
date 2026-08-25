"""Tests for the pure regression helpers in src/utils/train_utils.py:
the target scaler, the regression branches of init_metrics and
test/validate, and the scaling round trip.
"""

import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Data

from src.utils.train_utils import (
    compute_target_scaler,
    init_metrics,
    scale_targets,
    test as _test_entrypoint,
    unscale_predictions,
    validate,
)


class TestTargetScaler:
    def test_mean_and_std_exact_values(self):
        """std is the unbiased (ddof=1) torch default:
        values [1,2,3,4] -> mean 2.5, std sqrt(5/3)."""
        loader = SimpleNamespace(dataset=[
            Data(y=torch.tensor([[1.0], [2.0]])),
            Data(y=torch.tensor([[3.0], [4.0]])),
        ])
        mean, std = compute_target_scaler(loader)
        assert mean == pytest.approx(2.5)
        assert std == pytest.approx((5.0 / 3.0) ** 0.5)

    def test_constant_targets_fall_back_to_unit_std(self):
        loader = SimpleNamespace(dataset=[
            Data(y=torch.tensor([[5.0], [5.0]])),
            Data(y=torch.tensor([[5.0]])),
        ])
        mean, std = compute_target_scaler(loader)
        assert mean == pytest.approx(5.0)
        assert std == 1.0

    def test_scale_and_unscale_roundtrip(self):
        scaler = (2.5, 1.5)
        y = torch.tensor([1.0, 2.5, 4.0])
        scaled = scale_targets(y, scaler)
        assert torch.allclose(scaled, torch.tensor([-1.0, 0.0, 1.0]))
        assert torch.allclose(unscale_predictions(scaled, scaler), y)

    def test_none_scaler_is_identity(self):
        y = torch.tensor([1.0, 2.0])
        assert scale_targets(y, None) is y
        assert unscale_predictions(y, None) is y


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


class _EchoScaledModel(torch.nn.Module):
    """Predicts the z-scored ground truth exactly: after unscaling,
    every metric must report a perfect score."""

    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, batch):
        return scale_targets(batch.y, self.scaler)


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
    def test_validate_scales_loss_and_unscales_metrics(self):
        import torchmetrics

        scaler = (5.0, 2.0)
        model = _EchoScaledModel(scaler)
        metrics = validate(
            model, _batches(), torch.nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            _regression_cfg(), target_scaler=scaler,
        )
        assert metrics["val/loss"] == pytest.approx(0.0)
        assert metrics["val/mae"] == pytest.approx(0.0)

    def test_test_reports_metrics_in_original_units(self):
        import torchmetrics

        scaler = (5.0, 2.0)
        model = _EchoScaledModel(scaler)
        run = _FakeRun()
        _test_entrypoint(
            model, _batches(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            label_to_index={}, run=run, cfg=_regression_cfg(),
            epoch=3, target_scaler=scaler,
        )
        assert len(run.logged) == 1
        payload = run.logged[0]
        assert payload["epoch"] == 3
        assert payload["test/mae"] == pytest.approx(0.0)

    def test_off_by_one_predictions_keep_their_distance(self):
        """Without unscaling in place a (mean, std) scaler would shrink
        the reported error; the MAE must stay exactly 1."""
        import torchmetrics

        class OffByOne(torch.nn.Module):
            def forward(self, batch):
                return scale_targets(batch.y, (5.0, 2.0)) + 0.5

        run = _FakeRun()
        _test_entrypoint(
            OffByOne(), _batches(),
            [("mae", torchmetrics.MeanAbsoluteError())],
            label_to_index={}, run=run, cfg=_regression_cfg(),
            epoch=0, target_scaler=(5.0, 2.0),
        )
        assert run.logged[0]["test/mae"] == pytest.approx(1.0)
