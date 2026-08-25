"""Tests for the pure functions in src/utils/train_utils.py and
src/models/utils.py, the num_workers logic in
src/utils/data_handling.py, and the sum-readout options of
Graphormer/MLP.
"""

import numpy as np
import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Batch, Data

from src.models.utils import READOUTS, pool
from src.utils.data_handling import split_dataset
from src.utils.train_utils import (
    compute_target_scaler,
    init_metrics,
    scale_targets,
    test as _test_entrypoint,
    test_regression as _test_regression,
    unscale_predictions,
)


# ===========================================================================
# 1. pool (src/models/utils.py)
# ===========================================================================

class TestPool:
    def test_mean_reduce_exact_values(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pool(x, batch, reduce='mean')
        assert torch.equal(out, torch.tensor([[2.0, 3.0], [10.0, 20.0]]))

    def test_sum_reduce_exact_values(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]])
        batch = torch.tensor([0, 0, 1])
        out = pool(x, batch, reduce='sum')
        assert torch.equal(out, torch.tensor([[4.0, 6.0], [10.0, 20.0]]))

    def test_default_reduce_is_mean(self):
        x = torch.tensor([[2.0], [4.0]])
        batch = torch.tensor([0, 0])
        assert torch.equal(pool(x, batch), torch.tensor([[3.0]]))

    def test_unknown_reduce_rejected(self):
        with pytest.raises(AssertionError, match="unknown readout"):
            pool(torch.zeros(1, 1), torch.tensor([0]), reduce='max')

    def test_readouts_constant(self):
        assert READOUTS == ('mean', 'sum')


# ===========================================================================
# 2. compute_target_scaler / scale_targets / unscale_predictions
# ===========================================================================

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

    def test_multi_target_rows_are_flattened(self):
        loader = SimpleNamespace(dataset=[
            Data(y=torch.tensor([[1.0, 3.0]])),
            Data(y=torch.tensor([[5.0, 7.0]])),
        ])
        mean, std = compute_target_scaler(loader)
        assert mean == pytest.approx(4.0)
        # values [1,3,5,7]: var = (9+1+1+9)/3
        assert std == pytest.approx((20.0 / 3.0) ** 0.5)

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


# ===========================================================================
# 3. split_dataset num_workers logic (src/utils/data_handling.py)
# ===========================================================================

def _split_cfg(num_workers=None):
    trainer = SimpleNamespace(batch_size=2)
    if num_workers is not None:
        trainer.num_workers = num_workers
    return SimpleNamespace(
        trainer=trainer,
        seed=0,
        dataset=SimpleNamespace(
            use_stratified=False,
            train_parameters=SimpleNamespace(
                split=SimpleNamespace(train=0.6, val=0.2, test=0.2)
            ),
        ),
    )


def _tiny_dataset(n=10):
    return [
        Data(x=torch.randn(3, 2), y=torch.tensor([float(i)]))
        for i in range(n)
    ]


class TestSplitDatasetWorkers:
    def test_default_num_workers_zero_no_persistence(self):
        train, val, test = split_dataset(
            _tiny_dataset(), _split_cfg(), pin_memory=False
        )
        assert train.num_workers == 0
        assert val.num_workers == 0
        assert test.num_workers == 0
        assert train.persistent_workers is False

    def test_config_workers_split_between_train_and_eval(self):
        """8 train workers -> eval loaders get max(2, 8 // 4) = 2."""
        train, val, test = split_dataset(
            _tiny_dataset(), _split_cfg(num_workers=8), pin_memory=False
        )
        assert train.num_workers == 8
        assert val.num_workers == 2
        assert test.num_workers == 2
        assert train.persistent_workers is True
        assert val.persistent_workers is True

    def test_collate_fn_branch_uses_plain_dataloader(self):
        """With a collate_fn the torch DataLoader branch is taken and the
        same worker split applies."""
        def fake_collate(batch):
            return batch

        train, val, test = split_dataset(
            _tiny_dataset(), _split_cfg(num_workers=8), pin_memory=False,
            collate_fn=fake_collate,
        )
        assert train.collate_fn is fake_collate
        assert val.collate_fn is fake_collate
        assert train.num_workers == 8
        assert val.num_workers == 2
        assert test.num_workers == 2
        assert train.persistent_workers is True

    def test_split_sizes(self):
        train, val, test = split_dataset(
            _tiny_dataset(10), _split_cfg(), pin_memory=False
        )
        assert len(train.dataset) == 6
        assert len(val.dataset) == 2
        assert len(test.dataset) == 2


# ===========================================================================
# 4. New sum readouts of MLP and Graphormer
# ===========================================================================

class TestSumReadouts:
    def test_mlp_sum_readout_is_n_times_mean(self):
        """Scatter runs after the per-node readout MLP, so per graph
        out_sum = n * out_mean exactly (no bias after pooling)."""
        from src.models.MLP import MLP

        torch.manual_seed(0)
        model_mean = MLP(2, 8, 3, n_layers=2, dropout=0.0,
                         batch_norm=True, residuals=False, readout='mean')
        torch.manual_seed(0)
        model_sum = MLP(2, 8, 3, n_layers=2, dropout=0.0,
                        batch_norm=True, residuals=False, readout='sum')
        model_mean.eval()
        model_sum.eval()

        graphs = [
            Data(x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                 edge_index=torch.empty((2, 0), dtype=torch.long)),
            Data(x=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
                 edge_index=torch.empty((2, 0), dtype=torch.long)),
        ]
        batch = Batch.from_data_list(graphs)
        out_mean = model_mean(batch)
        out_sum = model_sum(batch)
        n = torch.tensor([[3.0], [2.0]])
        assert torch.allclose(out_sum, n * out_mean, atol=1e-5)

    def test_mlp_invalid_readout_rejected(self):
        from src.models.MLP import MLP

        with pytest.raises(AssertionError):
            MLP(2, 8, 3, n_layers=2, dropout=0.0, batch_norm=True,
                residuals=False, readout='max')

    def test_mlp_linear_head_can_predict_negative_values(self):
        """readout_activation=False replaces the ReLU-clamped head with a
        plain Linear: outputs must not be clamped to >= 0 (regression
        targets like c2_anticanonical ~ -98 need the lower half)."""
        from src.models.MLP import MLP

        torch.manual_seed(0)
        clamped = MLP(2, 8, 1, n_layers=2, dropout=0.0, batch_norm=True,
                      residuals=False, readout='mean',
                      readout_activation=True)
        linear = MLP(2, 8, 1, n_layers=2, dropout=0.0, batch_norm=True,
                     residuals=False, readout='mean',
                     readout_activation=False)
        assert isinstance(linear.readout, torch.nn.Linear)
        clamped.eval()
        linear.eval()

        graphs = [
            Data(x=torch.randn(4, 2),
                 edge_index=torch.empty((2, 0), dtype=torch.long))
            for _ in range(8)
        ]
        batch = Batch.from_data_list(graphs)
        assert (clamped(batch) >= 0).all()
        with torch.no_grad():
            linear.readout.weight.fill_(-1.0)
            linear.readout.bias.fill_(-1.0)
        assert (linear(batch) < 0).all()

    def test_graphormer_sum_readout_relation_and_validation(self):
        """out_sum - b = n_valid * (out_mean - b) per graph; padding is
        excluded from both readouts."""
        from src.models.Graphormer import Graphormer
        from src.utils.graphormer_spd import GraphormerSPDPreTransform

        def build(readout):
            torch.manual_seed(0)
            return Graphormer(
                hidden_channels=8, n_layers=1, in_channels=2,
                out_channels=2, num_heads=2, dropout=0.0,
                max_degree=4, max_distance=4, readout=readout,
            )

        model_mean = build('mean')
        model_sum = build('sum')
        model_sum.load_state_dict(model_mean.state_dict())
        model_mean.eval()
        model_sum.eval()

        pre = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        graphs = [
            Data(x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                 edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]],
                                         dtype=torch.long)),
            Data(x=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
                 edge_index=torch.empty((2, 0), dtype=torch.long)),
        ]
        batch = Batch.from_data_list([pre(g) for g in graphs])

        out_mean = model_mean(batch)
        out_sum = model_sum(batch)
        bias = model_mean.final_linear.bias
        n = torch.tensor([[3.0], [2.0]])
        assert torch.allclose(out_sum - bias, n * (out_mean - bias), atol=1e-5)

        with pytest.raises(ValueError, match="unknown readout"):
            build('max')


# ===========================================================================
# 5. init_metrics (regression branch) and test_regression
# ===========================================================================

class _FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)


class _EchoScaledModel(torch.nn.Module):
    """Predicts the z-scored ground truth exactly: after unscaling, every
    metric must report a perfect score."""

    def __init__(self, scaler):
        super().__init__()
        self.scaler = scaler

    def forward(self, batch):
        return scale_targets(batch.y, self.scaler)


class TestInitMetrics:
    def test_regression_branch_instantiates_all_metrics(self):
        from omegaconf import OmegaConf

        from src.metrics.regression import LatticeRoundedAccuracy

        cfg = OmegaConf.create({
            "task": {
                "task_type": "regression",
                "metrics": {
                    "lattice_rounded_accuracy": {
                        "_target_":
                            "src.metrics.regression.LatticeRoundedAccuracy",
                        "spacing": 2.0,
                    },
                    "mae": {
                        "_target_": "torchmetrics.MeanAbsoluteError",
                    },
                },
            },
            "trainer": {"device": "cpu"},
            "model": {"out_channels": 1},
        })
        accs = init_metrics(cfg)
        assert [name for name, _ in accs] == [
            "lattice_rounded_accuracy", "mae",
        ]
        assert isinstance(accs[0][1], LatticeRoundedAccuracy)
        assert accs[0][1].spacing == 2.0

    def test_classification_branch_keeps_accuracy_filter(self):
        from omegaconf import OmegaConf

        cfg = OmegaConf.create({
            "task": {
                "task_type": "classification",
                "metrics": {
                    "accuracy": {"_target_": "torchmetrics.Accuracy"},
                    "f1": {"_target_": "torchmetrics.F1Score"},
                },
            },
            "trainer": {"device": "cpu"},
            "model": {"out_channels": 3},
        })
        accs = init_metrics(cfg)
        # Only metrics whose target mentions "accuracy" are kept.
        assert [name for name, _ in accs] == ["accuracy"]


class TestTestRegression:
    def _make_cfg(self):
        return SimpleNamespace(
            task=SimpleNamespace(task_type="regression"),
            trainer=SimpleNamespace(device="cpu", max_epoch=5),
        )

    def _make_batches(self):
        return [
            Data(y=torch.tensor([[2.0], [4.0]])),
            Data(y=torch.tensor([[6.0], [8.0]])),
        ]

    def test_perfect_predictions_score_one(self):
        from src.metrics.regression import RoundedAccuracy

        scaler = (5.0, 2.0)
        model = _EchoScaledModel(scaler)
        run = _FakeRun()
        cfg = self._make_cfg()
        accuracies = [("rounded_accuracy", RoundedAccuracy())]

        _test_regression(
            model, self._make_batches(), accuracies, run, cfg,
            target_scaler=scaler,
        )

        assert len(run.logged) == 1
        payload = run.logged[0]
        assert payload["epoch"] == 5
        assert payload["test/rounded_accuracy"] == pytest.approx(1.0)

    def test_test_dispatches_regression_tasks(self):
        """The generic test() entrypoint routes regression tasks to
        test_regression instead of raising NotImplementedError."""
        from src.metrics.regression import RoundedAccuracy

        model = _EchoScaledModel(None)
        run = _FakeRun()
        cfg = self._make_cfg()
        accuracies = [("rounded_accuracy", RoundedAccuracy())]

        _test_entrypoint(
            model, self._make_batches(), accuracies,
            label_to_index={}, run=run, cfg=cfg, target_scaler=None,
        )
        assert run.logged[0]["test/rounded_accuracy"] == pytest.approx(1.0)

    def test_imperfect_predictions_scored_in_original_units(self):
        """A constant off-by-one model: unscaled predictions miss every
        target, so accuracy is exactly 0."""
        from src.metrics.regression import RoundedAccuracy

        class OffByOne(torch.nn.Module):
            def forward(self, batch):
                return batch.y + 1.0

        run = _FakeRun()
        cfg = self._make_cfg()
        accuracies = [("rounded_accuracy", RoundedAccuracy())]
        _test_regression(
            OffByOne(), self._make_batches(), accuracies, run, cfg,
            target_scaler=None,
        )
        assert run.logged[0]["test/rounded_accuracy"] == pytest.approx(0.0)
