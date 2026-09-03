"""Value-level tests for src/metrics/regression.py.

Covers RoundedAccuracy with hand-computed expectations, multi-target
handling, shape-mismatch reshaping, state accumulation across updates,
and reset behavior.

Style mirrors tests/test_values.py (plain pytest, exact asserts).
"""

import pytest
import torch

from src.metrics.regression import RoundedAccuracy


class TestRoundedAccuracy:
    def test_all_correct(self):
        """Predictions that round to the integer targets score 1.0."""
        metric = RoundedAccuracy()
        preds = torch.tensor([0.4, 1.6, 2.2, -0.7])
        target = torch.tensor([0.0, 2.0, 2.0, -1.0])
        metric(preds, target)
        assert metric.compute().item() == 1.0

    def test_half_correct(self):
        """round(1.4)=1 hits, round(1.6)=2 misses -> 0.5."""
        metric = RoundedAccuracy()
        metric(torch.tensor([1.4, 1.6]), torch.tensor([1.0, 1.0]))
        assert metric.compute().item() == 0.5

    def test_multi_target_counts_components_individually(self):
        """Every component of a multi-target vector counts on its own."""
        metric = RoundedAccuracy()
        preds = torch.tensor([[0.9, 2.1], [3.6, 4.9]])
        target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
        metric(preds, target)
        # 3.6 rounds to 4 != 3; the other three components hit.
        assert metric.compute().item() == pytest.approx(0.75)

    def test_column_vector_predictions_reshaped_to_target(self):
        """(N, 1) predictions against (N,) targets are reshaped, not broadcast."""
        metric = RoundedAccuracy()
        preds = torch.tensor([[1.1], [2.2], [2.8]])
        target = torch.tensor([1.0, 2.0, 3.0])
        metric(preds, target)
        assert metric.compute().item() == 1.0
        # Broadcasting instead of reshaping would have produced a 3x3
        # comparison and a denominator != 3.
        assert metric.total.item() == 3.0
        assert metric.correct.item() == 3.0

    def test_accumulates_across_updates(self):
        """State accumulates: 2/2 correct then 0/2 correct -> 0.5."""
        metric = RoundedAccuracy()
        metric(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0]))
        metric(torch.tensor([5.0, 5.0]), torch.tensor([1.0, 2.0]))
        assert metric.compute().item() == 0.5
        assert metric.total.item() == 4.0

    def test_reset_clears_state(self):
        metric = RoundedAccuracy()
        metric(torch.tensor([9.0]), torch.tensor([1.0]))
        assert metric.compute().item() == 0.0
        metric.reset()
        metric(torch.tensor([1.0]), torch.tensor([1.0]))
        assert metric.compute().item() == 1.0

    def test_negative_targets(self):
        """Rounding of negative predictions: -97.6 -> -98 (hit), -96.4 -> -96 (miss)."""
        metric = RoundedAccuracy()
        preds = torch.tensor([-97.6, -96.4])
        target = torch.tensor([-98.0, -98.0])
        metric(preds, target)
        assert metric.compute().item() == 0.5
