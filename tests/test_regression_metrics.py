"""Value-level tests for src/metrics/regression.py.

Covers RoundedAccuracy and LatticeRoundedAccuracy with hand-computed
expectations, multi-target handling, shape-mismatch reshaping, state
accumulation across updates, and reset behavior.

Style mirrors tests/test_values.py (plain pytest, exact asserts).
"""

import pytest
import torch

from src.metrics.regression import LatticeRoundedAccuracy, RoundedAccuracy


# ===========================================================================
# 1. RoundedAccuracy — exact values
# ===========================================================================

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


# ===========================================================================
# 2. LatticeRoundedAccuracy — exact values
# ===========================================================================

class TestLatticeRoundedAccuracy:
    def test_even_lattice_recovers_near_miss(self):
        """A prediction of 2.9 with even targets: nearest even is 2 (hit),
        while plain integer rounding gives 3 (miss). This is the exact
        motivating case from the docstring."""
        lattice = LatticeRoundedAccuracy(spacing=2.0)
        plain = RoundedAccuracy()
        preds = torch.tensor([2.9])
        target = torch.tensor([2.0])
        lattice(preds, target)
        plain(preds, target)
        assert lattice.compute().item() == 1.0
        assert plain.compute().item() == 0.0

    def test_even_lattice_still_misses_far_predictions(self):
        """3.2 / 2 = 1.6 rounds to 2 -> nearest even is 4, not the target 2."""
        metric = LatticeRoundedAccuracy(spacing=2.0)
        metric(torch.tensor([3.2]), torch.tensor([2.0]))
        assert metric.compute().item() == 0.0

    def test_spacing_one_matches_rounded_accuracy(self):
        """With spacing=1 the metric is exactly RoundedAccuracy."""
        preds = torch.tensor([0.4, 1.6, 2.5, -0.7, 3.49])
        target = torch.tensor([0.0, 2.0, 2.0, -1.0, 4.0])
        lattice = LatticeRoundedAccuracy(spacing=1.0)
        plain = RoundedAccuracy()
        lattice(preds, target)
        plain(preds, target)
        assert lattice.compute().item() == plain.compute().item()

    def test_negative_even_targets(self):
        """c2_anticanonical-like scale: -97.1 / 2 = -48.55 -> -49 -> hit for -98."""
        metric = LatticeRoundedAccuracy(spacing=2.0)
        preds = torch.tensor([-97.1, -95.0])
        target = torch.tensor([-98.0, -98.0])
        metric(preds, target)
        # -95.0 / 2 = -47.5 rounds (half-to-even) to -48 -> -96 != -98: miss.
        assert metric.compute().item() == 0.5

    def test_column_vector_predictions_reshaped(self):
        metric = LatticeRoundedAccuracy(spacing=2.0)
        preds = torch.tensor([[1.9], [6.3]])
        target = torch.tensor([2.0, 6.0])
        metric(preds, target)
        assert metric.compute().item() == 1.0
        assert metric.total.item() == 2.0

    def test_accumulates_and_resets(self):
        metric = LatticeRoundedAccuracy(spacing=2.0)
        metric(torch.tensor([2.1]), torch.tensor([2.0]))
        metric(torch.tensor([7.0]), torch.tensor([2.0]))
        assert metric.compute().item() == 0.5
        metric.reset()
        assert metric.total.item() == 0.0
        assert metric.correct.item() == 0.0
