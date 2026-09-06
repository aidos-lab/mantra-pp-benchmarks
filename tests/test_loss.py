"""Tests for src/metrics/loss.py."""

import torch

from src.metrics.loss import BaseLoss, Contrastive, LossType


class TestLossType:
    def test_enum_members_exist(self):
        assert LossType.BINARY is not None
        assert LossType.MULTICLASS is not None
        assert LossType.REGRESSION is not None


class TestBaseLoss:
    def test_init_calls_super(self):
        loss = BaseLoss(task_type=LossType.BINARY)
        assert isinstance(loss, torch.nn.Module)

    def test_turn_to_probs_binary(self):
        loss = BaseLoss(task_type=LossType.BINARY)
        x = torch.tensor([0.0, 1.0, -1.0])
        result = loss.turn_to_probs(x)
        assert torch.allclose(result, torch.sigmoid(x))

    def test_turn_to_probs_multiclass(self):
        loss = BaseLoss(task_type=LossType.MULTICLASS)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = loss.turn_to_probs(x)
        expected = torch.nn.functional.log_softmax(x)
        assert torch.allclose(result, expected)

    def test_turn_to_probs_regression_passthrough(self):
        loss = BaseLoss(task_type=LossType.REGRESSION)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = loss.turn_to_probs(x)
        assert torch.equal(result, x)


class TestContrastive:
    def test_identical_vectors_returns_max_similarity(self):
        loss = Contrastive(task_type=LossType.BINARY)
        v = torch.tensor([1.0, 0.0, 0.0])
        result = loss(v, v, gamma=0)
        # cosine similarity of identical vectors = 1.0
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors_returns_zero(self):
        loss = Contrastive(task_type=LossType.BINARY)
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        result = loss(v1, v2, gamma=0)
        assert abs(result) < 1e-6

    def test_gamma_margin(self):
        loss = Contrastive(task_type=LossType.BINARY)
        v = torch.tensor([1.0, 0.0])
        # cosine sim = 1.0, gamma = 0.5 -> max(0, 1.0 - 0.5) = 0.5
        result = loss(v, v, gamma=0.5)
        assert abs(result - 0.5) < 1e-6

    def test_negative_result_clamped_to_zero(self):
        loss = Contrastive(task_type=LossType.BINARY)
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        # cosine sim ≈ 0, gamma = 0.5 -> max(0, 0.0 - 0.5) = 0
        result = loss(v1, v2, gamma=0.5)
        assert result == 0
