"""Tests for the training objectives in src/metrics/loss.py."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.metrics.loss import (
    CrossEntropyObjective,
    PachnerConsistency,
    same_walk_mask,
)


def make_batch(walk_base, walk_step, y, with_walks=True):
    """Batch of complexes described only by their walk metadata."""
    data_list = []
    for base, step, label in zip(walk_base, walk_step, y):
        data = Data(x=torch.zeros(1, 1), y=torch.tensor(label))
        if with_walks:
            data.walk_base = int(base)
            data.walk_step = int(step)
        data_list.append(data)
    return Batch.from_data_list(data_list)


def base_loss():
    return torch.nn.CrossEntropyLoss()


class TestSameWalkMask:
    def test_mask_is_upper_triangular_and_counts_each_pair_once(self):
        batch = make_batch([0, 0, 0, 1, 2], [0, 1, 2, 0, 0], [0, 0, 0, 0, 1])
        mask = same_walk_mask(batch, "test")

        # Three snapshots of walk 0 form three unordered pairs.
        assert mask.sum() == 3
        assert mask.diagonal().sum() == 0
        assert not torch.tril(mask).any()
        assert not mask[3].any() and not mask[:, 3].any()

    def test_missing_walk_attributes_raise(self):
        batch = make_batch([0, 1], [0, 0], [0, 1], with_walks=False)
        with pytest.raises(ValueError, match="walk_length"):
            same_walk_mask(batch, "test")


class TestCrossEntropyObjective:
    def test_matches_the_plain_task_loss(self):
        batch = make_batch([0, 1], [0, 0], [0, 1], with_walks=False)
        logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])

        loss, parts = CrossEntropyObjective(base_loss())(
            logits, torch.zeros(2, 3), batch
        )

        assert torch.allclose(loss, base_loss()(logits, batch.y))
        assert set(parts) == {"ce"}


class TestPachnerConsistency:
    def test_identical_partners_add_nothing(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        logits = torch.tensor([[2.0, 0.0], [2.0, 0.0]])
        embedding = torch.ones(2, 4)

        loss, parts = PachnerConsistency(base_loss(), weight=1.0)(
            logits, embedding, batch
        )

        assert parts["consistency"] == pytest.approx(0.0)
        assert loss == pytest.approx(parts["ce"])

    def test_distance_between_partners_is_penalised(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        logits = torch.zeros(2, 2)
        embedding = torch.tensor([[0.0, 0.0], [3.0, 4.0]])

        _, parts = PachnerConsistency(base_loss(), weight=2.0)(
            logits, embedding, batch
        )

        assert parts["consistency"] == pytest.approx(25.0)

    def test_relative_target_is_scale_free(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        embedding = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
        objective = PachnerConsistency(base_loss(), target="relative")

        _, small = objective(torch.zeros(2, 2), embedding, batch)
        _, large = objective(torch.zeros(2, 2), 1000 * embedding, batch)

        # ||a - b||^2 / (||a||^2 + ||b||^2) = 25 / 25.
        assert small["consistency"] == pytest.approx(1.0)
        assert large["consistency"] == pytest.approx(1.0)

    def test_relative_target_is_not_minimised_by_shrinking(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        embedding = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        objective = PachnerConsistency(base_loss(), target="relative")

        loss, _ = objective(torch.zeros(2, 2), embedding, batch)
        loss.backward()

        # Scaling the embedding leaves the loss unchanged, so the gradient
        # has no radial component.
        assert (embedding.grad * embedding).sum().item() == pytest.approx(0.0, abs=1e-6)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="target"):
            PachnerConsistency(base_loss(), target="something")

    def test_gradient_reaches_the_embedding(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        embedding = torch.randn(2, 4, requires_grad=True)

        loss, _ = PachnerConsistency(base_loss())(
            torch.zeros(2, 2), embedding, batch
        )
        loss.backward()

        assert embedding.grad is not None
        assert embedding.grad.abs().sum() > 0
