"""Tests for the training objectives in src/metrics/loss.py."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.metrics.loss import (
    CrossEntropyObjective,
    PachnerConsistency,
    PachnerInfoNCE,
    PachnerPairwise,
    pair_masks,
    supervised_contrastive,
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


class TestPairMasks:
    def test_masks_are_upper_triangular_and_disjoint(self):
        batch = make_batch([0, 0, 1, 2], [0, 1, 0, 0], [0, 0, 0, 1])
        masks = pair_masks(batch, batch.y, "test")

        assert masks["same_walk"].sum() == 1
        assert masks["same_class_other_base"].sum() == 2
        assert masks["diff_class"].sum() == 3
        stacked = torch.stack(list(masks.values())).sum(0)
        assert stacked.max() <= 1
        assert stacked.diagonal().sum() == 0

    def test_missing_walk_attributes_raise(self):
        batch = make_batch([0, 1], [0, 0], [0, 1], with_walks=False)
        with pytest.raises(ValueError, match="walk_length"):
            pair_masks(batch, batch.y, "test")


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

    def test_logit_target_uses_symmetric_kl(self):
        batch = make_batch([0, 0], [0, 1], [0, 0])
        logits = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
        embedding = torch.zeros(2, 2)

        _, same = PachnerConsistency(base_loss(), target="logits")(
            logits, torch.ones(2, 2), batch
        )
        _, differing = PachnerConsistency(base_loss(), target="logits")(
            logits.flip(0) * 0, embedding, batch
        )

        assert same["consistency"] > 10.0
        assert differing["consistency"] == pytest.approx(0.0)

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


class TestPachnerPairwise:
    def test_perfectly_arranged_embedding_has_zero_loss(self):
        batch = make_batch([0, 0, 1, 2], [0, 2, 0, 0], [0, 0, 0, 1])
        # Same walk: 2 steps apart at 3 moves each -> distance 6.
        # Same class, other base: within the radius. Other class: beyond
        # the margin.
        embedding = torch.tensor(
            [
                [0.0, 0.0],
                [6.0, 0.0],
                [0.5, 0.0],
                [0.0, 50.0],
            ]
        )
        objective = PachnerPairwise(
            base_loss(),
            alpha=1.0,
            radius=6.0,
            margin=2.0,
            moves_per_step=3,
        )

        loss, parts = objective(torch.zeros(4, 2), embedding, batch)

        assert parts["walk"] == pytest.approx(0.0, abs=1e-4)
        assert parts["same_class"] == pytest.approx(0.0, abs=1e-4)
        assert parts["diff_class"] == pytest.approx(0.0, abs=1e-4)
        assert loss == pytest.approx(0.0, abs=1e-4)

    def test_margin_pushes_classes_apart(self):
        batch = make_batch([0, 1], [0, 0], [0, 1])
        embedding = torch.zeros(2, 2)
        objective = PachnerPairwise(base_loss(), margin=3.0)

        _, parts = objective(torch.zeros(2, 2), embedding, batch)

        assert parts["diff_class"] == pytest.approx(9.0, abs=1e-3)

    def test_wrong_number_of_kind_weights_raises(self):
        with pytest.raises(ValueError, match="kind_weights"):
            PachnerPairwise(base_loss(), kind_weights=[1.0, 1.0])

    def test_needs_walk_attributes(self):
        batch = make_batch([0, 1], [0, 0], [0, 1], with_walks=False)
        with pytest.raises(ValueError, match="walk_length"):
            PachnerPairwise(base_loss())(
                torch.zeros(2, 2), torch.zeros(2, 2), batch
            )


class TestPachnerInfoNCE:
    def test_aligned_classes_score_better(self):
        y = torch.tensor([0, 0, 1, 1])
        aligned = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        )
        mixed = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        assert supervised_contrastive(
            aligned, y, 0.1
        ) < supervised_contrastive(mixed, y, 0.1)

    def test_works_without_walk_attributes(self):
        batch = make_batch([0, 1, 2, 3], [0] * 4, [0, 0, 1, 1], False)
        embedding = torch.randn(4, 4, requires_grad=True)

        loss, parts = PachnerInfoNCE(base_loss(), ce_weight=1.0)(
            torch.zeros(4, 2), embedding, batch
        )
        loss.backward()

        assert set(parts) == {"infonce", "ce"}
        assert embedding.grad.abs().sum() > 0

    def test_batch_without_positives_contributes_nothing(self):
        y = torch.tensor([0, 1])
        embedding = torch.randn(2, 4)

        assert supervised_contrastive(embedding, y, 0.1) == 0.0
