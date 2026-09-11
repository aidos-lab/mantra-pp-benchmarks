"""Tests for the shared readout in src/models/readout.py."""

import pytest
import torch

from src.models.readout import RankReadout, Readout, pool


class TestPool:
    def test_default_is_sum(self):
        x = torch.tensor([[1.0], [2.0], [4.0]])
        batch = torch.tensor([0, 0, 1])
        assert torch.equal(pool(x, batch), torch.tensor([[3.0], [4.0]]))

    def test_dim_size_pads_missing_complexes(self):
        x = torch.tensor([[1.0], [2.0]])
        batch = torch.tensor([0, 0])
        out = pool(x, batch, dim_size=3)
        assert torch.equal(out, torch.tensor([[3.0], [0.0], [0.0]]))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="pool_method"):
            pool(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long), "median")


class TestReadout:
    def test_logits_are_the_head_of_the_embedding(self):
        readout = Readout(2, 3, "sum")
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch = torch.tensor([0, 0, 1])

        logits, embedding = readout(x, batch)

        assert torch.equal(embedding, torch.tensor([[4.0, 6.0], [5.0, 6.0]]))
        assert torch.allclose(logits, readout.head(embedding))

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="pool_method"):
            Readout(2, 3, "median")


class TestRankReadout:
    def two_complexes(self):
        """Constant features c_r on every r-simplex.

        Complex 0: a triangle boundary filled (3 vertices, 3 edges,
        1 face). Complex 1: two triangles sharing an edge (4, 5, 2).
        """
        xs = [
            torch.ones(7, 2) * 1.0,   # 0-simplices
            torch.ones(8, 2) * 10.0,  # 1-simplices
            torch.ones(3, 2) * 100.0,  # 2-simplices
        ]
        batches = [
            torch.tensor([0, 0, 0, 1, 1, 1, 1]),
            torch.tensor([0, 0, 0, 1, 1, 1, 1, 1]),
            torch.tensor([0, 1, 1]),
        ]
        return xs, batches

    def test_concat_stacks_the_pooled_ranks(self):
        xs, batches = self.two_complexes()
        readout = RankReadout([2, 2, 2], 3, "sum", "concat")

        logits, embedding = readout(xs, batches)

        expected = torch.tensor(
            [[3.0, 3.0, 30.0, 30.0, 100.0, 100.0], [4.0, 4.0, 50.0, 50.0, 200.0, 200.0]]
        )
        assert torch.equal(embedding, expected)
        assert torch.allclose(logits, readout.head(embedding))

    def test_alternating_sum_counts_with_signs(self):
        xs, batches = self.two_complexes()
        readout = RankReadout([2, 2, 2], 3, "sum", "alternating")

        _, embedding = readout(xs, batches)

        # f_0 c_0 - f_1 c_1 + f_2 c_2 per complex.
        expected = torch.tensor([[3 - 30 + 100.0] * 2, [4 - 50 + 200.0] * 2])
        assert torch.equal(embedding, expected)

    def test_alternating_sum_of_unit_features_is_the_euler_characteristic(self):
        xs, batches = self.two_complexes()
        xs = [torch.ones_like(x) for x in xs]
        readout = RankReadout([2, 2, 2], 3, "sum", "alternating")

        _, embedding = readout(xs, batches)

        # Both complexes are discs: chi = 1.
        assert torch.equal(embedding, torch.ones(2, 2))

    def test_complex_without_a_rank_still_gets_a_row(self):
        xs = [torch.ones(3, 1), torch.ones(2, 1), torch.ones(1, 1)]
        # Complex 1 has a vertex and an edge but no face.
        batches = [torch.tensor([0, 0, 1]), torch.tensor([0, 1]), torch.tensor([0])]
        readout = RankReadout([1, 1, 1], 2, "sum", "alternating")

        _, embedding = readout(xs, batches)

        assert embedding.shape == (2, 1)
        assert embedding[1].item() == pytest.approx(1 - 1 + 0)

    def test_alternating_needs_equal_widths(self):
        with pytest.raises(ValueError, match="same width"):
            RankReadout([4, 4, 8], 2, "sum", "alternating")

    def test_unknown_rank_readout_raises(self):
        with pytest.raises(ValueError, match="rank_readout"):
            RankReadout([4, 4, 4], 2, "sum", "stack")
