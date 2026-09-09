"""Tests for the Pachner pair batch sampler."""

from collections import Counter

import pytest
import torch
from torch_geometric.data import Data

from src.utils.sampling import PachnerPairSampler


def make_dataset(n_bases=12, walk_length=2, n_classes=3, with_walks=True):
    """Dataset of walk snapshots, ordered base-major as on disk."""
    entries = []
    for base in range(n_bases):
        for step in range(walk_length + 1):
            data = Data(y=torch.tensor(base % n_classes))
            if with_walks:
                data.walk_base = base
                data.walk_step = step
            entries.append(data)
    return entries


def pair_kinds(batch, dataset):
    """Count the pair kinds of the consecutive pairs of a batch."""
    counts = Counter()
    for first, second in zip(batch[::2], batch[1::2]):
        a, b = dataset[first], dataset[second]
        if int(a.walk_base) == int(b.walk_base):
            counts["walk"] += 1
        elif int(a.y) == int(b.y):
            counts["class"] += 1
        else:
            counts["diff"] += 1
    return counts


class TestValidation:
    def test_dataset_without_walks_raises(self):
        with pytest.raises(ValueError, match="walk_length"):
            PachnerPairSampler(make_dataset(with_walks=False), batch_size=8)

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            PachnerPairSampler(
                make_dataset(), batch_size=8, fractions=[0.5, 0.5, 0.5]
            )

    def test_wrong_number_of_fractions_raises(self):
        with pytest.raises(ValueError, match="fractions"):
            PachnerPairSampler(
                make_dataset(), batch_size=8, fractions=[0.5, 0.5]
            )

    def test_tiny_batch_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            PachnerPairSampler(make_dataset(), batch_size=1)


class TestBatches:
    def test_batches_respect_the_requested_fractions(self):
        dataset = make_dataset(n_bases=40)
        sampler = PachnerPairSampler(
            dataset, batch_size=20, fractions=[0.5, 0.25, 0.25]
        )

        # Ten pairs per batch; the walk pairs absorb the rounding of
        # the two 2.5-pair kinds.
        for batch in sampler:
            counts = pair_kinds(batch, dataset)
            assert sum(counts.values()) == 10
            assert counts["walk"] == 6
            assert counts["class"] == 2
            assert counts["diff"] == 2

    def test_indices_within_a_batch_are_unique(self):
        dataset = make_dataset(n_bases=40)
        sampler = PachnerPairSampler(dataset, batch_size=16)

        for batch in sampler:
            assert len(set(batch)) == len(batch)

    def test_epoch_count_covers_the_dataset(self):
        dataset = make_dataset(n_bases=40)
        sampler = PachnerPairSampler(dataset, batch_size=16)

        assert len(sampler) == len(dataset) // 16
        assert len(list(sampler)) == len(sampler)

    def test_same_seed_gives_the_same_first_epoch(self):
        dataset = make_dataset(n_bases=40)
        first = list(PachnerPairSampler(dataset, batch_size=16, seed=7))
        second = list(PachnerPairSampler(dataset, batch_size=16, seed=7))
        other = list(PachnerPairSampler(dataset, batch_size=16, seed=8))

        assert first == second
        assert first != other

    def test_consecutive_epochs_differ(self):
        sampler = PachnerPairSampler(make_dataset(n_bases=40), batch_size=16)

        assert list(sampler) != list(sampler)

    def test_walk_pairs_only_fall_back_when_classes_are_missing(self):
        dataset = make_dataset(n_bases=10, n_classes=1)
        sampler = PachnerPairSampler(
            dataset, batch_size=8, fractions=[0.0, 0.0, 1.0]
        )

        for batch in sampler:
            counts = pair_kinds(batch, dataset)
            assert counts["diff"] == 0
