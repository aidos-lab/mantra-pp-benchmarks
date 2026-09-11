"""Tests for the walk batch sampler."""

import pytest
import torch
from torch_geometric.data import Data

from src.utils.sampling import WalkBatchSampler


def make_dataset(n_bases=12, walk_length=2, with_walks=True):
    """Dataset of walk snapshots, ordered base-major as on disk."""
    entries = []
    for base in range(n_bases):
        for step in range(walk_length + 1):
            data = Data(y=torch.tensor(base % 3))
            if with_walks:
                data.walk_base = base
                data.walk_step = step
            entries.append(data)
    return entries


class TestValidation:
    def test_dataset_without_walks_raises(self):
        with pytest.raises(ValueError, match="walk_length"):
            WalkBatchSampler(make_dataset(with_walks=False), batch_size=8)

    def test_batch_smaller_than_a_walk_raises(self):
        with pytest.raises(ValueError, match="smaller than a walk"):
            WalkBatchSampler(make_dataset(walk_length=3), batch_size=3)


class TestBatches:
    def test_batches_hold_whole_walks(self):
        dataset = make_dataset(n_bases=12, walk_length=2)
        sampler = WalkBatchSampler(dataset, batch_size=8)

        for batch in sampler:
            bases = [int(dataset[i].walk_base) for i in batch]
            for base in set(bases):
                assert bases.count(base) == 3
            # 8 // 3 = 2 walks of 3 snapshots per batch.
            assert len(batch) == 6

    def test_an_epoch_is_a_pass_over_the_dataset(self):
        dataset = make_dataset(n_bases=10, walk_length=1)
        sampler = WalkBatchSampler(dataset, batch_size=6)

        indices = sorted(i for batch in sampler for i in batch)

        assert indices == list(range(len(dataset)))
        # 10 walks, 3 per batch -> 4 batches, the last one shorter.
        assert len(sampler) == 4
        assert len(list(sampler)) == 4

    def test_same_seed_gives_the_same_first_epoch(self):
        dataset = make_dataset()
        first = list(WalkBatchSampler(dataset, batch_size=8, seed=1))
        second = list(WalkBatchSampler(dataset, batch_size=8, seed=1))

        assert first == second

    def test_consecutive_epochs_differ(self):
        sampler = WalkBatchSampler(make_dataset(n_bases=30), batch_size=8)

        assert list(sampler) != list(sampler)
