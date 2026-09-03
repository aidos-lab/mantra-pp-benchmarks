"""Tests for src/utils/imbalance_handling.py."""

import numpy as np
import pytest
import torch

from src.utils.imbalance_handling import (
    convert_to_list,
    derive_weights,
    normalise_imbalance,
    sort_imbalance,
    compute_target_frequency,
    sorted_imbalance_weights,
)
from collections import Counter


class FakeDataset:
    def __init__(self, labels):
        self.y = torch.tensor(labels, dtype=torch.long)


class TestConvertToList:
    def test_basic(self):
        c = Counter({0: 10, 1: 20})
        keys, values = convert_to_list(c)
        assert set(keys) == {0, 1}
        assert set(values) == {10, 20}


class TestDeriveWeights:
    def test_balanced_classes(self):
        weights = derive_weights([100, 100])
        np.testing.assert_allclose(weights, [1.0, 1.0])

    def test_imbalanced_classes(self):
        # 90 of class 0, 10 of class 1 -> total=100, n=2
        # weight_0 = 100 / (90 * 2) = 0.556
        # weight_1 = 100 / (10 * 2) = 5.0
        weights = derive_weights([90, 10])
        np.testing.assert_allclose(weights[0], 100 / 180, rtol=1e-5)
        np.testing.assert_allclose(weights[1], 100 / 20, rtol=1e-5)

    def test_three_classes(self):
        weights = derive_weights([60, 30, 10])
        total = 100
        n = 3
        expected = [total / (60 * n), total / (30 * n), total / (10 * n)]
        np.testing.assert_allclose(weights, expected, rtol=1e-5)


class TestNormaliseImbalance:
    def test_sums_to_one(self):
        result = normalise_imbalance([0.5, 1.5, 3.0])
        assert abs(sum(result) - 1.0) < 1e-10

    def test_values(self):
        result = normalise_imbalance([2.0, 8.0])
        np.testing.assert_allclose(result, [0.2, 0.8])


class TestSortImbalance:
    def test_identity_sort(self):
        # sort_imbalance just returns [imbalance[i] for i in range(len(keys))]
        # This is an identity operation — it does NOT sort by keys
        result = sort_imbalance([10, 20, 30], [0, 1, 2])
        assert result == [10, 20, 30]

    def test_does_not_reorder(self):
        # Even if keys are out of order, it just indexes by position
        result = sort_imbalance([10, 20, 30], [2, 0, 1])
        assert result == [10, 20, 30]


class TestComputeTargetFrequency:
    def test_basic(self):
        ds = FakeDataset([0, 0, 0, 1, 1, 2])
        keys, values = compute_target_frequency(ds)
        freq = dict(zip(keys, values))
        assert freq[0] == 3
        assert freq[1] == 2
        assert freq[2] == 1


class TestSortedImbalanceWeights:
    def test_balanced_dataset(self):
        ds = FakeDataset([0] * 50 + [1] * 50)
        weights = sorted_imbalance_weights(ds)
        np.testing.assert_allclose(weights, [0.5, 0.5], rtol=1e-5)

    def test_imbalanced_dataset(self):
        ds = FakeDataset([0] * 90 + [1] * 10)
        weights = sorted_imbalance_weights(ds)
        # Minority class should get higher weight
        assert weights[1] > weights[0]
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_three_classes(self):
        ds = FakeDataset([0] * 100 + [1] * 50 + [2] * 50)
        weights = sorted_imbalance_weights(ds)
        assert abs(sum(weights) - 1.0) < 1e-10
        # Class 0 (majority) should have lowest weight
        assert weights[0] < weights[1]
