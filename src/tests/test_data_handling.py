"""Tests for src/utils/data_handling.py — split_dataset and filter_low_ocurrence_labels."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.utils.data_handling import filter_low_ocurrence_labels, split_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeDataset:
    """Minimal dataset-like object with .y and indexing support."""

    def __init__(self, labels):
        self.y = torch.tensor(labels, dtype=torch.long)
        self._data = [Data(x=torch.randn(2, 3), y=torch.tensor([l])) for l in labels]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
            indices = idx.nonzero(as_tuple=True)[0].tolist()
            return FakeDataset([self.y[i].item() for i in indices])
        return self._data[idx]


def _make_cfg(seed=42, test_split=0.2, val_split=0.2, train_split=0.6,
              batch_size=4, use_stratified=True):
    """Build a minimal config namespace matching what split_dataset expects."""
    return SimpleNamespace(
        seed=seed,
        dataset=SimpleNamespace(
            train_parameters=SimpleNamespace(
                split=SimpleNamespace(train=train_split, val=val_split, test=test_split)
            ),
            use_stratified=use_stratified,
        ),
        trainer=SimpleNamespace(batch_size=batch_size),
    )


# ---------------------------------------------------------------------------
# filter_low_ocurrence_labels
# ---------------------------------------------------------------------------

class TestFilterLowOccurrenceLabels:

    def test_keeps_classes_above_threshold(self):
        # 200 samples of class 0, 50 of class 1 — threshold is 100
        labels = [0] * 200 + [1] * 50
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds, n=100)
        # class 0 has 200 > 100, class 1 has 50 <= 100
        assert len(filtered) == 200

    def test_removes_classes_at_or_below_threshold(self):
        labels = [0] * 100 + [1] * 101
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds, n=100)
        # class 0 has exactly 100, NOT > 100, so filtered out
        # class 1 has 101 > 100, kept
        assert len(filtered) == 101

    def test_all_classes_above_threshold(self):
        labels = [0] * 200 + [1] * 150
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds, n=100)
        assert len(filtered) == 350

    def test_all_classes_below_threshold(self):
        labels = [0] * 50 + [1] * 30
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds, n=100)
        assert len(filtered) == 0

    def test_default_threshold_is_100(self):
        labels = [0] * 101 + [1] * 100
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds)
        assert len(filtered) == 101

    def test_preserves_label_values(self):
        labels = [2] * 200 + [5] * 50
        ds = FakeDataset(labels)
        filtered = filter_low_ocurrence_labels(ds, n=100)
        assert all(l == 2 for l in filtered.y.tolist())


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------

class TestSplitDataset:

    def test_split_sizes_are_correct(self):
        labels = [0] * 100 + [1] * 100
        ds = FakeDataset(labels)
        cfg = _make_cfg(use_stratified=False)
        train_loader, val_loader, test_loader = split_dataset(ds, cfg)

        train_count = sum(len(batch.y) for batch in train_loader)
        val_count = sum(len(batch.y) for batch in val_loader)
        test_count = sum(len(batch.y) for batch in test_loader)

        total = train_count + val_count + test_count
        assert total == 200
        # Approximate splits: 60/20/20
        assert abs(test_count - 40) <= 5
        assert abs(val_count - 40) <= 5

    def test_deterministic_with_same_seed(self):
        labels = [0] * 100 + [1] * 100
        ds = FakeDataset(labels)

        cfg1 = _make_cfg(seed=42, use_stratified=False)
        cfg2 = _make_cfg(seed=42, use_stratified=False)

        _, _, test1 = split_dataset(ds, cfg1)
        _, _, test2 = split_dataset(ds, cfg2)

        y1 = torch.cat([b.y for b in test1])
        y2 = torch.cat([b.y for b in test2])
        assert torch.equal(y1, y2)

    def test_different_seed_gives_different_split(self):
        labels = list(range(200))
        ds = FakeDataset(labels)

        cfg1 = _make_cfg(seed=42, use_stratified=False)
        cfg2 = _make_cfg(seed=99, use_stratified=False)

        _, _, test1 = split_dataset(ds, cfg1)
        _, _, test2 = split_dataset(ds, cfg2)

        y1 = set(torch.cat([b.y for b in test1]).tolist())
        y2 = set(torch.cat([b.y for b in test2]).tolist())
        assert y1 != y2
