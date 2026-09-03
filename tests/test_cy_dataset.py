"""Tests for the Calabi-Yau dataset path of main.create_dataset.

Skipped when the installed mantra does not ship CalabiYauDataset (the
mantra cy branch with its split support)."""

import os
from collections import Counter

import pytest
import torch

try:
    from mantra.datasets import CalabiYauDataset  # noqa: F401
except ImportError:
    pytest.skip(
        "requires the mantra cy branch with CalabiYauDataset",
        allow_module_level=True,
    )

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import main

CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config")
)

# 30 star triangulations of the cone over a square boundary; h11
# enumerates the samples and h12 cycles through three values.
N_ROWS = 30


def _rows(n=N_ROWS):
    simplices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    return [
        {
            "simplices": simplices,
            "vertices": vertices,
            "h11": i,
            "h12": i % 3,
        }
        for i in range(n)
    ]


def _write_parquet(tmp_path, rows=None):
    rows = rows or _rows()
    df = pd.DataFrame({key: [r[key] for r in rows] for key in rows[0]})
    path = tmp_path / "manifolds.parquet"
    pq.write_table(pa.Table.from_pandas(df), str(path))
    return str(path)


def _cfg(tmp_path, *overrides, rows=None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cy",
                f"dataset.local_path={_write_parquet(tmp_path, rows)}",
                f"dataset.root={tmp_path / 'root'}",
                *overrides,
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


class TestCYRegression:
    def test_split_sizes_and_targets(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg")
        train, val, test, label_to_index = main.create_dataset(cfg)

        # 0.7 / 0.1 / 0.2 of 30 samples; sklearn ceils the fractional
        # val cut (24 * (0.1 / 0.8) is just above 3 in floats).
        assert (len(train), len(val), len(test)) == (20, 4, 6)
        assert label_to_index == {}

        for dataset in (train, val, test):
            for data in dataset:
                assert data.y.shape == (1, 1)
                assert data.y.dtype == torch.float32
                assert data.y.item() == float(data.h11)

    def test_splits_partition_the_dataset(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg")
        train, val, test, _ = main.create_dataset(cfg)
        seen = sorted(
            int(data.h11) for ds in (train, val, test) for data in ds
        )
        assert seen == list(range(N_ROWS))

    def test_transform_chain_builds_features(self, tmp_path):
        """The default chain (dual graph, random-walk PE, feature
        selection) must leave a ready-to-batch graph sample."""
        cfg = _cfg(tmp_path, "task=hodge_reg")
        train, _, _, _ = main.create_dataset(cfg)
        sample = train[0]
        assert sample.edge_index is not None
        assert sample.x is not None
        assert sample.x.shape[0] == sample.num_nodes

    def test_alternate_target_via_override(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg", "task.target=h12")
        train, _, _, _ = main.create_dataset(cfg)
        for data in train:
            assert data.y.item() == float(data.h12)


class TestCYClassification:
    def test_mapping_spans_the_full_dataset(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        train, val, test, label_to_index = main.create_dataset(cfg)

        # h12 only attains 0, 1, 2; the mapping is sorted and
        # contiguous and every split respects it.
        assert label_to_index == {0: 0, 1: 1, 2: 2}
        for dataset in (train, val, test):
            for data in dataset:
                assert data.y.item() == label_to_index[int(data.h12)]
                assert int(data.label) == int(data.h12)


    def test_mapping_is_deterministic(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        _, _, _, first = main.create_dataset(cfg)
        _, _, _, again = main.create_dataset(cfg)
        assert first == again


class TestCYSplitOptions:
    def test_stratified_split_balances_label_source(self, tmp_path):
        cfg = _cfg(
            tmp_path,
            "task=hodge_reg",
            "dataset.use_stratified=True",
            "dataset.label_source=h12",
        )
        splits = main.create_dataset(cfg)[:3]
        assert sum(len(ds) for ds in splits) == N_ROWS
        for dataset in splits:
            counts = Counter(int(data.h12) for data in dataset)
            assert set(counts) == {0, 1, 2}
            assert max(counts.values()) - min(counts.values()) <= 1
