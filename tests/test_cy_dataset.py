"""Tests for the Calabi-Yau dataset path of main.create_dataset.

Skipped when the installed mantra does not ship CalabiYau with split
support (both only exist on the mantra cy branch)."""

import inspect
import os

import pytest
import torch

try:
    from mantra.datasets import CalabiYau
except ImportError:
    CalabiYau = None

if CalabiYau is None or (
    "split_type" not in inspect.signature(CalabiYau).parameters
):
    pytest.skip(
        "requires the mantra cy branch with CalabiYau split support",
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
# enumerates the samples, h12 cycles through three values, and the
# even c2_anticanonical column covers both signs.
N_ROWS = 30


def _write_parquet(tmp_path):
    simplices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    rows = [
        {
            "simplices": simplices,
            "vertices": vertices,
            "h11": i,
            "h12": i % 3,
            "c2_anticanonical": 2 * i - 10,
        }
        for i in range(N_ROWS)
    ]
    df = pd.DataFrame({key: [r[key] for r in rows] for key in rows[0]})
    path = tmp_path / "manifolds.parquet"
    pq.write_table(pa.Table.from_pandas(df), str(path))
    return str(path)


def _cfg(tmp_path, *overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name="config",
            overrides=[
                "dataset=cy",
                f"dataset.local_path={_write_parquet(tmp_path)}",
                f"dataset.root={tmp_path / 'root'}",
                *overrides,
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


class TestCYRegression:
    def test_split_sizes_and_targets(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge")
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
        cfg = _cfg(tmp_path, "task=hodge")
        train, val, test, _ = main.create_dataset(cfg)
        seen = sorted(
            int(data.h11) for ds in (train, val, test) for data in ds
        )
        assert seen == list(range(N_ROWS))

    def test_transform_chain_builds_features(self, tmp_path):
        """The default chain (dual graph, random-walk PE, feature
        selection) must leave a ready-to-batch graph sample."""
        cfg = _cfg(tmp_path, "task=hodge")
        train, _, _, _ = main.create_dataset(cfg)
        sample = train[0]
        assert sample.edge_index is not None
        assert sample.x is not None
        assert sample.x.shape[0] == sample.num_nodes

    def test_alternate_target_via_override(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge", "task.target=h12")
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

    def test_rare_values_keep_their_neuron(self, tmp_path):
        """h11 is unique per sample, so most values are absent from
        train; the mapping must still cover all of them."""
        cfg = _cfg(tmp_path, "task=hodge_class")
        train, val, test, label_to_index = main.create_dataset(cfg)
        assert label_to_index == {i: i for i in range(N_ROWS)}
        for dataset in (val, test):
            for data in dataset:
                assert data.y.item() == label_to_index[int(data.h11)]

    def test_mapping_is_deterministic(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        _, _, _, first = main.create_dataset(cfg)
        _, _, _, again = main.create_dataset(cfg)
        assert first == again
