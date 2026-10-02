"""Tests for the Calabi-Yau dataset path of main.create_dataset.

Skipped when the installed mantra does not ship CalabiYauDataset (the
mantra cy branch with its split support)."""

import os
from collections import Counter

import pytest
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree

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
# enumerates the samples and h12 cycles through three values. The
# per-vertex `c2` mimics c2 . D_i: vertex 0 is the origin and holds
# minus the sum of the other entries.
N_ROWS = 30


def _c2(i):
    return [-(20 + 2 * (i % 3)), 2 + 2 * (i % 3), 4, 6, 8]


def _rows(n=N_ROWS):
    simplices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    return [
        {
            "simplices": simplices,
            "vertices": vertices,
            "h11": i,
            "h12": i % 3,
            "c2": _c2(i),
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

    def test_min_sample_per_class_drops_rare_values(self, tmp_path):
        rows = _rows()
        rows[0]["h12"] = 7
        cfg = _cfg(
            tmp_path,
            "task=hodge_reg",
            "dataset.use_stratified=True",
            "dataset.label_source=h12",
            "dataset.min_sample_per_class=1",
            rows=rows,
        )
        splits = main.create_dataset(cfg)[:3]
        assert sum(len(ds) for ds in splits) == N_ROWS - 1
        assert all(int(data.h12) != 7 for ds in splits for data in ds)


ONE_SKELETON = "representation@dataset.transform.representation=one_skeleton"


class TestCYNodeLevel:
    def test_regression_targets_per_vertex(self, tmp_path):
        cfg = _cfg(tmp_path, "task=chern_node", ONE_SKELETON)
        train, val, test, label_to_index = main.create_dataset(cfg)

        assert label_to_index == {}
        for dataset in (train, val, test):
            for data in dataset:
                assert data.y.shape == (5, 1)
                assert data.y.dtype == torch.float32
                assert data.y.flatten().tolist() == _c2(int(data.h11))
                # one_skeleton: node k is vertex k
                assert data.num_nodes == 5

    def test_loaders_batch_per_vertex(self, tmp_path):
        cfg = _cfg(tmp_path, "task=chern_node", ONE_SKELETON,
                   "trainer.batch_size=3")
        train, val, test, _ = main.create_dataset(cfg)
        _, val_loader, _ = main.create_loaders(
            cfg, train, val, test, is_graph_representation=True
        )

        batch = next(iter(val_loader))
        n_graphs = batch.num_graphs
        assert batch.y.shape == (5 * n_graphs, 1)
        assert batch.num_nodes == 5 * n_graphs

        # The rows of `y` follow the node order of the batch.
        for g in range(n_graphs):
            rows = batch.y[batch.batch == g].flatten().tolist()
            assert rows == val[g].y.flatten().tolist()

    def test_shuffled_loader_keeps_targets_with_their_graph(self, tmp_path):
        cfg = _cfg(tmp_path, "task=chern_node", ONE_SKELETON,
                   "trainer.batch_size=4")
        train, val, test, _ = main.create_dataset(cfg)
        train_loader, _, _ = main.create_loaders(
            cfg, train, val, test, is_graph_representation=True
        )

        seen = []
        for batch in train_loader:
            degrees = degree(batch.edge_index[0], batch.num_nodes)
            for g in range(batch.num_graphs):
                rows = batch.y[batch.batch == g].flatten().tolist()
                # h11 identifies the sample; node ptr[g] is its origin,
                # the hub adjacent to the other four vertices.
                assert rows == _c2(int(batch.h11[g]))
                assert degrees[batch.ptr[g]] == 4
                seen.append(int(batch.h11[g]))
        assert sorted(seen) == sorted(int(data.h11) for data in train)

    def test_node_classification_remaps_train_split_values(self, tmp_path):
        cfg = _cfg(
            tmp_path, "task=hodge_class", "task.target=c2",
            "task.task_level=node", ONE_SKELETON,
        )
        train, val, test, label_to_index = main.create_dataset(cfg)

        values = sorted({v for data in train for v in _c2(int(data.h11))})
        assert label_to_index == {v: k for k, v in enumerate(values)}
        for dataset in (train, val, test):
            for data in dataset:
                assert data.y.dtype == torch.long
                assert data.y.tolist() == [
                    label_to_index[v] for v in _c2(int(data.h11))
                ]
                assert data.label.tolist() == _c2(int(data.h11))

    def test_vertex_value_missing_from_train_raises(self, tmp_path):
        # Vertex 4 carries a value unique to its sample, so every val
        # and test sample holds a class the train split never saw.
        rows = _rows()
        for i, row in enumerate(rows):
            row["c2"] = row["c2"][:4] + [100 + i]
        cfg = _cfg(
            tmp_path, "task=hodge_class", "task.target=c2",
            "task.task_level=node", ONE_SKELETON, rows=rows,
        )
        with pytest.raises(ValueError, match="not in train"):
            main.create_dataset(cfg)

    def test_graph_classification_keeps_the_train_split_remap(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        train, _, _, label_to_index = main.create_dataset(cfg)

        assert label_to_index == {0: 0, 1: 1, 2: 2}
        assert {int(data.y) for data in train} == {0, 1, 2}

    def test_train_and_validate_end_to_end(self, tmp_path):
        import torch.nn as nn
        import torchmetrics

        from src.utils.train_utils import train as _train, validate

        cfg = _cfg(tmp_path, "task=chern_node", ONE_SKELETON, "model=gcn",
                   "trainer.batch_size=8")
        train, val, test, label_to_index = main.create_dataset(cfg)
        train_loader, val_loader, _ = main.create_loaders(
            cfg, train, val, test, is_graph_representation=True
        )

        model = main.create_model(cfg, train[0].x.shape[1], label_to_index)
        assert cfg.model.task_level == "node"
        assert model.task_level == "node"

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        metrics = _train(
            model, optimizer, train_loader, nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())], cfg,
        )
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))
        metrics = validate(
            model, val_loader, nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())], cfg,
        )
        assert torch.isfinite(torch.tensor(metrics["val/loss"]))


class TestCYTransformCache:
    def test_chain_is_cached_and_slimmed(self, tmp_path):
        """The representation chain runs once at process time; the cached
        samples carry neither the Python objects nor the raw labels."""
        cfg = _cfg(tmp_path, "task=hodge_reg", ONE_SKELETON)
        train, _, _, _ = main.create_dataset(cfg)

        assert isinstance(train.pre_transform, Compose)
        assert not isinstance(train.transform, Compose)
        cached = train.get(0)
        assert "edge_index" in cached and "x" in cached
        for key in ("triangulation", "simplex_trie", "intersection_numbers", "random_walk_pe"):
            assert key not in cached
        assert train[0].y.shape == (1, 1)

    def test_cache_name_follows_the_transform_config(self, tmp_path):
        skeleton = _cfg(tmp_path, "task=hodge_reg", ONE_SKELETON)
        dual = _cfg(tmp_path, "task=hodge_reg",
                    "representation@dataset.transform.representation=dual_graph")
        assert main.cache_name(skeleton.dataset) != main.cache_name(dual.dataset)
        assert main.cache_name(skeleton.dataset) == main.cache_name(
            _cfg(tmp_path, "task=hodge_class", ONE_SKELETON).dataset
        )

        train_a, _, _, _ = main.create_dataset(skeleton)
        train_b, _, _, _ = main.create_dataset(dual)
        assert train_a.processed_dir != train_b.processed_dir
        assert train_a.get(0).num_nodes != train_b.get(0).num_nodes
