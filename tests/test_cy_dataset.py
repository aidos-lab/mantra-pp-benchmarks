"""Tests for the Calabi-Yau dataset path of main.create_dataset.

Skipped when the installed mantra does not ship CalabiYauDataset (the
mantra cy branch with its split support)."""

import inspect
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

N_ROWS = 30


def _c2(i):
    return [-(20 + 2 * (i % 3)), 2 + 2 * (i % 3), 4, 6, 8]


def _rows(n=N_ROWS):
    """`n` star triangulations of the cone over a square boundary: h11
    enumerates the samples, h12 cycles through three values, and `c2`
    mimics c2 . D_i with the origin (vertex 0) holding minus the sum of
    the other entries. The 2-d fan has no boundary triangle, so the
    intersection numbers reach no target; `_kappa_rows` has them."""
    simplices = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]]
    vertices = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]]
    return [
        {
            "simplices": simplices,
            "vertices": vertices,
            "h11": i,
            "h12": i % 3,
            "c2": _c2(i),
            "intersection_numbers": [[0, 1, 2, i], [2, 3, 0, 5], [1, 1, 2, 9]],
        }
        for i in range(n)
    ]


def _kappa_rows(n=N_ROWS):
    """Cone over the boundary of a tetrahedron: the boundary triangles
    (1,2,3), (1,2,4), (1,3,4), (2,3,4) follow the six through the origin.
    Triangle (1,2,3) holds h11 and (2,3,4) holds 5; the rows through the
    origin and the self-intersection are not targets."""
    rows = _rows(n)
    for i, row in enumerate(rows):
        row["simplices"] = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]]
        row["vertices"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
        row["intersection_numbers"] = [[1, 2, 3, i], [2, 3, 4, 5], [0, 1, 2, 7], [1, 1, 2, 9]]
    return rows


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
                "dataset.pre_transform_workers=1",
                *overrides,
            ],
        )
    OmegaConf.resolve(cfg)
    return cfg


class TestCYRegression:
    def test_split_sizes_and_targets(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg")
        train, val, test, _, label_to_index = main.create_dataset(cfg)

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
        train, val, test, _, _ = main.create_dataset(cfg)
        seen = sorted(
            int(data.h11) for ds in (train, val, test) for data in ds
        )
        assert seen == list(range(N_ROWS))

    def test_transform_chain_builds_features(self, tmp_path):
        """The default chain (dual graph, random-walk PE, feature
        selection) must leave a ready-to-batch graph sample."""
        cfg = _cfg(tmp_path, "task=hodge_reg")
        train, _, _, _, _ = main.create_dataset(cfg)
        sample = train[0]
        assert sample.edge_index is not None
        assert sample.x is not None
        assert sample.x.shape[0] == sample.num_nodes

    def test_alternate_target_via_override(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg", "task.target=h12")
        train, _, _, _, _ = main.create_dataset(cfg)
        for data in train:
            assert data.y.item() == float(data.h12)


class TestCYClassification:
    def test_mapping_spans_the_full_dataset(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        train, val, test, _, label_to_index = main.create_dataset(cfg)

        # h12 only attains 0, 1, 2; the mapping is sorted and
        # contiguous and every split respects it.
        assert label_to_index == {0: 0, 1: 1, 2: 2}
        for dataset in (train, val, test):
            for data in dataset:
                assert data.y.item() == label_to_index[int(data.h12)]
                assert int(data.label) == int(data.h12)


    def test_mapping_is_deterministic(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_class", "task.target=h12")
        _, _, _, _, first = main.create_dataset(cfg)
        _, _, _, _, again = main.create_dataset(cfg)
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
        train, val, test, _, label_to_index = main.create_dataset(cfg)

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
        train, val, test, _, _ = main.create_dataset(cfg)
        _, val_loader, _, _ = main.create_loaders(cfg, train, val, test)

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
        train, val, test, _, _ = main.create_dataset(cfg)
        train_loader, _, _, _ = main.create_loaders(cfg, train, val, test)

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
        train, val, test, _, label_to_index = main.create_dataset(cfg)

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
        train, _, _, _, label_to_index = main.create_dataset(cfg)

        assert label_to_index == {0: 0, 1: 1, 2: 2}
        assert {int(data.y) for data in train} == {0, 1, 2}

    def test_train_and_validate_end_to_end(self, tmp_path):
        import torch.nn as nn
        import torchmetrics

        from src.utils.train_utils import train as _train, validate

        cfg = _cfg(tmp_path, "task=chern_node", ONE_SKELETON, "model=gcn",
                   "trainer.batch_size=8")
        train, val, test, _, label_to_index = main.create_dataset(cfg)
        train_loader, val_loader, _, _ = main.create_loaders(cfg, train, val, test)

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


class TestCYTriangleLevel:
    def test_regression_targets_per_triangle(self, tmp_path):
        cfg = _cfg(tmp_path, "task=kappa_triangle", ONE_SKELETON, rows=_kappa_rows())
        train, _, _, _, _ = main.create_dataset(cfg)
        train_loader, _, _, _ = main.create_loaders(cfg, train, train, train)

        for data in train:
            assert data.y.flatten().tolist() == [float(data.h11), 0.0, 0.0, 5.0]
            assert data.simplex_index_2[:, 0].tolist() == [1, 2, 3]
            # The graph chain's select_attributes drops the mask of the simplicial models.
            assert "triangle_mask" not in data
        for batch in train_loader:
            assert batch.y.shape == (4 * batch.num_graphs, 1)
            assert batch.simplex_index_2.shape == (3, 4 * batch.num_graphs)
            graphs = torch.arange(batch.num_graphs).repeat_interleave(4)
            assert batch.batch[batch.simplex_index_2].eq(graphs).all()

    def test_simplicial_chain_masks_the_rank_2_skeleton(self, tmp_path):
        cfg = _cfg(
            tmp_path, "task=kappa_triangle",
            "representation@dataset.transform.representation=incidence_sc",
            "dataset.transform.representation.signed=True",
            "transform@dataset.transform.embedding=propagate_convex_comb",
            "~transform@dataset.transform.select_features",
            "transform@dataset.transform.select_attributes=select_attributes_sc",
            "dataset.transform_order=[representation,embedding,kappa,select_attributes]",
            rows=_kappa_rows(),
        )
        train, _, _, _, _ = main.create_dataset(cfg)
        train_loader, _, _, _ = main.create_loaders(cfg, train, train, train)
        for data in train:
            assert data.triangle_mask.tolist() == [False] * 6 + [True] * 4
            assert data.x_2.shape[0] == 10
            assert data.y.flatten().tolist() == [float(data.h11), 0.0, 0.0, 5.0]
        for batch in train_loader:
            n = batch.y.shape[0] // 4
            assert batch.triangle_mask.tolist() == ([False] * 6 + [True] * 4) * n
            assert batch.x_2.shape[0] == 10 * n

    def test_levi_graph_indices_point_at_vertex_nodes(self, tmp_path):
        cfg = _cfg(
            tmp_path, "task=kappa_triangle",
            "representation@dataset.transform.representation=levi_graph",
            rows=_kappa_rows(),
        )
        train, _, _, _, _ = main.create_dataset(cfg)
        train_loader, _, _, _ = main.create_loaders(cfg, train, train, train)
        for batch in train_loader:
            assert batch.vertex_mask[batch.simplex_index_2].all()
            assert batch.y.shape == (4 * batch.num_graphs, 1)


class TestCYTransformCache:
    def test_chain_is_cached_and_slimmed(self, tmp_path):
        """The representation chain runs once at process time; the cached
        samples carry neither the Python objects nor the raw labels."""
        cfg = _cfg(tmp_path, "task=hodge_reg", ONE_SKELETON)
        train, _, _, _, _ = main.create_dataset(cfg)

        assert isinstance(train.pre_transform, Compose)
        assert not isinstance(train.transform, Compose)
        cached = train.get(0)
        assert "edge_index" in cached and "x" in cached
        for key in ("triangulation", "simplex_trie", "intersection_numbers", "random_walk_pe", "chi"):
            assert key not in cached
        assert train[0].y.shape == (1, 1)

    def test_workers_reach_the_dataset(self, tmp_path):
        cfg = _cfg(tmp_path, "task=hodge_reg", ONE_SKELETON, "dataset.pre_transform_workers=2")
        train, _, _, _, _ = main.create_dataset(cfg)
        assert train.pre_transform_workers == 2

    def test_cache_name_follows_the_transform_config(self, tmp_path):
        skeleton = _cfg(tmp_path, "task=hodge_reg", ONE_SKELETON)
        dual = _cfg(tmp_path, "task=hodge_reg",
                    "representation@dataset.transform.representation=dual_graph")
        assert main.cache_name(skeleton.dataset) != main.cache_name(dual.dataset)
        assert main.cache_name(skeleton.dataset) == main.cache_name(
            _cfg(tmp_path, "task=hodge_class", ONE_SKELETON).dataset
        )


def _phase_rows(n=N_ROWS):
    """`_kappa_rows` as two phases per polytope: rows 2p and 2p + 1 share
    `polytope_id` p and `h11`, triangle (1,2,3) holds 2p in one phase
    and 2p + 1 in the other while (2,3,4) holds 5 in both, and `c2`
    differs between the phases."""
    rows = _kappa_rows(n)
    for i, row in enumerate(rows):
        row["polytope_id"] = i // 2
        row["h11"] = i // 2
    return rows


@pytest.mark.skipif(
    "group_by" not in inspect.signature(CalabiYauDataset.__init__).parameters,
    reason="requires mantra's grouped split (cy-experiments)",
)
class TestCYPhases:
    def _phases(self, tmp_path, task):
        cfg = _cfg(
            tmp_path, f"task={task}", ONE_SKELETON,
            "dataset.group_by=polytope_id", rows=_phase_rows(),
        )
        return main.create_dataset(cfg)[:3]

    def test_group_by_keeps_the_phases_of_a_polytope_together(self, tmp_path):
        train, val, test = self._phases(tmp_path, "kappa_triangle")
        ids = [{int(d.polytope_id) for d in ds} for ds in (train, val, test)]
        assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])
        assert sorted(ids[0] | ids[1] | ids[2]) == list(range(N_ROWS // 2))
        assert all(len(ds) % 2 == 0 for ds in (train, val, test))

    def test_oracle_takes_the_median_over_the_phases(self, tmp_path):
        import baseline
        from torch_geometric.data import Batch

        _, _, test = self._phases(tmp_path, "kappa_triangle")
        assert len(test) > 0
        oracle = baseline.PhaseOracle(test, "triangle")
        batch = Batch.from_data_list(list(test))
        pred = oracle(batch).reshape(-1)
        assert pred.shape == batch.y.reshape(-1).shape
        for data in test:
            p = int(data.polytope_id)
            rows = batch.polytope_id[batch.batch[batch.simplex_index_2[0]]] == p
            # Triangle (1,2,3) comes first: the median of 2p and 2p + 1;
            # then (1,2,4), (1,3,4) at 0 and (2,3,4) at 5 in both phases.
            assert pred[rows].tolist() == [2 * p + 0.5, 0.0, 0.0, 5.0] * 2

    def test_oracle_on_the_vertex_task(self, tmp_path):
        import baseline
        from torch_geometric.data import Batch

        _, _, test = self._phases(tmp_path, "chern_node")
        oracle = baseline.PhaseOracle(test, "node")
        batch = Batch.from_data_list(list(test))
        pred = oracle(batch).reshape(-1)
        assert pred.shape == batch.y.reshape(-1).shape
        for data in test:
            rows = batch.polytope_id[batch.batch] == int(data.polytope_id)
            phases = [d for d in test if int(d.polytope_id) == int(data.polytope_id)]
            assert len(phases) == 2
            # The median of two phases is their mean.
            expected = torch.stack([d.y.reshape(-1) for d in phases]).mean(0)
            assert pred[rows].tolist() == expected.tolist() * 2


try:
    from mantra.transforms import CandidateTriangleIntersectionNumbers  # noqa: F401

    HAS_CANDIDATES = True
except ImportError:
    HAS_CANDIDATES = False


def _candidate_rows(n=N_ROWS):
    """Two phases per polytope on the 2-face {1, 2, 3, 4}: the full cone
    (even rows, all four triples are triangles) and the half cone over
    (1,2,3) and (1,2,4) (odd rows). kappa(1,2,3) is the row index."""
    rows = _phase_rows(n)
    for i, row in enumerate(rows):
        row["face2_points"] = [[0, 1], [0, 2], [0, 3], [0, 4]]
        if i % 2:
            row["simplices"] = [[0, 1, 2, 3], [0, 1, 2, 4]]
            row["intersection_numbers"] = [[1, 2, 3, i], [0, 1, 2, 7]]
    return rows


CANDIDATE = (
    "task=kappa_candidate",
    "transform@dataset.transform.kappa=kappa_with_candidates",
)
CANDIDATES = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]


@pytest.mark.skipif(not HAS_CANDIDATES, reason="requires mantra cy-experiments")
class TestCYCandidateLevel:
    def test_targets_per_candidate_triangle(self, tmp_path):
        cfg = _cfg(tmp_path, *CANDIDATE, ONE_SKELETON, rows=_candidate_rows())
        train, _, _, _, _ = main.create_dataset(cfg)
        for data in train:
            full = data.candidate_present.tolist() == [True] * 4
            assert data.candidate_index_2.t().tolist() == CANDIDATES
            assert data.y.flatten().tolist() == [float(data.kappa_2[0]), 0.0, 0.0, 5.0 if full else 0.0]
        batch = next(iter(main.create_loaders(cfg, train, train, train)[0]))
        assert batch.y.shape == (4 * batch.num_graphs, 1)
        graphs = [g for g in range(batch.num_graphs) for _ in range(4)]
        assert batch.batch[batch.candidate_index_2[0]].tolist() == graphs

    def test_simplicial_chain_offsets_the_candidate_index(self, tmp_path):
        cfg = _cfg(
            tmp_path, *CANDIDATE,
            "representation@dataset.transform.representation=incidence_sc",
            "transform@dataset.transform.embedding=propagate_convex_comb",
            "~transform@dataset.transform.select_features",
            "transform@dataset.transform.select_attributes=select_attributes_sc",
            "dataset.transform_order=[representation,embedding,kappa,select_attributes]",
            rows=_candidate_rows(),
        )
        train, _, _, _, _ = main.create_dataset(cfg)
        batch = next(iter(main.create_loaders(cfg, train, train, train)[0]))
        # Five vertices per sample, so sample g starts at candidate (1, 2, 3) + 5g.
        n = batch.y.shape[0] // 4
        assert batch.candidate_index_2[:, ::4].t().tolist() == [[1 + 5 * g, 2 + 5 * g, 3 + 5 * g] for g in range(n)]
        # The oracle reads simplicial batches too, which have batch_0 instead of batch.
        import baseline

        assert baseline.PhaseOracle(train, "candidate")(batch).shape == batch.y.shape

    def test_oracle_takes_the_median_over_the_phases(self, tmp_path):
        import baseline
        from torch_geometric.data import Batch

        cfg = _cfg(tmp_path, *CANDIDATE, ONE_SKELETON, "dataset.group_by=polytope_id", rows=_candidate_rows())
        _, _, test, _, _ = main.create_dataset(cfg)
        batch = Batch.from_data_list(list(test))
        pred = baseline.PhaseOracle(test, "candidate")(batch).reshape(-1)
        for data in test:
            p = int(data.polytope_id)
            rows = batch.polytope_id[batch.batch[batch.candidate_index_2[0]]] == p
            # (1,2,3) takes 2p and 2p + 1 over the two phases, (2,3,4) takes 5 and 0.
            assert pred[rows].tolist() == [2 * p + 0.5, 0.0, 0.0, 2.5] * 2

    def test_train_and_validate_end_to_end(self, tmp_path):
        import torch.nn as nn
        import torchmetrics

        from src.utils.train_utils import train as _train, validate

        cfg = _cfg(tmp_path, *CANDIDATE, ONE_SKELETON, "model=gcn", "model.head_layers=2",
                   "trainer.batch_size=8", rows=_candidate_rows())
        train, val, test, _, label_to_index = main.create_dataset(cfg)
        train_loader, val_loader, _, _ = main.create_loaders(cfg, train, val, test)
        model = main.create_model(cfg, train[0].x.shape[1], label_to_index)
        assert model.task_level == "candidate"
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        mae = [("mae", torchmetrics.MeanAbsoluteError())]
        metrics = _train(model, optimizer, train_loader, nn.MSELoss(), mae, cfg)
        assert torch.isfinite(torch.tensor(metrics["train/loss"]))
        metrics = validate(model, val_loader, nn.MSELoss(), mae, cfg)
        assert torch.isfinite(torch.tensor(metrics["val/loss"]))
