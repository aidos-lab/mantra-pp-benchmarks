"""Tests for the Kähler-cone tasks: the graph and vertex targets, the
phase oracle at the graph level, the Kähler class as a fifth coordinate
and the membership label recomputed from a collated batch."""

import inspect
import json
import os
from itertools import combinations

import numpy as np
import pytest
import torch
from mantra.representations import HasseDiagram
from scipy.spatial import ConvexHull

import baseline
import main
from tests.test_cy_dataset import ONE_SKELETON, _cfg, _phase_rows

COORDS = (
    ONE_SKELETON,
    "~transform@dataset.transform.embedding",
    "dataset.transform.select_features.src=coords",
)
LIFT = (
    "+transform@dataset.transform.lift=lift_coords",
    "~transform@dataset.transform.kappa",
)
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "kclass_rows.json")


def _kahler_rows():
    """`_phase_rows` with the label columns: the volume and the divisor
    volume of the origin differ between the two phases of a polytope,
    the completeness flag is shared by them."""
    rows = _phase_rows()
    for i, row in enumerate(rows):
        row["skc_log_vol"] = 0.5 * i
        row["skc_log_tau"] = [float(i), 1.0, 2.0, 3.0, 4.0]
        row["toric_cone_complete"] = (i // 2) % 2
        row["kclass_height"] = [0.1 * i, 1.0, -1.0, 0.5, -0.5]
        row["kclass_member"] = i % 2
    return rows


def _datasets(tmp_path, *overrides, rows=None):
    cfg = _cfg(
        tmp_path, *overrides, "dataset.group_by=polytope_id",
        rows=rows or _kahler_rows(),
    )
    return cfg, main.create_dataset(cfg)


class TestKahlerTargets:
    def test_graph_target(self, tmp_path):
        _, (train, _, _, _, _) = _datasets(tmp_path, "task=skc_vol", ONE_SKELETON)
        for data in train:
            assert data.y.reshape(-1).tolist() == [float(data.skc_log_vol)]

    def test_vertex_target(self, tmp_path):
        _, (train, _, _, _, _) = _datasets(tmp_path, "task=skc_tau", ONE_SKELETON)
        for data in train:
            assert data.y.reshape(-1).tolist() == data.skc_log_tau.tolist()
            assert data.y.shape[0] == data.x.shape[0]

    def test_oracle_takes_the_median_of_a_graph_target(self, tmp_path):
        cfg, (_, _, test, _, _) = _datasets(tmp_path, "task=skc_vol", ONE_SKELETON)
        loader = main.create_loaders(cfg, test, test, test)[2]
        oracle = baseline.PhaseOracle(test, "graph")
        for batch in loader:
            # Rows 2p and 2p + 1 hold 0.5 * row: the median is p + 0.25.
            expected = batch.polytope_id.float() + 0.25
            assert oracle(batch).reshape(-1).tolist() == expected.tolist()

    def test_oracle_takes_the_majority_of_a_class(self, tmp_path):
        cfg, (_, _, test, _, label_to_index) = _datasets(
            tmp_path, "task=toric_complete", ONE_SKELETON
        )
        assert label_to_index == {0: 0, 1: 1}
        loader = main.create_loaders(cfg, test, test, test)[2]
        oracle = baseline.PhaseOracle(test, "graph", out_channels=2)
        for batch in loader:
            assert oracle(batch).argmax(1).tolist() == batch.y.reshape(-1).tolist()


class TestLiftedCoordinates:
    def test_graph_chain_reads_the_height_as_last_coordinate(self, tmp_path):
        _, (train, _, _, _, _) = _datasets(tmp_path, "task=kclass_member", *COORDS, *LIFT)
        for data in train:
            assert data.x.shape == (5, 4)
            assert data.x[:, 3].tolist() == pytest.approx(
                [0.1 * int(data.h11 * 2 + data.y), 1.0, -1.0, 0.5, -0.5]
            )

    def test_simplicial_chain_propagates_the_height(self, tmp_path):
        _, (train, _, _, _, _) = _datasets(
            tmp_path, "task=kclass_member", *LIFT,
            "representation@dataset.transform.representation=incidence_sc",
            "dataset.transform.representation.signed=True",
            "transform@dataset.transform.embedding=propagate_convex_comb",
            "~transform@dataset.transform.select_features",
            "transform@dataset.transform.select_attributes=select_attributes_sc",
            "dataset.transform_order=[lift,representation,embedding,select_attributes]",
        )
        for data in train:
            assert data.x_0.shape == (5, 4)
            # Edge (1, 2) comes after the four edges through the origin.
            assert data.x_1[4, 3].item() == pytest.approx(0.0)

    def test_coo_cache_keeps_the_propagated_height(self, tmp_path):
        from tests.test_cy_dataset import COO, SCCNN_CHAIN

        _, (train, _, _, _, _) = _datasets(
            tmp_path, "task=kclass_member", *LIFT, *SCCNN_CHAIN, *COO,
            "dataset.transform_order="
            "[lift,representation,up,down,hodge,embedding,select_attributes,coo]",
        )
        for data in train:
            assert data.x_0.shape == (5, 4) and data.x_2.shape[1] == 4
            assert data.x_1[4, 3].item() == pytest.approx(0.0)
            assert "x_3" not in data and "incidence_2_idx" in data

    def test_tasks_without_a_class_keep_the_coordinates(self, tmp_path):
        _, (train, _, _, _, _) = _datasets(tmp_path, "task=skc_vol", *COORDS)
        assert train[0].x.shape == (5, 3)


def _member_from_inputs(lifted, edges):
    """Whether the lower hull of the lifted points has the edges of the
    phase on every 2-face of the polytope, from the coordinates in `x`
    (the origin is point 0) and the edges alone."""
    points = lifted[1:, :4]
    facets = np.unique(np.round(ConvexHull(points).equations, 9), axis=0)
    on = np.abs(points @ facets[:, :4].T + facets[:, 4]) < 1e-7

    def on_a_2_face(pairs):
        return {
            (i, j) for i, j in pairs
            if i > 0 and (on[i - 1] & on[j - 1]).sum() >= 2
        }

    hull = ConvexHull(lifted)
    lower = hull.simplices[hull.equations[:, 4] < -1e-9]
    induced = {p for s in lower for p in combinations(sorted(s.tolist()), 2)}
    phase = {tuple(sorted(e)) for e in edges.T.tolist()}
    return on_a_2_face(induced) == on_a_2_face(phase)


class TestMembershipFromBatch:
    def test_label_is_a_function_of_the_collated_inputs(self, tmp_path):
        """The label written by kahler_labels.py agrees with the regular
        triangulation that the fifth coordinate induces, read off the
        collated `x` and `edge_index` of the coordinate chain."""
        with open(FIXTURE) as f:
            fixture = json.load(f)
        rows = [{**row, "polytope_id": copy} for copy in range(10) for row in fixture]
        cfg, (train, val, test, _, _) = _datasets(
            tmp_path, "task=kclass_member", *COORDS, *LIFT, rows=rows
        )
        assert len(train) + len(val) + len(test) == len(rows)
        seen = set()
        for batch in main.create_loaders(cfg, train, val, test)[0]:
            for g in range(batch.num_graphs):
                nodes = (batch.batch == g).nonzero().reshape(-1)
                edges = batch.edge_index[:, batch.batch[batch.edge_index[0]] == g]
                member = _member_from_inputs(
                    batch.x[nodes].double().numpy(), (edges - nodes[0]).numpy()
                )
                assert member == bool(batch.y[g])
                seen.add(member)
        assert seen == {True, False}


@pytest.mark.skipif(
    "max_rank" not in inspect.signature(HasseDiagram.__init__).parameters,
    reason="requires mantra's rank-capped Hasse diagram (cy-hasse-ranks)",
)
class TestHasseLadder:
    # The cone over the boundary of a tetrahedron has 5 vertices, 10
    # edges, 10 triangles and 4 tetrahedra.
    @pytest.mark.parametrize("max_rank, nodes", [(1, 15), (2, 25), (3, 29)])
    def test_vertex_targets_on_the_capped_diagram(self, tmp_path, max_rank, nodes):
        cfg, (train, _, _, _, _) = _datasets(
            tmp_path, "task=skc_tau",
            "representation@dataset.transform.representation=hasse_diagram",
            "dataset.transform.representation.feature_propagation=x",
            f"dataset.transform.representation.max_rank={max_rank}",
            "dataset.transform.representation.rank_feature=true",
            "transform@dataset.transform.embedding=propagate_convex_comb",
            "~transform@dataset.transform.select_features",
            "dataset.transform_order=[embedding,representation,kappa,select_attributes]",
        )
        for data in train:
            assert data.x.shape == (nodes, 3 + max_rank + 1)
            assert data.vertex_mask.tolist() == [True] * 5 + [False] * (nodes - 5)
            assert data.x[:5, :3].tolist() == data.coords.tolist()
            assert data.y.shape[0] == 5
        batch = next(iter(main.create_loaders(cfg, train, train, train)[0]))
        assert int(batch.vertex_mask.sum()) == batch.y.shape[0]
