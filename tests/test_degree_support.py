"""Size and degree support of the splits (src/metrics/degree_support.py)."""

import math

import pytest
from torch_geometric.data import Data

from src.metrics.degree_support import (
    degree_counts,
    degree_support_metrics,
    frequent_support,
    size_counts,
    total_variation,
    unseen_fraction,
    vertex_degrees,
)

TETRAHEDRON = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
# The 6-vertex projective plane: every vertex has degree 5.
RP2 = [
    [1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 4, 6], [1, 5, 6],
    [2, 3, 6], [2, 4, 5], [2, 5, 6], [3, 4, 5], [3, 4, 6],
]


class FakeDataset:
    """A dataset with `get` (raw entries) like a PyG in-memory dataset."""

    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def get(self, i):
        return self.entries[i]

    def __getitem__(self, i):  # transformed access, must not be used
        raise AssertionError("degree support must read raw entries")


def entry(triangulation):
    n = len({v for s in triangulation for v in s})
    return Data(triangulation=triangulation, n_vertices=n)


def test_vertex_degrees_of_the_tetrahedron_and_rp2():
    assert vertex_degrees(TETRAHEDRON) == [3, 3, 3, 3]
    assert vertex_degrees(RP2) == [5] * 6


def test_degree_sum_is_twice_the_edge_count():
    # Closed surface: 3 f2 = 2 f1, so sum deg = 2 f1 = 3 f2.
    for tri in (TETRAHEDRON, RP2):
        assert sum(vertex_degrees(tri)) == 3 * len(tri)


def test_counts_read_raw_entries():
    ds = FakeDataset([entry(TETRAHEDRON), entry(RP2)])
    assert degree_counts(ds) == {3: 4, 5: 6}
    assert size_counts(ds) == {4: 1, 6: 1}


def test_unseen_fraction():
    assert unseen_fraction({3: 4, 5: 6}, {3}) == pytest.approx(0.6)
    assert unseen_fraction({3: 4}, {3, 5}) == 0.0
    assert math.isnan(unseen_fraction({}, {3}))


def test_frequent_support_and_total_variation():
    counts = {3: 990, 5: 9, 7: 1}
    assert frequent_support(counts, 1e-2) == {3}
    assert frequent_support(counts, 5e-3) == {3, 5}
    assert frequent_support({}, 1e-3) == set()
    assert total_variation({3: 1}, {3: 1}) == 0.0
    assert total_variation({3: 1}, {5: 1}) == 1.0
    assert total_variation({3: 1, 5: 1}, {3: 1}) == pytest.approx(0.5)
    assert math.isnan(total_variation({}, {3: 1}))


def test_metrics_relative_to_training():
    train = FakeDataset([entry(TETRAHEDRON)] * 3)
    ood = FakeDataset([entry(RP2), entry(TETRAHEDRON)])
    metrics = degree_support_metrics(train, {"ood_x": ood})
    assert metrics["train/degree_min"] == 3
    assert metrics["train/degree_max"] == 3
    assert metrics["train/n_vertices_min"] == 4
    assert metrics["train/n_vertices_max"] == 4
    assert metrics["train/n_vertices_mean"] == 4.0
    assert metrics["train/degree_max_frequent"] == 3
    assert metrics["ood_x/unseen_degree_frac"] == pytest.approx(0.6)
    assert metrics["ood_x/rare_degree_frac"] == pytest.approx(0.6)
    assert metrics["ood_x/degree_tv"] == pytest.approx(0.6)
    assert metrics["ood_x/unseen_size_frac"] == pytest.approx(0.5)
    assert metrics["ood_x/degree_max"] == 5
    assert metrics["ood_x/n_vertices_mean"] == 5.0


def test_materialized_list_is_accepted():
    train = [entry(TETRAHEDRON)]
    metrics = degree_support_metrics(train, {"val": [entry(RP2)]})
    assert metrics["val/unseen_degree_frac"] == 1.0


def test_rare_degrees_count_as_unseen_only_in_the_thresholded_metric():
    # 999 tetrahedra and one RP2: degree 5 has a share of 6/4002 < 1e-2.
    train = FakeDataset([entry(TETRAHEDRON)] * 999 + [entry(RP2)])
    metrics = degree_support_metrics(train, {"v": FakeDataset([entry(RP2)])}, min_frequency=1e-2)
    assert metrics["v/unseen_degree_frac"] == 0.0
    assert metrics["v/rare_degree_frac"] == 1.0
    assert metrics["train/degree_max"] == 5
    assert metrics["train/degree_max_frequent"] == 3
