"""Tests for euler_baseline.py.

Runs without the real MANTRA dataset by feeding synthetic ``Data``-like
objects into ``compute_baseline``. Also exercises the Euler-characteristic
computation against known closed manifolds (S^2, RP^2, T^2, Klein bottle,
S^3) and checks the reporting / CLI layer.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Import the script as a module (it's at the repo root, not under `mantra/`).
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import euler_baseline as eb  # noqa: E402


# ---------------------------------------------------------------------------
# Known triangulations
# ---------------------------------------------------------------------------
# Boundary of the 3-simplex ~ S^2 :  V=4, E=6, F=4  -> chi=2
S2_TRI = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]

# Minimum 6-vertex RP^2 (antipodal icosahedron quotient)
# V=6, E=15, F=10  -> chi = 1
RP2_TRI = [
    [1, 2, 3], [1, 2, 4], [1, 3, 5], [1, 4, 6], [1, 5, 6],
    [2, 3, 6], [2, 4, 5], [2, 5, 6], [3, 4, 5], [3, 4, 6],
]

# Minimum 7-vertex torus (Möbius). V=7, E=21, F=14  -> chi = 0
T2_TRI = [
    [1, 2, 3], [1, 3, 5], [1, 2, 6], [1, 5, 6], [1, 4, 7],
    [1, 3, 4], [1, 2, 7], [2, 3, 7], [2, 4, 6], [2, 4, 5],
    [2, 5, 7], [3, 4, 5], [3, 6, 7], [4, 6, 7],
]

# For chi-collision tests we just reuse the torus triangulation under a
# different label — what we want to exercise is the baseline's handling
# of two distinct classes mapping to the same chi, not a faithful Klein
# bottle triangulation.
CHI0_COLLIDER_TRI = T2_TRI

# Boundary of 4-simplex ~ S^3 :  5 tets, V=5, E=10, F=10, T=5 -> chi=0
S3_TRI = [
    [1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5],
    [1, 3, 4, 5], [2, 3, 4, 5],
]


# ---------------------------------------------------------------------------
# Euler characteristic
# ---------------------------------------------------------------------------

class TestEulerCharacteristic:

    def test_s2_from_tetrahedron_boundary(self):
        assert eb.euler_characteristic(S2_TRI, dimension=2) == 2

    def test_rp2_min_triangulation(self):
        assert eb.euler_characteristic(RP2_TRI, dimension=2) == 1

    def test_torus_moebius_min(self):
        assert eb.euler_characteristic(T2_TRI, dimension=2) == 0

    def test_s3_boundary_of_4_simplex(self):
        assert eb.euler_characteristic(S3_TRI, dimension=3) == 0

    def test_permutation_of_vertex_order_does_not_matter(self):
        import random
        rng = random.Random(0)
        scrambled = []
        for s in S2_TRI:
            s = list(s)
            rng.shuffle(s)
            scrambled.append(s)
        assert eb.euler_characteristic(scrambled, dimension=2) == 2

    def test_disjoint_union_of_two_spheres(self):
        # relabel the second copy to use disjoint vertices
        copy = [[v + 10 for v in t] for t in S2_TRI]
        assert eb.euler_characteristic(S2_TRI + copy, dimension=2) == 4

    def test_string_and_float_vertices_coerced(self):
        weird = [["1", "2", "3"], [1.0, 2.0, 4.0], [1, 3, 4], [2, 3, 4]]
        assert eb.euler_characteristic(weird, dimension=2) == 2

    def test_rejects_unsupported_dimension(self):
        with pytest.raises(ValueError):
            eb.euler_characteristic([[1, 2]], dimension=1)
        with pytest.raises(ValueError):
            eb.euler_characteristic([[1, 2, 3, 4, 5]], dimension=4)


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _mk(name, tri):
    """Build a minimal PyG-like Data stand-in."""
    ns = SimpleNamespace(name=name, triangulation=tri)
    return ns


def _build_dataset(spec):
    """spec: list of (name, triangulation, count)."""
    ds = []
    for name, tri, count in spec:
        for _ in range(count):
            ds.append(_mk(name, tri))
    return ds


# ---------------------------------------------------------------------------
# compute_baseline
# ---------------------------------------------------------------------------

class TestComputeBaseline:

    def test_singletons_all_distinct_chi(self):
        ds = _build_dataset([
            ("S^2",  S2_TRI,  3),
            ("RP^2", RP2_TRI, 3),
            ("T^2",  T2_TRI,  3),
        ])
        out = eb.compute_baseline(ds, dimension=2)
        assert out["class_to_chi"] == {"S^2": 2, "RP^2": 1, "T^2": 0}
        assert set(out["chi_to_classes"].keys()) == {0, 1, 2}
        assert out["multi_chi_classes"] == {}
        assert out["excluded_classes"] == {}
        # all singleton chis -> BAcc = 3/3 = 1
        bacc = len(out["chi_to_classes"]) / len(out["class_to_chi"])
        assert bacc == 1.0

    def test_chi_collision_yields_bacc_below_one(self):
        # T^2 and Klein bottle both have chi=0; S^2 chi=2. 2 distinct chi, 3
        # classes -> BAcc = 2/3.
        ds = _build_dataset([
            ("S^2",          S2_TRI,    4),
            ("T^2",          T2_TRI,    4),
            ("Klein bottle", CHI0_COLLIDER_TRI, 4),
        ])
        out = eb.compute_baseline(ds, dimension=2)
        assert out["chi_to_classes"][0] == ["Klein bottle", "T^2"]
        assert out["chi_to_classes"][2] == ["S^2"]
        bacc = len(out["chi_to_classes"]) / len(out["class_to_chi"])
        assert bacc == pytest.approx(2 / 3)

    def test_min_count_filters_rare_classes_and_their_groups(self):
        # T^2 has 50 samples, Klein bottle has 4 samples, S^2 has 50.
        # With min_count=10 Klein is dropped -> chi=0 group shrinks to {T^2}.
        # Remaining 2 classes both have unique chi -> BAcc = 2/2 = 1.
        ds = _build_dataset([
            ("S^2",          S2_TRI,    50),
            ("T^2",          T2_TRI,    50),
            ("Klein bottle", CHI0_COLLIDER_TRI, 4),
        ])
        out = eb.compute_baseline(ds, dimension=2, min_count=10)
        assert set(out["class_to_chi"].keys()) == {"S^2", "T^2"}
        assert out["excluded_classes"] == {"Klein bottle": 4}
        assert len(out["chi_to_classes"]) == len(out["class_to_chi"])

    def test_min_count_zero_keeps_everything(self):
        ds = _build_dataset([
            ("S^2", S2_TRI, 1),
            ("T^2", T2_TRI, 1),
        ])
        out = eb.compute_baseline(ds, dimension=2, min_count=0)
        assert out["excluded_classes"] == {}
        assert len(out["class_to_chi"]) == 2

    def test_empty_dataset_is_degenerate_but_safe(self):
        out = eb.compute_baseline([], dimension=2)
        assert out["class_to_chi"] == {}
        assert out["chi_to_classes"] == {}
        assert out["multi_chi_classes"] == {}
        assert out["excluded_classes"] == {}

    def test_multi_chi_warning_is_populated_on_label_inconsistency(self):
        # Same label "bogus" used for an S^2 and a T^2 -> multi_chi flag.
        ds = [_mk("bogus", S2_TRI), _mk("bogus", T2_TRI)]
        out = eb.compute_baseline(ds, dimension=2)
        assert "bogus" in out["multi_chi_classes"]
        assert out["multi_chi_classes"]["bogus"] == {0, 2}

    def test_class_count_reflects_filter(self):
        ds = _build_dataset([
            ("S^2", S2_TRI, 7),
            ("T^2", T2_TRI, 2),
        ])
        out = eb.compute_baseline(ds, dimension=2, min_count=5)
        assert out["class_count"] == {"S^2": 7}
        assert "T^2" in out["excluded_classes"]


# ---------------------------------------------------------------------------
# report() — basic smoke + BAcc value check via stdout
# ---------------------------------------------------------------------------

class TestReport:

    def test_bacc_and_breakdown_appear_in_output(self, capsys):
        ds = _build_dataset([
            ("S^2",          S2_TRI,    4),
            ("T^2",          T2_TRI,    4),
            ("Klein bottle", CHI0_COLLIDER_TRI, 4),
        ])
        out = eb.compute_baseline(ds, dimension=2)
        eb.report(
            out, dimension=2, balanced=True,
            version="test", label_field="name",
        )
        captured = capsys.readouterr().out
        assert "Balanced accuracy (Euler baseline) = 0.6667" in captured
        assert "2 / 3" in captured
        assert "T^2" in captured and "Klein bottle" in captured
        assert "S^2" in captured

    def test_multi_chi_warning_printed(self, capsys):
        ds = [_mk("bogus", S2_TRI), _mk("bogus", T2_TRI)]
        out = eb.compute_baseline(ds, dimension=2)
        eb.report(
            out, dimension=2, balanced=False,
            version="test", label_field="name",
        )
        captured = capsys.readouterr().out
        assert "WARNING" in captured
        assert "'bogus'" in captured

    def test_excluded_classes_listed_when_filtered(self, capsys):
        ds = _build_dataset([
            ("S^2", S2_TRI, 50),
            ("rare", RP2_TRI, 2),
        ])
        out = eb.compute_baseline(ds, dimension=2, min_count=10)
        eb.report(
            out, dimension=2, balanced=False,
            version="test", label_field="name",
        )
        captured = capsys.readouterr().out
        assert "excluded classes" in captured
        assert "rare" in captured
        assert "n=2" in captured


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestCLI:

    def test_help_runs(self):
        result = subprocess.run(
            [sys.executable, str(_REPO / "euler_baseline.py"), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "--dimension" in result.stdout
        assert "--min-count" in result.stdout
        assert "--local-path" in result.stdout
