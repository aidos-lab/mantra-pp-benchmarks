"""Colour refinement and the pattern-shift measures of dpattern_tv.py."""

from collections import Counter

from dpattern_tv import Interner, class_weighted, compute, hasse, one_skeleton, refine, tv_distance

# Boundary of a tetrahedron and the octahedron: vertex-transitive, so
# refinement never splits them; the stellar subdivision of one face of
# the tetrahedron has a degree-3 vertex next to degree-4 ones.
TETRA = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
OCTA = [[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 2, 5], [2, 3, 6], [3, 4, 6], [4, 5, 6], [2, 5, 6]]
TETRA_STELLAR = [[1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 5], [2, 3, 5], [1, 3, 5]]


def colours(rep, tri, depth):
    nodes, neigh, init = rep(tri)
    return refine(nodes, neigh, init, depth, Interner())


def test_vertex_transitive_triangulations_stay_constant():
    for rep in (one_skeleton, hasse):
        for tri in (TETRA, OCTA):
            for col in colours(rep, tri, 3):
                nodes, _, init = rep(tri)
                # One colour per initial colour (rank) at every depth.
                assert len(set(col.values())) == len(set(init.values()))


def test_refinement_separates_degrees_after_one_round():
    # Vertices 1, 2, 3 have degree 4, the apexes 4 and 5 degree 3.
    col1 = colours(one_skeleton, TETRA_STELLAR, 1)[1]
    assert col1[1] == col1[2] == col1[3] != col1[4] == col1[5]
    # On the Hasse diagram the degree is the number of coboundary edges,
    # so one round shows it there as well.
    col1 = colours(hasse, TETRA_STELLAR, 1)[1]
    assert col1[frozenset({1})] != col1[frozenset({4})]


def test_tv_and_class_weighting():
    p, q = Counter(a=2, b=2), Counter(a=4)
    assert tv_distance(p, p) == 0
    assert tv_distance(p, q) == 0.5
    per_class = {"x": Counter(a=1), "y": Counter(b=1)}
    assert class_weighted(per_class, Counter(x=3, y=1)) == Counter(a=0.75, b=0.25)


def test_compute_flags_unseen_patterns_and_removes_class_mix():
    # Train: many tetrahedra of class S, one octahedron of class T. An
    # OOD split of octahedra only is a pure class-mix shift: tv > 0,
    # tv_cw = 0. The stellar split has 2-patterns train never saw
    # (a degree-4 vertex with degree-3 neighbours); its 1-patterns
    # (degrees 3 and 4) all occur in train.
    splits = {
        "train": [("S", TETRA)] * 5 + [("T", OCTA)],
        "ood_mix": [("T", OCTA)] * 3,
        "ood_new": [("S", TETRA_STELLAR)],
    }
    rows = {(r["split"], r["depth"]): r for r in compute(splits, 2, ["one_skeleton"])}
    assert rows["train", 2]["tv"] == rows["train", 2]["unseen"] == 0
    assert rows["ood_mix", 2]["tv"] > 0 and rows["ood_mix", 2]["tv_cw"] == 0
    assert rows["ood_mix", 2]["unseen"] == 0
    assert rows["ood_new", 1]["unseen"] == 0
    assert rows["ood_new", 2]["unseen"] == 1 and rows["ood_new", 2]["graphs_unseen"] == 1
