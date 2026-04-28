"""Pure utility functions for barycentric subdivision of raw triangulations.

No external ML dependencies — safe to import anywhere.
"""

import random
from itertools import combinations, permutations


def barycentric_subdivision_raw(triangulation):
    """Apply barycentric subdivision to a raw triangulation.

    Each sub-simplex of each top-dimensional simplex becomes a new vertex.
    Each permutation of vertices in a top-dimensional simplex produces
    a new top-dimensional simplex in the subdivision.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices (1-indexed vertices).

    Returns
    -------
    new_triangulation : list of list of int
        Subdivided top-dimensional simplices (1-indexed vertices).
    n_vertices : int
        Number of vertices in the subdivided triangulation.
    """
    all_simplices = {}
    next_vertex = 1

    for simplex in triangulation:
        simplex_sorted = sorted(simplex)
        for k in range(1, len(simplex_sorted) + 1):
            for face in combinations(simplex_sorted, k):
                key = frozenset(face)
                if key not in all_simplices:
                    all_simplices[key] = next_vertex
                    next_vertex += 1

    new_triangulation = []
    for simplex in triangulation:
        simplex_sorted = sorted(simplex)
        for perm in permutations(simplex_sorted):
            new_simplex = []
            for k in range(1, len(perm) + 1):
                prefix = frozenset(perm[:k])
                new_simplex.append(all_simplices[prefix])
            new_triangulation.append(new_simplex)

    return new_triangulation, next_vertex - 1


def stellar_subdivision_raw(triangulation, fraction=1.0, rng=None):
    """Stellar subdivision at a (random) fraction of top-dim simplices.

    For each chosen top-dim simplex σ, add a new vertex b_σ (its
    barycenter) and replace σ with the d+1 sub-simplices
    (σ \\ {v}) ∪ {b_σ} for v ∈ σ. Skipped simplices are passed through
    unchanged. ∂σ is unchanged for both chosen and skipped simplices, so
    partial application preserves the simplicial-complex condition
    across (d-1)-faces shared with neighbours. Original vertex labels
    are preserved.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices, 1-indexed vertices, dense labels.
    fraction : float
        Fraction of top-dim simplices to subdivide, in [0, 1]. 1.0
        (default) reproduces full stellar subdivision.
    rng : random.Random or None
        Source of randomness for selecting the subset. Ignored when
        fraction == 1.0. If None for fraction < 1.0, a fresh
        ``random.Random()`` is used.

    Returns
    -------
    new_triangulation : list of list of int
    n_vertices : int
    """
    if not triangulation:
        return [], 0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")

    original = set()
    for simplex in triangulation:
        original.update(simplex)
    next_v = max(original) + 1

    n_total = len(triangulation)
    if fraction == 1.0:
        chosen = None  # subdivide all; no RNG needed
    else:
        n_subdivide = round(fraction * n_total)
        if rng is None:
            rng = random.Random()
        chosen = set(rng.sample(range(n_total), n_subdivide))

    new_triangulation = []
    for idx, simplex in enumerate(triangulation):
        if chosen is not None and idx not in chosen:
            new_triangulation.append(list(simplex))
            continue
        b = next_v
        next_v += 1
        for i in range(len(simplex)):
            new_triangulation.append(
                sorted(list(simplex[:i]) + list(simplex[i + 1:]) + [b])
            )
    return new_triangulation, next_v - 1


def subdivide_dataset_json(data, n_subdivisions=1):
    """Apply n rounds of barycentric subdivision to all triangulations.

    Parameters
    ----------
    data : list of dict
        Raw dataset entries with 'triangulation' and 'n_vertices' keys.
    n_subdivisions : int
        Number of subdivision rounds.

    Returns
    -------
    list of dict
        Dataset entries with subdivided triangulations.
    """
    result = []
    for entry in data:
        tri = entry["triangulation"]
        n_v = entry["n_vertices"]
        for _ in range(n_subdivisions):
            tri, n_v = barycentric_subdivision_raw(tri)
        new_entry = dict(entry)
        new_entry["triangulation"] = tri
        new_entry["n_vertices"] = n_v
        result.append(new_entry)
    return result
