"""Pure utility functions for barycentric subdivision of raw triangulations.

No external ML dependencies — safe to import anywhere.
"""

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


def stellar_subdivision_raw(triangulation):
    """Stellar subdivision at every top-dim simplex.

    For each top-dim simplex σ, add a new vertex b_σ (its barycenter) and
    replace σ with the d+1 sub-simplices (σ \\ {v}) ∪ {b_σ} for v ∈ σ.
    The boundary ∂σ is unchanged, so the simplicial-complex condition is
    preserved across (d-1)-faces shared with neighbouring top-dim
    simplices. Original vertex labels are preserved.

    Parameters
    ----------
    triangulation : list of list of int
        Top-dimensional simplices, 1-indexed vertices, dense labels.

    Returns
    -------
    new_triangulation : list of list of int
    n_vertices : int
    """
    if not triangulation:
        return [], 0

    original = set()
    for simplex in triangulation:
        original.update(simplex)
    next_v = max(original) + 1

    new_triangulation = []
    for simplex in triangulation:
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
