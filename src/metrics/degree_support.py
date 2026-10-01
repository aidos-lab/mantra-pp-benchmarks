"""Size and vertex-degree support of the splits relative to training.

On a closed surface chi = sum_v (1 - deg(v) / 6), so a sum readout over
a per-vertex function of the degree can express chi exactly, but only
on the degrees it has seen: a model trained on one size has no
constraint on the degrees that appear only at larger sizes. These
metrics record, per split, the fraction of vertices whose degree lies
outside the training split's degree support — the support as seen at
all, and the support of degrees frequent enough to have been learned —
the total variation between the degree distributions, and the vertex
counts, so a cell's accuracy can be read against how far it leaves
that support. The degree is a 1-hop summary; the d-pattern table of
``dpattern_tv.py`` is the deeper one. Computed from the raw
triangulations before any transform.
"""

from collections import Counter
from typing import Dict, Iterable, List

import numpy as np


def vertex_degrees(triangulation: Iterable[Iterable[int]]) -> List[int]:
    """Degrees of the vertices of a triangulation given as top simplices.

    The degree of a vertex is the number of edges at it, i.e. the
    number of distinct neighbours, in any dimension.
    """
    neighbours: Dict[int, set] = {}
    for simplex in triangulation:
        vertices = [int(v) for v in simplex]
        for v in vertices:
            others = neighbours.setdefault(v, set())
            others.update(w for w in vertices if w != v)
    return [len(others) for others in neighbours.values()]


def raw_entries(dataset):
    """The untransformed entries of a PyG in-memory dataset."""
    get = getattr(dataset, "get", None)
    if get is None:  # a plain sequence (e.g. a materialized list)
        return iter(dataset)
    return (get(i) for i in range(len(dataset)))


def degree_counts(dataset) -> Counter:
    """Histogram of vertex degrees over all entries of ``dataset``."""
    counts: Counter = Counter()
    for data in raw_entries(dataset):
        counts.update(vertex_degrees(data.triangulation))
    return counts


def size_counts(dataset) -> Counter:
    """Histogram of vertex counts over the entries of ``dataset``."""
    return Counter(int(data.n_vertices) for data in raw_entries(dataset))


def unseen_fraction(counts: Counter, support: Iterable[int]) -> float:
    """Share of the mass of ``counts`` outside ``support``."""
    total = sum(counts.values())
    if total == 0:
        return float("nan")
    support = set(support)
    return sum(c for d, c in counts.items() if d not in support) / total


def frequent_support(counts: Counter, min_frequency: float) -> set:
    """Values whose share of the mass of ``counts`` is >= ``min_frequency``.

    A degree that occurs a handful of times in ~10^6 training vertices
    is formally "seen" but carries no gradient; the thresholded support
    is the one a model can be expected to have learned.
    """
    total = sum(counts.values())
    if total == 0:
        return set()
    return {d for d, c in counts.items() if c / total >= min_frequency}


def total_variation(counts: Counter, reference: Counter) -> float:
    """Total variation between the distributions of two histograms."""
    n, m = sum(counts.values()), sum(reference.values())
    if n == 0 or m == 0:
        return float("nan")
    keys = set(counts) | set(reference)
    return 0.5 * sum(
        abs(counts.get(k, 0) / n - reference.get(k, 0) / m) for k in keys
    )


RARE_DEGREE_FREQUENCY = 1e-3


def degree_support_metrics(
    train_dataset, others: Dict[str, object], min_frequency: float = RARE_DEGREE_FREQUENCY
) -> Dict[str, float]:
    """Flat metric dictionary of the size and degree support.

    ``train/degree_min``, ``train/degree_max``, ``train/degree_max_frequent``
    (largest degree with a share >= ``min_frequency`` of the training
    vertices), ``train/n_vertices_min``, ``train/n_vertices_max`` and
    ``train/n_vertices_mean`` describe the training split; every other
    split ``p`` (``val``, ``test`` and the ``ood_*`` cells) gets
    ``p/unseen_degree_frac`` (fraction of its vertices whose degree
    never occurs in training), ``p/rare_degree_frac`` (fraction whose
    degree has a share < ``min_frequency`` in training), ``p/degree_tv``
    (total variation between its degree distribution and the training
    one), ``p/unseen_size_frac`` (fraction of its entries whose vertex
    count never occurs in training), ``p/degree_max`` and
    ``p/n_vertices_mean``.
    """
    train_degrees = degree_counts(train_dataset)
    train_sizes = size_counts(train_dataset)
    frequent = frequent_support(train_degrees, min_frequency)
    metrics = {
        "train/degree_min": min(train_degrees),
        "train/degree_max": max(train_degrees),
        "train/degree_max_frequent": max(frequent),
        "train/n_vertices_min": min(train_sizes),
        "train/n_vertices_max": max(train_sizes),
        "train/n_vertices_mean": float(
            np.average(list(train_sizes), weights=list(train_sizes.values()))
        ),
    }
    for prefix, dataset in others.items():
        degrees = degree_counts(dataset)
        sizes = size_counts(dataset)
        metrics[f"{prefix}/unseen_degree_frac"] = unseen_fraction(degrees, train_degrees)
        metrics[f"{prefix}/rare_degree_frac"] = unseen_fraction(degrees, frequent)
        metrics[f"{prefix}/degree_tv"] = total_variation(degrees, train_degrees)
        metrics[f"{prefix}/unseen_size_frac"] = unseen_fraction(sizes, train_sizes)
        metrics[f"{prefix}/degree_max"] = max(degrees) if degrees else float("nan")
        metrics[f"{prefix}/n_vertices_mean"] = (
            float(np.average(list(sizes), weights=list(sizes.values())))
            if sizes
            else float("nan")
        )
    return metrics
