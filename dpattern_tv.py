"""d-pattern shift between the MANTRA splits (Yehudai et al., ICML 2021).

For every split S (train, val, test and each OOD cell of an ``ood``
config file) and every depth d <= --depth this computes the pooled node
distribution P_d^S over d-patterns -- the colour-refinement (1-WL)
colour after d rounds, i.e. what a d-layer message-passing network can
distinguish (Thm 4.2 of the paper; Thm 5.1: on patterns absent from
training a d+3-layer network can be perfect on train and arbitrary) --
and reports against the training split

    tv        1/2 sum_pi |P_d^train(pi) - P_d^S(pi)|
    tv_cw     the same, with train re-weighted to the class mix of S
    unseen    P_d^S mass on patterns that never occur in train
    graphs    fraction of complexes in S with >= 1 unseen pattern

"Pooled" means over all nodes of all complexes of the split, so a
complex counts with its node count. ``tv`` compares raw splits; the OOD
splits are class-capped (``max_ood_size_per_class``) while train is
not, and that re-mix alone moves ``tv``. ``tv_cw`` removes it: the
training histogram is built per class and mixed with the class shares
(in complexes) of S, so ``tv_cw`` of an isomorphic copy of the test
split is ~0.

Representations (initial colour in brackets):
    one_skeleton  vertices and edges                       [constant]
    hasse         every simplex, boundary / coboundary edges [rank]
                  -- vertex to vertex takes two rounds here, so Hasse
                  d = 2 is roughly one layer of higher-order message
                  passing on the simplicial complex.
The dual graph is omitted: it is 3-regular on a closed surface, so
featureless refinement never splits it.

Structural shift only: the benchmark's node features (moment curve by
vertex index) follow the labels and are left out by design. The
training distribution is that of the plain training split; runs with
``dataset.walk_length > 0`` train on its walk expansion, which is not
measured here.

Run with the project venv (exact MantraDataset splits; a separate
cache root so the training caches are untouched)::

    .venv/bin/python dpattern_tv.py \\
        --local-path data/mantra/2D/raw/2_manifolds.json \\
        --ood-config config/ood/size_vs_pattern.yaml \\
        --max-vertices 10 --depth 3
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Representations: (nodes, neigh, init)
#   nodes : list of node ids
#   neigh : dict node -> list of (edge_type, neighbour)
#   init  : dict node -> initial colour (hashable)
# --------------------------------------------------------------------------


def one_skeleton(tri):
    adj = defaultdict(set)
    for s in tri:
        for u, v in combinations(s, 2):
            adj[u].add(v)
            adj[v].add(u)
    nodes = sorted(adj)
    return nodes, {v: [("n", u) for u in adj[v]] for v in nodes}, dict.fromkeys(nodes, 0)


def hasse(tri):
    simplices = set()
    for s in tri:
        for k in range(1, len(s) + 1):
            simplices.update(frozenset(f) for f in combinations(s, k))
    nodes = sorted(simplices, key=lambda f: (len(f), sorted(f)))
    neigh = {f: [] for f in nodes}
    for f in nodes:
        for v in f if len(f) > 1 else ():
            g = f - {v}
            neigh[f].append(("b", g))
            neigh[g].append(("c", f))
    return nodes, neigh, {f: len(f) - 1 for f in nodes}


REPRESENTATIONS = {"one_skeleton": one_skeleton, "hasse": hasse}


# --------------------------------------------------------------------------
# Colour refinement. One interner per representation, shared by all
# splits, so colours are comparable across splits.
# --------------------------------------------------------------------------


class Interner(dict):
    def id(self, key):
        return self.setdefault(key, len(self))


def refine(nodes, neigh, init, depth, interner):
    """Colours after 0..depth rounds, each a dict node -> int."""
    col = {v: interner.id(("init", init[v])) for v in nodes}
    out = [col]
    for _ in range(depth):
        new = {}
        for v in nodes:
            groups = defaultdict(list)
            for t, u in neigh[v]:
                groups[t].append(col[u])
            key = (col[v], tuple((t, tuple(sorted(c))) for t, c in sorted(groups.items())))
            new[v] = interner.id(key)
        col = new
        out.append(col)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def tv_distance(p: Counter, q: Counter):
    np_, nq = sum(p.values()), sum(q.values())
    return 0.5 * sum(abs(p[k] / np_ - q[k] / nq) for k in set(p) | set(q))


def class_weighted(per_class: dict[str, Counter], shares: Counter):
    """Mix per-class node histograms with the class shares of another split."""
    total = sum(shares.values())
    mixed = Counter()
    for cls, share in shares.items():
        hist = per_class.get(cls)
        if hist:
            n = sum(hist.values())
            for k, c in hist.items():
                mixed[k] += share / total * c / n
    return mixed


def split_patterns(entries, rep, depth, interner):
    """Per depth: pooled Counter, per-class Counters, per-complex colour sets."""
    build = REPRESENTATIONS[rep]
    pooled = [Counter() for _ in range(depth + 1)]
    per_class = [defaultdict(Counter) for _ in range(depth + 1)]
    per_graph = [[] for _ in range(depth + 1)]
    for cls, tri in entries:
        nodes, neigh, init = build(tri)
        for d, col in enumerate(refine(nodes, neigh, init, depth, interner)):
            pooled[d].update(col.values())
            per_class[d][cls].update(col.values())
            per_graph[d].append(set(col.values()))
    return pooled, per_class, per_graph


def compute(splits: dict[str, list], depth: int, reps):
    """splits: name -> list of (class name, triangulation); train first."""
    rows = []
    for rep in reps:
        interner = Interner()
        stats = {name: split_patterns(entries, rep, depth, interner) for name, entries in splits.items()}
        train_pooled, train_per_class, _ = stats["train"]
        for name, (pooled, _, per_graph) in stats.items():
            shares = Counter(cls for cls, _ in splits[name])
            for d in range(depth + 1):
                unseen = sum(c for k, c in pooled[d].items() if k not in train_pooled[d])
                rows.append(
                    {
                        "representation": rep,
                        "split": name,
                        "depth": d,
                        "n_graphs": len(per_graph[d]),
                        "n_nodes": sum(pooled[d].values()),
                        "n_patterns": len(pooled[d]),
                        "tv": tv_distance(train_pooled[d], pooled[d]),
                        "tv_cw": tv_distance(class_weighted(train_per_class[d], shares), pooled[d]),
                        "unseen": unseen / sum(pooled[d].values()),
                        "graphs_unseen": sum(
                            1 for s in per_graph[d] if any(k not in train_pooled[d] for k in s)
                        )
                        / len(per_graph[d]),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_ood_specs(path):
    """The OOD config group as MantraDataset kwargs, as main.py reads it."""
    with open(path) as f:
        specs = yaml.safe_load(f)
    return {
        name: {"division_type": spec["division_type"], **(spec.get("subdivision") or {})}
        for name, spec in specs.items()
    }


def load_splits(args):
    from mantra.datasets import MantraDataset

    common = dict(
        root=args.root,
        dimension=2,
        version="latest",
        balanced=False,
        local_path=args.local_path,
        seed=args.seed,
        split_proportions=[0.6, 0.2, 0.2],
        min_sample_per_class=args.min_sample_per_class,
        stratified=True,
        max_vertices=args.max_vertices,
        max_ood_size_per_class=args.max_ood_size_per_class,
    )

    def entries(ds):
        return [(d.name, list(map(list, d.triangulation))) for d in ds]

    splits = {
        st: entries(MantraDataset(split_type=st, division_type="none", **common))
        for st in ("train", "val", "test")
    }
    for name, kwargs in load_ood_specs(args.ood_config).items():
        splits[f"ood_{name}"] = entries(MantraDataset(split_type="ood", **common, **kwargs))
    return splits


def print_table(rows, depth, metric):
    by = {(r["representation"], r["split"], r["depth"]): r for r in rows}
    splits = list(dict.fromkeys(r["split"] for r in rows))
    for rep in REPRESENTATIONS:
        if (rep, splits[0], 0) not in by:
            continue
        print(f"\n### {rep}: {metric} / unseen / graphs_unseen vs train")
        print("| split | n | " + " | ".join(f"d={d}" for d in range(depth + 1)) + " |")
        print("|" + "---|" * (depth + 3))
        for s in splits:
            cells = [
                f"{by[rep, s, d][metric]:.2f} / {by[rep, s, d]['unseen']:.2f} / {by[rep, s, d]['graphs_unseen']:.2f}"
                for d in range(depth + 1)
            ]
            print(f"| {s} | {by[rep, s, 0]['n_graphs']} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(REPO_ROOT / "data/dpattern_2d_unbalanced"))
    p.add_argument("--local-path", default=None, help="raw MANTRA json to copy into the cache root")
    p.add_argument("--ood-config", default=str(REPO_ROOT / "config/ood/size_vs_pattern.yaml"))
    p.add_argument("--seed", type=int, default=40, help="dataset.seed (the split seed)")
    p.add_argument("--max-vertices", type=int, default=10)
    p.add_argument("--min-sample-per-class", type=int, default=100)
    p.add_argument("--max-ood-size-per-class", type=int, default=100)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--reps", nargs="+", default=list(REPRESENTATIONS), choices=list(REPRESENTATIONS))
    p.add_argument("--metric", default="tv_cw", choices=["tv", "tv_cw"], help="column of the printed table")
    p.add_argument("--out", default=str(REPO_ROOT / "Claude outputs/dpattern/dpattern_tv.csv"))
    args = p.parse_args()

    splits = load_splits(args)
    for name, entries in splits.items():
        nv = [max(v for s in t for v in s) for _, t in entries]
        print(f"{name:24s} {len(entries):6d} complexes, vertices {min(nv)}..{max(nv)}")

    rows = compute(splits, args.depth, args.reps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print_table(rows, args.depth, args.metric)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
