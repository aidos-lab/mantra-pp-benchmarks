"""d-pattern discrepancy between MANTRA splits (Yehudai et al., ICML 2021).

For every split S (train, val, test and each OOD subdivision of an
``ood`` config file) and every depth d <= --depth this computes the
pooled node distribution P_d^S over d-patterns -- the colour-refinement
(1-WL) colour after d rounds, i.e. exactly what a d-layer message-passing
network can distinguish (Thm 4.2 of the paper) -- on three
representations of each triangulation, and reports

    tv             TV(P_d^train, P_d^S) = 1/2 sum_pi |P_train(pi) - P_S(pi)|
    unseen         P_d^S mass on patterns that never occur in train
    graphs_unseen  fraction of graphs in S with >= 1 unseen node pattern
    n_patterns     number of distinct d-patterns in S

Thm 5.1 of the paper: unseen > 0 already admits a network that is
perfect on train and arbitrarily wrong on S; tv is their empirical
predictor of the accuracy drop.

Representations (initial colour in brackets):
    one_skeleton  vertices and edges of the triangulation    [constant]
    dual_graph    top simplices, adjacent iff they share a codim-1 face
                  [constant]  -- every vertex has degree 3 on a closed
                  manifold, so patterns only separate from d = 2 on
    hasse         every simplex, typed boundary / coboundary adjacency
                  [rank]      -- the HOMP / Levi-graph patterns

Node features of the benchmark (moment curve, random features) are not
structural, so featureless refinement is the right comparison.

The OOD config may contain Pachner-walk cells (``division_type:
pachner`` with ``match``/``target``/``move_weights``/``mix``, mantra
branch ``pachner-walks``), e.g. ``config/ood/size_vs_pattern.yaml``,
which puts a size-matched Pachner walk next to every subdivision.

Run on the Mac with the project venv (uses the exact MantraDataset
splits of the 2D-unbalanced runs; the cache root is separate so the
training caches are untouched)::

    .venv/bin/python dpattern_tv.py \
        --local-path data/mantra/2D/raw/2_manifolds.json \
        --ood-config config/ood/paper_unbalanced.yaml \
        --max-vertices 10 --depth 4

Smoke test without torch (raw JSON, random 60/20/20 split, subdivisions
applied here with mantra.utils.triangulation)::

    python3 dpattern_tv.py --raw-json data/mantra/2D/raw/2_manifolds.json \
        --smoke 300 --max-vertices 10 --depth 3
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

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
    neigh = {v: [("n", u) for u in adj[v]] for v in nodes}
    return nodes, neigh, {v: 0 for v in nodes}


def dual_graph(tri):
    cofaces = defaultdict(list)
    for i, s in enumerate(tri):
        for f in combinations(sorted(s), len(s) - 1):
            cofaces[f].append(i)
    nodes = list(range(len(tri)))
    neigh = {
        i: [
            ("n", j)
            for f in combinations(sorted(s), len(s) - 1)
            for j in cofaces[f]
            if j != i
        ]
        for i, s in enumerate(tri)
    }
    return nodes, neigh, {i: 0 for i in nodes}


def hasse(tri):
    simplices = set()
    for s in tri:
        for k in range(1, len(s) + 1):
            simplices.update(frozenset(f) for f in combinations(s, k))
    nodes = sorted(simplices, key=lambda f: (len(f), sorted(f)))
    neigh = {f: [] for f in nodes}
    for f in nodes:
        if len(f) > 1:
            for v in f:
                g = f - {v}
                neigh[f].append(("b", g))
                neigh[g].append(("c", f))
    return nodes, neigh, {f: len(f) - 1 for f in nodes}


REPRESENTATIONS = {
    "one_skeleton": one_skeleton,
    "dual_graph": dual_graph,
    "hasse": hasse,
}


# --------------------------------------------------------------------------
# Colour refinement with a global interner so colours are comparable
# across splits (one interner per representation).
# --------------------------------------------------------------------------


class Interner(dict):
    def id(self, key):
        if key not in self:
            self[key] = len(self)
        return self[key]


def refine(nodes, neigh, init, depth, interner):
    """Return [colours_0, ..., colours_depth], each a dict node -> int."""
    col = {v: interner.id(("init", init[v])) for v in nodes}
    out = [col]
    for _ in range(depth):
        new = {}
        for v in nodes:
            groups = defaultdict(list)
            for t, u in neigh[v]:
                groups[t].append(col[u])
            key = (
                col[v],
                tuple((t, tuple(sorted(cs))) for t, cs in sorted(groups.items())),
            )
            new[v] = interner.id(key)
        col = new
        out.append(col)
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def split_patterns(triangulations, rep, depth, interner):
    """Per depth: pooled node Counter and list of per-graph colour sets."""
    build = REPRESENTATIONS[rep]
    pooled = [Counter() for _ in range(depth + 1)]
    per_graph = [[] for _ in range(depth + 1)]
    for tri in triangulations:
        nodes, neigh, init = build(tri)
        cols = refine(nodes, neigh, init, depth, interner)
        for d, col in enumerate(cols):
            pooled[d].update(col.values())
            per_graph[d].append(set(col.values()))
    return pooled, per_graph


def tv_distance(p: Counter, q: Counter):
    np_, nq = sum(p.values()), sum(q.values())
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p[k] / np_ - q[k] / nq) for k in keys)


def unseen_mass(train: Counter, q: Counter):
    nq = sum(q.values())
    return sum(c for k, c in q.items() if k not in train) / nq


def graphs_with_unseen(train: Counter, per_graph):
    return sum(1 for s in per_graph if any(k not in train for k in s)) / len(per_graph)


def compute(splits: dict[str, list], depth: int, reps):
    """splits: name -> list of triangulations (train first)."""
    rows = []
    for rep in reps:
        interner = Interner()
        stats = {
            name: split_patterns(tris, rep, depth, interner)
            for name, tris in splits.items()
        }
        train_pooled, _ = stats["train"]
        for name, (pooled, per_graph) in stats.items():
            for d in range(depth + 1):
                rows.append(
                    {
                        "representation": rep,
                        "split": name,
                        "depth": d,
                        "n_graphs": len(per_graph[d]),
                        "n_nodes": sum(pooled[d].values()),
                        "n_patterns": len(pooled[d]),
                        "tv": tv_distance(train_pooled[d], pooled[d]),
                        "unseen": unseen_mass(train_pooled[d], pooled[d]),
                        "graphs_unseen": graphs_with_unseen(
                            train_pooled[d], per_graph[d]
                        ),
                    }
                )
    return rows


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------


def load_ood_specs(path):
    import yaml

    with open(path) as f:
        specs = yaml.safe_load(f)
    return {
        name: {"division_type": spec["division_type"], **(spec.get("subdivision") or {})}
        for name, spec in specs.items()
    }


def splits_from_dataset(args):
    from mantra.datasets import MantraDataset

    common = dict(
        root=args.root,
        dimension=args.dimension,
        version="latest",
        balanced=False,
        local_path=args.local_path,
        seed=args.seed,
        split_proportions=[0.6, 0.2, 0.2],
        min_sample_per_class=args.min_sample_per_class,
        stratified=True,
        max_vertices=args.max_vertices,
        max_ood_size_per_class=args.max_ood_size_per_class,
        n_moves=5,
        target_count=100,
        use_surgery=True,
    )
    splits = {}
    for st in ("train", "val", "test"):
        ds = MantraDataset(split_type=st, division_type="none", **common)
        splits[st] = [list(map(list, d.triangulation)) for d in ds]
    for name, kwargs in load_ood_specs(args.ood_config).items():
        ds = MantraDataset(split_type="ood", **common, **kwargs)
        splits[f"ood_{name}"] = [list(map(list, d.triangulation)) for d in ds]
    return splits


def splits_from_raw(args):
    """Torch-free approximation: random stratification-free split and the
    same subdivision code, on a subsample. For smoke tests only."""
    sys.path.insert(0, str(REPO_ROOT.parent / "mantra"))
    from mantra.utils.triangulation import Triangulation

    with open(args.raw_json) as f:
        entries = json.load(f)
    if args.max_vertices is not None:
        entries = [e for e in entries if e["n_vertices"] <= args.max_vertices]
    rng = random.Random(args.seed)
    rng.shuffle(entries)
    if args.smoke:
        entries = entries[: args.smoke]
    n = len(entries)
    tris = [e["triangulation"] for e in entries]
    splits = {
        "train": tris[: int(0.6 * n)],
        "val": tris[int(0.6 * n) : int(0.8 * n)],
        "test": tris[int(0.8 * n) :],
    }
    for name, kw in load_ood_specs(args.ood_config).items():
        out = []
        for tri in splits["test"]:
            t = Triangulation.from_list(tri, rng=rng)
            kind = kw["division_type"]
            if kind == "barycentric":
                for _ in range(kw.get("round", 1)):
                    t.barycentric_subdivision()
            elif kind == "stellar":
                t.stellar_subdivision(fraction=kw.get("fraction", 1.0))
            elif kind == "graded":
                t.graded_subdivision(over_vrtx_cnt=kw["graded_vertex_number"])
            out.append(t.to_list())
        splits[f"ood_{name}"] = out
    return splits


# --------------------------------------------------------------------------


def print_table(rows, depth):
    reps = sorted({r["representation"] for r in rows}, key=list(REPRESENTATIONS).index)
    splits = list(dict.fromkeys(r["split"] for r in rows))
    for rep in reps:
        print(f"\n### {rep}  (tv / unseen / graphs_unseen vs train)")
        head = "| split | n_graphs | " + " | ".join(f"d={d}" for d in range(depth + 1)) + " |"
        print(head)
        print("|" + "---|" * (depth + 3))
        for s in splits:
            cells = []
            for d in range(depth + 1):
                r = next(
                    x for x in rows
                    if x["representation"] == rep and x["split"] == s and x["depth"] == d
                )
                cells.append(f"{r['tv']:.2f} / {r['unseen']:.2f} / {r['graphs_unseen']:.2f}")
            ng = next(x for x in rows if x["representation"] == rep and x["split"] == s)["n_graphs"]
            print(f"| {s} | {ng} | " + " | ".join(cells) + " |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data/dpattern_2d_unbalanced")
    p.add_argument("--local-path", default=None, help="raw MANTRA json to copy into the cache root")
    p.add_argument("--ood-config", default=str(REPO_ROOT / "config/ood/paper_unbalanced.yaml"))
    p.add_argument("--dimension", type=int, default=2)
    p.add_argument("--seed", type=int, default=40)
    p.add_argument("--max-vertices", type=int, default=10)
    p.add_argument("--min-sample-per-class", type=int, default=100)
    p.add_argument("--max-ood-size-per-class", type=int, default=100)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--reps", nargs="+", default=list(REPRESENTATIONS), choices=list(REPRESENTATIONS))
    p.add_argument("--out", default=str(REPO_ROOT / "Claude outputs/dpattern/dpattern_tv.csv"))
    p.add_argument("--raw-json", default=None, help="torch-free mode on the raw json")
    p.add_argument("--smoke", type=int, default=0, help="raw mode: subsample size (0 = all)")
    args = p.parse_args()

    splits = splits_from_raw(args) if args.raw_json else splits_from_dataset(args)
    for name, tris in splits.items():
        nv = [max(v for s in t for v in s) for t in tris]
        print(f"{name:16s} {len(tris):6d} graphs, vertices {min(nv)}..{max(nv)}")

    rows = compute(splits, args.depth, args.reps)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print_table(rows, args.depth)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
