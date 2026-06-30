"""
Stellar-subdivision ER experiment across all 21 manifold types in the 2D MANTRA dataset.

New edge-based measures (replacing the trivial triangle ratio):
  For each subdivision of σ = {a,b,c} that adds vertex v:
    - er_new_edges         : Σ ER({a,v}, {b,v}, {c,v})  after subdivision
    - er_boundary_before   : Σ ER({a,b}, {b,c}, {a,c})  BEFORE subdivision
    - er_boundary_after    : Σ ER({a,b}, {b,c}, {a,c})  AFTER  subdivision
    → edge_ratio           = er_new_edges / er_boundary_before
    → boundary_change      = er_boundary_after / er_boundary_before

Usage
-----
    uv run --extra cpu python experiment_stellar_er_types.py
"""

from __future__ import annotations
import json, random, sys
from itertools import combinations
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_stellar_er import build_all_faces, build_boundary_matrix, \
    compute_er_from_boundary, stellar_subdivide

DATA_PATH  = str(ROOT / "data/mantra/2D/raw/2_manifolds.json")
N_PER_TYPE = 5
MAX_STEPS  = 10
SEED       = 42
OUTPUT_CSV = str(ROOT / "results/results_stellar_er_types.csv")


def compute_er(top_simplices: list, d: int):
    faces = build_all_faces(top_simplices, d)
    B_d   = build_boundary_matrix(faces[d], faces[d - 1])
    er_d  = compute_er_from_boundary(B_d)
    B_1   = build_boundary_matrix(faces[1], faces[0])
    er_dm1 = compute_er_from_boundary(B_1)
    return faces, er_d, er_dm1


def run_one(entry: dict, max_steps: int, rng: random.Random, tri_id: int):
    d    = int(entry["dimension"])
    name = entry.get("name", "?")
    top  = [frozenset(s) for s in entry["triangulation"]]

    all_verts = {v for s in top for v in s}
    next_v    = max(all_verts) + 1

    pool = list(top)
    rng.shuffle(pool)
    n_steps = min(max_steps, len(pool))

    faces, er_d, er_dm1 = compute_er(top, d)
    rows = [_base_row(tri_id, name, 0, top, faces, er_d, er_dm1,
                      np.nan, np.nan, np.nan, np.nan, np.nan)]

    for step in range(1, n_steps + 1):
        sigma   = pool.pop()
        new_vtx = next_v                   # label of the vertex about to be added

        # Boundary edges of σ (BEFORE subdivision)
        b_edges = [frozenset(e) for e in combinations(sorted(sigma), 2)]
        er_bnd_before = sum(er_dm1[faces[d - 1][e]]
                            for e in b_edges if e in faces[d - 1])

        top, _, next_v = stellar_subdivide(top, sigma, new_vtx)

        faces, er_d, er_dm1 = compute_er(top, d)

        # New edges {vertex_of_σ, new_vtx} (AFTER subdivision)
        new_edges = [frozenset({v, new_vtx}) for v in sigma]
        er_new    = sum(er_dm1[faces[d - 1][e]]
                        for e in new_edges if e in faces[d - 1])

        # Boundary edges AFTER subdivision
        er_bnd_after = sum(er_dm1[faces[d - 1][e]]
                           for e in b_edges if e in faces[d - 1])

        edge_ratio   = er_new / er_bnd_before  if er_bnd_before > 1e-14 else np.nan
        bnd_change   = er_bnd_after / er_bnd_before if er_bnd_before > 1e-14 else np.nan

        rows.append(_base_row(tri_id, name, step, top, faces, er_d, er_dm1,
                              er_new, er_bnd_before, er_bnd_after,
                              edge_ratio, bnd_change))
    return rows


def _base_row(tri_id, name, step, top, faces, er_d, er_dm1,
              er_new, er_bnd_before, er_bnd_after, edge_ratio, bnd_change):
    d = 2
    return {
        "tri_id":            tri_id,
        "manifold_name":     name,
        "step":              step,
        "n_triangles":       len(faces[d]),
        "n_edges":           len(faces[d - 1]),
        "n_vertices":        len(faces[0]),
        # triangle ER
        "er_d_mean":         float(er_d.mean()),
        "er_d_max":          float(er_d.max()),
        "er_d_min":          float(er_d.min()),
        # edge ER
        "er_dm1_mean":       float(er_dm1.mean()),
        "er_dm1_max":        float(er_dm1.max()),
        "er_dm1_min":        float(er_dm1.min()),
        "er_dm1_std":        float(er_dm1.std()),
        # new edge measure
        "er_new_edges":      er_new,
        "er_bnd_before":     er_bnd_before,
        "er_bnd_after":      er_bnd_after,
        "edge_ratio":        edge_ratio,        # Σ ER(new) / Σ ER(boundary before)
        "boundary_change":   bnd_change,        # Σ ER(boundary after) / Σ ER(boundary before)
    }


def main():
    with open(DATA_PATH) as f:
        full_data = json.load(f)

    # Group by manifold type
    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    print(f"Found {len(by_type)} distinct manifold types.")

    rng = random.Random(SEED)

    # Sample up to N_PER_TYPE from each type (prefer smallest triangulations)
    sample: list[dict] = []
    for name, entries in sorted(by_type.items()):
        pool_sorted = sorted(entries, key=lambda e: e["n_vertices"])
        chosen = pool_sorted[:N_PER_TYPE]
        sample.extend(chosen)
        print(f"  {name:<20s}  {len(chosen):2d} chosen  "
              f"(V range: {chosen[0]['n_vertices']}–{chosen[-1]['n_vertices']})")

    print(f"\nTotal: {len(sample)} triangulations across {len(by_type)} types, "
          f"{MAX_STEPS} steps each.\n")

    all_rows = []
    for tri_id, entry in enumerate(sample):
        all_rows.extend(run_one(entry, MAX_STEPS, rng, tri_id))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} rows → {OUTPUT_CSV}")

    # Quick summary per type
    final = df[df["step"] == MAX_STEPS]
    print(f"\n{'Type':<20s} {'N':>3} {'ER_d (unif)':>12} {'edge_ratio mean':>16} "
          f"{'bnd_change mean':>16}")
    print("-" * 72)
    for name, g in final.groupby("manifold_name"):
        print(f"{name:<20s} {len(g):>3}  {g['er_d_mean'].mean():>10.6f}  "
              f"{g['edge_ratio'].mean():>16.4f}  {g['boundary_change'].mean():>16.4f}")


if __name__ == "__main__":
    main()
