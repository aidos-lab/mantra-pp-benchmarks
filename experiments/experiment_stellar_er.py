"""Experiment: how does simplicial effective resistance change under sequential stellar subdivisions?

For each triangulation K sampled from MANTRA:
  1. Compute ER of d-simplices and (d-1)-simplices (step 0).
  2. Iteratively pick an unused original top-dim simplex σ, perform a stellar
     subdivision (σ → d+1 new top simplices), recompute ER, and record how
     the following quantities change with the subdivision count:
       - Aggregations (sum, mean, L2-norm, max) of ER over all d-simplices,
         all (d-1)-simplices, and both combined.
       - The "ratio": sum of ER of the d+1 newly introduced d-simplices
         divided by ER(σ) before the subdivision.
  3. Save all results to a CSV for downstream analysis.

Usage
-----
    # Download MANTRA 2D data automatically, run on 20 triangulations, 15 steps each:
    python experiment_stellar_er.py --n_triangulations 20 --max_subdivisions 15

    # Use a local raw JSON file:
    python experiment_stellar_er.py --input path/to/2_manifolds.json

    # 3-manifolds, all original simplices, all triangulations:
    python experiment_stellar_er.py --dimension 3 --n_triangulations 0 --max_subdivisions 0

The output CSV has one row per (triangulation, subdivision step).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

import numpy as np
import pandas as pd

# ─── Simplicial complex utilities ────────────────────────────────────────────

def build_all_faces(top_simplices: list[frozenset], d: int) -> list[dict[frozenset, int]]:
    """
    Enumerate all k-faces (k=0..d) of a triangulation given its d-simplices.

    Returns
    -------
    faces : list of dicts, length d+1
        faces[k] maps frozenset(vertices) → 0-based index (sorted lexicographically
        for determinism).
    """
    face_sets: list[set[frozenset]] = [set() for _ in range(d + 1)]
    for sigma in top_simplices:
        verts = sorted(sigma)
        for k in range(d + 1):
            for combo in combinations(verts, k + 1):
                face_sets[k].add(frozenset(combo))

    faces = []
    for k in range(d + 1):
        ordered = sorted(face_sets[k], key=sorted)
        faces.append({s: i for i, s in enumerate(ordered)})
    return faces


def build_boundary_matrix(k_faces: dict, km1_faces: dict) -> np.ndarray:
    """
    Boundary matrix B_k of shape [n_{k-1}, n_k].

    B_k[i, j] = (-1)^r  if the i-th (k-1)-face is the r-th face of the j-th k-simplex
               (r = position of the omitted vertex in the sorted vertex list).
    """
    n_km1 = len(km1_faces)
    n_k = len(k_faces)
    B = np.zeros((n_km1, n_k), dtype=np.float64)
    for sigma, j in k_faces.items():
        verts = sorted(sigma)
        for r, _ in enumerate(verts):
            face = frozenset(verts[:r] + verts[r + 1 :])
            if face in km1_faces:
                B[km1_faces[face], j] = (-1.0) ** r
    return B


def compute_er_from_boundary(B_k: np.ndarray) -> np.ndarray:
    """
    Effective resistance of k-simplices via the up-Laplacian of (k-1)-chains.

    For the unweighted case (all weights = 1):
        L_up_{k-1} = B_k @ B_k^T          (up-Laplacian of (k-1)-simplices)
        ER_k        = diag(B_k^T @ L_up_{k-1}^+ @ B_k)

    This generalises the classical graph effective resistance (k=1) to higher
    dimensions and coincides with the formula used in EffectiveResistanceEmbedding.

    Parameters
    ----------
    B_k : np.ndarray, shape [n_{k-1}, n_k]

    Returns
    -------
    np.ndarray, shape [n_k]
    """
    n_k = B_k.shape[1]
    if B_k.shape[0] == 0 or n_k == 0:
        return np.zeros(n_k)
    L_up = B_k @ B_k.T                             # [n_{k-1}, n_{k-1}]
    L_plus = np.linalg.pinv(L_up)                  # pseudoinverse handles zero eigenvalues
    R = B_k.T @ L_plus @ B_k                       # [n_k, n_k]
    return np.diag(R)


def compute_er(
    top_simplices: list[frozenset], d: int
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """
    Compute ER vectors for d-simplices and (d-1)-simplices.

    Returns
    -------
    faces    : list of face dicts (index k gives k-face dict)
    er_d     : np.ndarray [n_d]   — ER of each d-simplex
    er_dm1   : np.ndarray [n_{d-1}] — ER of each (d-1)-simplex
                                       (empty array if d == 1)
    """
    faces = build_all_faces(top_simplices, d)

    # B_d: [n_{d-1}, n_d]
    B_d = build_boundary_matrix(faces[d], faces[d - 1])
    er_d = compute_er_from_boundary(B_d)

    if d <= 1:
        er_dm1 = np.empty(0)
    elif d == 2:
        # B_1: [n_0, n_1]  (vertices × edges)
        B_1 = build_boundary_matrix(faces[1], faces[0])
        er_dm1 = compute_er_from_boundary(B_1)
    else:
        # d >= 3: B_{d-1}: [n_{d-2}, n_{d-1}]
        B_dm1 = build_boundary_matrix(faces[d - 1], faces[d - 2])
        er_dm1 = compute_er_from_boundary(B_dm1)

    return faces, er_d, er_dm1


# ─── Stellar subdivision ──────────────────────────────────────────────────────

def stellar_subdivide(
    top_simplices: list[frozenset],
    sigma: frozenset,
    next_vertex: int,
) -> tuple[list[frozenset], list[frozenset], int]:
    """
    Stellar subdivision of σ: insert a new vertex b_σ, replace σ with
    (d+1) sub-simplices (σ \\ {v_i}) ∪ {b_σ} for each v_i ∈ σ.

    Returns
    -------
    new_top_simplices : list of frozensets
    new_faces         : list of frozensets — the d+1 newly added top simplices
    next_vertex + 1   : int
    """
    b = next_vertex
    verts = sorted(sigma)
    new_faces = [frozenset(verts[:i] + verts[i + 1 :] + [b]) for i in range(len(verts))]
    new_top = [s for s in top_simplices if s != sigma] + new_faces
    return new_top, new_faces, b + 1


# ─── Aggregation helpers ──────────────────────────────────────────────────────

def _agg(er: np.ndarray, prefix: str) -> dict:
    """Compute sum/mean/norm/max of an ER vector, prefixed for the output row."""
    if er.size == 0:
        return {f"{prefix}_sum": np.nan, f"{prefix}_mean": np.nan,
                f"{prefix}_norm": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_sum":  float(np.sum(er)),
        f"{prefix}_mean": float(np.mean(er)),
        f"{prefix}_norm": float(np.linalg.norm(er)),
        f"{prefix}_max":  float(np.max(er)),
    }


# ─── Per-triangulation experiment ─────────────────────────────────────────────

def run_experiment(
    entry: dict,
    max_subdivisions: int,
    rng: random.Random,
    tri_id: int,
) -> list[dict]:
    """
    Run the full sequential-stellar-subdivision experiment on one triangulation.

    Parameters
    ----------
    entry           : raw MANTRA JSON entry (keys: triangulation, dimension, n_vertices, name, …)
    max_subdivisions: cap on the number of subdivision steps (0 = no cap)
    rng             : seeded Random instance
    tri_id          : integer identifier for the output

    Returns
    -------
    List of row dicts, one per step (step 0 = initial state).
    """
    d = int(entry["dimension"])
    top_simplices: list[frozenset] = [frozenset(s) for s in entry["triangulation"]]
    manifold_name = entry.get("name", "unknown")

    all_verts = {v for s in top_simplices for v in s}
    next_vertex = max(all_verts) + 1

    # Pool: only original top-dim simplices can be subdivided.
    # We pick from this pool without replacement each step.
    pool = list(top_simplices)
    rng.shuffle(pool)

    n_original = len(pool)
    n_steps = n_original if max_subdivisions <= 0 else min(max_subdivisions, n_original)

    rows: list[dict] = []

    # ── Step 0: initial state ──
    faces, er_d, er_dm1 = compute_er(top_simplices, d)

    combined = np.concatenate([er_d, er_dm1]) if er_dm1.size > 0 else er_d
    base = {
        "tri_id": tri_id,
        "manifold_name": manifold_name,
        "dimension": d,
        "step": 0,
        "n_top_simplices": len(top_simplices),
        "n_faces_dm1": len(faces[d - 1]),
        "n_vertices": len(faces[0]),
        "er_subdivided_before": np.nan,
        "er_new_faces_sum": np.nan,
        "er_ratio_new_over_subdivided": np.nan,
    }
    rows.append({**base, **_agg(er_d, "er_d"), **_agg(er_dm1, "er_dm1"), **_agg(combined, "er_combined")})

    # ── Steps 1 … n_steps ──
    for step in range(1, n_steps + 1):
        if not pool:
            break

        sigma = pool.pop()                                    # random (pool was shuffled)

        # ER of σ before it is subdivided
        er_sigma_before = float(er_d[faces[d][sigma]])

        # Perform subdivision
        top_simplices, new_faces, next_vertex = stellar_subdivide(top_simplices, sigma, next_vertex)

        # Recompute ER on updated complex
        faces, er_d, er_dm1 = compute_er(top_simplices, d)

        # ER of newly introduced d-simplices
        new_er = np.array([er_d[faces[d][f]] for f in new_faces])
        er_new_sum = float(np.sum(new_er))
        ratio = (er_new_sum / er_sigma_before) if er_sigma_before > 1e-14 else np.nan

        combined = np.concatenate([er_d, er_dm1]) if er_dm1.size > 0 else er_d
        row = {
            "tri_id": tri_id,
            "manifold_name": manifold_name,
            "dimension": d,
            "step": step,
            "n_top_simplices": len(top_simplices),
            "n_faces_dm1": len(faces[d - 1]),
            "n_vertices": len(faces[0]),
            "er_subdivided_before": er_sigma_before,
            "er_new_faces_sum": er_new_sum,
            "er_ratio_new_over_subdivided": ratio,
            **_agg(er_d, "er_d"),
            **_agg(er_dm1, "er_dm1"),
            **_agg(combined, "er_combined"),
        }
        rows.append(row)

    return rows


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_or_download(input_path: str | None, dimension: int, data_root: str) -> list[dict]:
    """
    Load raw MANTRA JSON. If `input_path` is None, download via the MANTRA class.
    """
    if input_path is not None:
        with open(input_path) as f:
            return json.load(f)

    print(f"No --input provided; downloading MANTRA {dimension}D dataset to {data_root!r} …")
    try:
        from mantra.datasets.mantra import MANTRA
    except ImportError:
        sys.exit(
            "mantra-dataset is not installed. Install it or pass --input pointing to a raw JSON file."
        )

    # Instantiating MANTRA triggers download + processing.
    MANTRA(root=data_root, dimension=dimension)

    # Locate the raw JSON that was just downloaded.
    raw_dir = Path(data_root) / f"mantra/{dimension}D/raw"
    candidates = list(raw_dir.glob(f"{dimension}_manifolds*.json"))
    if not candidates:
        sys.exit(f"Could not find raw JSON in {raw_dir}. Check {data_root}.")
    raw_path = candidates[0]
    print(f"Using raw file: {raw_path}")
    with open(raw_path) as f:
        return json.load(f)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stellar-subdivision ER experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default=None,
        help="Path to raw MANTRA JSON file (e.g. 2_manifolds.json). "
             "If omitted, data is downloaded automatically.",
    )
    parser.add_argument(
        "--dimension", type=int, default=2, choices=[2, 3],
        help="Manifold dimension to use when downloading (ignored if --input is given).",
    )
    parser.add_argument(
        "--n_triangulations", type=int, default=20,
        help="Number of triangulations to sample (0 = use all).",
    )
    parser.add_argument(
        "--max_subdivisions", type=int, default=10,
        help="Maximum stellar subdivisions per triangulation "
             "(0 = subdivide every original top-dim simplex).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "results_stellar_er.csv"),
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--data_root", default=str(ROOT / "data"),
        help="Root directory for MANTRA download (used only when --input is omitted).",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    data = load_or_download(args.input, args.dimension, args.data_root)
    print(f"Loaded {len(data)} triangulations.")

    # Filter by dimension if using a file that may contain multiple dimensions.
    if args.input is not None:
        before = len(data)
        data = [e for e in data if int(e.get("dimension", args.dimension)) == args.dimension]
        if len(data) < before:
            print(f"Filtered to {len(data)} {args.dimension}D triangulations.")

    if args.n_triangulations > 0 and len(data) > args.n_triangulations:
        data = rng.sample(data, args.n_triangulations)
        print(f"Sampled {len(data)} triangulations.")

    print(
        f"Running experiment: {len(data)} triangulations × up to "
        f"{args.max_subdivisions or 'all'} subdivisions each."
    )

    all_rows: list[dict] = []
    for i, entry in enumerate(data):
        n_top = len(entry["triangulation"])
        print(
            f"  [{i + 1:3d}/{len(data)}] {str(entry.get('name', '?')):<20s} "
            f"d={entry['dimension']}  V={entry['n_vertices']:5d}  top={n_top:6d}"
        )
        try:
            rows = run_experiment(entry, args.max_subdivisions, rng, tri_id=i)
            all_rows.extend(rows)
        except Exception as exc:
            print(f"    SKIPPED — {exc}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows ({df['tri_id'].nunique()} triangulations, "
          f"{df['step'].max()} max steps) → {args.output}")

    # Quick summary table averaged across triangulations.
    agg_cols = [
        "er_d_sum", "er_d_mean", "er_d_norm", "er_d_max",
        "er_dm1_sum", "er_dm1_mean", "er_dm1_norm",
        "er_combined_sum", "er_combined_mean",
        "er_ratio_new_over_subdivided",
    ]
    present = [c for c in agg_cols if c in df.columns]
    summary = df.groupby("step")[present].mean()
    print("\nMean across triangulations per subdivision step:")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
