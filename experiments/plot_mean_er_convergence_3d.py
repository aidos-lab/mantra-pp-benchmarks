"""
Mean edge ER convergence for 3-manifold triangulations under stellar subdivision.

Left panel  — all trajectories together, one colour per manifold type.
Right panel — starting sparsity E₀ vs V₀ with threshold E = 4(V-1):
              triangulations above the line increase toward 1/4,
              those below decrease.  This is the actual control variable,
              not β₁ or any other topological invariant.
"""

from __future__ import annotations
import json, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_stellar_er import (
    build_all_faces, build_boundary_matrix, compute_er_from_boundary,
    stellar_subdivide,
)

DATA_3D = ROOT / "data" / "mantra" / "3D" / "raw" / "3_manifolds.json"
OUTPUT  = ROOT / "results" / "results_mean_er_convergence_3d.png"

N_PER_TYPE = 5
N_STEPS    = 200
SEED       = 42
D          = 3

LABELS = {
    "S^3":                       r"$S^3$",
    "RP^3":                      r"$RP^3$",
    "L(3,1)":                    r"$L(3,1)$",
    "L(4,1)":                    r"$L(4,1)$",
    "S^3/Q":                     r"$S^3/Q$",
    "S^2 x S^1":                 r"$S^2 \times S^1$",
    "S^2 twist S^1":             r"$S^2 \tilde\times S^1$",
    "(S^2 x S^1)#(S^2 x S^1)":  r"$(S^2{\times}S^1)^{\#2}$",
    "T^3":                       r"$T^3$",
}


# ── Core computation ──────────────────────────────────────────────────────────

def mean_edge_er_trajectory(entry: dict, n_steps: int, rng: random.Random
                            ) -> tuple[np.ndarray, int, int]:
    """Returns (trajectory, V₀, E₀)."""
    top = [frozenset(s) for s in entry["triangulation"]]
    nv  = max(v for s in top for v in s) + 1
    traj = []
    V0 = E0 = None
    for step in range(n_steps + 1):
        faces = build_all_faces(top, D)
        B1    = build_boundary_matrix(faces[1], faces[0])
        er    = compute_er_from_boundary(B1)
        traj.append(float(er.mean()))
        if step == 0:
            V0, E0 = len(faces[0]), len(faces[1])
        if step < n_steps:
            sigma = rng.choice(top)
            top, _, nv = stellar_subdivide(top, sigma, nv)
    return np.array(traj), V0, E0


def collect(data: list, rng: random.Random
            ) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[int, int]]]]:
    """Returns (trajectories, starting_v0e0) per type."""
    by_traj: dict[str, list] = defaultdict(list)
    by_v0e0: dict[str, list] = defaultdict(list)
    total = len(data)
    for i, entry in enumerate(data):
        name = entry.get("name", "?")
        print(f"  [{i+1:3d}/{total}] {name:<40s}  V={entry['n_vertices']}")
        traj, V0, E0 = mean_edge_er_trajectory(entry, N_STEPS, rng)
        by_traj[name].append(traj)
        by_v0e0[name].append((V0, E0))
    trajs = {k: np.stack(v) for k, v in by_traj.items()}
    return trajs, dict(by_v0e0)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(trajs: dict[str, np.ndarray],
         v0e0s: dict[str, list[tuple[int, int]]],
         output: Path) -> None:

    steps     = np.arange(N_STEPS + 1)
    all_names = sorted(trajs.keys(), key=lambda n: LABELS.get(n, n))

    # One colour per type from tab10
    cmap   = plt.colormaps["tab10"]
    colors = {n: cmap(i % 10) for i, n in enumerate(all_names)}

    fig, (ax_traj, ax_sc) = plt.subplots(
        1, 2, figsize=(14, 5.0),
        gridspec_kw={"wspace": 0.28, "left": 0.07, "right": 0.96,
                     "top": 0.93, "bottom": 0.12},
    )

    # ── Left: all trajectories ────────────────────────────────────────────────
    for name in all_names:
        arr  = trajs[name]
        m, s = arr.mean(0), arr.std(0)
        c    = colors[name]
        ls   = "--" if name == "S^3" else "-"
        lw   = 2.0  if name == "S^3" else 1.6
        ax_traj.plot(steps, m, color=c, lw=lw, ls=ls,
                     label=LABELS.get(name, name))
        ax_traj.fill_between(steps, m - s, m + s, color=c, alpha=0.10)

    ax_traj.axhline(1/4, color="#333333", lw=1.0, ls="--", alpha=0.5)
    ax_traj.text(N_STEPS + 2, 1/4, "1/4", va="center", fontsize=8.5,
                 color="#333333", alpha=0.7)

    ax_traj.set_xlabel("stellar subdivisions", fontsize=9.5)
    ax_traj.set_ylabel("mean edge ER", fontsize=9.5)
    ax_traj.set_title("All 3-manifold types", fontsize=11, pad=8)
    ax_traj.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_traj.tick_params(labelsize=8.5)
    ax_traj.grid(alpha=0.15, ls="--")
    ax_traj.spines[["top", "right"]].set_visible(False)
    ax_traj.legend(fontsize=8, loc="center right",
                   framealpha=0.9, frameon=True, edgecolor="#cccccc")

    # ── Right: sparsity scatter E₀ vs V₀ ─────────────────────────────────────
    all_v0 = [v for pairs in v0e0s.values() for v, _ in pairs]
    all_e0 = [e for pairs in v0e0s.values() for _, e in pairs]
    v_lo, v_hi = min(all_v0) - 1, max(all_v0) + 2
    e_lo, e_hi = 0, max(all_e0) * 1.08

    vv = np.linspace(v_lo, v_hi, 200)
    threshold = 4 * (vv - 1)

    # Shade regions
    ax_sc.fill_between(vv, threshold, e_hi, color="#d4eac8", alpha=0.45,
                       zorder=0)  # above → increases
    ax_sc.fill_between(vv, e_lo, threshold, color="#fde0d0", alpha=0.45,
                       zorder=0)  # below → decreases

    # Threshold line
    ax_sc.plot(vv, threshold, color="#555555", lw=1.4, ls="--", zorder=2,
               label=r"$E = 4(V-1)$")

    # Scatter one dot per triangulation
    for name in all_names:
        c = colors[name]
        first = True
        for V0, E0 in v0e0s[name]:
            marker = "v" if E0 < 4 * (V0 - 1) else "^"
            ax_sc.scatter(V0, E0, color=c, s=70, zorder=3, marker=marker,
                          label=LABELS.get(name, name) if first else "_")
            first = False

    # Region labels
    ax_sc.text(v_lo + 0.4, e_hi * 0.88, "ER increases  ↑",
               fontsize=8.5, color="#2e7d32", alpha=0.85)
    ax_sc.text(v_lo + 0.4, e_lo + (threshold[0] - e_lo) * 0.25,
               "ER decreases  ↓",
               fontsize=8.5, color="#c62828", alpha=0.85)

    ax_sc.set_xlim(v_lo, v_hi)
    ax_sc.set_ylim(e_lo, e_hi)
    ax_sc.set_xlabel(r"starting vertices $V_0$", fontsize=9.5)
    ax_sc.set_ylabel(r"starting edges $E_0$", fontsize=9.5)
    ax_sc.set_title(r"Starting sparsity: $E_0$ vs $4(V_0 - 1)$",
                    fontsize=11, pad=8)
    ax_sc.tick_params(labelsize=8.5)
    ax_sc.grid(alpha=0.15, ls="--")
    ax_sc.spines[["top", "right"]].set_visible(False)

    # Legend: marker shapes for direction + threshold line
    up_h   = mlines.Line2D([], [], color="gray", lw=0, marker="^",
                            markersize=7, label="increases toward 1/4")
    down_h = mlines.Line2D([], [], color="gray", lw=0, marker="v",
                            markersize=7, label="decreases toward 1/4")
    thr_h  = mlines.Line2D([], [], color="#555555", lw=1.4, ls="--",
                            label=r"$E_0 = 4(V_0-1)$")
    ax_sc.legend(handles=[thr_h, up_h, down_h], fontsize=8,
                 loc="upper left", framealpha=0.9, edgecolor="#cccccc")

    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_3D) as f:
        full_data = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    rng    = random.Random(SEED)
    sample = []
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        sample.extend(chosen)

    print(f"{len(sample)} triangulations × {N_STEPS} steps each.\n")
    trajs, v0e0s = collect(sample, rng)

    print("\nPlotting…")
    plot(trajs, v0e0s, OUTPUT)


if __name__ == "__main__":
    main()
