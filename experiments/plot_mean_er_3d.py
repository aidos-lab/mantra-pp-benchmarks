"""
Mean edge ER convergence for 3-manifold triangulations under stellar subdivision.

Key formulas (work in any dimension d):
  mean edge ER = (V - 1) / E             [exact: sum of edge ERs = V - 1]
  each subdivision adds 1 vertex, d+1 edges
  trajectory:  (V₀ + t - 1) / (E₀ + (d+1)·t)  →  1/(d+1)  as t → ∞

For d=3:  limit = 1/4  (vs 1/3 for d=2)

In 2D: χ fixed E = 3(V-χ), so trajectory depended only on (V₀, χ).
In 3D: χ = 0 for all closed 3-manifolds, so E₀ varies between triangulations.
       Monotone direction depends on whether E₀ > 4(V₀-1) [increase] or < [decrease].
"""

from __future__ import annotations
import json, re, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_stellar_er import (
    build_all_faces, build_boundary_matrix, compute_er_from_boundary,
    stellar_subdivide,
)

DATA_3D   = ROOT / "data" / "mantra" / "3D" / "raw" / "3_manifolds.json"
OUTPUT    = ROOT / "results" / "results_mean_er_3d.png"

N_PER_TYPE = 5
N_STEPS    = 200
SEED       = 42
D          = 3


def mean_edge_er_trajectory(entry: dict, n_steps: int, rng: random.Random) -> tuple[np.ndarray, int, int]:
    """Returns (trajectory, V₀, E₀)."""
    top = [frozenset(s) for s in entry["triangulation"]]
    nv  = max(v for s in top for v in s) + 1
    traj = []
    V0, E0 = None, None
    for step in range(n_steps + 1):
        faces  = build_all_faces(top, D)
        B1     = build_boundary_matrix(faces[1], faces[0])
        er     = compute_er_from_boundary(B1)
        traj.append(float(er.mean()))
        if step == 0:
            V0 = len(faces[0])
            E0 = len(faces[1])
        if step < n_steps:
            sigma = rng.choice(top)
            top, _, nv = stellar_subdivide(top, sigma, nv)
    return np.array(traj), V0, E0


def theoretical(V0: int, E0: int, n_steps: int) -> np.ndarray:
    """Exact formula: (V0 + t - 1) / (E0 + 4t)"""
    t = np.arange(n_steps + 1, dtype=float)
    return (V0 + t - 1) / (E0 + (D + 1) * t)


def _label(name: str) -> str:
    return (name
            .replace("S^3",         r"$S^3$")
            .replace("S^2 x S^1",   r"$S^2 \times S^1$")
            .replace("S^2 ~ S^1",   r"$S^2 \tilde\times S^1$")
            .replace("RP^3",        r"$RP^3$")
            .replace("L(3,1)",      r"$L(3,1)$"))


def main():
    with open(DATA_3D) as f:
        full_data = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    print(f"3D dataset: {len(full_data)} triangulations, "
          f"{len(by_type)} distinct types.")
    for name, entries in sorted(by_type.items(),
                                key=lambda kv: -len(kv[1])):
        print(f"  {name:<30s}  {len(entries):>6d}")

    rng = random.Random(SEED)

    # Smallest triangulations per type
    sample: list[dict] = []
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        sample.extend(chosen)

    # Collect trajectories
    trajs:  dict[str, list[np.ndarray]] = defaultdict(list)
    v0s:    dict[str, list[int]]        = defaultdict(list)
    e0s:    dict[str, list[int]]        = defaultdict(list)

    total = len(sample)
    for i, entry in enumerate(sample):
        name = entry.get("name", "?")
        V0 = entry["n_vertices"]
        print(f"  [{i+1:3d}/{total}] {name:<30s}  V={V0}")
        traj, V0_, E0 = mean_edge_er_trajectory(entry, N_STEPS, rng)
        trajs[name].append(traj)
        v0s[name].append(V0_)
        e0s[name].append(E0)

    # ── Plot ─────────────────────────────────────────────────────────────────
    all_names = sorted(trajs.keys())
    n_types   = len(all_names)
    steps     = np.arange(N_STEPS + 1)

    # Colour per type
    cmap   = plt.colormaps["tab10"]
    colors = {n: cmap(i % 10) for i, n in enumerate(all_names)}

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"wspace": 0.30, "left": 0.07, "right": 0.97,
                     "top": 0.82, "bottom": 0.12},
    )
    ax_traj, ax_v0e0 = axes

    # ── Left: trajectories ───────────────────────────────────────────────────
    for name in all_names:
        arr  = np.stack(trajs[name])
        m, s = arr.mean(0), arr.std(0)
        c    = colors[name]
        ax_traj.plot(steps, m, color=c, lw=2.0, label=_label(name))
        ax_traj.fill_between(steps, m - s, m + s, color=c, alpha=0.15)

        # Theoretical curve using mean V₀, E₀
        V0m = int(round(np.mean(v0s[name])))
        E0m = int(round(np.mean(e0s[name])))
        th  = theoretical(V0m, E0m, N_STEPS)
        ax_traj.plot(steps, th, color=c, lw=1.0, ls=":", alpha=0.9)

    ax_traj.axhline(1/4, color="black", lw=1.2, ls="--", alpha=0.7,
                    label="limit = 1/4")
    ax_traj.axhline(1/3, color="gray", lw=0.8, ls="--", alpha=0.4,
                    label="2D limit = 1/3  (ref.)")

    ax_traj.set_xlabel("# stellar subdivisions", fontsize=10)
    ax_traj.set_ylabel("Mean edge ER", fontsize=10)
    ax_traj.set_title(
        r"Mean edge ER $= (V-1)/E$ · 200 stellar subdivisions"
        "\n"
        r"solid = simulation ± 1σ,  dotted = $(V_0+t-1)/(E_0+4t)$",
        fontsize=10, fontweight="bold", pad=6,
    )
    ax_traj.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_traj.grid(alpha=0.2, ls="--")
    ax_traj.legend(fontsize=8.5, loc="center right", framealpha=0.9)

    # ── Right: starting E₀ vs V₀, showing E₀ vs 4(V₀-1) threshold ──────────
    plotted = set()
    for name in all_names:
        c = colors[name]
        for V0, E0 in zip(v0s[name], e0s[name]):
            lbl = _label(name) if name not in plotted else "_"
            ax_v0e0.scatter(V0, E0, color=c, s=55, zorder=3, label=lbl)
            plotted.add(name)

    # threshold line E = 4(V-1)
    all_v0 = [v for vlist in v0s.values() for v in vlist]
    vv = np.linspace(min(all_v0) - 0.5, max(all_v0) + 0.5, 100)
    ax_v0e0.plot(vv, 4 * (vv - 1), color="black", lw=1.5, ls="--",
                 label=r"$E = 4(V-1)$  (ER constant)")

    ax_v0e0.set_xlabel("Starting vertex count  $V_0$", fontsize=10)
    ax_v0e0.set_ylabel("Starting edge count  $E_0$", fontsize=10)
    ax_v0e0.set_title(
        r"$E_0$ vs $V_0$ for each starting triangulation"
        "\n"
        r"Above $E=4(V-1)$: ER ↑;   Below: ER ↓",
        fontsize=10, fontweight="bold", pad=6,
    )
    ax_v0e0.grid(alpha=0.2, ls="--")
    ax_v0e0.legend(fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        r"3-manifolds: mean edge ER $\to$ $\mathbf{1/4}$  "
        r"(vs $\mathbf{1/3}$ for 2-manifolds)"
        "\n"
        r"General rule for $d$-manifolds: limit $= 1/(d+1)$, "
        r"trajectory $= (V_0+t-1)\,/\,(E_0+(d+1)t)$",
        fontsize=11, y=1.01,
    )
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=150)
    print(f"\nSaved → {OUTPUT}")

    # Summary table
    print(f"\n{'Type':<30s}  {'N':>3}  {'V0':>4}  {'E0':>5}  "
          f"{'E0/V0':>6}  {'4(V0-1)/V0':>11}  {'mono':>6}  {'val@200':>8}")
    print("-" * 85)
    for name in all_names:
        for V0, E0, traj in zip(v0s[name], e0s[name], trajs[name]):
            mono = "↑" if E0 > 4*(V0-1) else ("=" if E0 == 4*(V0-1) else "↓")
            print(f"{name:<30s}  {1:>3}  {V0:>4}  {E0:>5}  "
                  f"{E0/V0:>6.2f}  {4*(V0-1)/V0:>11.2f}  {mono:>6}  {traj[-1]:>8.5f}")


if __name__ == "__main__":
    main()
