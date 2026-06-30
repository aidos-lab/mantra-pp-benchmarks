"""
Mean triangle ER convergence for 3-manifold triangulations under stellar subdivision.

Mirrors the style of plot_mean_er_convergence.py (2D edge ER), with β₁ playing
the role that χ plays there.

Formula (derived using χ=0 for all closed connected 3-manifolds):
  sum_f ER(f) = rank(B₂) = |T| + 1 - β₁
  mean triangle ER = (|T| + 1 - β₁) / (2|T|)   [since |F| = 2|T|]

Each subdivision adds exactly 3 tetrahedra and 6 triangles regardless of which
tetrahedron is chosen, so the trajectory is completely deterministic:
  f(t) = (T₀ + 1 - β₁ + 3t) / (2T₀ + 6t)  →  1/2

Because the formula is exact, no matrix computation is needed — we read T₀ directly
from the entry and evaluate the formula for 200 steps analytically.

Monotonicity is controlled entirely by β₁ (a topological invariant):
  β₁ = 0:  f'(t) < 0, decreases from (T₀+1)/(2T₀) > 1/2
  β₁ = 1:  f'(t) = 0, constant at exactly 1/2 for every triangulation
  β₁ ≥ 2:  f'(t) > 0, increases from below 1/2

Left panel  — β₁ = 0 types (S³, RP³, lens spaces, S³/Q)
Right panel — β₁ ≥ 1 types (S²×S¹, S²⋊S¹, connected sums, T³)
S³ shown as dashed reference on both panels (like S² in the 2D edge ER plot).
"""

from __future__ import annotations
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines

ROOT = Path(__file__).resolve().parent.parent

DATA_3D = ROOT / "data" / "mantra" / "3D" / "raw" / "3_manifolds.json"
OUTPUT  = ROOT / "results" / "results_mean_triangle_er_convergence_3d.png"

N_PER_TYPE = 5
N_STEPS    = 200

BETA1 = {
    "S^3":                       0,
    "RP^3":                      0,
    "L(3,1)":                    0,
    "L(4,1)":                    0,
    "S^3/Q":                     0,
    "S^2 x S^1":                 1,
    "S^2 twist S^1":             1,
    "(S^2 x S^1)#(S^2 x S^1)":  2,
    "T^3":                       3,
}

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


# ── Computation ───────────────────────────────────────────────────────────────

def trajectory(T0: int, beta1: int, n_steps: int) -> np.ndarray:
    """Exact closed-form trajectory: (T₀+1-β₁+3t) / (2T₀+6t)."""
    t = np.arange(n_steps + 1, dtype=float)
    return (T0 + 1 - beta1 + 3 * t) / (2 * T0 + 6 * t)


def collect(data: list) -> dict[str, list[np.ndarray]]:
    """Read T₀ from each entry and compute the exact trajectory analytically."""
    by_name: dict[str, list] = defaultdict(list)
    for entry in data:
        name  = entry.get("name", "?")
        T0    = len(entry["triangulation"])      # tetrahedra count
        beta1 = BETA1.get(name, 0)
        by_name[name].append(trajectory(T0, beta1, N_STEPS))
    return {k: np.stack(v) for k, v in by_name.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────

def _cmap_colors(names, cmap_name):
    cmap = plt.colormaps[cmap_name]
    n    = max(len(names) - 1, 1)
    return {name: cmap(0.28 + 0.58 * i / n) for i, name in enumerate(names)}


def plot(trajs: dict[str, np.ndarray], output: Path) -> None:
    steps    = np.arange(N_STEPS + 1)
    s3_color = "#2166ac"

    b0_names = [n for n in trajs if BETA1.get(n, 0) == 0 and n != "S^3"]
    b1_names = sorted([n for n in trajs if BETA1.get(n, 0) >= 1],
                      key=lambda n: BETA1.get(n, 0))

    b0_colors = _cmap_colors(sorted(b0_names), "Purples")
    b1_colors = _cmap_colors(b1_names,          "Oranges")

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.0),
        gridspec_kw={"wspace": 0.25, "left": 0.07, "right": 0.95,
                     "top": 0.93, "bottom": 0.12},
    )

    def _draw(ax, names, colors, title, legend_title):
        if "S^3" in trajs:
            arr  = trajs["S^3"]
            m, s = arr.mean(0), arr.std(0)
            ax.plot(steps, m, color=s3_color, lw=2.0, ls="--", zorder=5)
            ax.fill_between(steps, m - s, m + s, color=s3_color, alpha=0.10)

        for name in names:
            arr  = trajs[name]
            m, s = arr.mean(0), arr.std(0)
            c    = colors[name]
            ax.plot(steps, m, color=c, lw=1.6, label=LABELS.get(name, name))
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.12)

        ax.axhline(1/2, color="#333333", lw=1.0, ls="--", alpha=0.5)
        ax.text(N_STEPS + 2, 1/2, "1/2", va="center",
                fontsize=8.5, color="#333333", alpha=0.7)

        ax.set_xlabel("stellar subdivisions", fontsize=9.5)
        ax.set_ylabel("mean triangle ER", fontsize=9.5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlim(-2, N_STEPS + 12)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.15, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

        handles  = [mlines.Line2D([], [], color=s3_color, lw=2.0, ls="--",
                                   label=r"$S^3$")]
        handles += [mlines.Line2D([], [], color=colors[n], lw=1.6,
                                   label=LABELS.get(n, n))
                    for n in names]
        formula_h = mlines.Line2D(
            [], [], color="gray", lw=1.6,
            label=r"$(T_0\!+\!1\!-\!\beta_1\!+\!3t)\,/\,(2T_0\!+\!6t)$  [exact]",
        )
        ax.legend(handles=handles + [formula_h],
                  fontsize=7.5, loc="center right", title=legend_title,
                  title_fontsize=8.5, framealpha=0.9, edgecolor="#cccccc")

    _draw(axes[0], sorted(b0_names), b0_colors,
          r"$\beta_1 = 0$  (rational homology spheres)",
          r"$\beta_1 = 0$")
    _draw(axes[1], b1_names, b1_colors,
          r"$\beta_1 \geq 1$",
          r"$\beta_1$")

    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_3D) as f:
        full_data = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    sample = []
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        sample.extend(chosen)

    print(f"{len(sample)} triangulations — formula evaluated analytically "
          f"for {N_STEPS} steps (no matrix computation needed).\n")
    trajs = collect(sample)

    print("Plotting…")
    plot(trajs, output=OUTPUT)


if __name__ == "__main__":
    main()
