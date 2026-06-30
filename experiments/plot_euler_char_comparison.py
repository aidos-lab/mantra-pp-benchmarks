"""
Compare mean edge ER convergence for surfaces that share the same Euler characteristic.

Key identity:  mean edge ER = (V - 1) / (3(V - χ))
where V = current vertex count, χ = Euler characteristic of the surface.

This is exact and deterministic — the value does not depend on which
simplex is subdivided, only on (V, χ).

Surfaces sharing the same χ but different topology (orientable vs non-orientable)
must therefore follow identical trajectories if they start with the same V.
The theoretical curve (V₀ + t - 1) / (3(V₀ + t - χ)) is overlaid for each group.

Plot: one subplot per Euler characteristic value present in the dataset.
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

DATA_PATH = ROOT / "data" / "mantra" / "2D" / "raw" / "2_manifolds.json"
OUTPUT    = ROOT / "results" / "results_euler_char_comparison.png"

N_PER_TYPE = 5
N_STEPS    = 200
SEED       = 42


# ── Type helpers ──────────────────────────────────────────────────────────────

def euler_char(name: str) -> int:
    if name == "S^2":          return  2
    if name == "T^2":          return  0
    if name == "RP^2":         return  1
    if name == "Klein bottle": return  0
    m = re.match(r"#\^(\d+)\s*T\^2$",  name)
    if m: return 2 - 2 * int(m.group(1))
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m: return 2 - int(m.group(1))
    return 0


def _label(name: str) -> str:
    if name == "S^2":          return r"$S^2$"
    if name == "T^2":          return r"$T^2$"
    if name == "RP^2":         return r"$RP^2$"
    if name == "Klein bottle": return "Klein"
    m = re.match(r"#\^(\d+)\s*T\^2$",  name)
    if m: return rf"$\#{{{m.group(1)}}}T^2$"
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m: return rf"$\#{{{m.group(1)}}}RP^2$"
    return name


# ── Core computation ──────────────────────────────────────────────────────────

def mean_edge_er_trajectory(entry: dict, n_steps: int, rng: random.Random) -> np.ndarray:
    top = [frozenset(s) for s in entry["triangulation"]]
    nv  = max(v for s in top for v in s) + 1
    traj = []
    for step in range(n_steps + 1):
        faces  = build_all_faces(top, 2)
        B1     = build_boundary_matrix(faces[1], faces[0])
        er     = compute_er_from_boundary(B1)
        traj.append(float(er.mean()))
        if step < n_steps:
            sigma = rng.choice(top)
            top, _, nv = stellar_subdivide(top, sigma, nv)
    return np.array(traj)


def theoretical(V0: int, chi: int, n_steps: int) -> np.ndarray:
    """Exact formula: (V0 + t - 1) / (3 * (V0 + t - chi))"""
    t = np.arange(n_steps + 1, dtype=float)
    return (V0 + t - 1) / (3 * (V0 + t - chi))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_PATH) as f:
        full_data = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    rng = random.Random(SEED)

    # Collect trajectories
    trajs: dict[str, list[np.ndarray]] = defaultdict(list)  # name → list of arrays
    v0s:   dict[str, list[int]]        = defaultdict(list)   # name → V₀ values

    total = sum(min(len(v), N_PER_TYPE) for v in by_type.values())
    i = 0
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        for entry in chosen:
            i += 1
            V0 = entry["n_vertices"]
            print(f"  [{i:3d}/{total}] {name:<22s}  V={V0}  χ={euler_char(name)}")
            traj = mean_edge_er_trajectory(entry, N_STEPS, rng)
            trajs[name].append(traj)
            v0s[name].append(V0)

    # Group by χ
    by_chi: dict[int, list[str]] = defaultdict(list)
    for name in trajs:
        by_chi[euler_char(name)].append(name)

    chi_values = sorted(by_chi.keys(), reverse=True)  # high χ first

    # Layout: wrap into rows
    n_panels = len(chi_values)
    ncols    = 4
    nrows    = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 4.5, nrows * 3.8),
        gridspec_kw={"hspace": 0.52, "wspace": 0.32,
                     "left": 0.06, "right": 0.98,
                     "top": 0.93, "bottom": 0.06},
    )
    axes_flat = axes.flatten()

    steps = np.arange(N_STEPS + 1)

    # Colour: orientable = green family, non-orientable = red family, sphere = blue
    def _color(name):
        chi = euler_char(name)
        if name == "S^2":           return "#2166ac"
        if name == "RP^2":          return "#e08080"
        if name == "T^2":           return "#80c080"
        if name == "Klein bottle":  return "#c06060"
        m = re.match(r"#\^(\d+)\s*T\^2$",  name)
        if m: return "#2e7d32"
        m = re.match(r"#\^(\d+)\s*RP\^2$", name)
        if m: return "#b71c1c"
        return "#888888"

    for ax_idx, chi in enumerate(chi_values):
        ax   = axes_flat[ax_idx]
        names = by_chi[chi]

        for name in names:
            arr_list = trajs[name]
            mean_traj = np.stack(arr_list).mean(0)
            std_traj  = np.stack(arr_list).std(0)
            c = _color(name)
            ax.plot(steps, mean_traj, color=c, lw=2.0, label=_label(name))
            ax.fill_between(steps, mean_traj - std_traj, mean_traj + std_traj,
                            color=c, alpha=0.18)

            # Theoretical curve for the mean starting V₀
            V0_mean = int(round(np.mean(v0s[name])))
            th = theoretical(V0_mean, chi, N_STEPS)
            ax.plot(steps, th, color=c, lw=1.0, ls=":", alpha=0.9)

        ax.axhline(1 / 3, color="black", lw=1.0, ls="--", alpha=0.5)

        ax.set_title(
            rf"$\chi = {chi}$   " +
            "  /  ".join(_label(n) for n in names),
            fontsize=9, fontweight="bold"
        )
        ax.set_xlabel("# subdivisions", fontsize=8)
        ax.set_ylabel("Mean edge ER", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.grid(alpha=0.2, ls="--")

        # Legend only when there are multiple surface types
        if len(names) > 1:
            ax.legend(fontsize=8, loc="best", framealpha=0.85)

    # Hide unused panels
    for ax_idx in range(len(chi_values), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # Shared legend items
    solid = mlines.Line2D([], [], color="gray", lw=2.0, label="simulated mean ± 1σ")
    dotted = mlines.Line2D([], [], color="gray", lw=1.0, ls=":",
                            label=r"exact formula  $(V_0+t-1)\,/\,3(V_0+t-\chi)$")
    dashed = mlines.Line2D([], [], color="black", lw=1.0, ls="--",
                            alpha=0.5, label="limit = 1/3")
    fig.legend(handles=[solid, dotted, dashed],
               loc="lower center", ncols=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.00), framealpha=0.95)

    fig.suptitle(
        r"Mean edge ER grouped by Euler characteristic  $\chi$"
        "\n"
        r"Surfaces with the same $\chi$ follow the same trajectory — "
        r"solid = simulation, dotted = exact formula $(V_0+t-1)/3(V_0+t-\chi)$",
        fontsize=11, y=0.97,
    )
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=150)
    print(f"\nSaved → {OUTPUT}")


if __name__ == "__main__":
    main()
