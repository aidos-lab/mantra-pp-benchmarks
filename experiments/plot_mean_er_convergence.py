"""
Study convergence of mean edge ER under extended stellar subdivision.

Key difference from previous experiments: at each step we pick any current
top simplex uniformly at random (not just original ones), so subdivision
can continue indefinitely — the complex keeps refining.

Runs N_STEPS steps for each triangulation, tracking mean edge ER per step.
Plots trajectory for all 21 manifold types.
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

DATA_PATH  = ROOT / "data" / "mantra" / "2D" / "raw" / "2_manifolds.json"
OUTPUT = ROOT / "results" / "results_mean_er_convergence.png"

N_PER_TYPE = 5
N_STEPS    = 200
SEED       = 42


# ── Type helpers ──────────────────────────────────────────────────────────────

def classify(name: str) -> tuple[str, int]:
    if name == "S^2":          return "sphere", 0
    if name == "T^2":          return "orientable", 1
    if name == "RP^2":         return "nonorientable", 1
    if name == "Klein bottle": return "nonorientable", 2
    m = re.match(r"#\^(\d+)\s*T\^2$",  name)
    if m: return "orientable",    int(m.group(1))
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m: return "nonorientable", int(m.group(1))
    return "other", 99


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


def _family_sorted(names, family):
    return sorted([(classify(n)[1], n) for n in names if classify(n)[0] == family])


def _cmap_colors(kn_list, cmap_name):
    ks = [k for k, _ in kn_list]
    lo, hi = min(ks), max(ks)
    cmap = plt.colormaps[cmap_name]
    return {n: cmap(0.25 + 0.70 * (k - lo) / max(hi - lo, 1)) for k, n in kn_list}


# ── Core experiment ───────────────────────────────────────────────────────────

def mean_edge_er_trajectory(entry: dict, n_steps: int, rng: random.Random) -> np.ndarray:
    """
    Run n_steps stellar subdivisions, picking any current top simplex at random.
    Returns array of mean edge ER at each step (length n_steps + 1).
    """
    top = [frozenset(s) for s in entry["triangulation"]]
    all_verts = {v for s in top for v in s}
    next_v = max(all_verts) + 1

    traj = []

    for step in range(n_steps + 1):
        faces  = build_all_faces(top, 2)
        B1     = build_boundary_matrix(faces[1], faces[0])
        er_dm1 = compute_er_from_boundary(B1)
        traj.append(float(er_dm1.mean()))

        if step < n_steps:
            sigma = rng.choice(top)          # pick ANY current simplex
            top, _, next_v = stellar_subdivide(top, sigma, next_v)

    return np.array(traj)


# ── Data collection ───────────────────────────────────────────────────────────

def collect(data: list, rng: random.Random) -> dict[str, np.ndarray]:
    """Returns {name: 2-D array [n_triangulations, n_steps+1]}."""
    by_name: dict[str, list] = defaultdict(list)
    total = len(data)
    for i, entry in enumerate(data):
        name = entry.get("name", "?")
        print(f"  [{i+1:3d}/{total}] {name:<20s}  "
              f"V={entry['n_vertices']}  T={len(entry['triangulation'])}")
        traj = mean_edge_er_trajectory(entry, N_STEPS, rng)
        by_name[name].append(traj)
    return {k: np.stack(v) for k, v in by_name.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(trajs: dict[str, np.ndarray], output: Path) -> None:
    all_names = list(trajs.keys())
    nonorient = _family_sorted(all_names, "nonorientable")
    orient    = _family_sorted(all_names, "orientable")
    sphere    = _family_sorted(all_names, "sphere")
    sp_color  = "#2166ac"
    no_colors = _cmap_colors(nonorient, "Reds")
    or_colors = _cmap_colors(orient,    "Greens")

    steps = np.arange(N_STEPS + 1)

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5.0),
        gridspec_kw={"wspace": 0.25, "left": 0.07, "right": 0.95,
                     "top": 0.93, "bottom": 0.12},
    )

    def _draw(ax, kn_list, colors, title, legend_title):
        for _, name in sphere:
            arr  = trajs[name]
            m, s = arr.mean(0), arr.std(0)
            ax.plot(steps, m, color=sp_color, lw=2.0, ls="--", zorder=5)
            ax.fill_between(steps, m - s, m + s, color=sp_color, alpha=0.10)

        for k, name in kn_list:
            arr  = trajs[name]
            m, s = arr.mean(0), arr.std(0)
            c    = colors[name]
            ax.plot(steps, m, color=c, lw=1.6)
            ax.fill_between(steps, m - s, m + s, color=c, alpha=0.12)

        ax.axhline(1/3, color="#333333", lw=1.0, ls="--", alpha=0.5)
        ax.text(N_STEPS + 2, 1/3, "1/3", va="center", fontsize=8.5,
                color="#333333", alpha=0.7)

        ax.set_xlabel("stellar subdivisions", fontsize=9.5)
        ax.set_ylabel("mean edge ER", fontsize=9.5)
        ax.set_title(title, fontsize=11, pad=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.15, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

        handles = [mlines.Line2D([], [], color=sp_color, lw=2.0, ls="--",
                                  label=r"$S^2$")]
        handles += [mlines.Line2D([], [], color=colors[n], lw=1.6, label=_label(n))
                    for _, n in kn_list]
        ax.legend(handles=handles, fontsize=7.5, loc="center right",
                  title=legend_title, title_fontsize=8.5,
                  framealpha=0.9, frameon=True, edgecolor="#cccccc", ncols=1)

    _draw(axes[0], nonorient, no_colors,
          r"Non-orientable  $\#^k RP^2$",
          "crosscaps $k$")
    _draw(axes[1], orient, or_colors,
          r"Orientable  $\#^k T^2$",
          "handles $k$")

    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved → {output}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_PATH) as f:
        full_data = json.load(f)

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    rng = random.Random(SEED)
    sample = []
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        sample.extend(chosen)

    print(f"{len(sample)} triangulations × {N_STEPS} steps each.\n")
    trajs = collect(sample, rng)

    print("\nPlotting…")
    plot(trajs, OUTPUT)


if __name__ == "__main__":
    main()
