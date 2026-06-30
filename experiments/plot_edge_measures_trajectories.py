"""
Reproduce the 3-panel edge-ratio trajectory plot for four distinct edge-ER measures:
  1. Mean of edge ER
  2. Max of edge ER
  3. L2 norm of edge ER  (computed from saved mean/std/n_edges)
  4. Σ ER(new edges after) / Σ ER(boundary edges before)  [edge_ratio]

Layout: 4 rows × 3 columns.
  Col 0 — Non-orientable (#^k RP²) trajectories, coloured light→dark by k.
  Col 1 — Orientable (#^k T²) trajectories, coloured light→dark by k.
  Col 2 — Final value vs k scatter for both families.

Reads from results/results_stellar_er_types.csv.
"""

from __future__ import annotations
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "results" / "results_stellar_er_types.csv"
OUTPUT = ROOT / "results" / "results_edge_measures_trajectories.png"

MEASURES = [
    ("er_dm1_mean", "Mean edge ER"),
    ("er_dm1_max",  "Max edge ER"),
    ("er_dm1_l2",   "L2 norm of edge ER"),
    ("edge_ratio",  r"$\Sigma$ ER(new) / $\Sigma$ ER(boundary before)"),
]


# ── Type helpers ──────────────────────────────────────────────────────────────

def classify(name: str) -> tuple[str, int]:
    if name == "S^2":         return "sphere", 0
    if name == "T^2":         return "orientable", 1
    if name == "RP^2":        return "nonorientable", 1
    if name == "Klein bottle": return "nonorientable", 2
    m = re.match(r"#\^(\d+)\s*T\^2$", name);
    if m: return "orientable", int(m.group(1))
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


def _family_sorted(df, family):
    names = df["manifold_name"].unique()
    return sorted([(classify(n)[1], n) for n in names if classify(n)[0] == family])


def _cmap_colors(kn_list, cmap_name):
    ks = [k for k, _ in kn_list]
    lo, hi = min(ks), max(ks)
    cmap = plt.colormaps[cmap_name]
    return {n: cmap(0.25 + 0.70 * (k - lo) / max(hi - lo, 1)) for k, n in kn_list}


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_traj(ax, df, kn_list, colors, col, sphere_kn, sp_color, show_legend, legend_title):
    """Draw mean ± std trajectories for one family + optional sphere overlay."""
    for k, name in kn_list:
        sub  = df[(df["manifold_name"] == name) & (df["step"] >= 1) & df[col].notna()]
        if sub.empty:
            continue
        grp   = sub.groupby("step")[col]
        steps = np.array(sorted(grp.groups))
        means = np.array([grp.get_group(s).mean() for s in steps])
        stds  = np.array([grp.get_group(s).std(ddof=0) for s in steps])
        c = colors[name]
        ax.plot(steps, means, color=c, lw=1.8, label=_label(name))
        ax.fill_between(steps, means - stds, means + stds, color=c, alpha=0.15)

    # Sphere overlay (dashed blue)
    for k, name in sphere_kn:
        sub  = df[(df["manifold_name"] == name) & (df["step"] >= 1) & df[col].notna()]
        if sub.empty:
            continue
        grp   = sub.groupby("step")[col]
        steps = np.array(sorted(grp.groups))
        means = np.array([grp.get_group(s).mean() for s in steps])
        stds  = np.array([grp.get_group(s).std(ddof=0) for s in steps])
        ax.plot(steps, means, color=sp_color, lw=2.0, ls="--", zorder=5,
                label=r"$S^2$")
        ax.fill_between(steps, means - stds, means + stds,
                        color=sp_color, alpha=0.13)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2, ls="--")

    if show_legend:
        handles = [mlines.Line2D([], [], color=sp_color, lw=2.0, ls="--",
                                  label=r"$S^2$ (dashed)")]
        handles += [mlines.Line2D([], [], color=colors[n], lw=1.8, label=_label(n))
                    for _, n in kn_list]
        ax.legend(handles=handles, fontsize=6.5, loc="upper left",
                  title=legend_title, title_fontsize=7.5,
                  framealpha=0.85, ncols=1)


def _draw_scatter(ax, df, nonorient, no_colors, orient, or_colors, sphere, sp_color, col):
    """Scatter of final value vs k for all families."""
    final = df.loc[df.groupby("tri_id")["step"].idxmax()]

    for k, name in nonorient:
        vals = final[final["manifold_name"] == name][col].dropna()
        if vals.empty: continue
        ax.scatter([k] * len(vals), vals, color=no_colors[name],
                   s=40, alpha=0.75, marker="o", zorder=3)
        ax.errorbar(k, vals.mean(), yerr=vals.std(ddof=0),
                    color=no_colors[name], fmt="none", capsize=3, lw=1.2)

    for k, name in orient:
        vals = final[final["manifold_name"] == name][col].dropna()
        if vals.empty: continue
        ax.scatter([k] * len(vals), vals, color=or_colors[name],
                   s=40, alpha=0.75, marker="D", zorder=3)
        ax.errorbar(k, vals.mean(), yerr=vals.std(ddof=0),
                    color=or_colors[name], fmt="none", capsize=3, lw=1.2)

    for k, name in sphere:
        vals = final[final["manifold_name"] == name][col].dropna()
        if vals.empty: continue
        ax.scatter([k] * len(vals), vals, color=sp_color,
                   s=60, alpha=0.85, marker="*", zorder=4)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2, ls="--")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT)

    # Derive L2 norm: sqrt(n_edges * (std² + mean²))
    df["er_dm1_l2"] = np.sqrt(
        df["n_edges"] * (df["er_dm1_std"] ** 2 + df["er_dm1_mean"] ** 2)
    )

    nonorient = _family_sorted(df, "nonorientable")
    orient    = _family_sorted(df, "orientable")
    sphere    = _family_sorted(df, "sphere")
    sp_color  = "#2166ac"

    no_colors = _cmap_colors(nonorient, "Reds")
    or_colors = _cmap_colors(orient,    "Greens")

    n_measures = len(MEASURES)
    fig, axes = plt.subplots(
        n_measures, 3,
        figsize=(15, 4.2 * n_measures),
        gridspec_kw={"wspace": 0.30, "hspace": 0.48,
                     "left": 0.07, "right": 0.97,
                     "top": 0.95, "bottom": 0.05},
    )

    col_titles = [
        r"Non-orientable  $\#^k RP^2$  (light $\to$ dark = $k$ ↑)",
        r"Orientable  $\#^k T^2$  (light $\to$ dark = $k$ ↑)",
        "Final value vs  k",
    ]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight="bold", pad=8)

    for row, (col, measure_label) in enumerate(MEASURES):
        ax_no, ax_or, ax_sc = axes[row]

        _draw_traj(ax_no, df, nonorient, no_colors, col,
                   sphere, sp_color,
                   show_legend=(row == 0),
                   legend_title="k = crosscaps")

        _draw_traj(ax_or, df, orient, or_colors, col,
                   sphere, sp_color,
                   show_legend=(row == 0),
                   legend_title="k = handles")

        _draw_scatter(ax_sc, df, nonorient, no_colors, orient, or_colors,
                      sphere, sp_color, col)

        # Row y-labels
        ax_no.set_ylabel(measure_label, fontsize=9)

        # x-labels only on bottom row
        if row == n_measures - 1:
            for ax in (ax_no, ax_or):
                ax.set_xlabel("# stellar subdivisions", fontsize=9)
            ax_sc.set_xlabel("Connected-sum count  k", fontsize=9)

    # Shared legend for scatter column (bottom-right area)
    no_h = mlines.Line2D([], [], color="#c0392b", lw=0, marker="o",
                          markersize=6, label=r"Non-orient. $\#^k RP^2$")
    or_h = mlines.Line2D([], [], color="#27ae60", lw=0, marker="D",
                          markersize=6, label=r"Orientable $\#^k T^2$")
    sp_h = mlines.Line2D([], [], color=sp_color, lw=0, marker="*",
                          markersize=8, label=r"Sphere $S^2$")
    axes[-1, 2].legend(handles=[no_h, or_h, sp_h], fontsize=8, loc="upper left")

    fig.suptitle(
        "Edge ER measures under sequential stellar subdivisions — all 21 manifold types",
        fontsize=12, y=0.98,
    )
    fig.savefig(OUTPUT, bbox_inches="tight", dpi=150)
    print(f"Saved → {OUTPUT}")


if __name__ == "__main__":
    main()
