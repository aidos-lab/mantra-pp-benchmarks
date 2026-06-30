"""
Plot how  edge_ratio = Σ ER(new edges) / Σ ER(boundary edges before)  evolves
with the number of stellar subdivisions, for all 21 2D manifold types.

Layout:
  Left  — Trajectory lines for non-orientable (#^k RP²) family, k = 1…17.
           Colour encodes k: light pink (k=1) → dark red (k=17).
  Centre — Same for orientable (#^k T²) family, k = 1…8.
           Colour encodes k: light green → dark green.
  Right  — Final edge_ratio vs k for both families on one axes,
           showing the monotone relationship.

Reads from results/results_stellar_er_types.csv (already computed).

Usage
-----
    uv run python experiments/plot_edge_ratio_trajectories.py
"""

from __future__ import annotations
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import matplotlib.cm as cm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT  = ROOT / "results" / "results_stellar_er_types.csv"
OUTPUT = ROOT / "results" / "results_edge_ratio_trajectories.png"


# ── Type helpers ──────────────────────────────────────────────────────────────

def classify(name: str) -> tuple[str, int]:
    if name == "S^2":
        return "sphere", 0
    if name == "T^2":
        return "orientable", 1
    if name == "RP^2":
        return "nonorientable", 1
    if name == "Klein bottle":
        return "nonorientable", 2
    m = re.match(r"#\^(\d+)\s*T\^2$", name)
    if m:
        return "orientable", int(m.group(1))
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m:
        return "nonorientable", int(m.group(1))
    return "other", 99


def _label(name: str) -> str:
    if name == "S^2":    return r"$S^2$"
    if name == "T^2":    return r"$T^2$"
    if name == "RP^2":   return r"$RP^2$"
    if name == "Klein bottle": return "Klein"
    m = re.match(r"#\^(\d+)\s*T\^2$", name)
    if m:  return rf"$\#{{{m.group(1)}}}T^2$"
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m:  return rf"$\#{{{m.group(1)}}}RP^2$"
    return name


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, output: Path) -> None:
    all_types = sorted(df["manifold_name"].unique(), key=lambda n: classify(n))

    nonorient = sorted([(classify(n)[1], n) for n in all_types
                        if classify(n)[0] == "nonorientable"])
    orient    = sorted([(classify(n)[1], n) for n in all_types
                        if classify(n)[0] == "orientable"])
    sphere    = [(classify(n)[1], n) for n in all_types if classify(n)[0] == "sphere"]

    # Colour maps: lightness encodes k within each family
    def _cmap_colors(kn_list: list, cmap_name: str) -> dict:
        ks = [k for k, _ in kn_list]
        lo, hi = min(ks), max(ks)
        cmap = plt.colormaps[cmap_name]
        return {n: cmap(0.25 + 0.70 * (k - lo) / max(hi - lo, 1))
                for k, n in kn_list}

    no_colors = _cmap_colors(nonorient, "Reds")
    or_colors = _cmap_colors(orient,    "Greens")
    sp_color  = "#2166ac"

    fig, axes = plt.subplots(
        1, 3, figsize=(15, 5.5),
        gridspec_kw={"wspace": 0.32, "left": 0.06, "right": 0.97,
                     "top": 0.88, "bottom": 0.13},
    )
    ax_no, ax_or, ax_sc = axes

    def _draw_traj(ax, kn_list, colors, title, family_label, ref_color):
        for k, name in kn_list:
            sub = df[df["manifold_name"] == name]
            grp = sub.groupby("step")["edge_ratio"]
            steps = np.array(sorted(grp.groups))
            steps = steps[steps >= 1]              # skip step 0 (NaN)
            means = np.array([grp.get_group(s).mean() for s in steps])
            stds  = np.array([grp.get_group(s).std(ddof=0)  for s in steps])
            c = colors[name]
            ax.plot(steps, means, color=c, lw=2.0, label=_label(name))
            ax.fill_between(steps, means - stds, means + stds, color=c, alpha=0.15)

        ax.set_xlabel("# stellar subdivisions", fontsize=10)
        ax.set_ylabel(r"$\Sigma\,\mathrm{ER}(\mathrm{new\;edges})\ /\ \Sigma\,\mathrm{ER}(\mathrm{boundary\;before})$",
                      fontsize=9)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(alpha=0.2, ls="--")

        # Colourbar-style legend: type labels, ordered by k
        handles = [mlines.Line2D([], [], color=colors[n], lw=2, label=_label(n))
                   for _, n in kn_list]
        ax.legend(handles=handles, fontsize=7.5, loc="upper left",
                  title=family_label, title_fontsize=8,
                  framealpha=0.85, ncols=1)

    _draw_traj(ax_no, nonorient, no_colors,
               "Non-orientable surfaces  $\\#^k RP^2$\n(k = 1 … 17, light → dark)",
               "type (k = crosscaps)", "#d73027")

    # Add S² on the non-orientable panel for comparison (it's the simplest case)
    for k, name in sphere:
        sub  = df[df["manifold_name"] == name]
        grp  = sub.groupby("step")["edge_ratio"]
        stps = np.array(sorted(grp.groups))
        stps = stps[stps >= 1]
        mns  = np.array([grp.get_group(s).mean() for s in stps])
        stds = np.array([grp.get_group(s).std(ddof=0) for s in stps])
        ax_no.plot(stps, mns, color=sp_color, lw=2.2, ls="--",
                   label=_label(name), zorder=5)
        ax_no.fill_between(stps, mns - stds, mns + stds, color=sp_color, alpha=0.13)
        # Prepend S² to the legend
        extra = mlines.Line2D([], [], color=sp_color, lw=2.2, ls="--",
                              label=r"$S^2$ (sphere, dashed)")
        handles, labels = ax_no.get_legend_handles_labels()
        ax_no.legend([extra] + handles[:-1], [extra.get_label()] + labels[:-1],
                     fontsize=7.5, loc="upper left",
                     title="type (k = crosscaps)", title_fontsize=8,
                     framealpha=0.85, ncols=1)

    _draw_traj(ax_or, orient, or_colors,
               "Orientable surfaces  $\\#^k T^2$\n(k = 1 … 8, light → dark)",
               "type (k = handles)", "#4dac26")

    # ── Right panel: final value vs k ─────────────────────────────────────────
    final = df.loc[df.groupby("tri_id")["step"].idxmax()]

    for k, name in nonorient:
        rows = final[final["manifold_name"] == name]
        vals = rows["edge_ratio"].dropna()
        ax_sc.scatter([k] * len(vals), vals,
                      color=no_colors[name], s=55, alpha=0.75,
                      marker="o", zorder=3)
        ax_sc.errorbar(k, vals.mean(), yerr=vals.std(ddof=0),
                       color=no_colors[name], fmt="none", capsize=4, lw=1.5)

    for k, name in orient:
        rows = final[final["manifold_name"] == name]
        vals = rows["edge_ratio"].dropna()
        ax_sc.scatter([k] * len(vals), vals,
                      color=or_colors[name], s=55, alpha=0.75,
                      marker="D", zorder=3)
        ax_sc.errorbar(k, vals.mean(), yerr=vals.std(ddof=0),
                       color=or_colors[name], fmt="none", capsize=4, lw=1.5)

    for k, name in sphere:
        rows = final[final["manifold_name"] == name]
        vals = rows["edge_ratio"].dropna()
        ax_sc.scatter([k] * len(vals), vals,
                      color=sp_color, s=65, alpha=0.8,
                      marker="*", zorder=4)

    no_patch = mlines.Line2D([], [], color="#c0392b", lw=0, marker="o",
                              markersize=7, label=r"Non-orient. $\#^k RP^2$")
    or_patch = mlines.Line2D([], [], color="#27ae60", lw=0, marker="D",
                              markersize=7, label=r"Orientable $\#^k T^2$")
    sp_patch = mlines.Line2D([], [], color=sp_color,  lw=0, marker="*",
                              markersize=9, label=r"Sphere $S^2$ (k=0)")
    ax_sc.legend(handles=[no_patch, or_patch, sp_patch], fontsize=8.5)
    ax_sc.set_xlabel("Connected-sum count  k", fontsize=10)
    ax_sc.set_ylabel("Final edge ratio", fontsize=10)
    ax_sc.set_title("Final edge ratio vs k\n(dots = individual triangulations)",
                    fontsize=10.5, fontweight="bold")
    ax_sc.grid(alpha=0.2, ls="--")

    fig.suptitle(
        r"Edge ratio  $\Sigma\,\mathrm{ER}(\mathrm{new\;edges})\ /\ "
        r"\Sigma\,\mathrm{ER}(\mathrm{boundary\;edges\;before})$"
        "  across all 21 manifold types",
        fontsize=11.5,
    )

    fig.savefig(output, bbox_inches="tight", dpi=150)
    print(f"Saved → {output}")


def main():
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rows, {df['manifold_name'].nunique()} types.")
    plot(df, OUTPUT)


if __name__ == "__main__":
    main()
