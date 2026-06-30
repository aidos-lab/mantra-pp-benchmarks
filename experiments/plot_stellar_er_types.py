"""
Visualise stellar-subdivision ER experiment across all 21 2D manifold types.

New edge-based measures (replacing the trivial Σ ER(triangles)/ER(σ) ratio):
  edge_ratio     = Σ ER(new edges {v_i, b}) / Σ ER(boundary edges of σ before)
  boundary_change = Σ ER(boundary edges after) / Σ ER(boundary edges before)

Layout (2×2):
  (0,0) Strip plot  — final d-simplex ER per triangulation, by type.
                      Confirms uniform convergence; value is type-specific.
  (0,1) Final edge statistics — violin of edge ER at last step, by type family.
  (1,0) edge_ratio trajectory  — does the new-edge ER / boundary-before ratio converge?
  (1,1) boundary_change trajectory — how much does subdivision alter boundary-edge ER?

Usage
-----
    uv run python plot_stellar_er_types.py --input results_stellar_er_types.csv
"""

from __future__ import annotations
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
import numpy as np
import pandas as pd

# ── Manifold type classification ───────────────────────────────────────────────

def classify(name: str) -> tuple[str, int]:
    """
    Returns (family, order) where family in {"sphere","orientable","nonorientable"}
    and order is the genus / crosscap count (for ordering within family).
    """
    if name == "S^2":
        return ("sphere", 0)
    m = re.match(r"#\^(\d+)\s*T\^2", name)
    if m:
        return ("orientable", int(m.group(1)))
    if name == "T^2":
        return ("orientable", 1)
    m = re.match(r"#\^(\d+)\s*RP\^2", name)
    if m:
        return ("nonorientable", int(m.group(1)))
    if name == "RP^2":
        return ("nonorientable", 1)
    if name == "Klein bottle":
        return ("nonorientable", 2)
    return ("other", 99)


def _label(name: str) -> str:
    """Return a short display label for a manifold name."""
    if name == "Klein bottle":
        return "Klein"
    if name == "S^2":
        return r"$S^2$"
    if name == "T^2":
        return r"$T^2$"
    if name == "RP^2":
        return r"$RP^2$"
    m = re.match(r"#\^(\d+)\s*T\^2$", name)
    if m:
        k = m.group(1)
        return rf"$\#{{{k}}}T^2$"
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m:
        k = m.group(1)
        return rf"$\#{{{k}}}RP^2$"
    return name


def type_color(name: str, all_types: list[str]) -> str:
    """
    Colour scheme:
      S²                     → steel blue
      Orientable (T²family)  → green gradient by genus (1–8)
      Non-orientable         → red/orange gradient by crosscap count (1–17)
    """
    family, order = classify(name)
    if family == "sphere":
        return "#2166ac"   # deep blue

    orientable_types = sorted(
        [t for t in all_types if classify(t)[0] == "orientable"],
        key=lambda t: classify(t)[1]
    )
    nonorient_types = sorted(
        [t for t in all_types if classify(t)[0] == "nonorientable"],
        key=lambda t: classify(t)[1]
    )

    if family == "orientable":
        # green gradient: light → dark
        greens = plt.cm.Greens(np.linspace(0.35, 0.9, max(len(orientable_types), 1)))
        idx = next((i for i, t in enumerate(orientable_types) if t == name), 0)
        return tuple(greens[idx])  # type: ignore

    if family == "nonorientable":
        # red/orange gradient
        reds = plt.cm.Reds(np.linspace(0.25, 0.9, max(len(nonorient_types), 1)))
        idx = next((i for i, t in enumerate(nonorient_types) if t == name), 0)
        return tuple(reds[idx])  # type: ignore

    return "#999999"


# ── Trajectory helper ──────────────────────────────────────────────────────────

def _traj(df: pd.DataFrame, col: str, name: str):
    sub = df[(df["manifold_name"] == name) & df[col].notna()]
    if sub.empty:
        return np.array([]), np.array([]), np.array([])
    by_step = sub.groupby("step")[col]
    steps = np.array(sorted(by_step.groups))
    means = np.array([by_step.get_group(s).mean() for s in steps])
    stds  = np.array([by_step.get_group(s).std(ddof=0) for s in steps])
    return steps, means, stds


# ── Main plot ──────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, output: str | None) -> None:
    # Sort types by family then order
    all_types = sorted(df["manifold_name"].unique(), key=lambda n: (classify(n)[0], classify(n)[1]))
    colors = {n: type_color(n, list(df["manifold_name"].unique())) for n in all_types}

    # For each triangulation, take the last available step
    final = df.loc[df.groupby("tri_id")["step"].idxmax()]

    fig, axes = plt.subplots(
        2, 2, figsize=(15, 10),
        gridspec_kw={"hspace": 0.44, "wspace": 0.30},
    )
    ax_strip, ax_violin = axes[0]
    ax_ratio, ax_change = axes[1]

    # ── (0,0): strip plot — final d-simplex ER ───────────────────────────────
    rng_jitter = np.random.default_rng(7)
    xticks, xlabels = [], []
    for xi, name in enumerate(all_types):
        rows = final[final["manifold_name"] == name]
        vals = rows["er_d_mean"].values
        jitter = rng_jitter.uniform(-0.22, 0.22, size=len(vals))
        ax_strip.scatter(xi + jitter, vals, color=colors[name],
                         s=38, alpha=0.85, linewidths=0)
        xticks.append(xi)
        xlabels.append(_label(name))

    ax_strip.axhline(1.0,      color="black", lw=1.0, ls="--", alpha=0.4, label="ER = 1.0")
    ax_strip.axhline(35 / 36,  color="#888888", lw=1.0, ls=":",  alpha=0.6, label="35/36 ($T^2$)")
    ax_strip.set_xticks(xticks)
    ax_strip.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7.5)
    ax_strip.set_ylabel("Final uniform ER of triangles", fontsize=9)
    ax_strip.set_title("Top-simplex ER at last step  (one dot = one triangulation)",
                        fontsize=9.5, fontweight="bold")
    ax_strip.legend(fontsize=8, loc="lower right")
    ax_strip.grid(axis="y", alpha=0.2, ls="--")
    # Dynamic y-limits
    ymin = final["er_d_mean"].min()
    ymax = final["er_d_mean"].max()
    margin = max((ymax - ymin) * 0.4, 0.005)
    ax_strip.set_ylim(ymin - margin, ymax + margin)
    ax_strip.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # ── (0,1): violin — final edge ER distribution by family ─────────────────
    #   Group into 3 families to avoid 21-violin crowding
    fam_data = {"sphere": [], "orientable\n(tori)": [], "non-orientable\n(proj. planes)": []}
    fam_colors = {"sphere": "#2166ac", "orientable\n(tori)": "#4dac26",
                  "non-orientable\n(proj. planes)": "#d73027"}

    for name in all_types:
        rows = final[final["manifold_name"] == name]
        tri_ids = rows["tri_id"].values
        family, _ = classify(name)
        fam_key = {"sphere": "sphere", "orientable": "orientable\n(tori)",
                   "nonorientable": "non-orientable\n(proj. planes)"}.get(family, "sphere")
        # Use last-step edge ER mean per triangulation (we don't store full distribution)
        fam_data[fam_key].extend(rows["er_dm1_mean"].values.tolist())

    fam_keys = list(fam_data.keys())
    vdata = [np.array(fam_data[k]) for k in fam_keys]
    vdata = [v[~np.isnan(v)] for v in vdata]
    vpositions = list(range(len(fam_keys)))

    vp = ax_violin.violinplot(vdata, positions=vpositions, widths=0.65,
                               showmedians=True, showextrema=True)
    for body, key in zip(vp["bodies"], fam_keys):
        body.set_facecolor(fam_colors[key])
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.5)
    vp["cbars"].set_color("black")
    vp["cmins"].set_color("black")
    vp["cmaxes"].set_color("black")

    ax_violin.set_xticks(vpositions)
    ax_violin.set_xticklabels(fam_keys, fontsize=9)
    ax_violin.set_ylabel("Mean edge ER at last step", fontsize=9)
    ax_violin.set_title("Edge ER distribution at last step\nby manifold family",
                         fontsize=9.5, fontweight="bold")
    ax_violin.grid(axis="y", alpha=0.2, ls="--")

    # ── (1,0): edge_ratio trajectory ─────────────────────────────────────────
    _plot_traj(ax_ratio, df, all_types, colors, "edge_ratio",
               title="New-edge ratio  =  Σ ER(new edges) / Σ ER(boundary edges before)",
               ylabel="edge ratio",
               refval=None)

    # ── (1,1): boundary_change trajectory ────────────────────────────────────
    _plot_traj(ax_change, df, all_types, colors, "boundary_change",
               title="Boundary-edge change  =  Σ ER(boundary after) / Σ ER(boundary before)",
               ylabel="ΣER(boundary) after / before",
               refval=1.0)

    # ── Legend for bottom panels ──────────────────────────────────────────────
    sphere_patch    = mpatches.Patch(color="#2166ac", label="$S^2$ (sphere)")
    orientable_patch = mpatches.Patch(color="#4dac26", label="Orientable surface (tori family)")
    nonorient_patch  = mpatches.Patch(color="#d73027", label="Non-orientable surface (RP² family)")
    # Add type-specific thin lines for detail
    type_handles = []
    for name in all_types:
        family, order = classify(name)
        lbl = _label(name)
        type_handles.append(mlines.Line2D([], [], color=colors[name], lw=1.5, label=lbl))

    fig.legend(
        handles=[sphere_patch, orientable_patch, nonorient_patch] + type_handles,
        loc="lower center",
        ncols=7,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.09),
        framealpha=0.96,
        title="Manifold types  (colour = family, shade = genus/crosscap count)",
        title_fontsize=8,
    )

    fig.suptitle(
        "Stellar Subdivision — Simplicial ER across all 21 2D Manifold Types",
        fontsize=12, y=1.01,
    )

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved → {output}")
    else:
        plt.show()


def _plot_traj(ax, df, all_types, colors, col, title, ylabel, refval):
    for name in all_types:
        steps, means, stds = _traj(df, col, name)
        if steps.size == 0:
            continue
        c = colors[name]
        ax.plot(steps, means, color=c, lw=1.8, label=_label(name))
        ax.fill_between(steps, means - stds, means + stds, color=c, alpha=0.12)

    if refval is not None:
        ax.axhline(refval, color="black", lw=1.0, ls="--", alpha=0.5,
                   label=f"reference = {refval}")

    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("# stellar subdivisions", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.2, ls="--")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input",  default=str(ROOT / "results/results_stellar_er_types.csv"))
    parser.add_argument("--output", default=str(ROOT / "results/results_stellar_er_types.png"))
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows — "
          f"{df['tri_id'].nunique()} triangulations, "
          f"{df['manifold_name'].nunique()} types.")
    plot(df, args.output)


if __name__ == "__main__":
    main()
