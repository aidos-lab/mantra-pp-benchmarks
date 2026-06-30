"""
Plot effective-resistance trajectories from experiment_stellar_er.py.

Layout: 2 rows × N_triangulations columns.
  Row 0 — Normalised ER aggregations (each metric divided by its step-0 value,
           so every curve starts at 1.0 and deviations from 1 mean change).
           Solid lines: d-dim simplices.  Dashed lines: (d-1)-dim simplices.
  Row 1 — Raw ratio  Σ ER(new faces) / ER(σ_before).
           A horizontal dashed reference line is drawn at y = d+1,
           the value that would hold if stellar subdivision preserves the
           per-simplex mean ER exactly.

Usage
-----
    python plot_stellar_er.py                          # reads results_stellar_er.csv
    python plot_stellar_er.py --input my_results.csv
    python plot_stellar_er.py --output fig.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ── Palette ───────────────────────────────────────────────────────────────────
_COLORS = {
    "sum":  "tab:blue",
    "mean": "tab:orange",
    "norm": "tab:green",
    "max":  "tab:red",
}
_RATIO_COLOR  = "mediumpurple"
_REF_COLOR    = "black"

_AGG_KEYS = ["sum", "mean", "norm", "max"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    v0 = arr[0]
    return arr / v0 if abs(v0) > 1e-14 else np.ones_like(arr)


def _format_name(name: str) -> str:
    """Light LaTeX formatting for common manifold names."""
    name = (name
            .replace("T^2",  r"$T^2$")
            .replace("S^2",  r"$S^2$")
            .replace("S^3",  r"$S^3$")
            .replace("RP^2", r"$\mathbb{RP}^2$")
            .replace("CP^2", r"$\mathbb{CP}^2$"))
    # #^k X  →  $\#^{k}$ X
    name = re.sub(r"#\^(\d+)\s*(.*)", r"$\#^{\1}$\2", name)
    return name


# ── Main plot ─────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, output: str | None) -> None:
    tri_ids = df["tri_id"].unique()
    n = len(tri_ids)

    fig, axes = plt.subplots(
        2, n,
        figsize=(4.2 * n, 8),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.38},
        squeeze=False,
    )
    top_axes = axes[0]    # normalised ER metrics
    bot_axes = axes[1]    # ratio

    for col, tri_id in enumerate(tri_ids):
        sub = df[df["tri_id"] == tri_id].sort_values("step")
        steps = sub["step"].values
        name  = str(sub["manifold_name"].iloc[0])
        dim   = int(sub["dimension"].iloc[0])

        n0   = int(sub.loc[sub["step"] == 0, "n_top_simplices"].iloc[0])
        v0   = int(sub.loc[sub["step"] == 0, "n_vertices"].iloc[0])

        # ── Row 0: normalised ER aggregations ─────────────────────────────────
        ax = top_axes[col]

        for agg in _AGG_KEYS:
            color = _COLORS[agg]

            col_d   = f"er_d_{agg}"
            col_dm1 = f"er_dm1_{agg}"

            if col_d in sub.columns:
                y = _normalise(sub[col_d].values.astype(float))
                ax.plot(steps, y,
                        color=color, linewidth=2.0, linestyle="solid",
                        label=f"{agg}" if col == 0 else "_")

            if col_dm1 in sub.columns and not sub[col_dm1].isna().all():
                y = _normalise(sub[col_dm1].values.astype(float))
                ax.plot(steps, y,
                        color=color, linewidth=1.5, linestyle="dashed", alpha=0.75,
                        label=f"{agg} (d−1)" if col == 0 else "_")

        ax.axhline(1.0, color=_REF_COLOR, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.set_ylabel("metric / step-0 value", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.2, linestyle="--")

        title_tex = _format_name(name)
        ax.set_title(f"{title_tex}\n"
                     f"({v0} vertices, {n0} top simplices)",
                     fontsize=9.5, fontweight="bold", pad=5)

        # ── Row 1: ratio ──────────────────────────────────────────────────────
        bx = bot_axes[col]

        ref = dim + 1   # expected ratio if per-simplex mean ER is preserved

        rcol = "er_ratio_new_over_subdivided"
        r_valid = np.array([])
        if rcol in sub.columns:
            r = sub[rcol].values.astype(float)
            r_valid = r[1:][~np.isnan(r[1:])]

            # Clip wild outliers (er_sigma_before ≈ 0 edge case)
            hi = np.nanpercentile(r_valid, 98) if r_valid.size else ref + 1
            r_plot = np.where(np.isnan(r), np.nan, np.clip(r, 0, max(hi * 1.1, ref + 1)))

            bx.plot(steps[1:], r_plot[1:],
                    color=_RATIO_COLOR, linewidth=1.8, linestyle="solid",
                    marker="o", markersize=4,
                    label=r"$\Sigma\,\mathrm{ER}_\mathrm{new}\ /\ \mathrm{ER}(\sigma)$"
                    if col == 0 else "_")

        # Reference line at d+1
        bx.axhline(ref, color=_REF_COLOR, linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"d+1 = {ref}  (mean ER preserved)" if col == 0 else "_")

        # Fix y-axis: centre on ref with a sensible margin; disable offset notation
        if r_valid.size:
            span = max(np.nanmax(np.abs(r_valid - ref)), 0.05)
        else:
            span = 0.5
        margin = max(span * 1.4, 0.15)
        bx.set_ylim(ref - margin, ref + margin)
        bx.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        bx.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

        bx.set_xlabel("# stellar subdivisions", fontsize=8.5)
        bx.set_ylabel(r"$\Sigma\,\mathrm{ER}(\mathrm{new})\ /\ \mathrm{ER}(\sigma)$",
                      fontsize=8)
        bx.tick_params(labelsize=8)
        bx.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        bx.grid(True, alpha=0.2, linestyle="--")

    # ── Shared legend for row 0 ───────────────────────────────────────────────
    legend_handles = []
    dim_ex = int(df["dimension"].iloc[0])
    for agg in _AGG_KEYS:
        c = _COLORS[agg]
        legend_handles.append(
            mlines.Line2D([], [], color=c, linewidth=2.0, linestyle="solid",
                          label=f"{agg}  (d={dim_ex})"))
        legend_handles.append(
            mlines.Line2D([], [], color=c, linewidth=1.5, linestyle="dashed", alpha=0.75,
                          label=f"{agg}  (d−1={dim_ex-1})"))
    legend_handles.append(
        mlines.Line2D([], [], color=_REF_COLOR, linewidth=0.8, linestyle=":",
                      label="reference: no change (y = 1)"))

    # Legend for row 1
    legend_handles.append(
        mlines.Line2D([], [], color=_RATIO_COLOR, linewidth=1.8, linestyle="solid",
                      marker="o", markersize=4,
                      label=r"ratio  $\Sigma\,\mathrm{ER}(\mathrm{new\,faces})\ /\ \mathrm{ER}(\sigma_\mathrm{before})$"))
    legend_handles.append(
        mlines.Line2D([], [], color=_REF_COLOR, linewidth=1.2, linestyle="--",
                      label=f"reference: d+1 = {dim_ex+1}  (mean ER preserved)"))

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncols=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.09),
        framealpha=0.96,
    )

    fig.suptitle(
        "Simplicial Effective Resistance under Sequential Stellar Subdivisions",
        fontsize=11.5, y=1.01,
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved → {output}")
    else:
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ER trajectories from experiment_stellar_er.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  default=str(ROOT / "results/results_stellar_er.csv"))
    parser.add_argument("--output", default=None,
                        help="Save to file (PDF/PNG/SVG). Omit to show interactively.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows — "
          f"{df['tri_id'].nunique()} triangulations, "
          f"{df['step'].max()} max steps.")
    plot(df, args.output)


if __name__ == "__main__":
    main()
