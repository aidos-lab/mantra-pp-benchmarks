"""
Visualise aggregate results from the 100-triangulation 2D stellar-subdivision experiment.

Four panels:
  (0,0) Strip plot  — final top-simplex ER value per triangulation, by manifold type.
                      Reveals two-value clustering: T² at 35/36, everything else at 1.
  (0,1) Violin plot — final edge (d-1 simplex) ER distribution per manifold type.
                      Reveals the two-stratum structure: new edges high, original low.
  (1,0) Trajectory  — mean(ER_d) and mean(ER_{d-1}) normalised to step 0, ±1σ band.
                      Grouped and coloured by manifold type.
  (1,1) Ratio       — Σ ER(new faces) / ER(σ_before) per step, ±1σ, reference at d+1=3.
"""

from __future__ import annotations
import json, random, sys, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_stellar_er import compute_er, stellar_subdivide

# ── Config ────────────────────────────────────────────────────────────────────
N          = 100
MAX_STEPS  = 8
SEED       = 0
DATA_PATH  = str(ROOT / "data/mantra/2D/raw/2_manifolds.json")
OUTPUT     = str(ROOT / "results/results_stellar_er_100.png")

# Merge rare types into a single "other" bucket for readability
MIN_COUNT_FOR_OWN_GROUP = 3

# ── Colour palette (type → colour) ───────────────────────────────────────────
TYPE_COLORS = {
    "T^2":         "#e41a1c",   # red  — different ER value
    "Klein bottle":"#377eb8",   # blue
    "#^3 RP^2":    "#4daf4a",   # green
    "#^4 RP^2":    "#984ea3",   # purple
    "#^5 RP^2":    "#ff7f00",   # orange
    "#^6 RP^2":    "#a65628",   # brown
    "other":       "#999999",   # grey
}
DEFAULT_COLOR = "#999999"


def _color(name: str) -> str:
    return TYPE_COLORS.get(name, DEFAULT_COLOR)


def _label(name: str) -> str:
    """Short display label for a manifold name."""
    name = re.sub(r"#\^(\d+)\s*RP\^2", r"$\#^{\1}$RP²", name)
    name = name.replace("T^2", "$T^2$").replace("Klein bottle", "Klein")
    return name


# ── Run experiment, collect data ──────────────────────────────────────────────

def collect(data: list[dict], rng: random.Random):
    """
    Returns
    -------
    step_data : dict[name -> list of per-step dicts]
        Keys per dict: step, er_d_mean_norm, er_dm1_mean_norm, ratio
    final_data : dict[name -> list of (er_d array, er_dm1 array)]
    """
    step_data: dict[str, list] = defaultdict(list)
    final_data: dict[str, list] = defaultdict(list)

    for entry in data:
        d    = int(entry["dimension"])
        name = entry.get("name", "other")
        top  = [frozenset(s) for s in entry["triangulation"]]

        all_verts = {v for s in top for v in s}
        next_v    = max(all_verts) + 1
        pool      = list(top)
        rng.shuffle(pool)
        n_steps   = min(MAX_STEPS, len(pool))

        faces, er_d, er_dm1 = compute_er(top, d)
        er_d0   = float(er_d.mean())
        er_dm10 = float(er_dm1.mean())

        step_data[name].append({"step": 0, "er_d_norm": 1.0,
                                 "er_dm1_norm": 1.0, "ratio": np.nan})

        for step in range(1, n_steps + 1):
            sigma  = pool.pop()
            er_sig = float(er_d[faces[d][sigma]])
            top, new_faces, next_v = stellar_subdivide(top, sigma, next_v)
            faces, er_d, er_dm1    = compute_er(top, d)

            new_er_sum = float(sum(er_d[faces[d][f]] for f in new_faces))
            ratio = new_er_sum / er_sig if er_sig > 1e-14 else np.nan

            step_data[name].append({
                "step":       step,
                "er_d_norm":  float(er_d.mean()) / er_d0,
                "er_dm1_norm": float(er_dm1.mean()) / er_dm10,
                "ratio":      ratio,
            })

        final_data[name].append((er_d.copy(), er_dm1.copy()))

    return step_data, final_data


# ── Aggregate helpers ─────────────────────────────────────────────────────────

def _traj(rows: list[dict], key: str):
    """Per-step mean and std of `key` across a list of per-step row lists."""
    max_step = max(r["step"] for r in rows)
    by_step: dict[int, list] = defaultdict(list)
    for r in rows:
        v = r[key]
        if not np.isnan(v):
            by_step[r["step"]].append(v)
    steps = sorted(by_step)
    means = [np.mean(by_step[s]) for s in steps]
    stds  = [np.std(by_step[s])  for s in steps]
    return np.array(steps), np.array(means), np.array(stds)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(step_data, final_data, type_counts, output: str | None):
    # Merge rare types
    merged_step: dict[str, list] = defaultdict(list)
    merged_final: dict[str, list] = defaultdict(list)
    for name, rows_list in step_data.items():
        key = name if type_counts[name] >= MIN_COUNT_FOR_OWN_GROUP else "other"
        merged_step[key].extend(rows_list)
    for name, fd_list in final_data.items():
        key = name if type_counts[name] >= MIN_COUNT_FOR_OWN_GROUP else "other"
        merged_final[key].extend(fd_list)

    # Order types: T² first (different ER), then by count
    def _order(n):
        if n == "T^2":
            return (0, 0)
        return (1, -type_counts.get(n, 0))

    types = sorted(merged_step.keys(), key=_order)

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={"hspace": 0.42, "wspace": 0.32},
    )
    ax_strip, ax_violin = axes[0]
    ax_traj,  ax_ratio  = axes[1]

    # ── Panel (0,0): strip plot — final top-simplex ER ───────────────────────
    rng_jitter = np.random.default_rng(1)
    xticks, xlabels = [], []
    for xi, name in enumerate(types):
        er_vals = [float(er_d.mean()) for er_d, _ in merged_final[name]]
        jitter  = rng_jitter.uniform(-0.18, 0.18, size=len(er_vals))
        ax_strip.scatter(xi + jitter, er_vals,
                         color=_color(name), alpha=0.7, s=28, linewidths=0)
        xticks.append(xi)
        xlabels.append(_label(name))

    ax_strip.axhline(1.0,     color="black", lw=1.0, ls="--", alpha=0.5,
                     label="ER = 1")
    ax_strip.axhline(35 / 36, color="#e41a1c", lw=1.0, ls=":",  alpha=0.7,
                     label="ER = 35/36")
    ax_strip.set_xticks(xticks)
    ax_strip.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8.5)
    ax_strip.set_ylabel("Final uniform ER of d-simplices", fontsize=9)
    ax_strip.set_title("Top-simplex ER at final step\n(one dot = one triangulation)",
                        fontsize=9.5, fontweight="bold")
    ax_strip.set_ylim(0.955, 1.015)
    ax_strip.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))
    ax_strip.legend(fontsize=8, loc="lower right")
    ax_strip.grid(axis="y", alpha=0.25, ls="--")

    # ── Panel (0,1): violin — final edge ER distribution ─────────────────────
    vdata, vcolors, vpos = [], [], []
    for xi, name in enumerate(types):
        all_edges = np.concatenate([er_dm1 for _, er_dm1 in merged_final[name]])
        vdata.append(all_edges)
        vcolors.append(_color(name))
        vpos.append(xi)

    vp = ax_violin.violinplot(vdata, positions=vpos, widths=0.7,
                               showmedians=True, showextrema=False)
    for body, col in zip(vp["bodies"], vcolors):
        body.set_facecolor(col)
        body.set_alpha(0.6)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.2)

    ax_violin.set_xticks(vpos)
    ax_violin.set_xticklabels(xlabels, rotation=20, ha="right", fontsize=8.5)
    ax_violin.set_ylabel("ER of (d−1)-simplices (edges)", fontsize=9)
    ax_violin.set_title("Edge ER distribution at final step\n"
                         "(violin = KDE; bar = median)",
                         fontsize=9.5, fontweight="bold")
    ax_violin.grid(axis="y", alpha=0.25, ls="--")

    # ── Panel (1,0): normalised ER trajectory ────────────────────────────────
    for name in types:
        rows = merged_step[name]
        steps, m_d, s_d = _traj(rows, "er_d_norm")
        c = _color(name)
        label = _label(name)
        ax_traj.plot(steps, m_d, color=c, lw=2.0, label=f"{label} (d)")
        ax_traj.fill_between(steps, m_d - s_d, m_d + s_d, color=c, alpha=0.12)

        steps, m_dm1, s_dm1 = _traj(rows, "er_dm1_norm")
        ax_traj.plot(steps, m_dm1, color=c, lw=1.4, ls="dashed")
        ax_traj.fill_between(steps, m_dm1 - s_dm1, m_dm1 + s_dm1, color=c, alpha=0.07)

    ax_traj.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.5)
    solid  = mlines.Line2D([], [], color="gray", lw=2.0, ls="solid",  label="d-simplices")
    dashed = mlines.Line2D([], [], color="gray", lw=1.4, ls="dashed", label="(d−1)-simplices")
    handles, labels = ax_traj.get_legend_handles_labels()
    ax_traj.legend(handles + [solid, dashed], labels + ["d-simplices", "(d−1)-simplices"],
                   fontsize=7.5, ncols=2, loc="upper left")
    ax_traj.set_xlabel("# stellar subdivisions", fontsize=9)
    ax_traj.set_ylabel("mean ER / step-0 value", fontsize=9)
    ax_traj.set_title("Normalised mean ER trajectory  (mean ± 1σ across triangulations)\n"
                       "solid = d-simplices · dashed = (d−1)-simplices",
                       fontsize=9.5, fontweight="bold")
    ax_traj.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_traj.grid(alpha=0.2, ls="--")

    # ── Panel (1,1): ratio trajectory ────────────────────────────────────────
    for name in types:
        rows = merged_step[name]
        steps, m_r, s_r = _traj(rows, "ratio")
        c = _color(name)
        ax_ratio.plot(steps, m_r, color=c, lw=2.0, marker="o",
                      markersize=3.5, label=_label(name))
        ax_ratio.fill_between(steps, m_r - s_r, m_r + s_r, color=c, alpha=0.12)

    ax_ratio.axhline(3.0, color="black", lw=1.2, ls="--", alpha=0.7,
                     label="d+1 = 3")
    ax_ratio.legend(fontsize=8, ncols=2)
    ax_ratio.set_xlabel("# stellar subdivisions", fontsize=9)
    ax_ratio.set_ylabel(r"$\Sigma\,\mathrm{ER}(\mathrm{new})\;/\;\mathrm{ER}(\sigma)$",
                         fontsize=9)
    ax_ratio.set_title("Ratio convergence to d+1 = 3  (mean ± 1σ)",
                        fontsize=9.5, fontweight="bold")
    ax_ratio.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax_ratio.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax_ratio.grid(alpha=0.2, ls="--")

    fig.suptitle(
        f"Stellar Subdivision ER — {N} random 2-manifold triangulations, "
        f"{MAX_STEPS} subdivisions each",
        fontsize=11, y=1.01,
    )

    if output:
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"Saved → {output}")
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(DATA_PATH) as f:
        full_data = json.load(f)

    rng = random.Random(SEED)
    data = rng.sample(full_data, N)

    from collections import Counter
    type_counts = Counter(e.get("name", "other") for e in data)

    print(f"Running on {N} triangulations ({MAX_STEPS} steps each)…")
    step_data, final_data = collect(data, rng)
    print("Done. Plotting…")

    plot(step_data, final_data, type_counts, OUTPUT)


if __name__ == "__main__":
    main()
