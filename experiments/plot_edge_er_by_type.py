"""
Show the full edge-ER distribution at the last step, stratified by all 21
manifold types.  This recovers the two-cluster structure (new vs old edges)
and lets us see whether the bimodal shape / cluster positions vary with the
number of connected summands.

Layout (2 rows):
  Top   : 21 violin plots, one per type, ordered by family and connected-sum
           count.  Each violin is the concatenation of ALL edge ER values from
           ALL triangulations of that type at the final step.
  Bottom: Two scatter subplots (orientable / non-orientable), one dot per
           type.  x = connected-sum count k, y = mean edge ER (median of the
           upper cluster).  Shows how ER scales with genus / crosscap number.

Usage
-----
    uv run --extra cpu python plot_edge_er_by_type.py
"""

from __future__ import annotations
import json, re, random, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiment_stellar_er import (
    build_all_faces, build_boundary_matrix,
    compute_er_from_boundary, stellar_subdivide,
)

DATA_PATH  = str(ROOT / "data/mantra/2D/raw/2_manifolds.json")
N_PER_TYPE = 5
MAX_STEPS  = 10
SEED       = 42
OUTPUT     = str(ROOT / "results/results_edge_er_by_type.png")


# ── Type helpers ──────────────────────────────────────────────────────────────

def classify(name: str) -> tuple[str, int]:
    if name == "S^2":
        return ("sphere", 0)
    m = re.match(r"#\^(\d+)\s*T\^2$", name)
    if m:
        return ("orientable", int(m.group(1)))
    if name == "T^2":
        return ("orientable", 1)
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m:
        return ("nonorientable", int(m.group(1)))
    if name == "RP^2":
        return ("nonorientable", 1)
    if name == "Klein bottle":
        return ("nonorientable", 2)
    return ("other", 99)


def _label(name: str) -> str:
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
        return rf"$\#{{{m.group(1)}}}T^2$"
    m = re.match(r"#\^(\d+)\s*RP\^2$", name)
    if m:
        return rf"$\#{{{m.group(1)}}}RP^2$"
    return name


def type_color(name: str, all_nonorient: list, all_orient: list) -> str:
    family, _ = classify(name)
    if family == "sphere":
        return "#2166ac"
    if family == "orientable":
        g = plt.cm.Greens(np.linspace(0.35, 0.9, max(len(all_orient), 1)))
        idx = next((i for i, t in enumerate(all_orient) if t == name), 0)
        return tuple(g[idx])
    if family == "nonorientable":
        r = plt.cm.Reds(np.linspace(0.25, 0.9, max(len(all_nonorient), 1)))
        idx = next((i for i, t in enumerate(all_nonorient) if t == name), 0)
        return tuple(r[idx])
    return "#999999"


# ── Compute full edge-ER distributions ───────────────────────────────────────

def final_edge_er(entry: dict, rng: random.Random) -> np.ndarray:
    """Run MAX_STEPS stellar subdivisions and return full edge-ER array at the end."""
    d   = int(entry["dimension"])
    top = [frozenset(s) for s in entry["triangulation"]]
    all_verts = {v for s in top for v in s}
    next_v = max(all_verts) + 1
    pool   = list(top)
    rng.shuffle(pool)
    n_steps = min(MAX_STEPS, len(pool))

    for _ in range(n_steps):
        sigma = pool.pop()
        top, _, next_v = stellar_subdivide(top, sigma, next_v)

    faces = build_all_faces(top, d)
    B1    = build_boundary_matrix(faces[1], faces[0])
    return compute_er_from_boundary(B1)


def collect_distributions(data: list, rng: random.Random) -> dict[str, np.ndarray]:
    """Returns {manifold_name: array of all edge ER values at the final step}."""
    result: dict[str, list] = defaultdict(list)
    for entry in data:
        name  = entry.get("name", "?")
        er_dm1 = final_edge_er(entry, rng)
        result[name].extend(er_dm1.tolist())
    return {k: np.array(v) for k, v in result.items()}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(edge_dists: dict[str, np.ndarray], output: str | None) -> None:
    all_types = sorted(edge_dists.keys(), key=lambda n: (classify(n)[0], classify(n)[1]))
    nonorient_types = [t for t in all_types if classify(t)[0] == "nonorientable"]
    orient_types    = [t for t in all_types if classify(t)[0] == "orientable"]
    sphere_types    = [t for t in all_types if classify(t)[0] == "sphere"]

    colors = {n: type_color(n, nonorient_types, orient_types) for n in all_types}

    # Sort for display: non-orientable first (more of them), then orientable, then sphere
    display_order = (
        sorted(nonorient_types, key=lambda n: classify(n)[1]) +
        sorted(orient_types,    key=lambda n: classify(n)[1]) +
        sphere_types
    )

    fig = plt.figure(figsize=(18, 11))
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[1.7, 1],
        hspace=0.50, wspace=0.35,
        left=0.06, right=0.98, top=0.93, bottom=0.12,
    )
    ax_viol  = fig.add_subplot(gs[0, :])        # full-width violins
    ax_sc_no = fig.add_subplot(gs[1, 0])        # scatter: non-orientable
    ax_sc_or = fig.add_subplot(gs[1, 1])        # scatter: orientable
    ax_ratio = fig.add_subplot(gs[1, 2])        # histogram overlay: cluster gap

    # ── Row 0: 21 violin plots ───────────────────────────────────────────────
    positions = list(range(len(display_order)))
    vdata     = [edge_dists[n] for n in display_order]
    clrs      = [colors[n] for n in display_order]

    vp = ax_viol.violinplot(
        vdata, positions=positions, widths=0.75,
        showmedians=True, showextrema=False,
    )
    for body, col in zip(vp["bodies"], clrs):
        body.set_facecolor(col)
        body.set_alpha(0.72)
        body.set_edgecolor("none")
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(1.3)

    ax_viol.set_xticks(positions)
    ax_viol.set_xticklabels([_label(n) for n in display_order],
                             rotation=35, ha="right", fontsize=8)
    ax_viol.set_ylabel("Edge effective resistance", fontsize=10)
    ax_viol.set_title(
        f"Full edge-ER distribution at final step (after {MAX_STEPS} stellar subdivisions), "
        "by manifold type",
        fontsize=10.5, fontweight="bold",
    )
    ax_viol.grid(axis="y", alpha=0.2, ls="--")

    # Separator lines between families
    sep1 = len(nonorient_types) - 0.5
    sep2 = len(nonorient_types) + len(orient_types) - 0.5
    for sx in [sep1, sep2]:
        ax_viol.axvline(sx, color="black", lw=0.7, ls=":", alpha=0.4)

    # Family labels above
    xc_no = (len(nonorient_types) - 1) / 2
    xc_or = len(nonorient_types) + (len(orient_types) - 1) / 2
    xc_sp = len(nonorient_types) + len(orient_types)
    ymax  = ax_viol.get_ylim()[1]
    for xc, txt, c in [
        (xc_no, "Non-orientable (RP² family)", "#d73027"),
        (xc_or, "Orientable (tori family)",    "#4dac26"),
        (xc_sp, "Sphere",                      "#2166ac"),
    ]:
        ax_viol.text(xc, ymax * 1.02, txt, ha="center", va="bottom",
                     fontsize=8, color=c, fontweight="bold",
                     transform=ax_viol.get_xaxis_transform())

    # ── Row 1 left: scatter — non-orientable ─────────────────────────────────
    def _upper_median(arr: np.ndarray) -> float:
        """Median of the upper half (above the overall median) — tracks the high cluster."""
        med = np.median(arr)
        upper = arr[arr > med]
        return float(np.median(upper)) if upper.size else float(med)

    def _lower_median(arr: np.ndarray) -> float:
        med = np.median(arr)
        lower = arr[arr <= med]
        return float(np.median(lower)) if lower.size else float(med)

    for name in nonorient_types:
        _, k = classify(name)
        arr = edge_dists[name]
        ax_sc_no.scatter(k, _upper_median(arr), color=colors[name],
                         s=80, zorder=3, marker="^", label=_label(name))
        ax_sc_no.scatter(k, _lower_median(arr), color=colors[name],
                         s=80, zorder=3, marker="v")
        # Connect the two clusters with a thin line
        ax_sc_no.plot([k, k], [_lower_median(arr), _upper_median(arr)],
                      color=colors[name], lw=1.0, alpha=0.5)

    ax_sc_no.set_xlabel("Connected-sum count  k  (crosscaps)", fontsize=9)
    ax_sc_no.set_ylabel("Cluster median edge ER\n(▲ upper  /  ▼ lower)", fontsize=9)
    ax_sc_no.set_title("Non-orientable: $\\#^k RP^2$\ncluster positions vs k",
                        fontsize=9.5, fontweight="bold")
    ax_sc_no.grid(alpha=0.2, ls="--")

    for name in orient_types:
        _, k = classify(name)
        arr = edge_dists[name]
        ax_sc_or.scatter(k, _upper_median(arr), color=colors[name],
                         s=80, zorder=3, marker="^", label=_label(name))
        ax_sc_or.scatter(k, _lower_median(arr), color=colors[name],
                         s=80, zorder=3, marker="v")
        ax_sc_or.plot([k, k], [_lower_median(arr), _upper_median(arr)],
                      color=colors[name], lw=1.0, alpha=0.5)

    ax_sc_or.set_xlabel("Connected-sum count  k  (handles)", fontsize=9)
    ax_sc_or.set_ylabel("Cluster median edge ER\n(▲ upper  /  ▼ lower)", fontsize=9)
    ax_sc_or.set_title("Orientable: $\\#^k T^2$\ncluster positions vs k",
                        fontsize=9.5, fontweight="bold")
    ax_sc_or.grid(alpha=0.2, ls="--")

    # ── Row 1 right: cluster gap by type ─────────────────────────────────────
    # Show the gap (upper – lower cluster median) for each type vs k
    # This directly answers: does the bimodal gap grow with k?
    for name in nonorient_types:
        _, k = classify(name)
        arr = edge_dists[name]
        gap = _upper_median(arr) - _lower_median(arr)
        ax_ratio.scatter(k, gap, color=colors[name], s=80, marker="s", zorder=3)

    for name in orient_types:
        _, k = classify(name)
        arr = edge_dists[name]
        gap = _upper_median(arr) - _lower_median(arr)
        ax_ratio.scatter(k, gap, color=colors[name], s=80, marker="D", zorder=3)

    for name in sphere_types:
        arr = edge_dists[name]
        gap = _upper_median(arr) - _lower_median(arr)
        ax_ratio.scatter(0, gap, color=colors[name], s=80, marker="o", zorder=3)

    no_patch = mpatches.Patch(color="#d73027", label="Non-orient. ■")
    or_patch = mpatches.Patch(color="#4dac26", label="Orientable  ◆")
    sp_patch = mpatches.Patch(color="#2166ac", label="Sphere      ●")
    ax_ratio.legend(handles=[no_patch, or_patch, sp_patch], fontsize=8)
    ax_ratio.set_xlabel("Connected-sum count k  (0 = sphere)", fontsize=9)
    ax_ratio.set_ylabel("Upper − lower cluster gap", fontsize=9)
    ax_ratio.set_title("Bimodal cluster gap  vs  k\n(does subdivision gap grow with genus?)",
                        fontsize=9.5, fontweight="bold")
    ax_ratio.grid(alpha=0.2, ls="--")

    fig.suptitle(
        "Edge ER two-cluster structure — stratified by all 21 manifold types",
        fontsize=12, y=0.98,
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

    by_type: dict[str, list] = defaultdict(list)
    for e in full_data:
        by_type[e.get("name", "?")].append(e)

    rng = random.Random(SEED)
    sample: list[dict] = []
    for name, entries in sorted(by_type.items()):
        chosen = sorted(entries, key=lambda e: e["n_vertices"])[:N_PER_TYPE]
        sample.extend(chosen)
        print(f"  {name:<20s}  {len(chosen)} triangulations")

    print(f"\nTotal: {len(sample)} triangulations, {MAX_STEPS} subdivision steps each.")
    print("Computing final edge-ER distributions…")

    edge_dists = collect_distributions(sample, rng)

    print("Plotting…")
    plot(edge_dists, OUTPUT)


if __name__ == "__main__":
    main()
