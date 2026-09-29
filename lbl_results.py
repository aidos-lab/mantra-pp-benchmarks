"""Tables and figures of the labelling study (W&B project mantra-pachner-2d-label).

Input: the CSV of ``cluster/wandb_dump.py`` for that project. Arms are
the manifests of ``cluster/make_lbl_manifests.py`` (first tag, ``lbl-``
dropped): per model a feature condition (mc = moment curve on the
canonical labels, mc-perm = rows permuted every training batch, or a
label-free reference) and, on the complex, the orientation of the
boundary matrices (signed / unsigned). Cells as in the size-vs-pattern
study (config/ood/size_vs_pattern.yaml).

Outputs, in --out:
    lbl_table.csv / .md   mean / std over seeds per arm and cell
    fig_lbl_homp.png      CWN and SCCNN: features x orientation, bars
    fig_lbl_gnn.png       RG-GCN dual / Levi / Hasse and GCN dual, bars
    fig_lbl_heatmap.png   every arm x every cell, diverging around 7/9
    fig_lbl_size.png      accuracy over size for the arms without any
                          labelling channel, one panel per generator

    .venv/bin/python lbl_results.py "Claude outputs/wandb_label.csv" \\
        --out "Claude outputs/lbl"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from svp_results import (
    BAR_CELLS,
    BAR_LABELS,
    CHANCE,
    CHI,
    CHI_CEILING,
    GEN_COLOUR,
    GEN_LABEL,
    GEN_MARKER,
    GENERATORS,
    GRID,
    INK,
    METRIC,
    MUTED,
    SERIES,
    SIZES,
    TRAIN_SIZE,
    cells,
    fmt,
    reference_lines,
    size_axis,
    style,
)

# Colour = feature condition, hatch = unsigned orientation.
FEATURE_COLOUR = {"mc": SERIES[0], "mc-perm": SERIES[1], "ref": SERIES[2], "mean": SERIES[3]}


def arm_of(tags: str) -> str:
    m = tags.split(";")[0]
    return m[4:] if m.startswith("lbl-") else m


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in runs.groupby("arm"):
        for cell, prefix in cells().items():
            acc, chi = group[f"{prefix}/{METRIC}"], group[f"{prefix}/{CHI}"]
            rows.append(
                {"arm": arm, "cell": cell, "n": int(acc.notna().sum()),
                 "acc_mean": acc.mean(), "acc_std": acc.std(ddof=0),
                 "chi_mean": chi.mean(), "chi_std": chi.std(ddof=0)}
            )
    return pd.DataFrame(rows)


ORDER = [
    "cwn-mc-signed", "cwn-mc-perm-signed", "cwn-mc-unsigned", "cwn-mc-perm-unsigned", "cwn-degree-unsigned",
    "sccnn-mc-signed", "sccnn-mc-perm-signed", "sccnn-mc-unsigned", "sccnn-mc-perm-unsigned", "sccnn-degree-unsigned",
    "rggcn-dual-mc", "rggcn-dual-mc-perm", "rggcn-dual-rwpe", "rggcn-dual-random",
    "rggcn-levi-mc", "rggcn-levi-mc-perm", "rggcn-levi-type",
    "rggcn-hasse-mc", "rggcn-hasse-mc-perm", "rggcn-hasse-rank",
    "gcn-dual-mc", "gcn-dual-mc-perm", "gcn-dual-mc-mean",
]


def markdown(table: pd.DataFrame) -> str:
    cols = BAR_CELLS
    by = table.set_index(["arm", "cell"])
    lines = [
        "Balanced accuracy, mean±std over seeds; second line: chi-level balanced accuracy. chi ceiling 7/9 = 0.778.",
        "", "| arm | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols),
    ]
    for arm in [a for a in ORDER if a in set(table.arm)] + sorted(set(table.arm) - set(ORDER)):
        n = int(by.loc[(arm, "test"), "n"])
        lines.append(f"| {arm} | {n} | " + " | ".join(fmt(by.loc[(arm, c), "acc_mean"], by.loc[(arm, c), "acc_std"]) for c in cols) + " |")
        lines.append("| ↳ chi | | " + " | ".join(fmt(by.loc[(arm, c), "chi_mean"], by.loc[(arm, c), "chi_std"]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------


def bars(ax, by, spec, title, seeds=None):
    """spec: list of (arm, label, feature key, hatched); seeds: arm -> DataFrame of runs (dots)."""
    spec = [s for s in spec if s[0] in by.index.get_level_values(0)]
    n = len(spec)
    width = 0.8 / max(n, 1)
    x = np.arange(len(BAR_CELLS))
    for k, (arm, label, feat, hatched) in enumerate(spec):
        ys = [by.loc[(arm, c), "acc_mean"] for c in BAR_CELLS]
        es = [by.loc[(arm, c), "acc_std"] for c in BAR_CELLS]
        ax.bar(x + (k - (n - 1) / 2) * width, ys, width * 0.9, yerr=es, label=label,
               color=FEATURE_COLOUR[feat], hatch="////" if hatched else None, edgecolor="white" if hatched else FEATURE_COLOUR[feat],
               linewidth=0.5, error_kw={"lw": 0.8, "capsize": 1.5, "ecolor": MUTED})
        if seeds is not None and arm in seeds:
            for j, c in enumerate(BAR_CELLS):
                vals = seeds[arm][f"{cells()[c]}/{METRIC}"].values
                ax.plot(np.full(len(vals), x[j] + (k - (n - 1) / 2) * width), vals, ".", color=INK, ms=3.5, zorder=3)
    reference_lines(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(BAR_LABELS, fontsize=7.5)
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.legend(fontsize=7, loc="upper right", ncol=1)


def fig_homp(table, path, seeds):
    by = table.set_index(["arm", "cell"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, (model, name) in zip(axes, [("cwn", "CWN"), ("sccnn", "SCCNN")]):
        bars(ax, by, [
            (f"{model}-mc-signed", "MC canonical, signed (paper)", "mc", False),
            (f"{model}-mc-perm-signed", "MC permuted, signed", "mc-perm", False),
            (f"{model}-mc-unsigned", "MC canonical, unsigned", "mc", True),
            (f"{model}-mc-perm-unsigned", "MC permuted, unsigned", "mc-perm", True),
            (f"{model}-degree-unsigned", "vertex degree, unsigned (label-free)", "ref", True),
        ], name, seeds)
    axes[0].set_ylabel("balanced accuracy")
    fig.suptitle("HOMP: feature channel (colour) x orientation channel (hatched = unsigned boundary matrices); dots = seeds; dashed = $\\chi$ ceiling 7/9",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_gnn(table, path, seeds):
    by = table.set_index(["arm", "cell"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), sharey=True)
    panels = [
        ("RG-GCN on the dual graph", [
            ("rggcn-dual-mc", "MC canonical (paper)", "mc", False),
            ("rggcn-dual-mc-perm", "MC permuted (5 columns)", "mc-perm", False),
            ("rggcn-dual-rwpe", "RWPE-8 (label-free)", "ref", False),
            ("rggcn-dual-random", "random-8, redrawn per access (label-free)", "ref", True),
        ]),
        ("GCN on the dual graph", [
            ("gcn-dual-mc", "MC canonical, sum readout", "mc", False),
            ("gcn-dual-mc-perm", "MC permuted, sum readout", "mc-perm", False),
            ("gcn-dual-mc-mean", "MC canonical, mean readout (paper)", "mean", False),
        ]),
        ("RG-GCN on the Levi graph", [
            ("rggcn-levi-mc", "MC canonical + node type", "mc", False),
            ("rggcn-levi-mc-perm", "MC permuted + node type", "mc-perm", False),
            ("rggcn-levi-type", "node type only (label-free)", "ref", False),
        ]),
        ("RG-GCN on the Hasse diagram", [
            ("rggcn-hasse-mc", "MC canonical + rank", "mc", False),
            ("rggcn-hasse-mc-perm", "MC permuted + rank", "mc-perm", False),
            ("rggcn-hasse-rank", "rank only (label-free)", "ref", False),
        ]),
    ]
    for ax, (title, spec) in zip(axes.flat, panels):
        bars(ax, by, spec, title, seeds)
    for ax in axes[:, 0]:
        ax.set_ylabel("balanced accuracy")
    fig.suptitle("GNNs: canonical vs permuted moment curve vs a label-free encoding, per representation; dashed = $\\chi$ ceiling 7/9",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_heatmap(table, path):
    by = table.set_index(["arm", "cell"])
    arms = [a for a in ORDER if a in set(table.arm)]
    cols = ["test", "P-relabel", "P-flip"] + [f"{g}@{s}" for g in GENERATORS for s in SIZES]
    data = np.array([[by.loc[(a, c), "acc_mean"] for c in cols] for a in arms])
    cmap = LinearSegmentedColormap.from_list("chi", ["#0d366b", "#6da7ec", "#ffffff", "#f5a26a", "#a63a00"])
    norm = TwoSlopeNorm(vmin=CHANCE, vcenter=CHI_CEILING, vmax=1.0)
    fig, ax = plt.subplots(figsize=(11, 0.3 * len(arms) + 2.4))
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cols)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=7, color="white" if abs(v - CHI_CEILING) > 0.3 else INK)
    for x in (2.5, 7.5, 12.5):
        ax.axvline(x, color=INK, lw=1.2)
    blocks = [a.split("-")[0] + a.split("-")[1] * (a.startswith(("rggcn", "gcn"))) for a in arms]
    for i in range(1, len(arms)):
        if blocks[i] != blocks[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["test", "P-relabel", "P-flip"] + [f"{SIZES[s]:.0f}" for _ in GENERATORS for s in SIZES], fontsize=7.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels(arms, fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([1, 5, 10, 15])
    sec.set_xticklabels(["$f_0 = 10$"] + [GEN_LABEL[g] for g in GENERATORS], fontsize=8)
    sec.tick_params(length=0)
    sec.spines["top"].set_visible(False)
    ax.set_xlabel("mean vertex count $f_0$ of the cell", fontsize=8)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", fraction=0.02, pad=0.06, aspect=60)
    cb.set_ticks([CHANCE, 0.4, CHI_CEILING, 0.9, 1.0])
    cb.set_ticklabels(["1/9 chance", ".40", "7/9 = $\\chi$ ceiling", ".90", "1"])
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title("Labelling study: balanced accuracy of every arm on every cell (white = $\\chi$ ceiling)", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


CLEAN = [  # arms with no labelling channel left
    ("cwn-mc-perm-unsigned", "CWN, MC permuted, unsigned"),
    ("sccnn-mc-perm-unsigned", "SCCNN, MC permuted, unsigned"),
    ("rggcn-dual-mc-perm", "RG-GCN dual, MC permuted"),
    ("rggcn-levi-mc-perm", "RG-GCN Levi, MC permuted"),
    ("rggcn-hasse-mc-perm", "RG-GCN Hasse, MC permuted"),
    ("gcn-dual-mc-perm", "GCN dual, MC permuted"),
]


def fig_size(table, path):
    by = table.set_index(["arm", "cell"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    xs = [TRAIN_SIZE, *SIZES.values()]
    for ax, gen in zip(axes, GENERATORS):
        for colour, (arm, label) in zip(SERIES, CLEAN):
            if arm not in by.index.get_level_values(0):
                continue
            ys = [by.loc[(arm, "test"), "acc_mean"]] + [by.loc[(arm, f"{gen}@{s}"), "acc_mean"] for s in SIZES]
            es = [by.loc[(arm, "test"), "acc_std"]] + [by.loc[(arm, f"{gen}@{s}"), "acc_std"] for s in SIZES]
            ax.errorbar(xs, ys, yerr=es, color=colour, marker="o", ms=4, lw=1.8, capsize=2, label=label)
        size_axis(ax)
        ax.set_title(GEN_LABEL[gen], fontsize=9.5)
        ax.set_xlabel("mean vertex count $f_0$")
    axes[0].set_ylabel("balanced accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Size and generator, models without a labelling channel (all trained on $f_0 = 10$, plain cross-entropy)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--out", default="Claude outputs/lbl")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(args.dump)
    runs["arm"] = runs.tags.map(arm_of)
    unfinished = runs[runs.state != "finished"].groupby("arm").state.value_counts()
    if len(unfinished):
        print("not finished (excluded):\n" + unfinished.to_string())
    runs = runs[runs.state == "finished"]

    table = aggregate(runs)
    table.to_csv(out / "lbl_table.csv", index=False)
    (out / "lbl_table.md").write_text(markdown(table))
    style()
    seeds = {arm: grp for arm, grp in runs.groupby("arm")}
    fig_homp(table, out / "fig_lbl_homp.png", seeds)
    fig_gnn(table, out / "fig_lbl_gnn.png", seeds)
    fig_heatmap(table, out / "fig_lbl_heatmap.png")
    fig_size(table, out / "fig_lbl_size.png")
    print(markdown(table))
    print(f"wrote {out}/lbl_table.{{csv,md}}, fig_lbl_{{homp,gnn,heatmap,size}}.png")


if __name__ == "__main__":
    main()
