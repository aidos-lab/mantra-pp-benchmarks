"""Tables and figures of the size-vs-pattern study from a W&B dump.

Input is the CSV written by ``cluster/wandb_dump.py`` for the project
``mantra-pachner-2d-svp`` (one row per run, config and summary columns).
Every finished run belongs to one arm (its first tag, the manifest
name, with the ``svp-``/``svpt-`` prefix dropped, so an evaluation of a
readout-study checkpoint and a fresh training of the same recipe get
the same arm name). Per arm, seeds are aggregated as mean and standard
deviation of the balanced accuracy.

Cells (config/ood/size_vs_pattern.yaml): five vertex counts x three
generators (S = the paper's subdivisions, P-refine = Pachner walk
stopped at the same vertex count, P-mixed = P-refine + 5 flips per
vertex), plus P-flip and P-relabel at the training size.

Outputs, in --out:
    svp_table.csv / .md    one row per arm and cell (csv: mean, std, n,
                           also of the chi-level accuracy where logged)
    fig_labelling.png      test vs P-relabel per arm: how much of the
                           in-distribution accuracy is the labelling
    fig_heatmap.png        every arm x every cell, diverging around the
                           chi ceiling 7/9 (white = exactly chi)
    fig_story.png          accuracy over size for six representative
                           arms, one column per generator
    fig_controls.png       the label-free control arms over size,
                           plain vs walk training, one row per generator

    .venv/bin/python svp_results.py "Claude outputs/wandb_svp.csv" \\
        --out "Claude outputs/svp"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

CHI_CEILING = 7 / 9  # 9 classes, two chi-degenerate pairs
CHANCE = 1 / 9

# Mean vertex count of every cell (dpattern_size_vs_pattern.csv, one_skeleton d=0).
SIZES = {"graded16": 16.0, "graded20": 20.0, "stellar075": 26.6, "stellar1": 32.1, "bary1": 65.3}
TRAIN_SIZE = 10.0
GENERATORS = {"S": "{}", "P-refine": "prefine_{}", "P-mixed": "pmixed_{}"}
GEN_LABEL = {
    "S": "S: paper subdivisions",
    "P-refine": "P-refine: Pachner walk to the same $f_0$",
    "P-mixed": "P-mixed: P-refine + 5 flips per vertex",
}
# Fixed categorical order; both palettes pass the CVD/contrast checks.
GEN_COLOUR = {"S": "#0072B2", "P-refine": "#D55E00", "P-mixed": "#009E73"}
GEN_MARKER = {"S": "o", "P-refine": "s", "P-mixed": "^"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#d9d8d0"

METRIC = "balanced_accuracy"
CHI = "chi_balanced_accuracy"

FAMILIES = ["baseline", "consistency", "augment", "ctl"]
FAMILY_LABEL = {
    "baseline": "baseline (plain training, MC features)",
    "consistency": "consistency loss on Pachner walks",
    "augment": "plain loss on Pachner walks",
    "ctl": "label-free controls",
}


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def arm_of(tags: str) -> str:
    manifest = tags.split(";")[0]
    for prefix in ("svpt-", "svp-"):
        if manifest.startswith(prefix):
            return manifest[len(prefix):]
    return manifest


def describe(arm: str) -> dict:
    """Family, model, representation and options of an arm, from its name."""
    family = arm.split("-")[0]
    if family == "ctl":
        m = re.fullmatch(r"ctl-(.+?)(-mlp)?-(plain|walks)", arm)
        feature, mlp, training = m.groups()
        model = "MLP" if mlp else ("CWN" if feature.startswith("cwn") else "GCN")
        return {"family": family, "model": model, "feature": feature, "training": training, "options": ""}
    m = re.fullmatch(r"(\w+)-(cwn|sccnn|gcn|rggcn)(-levi)?-2d-unb-(sum|altsum)((?:-\w+)*)", arm)
    family, model, levi, readout, options = m.groups()
    return {
        "family": family,
        "model": model.upper().replace("RGGCN", "RG-GCN") + (" Levi" if levi else ""),
        "readout": readout,
        "options": options.strip("-"),
    }


def short(arm: str) -> str:
    """A readable arm label."""
    d = describe(arm)
    if d["family"] == "ctl":
        return f"{d['feature']} · {d['model']} · {d['training']}"
    label = d["model"] + (" alt-sum" if d["readout"] == "altsum" else "")
    return label + (f" · {d['options'].replace('-', ' ')}" if d["options"] else "")


def order(arms) -> list[str]:
    """Family order, then model, then options, so the same models sit together."""
    return sorted(
        arms,
        key=lambda a: (
            FAMILIES.index(describe(a)["family"]),
            describe(a).get("feature", ""),
            describe(a)["model"],
            describe(a).get("training", ""),
            describe(a).get("readout", ""),
            describe(a)["options"],
        ),
    )


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def cells() -> dict[str, str]:
    """display name -> W&B metric prefix."""
    out = {"test": "test", "P-relabel": "ood_prelabel", "P-flip": "ood_pflip"}
    for size in SIZES:
        for gen, pattern in GENERATORS.items():
            out[f"{gen}@{size}"] = "ood_" + pattern.format(size)
    return out


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, group in runs.groupby("arm"):
        for cell, prefix in cells().items():
            acc = group[f"{prefix}/{METRIC}"]
            chi = group[f"{prefix}/{CHI}"]
            rows.append(
                {
                    "arm": arm,
                    "family": describe(arm)["family"],
                    "cell": cell,
                    "n": int(acc.notna().sum()),
                    "acc_mean": acc.mean(),
                    "acc_std": acc.std(ddof=0),
                    "chi_mean": chi.mean(),
                    "chi_std": chi.std(ddof=0),
                }
            )
    return pd.DataFrame(rows)


def fmt(mean, std):
    return "–" if pd.isna(mean) else f"{mean:.2f}±{std:.2f}"


def markdown(table: pd.DataFrame) -> str:
    cols = ["test", "P-relabel", "S@graded16", "P-refine@graded16", "P-mixed@graded16", "S@bary1", "P-mixed@bary1"]
    by = table.set_index(["arm", "cell"])
    lines = [
        "Balanced accuracy, mean±std over seeds; second line per arm: chi-level balanced accuracy "
        "(is the predicted class's Euler characteristic right; logged only by runs after f06c731). "
        "chi ceiling: 7/9 = 0.778.",
        "",
        "| arm | n | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for arm in order(table.arm.unique()):
        n = int(by.loc[(arm, "test"), "n"])
        acc = [fmt(by.loc[(arm, c), "acc_mean"], by.loc[(arm, c), "acc_std"]) for c in cols]
        chi = [fmt(by.loc[(arm, c), "chi_mean"], by.loc[(arm, c), "chi_std"]) for c in cols]
        lines.append(f"| {arm} | {n} | " + " | ".join(acc) + " |")
        if any(c != "–" for c in chi):
            lines.append("| ↳ chi | | " + " | ".join(chi) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def style():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def reference_lines(ax, axis="y"):
    line = ax.axhline if axis == "y" else ax.axvline
    line(CHI_CEILING, color=MUTED, lw=1, ls="--", zorder=0)
    line(CHANCE, color=GRID, lw=1, ls=":", zorder=0)


def family_bands(ax, arms, axis="y"):
    """Shade alternate families and label them; arms are in plot order."""
    fams = [describe(a)["family"] for a in arms]
    start = 0
    for i in range(1, len(fams) + 1):
        if i == len(fams) or fams[i] != fams[start]:
            if (FAMILIES.index(fams[start]) % 2) == 1:
                span = ax.axhspan if axis == "y" else ax.axvspan
                span(start - 0.5, i - 0.5, color="#f2f1ec", zorder=-1)
            start = i


def fig_labelling(table, path):
    """Dumbbell per arm: test (filled) -> P-relabel (hollow)."""
    by = table.set_index(["arm", "cell"])
    arms = order(table.arm.unique())[::-1]  # top-down in family order
    fig, ax = plt.subplots(figsize=(7.5, 0.24 * len(arms) + 1.6))
    ys = np.arange(len(arms))
    for y, arm in zip(ys, arms):
        t, r = by.loc[(arm, "test"), "acc_mean"], by.loc[(arm, "P-relabel"), "acc_mean"]
        ax.plot([r, t], [y, y], color=GRID, lw=2, zorder=1)
        ax.plot(t, y, "o", color=SERIES[0], ms=6, zorder=2)
        ax.plot(r, y, "o", mfc="white", mec=SERIES[1], mew=1.8, ms=6, zorder=3)
    family_bands(ax, arms)
    reference_lines(ax, axis="x")
    ax.set_yticks(ys)
    ax.set_yticklabels([short(a) for a in arms], fontsize=7.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("balanced accuracy at the training size ($f_0 = 10$)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    # family labels on the right
    for fam in FAMILIES:
        idx = [i for i, a in enumerate(arms) if describe(a)["family"] == fam]
        if idx:
            ax.text(1.03, np.mean(idx), FAMILY_LABEL[fam], fontsize=7.5, color=MUTED, va="center", rotation=90)
    ax.plot([], [], "o", color=SERIES[0], label="test split (canonical labels)")
    ax.plot([], [], "o", mfc="white", mec=SERIES[1], mew=1.8, label="P-relabel: same complexes, random vertex labels")
    ax.plot([], [], ls="--", color=MUTED, label="$\\chi$ ceiling 7/9")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("How much of the in-distribution accuracy is the vertex labelling?", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_heatmap(table, path):
    """Arms x cells; diverging colour around the chi ceiling."""
    by = table.set_index(["arm", "cell"])
    arms = order(table.arm.unique())
    cols = ["test", "P-relabel", "P-flip"] + [f"{g}@{s}" for g in GENERATORS for s in SIZES]
    data = np.array([[by.loc[(a, c), "acc_mean"] for c in cols] for a in arms])
    cmap = LinearSegmentedColormap.from_list("chi", ["#0d366b", "#6da7ec", "#ffffff", "#f5a26a", "#a63a00"])
    norm = TwoSlopeNorm(vmin=CHANCE, vcenter=CHI_CEILING, vmax=1.0)
    fig, ax = plt.subplots(figsize=(11, 0.26 * len(arms) + 2.2))
    ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cols)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}"[1:], ha="center", va="center", fontsize=6.5, color="white" if abs(v - CHI_CEILING) > 0.3 else INK)
    # block separators: after the 3 training-size cells and between generators
    for x in (2.5, 7.5, 12.5):
        ax.axvline(x, color=INK, lw=1.2)
    fams = [describe(a)["family"] for a in arms]
    for i in range(1, len(arms)):
        if fams[i] != fams[i - 1]:
            ax.axhline(i - 0.5, color=INK, lw=1.2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["test", "P-relabel", "P-flip"] + [f"{SIZES[s]:.0f}" for _ in GENERATORS for s in SIZES], fontsize=7.5)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([short(a) for a in arms], fontsize=7.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # generator headers above the blocks
    sec = ax.secondary_xaxis("top")
    sec.set_xticks([1, 5, 10, 15])
    sec.set_xticklabels(["$f_0 = 10$"] + [GEN_LABEL[g] for g in GENERATORS], fontsize=8)
    sec.tick_params(length=0)
    sec.spines["top"].set_visible(False)
    for fam in FAMILIES:
        idx = [i for i, f in enumerate(fams) if f == fam]
        if idx:
            ax.text(len(cols) - 0.3, np.mean(idx), FAMILY_LABEL[fam], fontsize=7.5, color=MUTED, va="center", rotation=270)
    ax.set_xlabel("mean vertex count $f_0$ of the cell", fontsize=8)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation="horizontal", fraction=0.012, pad=0.035, aspect=60
    )
    cb.set_ticks([CHANCE, 0.4, CHI_CEILING, 0.9, 1.0])
    cb.set_ticklabels(["1/9 chance", ".40", "7/9 = $\\chi$ ceiling", ".90", "1"])
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title(
        "Balanced accuracy of every arm on every cell (white = exactly the $\\chi$ ceiling; warm = above it, cool = below)",
        loc="left",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


STORY = [
    ("baseline-cwn-2d-unb-altsum", "CWN, MC features (paper setup)"),
    ("baseline-cwn-2d-unb-altsum-perm", "CWN, MC features, labels permuted per batch"),
    ("consistency-cwn-2d-unb-altsum", "CWN, consistency loss on walks"),
    ("consistency-cwn-2d-unb-altsum-refine", "CWN, consistency loss on refining walks"),
    ("ctl-cwn-degree-unsigned-walks", "CWN, degree only, unsigned, plain loss on walks"),
    ("ctl-skel-degree-mlp-walks", "DeepSets on vertex degrees, plain loss on walks"),
]


def size_curve(by, arm, gen):
    xs = [TRAIN_SIZE, *SIZES.values()]
    ys = [by.loc[(arm, "test"), "acc_mean"]] + [by.loc[(arm, f"{gen}@{s}"), "acc_mean"] for s in SIZES]
    es = [by.loc[(arm, "test"), "acc_std"]] + [by.loc[(arm, f"{gen}@{s}"), "acc_std"] for s in SIZES]
    return xs, ys, es


def size_axis(ax):
    xs = [TRAIN_SIZE, *SIZES.values()]
    ax.set_xscale("log")
    ax.set_xlim(8.5, 78)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:.0f}" for x in xs], fontsize=7.5)
    ax.minorticks_off()
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, lw=0.6)
    reference_lines(ax)


def fig_story(table, path):
    by = table.set_index(["arm", "cell"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, gen in zip(axes, GENERATORS):
        for colour, (arm, label) in zip(SERIES, STORY):
            xs, ys, es = size_curve(by, arm, gen)
            ax.errorbar(xs, ys, yerr=es, color=colour, marker="o", ms=4, lw=1.8, capsize=2, label=label)
        size_axis(ax)
        ax.set_title(GEN_LABEL[gen], fontsize=9.5)
        ax.set_xlabel("mean vertex count $f_0$")
    axes[0].set_ylabel("balanced accuracy")
    axes[2].text(76, CHI_CEILING - 0.045, "$\\chi$ ceiling 7/9", fontsize=7.5, color=MUTED, ha="right")
    axes[2].text(76, CHANCE + 0.02, "chance 1/9", fontsize=7.5, color=MUTED, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Accuracy over size: the same six models under the three generators (trained on $f_0 \\leq 10$, walks reach $f_0 \\leq 16$)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


CTL_FEATURES = {
    "skel-degree": ("1-skeleton · degree", SERIES[0]),
    "dual-rwpe": ("dual graph · RWPE", SERIES[1]),
    "dual-degree": ("dual graph · degree", SERIES[4]),
    "levi-type": ("Levi graph · node type", SERIES[2]),
    "hasse-rank": ("Hasse diagram · rank", SERIES[3]),
    "cwn-degree": ("complex · degree, signed", SERIES[5]),
    "cwn-degree-unsigned": ("complex · degree, unsigned", "#5c5c5c"),
}
CTL_MODEL_STYLE = {"GCN": "-", "CWN": "-", "MLP": "--"}


def fig_controls(table, path):
    by = table.set_index(["arm", "cell"])
    arms = [a for a in table.arm.unique() if a.startswith("ctl-")]
    fig, axes = plt.subplots(3, 2, figsize=(9.5, 9), sharex=True, sharey=True)
    for i, gen in enumerate(GENERATORS):
        for j, training in enumerate(("plain", "walks")):
            ax = axes[i, j]
            for arm in arms:
                d = describe(arm)
                if d["training"] != training:
                    continue
                label, colour = CTL_FEATURES[d["feature"]]
                xs, ys, _ = size_curve(by, arm, gen)
                ax.plot(xs, ys, color=colour, ls=CTL_MODEL_STYLE[d["model"]], marker="o", ms=3.5, lw=1.6,
                        label=f"{label} · {d['model']}")
            size_axis(ax)
            if i == 0:
                ax.set_title(
                    {"plain": "trained on the plain split ($f_0 = 10$)", "walks": "trained on Pachner walks ($f_0 \\leq 16$)"}[training],
                    fontsize=9.5,
                )
            if j == 0:
                ax.set_ylabel(f"{gen}\nbalanced accuracy")
            if i == 2:
                ax.set_xlabel("mean vertex count $f_0$")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Label-free control arms (no positional features): rows = generator, columns = training set; dashed = DeepSets (MLP) on the same features", fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dump")
    p.add_argument("--out", default="Claude outputs/svp")
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
    table.to_csv(out / "svp_table.csv", index=False)
    (out / "svp_table.md").write_text(markdown(table))
    style()
    fig_labelling(table, out / "fig_labelling.png")
    fig_heatmap(table, out / "fig_heatmap.png")
    fig_story(table, out / "fig_story.png")
    fig_controls(table, out / "fig_controls.png")
    print(markdown(table))
    print(f"wrote {out}/svp_table.{{csv,md}}, fig_labelling/heatmap/story/controls.png")


if __name__ == "__main__":
    main()
