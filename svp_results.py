"""Tables and figures of the size-vs-pattern study from a W&B dump.

Input is the CSV written by ``cluster/wandb_dump.py`` for the project
``mantra-pachner-2d-svp`` (one row per run, config and summary columns).
Every finished run belongs to one arm (its first tag, the manifest
name, with the ``svp-``/``svpt-`` prefix dropped, so an evaluation of a
readout-study checkpoint and a fresh training of the same recipe get
the same arm name). Per arm, seeds are aggregated as mean and standard
deviation.

Cells (config/ood/size_vs_pattern.yaml): five vertex counts x three
generators (S = the paper's subdivisions, P-refine = Pachner walk
stopped at the same vertex count, P-mixed = P-refine + 5 flips per
vertex), plus P-flip and P-relabel at the training size.

Outputs, in --out:
    svp_table.csv         one row per arm and cell, mean/std of the
                          balanced accuracy and of the chi-level
                          balanced accuracy, number of seeds
    svp_table.md          the compact view: test, P-relabel, the three
                          generators at graded16 and at bary1
    svp_size_<family>.png accuracy over vertex count, one panel per
                          arm, one line per generator, chi ceiling 7/9

    .venv/bin/python svp_results.py "Claude outputs/wandb_svp.csv" \\
        --out "Claude outputs/svp"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CHI_CEILING = 7 / 9  # 9 classes, two chi-degenerate pairs

# Mean vertex count of every cell (dpattern_size_vs_pattern.csv, one_skeleton d=0).
SIZES = {"graded16": 16.0, "graded20": 20.0, "stellar075": 26.6, "stellar1": 32.1, "bary1": 65.3}
TRAIN_SIZE = 10.0
GENERATORS = {"S": "{}", "P-refine": "prefine_{}", "P-mixed": "pmixed_{}"}
# Fixed categorical order, Okabe-Ito (colour-blind safe).
COLOURS = {"S": "#0072B2", "P-refine": "#E69F00", "P-mixed": "#009E73"}
MARKERS = {"S": "o", "P-refine": "s", "P-mixed": "^"}

METRIC = "balanced_accuracy"
CHI = "chi_balanced_accuracy"


def arm_of(tags: str) -> str:
    manifest = tags.split(";")[0]
    for prefix in ("svpt-", "svp-"):
        if manifest.startswith(prefix):
            return manifest[len(prefix):]
    return manifest


def family_of(arm: str) -> str:
    return arm.split("-")[0]


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
                    "family": family_of(arm),
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
        "(is the predicted class's Euler characteristic right). chi ceiling: 7/9 = 0.778.",
        "",
        "| arm | n | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for arm in sorted(table.arm.unique()):
        n = int(by.loc[(arm, "test"), "n"])
        acc = [fmt(by.loc[(arm, c), "acc_mean"], by.loc[(arm, c), "acc_std"]) for c in cols]
        chi = [fmt(by.loc[(arm, c), "chi_mean"], by.loc[(arm, c), "chi_std"]) for c in cols]
        lines.append(f"| {arm} | {n} | " + " | ".join(acc) + " |")
        if any(c != "–" for c in chi):
            lines.append("| ↳ chi | | " + " | ".join(chi) + " |")
    return "\n".join(lines) + "\n"


def size_figure(table: pd.DataFrame, family: str, path: Path, metric: str):
    arms = sorted(table[table.family == family].arm.unique())
    ncol = min(4, len(arms))
    nrow = -(-len(arms) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), sharex=True, sharey=True, squeeze=False)
    by = table.set_index(["arm", "cell"])
    xs = [TRAIN_SIZE, *SIZES.values()]
    for ax, arm in zip(axes.flat, arms):
        for gen in GENERATORS:
            ys = [by.loc[(arm, "test"), f"{metric}_mean"]] + [by.loc[(arm, f"{gen}@{s}"), f"{metric}_mean"] for s in SIZES]
            es = [by.loc[(arm, "test"), f"{metric}_std"]] + [by.loc[(arm, f"{gen}@{s}"), f"{metric}_std"] for s in SIZES]
            ax.errorbar(xs, ys, yerr=es, color=COLOURS[gen], marker=MARKERS[gen], ms=4, lw=1.5, capsize=2, label=gen)
        rel = by.loc[(arm, "P-relabel"), f"{metric}_mean"]
        ax.plot([TRAIN_SIZE], [rel], marker="x", ms=8, mew=2, color="#444444", ls="none", label="P-relabel")
        ax.axhline(CHI_CEILING, color="#999999", lw=1, ls="--")
        ax.axhline(1 / 9, color="#cccccc", lw=1, ls=":")
        ax.set_title(arm, fontsize=9)
        ax.set_xscale("log")
        ax.set_xlim(8.5, 78)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{x:.0f}" for x in xs], fontsize=7)
        ax.minorticks_off()
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(arms):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("mean vertex count $f_0$", fontsize=8)
    label = "balanced accuracy" if metric == "acc" else "$\\chi$-level balanced accuracy"
    for ax in axes[:, 0]:
        ax.set_ylabel(label, fontsize=8)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"{family}: {label} over size, per generator (dashed: $\\chi$ ceiling 7/9, dotted: chance 1/9)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=150)
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
    for family in sorted(table.family.unique()):
        size_figure(table, family, out / f"svp_size_{family}.png", "acc")
        if table[table.family == family].chi_mean.notna().any():
            size_figure(table, family, out / f"svp_size_{family}_chi.png", "chi")
    print(markdown(table))
    print(f"wrote {out}/svp_table.{{csv,md}} and svp_size_<family>[_chi].png")


if __name__ == "__main__":
    main()
