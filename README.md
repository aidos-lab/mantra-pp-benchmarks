#  No Triangulation Without Representation: Generalization in Topological Deep Learning

Code accompanying the paper *No Triangulation Without Representation:
Generalization in Topological Deep Learning*.

This repository contains:

- Implementation of Pachner moves, rebalancing and duplication checks of topological datasets
- Training and evaluation pipeline for graph- and simplicial / cell-complex
  models on the (extended) MANTRA manifold-triangulation classification benchmarks.
- Out-of-distribution evaluation on subdivided triangulations (full
  barycentric, *stellar*, and *graded*); the subdivision machinery itself
  lives in the MANTRA library, which serves the subdivided test set as the
  `ood` split of `MantraDataset`.

## MANTRA dependency

The `mantra-dataset` dependency is the upstream MANTRA dataset library
(<https://github.com/aidos-lab/mantra>) and is installed as an editable
package from a sibling checkout expected at `../mantra` (see
`pyproject.toml` / `uv.lock`).

## Repository layout

```
.
├── main.py                      # Hydra entry point: train + test a model
├── evaluate_barycentric.py      # Load a checkpoint, evaluate on a subdivided cache
├── config/                      # Hydra config groups (model, dataset, transform, task, …)
├── src/
│   ├── models/                  # GCN, MLP, ResidualGatedGCN, Graphormer, SAN, SAN-TMX, CWN, SCCNN
│   ├── utils/                   # data loading, pair sampler, train loop, barycentric helpers
│   ├── metrics/                 # training objectives, task and embedding metrics
│   └── schema.py                # OmegaConf dataclasses for Hydra structured configs
└── tests/                       # pytest suite (math utils, transforms, models, losses, …)
```

## Installation

The project is built with [`uv`](https://docs.astral.sh/uv/) and pinned via
`uv.lock`. There are two mutually-exclusive optional dependency sets,
`cpu` and `cu130` (CUDA 13.0 wheels for `torch` / `torch-geometric` /
`pyg-lib` / `torch-sparse` / `torch-scatter`).

### CPU install (Linux or macOS)

```bash
uv sync --extra cpu
```

### GPU install (Linux + CUDA 13.0)

```bash
uv sync --extra cu130
```

This will:

1. Create `.venv/` with Python ≥ 3.12 and install all pinned dependencies.
2. Install the MANTRA library from the sibling checkout `../mantra` as
   an editable `mantra-dataset` package, so all `from mantra.…` imports
   resolve to that checkout.
3. Install this repository as `mantra-pp-benchmarks==0.1.0`.

All commands below assume `.venv/bin/python` (or `uv run python …`).

> **zsh users:** the bracketed Hydra list overrides
> (e.g. `dataset.transform_order=[representation,…]`) need to be quoted —
> `'dataset.transform_order=[representation,embedding,select_features]'`
> — because zsh treats `[…]` as a glob. In bash they work either way.

### Datasets

The repository does **not** ship the raw or subdivided MANTRA JSONs.
There are three dataset variants used in the paper, all derived from the
upstream MANTRA release:

1. **Original (unbalanced) MANTRA.** The 2- and 3-manifold JSONs are
   auto-downloaded from the upstream MANTRA release the first time you
   instantiate `ManifoldTriangulations(root=..., balanced=False)`.
   After one run the file is on disk at
   `${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds.json` and is
   reused in subsequent runs (see [Running a single training run](#running-a-single-training-run)).
2. **Balanced MANTRA.** Produced by Pachner-move augmentation +
   deduplication, applied on the fly by the MANTRA library
   (`MantraDataset(balanced=True)`; see
   [Generating the balanced dataset](#generating-the-balanced-dataset)).
3. **Subdivided MANTRA.** Full-barycentric / stellar / graded
   subdivisions of the test split, served by the MANTRA library as the
   `ood` split (see
   [Out-of-distribution evaluation](#out-of-distribution-evaluation-on-subdivided-triangulations)).
4. **Pachner walks.** Every entry of the in-distribution splits expanded
   into a random Pachner walk (see
   [Pachner walks](#pachner-walks-and-the-pachner-objectives)).


### Generating the balanced dataset

The balanced variant is produced by Pachner-move augmentation +
deduplication on the unbalanced JSON. The balancing logic lives in the
MANTRA library (`mantra/utils/balancing.py` upstream) and is applied on
the fly by `MantraDataset(balanced=True)`.

A reduced version can be generated and run locally on a CPU in a reasonable time; the full version is best run on a cluster. The unbalanced source JSON has to be on disk first — trigger the
auto-download by running `main.py` once with `dataset.balanced=False`,
or load it explicitly via `ManifoldTriangulations(root='data', balanced=False)`.

Enable it through the dataset config, e.g. for a quick CPU run:

```bash
.venv/bin/python main.py dataset.balanced=True dataset.target_count=200
```

The relevant knobs (`target_count`, `n_moves`, `max_vertices`, and the
split seed) live in `config/dataset/dataset.yaml` and are forwarded to
`MantraDataset`. The paper runs use `dataset.target_count=2500` (2D) and
`dataset.target_count=5000` (3D).

#### Loading a locally generated balanced JSON in training

A JSON that is *already* balanced is loaded like any other local file,
with `dataset.balanced=False` — the flag asks for balancing to be done
now, not for the file to be declared balanced:

```bash
.venv/bin/python main.py \
    dataset.balanced=False \
    dataset.local_path=/tmp/2_manifolds_balanced_demo.json \
    dataset.root=data/2d_mantra_balanced \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features] \
    task=name \
    trainer.device=cpu trainer.max_epoch=2
```

The first run copies the JSON into
`${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds.json` and
materializes the processed cache. Subsequent runs reuse the cache; pass
`dataset.reload=True` to force re-processing.

Caches of `dataset.balanced=True` built before the fix in the MANTRA
library (`MantraDataset(balanced=True)` used to discard the balanced
entries) hold unbalanced data; rebuild them with `dataset.reload=True`.

## Running a single training run

`main.py` is a Hydra app with the default composition defined in
`config/config.yaml`. A minimal CPU run on the unbalanced 2-manifolds
(JSON is auto-downloaded into `data/mantra/2D/raw/` on first invocation):

```bash
.venv/bin/python main.py \
    dataset.balanced=False \
    dataset.root=data \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features] \
    task=name \
    trainer.device=cpu trainer.max_epoch=2
```

The task config carries its metrics (`accuracy` and the macro-averaged
`balanced_accuracy`), so no `+task/metrics@...` override is needed.

For the balanced variant, generate the JSON first (see
[Generating the balanced dataset](#generating-the-balanced-dataset))
and then add the `dataset.balanced=True` /
`dataset.local_path=<path-to-generated-json>` overrides.

Key Hydra groups:

| Group                                         | Options                                                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `model`                                       | `gcn`, `mlp`, `residual_gated_gcn`, `graphormer`, `san`, `san_tmx`, `cwn`, `sccnn`                  |
| `representation@dataset.transform.representation` | `dual_graph`, `one_skeleton`, `hasse_diagram`, `incidence_sc`, `adjacency_sc`, `up_/down_/hodge_laplacian_sc` |
| `transform@dataset.transform.embedding`       | `random_walk_pe`, `moment_curve_embedding`, `random_features`, `degree`, `effective_resistance`, `er_stats` |
| `task`                                        | `name`, `orientability`, `betti_numbers`                                                            |
| `loss`                                        | `cross_entropy`, `pachner_consistency`, `pachner_pairwise`, `pachner_infonce`                       |
| `trainer`                                     | `default`                                                                                           |
| `logger`                                      | `wandb`                                                                                             |

Outputs:

- The best validation checkpoint and the resolved Hydra config are written
  to `${checkpoint_path}/<wandb_run_name>/{best_val.pt, config.yaml}`.
- Train / val / test metrics and a confusion matrix are logged to
  Weights & Biases under `${logger.entity}/${logger.project}`.
- With `dataset.division_type != none` the subdivided `ood` split is
  built and evaluated in the same run, logged under the `ood/` prefix.
  The subdivision arguments go into `dataset.subdivision`, e.g.
  `dataset.division_type=graded +dataset.subdivision.graded_vertex_number=16`,
  `dataset.division_type=stellar +dataset.subdivision.fraction=0.75` or
  `dataset.division_type=barycentric +dataset.subdivision.round=1`. The
  `+` is needed because `dataset.subdivision` starts out empty and the
  configs are in struct mode.
  This replaces the three-step `precompute_barycentric.py` /
  `evaluate_barycentric.py` pipeline for a fresh run; that pipeline is
  still the way to evaluate an *existing* checkpoint.

To turn off W&B sync, run with `WANDB_MODE=offline`. To override the
project name use `logger.project=<name>` (and `logger.entity=<entity>`).

## Out-of-distribution evaluation on subdivided triangulations

The subdivided test set is the `ood` split of `MantraDataset`: the
MANTRA library subdivides the entries of the test split, so no separate
JSON has to be generated. Add the subdivision to any training run and it
is built, evaluated and logged (prefix `ood/`) in the same job:

```bash
.venv/bin/python main.py \
    dataset.balanced=False \
    dataset.root=data \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features] \
    task=name \
    dataset.division_type=graded \
    +dataset.subdivision.graded_vertex_number=16 \
    dataset.max_ood_size_per_class=20 \
    trainer.device=cpu trainer.max_epoch=2
```

| `dataset.division_type` | `dataset.subdivision`        |
| ----------------------- | ---------------------------- |
| `graded`                | `+dataset.subdivision.graded_vertex_number=N` — every OOD entry is grown to exactly `N` vertices |
| `stellar`               | `+dataset.subdivision.fraction=f` — subdivide a fraction `f` of the top simplices |
| `barycentric`           | `+dataset.subdivision.round=k` — `k` full barycentric subdivisions |
| `none` (default)        | no OOD split is built        |

`graded` requires every in-distribution triangulation to have fewer than
`graded_vertex_number` vertices; combine it with `dataset.max_vertices`
to guarantee that. Note that Pachner walks (below) grow the
in-distribution triangulations after that check, so with
`dataset.walk_length > 0` the OOD entries are no longer guaranteed to be
the larger ones.

### The subdivision table

To get the paper's generalization table out of a single run, select the
`ood` config group instead of a single `division_type`:

```bash
.venv/bin/python main.py ... ood=paper_table
```

`config/ood/paper_table.yaml` lists four splits — graded to 16 vertices,
0.75-top stellar, full stellar and one full barycentric subdivision —
and each is built, evaluated and logged under its own prefix
(`ood_graded16/`, `ood_stellar075/`, `ood_stellar1/`, `ood_bary1/`),
next to the in-distribution `test/` column. Add or drop columns by
editing that file or with `~dataset.ood_evaluations.<name>`; `ood=none`
(the default) falls back to the single `division_type` split.

Every column re-runs `process()`, which rewrites the identical
train/val/test files and adds its own `ood` file, so the four splits
share one cache root at the cost of four preprocessing passes.

`evaluate_barycentric.py` remains the way to evaluate an *existing*
checkpoint on a subdivided dataset that was cached separately.

## Pachner walks and the Pachner objectives

Labels in MANTRA are topological invariants, so they are constant along
a Pachner orbit: bistellar moves change the triangulation but not the
manifold. `PachnerWalkDataset` (in the MANTRA library) turns that into
supervision by expanding every train/val/test entry into a random walk
of `dataset.walk_length` snapshots, `dataset.moves_per_step` moves
apart:

```bash
.venv/bin/python main.py \
    ... \
    dataset.walk_length=2 \
    dataset.moves_per_step=3 \
    trainer.sampler=pachner_pairs \
    loss=pachner_consistency
```

`dataset.walk_length=0` (the default) reproduces the plain dataset
exactly, including its cache files. Walk snapshots carry `walk_base`
(the entry they came from, counted within the split) and `walk_step`.

Two snapshots `a` and `b` of one walk are
`|walk_step_a - walk_step_b| * moves_per_step` **random moves** apart.
That is an upper bound on their Pachner distance, not the distance
itself: a random walk can undo its own moves. Treat it as a monotone
proxy.

`trainer.sampler=pachner_pairs` fills each training batch with index
pairs instead of a random sample — same walk, same class from a
different walk, and different class — in the proportions
`trainer.pair_fractions` (default `[0.5, 0.25, 0.25]`). Without it a
batch almost never contains two snapshots of one walk, and the
objectives below have nothing to work with. The validation batches are
sequential, but the snapshots of a walk are stored consecutively, so
they contain same-walk pairs by construction.

| `loss=`               | objective |
| --------------------- | --------- |
| `cross_entropy`       | the task loss alone (baseline) |
| `pachner_consistency` | task loss + `weight` · mean over same-walk pairs of ‖g(T_a) − g(T_b)‖² (`target=embedding`) or of the symmetric KL of the softmax outputs (`target=logits`) |
| `pachner_pairwise`    | (‖g(T_a) − g(T_b)‖ − `alpha`·d_ab)² on same-walk pairs, `relu(‖·‖ − radius)²` on same-class pairs, `relu(margin − ‖·‖)²` across classes, plus `ce_weight` · task loss |
| `pachner_infonce`     | supervised contrastive loss with same-class positives, temperature `temperature`, plus `ce_weight` · task loss |

Every objective returns its components, which are logged as
`train/loss_<name>` and `val/loss_<name>`. Early stopping and the LR
schedule key on `val/loss`, i.e. on the objective that is being trained.

Because the objectives shape the embedding rather than the classifier,
each run also reports (unless `trainer.eval_embeddings=False`), for the
test and OOD splits:

- `<split>/emb_spearman` — rank correlation of embedding distance with
  the number of moves between two snapshots of one walk,
- `<split>/emb_auroc` — how well the distance separates homeomorphism
  types,
- `<split>/emb_knn_bal_acc`, `<split>/emb_probe_bal_acc` — balanced
  accuracy of a 5-NN and a linear probe fitted on the *train*
  embeddings.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite under [`tests/`](./tests) covers the math utilities,
imbalance handling, the barycentric / stellar subdivision routines, the
model layers (Graphormer SPD, SAN-TMX, GCN/MLP, residual gated GCN),
and the Pachner pieces: the objectives, the pair sampler, the embedding
metrics, and that every model's `forward_with_embedding` returns the
logits its `forward` returns.


## Citation

A BibTeX entry will be added on de-anonymization.
