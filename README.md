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
├── config/                      # Hydra config groups (model, dataset, transform, task, …)
├── src/
│   ├── models/                  # GCN, MLP, ResidualGatedGCN, Graphormer, SAN, SAN-TMX, CWN, SCCNN
│   ├── utils/                   # data loading, train loop, metrics, barycentric helpers
│   ├── metrics/                 # loss + task metrics
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
> `'dataset.transform_order=[representation,embedding,select_features,create_labels]'`
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
   subdivisions of the test split, built by `MantraDataset` inside the
   training run (see
   [Evaluating subdivisions in the training run](#evaluating-subdivisions-in-the-training-run)).


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

To train on a balanced JSON you produced (here the demo file from the
example above):

```bash
.venv/bin/python main.py \
    dataset.balanced=True \
    dataset.local_path=/tmp/2_manifolds_balanced_demo.json \
    dataset.root=data/2d_mantra_balanced \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features,create_labels] \
    task=name \
    +task/metrics@task.metrics.bal_acc=accuracy \
    trainer.device=cpu trainer.max_epoch=2
```

The first run copies the JSON into
`${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds_balanced.json` and
materializes the processed cache. Subsequent runs reuse the cache; pass
`dataset.reload=True` to rebuild the splits and every OOD split of the
run. OOD splits that are not part of the run are deleted, not rebuilt.

#### How `dataset.local_path` and `dataset.balanced` interact

`main.py` enforces the invariant

```python
assert not ((dataset.local_path is not None) ^ dataset.balanced)
```

i.e. they must both be set or both be unset.

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
    dataset.transform_order=[representation,embedding,select_features,create_labels] \
    task=name \
    +task/metrics@task.metrics.bal_acc=accuracy \
    trainer.device=cpu trainer.max_epoch=2
```

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
| `loss`                                        | `cross_entropy`, `contrastive`                                                                      |
| `trainer`                                     | `default`                                                                                           |
| `logger`                                      | `wandb`                                                                                             |

Outputs:

- The best validation checkpoint and the resolved Hydra config are written
  to `${checkpoint_path}/<wandb_run_name>/{best_val.pt, config.yaml}`.
- Train / val / test metrics and a confusion matrix are logged to
  Weights & Biases under `${logger.entity}/${logger.project}`.

To turn off W&B sync, run with `WANDB_MODE=offline`. To override the
project name use `logger.project=<name>` (and `logger.entity=<entity>`).

## Evaluating subdivisions in the training run

The `ood` config group lists subdivisions of the test split that are
built next to the train/val/test splits and evaluated right after the
test set. Each entry is the `MantraDataset` subdivision arguments, and
its key names the metric prefix (`ood_<key>/...`); `test/...` stays the
in-distribution column. No `ood` option is selected by default, so
`dataset.ood_evaluations` stays empty and no OOD split is evaluated.

| `ood=`              | Paper                        | Splits                                         | Needs                    |
| ------------------- | ---------------------------- | ---------------------------------------------- | ------------------------ |
| `paper_unbalanced`  | Figure 5, Table S.10         | graded 16-20, 0.75- and 1-stellar, barycentric | `dataset.max_vertices=15` |
| `paper_balanced_2d` | Table S.11                   | graded 21-24, 0.75- and 1-stellar, barycentric | `dataset.max_vertices=20` |
| `paper_balanced_3d` | Table S.12                   | graded 31-34, 0.75- and 1-stellar, barycentric | `dataset.max_vertices=30` |

The vertex cap keeps every graded split strictly larger than any
training triangulation. For a one-off split, override the table
directly, e.g. `'+dataset.ood_evaluations={stellar05: {division_type:
stellar, fraction: 0.5}}'`.

Every OOD variant re-runs `MantraDataset.process()`, which rewrites the
train/val/test files (identical up to freshly drawn random features and
vertex labellings) and adds its own `ood_*` file, so all variants share
one cache root. The representation and encoding transforms are cached
there too, as `pre_transform`, in a processed directory named after a
hash of `dataset.transform` and `dataset.transform_order`, so several
featurizations coexist under one `dataset.root` and share its raw data.
Only the label transform runs on access. Random encodings are therefore
drawn once, when the cache is built, and reused by every run on that
cache, whatever its seed.

Simplicial models get random vertex features from
`transform@dataset.transform.embedding=simplex_random_features`; they
are written to `random_features_0`, and `propagate_vertex_features`
lifts them to the higher simplices.

`+transform@dataset.transform.permute=permute_vertices` relabels every
triangulation by a random vertex permutation (seeded with
`dataset.seed`), so label-dependent encodings such as the moment curve
and the orientations of signed incidence matrices do not inherit the
stored vertex order. It must come first in `dataset.transform_order`,
before any representation; the camera-ready manifests use it on every
arm.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite under [`tests/`](./tests) covers the math utilities,
imbalance handling, the barycentric / stellar subdivision routines, and
the model layers (Graphormer SPD, SAN-TMX, GCN/MLP, residual gated GCN).


## Citation

A BibTeX entry will be added on de-anonymization.
