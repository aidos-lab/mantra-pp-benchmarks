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
│   ├── models/                  # GCN, MLP, ResidualGatedGCN, Graphormer, GIN, DeepSets, PointNet++, SAN, SAN-TMX, CWN, SCCNN
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

CUDA 13 wheels have no kernels for Volta GPUs (V100 and older). For such
a node install the CUDA 12.8 build into a second environment and point
the job scripts at it:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cu128 uv sync --extra cu128
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
    task=name \
    trainer.device=cpu trainer.max_epoch=2
```

The first run copies the JSON into
`${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds_balanced.json` and
materializes the processed cache. Subsequent runs reuse the cache; pass
`dataset.reload=True` to rebuild the splits and every OOD split of the
run. OOD splits that are not part of the run are deleted, not rebuilt.

#### How `dataset.local_path` and `dataset.balanced` interact

`dataset.local_path` replaces the download with a local JSON and
`dataset.balanced` selects the balanced variant of the cache; `main.py`
does not check that they agree, so pass both when training on a
generated balanced JSON.

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
    task=name \
    trainer.device=cpu trainer.max_epoch=2
```

For the balanced variant, generate the JSON first (see
[Generating the balanced dataset](#generating-the-balanced-dataset))
and then add the `dataset.balanced=True` /
`dataset.local_path=<path-to-generated-json>` overrides.

Key Hydra groups:

| Group                                         | Options                                                                                             |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `model`                                       | `gcn`, `mlp`, `residual_gated_gcn`, `graphormer`, `gin`, `deepsets`, `pointnetpp`, `san`, `san_tmx`, `cwn`, `sccnn` |
| `representation@dataset.transform.representation` | `dual_graph`, `one_skeleton`, `hasse_diagram`, `levi_graph`, `incidence_sc`, `adjacency_sc`, `up_/down_/hodge_laplacian_sc` |
| `transform@dataset.transform.embedding`       | `random_walk_pe`, `moment_curve_embedding`, `random_features`, `degree`, `effective_resistance`, `er_stats` |
| `task`                                       | `name`, `orientability`, `betti_numbers`; Calabi-Yau: `hodge_reg`, `hodge_class`, `chern_node`     |
| `trainer`                                     | `default`                                                                                           |
| `logger`                                      | `wandb`                                                                                             |

Losses are not a Hydra group: every `config/task/*.yaml` carries its own
`loss:` block (cross-entropy for the MANTRA tasks and `hodge_class`, MSE
for `hodge_reg`).

Outputs:

- The best validation checkpoint and the resolved Hydra config are written
  to `${checkpoint_path}/<wandb_run_name>/{best_val.pt, config.yaml}`.
- Train / val / test metrics and a confusion matrix are logged to
  Weights & Biases under `${logger.entity}/${logger.project}`.

To turn off W&B sync, run with `WANDB_MODE=offline`. To override the
project name use `logger.project=<name>` (and `logger.entity=<entity>`).

## Calabi-Yau tasks

The `cy` dataset serves Calabi-Yau star triangulations from a parquet
file through `mantra.datasets.CalabiYauDataset` (mantra `cy` branch with
aidos-lab/mantra#105), which splits the full `CalabiYau` data into
seeded train/val/test parts the way `MantraDataset` splits the MANTRA
data. Downloading is not implemented yet, so point `dataset.local_path`
at a local parquet file:

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet task=hodge_reg
```

Tasks:

- `task=hodge_reg` — graph-level regression of `h11`, scored with MAE
  and rounded accuracy; `task.target=h12` for the other Hodge number,
  `task.target=c2_anticanonical` for the second Chern class paired with
  the anticanonical divisor. The targets are used as they are.
- `task=hodge_class` — classification of the target value. The values
  of the train split become the classes, remapped to contiguous
  indices like the MANTRA tasks and saved next to the checkpoint as
  `label_to_index.json`.
- `task=chern_node` — node-level regression of `c2`, the second Chern
  class paired with the toric divisor of each point (c2 . D_i). Vertex
  0 is the origin, whose entry is c2 . K = -(sum of the others); see
  below.

Splits are seeded with `dataset.seed` and sized by
`dataset.train_parameters.split` (0.7/0.1/0.2); the split files encode
both, so variants coexist under `dataset.root`. Stratification mirrors
the MANTRA path: `dataset.use_stratified=True` stratifies on the values
of `dataset.label_source` (`h11` by default) and
`dataset.min_sample_per_class=<n>` first drops the values with at most
`n` samples. Every kept value needs a sample in each split, so choose
`n` for the data size, e.g. on the small `ks_n05` file:

```bash
python main.py dataset=cy dataset.local_path=/path/to/ks_n05_cylabels.parquet \
    task=hodge_class task.target=h12 \
    dataset.use_stratified=True dataset.label_source=h12 dataset.min_sample_per_class=8
```

`dataset.limit=<n>` processes only the first `n` triangulations for
smoke tests; the limit applies before the class filter.

To feed the vertex coordinates instead of the random-walk embedding,
propagate them to all simplex ranks with `propagate_convex_comb` before
the representation and let the representation read them (works for
`hasse_diagram`, `levi_graph` and `dual_graph`):

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet task=hodge_reg \
    representation@dataset.transform.representation=hasse_diagram \
    dataset.transform.representation.feature_propagation=x \
    transform@dataset.transform.embedding=propagate_convex_comb \
    '~dataset.transform.select_features' \
    'dataset.transform_order=[embedding,representation]'
```

The tests in `tests/test_cy_dataset.py` skip unless the installed
mantra ships `CalabiYauDataset`.

The representation and embedding transforms of the `cy` dataset are
cached: `create_dataset` passes them as `pre_transform`, drops the
attributes nothing reads afterwards (the simplex trie, the raw
triangulation, the intersection numbers, connectivity of rank 3 and
above, the copied embedding) and runs only the label transform on
access. The processed directory carries a hash of the transform config,
so variants with different transforms coexist under one `dataset.root`
and a changed chain never reads a stale cache; the splits are still
built per root, so one root per transform stack keeps precompute jobs
from racing.

### Point-set models

`deepsets` and `pointnetpp` read only `batch.x` and
`batch.batch`, never the connectivity. With the one-skeleton as the
representation (its nodes are the vertices, in order) and the lattice
coordinates selected as the features, `x` is the raw `(n_points, 4)`
coordinate matrix with the origin in row 0, so these models see the
polytope and nothing of the triangulation:

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet task=hodge_reg \
    model=deepsets \
    representation@dataset.transform.representation=one_skeleton \
    dataset.transform.select_features.src=coords
```

`deepsets` is a DeepSets network (shared point-wise MLP, sum pooling,
MLP head); with `model.pool_method=max` it is vanilla PointNet without
the transform nets. `pointnetpp` is PointNet++ in the form of PyTorch
Geometric's `pointnet2` examples (farthest point sampling starting at
the origin,
radius grouping with `sa_radii` sized for integer coordinates, and
feature-propagation levels at node level). It needs `torch-cluster`,
which `uv sync` installs. The random-walk embedding is still computed
and then replaced by `select_features`; dropping it also needs
`'~dataset.transform.embedding'` with
`'dataset.transform_order=[representation,select_features]'`.

Read the Hodge results with the counting shortcut in mind: for a
favorable polytope `h11_cy` (the `h12` column in the Kreuzer-Skarke
convention) equals `n_points - 5`, so a sum-pooled point model, like
any graph model, can count its way to it. The second Chern class and
the intersection numbers depend on the triangulation, which is where
the gap between the point-set models and the models on the simplicial
structure is the quantity of interest.

### Node-level tasks

`task.task_level=node` predicts one value per triangulation vertex:
the label transform (`mantra.transforms.AttributeToNodeRegressionTransform`
/ `AttributeToNodeClassTransform`, mantra PR #96) writes one row of
`data.y` per vertex, in vertex order, and the loss and the metrics run
over all rows. Two things must line up with that:

- the graph nodes have to be the vertices, so pick the one-skeleton
  (or a simplicial representation) instead of the default dual graph;
- the model needs a node-level head: `create_model` sets
  `model.task_level` from the task; `gcn`, `mlp`, `gin`,
  `residual_gated_gcn`, `graphormer` then skip the pooling (one output
  per node), `cwn`, `sccnn` read out per 0-cell, `deepsets` predicts
  each point from its embedding concatenated with the pooled set
  feature and `pointnetpp` through its feature-propagation levels.

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet \
    task=chern_node model=gcn \
    representation@dataset.transform.representation=one_skeleton
```

The origin is a target like every other vertex. Its value is the
negated sum of the other entries, so it is much larger in magnitude: on
the `ks_n05` subsample it is 2.4% of the vertices but 75% of the target
variance (a mean predictor scores MSE 1955 with the origin, 473
without). Read MSE and MAE with that in mind.

Any attribute task can be run per vertex, e.g. node-level
classification with `task=hodge_class task.target=c2
task.task_level=node`. As at graph level, the classes are the values
seen in the train split; a vertex value that occurs only in val or
test is an error at start-up, since no model could predict it.

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

`MantraDataset` builds the train/val/test files once per processed
directory; every OOD variant only subdivides the cached test entries and
adds its own `ood_*` file, so all variants share one cache root. The
representation and encoding transforms are cached
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
the model layers (Graphormer SPD, SAN-TMX, GCN/MLP, residual gated GCN,
GIN, DeepSets, PointNet++).


## Citation

A BibTeX entry will be added on de-anonymization.
