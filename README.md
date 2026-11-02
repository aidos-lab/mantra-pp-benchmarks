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

### GPU install (Linux + Cuda 12.8)

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
| `model`                                       | `gcn`, `mlp`, `residual_gated_gcn`, `graphormer`, `gin`, `deepsets`, `pointnetpp`, `san`, `san_tmx`, `cwn`, `sccnn`, `ect_vit`, `inv_feature_mlp` |
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

The paper's runs compose three config groups on top of that:
`+experiment=cy/<model>` holds the protocol (`config/experiment/cy/base.yaml`:
split seed, batch size, epochs, early stopping, scheduler patience) and
the model's size and learning rate; `+chain=<stack>` the representation
and feature stack (`skel`, `coords`, `coordsh`, `dualc`, `levic`, `cwn`,
`sccnn`, `ect1`, `ectfull`, `fvec1`, `fvecfull`, see `config/chain/`);
`+cy_task=<row>` the task with its split settings (`h11_class`,
`h21_class`, `h21_reg`, `chern_node`). `deepsets_node` and
`pointnetpp_node` are the node-level sizes of those two models, and
`invmlp` needs `model.max_points` for the file. Anything on the command
line still overrides them:

```bash
python main.py +experiment=cy/gcn +chain=levic +cy_task=h11_class \
    dataset.local_path=/path/to/manifolds.parquet trainer.device=cuda
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

`baseline.py` takes the same overrides and logs the trivial baselines
as W&B runs with `model.name=baseline`: the majority class or the train
median, and for `h11` the point count minus five (`kind=count`), which
is exact for favorable polytopes, 94% of the `ks_h12lt10` rows. The
count reads `n_vertices`, which the Hasse and dual representations
overwrite, so pass `+chain=coords`; the split does not depend on the
representation.

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
    +chain=levic representation@dataset.transform.representation=hasse_diagram
```

The tests in `tests/test_cy_dataset.py` skip unless the installed
mantra ships `CalabiYauDataset`.

The `cy` dataset caches its transform chain like `MantraDataset` (see
the OOD section). The chain ends with mantra's `SelectAttributes`
(`config/transform/select_attributes.yaml`), so a cached sample holds
only its `keep_keys`: the graph inputs, the task targets and what
`baseline.py` reads. The chain also holds `kappa`, which writes the
per-triangle intersection numbers for the triangle task, so every cy
parquet needs the `intersection_numbers` column and any
`dataset.transform_order` you pass must end with `kappa,
select_attributes`. Simplicial and ECT chains select their own keep
lists with `transform@dataset.transform.select_attributes=select_attributes_sc`
or `=select_attributes_ect`, as the ECT example below does. Simplicial
chains also add `+transform@dataset.transform.scale=scale_operators`
before `kappa`: it divides every Laplacian and adjacency operator by
its largest absolute row sum, without which the SCCNN's Chebyshev
convolutions blow the features up by the operators' eigenvalues (up to
14 here) at every layer.

### Point-set models

`deepsets` and `pointnetpp` read only `batch.x` and
`batch.batch`, never the connectivity. With the one-skeleton as the
representation (its nodes are the vertices, in order) and the lattice
coordinates selected as the features, `x` is the raw `(n_points, 4)`
coordinate matrix with the origin in row 0, so these models see the
polytope and nothing of the triangulation:

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet task=hodge_reg \
    model=deepsets +chain=coords
```

`deepsets` is a DeepSets network (shared point-wise MLP, sum pooling,
MLP head); with `model.pool_method=max` it is vanilla PointNet without
the transform nets. `pointnetpp` is PointNet++ in the form of PyTorch
Geometric's `pointnet2` examples (farthest point sampling starting at
the origin,
radius grouping with `sa_radii` sized for integer coordinates, and
feature-propagation levels at node level). It needs `torch-cluster`,
which `uv sync` installs.

`inv_feature_mlp` is the MLP with invariant features of Ewert et al.
(2024): a point-wise column encoder max-pooled over the points plus an
axis-wise row encoder max-pooled over the four axes, on the same
coordinate matrix zero-padded to `model.max_points`. The paper used the
vertex matrix of the polytope; `+transform@dataset.transform.hull=hull_vertices`
marks the hull vertices among the lattice points and
`model.point_mask=hull_mask` restricts the model to them (`hull` goes
before `select_features` in `transform_order`).

### Euler characteristic transform

`ect_vit` reads only `batch.ect`, written by the mantra
`EulerCharacteristicTransform`: per seeded direction, the Euler
characteristic of the sublevel subcomplex at `resolution` thresholds,
of the `max_rank`-skeleton or of every face. Every direction's curve is
a token of a small pre-norm transformer; `main.py` standardises the
curves with train-split statistics (`feature_stats`), without which the
raw counts make the model collapse on validation. The transform reads
the coordinates, so it must precede `select_features` in the order:

```bash
python main.py dataset=cy dataset.local_path=/path/to/manifolds.parquet task=hodge_class \
    model=ect_vit +chain=ect1
```

Every top simplex of a Calabi-Yau triangulation contains the origin, so
the ECT of the full complex is only the support function of the
polytope; `dataset.transform.ect.per_rank=true` keeps the face count of
every rank instead (`model.num_ranks=max_rank+1`), which is strictly
more informative. At node level the lattice points join the token
sequence and are predicted one by one.

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
    task=chern_node model=gcn +chain=coords
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

### Triangle-level task

`task=kappa_triangle` regresses the intersection number of the three
toric divisors of every boundary triangle of the triangulation, i.e.
every 3-cone of the fan. Point 0 is the origin, which CYTools identifies
with the canonical class `K = -sum_l D_l`, so the `intersection_numbers`
rows through index 0 are not triple intersections of toric divisors (a
triangle through the origin would carry `K . D_j . D_k`, minus the
anticanonical degree of the curve `D_j . D_k`, which the linear
relations determine from the boundary numbers and the coordinates);
the `kappa` transform ignores them like the self-intersections. It
writes `kappa_2`, one value per boundary triangle in lexicographic order
(zero where the sparse `intersection_numbers` have no entry),
`simplex_index_2`, the triangles' vertex indices, and `triangle_mask`,
true on the boundary triangles in the rank-2 order of the simplicial
representations. Models with one embedding per vertex sum the three
vertex embeddings of a triangle (`target_rows`), which is symmetric like
the target; `sccnn` reads out its rank-2 states and `cwn`, whose layers
update edges only, sums the final edge states of each triangle, both
restricted by `triangle_mask`. A linear layer on the summed rows is
additive in the vertices and cannot fit the target, so the task sets
`model.head_layers=2`: the readout of every model becomes `head`, two
Linear-LayerNorm-ReLU blocks before the final linear layer (`deepsets`
already has one hidden layer in `rho` and `inv_feature_mlp` its
`decoder_layers`; the head comes on top of those). The default
`head_layers=0` is the plain linear layer, so the graph and node
results are unchanged. About 40 % of the boundary triangles have
intersection number 0, which is what the median baseline scores in
rounded accuracy. The `kappa` transform with the `mask` argument and
the boundary-only targets needs mantra's `cy-experiments` branch with
`scripts/cy/patches/mantra-kappa-boundary.patch` applied; an older
mantra fails at dataset construction.

### Several phases per polytope

Every polytope of the parquet files above carries one triangulation,
the Delaunay one, so all targets are functions of the point set and
the point-set models compete with the simplicial ones. A parquet with
several Calabi-Yau phases of each polytope makes the per-simplex
targets depend on the simplices: `scripts/cy/make_phases_parquet.py`
(run in `.venv-cytools`) enumerates the fine regular star
triangulations of a seeded random subset of the polytopes with at
least two phases and within its enumeration caps, keeps one
representative per 2-face restriction (the Calabi-Yau phase; the
Delaunay triangulation when its phase is among them) and writes the
usual columns plus `polytope_id`, `phase`, `n_phases`, `n_frst` and
`is_delaunay`. `scripts/cy/phase_stats.py` checks such a file against
the local formula for the intersection numbers, the c2 star sum and the
chi identity, and prints how many phases the 1-skeleton and the
boundary triangles tell apart and the ceiling of a phase-blind
predictor.

Training on it needs `dataset.group_by=polytope_id`, which puts all
phases of a polytope into the same split (mantra `cy-experiments` with
`scripts/cy/patches/mantra-group-split.patch`; `label_source` must then
be constant within a polytope, which the Hodge numbers are). mantra
reads the group before the pre-transform, so training itself needs no
extra attribute; `polytope_id` is kept in the caches only for
`baseline.py`, which then logs the `phase_median_oracle` of the
per-vertex and per-triangle tasks: the median target of each vertex or
boundary triangle over the phases of its polytope containing it, the
best MAE any model that does not see the phase can reach. Adding the
key changed the cache name of every Calabi-Yau chain once.

Note that the intersection number of a boundary triangle is the same
in every phase containing it, so the triangle task stays solvable
without the phase; the per-vertex c2 sums over the triangles present
and is the task that changes with the phase.

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
