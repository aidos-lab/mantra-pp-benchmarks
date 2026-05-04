# MANTRA++ Benchmarks

Code accompanying the paper *No Triangulation Without Representation:
Generalization in Topological Deep Learning*.

This repository contains:

- Implementation of Pachner moves, rebalancing and duplication checks of topological datasets
- Training and evaluation pipeline for graph- and simplicial / cell-complex
  models on the (extended) MANTRA manifold-triangulation classification benchmarks.
- A subdivision pipeline (full barycentric, *stellar*, and *graded*) used to
  generate out-of-distribution test sets from the original triangulations,
  together with the corresponding precompute and evaluation scripts.

## Anonymity note (double-blind submission)

To preserve double-blind review, the [`mantra/`](./mantra) subdirectory
contains a vendored snapshot of the development state of the upstream
MANTRA dataset library
(<https://github.com/aidos-lab/mantra>, third-party authors), so this
repository can be built and run without referencing any author-specific
fork or local checkout. Original third-party authorship of the MANTRA
package is preserved in `mantra/pyproject.toml`, `mantra/LICENSE.md`, and
`mantra/README.md`.

Modifications we made to the MANTRA library while developing this paper
are included in the vendored copy and will be made available after review period ends.

## Repository layout

```
.
├── main.py                      # Hydra entry point: train + test a model
├── precompute_barycentric.py    # Build pre_transform cache for a subdivided dataset (no training)
├── generate_barycentric_data.py # Subdivide a raw MANTRA JSON into level-k JSONs
├── evaluate_barycentric.py      # Load a checkpoint, evaluate on a subdivided cache
├── config/                      # Hydra config groups (model, dataset, transform, task, …)
├── src/
│   ├── models/                  # GCN, MLP, ResidualGatedGCN, Graphormer, SAN, SAN-TMX, CWN, SCCNN
│   ├── utils/                   # data loading, train loop, metrics, barycentric helpers
│   ├── metrics/                 # loss + task metrics
│   └── schema.py                # OmegaConf dataclasses for Hydra structured configs
├── tests/                       # pytest suite (math utils, transforms, models, losses, …)
├── datasets/                    # Raw + subdivided MANTRA JSONs shipped with the repo (see Datasets section)
└── mantra/                      # Vendored MANTRA dataset library (see note above)
    └── mantra/                  # Python package
        ├── augmentations/       # Pachner moves + balancing logic
        ├── datasets/            # ManifoldTriangulations (PyG InMemoryDataset)
        ├── representations/     # Dual graph, Hasse diagram, simplicial connectivity, …
        ├── transforms/          # Embeddings, label/feature selection
        └── generate_balanced.py # CLI for producing the balanced dataset variant
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
2. Install the vendored `mantra/` directory as an editable
   `mantra-dataset==0.0.19` package, so all `from mantra.…` imports
   resolve to `mantra/mantra/`.
3. Install this repository as `mantra-pp-benchmarks==0.1.0`.

All commands below assume `.venv/bin/python` (or `uv run python …`).

> **zsh users:** the bracketed Hydra list overrides
> (e.g. `dataset.transform_order=[representation,…]`) need to be quoted —
> `'dataset.transform_order=[representation,embedding,select_features,create_labels]'`
> — because zsh treats `[…]` as a glob. In bash they work either way.

### Datasets

All JSONs needed to reproduce the experiments are checked in under
[`datasets/`](./datasets):

```
datasets/
├── 2_manifolds.json                                  # Unbalanced 2-manifolds (with updated class labels)
├── 2_manifolds_balanced.json                         # Balanced 2-manifolds (target_count=1000)
├── 3_manifolds.json                                  # Unbalanced 3-manifolds
├── 3_manifolds_balanced.json                         # Balanced 3-manifolds
├── mantra_bary_2d_unbalanced_full/                   # Full barycentric, level 1
├── mantra_bary_2d_unbalanced_stellar/                # Stellar, levels 0.75 and 1
├── mantra_bary_2d_unbalanced_graded/                 # Graded subdivisions n=16..20
├── mantra_bary_2d_balanced_full/
├── mantra_bary_2d_balanced_stellar/
├── mantra_bary_3d_balanced_full/
└── mantra_bary_3d_balanced_stellar/
```

There are three dataset variants used in the paper:

1. **Original (unbalanced) MANTRA.** The version with updated class
   labels lives at
   [`datasets/2_manifolds.json`](./datasets/2_manifolds.json) /
   [`datasets/3_manifolds.json`](./datasets/3_manifolds.json). The same
   JSONs are also auto-downloaded from upstream by
   `ManifoldTriangulations(root=..., balanced=False)` if you don't pre-populate
   the cache (see *Pre-populating the unbalanced cache* below).
2. **Balanced MANTRA.** Pre-generated at
   [`datasets/2_manifolds_balanced.json`](./datasets/2_manifolds_balanced.json) /
   [`datasets/3_manifolds_balanced.json`](./datasets/3_manifolds_balanced.json).
   You can also re-generate them yourself via
   `mantra/mantra/generate_balanced.py`
   (see [Generating the balanced dataset](#generating-the-balanced-dataset)).
3. **Subdivided MANTRA.** Pre-generated under `datasets/mantra_bary_*/`
   for every (dim × balanced × subdivision-mode) combination used in the
   paper. You can also re-generate them via
   `generate_barycentric_data.py` (see
   [Reproducing the subdivision experiments](#reproducing-the-subdivision-experiments)).

#### Pre-populating the unbalanced cache (optional)

`main.py` enforces `dataset.local_path is None XOR dataset.balanced`,
so to train on `datasets/2_manifolds.json` *without* the upstream
download, copy the JSON into the cache layout PyG expects:

```bash
mkdir -p data/mantra/2D/raw && cp datasets/2_manifolds.json data/mantra/2D/raw/
mkdir -p data/mantra/3D/raw && cp datasets/3_manifolds.json data/mantra/3D/raw/
```

After this, `dataset.root=data` + `dataset.balanced=False` will skip
the download.


### Generating the balanced dataset

The balanced variant is produced by Pachner-move augmentation +
deduplication on the unbalanced JSON. Source code:
[`mantra/mantra/augmentations/balancing.py`](./mantra/mantra/augmentations/balancing.py);
CLI: [`mantra/mantra/generate_balanced.py`](mantra/mantra/generate_balanced.py).

A reduced version can be generated and run locally on a CPU in a reasonable time; the full version (target count 1000) is best run on a cluster. 

For the 2-manifolds (small target count for a quick CPU run):

```bash
.venv/bin/python mantra/mantra/generate_balanced.py \
    datasets/2_manifolds.json \
    /tmp/2_manifolds_balanced_demo.json \
    --dimension 2 \
    --target-count 100 \
    --dedup-max-rounds 5
```

For the 3-manifolds:

```bash
.venv/bin/python mantra/mantra/generate_balanced.py \
    datasets/3_manifolds.json \
    /tmp/3_manifolds_balanced_demo.json \
    --dimension 3 \
    --target-count 100 \
    --dedup-max-rounds 5
```

The full-target-count `(target_count=1000)` versions used in the paper
are already shipped at
`datasets/{2,3}_manifolds_balanced.json`.

Useful flags:

| Flag                     | Default | Effect                                                                                       |
| ------------------------ |---------| -------------------------------------------------------------------------------------------- |
| `--target-count`         | 1000    | Target number of triangulations per class (over- and under-sampled symmetrically).           |
| `--n-moves`              | 5       | Number of random Pachner moves applied to each augmented sample.                             |
| `--seed`                 | 42      | RNG seed.                                                                                    |
| `--max-vertices`         | none    | Drop entries (original or augmented) with more than this many vertices.                      |
| `--no-topology-changes`  | off     | Disable torus / cross-cap gluing (2D only). With this flag, deficits are filled by Pachner moves alone. |
| `--no-dedup`             | off     | Skip the post-augmentation isomorphic-duplicate removal ().                |
| `--dedup-max-rounds`     | 5       | Number of dedup-and-regenerate rounds before falling back to "remove only".                  |

The unbalanced source JSONs `${dim}_manifolds.json` are produced by the
auto-download path of `ManifoldTriangulations`; after one
`balanced=False` run the file is in
`${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds.json` and can be
fed straight into `generate_balanced.py`.

#### Loading a locally generated balanced JSON in training

To train on the in-repo balanced 2-manifolds JSON:

```bash
.venv/bin/python main.py \
    dataset.balanced=True \
    dataset.local_path=datasets/2_manifolds_balanced.json \
    dataset.root=data/2d_mantra_balanced \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features,create_labels] \
    task=name \
    +task/metrics@task.metrics.acc=accuracy \
    trainer.device=cpu trainer.max_epoch=2
```

The first run copies the JSON into
`${dataset.root}/mantra/${dim}D/raw/${dim}_manifolds_balanced.json` and
materializes the processed cache. Subsequent runs reuse the cache; pass
`dataset.reload=True` to force re-processing.

#### How `dataset.local_path` and `dataset.balanced` interact

`main.py` enforces the invariant

```python
assert not ((dataset.local_path is not None) ^ dataset.balanced)
```

i.e. they must both be set or both be unset.

## Running a single training run

`main.py` is a Hydra app with the default composition defined in
`config/config.yaml`. A minimal CPU run on the in-repo balanced
2-manifolds:

```bash
.venv/bin/python main.py \
    dataset.balanced=True \
    dataset.local_path=datasets/2_manifolds_balanced.json \
    dataset.root=data/2d_mantra_balanced \
    dataset.dimension=2 \
    model=gcn \
    representation@dataset.transform.representation=one_skeleton \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features,create_labels] \
    task=name \
    +task/metrics@task.metrics.acc=accuracy \
    trainer.device=cpu trainer.max_epoch=2
```

For the unbalanced variant, first pre-populate the cache (see above)
and then drop the `dataset.{balanced,local_path}` overrides.

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

## Reproducing the subdivision experiments

The barycentric / stellar / graded-subdivision experiments follow a
three-step pipeline. Throughout, `${LEVEL}` denotes the subdivision level
(1, 2, …; or a fraction in `(0, 1)` for partial-stellar).

### 1. Generate the subdivided JSONs

```bash
.venv/bin/python generate_barycentric_data.py \
    input=datasets/2_manifolds.json \
    output_dir=/tmp/mantra_bary_demo \
    prefix=2_manifolds \
    n_levels=1 \
    min_vertices=16 \
    mode=stellar
```

Writes `${output_dir}/${prefix}_bary_<level>.json` for every level
`1..n_levels`. Entries with fewer than `min_vertices` after subdivision
are dropped from the output. See
[`config/generate_barycentric_data.yaml`](./config/generate_barycentric_data.yaml)
for all options (notably `n_smallest`, `min_class_count`, and the
`mode` validity rules; valid `mode` values are `stellar`,
`full_barycentric`, `graded`).

The pre-generated outputs used in the paper ship under
`datasets/mantra_bary_*/` and can be used directly without re-running
this step.

### 2. Build the pre_transform cache for that level

Using the in-repo unbalanced stellar level-1 JSON:

```bash
.venv/bin/python precompute_barycentric.py \
    +bary_level=1 \
    dataset.root=data/bary1_2d_unbalanced_random_features_dual_graph \
    dataset.local_path=datasets/mantra_bary_2d_unbalanced_stellar/2_manifolds_bary_1.json \
    dataset.dimension=2 \
    dataset.balanced=False \
    representation@dataset.transform.representation=dual_graph \
    transform@dataset.transform.embedding=random_features \
    dataset.transform.select_features.src=random_features \
    dataset.transform_order=[representation,embedding,select_features,create_labels] \
    task=name model=gcn trainer.device=cpu \
    +task/metrics@task.metrics.acc=accuracy
```

`precompute_barycentric.py` deliberately drops the
`local_path XOR balanced` assertion that `main.py` enforces, so it
can take an unbalanced subdivided JSON via `dataset.local_path`.

The aggregated launchers used in the paper are
`scripts/precompute_{gnn,cwn,san,sccnn}_barycentric_all*.sh` and
`run_precompute_array.sh` (SLURM job array form).

### 3. Evaluate a trained checkpoint on the subdivided cache

```bash
.venv/bin/python evaluate_barycentric.py \
    checkpoint_dir=checkpoints/<run_name> \
    bary_level=1 \
    bary_dataset_root=data/bary1_2d_unbalanced_random_features_dual_graph \
    bary_local_path=datasets/mantra_bary_2d_unbalanced_stellar/2_manifolds_bary_1.json \
    bary_balanced=False \
    device=cpu
```

`evaluate_barycentric.py` loads the *training* config from
`<checkpoint_dir>/config.yaml`, replays the same model-shape mutations
that `main.py` applies (in particular `model.out_channels` from
`label_to_index`), strips the `_orig_mod.` prefix introduced by
`torch.compile`, and runs the test loop on the subdivided dataset.

The `bary_balanced` flag **must** match the `dataset.balanced` value used
when the cache was built; the script aborts with a clear error otherwise.

To sweep over levels and / or checkpoints, use Hydra multirun:

```bash
.venv/bin/python evaluate_barycentric.py --multirun \
    checkpoint_dir=checkpoints/run_a,checkpoints/run_b \
    bary_level=1 \
    bary_dataset_root='data/bary${bary_level}_2d_unbalanced_random_features_dual_graph' \
    bary_local_path='datasets/mantra_bary_2d_unbalanced_stellar/2_manifolds_bary_${bary_level}.json'
```

Per-checkpoint results are written to
`<checkpoint_dir>/barycentric_results_level_<L>.json`.

## Tests

```bash
.venv/bin/python -m pytest -q
```

The suite under [`tests/`](./tests) covers the math utilities,
imbalance handling, the barycentric / stellar subdivision routines, and
the model layers (Graphormer SPD, SAN-TMX, GCN/MLP, residual gated GCN).


## Citation

A BibTeX entry will be added on de-anonymization.
