"""
Evaluate trained model checkpoints on barycentric subdivisions of test data.

Usage:
    python src/evaluate_barycentric.py \
        checkpoint_dir=checkpoints/run_name \
        data_path=/path/to/2_manifolds_balanced.json \
        dimension=2 \
        max_subdivisions=1 \
        device=cpu

The checkpoint directory must contain:
    - best_val.pt   (model state dict)
    - config.yaml   (Hydra config from training)
"""

import json
import os
import sys
import tempfile
from collections import defaultdict

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations

from src.schema import EvalBarycentricConfig
from src.utils.barycentric_utils import subdivide_dataset_json
from src.utils.data_handling import filter_low_ocurrence_labels
from src.utils.train_utils import init_metrics


cs = ConfigStore.instance()
cs.store(name="eval_barycentric_conf", node=EvalBarycentricConfig)


def load_config(checkpoint_dir):
    """Load the Hydra config from a checkpoint directory."""
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    return cfg


def build_transforms(cfg):
    """Reconstruct the transform pipeline from config.

    Returns the full pre_transform Compose and the label_transform separately.
    """
    transform_dict = cfg.dataset.transform
    transform_order = cfg.dataset.transform_order

    transforms = [None] * len(transform_dict)
    label_transform = None

    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == "create_labels":
            label_transform = trans

        assert name in transform_order, f"Transform {name} not in transform_order"
        transforms[transform_order.index(name)] = trans

    return Compose(transforms), label_transform


def create_dataset(cfg, local_path, transforms, label_transform, dataset_name=None):
    """Create a ManifoldTriangulations dataset with the given config and data path."""
    dataset = ManifoldTriangulations(
        root=cfg.dataset.root,
        dimension=cfg.dataset.dimension,
        version=cfg.dataset.version,
        local_path=local_path,
        pre_transform=transforms,
        force_reload=True,
        transform=label_transform,
        name=dataset_name,
    )
    dataset = filter_low_ocurrence_labels(dataset)

    # Force a pass to build label_to_index
    for _ in dataset:
        pass

    return dataset


def get_test_loader(dataset, cfg, collate_fn=None):
    """Get only the test dataloader using the same split as training."""
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch.utils.data import DataLoader, Subset

    split_params = cfg.dataset.train_parameters.split

    indices = np.arange(len(dataset))
    from sklearn.model_selection import train_test_split

    train_val_index, test_index = train_test_split(
        indices,
        test_size=split_params.test,
        shuffle=True,
        stratify=dataset.y if cfg.dataset.use_stratified else None,
        random_state=cfg.seed,
    )

    test_dataset = Subset(dataset, test_index)

    if collate_fn is not None:
        return DataLoader(
            test_dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    else:
        return PyGDataLoader(
            test_dataset,
            batch_size=cfg.trainer.batch_size,
            shuffle=False,
        )


def evaluate_model(model, test_loader, accuracies, device):
    """Evaluate model on test data and return metrics."""
    model.eval()
    metrics = {}
    y_all = []
    y_hat_all = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            y_hat = model(batch).argmax(dim=1)
            y = batch.y.long()

            for _, acc in accuracies:
                acc(y_hat, y)

            y_all.append(y)
            y_hat_all.append(y_hat)

    for name, acc in accuracies:
        metrics[f"test/{name}"] = acc.compute().item()
        acc.reset()

    y_total = torch.cat(y_all)
    y_hat_total = torch.cat(y_hat_all)

    return metrics, y_total, y_hat_total


@hydra.main(version_base=None, config_path="../config", config_name="eval_barycentric")
def main(eval_cfg: EvalBarycentricConfig):

    # Detect whether checkpoint_dir is a single run or a parent with multiple runs
    checkpoint_dirs = []
    if os.path.exists(os.path.join(eval_cfg.checkpoint_dir, "best_val.pt")):
        checkpoint_dirs = [eval_cfg.checkpoint_dir]
    else:
        for entry in sorted(os.listdir(eval_cfg.checkpoint_dir)):
            full_path = os.path.join(eval_cfg.checkpoint_dir, entry)
            if os.path.isdir(full_path) and os.path.exists(
                os.path.join(full_path, "best_val.pt")
            ):
                checkpoint_dirs.append(full_path)

    if not checkpoint_dirs:
        print(f"[ERROR] No checkpoints found in {eval_cfg.checkpoint_dir}")
        sys.exit(1)

    print(f"[INFO] Found {len(checkpoint_dirs)} checkpoint(s) to evaluate")

    # Load and optionally filter the raw dataset
    print(f"[INFO] Loading raw dataset from {eval_cfg.data_path}")
    with open(eval_cfg.data_path) as f:
        raw_data = json.load(f)

    if eval_cfg.max_nodes is not None or eval_cfg.min_nodes is not None:
        original_count = len(raw_data)
        if eval_cfg.min_nodes is not None:
            raw_data = [d for d in raw_data if d["n_vertices"] >= eval_cfg.min_nodes]
        if eval_cfg.max_nodes is not None:
            raw_data = [d for d in raw_data if d["n_vertices"] <= eval_cfg.max_nodes]
        print(
            f"[INFO] Filtered to {len(raw_data)}/{original_count} samples "
            f"with nodes in [{eval_cfg.min_nodes or 0}, {eval_cfg.max_nodes or 'inf'}]"
        )

    # Pre-compute subdivided datasets and save to temp files
    subdivision_paths = {}

    with tempfile.TemporaryDirectory(prefix="mantra_bary_") as tmpdir:
        prev_data = raw_data
        for n_sub in range(eval_cfg.max_subdivisions + 1):
            if n_sub == 0:
                path = os.path.join(tmpdir, "original.json")
                with open(path, "w") as f:
                    json.dump(raw_data, f)
            else:
                print(f"[INFO] Computing barycentric subdivision level {n_sub}...")
                subdivided = subdivide_dataset_json(prev_data, n_subdivisions=1)
                prev_data = subdivided
                path = os.path.join(tmpdir, f"subdivision_{n_sub}.json")
                with open(path, "w") as f:
                    json.dump(subdivided, f)
                # Report size statistics
                n_nodes = [d["n_vertices"] for d in subdivided]
                print(
                    f"  -> {len(subdivided)} samples, "
                    f"node count: min={min(n_nodes)}, max={max(n_nodes)}, "
                    f"mean={np.mean(n_nodes):.0f}"
                )
            subdivision_paths[n_sub] = path

        all_results = []

        for ckpt_dir in checkpoint_dirs:
            run_name = os.path.basename(ckpt_dir)
            print(f"\n{'='*60}")
            print(f"[INFO] Evaluating checkpoint: {run_name}")
            print(f"{'='*60}")

            train_cfg = load_config(ckpt_dir)

            # Determine if graph-based representation
            is_graph_representation = (
                train_cfg.dataset.transform.select_features.representation == "graph"
            )
            collate_fn = None
            if not is_graph_representation:
                from src.utils.data_loading import collate_simplicial

                collate_fn = collate_simplicial

            # Build transforms
            transforms, label_transform = build_transforms(train_cfg)

            # Load model
            device = eval_cfg.device

            for n_sub in range(eval_cfg.max_subdivisions + 1):
                print(f"\n--- Subdivision level {n_sub} ---")

                local_path = subdivision_paths[n_sub]
                dataset_name = f"bary_eval_{n_sub}_{run_name}"

                # Override dataset root to use temp directory
                dataset_root = os.path.join(tmpdir, f"ds_{n_sub}_{run_name}")

                with OmegaConf.read_write(train_cfg):
                    original_root = train_cfg.dataset.root
                    train_cfg.dataset.root = dataset_root

                dataset = create_dataset(
                    train_cfg, local_path, transforms, label_transform, dataset_name
                )

                with OmegaConf.read_write(train_cfg):
                    train_cfg.dataset.root = original_root

                label_to_index = label_transform.label_to_index
                num_classes = len(label_to_index)

                # Initialize model (only once per checkpoint, but need num_classes)
                if n_sub == 0:
                    with OmegaConf.read_write(train_cfg):
                        train_cfg.model.out_channels = num_classes

                    if is_graph_representation:
                        in_channels = (
                            dataset[0].x.shape[1]
                            if train_cfg.dataset.num_features == "auto"
                            else int(train_cfg.dataset.num_features)
                        )
                    else:
                        feature_keys = [
                            k for k in dataset[0].keys() if k.split("_")[0] == "x"
                        ]
                        in_channels = (
                            [dataset[0][k].shape[1] for k in feature_keys]
                            if train_cfg.dataset.num_features == "auto"
                            else [int(train_cfg.dataset.num_features)] * len(feature_keys)
                        )

                    model = hydra.utils.instantiate(
                        train_cfg.model, in_channels=in_channels
                    )
                    state_dict = torch.load(
                        os.path.join(ckpt_dir, "best_val.pt"),
                        weights_only=True,
                        map_location=device,
                    )
                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()

                    with OmegaConf.read_write(train_cfg):
                        train_cfg.trainer.device = device

                    accuracies = init_metrics(train_cfg)

                test_loader = get_test_loader(dataset, train_cfg, collate_fn)

                metrics, y_total, y_hat_total = evaluate_model(
                    model, test_loader, accuracies, device
                )

                print(f"  Results:")
                for k, v in metrics.items():
                    print(f"    {k}: {v:.4f}")

                all_results.append(
                    {
                        "checkpoint": run_name,
                        "subdivision_level": n_sub,
                        "metrics": metrics,
                        "num_test_samples": len(y_total),
                    }
                )

        # Aggregate results
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")

        # Group by subdivision level
        by_subdivision = defaultdict(list)
        for r in all_results:
            by_subdivision[r["subdivision_level"]].append(r)

        for n_sub in sorted(by_subdivision.keys()):
            runs = by_subdivision[n_sub]
            print(f"\nSubdivision level {n_sub}:")
            all_metrics = defaultdict(list)
            for r in runs:
                for k, v in r["metrics"].items():
                    all_metrics[k].append(v)
            for k, values in all_metrics.items():
                arr = np.array(values)
                if len(arr) > 1:
                    print(f"  {k}: {arr.mean():.4f} +/- {arr.std():.4f} (n={len(arr)})")
                else:
                    print(f"  {k}: {arr[0]:.4f}")

        # Save results
        output_path = eval_cfg.output or os.path.join(
            eval_cfg.checkpoint_dir, "barycentric_results.json"
        )
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
