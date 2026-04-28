"""Evaluate a trained checkpoint on a precomputed barycentric-subdivision cache.

This script expects the subdivision pre_transform cache to have been built
beforehand by ``src/precompute_barycentric.py``. It points the dataset at
that cache (``force_reload=False``), reconstructs the model from the
training config stored next to the checkpoint (replaying the same cfg
mutations that ``main.py`` applies before instantiating the model), and
evaluates on the same test split that training used.

Each invocation evaluates a single checkpoint. To sweep over subdivision
levels and/or multiple checkpoints, use Hydra multirun::

    python src/evaluate_barycentric.py --multirun \\
        checkpoint_dir=checkpoints/<run_name> \\
        bary_level=1,2,3,4 \\
        bary_dataset_root='/tmp/bary${bary_level}_2d_mantra_random_features_cwn' \\
        bary_local_path='/tmp/mantra_bary/2_manifolds_bary_${bary_level}.json'
"""

import json
import os
import sys

import hydra
import wandb
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, read_write
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations

from src.schema import EvalBarycentricConfig
from src.utils.data_handling import filter_low_ocurrence_labels
from src.utils.train_utils import init_metrics, test


cs = ConfigStore.instance()
cs.store(name="eval_barycentric_conf", node=EvalBarycentricConfig)


def load_train_config(checkpoint_dir):
    return OmegaConf.load(os.path.join(checkpoint_dir, "config.yaml"))


def extract_train_metadata(train_cfg):
    """Pull model / embedding / representation class names out of train_cfg
    so they can be logged to wandb alongside eval results.
    """
    def _last_segment(target):
        return str(target).rsplit(".", 1)[-1] if target else None

    transform_dict = train_cfg.dataset.transform

    embedding_target = None
    if "embedding" in transform_dict and "_target_" in transform_dict.embedding:
        embedding_target = transform_dict.embedding._target_

    representation_target = None
    for key, sub in transform_dict.items():
        if key.startswith("representation") and "_target_" in sub:
            representation_target = sub._target_
            break

    select_repr = None
    if "select_features" in transform_dict:
        select_repr = transform_dict.select_features.get("representation")

    return {
        "model": _last_segment(train_cfg.model._target_),
        "embedding": _last_segment(embedding_target),
        "representation": _last_segment(representation_target),
        "select_features_representation": select_repr,
        "seed": train_cfg.get("seed"),
        "dataset_dimension": train_cfg.dataset.dimension,
        "dataset_balanced": bool(train_cfg.dataset.balanced),
    }


def build_transforms(train_cfg):
    transform_dict = train_cfg.dataset.transform
    transform_order = train_cfg.dataset.transform_order

    transforms = [None] * len(transform_dict)
    label_transform = None

    for name, transform in transform_dict.items():
        trans = hydra.utils.instantiate(transform)
        if name == "create_labels":
            label_transform = trans
        assert name in transform_order, f"Transform {name} not in transform_order"
        transforms[transform_order.index(name)] = trans

    return Compose(transforms), label_transform


def create_dataset(train_cfg, root, local_path, transforms, label_transform, balanced):
    # `balanced` must match what precompute_barycentric.py wrote — it controls
    # the processed filename (`data_{dim}.pt` vs `data_{dim}_balanced.pt`).
    dataset = ManifoldTriangulations(
        root=root,
        dimension=train_cfg.dataset.dimension,
        version=train_cfg.dataset.version,
        balanced=balanced,
        local_path=local_path,
        pre_transform=transforms,
        force_reload=False,
        transform=label_transform,
    )
    return dataset


def apply_main_cfg_mutations(train_cfg, dataset, label_transform):
    """Replay the cfg mutations that ``main.py`` performs before model init.

    Mirrors ``main.py`` so the model we instantiate here matches what
    training built: sets ``model.out_channels`` from the label_to_index size
    (for classification) and computes ``in_channels`` with the same
    graph / SCCNN-CWN / SAN branching.
    """
    is_graph_representation = (
        train_cfg.dataset.transform.select_features.representation == "graph"
    )

    # CreateLabels is stateful; main.py walks the whole dataset to populate
    # label_to_index before reading it.
    for _ in dataset:
        pass

    with read_write(train_cfg):
        if train_cfg.task.task_type == "classification":
            train_cfg.model.out_channels = len(label_transform.label_to_index)

    OmegaConf.resolve(train_cfg)

    model_target = train_cfg.model._target_
    if is_graph_representation or (
        "SCCNN" not in model_target and "CWN" not in model_target
    ):
        if train_cfg.dataset.num_features != "auto":
            in_channels = int(train_cfg.dataset.num_features)
        else:
            in_channels = (
                dataset[0]["x_1"].shape[1]
                if hasattr(dataset[0], "x_1")
                else dataset[0].x.shape[1]
            )
    else:
        feature_keys = [k for k in dataset[0].keys() if k.split("_")[0] == "x"]
        in_channels = (
            [dataset[0][k].shape[1] for k in feature_keys]
            if train_cfg.dataset.num_features == "auto"
            else [int(train_cfg.dataset.num_features)] * len(feature_keys)
        )

    return in_channels, is_graph_representation


def get_test_loader(dataset, train_cfg, collate_fn=None):
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch.utils.data import DataLoader

    if collate_fn is not None:
        return DataLoader(
            dataset,
            batch_size=train_cfg.trainer.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
    return PyGDataLoader(
        dataset,
        batch_size=train_cfg.trainer.batch_size,
        shuffle=False,
    )




@hydra.main(version_base=None, config_path="config", config_name="eval_barycentric")
def main(eval_cfg: EvalBarycentricConfig):
    ckpt_dir = eval_cfg.checkpoint_dir
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        print(f"[ERROR] checkpoint_dir is not a directory: {ckpt_dir}")
        sys.exit(1)

    print(f"[INFO] checkpoint_dir    = {ckpt_dir}")
    print(f"[INFO] bary_level        = {eval_cfg.bary_level}")
    print(f"[INFO] bary_dataset_root = {eval_cfg.bary_dataset_root}")
    print(f"[INFO] bary_local_path   = {eval_cfg.bary_local_path}")

    if not os.path.exists(eval_cfg.bary_local_path):
        print(
            f"[ERROR] bary_local_path does not exist: {eval_cfg.bary_local_path}\n"
            "        Run scripts/generate_barycentric_data.py first."
        )
        sys.exit(1)
    if not os.path.exists(eval_cfg.bary_dataset_root):
        print(
            f"[ERROR] bary_dataset_root does not exist: {eval_cfg.bary_dataset_root}\n"
            "        Run scripts/precompute_{cwn,san,gnn}_barycentric.sh first."
        )
        sys.exit(1)

    # Single-checkpoint eval — use Hydra multirun (checkpoint_dir=a,b,c) to
    # sweep over multiple runs.
    run_name = os.path.basename(os.path.normpath(ckpt_dir))
    print(f"\n=== Evaluating {run_name} ===")

    train_cfg = load_train_config(ckpt_dir)

    # The bary cache must have been generated from the same dataset
    # variant the model was trained on (balanced vs unbalanced)
    if bool(train_cfg.dataset.balanced) != bool(eval_cfg.bary_balanced):
        raise ValueError(
            f"balanced-flag mismatch for checkpoint {run_name}: "
            f"train_cfg.dataset.balanced={train_cfg.dataset.balanced} but "
            f"eval_cfg.bary_balanced={eval_cfg.bary_balanced}. "
            "The bary precompute must use the same `dataset.balanced` as "
            "training."
        )

    transforms, label_transform = build_transforms(train_cfg)

    dataset = create_dataset(
        train_cfg,
        root=eval_cfg.bary_dataset_root,
        local_path=eval_cfg.bary_local_path,
        transforms=transforms,
        label_transform=label_transform,
        balanced=eval_cfg.bary_balanced,
    )

    with read_write(train_cfg):
        train_cfg.trainer.device = eval_cfg.device

    in_channels, is_graph_repr = apply_main_cfg_mutations(
        train_cfg, dataset, label_transform
    )

    #NOTE: UGLY FIX
    label_str = f'/tmp/label_to_index_{train_cfg.dataset.dimension}d_'
    if not train_cfg.dataset.balanced:
        label_str += 'u'
    label_str +='bal.json'

    # Forward and inverted map
    label_to_index = label_transform.label_to_index
    index_to_label = {v: k for k,v in label_to_index.items()}

    # Same for the correct one
    with open(label_str, "r") as f:
        correct_label_to_index = json.load(f)
        correct_index_to_label = {v: k for k,v in correct_label_to_index.items()}

    print(correct_index_to_label)
    print(index_to_label)
    # This should be a map new_idx -> old_idx
    idx_to_idx = {
        previous_idx: correct_label_to_index[previous_label]
        for previous_idx, previous_label in  index_to_label.items()
    }

    # Iterate through all the dataset and remap everything
    for data in dataset:
        data.y = torch.tensor(idx_to_idx[data.y.item()])

    collate_fn = None
    if not is_graph_repr:
        from src.utils.data_loading import collate_simplicial
        collate_fn = collate_simplicial

    state_dict = torch.load(
        os.path.join(ckpt_dir, "best_val.pt"),
        weights_only=True,
        map_location=eval_cfg.device,
    )
    # Checkpoints saved from a torch.compile()-wrapped model carry an
    # `_orig_mod.` prefix on every key; strip it so load_state_dict matches
    # the uncompiled module we instantiate here.
    state_dict = {
        (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }

    model = hydra.utils.instantiate(train_cfg.model, in_channels=in_channels)
    model.load_state_dict(state_dict)
    model = model.to(eval_cfg.device)
    model.eval()

    accuracies = init_metrics(train_cfg)
    test_loader = get_test_loader(dataset, train_cfg, collate_fn)

    conf_dict = OmegaConf.to_container(eval_cfg, resolve=True)
    train_meta = extract_train_metadata(train_cfg)
    conf_dict.update(train_meta)
    conf_dict["checkpoint"] = run_name


    with wandb.init(entity=eval_cfg.logger.entity, project=eval_cfg.logger.project, config=conf_dict) as run:
        run.summary.update(train_meta)
        test(
            model, test_loader, accuracies, label_to_index, run, train_cfg
        )

    result = {
        "checkpoint": run_name,
        "bary_level": eval_cfg.bary_level,
        "bary_dataset_root": eval_cfg.bary_dataset_root,
        "bary_local_path": eval_cfg.bary_local_path,
        "num_test_samples": int(len(dataset)),
    }

    output_path = eval_cfg.output or os.path.join(
        ckpt_dir,
        f"barycentric_results_level_{eval_cfg.bary_level}.json",
    )
    with open(output_path, "w") as f:
        json.dump([result], f, indent=2)
    print(f"\n[INFO] Results saved to {output_path}")


if __name__ == "__main__":
    main()
