"""Hydra-configured precompute entry point for barycentric-subdivision datasets.

This is a slim counterpart to ``main.py``: it builds the pre_transform pipeline
from the standard config groups, instantiates ``ManifoldTriangulations`` (which
caches processed data under ``dataset.root``), and exits. No model, no training,
no WandB.

Unlike ``main.py`` this script does NOT enforce the
``local_path XNOR balanced`` assertion — it lets you feed a local subdivided
JSON (``dataset.local_path``) while running under ``dataset.balanced=False``,
which is exactly what the barycentric-subdivision precompute needs.

Typical invocation via Hydra multirun over subdivision levels::

    python src/precompute_barycentric.py --multirun \\
        +bary_level=1,2,3,4 \\
        dataset.root='/tmp/bary${bary_level}_2d_mantra_random_features_cwn' \\
        dataset.local_path='/tmp/mantra_bary/2_manifolds_bary_${bary_level}.json' \\
        dataset.dimension=2 \\
        dataset.balanced=False \\
        transform@dataset.transform.embedding=random_features \\
        dataset.transform.select_features.src=random_features \\
        ... (representation/transform_order/task/model as in main.py)

The ``model`` group is instantiated purely to mirror the config shape that
downstream training expects and to validate feature-dim plumbing; weights are
not trained or saved.
"""

import os

import hydra
import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch_geometric.transforms import Compose

from mantra.datasets import ManifoldTriangulations

from src.schema import RootConfig
from src.utils.data_handling import filter_low_ocurrence_labels


cs = ConfigStore.instance()
cs.store(name="precompute_barycentric_conf", node=RootConfig)


def _is_gpu(cfg):
    if cfg.trainer.device == "cpu":
        return False
    return torch.cuda.is_available()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: RootConfig):
    if not _is_gpu(cfg):
        n_threads = os.cpu_count() or 1
        torch.set_num_threads(n_threads)
        print(f"[INFO] CPU mode: torch.set_num_threads({n_threads})")

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

    assert label_transform is not None, "Missing the `create_labels` transform"
    transforms = Compose(transforms)

    print(
        f"[INFO] Building dataset at root={cfg.dataset.root} "
        f"local_path={cfg.dataset.local_path} balanced={cfg.dataset.balanced}"
    )

    dataset = ManifoldTriangulations(
        balanced=cfg.dataset.balanced,
        root=cfg.dataset.root,
        dimension=cfg.dataset.dimension,
        version=cfg.dataset.version,
        local_path=cfg.dataset.local_path,
        pre_transform=transforms,
        force_reload=cfg.dataset.reload,
        transform=label_transform,
    )
    dataset = filter_low_ocurrence_labels(dataset)

    # Trigger one full iteration so CreateLabels populates label_to_index and
    # the pre_transform cache is completed before exit.
    n = 0
    for _ in dataset:
        n += 1

    print(
        f"[INFO] Precompute complete: {n} samples cached under {cfg.dataset.root} "
        f"({len(label_transform.label_to_index)} classes)"
    )


if __name__ == "__main__":
    main()
