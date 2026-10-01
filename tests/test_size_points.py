"""The size_points OOD table and the size-walk dataset parameters."""

from hydra import compose, initialize_config_dir
from pathlib import Path

import pytest

from main import create_dataset, ood_evaluations

CONFIG = str(Path(__file__).resolve().parents[1] / "config")
POINTS = [13, 16, 20, 24, 27, 32, 40, 48, 65, 100]


def compose_cfg(*overrides):
    with initialize_config_dir(version_base=None, config_dir=CONFIG):
        return compose(config_name="config", overrides=list(overrides))


def test_size_points_extends_size_vs_pattern():
    base = compose_cfg("ood=size_vs_pattern")
    cfg = compose_cfg("ood=size_points")
    cells = dict(ood_evaluations(cfg))
    for name, spec in ood_evaluations(base):
        assert cells[name] == spec
    for n in POINTS:
        assert cells[f"ood_prefine_{n}"] == {"division_type": "pachner", "target": n}
        assert cells[f"ood_pmixed_{n}"] == {"division_type": "pachner", "target": n, "mix": 5}
    assert len(cells) == len(ood_evaluations(base)) + 2 * len(POINTS)


def test_size_walk_parameters_reach_the_dataset(monkeypatch):
    """create_dataset hands size_targets & co. to PachnerWalkDataset, and the
    OOD cells of the table do not clash with them."""
    seen = []

    class FakeDataset:
        def __init__(self, split_type, **kwargs):
            seen.append((split_type, kwargs))
            self.transform = None

        def __iter__(self):
            return iter([])

    monkeypatch.setattr("main.PachnerWalkDataset", FakeDataset)
    cfg = compose_cfg(
        "ood=size_points",
        "dataset.size_targets=[20,32]",
        "dataset.keep_base=False",
        "dataset.max_vertices=10",
        "task=name",
    )
    create_dataset(cfg)
    splits = [s for s, _ in seen]
    assert splits[:3] == ["train", "val", "test"]
    assert splits.count("ood") == 37
    for split, kwargs in seen:
        if split == "ood":
            continue
        assert kwargs["size_targets"] == [20, 32]
        assert kwargs["size_splits"] == ("train", "val")
        assert kwargs["keep_base"] is False
        assert kwargs["size_mix"] == 0.0
    ood = [kw for s, kw in seen if s == "ood"]
    assert ood[-1]["division_type"] == "pachner"
    assert ood[-1]["target"] == 100 and ood[-1]["mix"] == 5
    # The OOD split is never expanded, so its instances do not carry the
    # expansion parameters (they would only rebuild the train/val walks
    # once per cell).
    assert ood[-1]["size_targets"] is None and ood[-1]["walk_length"] == 0


def test_size_targets_default_off():
    cfg = compose_cfg()
    assert cfg.dataset.size_targets is None
    assert list(cfg.dataset.size_splits) == ["train", "val"]
    assert cfg.dataset.keep_base is True
