import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from main import create_loaders, create_model, training_loop


@pytest.mark.parametrize("num_workers", [0, 1])
def test_create_loaders_keeps_workers_resident_only_for_train_and_val(num_workers):
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            batch_size=1, num_workers=num_workers, pin_memory=False
        ),
        dataset=SimpleNamespace(
            transform=SimpleNamespace(
                select_features=SimpleNamespace(representation="graph")
            )
        ),
    )
    dataset = [
        Data(
            x=torch.ones(1, 1),
            y=torch.tensor(0),
            edge_index=torch.empty((2, 0), dtype=torch.long),
        )
    ]

    train, val, test, ood_loaders = create_loaders(
        cfg, dataset, dataset, dataset, {"ood_bary1": dataset}
    )

    resident = num_workers > 0
    assert train.persistent_workers == resident
    assert val.persistent_workers == resident
    assert not test.persistent_workers
    assert list(ood_loaders) == ["ood_bary1"]
    assert not any(loader.persistent_workers for loader in ood_loaders.values())


def test_create_model_updates_runtime_output_channels(monkeypatch):
    captured = {}

    def fake_instantiate(model_cfg, **kwargs):
        captured["configured_out_channels"] = model_cfg.out_channels
        captured.update(kwargs)
        return torch.nn.Linear(1, model_cfg.out_channels)

    monkeypatch.setattr("main.hydra.utils.instantiate", fake_instantiate)
    cfg = SimpleNamespace(
        model=SimpleNamespace(in_channels=1, out_channels=1),
        task=SimpleNamespace(task_type="classification", task_level="graph"),
        dataset=SimpleNamespace(
            transform=SimpleNamespace(
                select_features=SimpleNamespace(representation="graph")
            )
        ),
        trainer=SimpleNamespace(device="cpu"),
        compile=False,
    )

    create_model(cfg, num_features=1, label_to_index={4: 0, 12: 1, 20: 2})

    assert cfg.model.out_channels == 3
    assert captured == {"configured_out_channels": 3, "in_channels": 1}


@pytest.mark.parametrize(
    ("max_epoch", "early_stopping", "message"),
    [(0, 1, "max_epoch"), (1, 0, "early_stopping")],
)
def test_training_loop_rejects_configuration_without_an_epoch(
    max_epoch, early_stopping, message
):
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            max_epoch=max_epoch,
            early_stopping=early_stopping,
        )
    )

    with pytest.raises(ValueError, match=message):
        training_loop(None, None, None, None, None, cfg, {})


class BatchLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)

    def forward(self, batch):
        return self.linear(batch.x)


class FakeRun:
    name = "test-run"
    id = "test-run"
    summary = {}

    def log(self, values):
        pass


def test_training_loop_saves_typed_label_mapping(tmp_path):
    cfg = OmegaConf.create(
        {
            "checkpoint_path": str(tmp_path),
            "trainer": {
                "max_epoch": 1,
                "early_stopping": 1,
                "device": "cpu",
                "scheduler_patience": 10,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "task": {"loss": {"_target_": "torch.nn.CrossEntropyLoss"}},
        }
    )
    batch = Data(
        x=torch.tensor([[0.0], [1.0]]),
        y=torch.tensor([0, 1]),
    )

    run = FakeRun()
    training_loop(
        [batch], [batch], BatchLinear(), [], run, cfg, {4: 0, 12: 1}
    )

    with open(tmp_path / "test-run" / "label_to_index.json") as file:
        records = json.load(file)
    assert records == [{"label": 4, "index": 0}, {"label": 12, "index": 1}]
    assert {"val/loss", "train_eval/loss"} <= set(run.summary)


class FakeMantraDataset:
    """Stands in for MantraDataset: records kwargs, applies `transform` on access."""

    instances = []

    def __init__(self, split_type, transform=None, pre_transform=None, **kwargs):
        self.split_type = split_type
        self.transform = transform
        self.pre_transform = pre_transform
        self.kwargs = kwargs
        self.items = [
            Data(x=torch.ones(1, 1), betti_numbers=[1, 0, 1]),
            Data(x=torch.ones(1, 1), betti_numbers=[1, 1, 0]),
        ]
        FakeMantraDataset.instances.append(self)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        data = self.items[index].clone()
        return data if self.transform is None else self.transform(data)


def test_create_dataset_keeps_task_transform_out_of_the_cache(monkeypatch):
    import main as main_module

    FakeMantraDataset.instances.clear()
    monkeypatch.setattr(main_module, "MantraDataset", FakeMantraDataset)
    cfg = OmegaConf.create(
        {
            "torch_seed": 0,
            "task": {
                "name": "Orientability",
                "task_type": "classification",
                "task_level": "graph",
            },
            "dataset": {
                "type": "mantra",
                "transform": {},
                "transform_order": [],
                "root": "unused",
                "dimension": 2,
                "version": "latest",
                "balanced": False,
                "local_path": None,
                "seed": 0,
                "reload": False,
                "train_parameters": {"split": {"train": 0.6, "val": 0.2, "test": 0.2}},
                "min_sample_per_class": 0,
                "use_stratified": False,
                "max_vertices": None,
                "n_moves": 1,
                "target_count": None,
                "use_surgery": False,
                "max_ood_size_per_class": 100,
                "ood_evaluations": {"bary1": {"division_type": "barycentric"}},
            },
        }
    )

    train, val, test_ds, ood, label_to_index = main_module.create_dataset(cfg)

    assert label_to_index == {0: 0, 1: 1}
    assert list(ood) == ["ood_bary1"]
    for dataset in (train, val, test_ds, *ood.values()):
        assert len(dataset.pre_transform.transforms) == 0
        assert [int(dataset[i].y) for i in range(2)] == [1, 0]
    assert ood["ood_bary1"].kwargs["division_type"] == "barycentric"


def test_create_dataset_follows_transform_order_with_gaps(monkeypatch):
    import main as main_module

    FakeMantraDataset.instances.clear()
    monkeypatch.setattr(main_module, "MantraDataset", FakeMantraDataset)
    cfg = OmegaConf.create(
        {
            "torch_seed": 0,
            "task": {
                "name": "Orientability",
                "task_type": "classification",
                "task_level": "graph",
            },
            "dataset": {
                "type": "mantra",
                "transform": {
                    "second": {"_target_": "torch_geometric.transforms.ToUndirected"},
                    "first": {"_target_": "torch_geometric.transforms.ToUndirected"},
                },
                "transform_order": ["first", "unused", "second"],
                "root": "unused",
                "dimension": 2,
                "version": "latest",
                "balanced": False,
                "local_path": None,
                "seed": 0,
                "reload": False,
                "train_parameters": {"split": {"train": 0.6, "val": 0.2, "test": 0.2}},
                "min_sample_per_class": 0,
                "use_stratified": False,
                "max_vertices": None,
                "n_moves": 1,
                "target_count": None,
                "use_surgery": False,
                "max_ood_size_per_class": 100,
                "ood_evaluations": {},
            },
        }
    )

    train, *_ = main_module.create_dataset(cfg)

    assert len(train.pre_transform.transforms) == 2
    assert all(t is not None for t in train.pre_transform.transforms)

    cfg.dataset.transform_order = ["first"]
    with pytest.raises(ValueError, match="second"):
        main_module.create_dataset(cfg)
