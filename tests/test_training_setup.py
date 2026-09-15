import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from main import create_loaders, create_model, training_loop


def test_create_loaders_disables_persistence_without_workers():
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            sampler="random",
            materialize=False,
        ),
        dataset=SimpleNamespace(
            transform=SimpleNamespace(
                select_features=SimpleNamespace(representation="graph")
            )
        ),
    )
    dataset = [Data(x=torch.ones(1, 1), y=torch.tensor(0))]

    train_loader, val_loader, test_loader, ood_loaders = create_loaders(
        cfg, dataset, dataset, dataset
    )

    assert ood_loaders == {}
    assert all(
        not loader.persistent_workers
        for loader in (train_loader, val_loader, test_loader)
    )


def test_create_model_updates_runtime_output_channels(monkeypatch):
    captured = {}

    def fake_instantiate(model_cfg, **kwargs):
        captured["configured_out_channels"] = model_cfg.out_channels
        captured.update(kwargs)
        return torch.nn.Linear(1, model_cfg.out_channels)

    monkeypatch.setattr("main.hydra.utils.instantiate", fake_instantiate)
    cfg = SimpleNamespace(
        model=SimpleNamespace(in_channels=1, out_channels=1),
        task=SimpleNamespace(task_type="classification"),
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

    def forward(self, batch, return_embedding=False):
        logits = self.linear(batch.x)
        return (logits, batch.x) if return_embedding else logits


class FakeRun:
    name = "test-run"
    id = "test-run"

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
                "learning_rate": 0.01,
                "weight_decay": 0.0,
            },
            "task": {"loss": {"_target_": "torch.nn.CrossEntropyLoss"}},
            "loss": {"_target_": "src.metrics.loss.CrossEntropyObjective"},
        }
    )
    batch = Data(
        x=torch.tensor([[0.0], [1.0]]),
        y=torch.tensor([0, 1]),
    )

    training_loop(
        [batch], [batch], BatchLinear(), [], FakeRun(), cfg, {4: 0, 12: 1}
    )

    with open(tmp_path / "test-run" / "label_to_index.json") as file:
        records = json.load(file)
    assert records == [{"label": 4, "index": 0}, {"label": 12, "index": 1}]


def test_materialize_applies_transforms_once():
    from src.utils.materialize import MaterializedDataset

    calls = []

    class Counting:
        def __len__(self):
            return 3

        def __getitem__(self, index):
            calls.append(index)
            return Data(x=torch.full((1, 1), float(index)), y=torch.tensor(index))

    materialized = MaterializedDataset(Counting())
    for _ in range(3):
        for i in range(3):
            assert materialized[i].y.item() == i

    assert len(materialized) == 3
    assert calls == [0, 1, 2]


def test_materialize_drops_the_raw_payload():
    from src.utils.materialize import MaterializedDataset

    dataset = [
        Data(x=torch.ones(1, 1), y=torch.tensor(0), triangulation=[[1, 2, 3]], n_vertices=3)
    ]

    entry = MaterializedDataset(dataset)[0]

    assert "triangulation" not in entry
    assert int(entry.n_vertices) == 3


def test_create_loaders_materializes_every_split():
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            batch_size=2,
            num_workers=4,
            pin_memory=False,
            sampler="random",
            materialize=True,
        ),
        dataset=SimpleNamespace(transform=SimpleNamespace()),
    )
    dataset = [Data(x=torch.ones(1, 1), y=torch.tensor(i)) for i in range(4)]

    train_loader, _, _, ood_loaders = create_loaders(
        cfg, dataset, dataset, dataset, {"ood_x": dataset}
    )

    # Materializing forces in-process loading.
    assert train_loader.num_workers == 0
    assert len(train_loader.dataset) == 4
    assert type(train_loader.dataset).__name__ == "MaterializedDataset"
    assert type(ood_loaders["ood_x"].dataset).__name__ == "MaterializedDataset"
    assert sum(len(b.y) for b in train_loader) == 4
