import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

from main import create_loaders, training_loop


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
    dataset = [Data(x=torch.ones(1, 1), y=torch.tensor(0))]

    train, val, test, ood_loaders = create_loaders(
        cfg, dataset, dataset, dataset, {"ood_bary1": dataset}
    )

    resident = num_workers > 0
    assert train.persistent_workers == resident
    assert val.persistent_workers == resident
    assert not test.persistent_workers
    assert list(ood_loaders) == ["ood_bary1"]
    assert not any(loader.persistent_workers for loader in ood_loaders.values())


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

    training_loop(
        [batch], [batch], BatchLinear(), [], FakeRun(), cfg, {4: 0, 12: 1}
    )

    with open(tmp_path / "test-run" / "label_to_index.json") as file:
        records = json.load(file)
    assert records == [{"label": 4, "index": 0}, {"label": 12, "index": 1}]
