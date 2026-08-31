import pytest
import torch
from types import SimpleNamespace
from torch_geometric.data import Data

from src.utils.train_utils import make_labels, test as run_test


def test_record_class_labels_uses_labels_paired_with_targets():
    index_to_label = {}

    make_labels(
        index_to_label,
        torch.tensor([1, 0, 1]),
        ["torus", "sphere", "torus"],
    )

    assert index_to_label == {0: "sphere", 1: "torus"}


def test_record_class_labels_rejects_inconsistent_names():
    with pytest.raises(AssertionError, match="label mismatch"):
        make_labels({}, torch.tensor([0, 0]), ["torus", "sphere"])


class IdentityModel(torch.nn.Module):
    def forward(self, batch):
        return batch.x


class FakeRun:
    def __init__(self):
        self.logged = []

    def log(self, values):
        self.logged.append(values)


def test_regression_test_loop_does_not_build_confusion_matrix():
    run = FakeRun()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(device="cpu"),
        task=SimpleNamespace(task_type="regression"),
    )
    batch = Data(x=torch.tensor([[1.5]]), y=torch.tensor([[1.0]]))

    run_test(IdentityModel(), [batch], [], {}, run, cfg, epoch=0)

    assert all("conf_mat" not in values for values in run.logged)


def test_missing_test_class_uses_original_numeric_label(monkeypatch):
    captured = {}

    def fake_confusion_matrix(**kwargs):
        captured.update(kwargs)
        return "plot"

    monkeypatch.setattr(
        "src.utils.train_utils.wandb.plot.confusion_matrix",
        fake_confusion_matrix,
    )
    run = FakeRun()
    cfg = SimpleNamespace(
        trainer=SimpleNamespace(device="cpu"),
        task=SimpleNamespace(task_type="classification"),
    )
    batch = Data(
        x=torch.tensor([[2.0, 0.0]]),
        y=torch.tensor([0]),
        label="sphere",
    )

    run_test(IdentityModel(), [batch], [], {4: 0, 12: 1}, run, cfg, epoch=0)

    assert captured["class_names"] == ["sphere", "12"]
