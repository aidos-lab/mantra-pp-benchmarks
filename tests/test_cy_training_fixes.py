import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from main import TargetAffine, create_model
from src.models.graph.gcn import GCN
from src.models.graph.gin import GIN
from src.models.graph.res_gcn import ResidualGatedGCN
from src.utils.baselines import regression_target_stats, trivial_baselines


def _graph_batch(sizes=(5, 300), features=4):
    graphs = [
        Data(
            x=torch.randn(n, features),
            edge_index=torch.stack([torch.arange(n), torch.roll(torch.arange(n), 1)]),
            y=torch.tensor([[70.0]]),
        )
        for n in sizes
    ]
    return Batch.from_data_list(graphs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda norm: GCN(4, 16, 1, 2, True, True, 0.0, "sum", "graph", readout_norm=norm),
        lambda norm: GIN(4, 16, 1, 2, True, True, 0.0, "sum", "graph", readout_norm=norm),
        lambda norm: ResidualGatedGCN(16, 2, 4, 1, True, True, 0.0, "sum", "graph", readout_norm=norm),
    ],
)
def test_readout_norm_is_a_layer_norm_after_pooling(factory):
    torch.manual_seed(0)
    batch = _graph_batch()
    plain, normed = factory(False), factory(True)
    assert isinstance(plain.readout.norm, torch.nn.Identity)
    assert isinstance(normed.readout.norm, torch.nn.LayerNorm)

    normed.train()
    assert normed(batch).shape == (2, 1)
    assert plain(batch).shape == (2, 1)


def test_create_model_wraps_regression_heads_only():
    cfg = OmegaConf.create(
        {
            "task": {"task_type": "regression", "task_level": "graph"},
            "trainer": {"device": "cpu"},
            "compile": False,
            "model": {
                "_target_": "src.models.graph.gcn.GCN",
                "hidden_channels": 8,
                "n_layers": 1,
                "out_channels": 1,
                "batch_norm": True,
                "residuals": True,
                "dropout": 0.0,
                "pool_method": "sum",
                "task_level": "graph",
            },
        }
    )
    model = create_model(cfg, 4, {}, target_stats=(70.0, 16.0))
    assert isinstance(model, TargetAffine)
    assert model.std.item() == 16.0

    cfg.task.task_type = "classification"
    model = create_model(cfg, 4, {0: 0, 1: 1}, target_stats=(70.0, 16.0))
    assert isinstance(model, GCN)
    assert cfg.model.out_channels == 2


class _Split:
    """Minimal InMemoryDataset stand-in: concatenated raw columns."""

    def __init__(self, h11, c2):
        self._data = {"h11": torch.tensor(h11), "c2": torch.cat([torch.tensor(c) for c in c2])}


def test_trivial_baselines_and_target_stats():
    train = _Split([2, 2, 3], [[-6.0, 3.0, 3.0], [-4.0, 2.0, 2.0], [-10.0, 5.0, 5.0]])
    test = _Split([2, 3], [[-8.0, 4.0, 4.0], [-2.0, 1.0, 1.0]])

    cls = OmegaConf.create({"task": {"task_type": "classification", "task_level": "graph", "target": "h11"}})
    assert trivial_baselines(cls, train, test) == {
        "baseline/test_accuracy": 0.5,
        "baseline/test_balanced_accuracy": 0.5,
    }

    node = OmegaConf.create({"task": {"task_type": "regression", "task_level": "node", "target": "c2"}})
    mean, _ = regression_target_stats(node, train)
    assert mean == pytest.approx(0.0)
    # Train entries -6,3,3,-4,2,2,-10,5,5 (median 2); test entries -8,4,4,-2,1,1.
    assert trivial_baselines(node, train, test)["baseline/test_mae"] == pytest.approx(20 / 6)

    graph = OmegaConf.create({"task": {"task_type": "regression", "task_level": "graph", "target": "h11"}})
    assert trivial_baselines(graph, train, test)["baseline/test_mae"] == pytest.approx(0.5)

