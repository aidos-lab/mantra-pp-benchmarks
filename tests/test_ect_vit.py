import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from main import create_model
from src.models.ect_vit import ECTViT
from src.utils.baselines import feature_stats

import pytest


def _batch(n, directions=8, resolution=6):
    return Batch.from_data_list(
        [Data(ect=torch.randint(-50, 2, (1, directions, resolution), dtype=torch.int16)) for _ in range(n)]
    )


def test_input_stats_standardise_before_embedding():
    torch.manual_seed(0)
    model = ECTViT(4, 3, 8, 6, hidden_channels=16, n_layers=1).eval()
    batch = _batch(5)
    mean, std = torch.randn(8, 6), torch.rand(8, 6) + 1.0
    reference = model(batch)
    model.set_input_stats(mean, std)
    batch.ect = mean + std * batch.ect.float()
    assert torch.allclose(model(batch), reference, atol=1e-4)
    assert "input_mean" in model.state_dict()


def test_feature_stats_floor_the_std_at_one():
    train = type("Split", (), {})()
    train._data = Data(ect=torch.tensor([[[0.0, 10.0]], [[0.0, 30.0]]]))
    mean, std = feature_stats(train, "ect")
    assert mean.tolist() == [[0.0, 20.0]]
    assert torch.allclose(std, torch.tensor([[1.0, 10.0 * 2**0.5]]))


def test_create_model_passes_input_stats_to_the_ect_vit():
    cfg = OmegaConf.create(
        {
            "task": {"task_type": "classification", "task_level": "graph"},
            "trainer": {"device": "cpu"},
            "compile": False,
            "model": {
                "_target_": "src.models.ect_vit.ECTViT",
                "num_directions": 8,
                "resolution": 6,
                "hidden_channels": 16,
                "n_layers": 1,
                "out_channels": 1,
                "task_level": "graph",
            },
        }
    )
    stats = (torch.full((8, 6), 2.0), torch.full((8, 6), 3.0))
    model = create_model(cfg, 4, {0: 0, 1: 1}, input_stats=stats)
    assert model.input_std.unique().item() == 3.0
    assert model(_batch(2)).shape == (2, 2)


def test_feature_stats_match_torch_in_chunks():
    train = type("Split", (), {})()
    train._data = Data(ect=torch.randint(-50, 2, (10, 3, 4), dtype=torch.int16))
    mean, std = feature_stats(train, "ect", chunk=3)
    values = train._data.ect.float()
    assert torch.allclose(mean, values.mean(0), atol=1e-5)
    assert torch.allclose(std, values.std(0).clamp_min(1.0), atol=1e-4)


def test_per_rank_input_fills_one_token_per_direction():
    torch.manual_seed(0)
    model = ECTViT(4, 3, 8, 6, num_ranks=2, hidden_channels=16, n_layers=1).eval()
    batch = Batch.from_data_list(
        [Data(ect=torch.randint(0, 9, (1, 2, 8, 6), dtype=torch.int16)) for _ in range(3)]
    )
    assert model.input_mean.shape == (2, 8, 6)
    assert model(batch).shape == (3, 3)
    with pytest.raises(ValueError):
        model(_batch(2))


def test_node_level_predicts_every_point_in_batch_order():
    torch.manual_seed(0)
    model = ECTViT(4, 2, 8, 6, hidden_channels=16, n_layers=1, task_level="node").eval()
    sets = [
        Data(ect=torch.randint(-50, 2, (1, 8, 6), dtype=torch.int16), x=torch.randn(n, 4))
        for n in (3, 5)
    ]
    out = model(Batch.from_data_list(sets))
    assert out.shape == (8, 2)
    alone = model(Batch.from_data_list(sets[:1]))
    assert torch.allclose(out[:3], alone, atol=1e-5)
