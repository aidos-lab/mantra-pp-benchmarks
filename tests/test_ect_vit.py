import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.ect_vit import ECTViT
from src.utils.feature_stats import feature_stats


def _batch(n, shape=(8, 6), points=None):
    return Batch.from_data_list(
        [
            Data(ect=torch.randint(-50, 2, (1, *shape), dtype=torch.int16), x=torch.randn(p, 4))
            for p in (points or [1] * n)
        ]
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


def test_feature_stats_match_torch_in_chunks_with_the_std_floored():
    train = type("Split", (), {})()
    train._data = Data(ect=torch.randint(-50, 2, (10, 3, 4), dtype=torch.int16))
    train._data.ect[:, 0, 0] = 0
    mean, std = feature_stats(train, "ect", chunk=3)
    values = train._data.ect.float()
    assert torch.allclose(mean, values.mean(0), atol=1e-5)
    assert torch.allclose(std, values.std(0).clamp_min(1.0), atol=1e-4)
    assert std[0, 0] == 1.0


def test_per_rank_input_fills_one_token_per_direction():
    torch.manual_seed(0)
    model = ECTViT(4, 3, 8, 6, num_ranks=2, hidden_channels=16, n_layers=1).eval()
    assert model(_batch(3, shape=(2, 8, 6))).shape == (3, 3)
    with pytest.raises(ValueError):
        model(_batch(2))


def test_node_level_predicts_every_point_in_batch_order():
    torch.manual_seed(0)
    model = ECTViT(4, 2, 8, 6, hidden_channels=16, n_layers=1, task_level="node").eval()
    batch = _batch(2, points=[3, 5])
    out = model(batch)
    assert out.shape == (8, 2)
    alone = model(Batch.from_data_list(batch.to_data_list()[:1]))
    assert torch.allclose(out[:3], alone, atol=1e-5)
