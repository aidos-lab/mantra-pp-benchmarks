import torch
from torch_geometric.data import Data

from src.transforms import AppendOneHot


def test_append_one_hot_extends_the_feature_tensor():
    data = Data(
        feat=torch.tensor([[1.0], [2.0], [3.0]]),
        node_type=torch.tensor([0, 1, 1]),
    )

    result = AppendOneHot(src="feat")(data)

    expected = torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]])
    assert torch.equal(result.feat, expected)
    assert result.feat.dtype == torch.float32


def test_one_hot_alone_is_the_feature_tensor():
    data = Data(rank=torch.tensor([0, 1, 2, 1]))

    result = AppendOneHot(src=None, attr="rank", num_classes=3)(data)

    expected = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]]).float()
    assert torch.equal(result.rank, expected)  # stored under attr by default
    assert result.rank.dtype == torch.float32


def test_dst_keeps_the_source():
    data = Data(feat=torch.ones(2, 1), node_type=torch.tensor([0, 1]))

    result = AppendOneHot(src="feat", dst="out")(data)

    assert torch.equal(result.feat, torch.ones(2, 1))
    assert result.out.shape == (2, 3)
