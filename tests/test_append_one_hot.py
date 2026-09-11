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
