import torch
from torch_geometric.data import Data

from src.transforms import RemapLabels


def test_remap_labels_uses_y_and_preserves_semantic_label():
    data = Data(label="torus", y=torch.tensor(12))

    result = RemapLabels({4: 0, 12: 1})(data)

    assert result.label == "torus"
    assert result.y.dtype == torch.long
    assert result.y.item() == 1
