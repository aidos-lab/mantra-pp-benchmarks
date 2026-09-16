import pytest
import torch
from torch_geometric.data import Data

from src.transforms import RemapLabels


def test_remap_labels_uses_y_and_preserves_semantic_label():
    data = Data(label="torus", y=torch.tensor(12))

    result = RemapLabels({4: 0, 12: 1})(data)

    assert result.label == "torus"
    assert result.y.dtype == torch.long
    assert result.y.item() == 1


def test_remap_labels_keeps_per_vertex_shape():
    data = Data(y=torch.tensor([12, 4, 12]))

    result = RemapLabels({4: 0, 12: 1})(data)

    assert result.y.dtype == torch.long
    assert result.y.tolist() == [1, 0, 1]


def test_remap_labels_rejects_unseen_label():
    with pytest.raises(KeyError):
        RemapLabels({4: 0})(Data(y=torch.tensor([4, 7])))
