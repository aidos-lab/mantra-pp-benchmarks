import torch
from torch_geometric.transforms import BaseTransform


class RemapLabels(BaseTransform):
    """Remap scalar class labels to contiguous cross-entropy targets."""

    def __init__(self, label_to_index: dict[int, int]):
        super().__init__()
        self.label_to_index = label_to_index

    def forward(self, data):
        label = int(data.y.item())
        data.y = torch.tensor(self.label_to_index[label], dtype=torch.long)
        return data
