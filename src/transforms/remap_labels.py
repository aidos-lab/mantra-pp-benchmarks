import torch
from torch_geometric.transforms import BaseTransform


class RemapLabels(BaseTransform):
    """Remap scalar class labels to contiguous cross-entropy targets."""

    def __init__(self, label_to_index: dict[int, int]):
        super().__init__()
        self.label_to_index = label_to_index

    def forward(self, data):
        if "y" not in data:
            raise ValueError("Cannot remap a sample without a 'y' label")
        if not isinstance(data.y, torch.Tensor) or data.y.numel() != 1:
            raise ValueError("Class labels must be scalar tensors")

        label = int(data.y.item())
        if label not in self.label_to_index:
            raise ValueError(f"Label {label} is not present in the training split")

        data.y = torch.tensor(self.label_to_index[label], dtype=torch.long)
        return data
