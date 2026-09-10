import torch
from torch_geometric.transforms import BaseTransform


class RemapLabels(BaseTransform):
    """Remap class labels, scalar or one per vertex, to contiguous targets."""

    def __init__(self, label_to_index: dict[int, int]):
        super().__init__()
        self.label_to_index = label_to_index

    def forward(self, data):
        y = data.y
        indices = [self.label_to_index[int(v)] for v in y.reshape(-1).tolist()]
        data.y = torch.tensor(indices, dtype=torch.long).reshape(y.shape)
        return data
