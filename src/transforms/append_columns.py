import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform


class AppendColumns(BaseTransform):
    """Append a per-vertex attribute to another as further columns."""

    def __init__(self, source: str, target: str):
        """
        Parameters
        ----------
        source : str
            Attribute with one row per vertex, of shape `(n,)` or `(n, k)`.
        target : str
            Attribute of shape `(n, d)` that receives the columns.
        """
        super().__init__()
        self.source = source
        self.target = target

    def forward(self, data: Data) -> Data:
        target = data[self.target]
        columns = data[self.source].reshape(target.shape[0], -1)
        data[self.target] = torch.cat([target, columns.to(target.dtype)], dim=1)
        return data
