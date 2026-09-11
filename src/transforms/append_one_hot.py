import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform


class AppendOneHot(BaseTransform):
    """Append the one-hot encoding of an integer node attribute to a feature tensor.

    Used on the Levi graph to give the model the ``node_type`` of every
    node (0-simplex or maximal simplex) next to its positional features,
    so that a sum readout can count the two parts of the bipartition.

    Parameters
    ----------
    src : str
        Name of the ``[N, F]`` feature tensor to extend; it is replaced
        by the ``[N, F + num_classes]`` result.
    attr : str
        Name of the ``[N]`` integer attribute to encode.
    num_classes : int
        Number of one-hot columns.
    """

    def __init__(self, src: str, attr: str = "node_type", num_classes: int = 2):
        super().__init__()
        self.src = src
        self.attr = attr
        self.num_classes = num_classes

    def forward(self, data):
        features = data[self.src]
        one_hot = F.one_hot(data[self.attr].long(), self.num_classes)
        data[self.src] = torch.cat([features, one_hot.to(features.dtype)], dim=-1)
        return data
