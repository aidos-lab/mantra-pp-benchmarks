import torch
import torch.nn.functional as F
from torch_geometric.transforms import BaseTransform


class AppendOneHot(BaseTransform):
    """Append the one-hot encoding of an integer node attribute to a feature tensor.

    Used on the Levi graph to give the model the ``node_type`` of every
    node (0-simplex or maximal simplex) next to its positional features,
    and on the Hasse diagram for the ``rank``, so that a sum readout can
    count the parts. Without ``src`` the one-hot is the whole feature
    tensor: a label-free encoding that carries nothing but the counts.

    Parameters
    ----------
    src : str or None
        Name of the ``[N, F]`` feature tensor to extend; ``None`` starts
        from no features.
    attr : str
        Name of the ``[N]`` integer attribute to encode.
    num_classes : int
        Number of one-hot columns.
    dst : str or None
        Name under which the ``[N, F + num_classes]`` result is stored;
        defaults to ``src``, or to ``attr`` when there is no ``src``.
    """

    def __init__(
        self,
        src: str | None,
        attr: str = "node_type",
        num_classes: int = 2,
        dst: str | None = None,
    ):
        super().__init__()
        self.src = src
        self.attr = attr
        self.num_classes = num_classes
        self.dst = dst or src or attr

    def forward(self, data):
        one_hot = F.one_hot(data[self.attr].long(), self.num_classes).float()
        if self.src is None:
            data[self.dst] = one_hot
        else:
            features = data[self.src]
            data[self.dst] = torch.cat([features, one_hot.to(features.dtype)], dim=-1)
        return data
