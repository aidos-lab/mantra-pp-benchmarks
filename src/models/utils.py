from torch import nn 
from torch_geometric.utils import scatter

def mlp(channels, dropout=0.0, layer_norm=True):
    """Linear -> LayerNorm -> ReLU -> Dropout per consecutive `channels` pair.

    LayerNorm rather than BatchNorm: point sets have variable size, and
    per-point statistics keep the layers independent of the batch.
    """
    layers = []
    for c_in, c_out in zip(channels[:-1], channels[1:]):
        layers.append(nn.Linear(c_in, c_out))
        if layer_norm:
            layers.append(nn.LayerNorm(c_out))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def pool(x, batch, reduce="mean"):
    """Readout of the rows of `x` into one row per graph of `batch`."""
    return scatter(x, batch, dim=-2, reduce=reduce)


def target_rows(x, batch, task_level):
    """Rows of `x` that carry the targets of `task_level`.

    `node`: the vertex rows; representations whose nodes are not all
    vertices (e.g. the Levi graph) mark them in `vertex_mask`.
    `triangle`: the sum of the three vertex rows of every triangle in
    `simplex_index_2`, symmetric like the intersection numbers.
    """
    if task_level == "triangle":
        return x[batch.simplex_index_2].sum(0)
    mask = getattr(batch, "vertex_mask", None)
    return x if mask is None else x[mask]


class SumReadout(nn.Module):
    def __init__(self, in_features, out_features):
        super(SumReadout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection_needed = in_features != out_features
        if self.projection_needed:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, signal_belongings):
        if self.projection_needed:
            x = self.linear(x)
        x = scatter(x, signal_belongings, dim=-2, reduce='sum')
        return x
