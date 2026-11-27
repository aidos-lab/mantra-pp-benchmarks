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


def resolve_max_rank(max_rank, in_channels):
    """Top rank a simplicial model works on.

    `in_channels` has one entry per rank of the data. None takes every
    rank, so a model follows the dimension of its dataset; an int keeps
    the ranks 0..max_rank and ignores the simplices above them.
    """
    available = len(in_channels) - 1
    if max_rank is None:
        max_rank = available
    if not 2 <= max_rank <= available:
        raise ValueError(
            f"max_rank must be between 2 and the top rank of the data "
            f"({available}), got {max_rank}"
        )
    return max_rank


def per_rank(channels, num_ranks):
    """`channels` as a list with one width per rank; an int is repeated."""
    if isinstance(channels, int):
        return [channels] * num_ranks
    channels = list(channels)
    if len(channels) != num_ranks:
        raise ValueError(
            f"hidden_channels has {len(channels)} entries for the ranks "
            f"0..{num_ranks - 1}: pass one per rank or a single int"
        )
    return channels


def pool(x, batch, reduce="mean"):
    """Readout of the rows of `x` into one row per graph of `batch`."""
    return scatter(x, batch, dim=-2, reduce=reduce)


def target_rows(x, batch, task_level):
    """Rows of `x` that carry the targets of `task_level`.

    `node`: the vertex rows; representations whose nodes are not all
    vertices (e.g. the Levi graph) mark them in `vertex_mask`.
    `triangle`: the sum of the three vertex rows of every boundary
    triangle in `simplex_index_2`, symmetric like the intersection numbers;
    where the triangles have rows of their own (`triangle_node_index` of
    `src.transforms.SimplexPoints`), those rows.
    `candidate`: the same sum over the candidate triangles of the 2-faces
    in `candidate_index_2`, present in the triangulation or not.
    """
    if task_level == "triangle":
        if "triangle_node_index" in batch:
            return x[batch.triangle_node_index]
        return x[batch.simplex_index_2].sum(0)
    if task_level == "candidate":
        return x[batch.candidate_index_2].sum(0)
    mask = getattr(batch, "vertex_mask", None)
    return x if mask is None else x[mask]


def triangle_rows(x, batch):
    """Rows of the per-triangle states `x` that carry targets: the
    boundary triangles, which the `kappa` transform marks in
    `triangle_mask`. Raises instead of falling back to every row, so a
    cache without the mask cannot pass as the triangle task."""
    if "triangle_mask" not in batch:
        raise ValueError("the triangle task needs `triangle_mask` in the batch")
    return x[batch.triangle_mask]


def head(in_features, out_features, hidden_layers=0):
    """Readout of every model: `hidden_layers` blocks of `mlp`, then a
    linear layer; the plain linear layer when 0.

    Shared so that `model.head_layers` gives all models the same capacity
    after their symmetric sum, where a linear layer is additive in the
    summed rows and cannot fit the triangle targets. The hidden layers
    use LayerNorm and no dropout whatever the model's own settings.
    """
    if hidden_layers < 0:
        raise ValueError(f"hidden_layers must be >= 0, got {hidden_layers}")
    if hidden_layers == 0:
        return nn.Linear(in_features, out_features)
    return nn.Sequential(
        mlp([in_features] * (hidden_layers + 1)),
        nn.Linear(in_features, out_features),
    )


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
