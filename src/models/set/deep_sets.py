"""DeepSets and PointNet over the lattice points of a polytope.

Zaheer et al., "Deep Sets", NeurIPS 2017 (arXiv:1703.06114), Theorem 2:
a permutation-invariant set function is `rho(sum_i phi(x_i))` with a
shared point-wise network `phi` and a network `rho` on the pooled
feature. With max pooling this is PointNet (Qi et al., CVPR 2017,
arXiv:1612.00593) without its input and feature transform networks
(`model.pool_method=max`). At node level a
point is predicted from its embedding concatenated with the pooled set
feature, the segmentation head of PointNet (one permutation-equivariant
layer in the terms of Deep Sets, Lemma 3). Deep Sets uses plain layers
and PointNet BatchNorm; here every layer is `utils.mlp`'s
Linear -> LayerNorm -> ReLU, which is independent of the set size.

The model reads only `batch.x` and `batch.batch`: with `x` set to the
lattice coordinates it sees the polytope and nothing of the
triangulation, the geometry-only control of the Calabi-Yau benchmarks
(README, "Point-set models").
"""

import torch
import torch.nn as nn

from src.models.utils import head, mlp, pool, target_rows


class DeepSets(nn.Module):
    """DeepSets over a batch of point sets.

    Parameters
    ----------
    in_channels : int
        Point feature size (4 for the lattice coordinates).
    hidden_channels : int
        Width of the shared point-wise network and of the head.
    out_channels : int
        Prediction width; set from the task by `main.py`.
    n_layers : int
        Number of shared point-wise layers (`phi`).
    dropout : float
        Dropout after every hidden layer.
    layer_norm : bool
        LayerNorm after every hidden linear layer.
    pool_method : str
        Symmetric aggregation over the points: `'sum'` (DeepSets, keeps
        the prediction extensive), `'mean'`, or `'max'` (PointNet).
    task_level : str
        `'graph'` for one prediction per set, `'node'` for one per point.
    head_layers : int
        Hidden layers of the readout (`head`).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        n_layers: int = 3,
        dropout: float = 0.0,
        layer_norm: bool = True,
        pool_method: str = "sum",
        task_level: str = "graph",
        head_layers: int = 0,
    ):
        super().__init__()
        self.pool_method = pool_method
        self.task_level = task_level

        h = hidden_channels
        self.phi = mlp([in_channels] + [h] * n_layers, dropout, layer_norm)
        rho_in = h if task_level == "graph" else 2 * h
        self.rho = nn.Sequential(
            mlp([rho_in, h], dropout, layer_norm),
            head(h, out_channels, head_layers),
        )

    def forward(self, batch) -> torch.Tensor:
        h = self.phi(batch.x)
        g = pool(h, batch.batch, self.pool_method)
        if self.task_level == "graph":
            return self.rho(g)
        # Each point sees its own embedding and the pooled set feature
        z = torch.cat([h, g[batch.batch]], dim=1)
        return self.rho(target_rows(z, batch, self.task_level))
