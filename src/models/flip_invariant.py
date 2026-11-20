from typing import List, Tuple

import torch
from torch import nn
from torch_sparse import SparseTensor

from src.models.utils import SumReadout


def _reconstruct_sparse(batch, name: str) -> SparseTensor:
    """Rebuild a batched SparseTensor from the `<name>_idx`/`<name>_val` COO pair
    produced by `collate_simplicial` (same convention as `SCCNN._reconstruct`)."""
    idx = batch[f"{name}_idx"]
    val = batch[f"{name}_val"]
    if "incidence" in name:
        r_to = int(name.split("_")[-1])
        r_from = r_to - 1
    else:
        r_from = r_to = int(name.split("_")[-1])
    n_rows = batch[f"x_{r_from}"].size(0)
    n_cols = batch[f"x_{r_to}"].size(0)
    return SparseTensor(row=idx[0], col=idx[1], value=val, sparse_sizes=(n_rows, n_cols))


def build_link_operator(up_laplacian_1: SparseTensor, incidence_1: SparseTensor) -> SparseTensor:
    r"""Edge -> vertex "link" operator L: R^{n0} -> R^{n1}, with

        L[e, z] = 1  iff z completes edge e to one of its incident triangles
                     (i.e. z is in the *link* of e, lk(e)),
        L[e, z] = 0  otherwise.

    For a closed 2-manifold triangulation every edge has exactly two incident
    triangles, so lk(e) = {p, q}, the two apex vertices. This operator is the
    other half of the edge-flip-invariant coboundary

        wE(e, i) = wV(u, i) + wV(v, i) - sum_{z in lk(e)} wV(z, i - 1)

    (the first half being the plain unsigned incidence-transpose "sum of
    endpoints" operator, `incidence_1.t()`).

    Derivation: with U1 = up_laplacian_1 (unsigned, = B2 @ B2^T) and
    B1t = incidence_1^T (unsigned vertex-edge incidence, transposed), one has
    (U1 @ B1t)[e, z] = 2 * [z in e] + [z in lk(e)] for every interior edge e
    (each of e's two incident triangles contributes 1 for z in lk(e), and 2
    for z in e since both endpoints belong to both incident triangles). So

        L = (U1 @ B1t) / 2 - 2 * B1t

    recovers exactly the link indicator. This assumes a manifold without
    boundary (every edge has exactly two incident triangles), matching the
    setting the operator is defined for.
    """
    b1t_dense = incidence_1.t().to_dense()
    up1_dense = up_laplacian_1.to_dense()
    link_dense = (up1_dense @ b1t_dense) / 2.0 - 2.0 * b1t_dense
    return SparseTensor.from_dense(link_dense)


class FlipInvariantLayer(nn.Module):
    r"""One propagation step of the flip-invariant simplicial network.

    Vertices are smoothed with the graph Hodge Laplacian; edges are then built
    from vertex features via the flip-invariant coboundary

        x1' = Theta_S(S @ x0_new) - Theta_L(L @ x0_old) + Theta_x1(x1),

    whose first two terms are the linear-algebra form of
    `wE(e,i) = wV(u,i)+wV(v,i) - sum_{lk(e)} wV(z,i-1)`, invariant to a local
    edge-flip (2-2 Pachner move) of e's neighboring triangulation. The third
    term is a plain residual pull from the incoming edge state (so real input
    edge features are used rather than discarded) and is not part of the
    invariant construction itself. Triangles are updated by ordinary
    down-Laplacian smoothing plus a pull from the freshly computed edge
    features.
    """

    def __init__(self, hidden_channels: List[int]):
        super().__init__()
        h0, h1, h2 = hidden_channels
        self.vertex_lin = nn.Linear(h0, h0)
        self.theta_s = nn.Linear(h0, h1, bias=False)
        self.theta_l = nn.Linear(h0, h1, bias=False)
        # residual pull from the incoming edge state, so real edge-level input
        # features (e.g. propagated embeddings) aren't simply discarded by the
        # flip-invariant term, which otherwise depends only on vertex features
        self.theta_x1 = nn.Linear(h1, h1, bias=False)
        self.tri_self_lin = nn.Linear(h2, h2)
        self.tri_edge_lin = nn.Linear(h1, h2, bias=False)
        self.act = nn.ReLU()

    def forward(
        self,
        x_all: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        laplacians: Tuple[SparseTensor, SparseTensor],
        incidences: Tuple[SparseTensor, SparseTensor],
        link_operator: SparseTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_0, x_1, x_2 = x_all
        hodge_laplacian_0, down_laplacian_2 = laplacians
        b1, b2 = incidences

        x_0_new = self.act(self.vertex_lin(hodge_laplacian_0 @ x_0))
        x_1_new = self.act(
            self.theta_s(b1.t() @ x_0_new)
            - self.theta_l(link_operator @ x_0)
            + self.theta_x1(x_1)
        )
        x_2_new = self.act(
            self.tri_self_lin(down_laplacian_2 @ x_2)
            + self.tri_edge_lin(b2.t() @ x_1_new)
        )

        return x_0_new, x_1_new, x_2_new


class FlipInvariantBackbone(nn.Module):
    def __init__(self, in_channels: List[int], hidden_channels: List[int], n_layers: int = 2):
        super().__init__()
        self.in_linear_0 = nn.Linear(in_channels[0], hidden_channels[0])
        self.in_linear_1 = nn.Linear(in_channels[1], hidden_channels[1])
        self.in_linear_2 = nn.Linear(in_channels[2], hidden_channels[2])
        self.layers = nn.ModuleList(
            FlipInvariantLayer(hidden_channels) for _ in range(n_layers)
        )

    def forward(self, x_all, laplacians, incidences, link_operator):
        x_0, x_1, x_2 = x_all
        x_all = (self.in_linear_0(x_0), self.in_linear_1(x_1), self.in_linear_2(x_2))
        for layer in self.layers:
            x_all = layer(x_all, laplacians, incidences, link_operator)
        return x_all


class FlipInvariantSCNN(nn.Module):
    r"""Simplicial-complex network whose vertex-to-edge propagation is the
    edge-flip-invariant coboundary operator (the "S - L" construction) instead
    of a generic Hodge-Laplacian convolution.

    Parameters
    ----------
    in_channels : list of int
        Input feature width per dimension, [dim(x_0), dim(x_1), dim(x_2)].
    hidden_channels : list of int
        Hidden feature width per dimension.
    out_channels : int
        Number of output classes.
    n_layers : int
        Number of stacked `FlipInvariantLayer`s.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: List[int],
        out_channels: int = 3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.backbone = FlipInvariantBackbone(in_channels, hidden_channels, n_layers)
        self.readout_0 = SumReadout(hidden_channels[0], out_channels)
        self.readout_1 = SumReadout(hidden_channels[1], out_channels)
        self.readout_2 = SumReadout(hidden_channels[2], out_channels)

    def forward(self, batch):
        x = {
            int(k.split("_")[-1]): batch[k]
            for k in batch.keys()
            if isinstance(k, str) and k.split("_")[0] == "x" and k.split("_")[1].isdigit()
        }
        batches = {
            int(k.split("_")[-1]): batch[k]
            for k in batch.keys()
            if isinstance(k, str) and k.split("_")[0] == "batch" and k.split("_")[1].isdigit()
        }
        x_0, x_1, x_2 = x[0], x[1], x[2]
        x_bel_0, x_bel_1, x_bel_2 = batches[0], batches[1], batches[2]
        assert x_bel_0.max() == x_bel_1.max() == x_bel_2.max(), (
            f"{x_bel_0.max().item()}, {x_bel_1.max().item()}, {x_bel_2.max().item()}"
        )

        hodge_laplacian_0 = _reconstruct_sparse(batch, "hodge_laplacian_0")
        down_laplacian_2 = _reconstruct_sparse(batch, "down_laplacian_2")
        up_laplacian_1 = _reconstruct_sparse(batch, "up_laplacian_1")
        incidence_1 = _reconstruct_sparse(batch, "incidence_1")
        incidence_2 = _reconstruct_sparse(batch, "incidence_2")

        link_operator = build_link_operator(up_laplacian_1, incidence_1)

        x_0, x_1, x_2 = self.backbone(
            (x_0, x_1, x_2),
            (hodge_laplacian_0, down_laplacian_2),
            (incidence_1, incidence_2),
            link_operator,
        )

        out_0 = self.readout_0(x_0, x_bel_0)
        out_1 = self.readout_1(x_1, x_bel_1)
        out_2 = self.readout_2(x_2, x_bel_2)
        return out_0 + out_1 + out_2
