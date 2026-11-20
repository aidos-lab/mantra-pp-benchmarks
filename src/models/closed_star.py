from typing import List, Tuple

import torch
from torch import nn
from torch_sparse import SparseTensor

from src.models.flip_invariant import _reconstruct_sparse
from src.models.utils import SumReadout


class ClosedStarLayer(nn.Module):
    """
    For each dimension the update combines:
      - `hodge_laplacian_d @ x_d`
      - `incidence_d^T @ x_{d-1}` 
      - `incidence_{d+1} @ x_{d+1}` 
    """

    def __init__(self, hidden_channels: List[int]):
        super().__init__()
        h0, h1, h2 = hidden_channels

        self.theta_v_self = nn.Linear(h0, h0)
        self.theta_v_div = nn.Linear(h1, h0, bias=False)

        self.theta_e_self = nn.Linear(h1, h1)
        self.theta_e_grad = nn.Linear(h0, h1, bias=False)
        self.theta_e_div = nn.Linear(h2, h1, bias=False)

        self.theta_e_res = nn.Linear(h1, h1, bias=False)

        self.theta_t_self = nn.Linear(h2, h2)
        self.theta_t_curl = nn.Linear(h1, h2, bias=False)

        self.act = nn.ReLU()

    def forward(
        self,
        x_all: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        hodge_laplacians: Tuple[SparseTensor, SparseTensor, SparseTensor],
        incidences: Tuple[SparseTensor, SparseTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_0, x_1, x_2 = x_all
        l0, l1, l2 = hodge_laplacians
        b1, b2 = incidences

        x_0_new = self.act(
            self.theta_v_self(l0 @ x_0)
            + self.theta_v_div(b1 @ x_1)
        )
        x_1_new = self.act(
            self.theta_e_self(l1 @ x_1)
            + self.theta_e_grad(b1.t() @ x_0_new)
            + self.theta_e_div(b2 @ x_2)
            + self.theta_e_res(x_1)
        )
        x_2_new = self.act(
            self.theta_t_self(l2 @ x_2)
            + self.theta_t_curl(b2.t() @ x_1_new)
        )

        return x_0_new, x_1_new, x_2_new


class ClosedStarBackbone(nn.Module):
    def __init__(self, in_channels: List[int], hidden_channels: List[int], n_layers: int = 2):
        super().__init__()
        self.in_linear_0 = nn.Linear(in_channels[0], hidden_channels[0])
        self.in_linear_1 = nn.Linear(in_channels[1], hidden_channels[1])
        self.in_linear_2 = nn.Linear(in_channels[2], hidden_channels[2])
        self.layers = nn.ModuleList(
            ClosedStarLayer(hidden_channels) for _ in range(n_layers)
        )

    def forward(self, x_all, hodge_laplacians, incidences):
        x_0, x_1, x_2 = x_all
        x_all = (self.in_linear_0(x_0), self.in_linear_1(x_1), self.in_linear_2(x_2))
        for layer in self.layers:
            x_all = layer(x_all, hodge_laplacians, incidences)
        return x_all


class ClosedStarSCNN(nn.Module):
    """
    Parameters
    ----------
    in_channels : list of int
        Input feature width per dimension, [dim(x_0), dim(x_1), dim(x_2)].
    hidden_channels : list of int
        Hidden feature width per dimension.
    out_channels : int
        Number of output classes.
    n_layers : int
        Number of stacked `ClosedStarLayer`s.
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: List[int],
        out_channels: int = 3,
        n_layers: int = 2,
    ):
        super().__init__()
        self.backbone = ClosedStarBackbone(in_channels, hidden_channels, n_layers)
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
        hodge_laplacian_1 = _reconstruct_sparse(batch, "hodge_laplacian_1")
        hodge_laplacian_2 = _reconstruct_sparse(batch, "hodge_laplacian_2")
        incidence_1 = _reconstruct_sparse(batch, "incidence_1")
        incidence_2 = _reconstruct_sparse(batch, "incidence_2")

        x_0, x_1, x_2 = self.backbone(
            (x_0, x_1, x_2),
            (hodge_laplacian_0, hodge_laplacian_1, hodge_laplacian_2),
            (incidence_1, incidence_2),
        )

        out_0 = self.readout_0(x_0, x_bel_0)
        out_1 = self.readout_1(x_1, x_bel_1)
        out_2 = self.readout_2(x_2, x_bel_2)
        return out_0 + out_1 + out_2
