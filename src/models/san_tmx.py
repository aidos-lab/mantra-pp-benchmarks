# -*- coding: utf-8 -*-
"""
Simplicial Attention Network (SAN) — implementation inspired by TopoModelX.

Uses torch_sparse.SparseTensor throughout and precomputes the harmonic
projector once per batch in the forward pass.

Reference
---------
Giusti, Battiloro, Di Lorenzo, Sardellitti and Barbarossa.
Simplicial attention neural networks (2022).
https://arxiv.org/abs/2203.07485
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor, spmm
from torch_geometric.utils import softmax as sparse_softmax

from src.models.utils import SumReadout
from src.models.conv_down import Conv


class SANConv(nn.Module):
    """SAN convolution: attention-weighted message passing on a Laplacian.

    Computes: out = sum_{k=0}^{K-1} (att_L)^{k+1} @ x @ W_k

    where att_L is the Laplacian with learned attention coefficients,
    and the power is computed via repeated sparse matmul.

    Parameters
    ----------
    in_channels : int
        Input feature dimension.
    out_channels : int
        Output feature dimension.
    n_filters : int
        Number of filter orders (kappa).
    """

    def __init__(self, in_channels, out_channels, n_filters):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        self.weight = Parameter(
            torch.empty(n_filters, in_channels, out_channels)
        )
        self.att_weight = Parameter(
            torch.empty(2 * out_channels * n_filters)
        )

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_weight.unsqueeze(0), gain=gain)

    def forward(self, x, neighborhood):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (n_cells, in_channels)
        neighborhood : SparseTensor, shape (n_cells, n_cells)

        Returns
        -------
        torch.Tensor, shape (n_cells, out_channels)
        """
        n = x.size(0)

        # Compute messages for each filter: (n_filters, n_cells, out_channels)
        x_message = torch.matmul(x, self.weight)

        # Reshape for attention: (n_cells, out_channels * n_filters)
        x_flat = x_message.permute(1, 0, 2).reshape(n, -1)

        # Compute attention scores on the sparsity pattern of the neighborhood
        row, col, _ = neighborhood.coo()

        # Concatenate source and target features, project with att_weight
        e = torch.cat([x_flat[row], x_flat[col]], dim=-1) @ self.att_weight
        e = F.elu(e)

        # Sparse softmax over each row
        alpha = sparse_softmax(e, row, num_nodes=n)

        # Mirrors TopoModelX: out = sum_{k=0..K-1} att_lap^{k+1} @ x_message[k].
        # Each power is applied via repeated sparse-dense spmm
        index = torch.stack([row, col], dim=0)
        out = torch.zeros_like(x_message[0])
        for k in range(self.n_filters):
            v = x_message[k]
            for _ in range(k + 1):
                v = spmm(index, alpha, n, n, v)
            out = out + v
        return out


class SANTMXLayer(nn.Module):
    """Single SAN layer with separate down and up convolutions.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    n_filters : int
        Approximation order (kappa).
    """

    def __init__(self, in_channels, out_channels, n_filters=2):
        super().__init__()
        self.conv_down = SANConv(in_channels, out_channels, n_filters)
        self.conv_up = SANConv(in_channels, out_channels, n_filters)
        self.conv_harmonic = Conv(in_channels, out_channels)

    def forward(self, x, laplacian_down, laplacian_up, P):
        z_down = self.conv_down(x, laplacian_down)
        z_up = self.conv_up(x, laplacian_up)
        z_har = self.conv_harmonic(x, P)
        return F.relu(z_down + z_up + z_har)


class SANTMX(nn.Module):
    """Simplicial Attention Network (TopoModelX-style).

    Parameters
    ----------
    in_channels : int
    hidden_channels : int
    out_channels : int
    n_layers : int
        Must be > 2.
    kappa : int
        Filter order.
    epsilon : float
        Harmonic projector parameter.
    k_proj : int
        Exponent for the harmonic projector: P = (I - ε·L)^k_proj.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers,
        kappa=2,
        epsilon=0.1,
        k_proj=5,
        # Unused but kept for config compatibility with original SAN
        alpha_leaky_relu=0.01,
        dropout=0.0,
        attention=True,
    ):
        super().__init__()
        assert n_layers > 2, "Needs at least 3 layers in `n_layers`"

        self.epsilon = epsilon
        self.k_proj = k_proj

        self.layers = nn.ModuleList()
        self.layers.append(SANTMXLayer(in_channels, hidden_channels, kappa))
        for _ in range(n_layers - 2):
            self.layers.append(SANTMXLayer(hidden_channels, hidden_channels, kappa))
        self.layers.append(SANTMXLayer(hidden_channels, hidden_channels, kappa))

        self.linear = nn.Linear(hidden_channels, hidden_channels)
        self.readout = SumReadout(hidden_channels, out_channels)

    def _reconstruct(self, batch, name):
        """Rebuild a SparseTensor from batched _idx/_val keys."""
        idx = batch[f"{name}_idx"]
        val = batch[f"{name}_val"]
        if 'incidence' in name:
            r_to = int(name.split('_')[-1])
            r_from = r_to - 1
        else:
            r_from = r_to = int(name.split('_')[-1])
        n_rows = batch[f"x_{r_from}"].size(0)
        n_cols = batch[f"x_{r_to}"].size(0)
        return SparseTensor(
            row=idx[0], col=idx[1], value=val,
            sparse_sizes=(n_rows, n_cols)
        )

    def _sparse_tensor_power(self, sparse_mat: SparseTensor, n):
        if n == 0:
            return SparseTensor.eye(*sparse_mat.shape, device=sparse_mat.device())
            
        if n == 1:
            return sparse_mat 
            
        if n % 2 == 0:
            half_power = self._sparse_tensor_power(sparse_mat, n // 2)
            return half_power @ half_power
        else:
            return sparse_mat @ self._sparse_tensor_power(sparse_mat, n - 1)

    def _compute_projection_matrix(self, L_g):
        """ Naive projector without using harmonics.

            Parameters
            ----------
                hodge_laplacian: Tensor
                    Tensor containing the hodge_laplacian L_1.
                batch_1: Tensor
                    Tensor containing the simplicial complext that each 1-simplex is a part of.
            Returns
            -------
                Tensor
                    Projection matrix.
        """        
        n = L_g.size(0)
        eye = SparseTensor.eye(n, device=L_g.device())

        row_t, col_t, value_t = L_g.coo()
        row_e, col_e, value_e = eye.coo()

        proj = SparseTensor(
            row=torch.cat([row_e, row_t]),
            col=torch.cat([col_e, col_t]),
            value = torch.cat([value_e, -self.epsilon*value_t]),
            sparse_sizes=L_g.sparse_sizes()
        ).coalesce()
        return self._sparse_tensor_power(proj, self.k_proj)

    def _harmonic_projector(self, hodge_laplacian, batch_1):
        """Per-graph harmonic projectors assembled as block-diagonal SparseTensor."""
        E = batch_1.size(0)
        device = batch_1.device
        num_graphs = batch_1.max().item() + 1

        _, counts = batch_1.unique_consecutive(return_counts=True)
        offsets = torch.zeros(num_graphs + 1, dtype=torch.long, device=device)
        offsets[1:] = counts.cumsum(0)

        all_rows, all_cols, all_vals = [], [], []

        for g in range(num_graphs):
            start, end = offsets[g].item(), offsets[g + 1].item()
            if start == end:
                continue

            L_block = hodge_laplacian[start:end, start:end].to_dense()
            P_g = self._compute_projection_matrix(L_block)

            r, c = torch.where(P_g.abs() > 1e-8)
            all_rows.append(r + start)
            all_cols.append(c + start)
            all_vals.append(P_g[r, c])

        if not all_rows:
            return SparseTensor(
                row=torch.zeros(0, dtype=torch.long, device=device),
                col=torch.zeros(0, dtype=torch.long, device=device),
                value=torch.zeros(0, device=device),
                sparse_sizes=(E, E),
            )
        rows = torch.cat(all_rows)
        cols = torch.cat(all_cols)
        vals = torch.cat(all_vals)
        return SparseTensor(row=rows, col=cols, value=vals, sparse_sizes=(E, E))

    def forward(self, batch):
        x = batch['x_1']
        batch_1 = batch['batch_1']

        up_laplacian = self._reconstruct(batch, "up_laplacian_1")
        down_laplacian = self._reconstruct(batch, "down_laplacian_1")

        hodge_laplacian = up_laplacian + down_laplacian
        P = self._compute_projection_matrix(hodge_laplacian)

        for layer in self.layers:
            x = layer(x, down_laplacian, up_laplacian, P)

        x = self.linear(x)
        out = self.readout(x, batch_1)
        return out
