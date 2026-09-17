# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:00:00 2022

@author: Lorenzo Giusti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, spmm
from torch_geometric.utils import softmax as sparse_softmax
from src.models.utils import SumReadout


class SALayer(nn.Module):
    """
    SALayer is a simplicial attention layer for signals on simplicial complexes.
    It processes the attention mechanism on upper (solenoidal) and lower (irrotational) neighbourhoods,
    as well as the harmonic component of an edge signal, providing a generalization of graph attention
    networks to higher-order domains.
    """
    def __init__(self, F_in, F_out,  kappa, p_dropout, alpha_leaky_relu, act_fn):
        super(SALayer, self).__init__()

        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))
        self.att_irr = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))
        self.att_sol = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))

        self.dropout = p_dropout
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)
        self.act_fn = act_fn

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)
        nn.init.xavier_uniform_(self.att_irr.data, gain=gain)
        nn.init.xavier_uniform_(self.att_sol.data, gain=gain)

    def forward(self, x, down_laplacian, up_laplacian, P):
        E = x.shape[0]

        x_irr = torch.cat([x @ self.Wirr[k] for k in range(self.K)], dim=1)
        x_sol = torch.cat([x @ self.Wsol[k] for k in range(self.K)], dim=1)
        x_har = (P @ x) @ self.Whar

        # Get sparse indices for attention
        d_row, d_col, _ = down_laplacian.coo()
        u_row, u_col, _ = up_laplacian.coo()

        # Split attention vectors
        a_l_irr = self.att_irr[:self.F_out * self.K]
        a_r_irr = self.att_irr[self.F_out * self.K:]
        a_l_sol = self.att_sol[:self.F_out * self.K]
        a_r_sol = self.att_sol[self.F_out * self.K:]

        # Compute attention scores only on non-zero entries
        e_irr = self.leakyrelu(
            (x_irr[d_row] @ a_l_irr) + (x_irr[d_col] @ a_r_irr)
        ).squeeze(-1)

        e_sol = self.leakyrelu(
            (x_sol[u_row] @ a_l_sol) + (x_sol[u_col] @ a_r_sol)
        ).squeeze(-1)

        # Sparse softmax
        alpha_irr = sparse_softmax(e_irr, d_row, num_nodes=E)
        alpha_sol = sparse_softmax(e_sol, u_row, num_nodes=E)

        alpha_irr = F.dropout(alpha_irr, self.dropout, training=self.training)
        alpha_sol = F.dropout(alpha_sol, self.dropout, training=self.training)

        # Sum_{k=0}^{K-1} alpha^{k+1} @ (x @ W[k]) via Horner's scheme with
        # explicit torch_sparse.spmm — grad-safe on the attention values
        # (SparseTensor @ SparseTensor would silently drop them).
        d_index = torch.stack([d_row, d_col], dim=0)
        u_index = torch.stack([u_row, u_col], dim=0)

        acc_irr = x @ self.Wirr[self.K - 1]
        acc_sol = x @ self.Wsol[self.K - 1]
        for k in range(self.K - 2, -1, -1):
            acc_irr = spmm(d_index, alpha_irr, E, E, acc_irr) + x @ self.Wirr[k]
            acc_sol = spmm(u_index, alpha_sol, E, E, acc_sol) + x @ self.Wsol[k]
        z_i = spmm(d_index, alpha_irr, E, E, acc_irr)
        z_s = spmm(u_index, alpha_sol, E, E, acc_sol)

        out = (z_i + z_s + x_har)

        if self.act_fn is None:
            return out

        return self.act_fn(out)


class SCLayer(nn.Module):

    def __init__(self, F_in, F_out, kappa, p_dropout, alpha_leaky_relu, act_fn):
        super(SCLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))

        self.dropout = p_dropout
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)

        self.act_fn = act_fn
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)

    def forward(self, x, down_laplacian, up_laplacian, P):
        x_har = (P @ x) @ self.Whar

        # Sum_{k=0}^{K-1} L^{k+1} @ (x @ W[k]) via Horner's scheme with explicit
        # torch_sparse.spmm. Laplacian values here are fixed, so the gradient
        # issue doesn't bite, but we keep the same pattern as SALayer.
        d_row, d_col, d_val = down_laplacian.coo()
        u_row, u_col, u_val = up_laplacian.coo()
        E = x.shape[0]
        d_index = torch.stack([d_row, d_col], dim=0)
        u_index = torch.stack([u_row, u_col], dim=0)

        acc_irr = x @ self.Wirr[self.K - 1]
        acc_sol = x @ self.Wsol[self.K - 1]
        for k in range(self.K - 2, -1, -1):
            acc_irr = spmm(d_index, d_val, E, E, acc_irr) + x @ self.Wirr[k]
            acc_sol = spmm(u_index, u_val, E, E, acc_sol) + x @ self.Wsol[k]
        z_i = spmm(d_index, d_val, E, E, acc_irr)
        z_s = spmm(u_index, u_val, E, E, acc_sol)

        out = F.dropout((z_i + z_s + x_har), self.dropout)
        if self.act_fn is None:
            return out
        return self.act_fn(out)


class SAN(nn.Module):
    """

    @author: Lorenzo Giusti
    """
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, alpha_leaky_relu=0.01, kappa=2, dropout=0.0, epsilon=0.1, k_proj=10, attention=False, task_level: str = "graph"):
        super().__init__()

        if task_level != "graph":
            raise ValueError("SAN has only a graph-level head (edge readout)")
        assert n_layers > 2, "Needs at least 3 layers in `n_layers`"
        self.n_layers = n_layers
        self.input_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.attention = attention
        self.epsilon = epsilon
        self.k_proj = k_proj
        self.act = torch.nn.functional.relu

        if self.attention:
            self.layers = nn.ModuleList(
                [
                    SALayer(
                        F_in = hidden_channels,
                        F_out = hidden_channels,
                        kappa=kappa,
                        p_dropout=dropout,
                        alpha_leaky_relu=alpha_leaky_relu,
                        act_fn=self.act
                    )
                    for _ in range(n_layers - 2)
                ]
            )
            self.layers.insert(0,
                SALayer(
                    F_in = in_channels,
                    F_out = hidden_channels,
                    kappa=kappa,
                    p_dropout=dropout,
                    alpha_leaky_relu=alpha_leaky_relu,
                    act_fn=self.act
                )
            )
            self.layers.append(
                SALayer(
                    F_in = hidden_channels,
                    F_out = hidden_channels,
                    kappa=kappa,
                    p_dropout=dropout,
                    alpha_leaky_relu=alpha_leaky_relu,
                    act_fn=None
                )
            )
        else:
            self.layers = nn.ModuleList(
                [
                    SCLayer(
                        F_in = hidden_channels,
                        F_out = hidden_channels,
                        kappa=kappa,
                        p_dropout=dropout,
                        alpha_leaky_relu=alpha_leaky_relu,
                        act_fn=self.act
                    )
                    for _ in range(n_layers - 2)
                ]
            )
            self.layers.insert(0,
                SCLayer(
                    F_in = in_channels,
                    F_out = hidden_channels,
                    kappa=kappa,
                    p_dropout=dropout,
                    alpha_leaky_relu=alpha_leaky_relu,
                    act_fn=self.act
                )
            )
            self.layers.append(
                SCLayer(
                    F_in = hidden_channels,
                    F_out = hidden_channels,
                    kappa=kappa,
                    p_dropout=dropout,
                    alpha_leaky_relu=alpha_leaky_relu,
                    act_fn=None
                )
            )
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

    def _single_graph_projector(self, L_g):
        """P_g ≈ (I - ε·L_norm)^(2^k) for one graph's Hodge Laplacian."""
        n = L_g.size(0)
        lam = torch.linalg.eigvalsh(L_g)[-1].item()
        L_norm = L_g * (2.0 / lam) if lam > 0 else L_g
        P = torch.eye(n, device=L_g.device) - self.epsilon * L_norm
        for _ in range(self.k_proj):
            P = P @ P
        return P

    def _harmonic_projector(self, hodge_laplacian, batch_1):
        """Per-graph harmonic projectors assembled as block-diagonal sparse."""
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
            n = end - start

            # Extract per-graph block efficiently via SparseTensor slicing
            L_block = hodge_laplacian[start:end, start:end].to_dense()

            P_g = self._single_graph_projector(L_block)
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

        P = self._harmonic_projector(hodge_laplacian, batch_1)

        for layer in self.layers:
            x = layer(x, down_laplacian, up_laplacian, P)

        x = self.linear(x)

        out = self.readout(x, batch_1)

        return out
