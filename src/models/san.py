# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:00:00 2022

@author: Lorenzo Giusti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import MLP
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
        """
        Initialize the SALayer.
        Args:
            F_in (int): Number of features of the input signal.
            F_out (int): Number of features of the output component.
            L (tuple): Tuple containing matrices defining the simplicial complex structure.
            kappa (int): Maximum order of neighborhood to consider in attention computation.
            p_dropout (float): Dropout rate.
            alpha_leaky_relu (float): Negative slope coefficient for LeakyReLU.
        """
        super(SALayer, self).__init__()

        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        # Initialize the transformation matrices for the three types of components: 
        # Irr, Sol, and Har.
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))
        # Initialize the attention matrices.
        self.att_irr = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))
        self.att_sol = nn.Parameter(torch.empty(size=(2*F_out*self.K, 1)))

        self.dropout = p_dropout
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)
        self.act_fn = act_fn

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)
        nn.init.xavier_uniform_(self.att_irr.data, gain=gain)
        nn.init.xavier_uniform_(self.att_sol.data, gain=gain)
        
    def forward(self, x, down_laplacian, up_laplacian, P):
        """
        Perform the forward computation of the layer.
        Args:
            x (torch.Tensor): Input signal with shape (num_nodes, F_in).
        Returns:
            torch.Tensor: Output signal with shape (num_nodes, F_out).
        """
        # Ld, Lu , P = self.L

        E = x.shape[0]
        # Compute the Irrotational, Solenoidal and Harmonic component transformations.
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_irr = torch.cat([torch.spmm(x, self.Wirr[k]) for k in range(self.K)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_sol = torch.cat([torch.spmm(x, self.Wsol[k]) for k in range(self.K)], dim=1)
        # (ExE) x (ExF_in) x (F_inxF_out) -> (ExF_out)
        x_har = torch.spmm(torch.spmm(P, x),  self.Whar)

        # Compute attention coefficients. # Broadcast add: (Ex1) + (1xE) -> (ExE)
        
        d_idx = down_laplacian.coalesce().indices()   # (2, nnz_d)
        u_idx = up_laplacian.coalesce().indices()     # (2, nnz_u)

        row_d, col_d = d_idx[0], d_idx[1]
        row_u, col_u = u_idx[0], u_idx[1]
        
        # --- Split attention vectors ---
        a_l_irr = self.att_irr[:self.F_out * self.K]   # (F_out*K, 1)
        a_r_irr = self.att_irr[self.F_out * self.K:]
        a_l_sol = self.att_sol[:self.F_out * self.K]
        a_r_sol = self.att_sol[self.F_out * self.K:]

        # --- Compute attention scores only on non-zero entries ---
        # Instead of outer product (E×E), we index only the nnz pairs
        # x_irr[row_d]: (nnz_d, F_out*K), result: (nnz_d, 1) -> (nnz_d,)
        e_irr = self.leakyrelu(
            (x_irr[row_d] @ a_l_irr) + (x_irr[col_d] @ a_r_irr)
        ).squeeze(-1)  # (nnz_d,)

        e_sol = self.leakyrelu(
            (x_sol[row_u] @ a_l_sol) + (x_sol[col_u] @ a_r_sol)
        ).squeeze(-1)  # (nnz_u,)

        # --- Sparse softmax: normalises per-row over neighbours only ---
        # pyg's softmax takes (values, row_indices, num_nodes)
        alpha_irr = sparse_softmax(e_irr, row_d, num_nodes=E)  # (nnz_d,)
        alpha_sol = sparse_softmax(e_sol, row_u, num_nodes=E)  # (nnz_u,)

        alpha_irr = F.dropout(alpha_irr, self.dropout, training=self.training)
        alpha_sol = F.dropout(alpha_sol, self.dropout, training=self.training)

        # --- Rebuild sparse attention matrices for aggregation ---
        alpha_irr_sp = torch.sparse_coo_tensor(d_idx, alpha_irr, (E, E))
        alpha_sol_sp = torch.sparse_coo_tensor(u_idx, alpha_sol, (E, E))

        # Exclude self-loops by setting diagonal entries to a large negative value.
        # zero_vec = -9e15*torch.ones_like(E_irr)
        # E_irr = torch.where(down_laplacian != 0, E_irr, zero_vec)
        # E_sol = torch.where(up_laplacian != 0, E_sol, zero_vec)

        z_i = torch.spmm(alpha_irr_sp, torch.clone(torch.spmm(x,  self.Wirr[0])))
        z_s = torch.spmm(alpha_sol_sp, torch.clone(torch.spmm(x,  self.Wsol[0])))

        alpha_irr_k = alpha_irr_sp
        alpha_sol_k = alpha_sol_sp
        for k in range(1, self.K):
            alpha_irr_k = torch.spmm(alpha_irr_k, down_laplacian)
            alpha_sol_k = torch.spmm(alpha_sol_k, up_laplacian)
            z_i += torch.spmm(torch.spmm(alpha_irr_k, x), self.Wirr[k])
            z_s += torch.spmm(torch.spmm(alpha_sol_k,  x), self.Wsol[k])

        out = (z_i + z_s + x_har)

        if self.act_fn is None:
            return out

        return self.act_fn(out)


class SCLayer(nn.Module):

    def __init__(self, F_in, F_out, kappa, p_dropout, alpha_leaky_relu, act_fn):
        """
        F_in: Numer of features of the input signal
        F_out: Numer of features of the output *component*
        """
        super(SCLayer, self).__init__()
        self.K = kappa
        self.F_in = F_in
        self.F_out = F_out
        self.Wirr = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Wsol = nn.Parameter(torch.empty(size=(self.K, F_in, F_out)))
        self.Whar = nn.Parameter(torch.empty(size=(F_in, F_out)))


        self.dropout = p_dropout  # 0.0#0.6
        self.leakyrelu = nn.LeakyReLU(alpha_leaky_relu)

        self.act_fn = act_fn
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.Wirr.data, gain=gain)
        nn.init.xavier_uniform_(self.Wsol.data, gain=gain)
        nn.init.xavier_uniform_(self.Whar.data, gain=gain)

    def forward(self, x, down_laplacian, up_laplacian, P):
        # This is what we need
        # Ld, Lu , P = self.L


        x_har = torch.spmm(P, x) @ self.Whar
        
        alpha_irr_k = down_laplacian # torch.clone(down_laplacian).requires_grad_(False)
        alpha_sol_k = up_laplacian # torch.clone(up_laplacian).requires_grad_(False)


        tmp_Wirr = torch.spmm(x, self.Wirr[0])
        tmp_Wsol = torch.spmm(x, self.Wsol[0])

        z_i = torch.spmm(alpha_irr_k, tmp_Wirr)
        z_s = torch.spmm(alpha_sol_k,  tmp_Wsol)

        for k in range(1, self.K):
            alpha_irr_k = torch.spmm(alpha_irr_k, down_laplacian) # alpha_irr
            alpha_sol_k = torch.spmm(alpha_sol_k, up_laplacian) #alpha_sol
            z_i += torch.spmm(alpha_irr_k, torch.spmm(x, self.Wirr[k]))
            # (ExE) x (ExF_out) -> (ExF_out)
            z_s += torch.spmm(alpha_sol_k, torch.spmm(x, self.Wsol[k]))

        out = F.dropout((z_i + z_s + x_har), self.dropout)  # (ExE) -> (ExE)
        if self.act_fn is None:
            return out
        return self.act_fn(out)


class SAN(nn.Module):
    """

    @author: Lorenzo Giusti
    """
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, alpha_leaky_relu=0.01, kappa=2, dropout=0.0, epsilon=0.1, k_proj=10, attention=False):
        super().__init__()

        assert n_layers > 2, "Needs at least 3 layers in `n_layers`"
        self.n_layers = n_layers
        self.input_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.attention = attention
        self.epsilon = epsilon
        self.k_proj = k_proj
        self.act = torch.nn.functional.relu #NOTE: Can be changed


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

    def _reconstruct_coo(self, batch, name):
        """Rebuild a torch.sparse_coo_tensor from batched _idx/_val keys."""
        idx = batch[f"{name}_idx"]
        val = batch[f"{name}_val"]
        if 'incidence' in name:
            r_to = int(name.split('_')[-1])
            r_from = r_to - 1
        else:
            r_from = r_to = int(name.split('_')[-1])
        n_rows = batch[f"x_{r_from}"].size(0)
        n_cols = batch[f"x_{r_to}"].size(0)
        return torch.sparse_coo_tensor(idx, val, (n_rows, n_cols)).coalesce()

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

        L_dense = hodge_laplacian.to_dense() if hodge_laplacian.is_sparse else hodge_laplacian
        all_rows, all_cols, all_vals = [], [], []

        for g in range(num_graphs):
            start, end = offsets[g].item(), offsets[g + 1].item()
            if start == end:
                continue
            P_g = self._single_graph_projector(L_dense[start:end, start:end])
            r, c = torch.where(P_g.abs() > 1e-8)
            all_rows.append(r + start)
            all_cols.append(c + start)
            all_vals.append(P_g[r, c])

        if not all_rows:
            return torch.sparse_coo_tensor(
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, device=device), (E, E),
            )
        P_idx = torch.stack([torch.cat(all_rows), torch.cat(all_cols)])
        return torch.sparse_coo_tensor(P_idx, torch.cat(all_vals), (E, E)).coalesce()

    def forward(self, batch):
        x = batch['x_1']
        batch_1 = batch['batch_1']

        up_laplacian = self._reconstruct_coo(batch, "up_laplacian_1")
        down_laplacian = self._reconstruct_coo(batch, "down_laplacian_1")

        hodge_laplacian = up_laplacian + down_laplacian

        P = self._harmonic_projector(hodge_laplacian, batch_1)

        for layer in self.layers:
            x = layer(x, down_laplacian, up_laplacian, P)

        x = self.linear(x)

        out = self.readout(x, batch_1)

        return out