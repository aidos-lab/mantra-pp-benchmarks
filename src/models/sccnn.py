from typing import List

import torch
from torch import nn
from torch_sparse import SparseTensor

from src.models.utils import SumReadout


class SCCNNLayer(nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features on nodes, edges, and faces.
    out_channels : tuple of int
        Dimensions of output features on nodes, edges, and faces.
    conv_order : int
        Convolution order of the simplicial filters.
    sc_order : int
        SC order.
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size (default: False).
    update_func : str, optional
        Activation function used in aggregation layers (default: None).
    initialization : str, optional
        Initialization method for the weights (default: "xavier_normal").
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order,
        sc_order,
        aggr_norm: bool = False,
        initialization: str = "xavier_normal",
    ) -> None:
        super().__init__()

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.in_channels_0 = in_channels_0
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels_0 = out_channels_0
        self.out_channels_1 = out_channels_1
        self.out_channels_2 = out_channels_2

        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update = nn.ReLU()
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        self.weight_0 = nn.Parameter(
            torch.Tensor(
                self.in_channels_0,
                self.out_channels_0,
                1 + conv_order + 1 + conv_order,
            )
        )

        self.weight_1 = nn.Parameter(
            torch.Tensor(
                self.in_channels_1,
                self.out_channels_1,
                6 * conv_order + 3,
            )
        )

        # determine the third dimensions of the weights
        # because when SC order is larger than 2, there are lower and upper
        # parts for L_2; otherwise, L_2 contains only the lower part

        if sc_order > 2:
            self.weight_2 = nn.Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order
                    + 2,  # in the future for arbitrary sc_order we should have this 6*conv_order + 3,
                )
            )

        elif sc_order == 2:
            self.weight_2 = nn.Parameter(
                torch.Tensor(
                    self.in_channels_2,
                    self.out_channels_2,
                    4 * conv_order + 2,
                )
            )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight_0, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight_2, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Normalized feature tensor.
        """
        neighborhood_size = conv_operator.sum(dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse
            Convolution operator.
        conv_order : int
            Order of the convolution.
        x : torch.Tensor
            Feature tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order)).to(
            x.device
        )

        if self.aggr_norm:
            X[:, :, 0] = conv_operator @ x
            X[:, :, 0] = self.aggr_norm_func(conv_operator, X[:, :, 0])
            for k in range(1, conv_order):
                X[:, :, k] = conv_operator @ X[:, :, k - 1]
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])
        else:
            X[:, :, 0] = conv_operator @ x
            for k in range(1, conv_order):
                X[:, :, k] = conv_operator @ X[:, :, k - 1]
        return X

    def forward(self, x_all, laplacian_all, incidence_all):
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of input feature tensors (node, edge, face).
        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
        incidence_all : tuple of tensors
            Tuple of order 1 and 2 incidence matrices.

        Returns
        -------
        torch.Tensor
            Output tensor for each 0-cell.
        torch.Tensor
            Output tensor for each 1-cell.
        torch.Tensor
            Output tensor for each 2-cell.
        """
        x_0, x_1, x_2 = x_all

        if self.sc_order == 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
            ) = laplacian_all
            device = laplacian_down_2.device()
            laplacian_up_2 = SparseTensor(
                row=torch.empty(0, dtype=torch.long, device=device),
                col=torch.empty(0, dtype=torch.long, device=device),
                value=torch.empty(0, device=device),
                sparse_sizes=laplacian_down_2.sparse_sizes(),
            )
        elif self.sc_order > 2:
            (
                laplacian_0,
                laplacian_down_1,
                laplacian_up_1,
                laplacian_down_2,
                laplacian_up_2,
            ) = laplacian_all

        b1, b2 = incidence_all

        """
        Convolution in the node space
        """
        # -----------Logic to obtain update for 0-cells --------

        x_0_laplacian = self.chebyshev_conv(laplacian_0, self.conv_order, x_0)
        x_0_to_0 = torch.cat([x_0.unsqueeze(2), x_0_laplacian], dim=2)
        # -------------------

        x_1_to_0_upper = b1 @ x_1
        x_1_to_0_laplacian = self.chebyshev_conv(
            laplacian_0, self.conv_order, x_1_to_0_upper
        )
        x_1_to_0 = torch.cat(
            [x_1_to_0_upper.unsqueeze(2), x_1_to_0_laplacian], dim=2
        )
        # -------------------

        x_0_all = torch.cat((x_0_to_0, x_1_to_0), 2)

        # -------------------
        """
        Convolution in the edge space
        """

        # -----------Logic to obtain update for 1-cells --------

        x_1_down = self.chebyshev_conv(laplacian_down_1, self.conv_order, x_1)
        x_1_up = self.chebyshev_conv(laplacian_up_1, self.conv_order, x_1)
        x_1_to_1 = torch.cat((x_1.unsqueeze(2), x_1_down, x_1_up), 2)

        # -------------------

        # Lower projection
        x_0_1_lower = b1.t() @ x_0

        # Calculate lowwer chebyshev_conv
        x_0_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_0_1_lower
        )

        # Calculate upper chebyshev_conv (Note: in case of signed incidence should be always zero)
        x_0_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_0_1_lower
        )

        # Concatenate output of filters
        x_0_to_1 = torch.cat(
            [x_0_1_lower.unsqueeze(2), x_0_1_down, x_0_1_up], dim=2
        )
        # -------------------

        x_2_1_upper = b2 @ x_2

        # Calculate lowwer chebyshev_conv (Note: In case of signed incidence should be always zero)
        x_2_1_down = self.chebyshev_conv(
            laplacian_down_1, self.conv_order, x_2_1_upper
        )

        # Calculate upper chebyshev_conv
        x_2_1_up = self.chebyshev_conv(
            laplacian_up_1, self.conv_order, x_2_1_upper
        )

        x_2_to_1 = torch.cat(
            [x_2_1_upper.unsqueeze(2), x_2_1_down, x_2_1_up], dim=2
        )

        # -------------------
        x_1_all = torch.cat((x_0_to_1, x_1_to_1, x_2_to_1), 2)
        """Convolution in the face (triangle) space, depending on the SC order,
        the exact form maybe a little different."""
        # -------------------Logic to obtain update for 2-cells --------

        x_2_down = self.chebyshev_conv(laplacian_down_2, self.conv_order, x_2)
        x_2_up = self.chebyshev_conv(laplacian_up_2, self.conv_order, x_2)

        x_2_to_2 = torch.cat((x_2.unsqueeze(2), x_2_down, x_2_up), 2)

        # -------------------

        x_1_2_lower = b2.t() @ x_1

        x_1_2_down = self.chebyshev_conv(
            laplacian_down_2, self.conv_order, x_1_2_lower
        )
        x_1_2_up = self.chebyshev_conv(
            laplacian_up_2, self.conv_order, x_1_2_lower
        )

        x_1_to_2 = torch.cat(
            [x_1_2_lower.unsqueeze(2), x_1_2_down, x_1_2_up], dim=2
        )

        # -------------------

        x_2_all = torch.cat([x_1_to_2, x_2_to_2], dim=2)

        # -------------------

        # Need to check that this einsums are correct
        y_0 = torch.einsum("nik,iok->no", x_0_all, self.weight_0)
        y_1 = torch.einsum("nik,iok->no", x_1_all, self.weight_1)
        y_2 = torch.einsum("nik,iok->no", x_2_all, self.weight_2)

        # Perform activations

        return self.update(y_0), self.update(y_1), self.update(y_2)


class SCCNNBackbone(nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all : tuple of int
        Dimension of input features on (nodes, edges, faces).
    hidden_channels_all : tuple of int
        Dimension of features of hidden layers on (nodes, edges, faces).
    conv_order : int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order : int
        Order of simplicial complex.
    aggr_norm : bool, optional
        Whether to normalize the aggregation (default: False).
    update_func : str, optional
        Update function for the simplicial complex convolution (default: None).
    n_layers : int, optional
        Number of layers (default: 2).
    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = nn.Linear(
            in_channels_all[0], hidden_channels_all[0]
        )
        self.in_linear_1 = nn.Linear(
            in_channels_all[1], hidden_channels_all[1]
        )
        self.in_linear_2 = nn.Linear(
            in_channels_all[2], hidden_channels_all[2]
        )

        self.layers = nn.ModuleList(
            SCCNNLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple(tensors)
            Tuple of feature tensors (node, edge, face).
        laplacian_all : tuple(tensors)
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
        incidence_all : tuple(tensors)
            Tuple of order 1 and 2 incidence matrices.

        Returns
        -------
        tuple(tensors)
            Tuple of final hidden state tensors (node, edge, face).
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        return x_all


class SCCNN(nn.Module):
    """SCCNN implementation for complex classification.

    Original paper: Convolutional Learning on Simplicial Complexes (https://arxiv.org/pdf/2301.11163)

     Note: In this task, we can consider the output on any order of simplices for the
     classification task, which of course can be amended by a readout layer.

     Parameters
     ----------
     in_channels : tuple of int
         Dimension of input features on (nodes, edges, faces).
     hidden_channels_all : tuple of int
         Dimension of features of hidden layers on (nodes, edges, faces).
     out_channels : int
         Dimension of output features.
     conv_order : int
         Order of convolutions, we consider the same order for all convolutions.
     sc_order : int
         Order of simplicial complex.
     aggr_norm : bool, optional
         Whether to normalize the aggregation (default: False).
     n_layers : int, optional
         Number of layers (default: 2).
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: List[int],
        out_channels: int = 3,
        conv_order: int = 1,
        sc_order: int = 3,
        aggr_norm: bool = True,
        n_layers: int = 2,
    ):
        super().__init__()
        self.sc_order = sc_order

        self.sccnn_backbone = SCCNNBackbone(
            in_channels,
            hidden_channels,
            conv_order,
            sc_order,
            aggr_norm,
            n_layers,
        )

        self.readout_0 = SumReadout(hidden_channels[0], out_channels)
        self.readout_1 = SumReadout(hidden_channels[1], out_channels)
        self.readout_2 = SumReadout(hidden_channels[2], out_channels)

    def _reconstruct(self, batch, name):
        idx = batch[f"{name}_idx"]
        val = batch[f"{name}_val"]
        if "incidence" in name:
            r_to = int(name.split("_")[-1])
            r_from = r_to - 1
        else:
            r_from = r_to = int(name.split("_")[-1])

        n_rows = batch[f"x_{r_from}"].size(0)
        n_cols = batch[f"x_{r_to}"].size(0)
        return SparseTensor(
            row=idx[0], col=idx[1], value=val, sparse_sizes=(n_rows, n_cols)
        )

    def forward(self, batch):
        x = {
            int(x_key.split("_")[-1]): batch[x_key]
            for x_key in batch.keys()
            if isinstance(x_key, str)
            and x_key.split("_")[0] == "x"
            and x_key.split("_")[1].isdigit()
        }
        batches = {
            int(x_key.split("_")[-1]): batch[x_key]
            for x_key in batch.keys()
            if isinstance(x_key, str)
            and x_key.split("_")[0] == "batch"
            and x_key.split("_")[1].isdigit()
        }

        x_0, x_1, x_2 = x[0], x[1], x[2]

        x_bel_0, x_bel_1, x_bel_2 = (
            batches[0],
            batches[1],
            batches[2],
        )

        # Make sure there are the same number of SCs for each batch
        assert (
            x_bel_0.max() == x_bel_1.max() and x_bel_1.max() == x_bel_2.max()
        ), f"{x_bel_0.max().item()}, {x_bel_1.max().item()}, {x_bel_2.max().item()}"

        x_all = (x_0, x_1, x_2)

        if self.sc_order > 2:
            laplacian_names = [
                "hodge_laplacian_0",
                "down_laplacian_1",
                "up_laplacian_1",
                "down_laplacian_2",
                "up_laplacian_2",
            ]
        else:
            laplacian_names = [
                "hodge_laplacian_0",
                "down_laplacian_1",
                "up_laplacian_1",
                "down_laplacian_2",
            ]
        laplacian_all = tuple(
            self._reconstruct(batch, name) for name in laplacian_names
        )
        incidence_all = tuple(
            self._reconstruct(batch, name)
            for name in ["incidence_1", "incidence_2"]
        )

        x_0, x_1, x_2 = self.sccnn_backbone(
            x_all, laplacian_all, incidence_all
        )

        # TODO: Figure out what to do about this readout
        out_0 = self.readout_0(x_0, x_bel_0)
        out_1 = self.readout_1(x_1, x_bel_1)
        out_2 = self.readout_2(x_2, x_bel_2)

        return out_0 + out_1 + out_2
