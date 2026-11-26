from typing import Optional, List

from pydantic import BaseModel
from torch import nn
import torch

from torch_sparse import SparseTensor

from torch_geometric.utils import coalesce, scatter, to_dense_adj
from src.models.utils import (
    SumReadout,
    head,
    per_rank,
    resolve_max_rank,
    target_rows,
    triangle_rows,
)


class SCCNNLayer(nn.Module):
    r"""Layer of a Simplicial Complex Convolutional Neural Network.

    The simplices of every rank k = 0..K, with K = len(in_channels) - 1,
    receive three signals: their own, the one of rank k - 1 through the
    boundary matrix B_k and the one of rank k + 1 through B_{k+1}. Each
    is filtered with the powers 1..conv_order of the down and of the up
    Laplacian of rank k, and a weight per filter mixes the channels.
    The vertices have no simplices below them and the top rank none
    above, so they receive two signals.

    Parameters
    ----------
    in_channels : tuple of int
        Dimensions of input features, one per rank (nodes, edges, faces, ...).
    out_channels : tuple of int
        Dimensions of output features, one per rank.
    conv_order : int
        Convolution order of the simplicial filters.
    sc_order : int
        Dimension of the complex. The top rank K has an up Laplacian
        only in a complex of higher dimension (sc_order > K); without
        one its up filters are zero.
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size (default: False).
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

        self.max_rank = len(in_channels) - 1
        self.conv_order = conv_order
        self.sc_order = sc_order

        self.aggr_norm = aggr_norm
        self.update = nn.ReLU()
        self.initialization = initialization

        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert self.conv_order > 0

        # Go thourgh each rank where the max rank has no upper 
        # counterpar and the 0-th rank has no lower counerpart
        for rank, (channels_in, channels_out) in enumerate(
            zip(in_channels, out_channels)
        ):
            signals = (rank > 0) + 1 + (rank < self.max_rank)
            laplacians = 1 if rank == 0 else 2
            setattr(
                self,
                f"weight_{rank}",
                nn.Parameter(
                    torch.Tensor(
                        channels_in,
                        channels_out,
                        signals * (1 + laplacians * conv_order),
                    )
                ),
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
            init = torch.nn.init.xavier_uniform_
        elif self.initialization == "xavier_normal":
            init = torch.nn.init.xavier_normal_
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )
        for rank in range(self.max_rank + 1):
            init(getattr(self, f"weight_{rank}"), gain=gain)

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
        conv_operator : torch.sparse or None
            Convolution operator; None is the zero operator.
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
        if conv_operator is None:
            return x.new_zeros(num_simplices, num_channels, conv_order)

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

    def forward(self, x_all, down, up, incidence):
        r"""Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Input features, one tensor per rank 0..K.
        down : list
            `down[k]` is the down Laplacian of rank k; `down[0]` is None.
        up : list
            `up[k]` is the up Laplacian of rank k (L_0 for the vertices);
            None for the top rank of a complex of dimension K.
        incidence : list
            `incidence[k]` is the boundary matrix B_k, of shape
            (n_{k-1}, n_k); `incidence[0]` is None.

        Returns
        -------
        tuple of tensors
            Output features, one tensor per rank.
        """
        out = []
        for rank, x in enumerate(x_all):
            signals = []
            if rank > 0:
                signals.append(incidence[rank].t() @ x_all[rank - 1])
            signals.append(x)
            if rank < self.max_rank:
                signals.append(incidence[rank + 1] @ x_all[rank + 1])

            filtered = []
            for signal in signals:
                filtered.append(signal.unsqueeze(2))
                if rank > 0:
                    filtered.append(
                        self.chebyshev_conv(down[rank], self.conv_order, signal)
                    )
                filtered.append(
                    self.chebyshev_conv(up[rank], self.conv_order, signal)
                )

            y = torch.einsum(
                "nik,iok->no",
                torch.cat(filtered, dim=2),
                getattr(self, f"weight_{rank}"),
            )
            out.append(self.update(y))

        return tuple(out)


class SCCNNBackbone(nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all : tuple of int
        Dimension of input features, one per rank (nodes, edges, faces, ...).
    hidden_channels_all : tuple of int
        Dimension of features of hidden layers, one per rank.
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
        for rank, (channels_in, channels_hidden) in enumerate(
            zip(in_channels_all, hidden_channels_all)
        ):
            setattr(
                self, f"in_linear_{rank}", nn.Linear(channels_in, channels_hidden)
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

    def forward(self, x_all, down, up, incidence):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple(tensors)
            Tuple of feature tensors, one per rank.
        down, up, incidence : list
            Laplacians and boundary matrices by rank, see `SCCNNLayer.forward`.

        Returns
        -------
        tuple(tensors)
            Tuple of final hidden state tensors, one per rank.
        """
        x_all = tuple(
            getattr(self, f"in_linear_{rank}")(x) for rank, x in enumerate(x_all)
        )

        # Forward through SCCNN
        for layer in self.layers:
            x_all = layer(x_all, down, up, incidence)

        return x_all



class SCCNN(nn.Module):
    """SCCNN implementation for complex classification.

    Original paper: Convolutional Learning on Simplicial Complexes (https://arxiv.org/pdf/2301.11163)

     Note: In this task, we can consider the output on any order of simplices for the
     classification task, which of course can be amended by a readout layer.

     Parameters
     ----------
     in_channels : tuple of int
         Dimension of input features, one per rank of the data.
     hidden_channels : int or tuple of int
         Dimension of features of hidden layers: one per rank
         0..max_rank, or the same for all of them.
     out_channels : int
         Dimension of output features.
     conv_order : int
         Order of convolutions, we consider the same order for all convolutions.
     sc_order : int
         Order of simplicial complex; at least max_rank.
     aggr_norm : bool, optional
         Whether to normalize the aggregation (default: False).
     n_layers : int, optional
         Number of layers (default: 2).
     max_rank : int, optional
         Top rank with features and weights (default: 2, up to the
         triangles). 3 adds the tetrahedra of a 3-dimensional complex;
         None takes every rank of the data.
    """

    def __init__(self, in_channels: List[int], hidden_channels: List[int], out_channels: int = 3, conv_order: int = 1, sc_order: int = 3, aggr_norm: bool = True, n_layers: int = 2, task_level: str = 'graph', head_layers: int = 0, max_rank: Optional[int] = 2):
        super().__init__()
        self.max_rank = resolve_max_rank(max_rank, in_channels)
        self.sc_order = max(sc_order, self.max_rank)
        self.task_level = task_level
        hidden_channels = per_rank(hidden_channels, self.max_rank + 1)

        self.sccnn_backbone = SCCNNBackbone(
            list(in_channels)[: self.max_rank + 1],
            hidden_channels,
            conv_order,
            self.sc_order,
            aggr_norm,
            n_layers,
        )

        if task_level == 'node':
            self.node_readout = head(hidden_channels[0], out_channels, head_layers)
        elif task_level == 'triangle':
            self.triangle_readout = head(hidden_channels[2], out_channels, head_layers)
        elif task_level == 'candidate':
            self.candidate_readout = head(hidden_channels[0], out_channels, head_layers)
        else:
            for rank, channels in enumerate(hidden_channels):
                setattr(
                    self, f"readout_{rank}", SumReadout(channels, out_channels)
                )

    def _reconstruct(self, batch, name):
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

    def forward(self, batch):
        ranks = range(self.max_rank + 1)
        x_all = tuple(batch[f"x_{rank}"] for rank in ranks)
        x_bel = [batch[f"batch_{rank}"] for rank in ranks]

        # Make sure there are the same number of SCs for each batch
        assert all(bel.max() == x_bel[0].max() for bel in x_bel), ", ".join(
            str(bel.max().item()) for bel in x_bel
        )

        down = [None] + [
            self._reconstruct(batch, f"down_laplacian_{rank}") for rank in ranks[1:]
        ]
        up = [self._reconstruct(batch, "hodge_laplacian_0")] + [
            self._reconstruct(batch, f"up_laplacian_{rank}")
            if rank < self.sc_order
            else None
            for rank in ranks[1:]
        ]
        incidence = [None] + [
            self._reconstruct(batch, f"incidence_{rank}") for rank in ranks[1:]
        ]

        x_all = self.sccnn_backbone(x_all, down, up, incidence)

        if self.task_level == 'node':
            return self.node_readout(x_all[0])
        if self.task_level == 'triangle':
            return self.triangle_readout(triangle_rows(x_all[2], batch))
        if self.task_level == 'candidate':
            return self.candidate_readout(target_rows(x_all[0], batch, 'candidate'))

        #TODO: Figure out what to do about this readout
        out = self.readout_0(x_all[0], x_bel[0])
        for rank in ranks[1:]:
            out = out + getattr(self, f"readout_{rank}")(x_all[rank], x_bel[rank])

        return out
