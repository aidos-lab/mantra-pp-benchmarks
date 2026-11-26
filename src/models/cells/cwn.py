"""CWN class."""

import torch

import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_sparse import SparseTensor
from torch_scatter import scatter_add
from torch_geometric.nn import MLP

from src.models.conv_down import Conv, PropagateSignalDown
from src.models.utils import (
    head,
    per_rank,
    resolve_max_rank,
    target_rows,
    triangle_rows,
)


class CWNLayer(nn.Module):
    r"""Layer of a CW Network (CWN).

    Implementation of the CWN layer proposed in [1]_.

    This module is composed of the following layers:
    1. A convolutional layer that sends messages from r-cells to r-cells.
    2. A convolutional layer that sends messages from (r-1)-cells to r-cells.
    3. A layer that creates representations in r-cells based on the received messages.
    4. A layer that updates representations in r-cells.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on (r-1)-cells (nodes in case r = 1).

    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).

    in_channels_2 : int
        Dimension of input features on (r+1)-cells (faces in case r = 1).

    out_channels : int
        Dimension of output features on r-cells.

    conv_1_to_1 : torch.nn.Module, optional
        A module that convolves the representations of upper-adjacent neighbors of r-cells
        and their corresponding co-boundary (r+1) cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultFirstConv for more detail).

    conv_0_to_1 : torch.nn.Module, optional
        A module that convolves the representations of (r-1)-cells on the boundary of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultSecondConv for more detail).

    aggregate_fn : torch.nn.Module, optional
        A module that aggregates the representations of r-cells obtained by convolutional layers.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultAggregate for more detail).

    update_fn : torch.nn.Module, optional
        A module that updates the aggregated representations of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultUpdate for more detail).

    **kwargs : optional
        Additional arguments for the modules of the CWN layer.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels,
        conv_1_to_1=None,
        conv_0_to_1=None,
        aggregate_fn=None,
        update_fn=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv_1_to_1 = (
            conv_1_to_1
            if conv_1_to_1 is not None
            else _CWNDefaultFirstConv(in_channels[1], in_channels[2], out_channels)
        )
        self.conv_0_to_1 = (
            conv_0_to_1
            if conv_0_to_1 is not None
            else _CWNDefaultSecondConv(in_channels[0], in_channels[1], out_channels)
        )
        self.aggregate_fn = (
            aggregate_fn if aggregate_fn is not None else _CWNDefaultAggregate()
        )
        self.update_fn = (
            update_fn
            if update_fn is not None
            else _CWNDefaultUpdate(out_channels, out_channels)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_1,
        incidence_2,
        incidence_1_t,
    ):
        r"""Forward pass.

        The forward pass was initially proposed in [1]_.
        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of two convolutional steps
        that are followed by an aggregation step and a final update step.

        1. The first convolution between r-cells through (r+1)-cells exploits
        upper-adjacency neighborhood matrix and co-boundary matrix:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
                = M_{\mathcal{L}\uparrow}(h_x^{t,(r)}, h_y^{t,(r)}, h_z^{t,(r')})\\
            &🟧 \quad m_x^{(r \rightarrow r' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{L}(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
            \end{align*}

        2. The second convolution from (r-1)-cells to r-cells exploits
        boundary neighborhood matrix:

        .. math::
            \begin{align*}
            &🟥 m_{y \rightarrow x}^{(r'' \rightarrow r)} = M_{\mathcal{B}}(h_x^{t,(r)}, h_y^{t,(r'')})\\
            &🟧 \quad m_x^{(r'' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}
            \end{align*}

        3. Then, an aggregation step is applied:

        .. math::
            \begin{align*}
            &🟧 \quad m_x^{(r)} = AGG_{\mathcal{N}\_k \in \mathcal{N}} (m_x^k)
            \end{align*}

        4. Finally, an update step is applied:

        .. math::
            \begin{align*}
            &🟦 \quad h_x^{t+1,(r)} = U\left(h_x^{t,(r)}, m_x^{(r)}\right)
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        x_2 : torch.Tensor, shape = (n_{r+1}_cells, in_channels_{r+1})
            Input features on the (r+1)-cells.
        adjacency_0 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        incidence_2 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).
        incidence_1_t : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations of the r-cells.

        References
        ----------
        .. [2] Papillon, Sanborn, Hajij, Miolane.
            Equations of topological neural networks (2023).
            https://github.com/awesome-tnns/awesome-tnns/
        .. [3] Papillon, Sanborn, Hajij, Miolane.
            Architectures of topological deep learning: a survey on topological neural networks (2023).
            https://arxiv.org/abs/2304.10031.
        """
        x_convolved_1_to_1 = self.conv_1_to_1(x_1, x_2, adjacency_1, incidence_2)
        x_convolved_0_to_1 = self.conv_0_to_1(x_0, x_1, incidence_1_t)

        x_aggregated = self.aggregate_fn(x_convolved_1_to_1, x_convolved_0_to_1)
        return self.update_fn(x_aggregated, x_1)


class _CWNDefaultFirstConv(nn.Module):
    r"""
    Default implementation of the first convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the first convolutional step in CWN layer.

    Parameters
    ----------
    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).
    in_channels_2 : int
        Dimension of input features on (r+1)-cells (faces in case r = 1).
    out_channels : int
        Dimension of output features on r-cells.
    """

    def __init__(self, in_channels_1, in_channels_2, out_channels) -> None:
        super().__init__()
        self.conv_1_to_1 = Conv(
            in_channels_1, out_channels
        )
        self.conv_2_to_1 = Conv(
            in_channels_2, out_channels
        )

    def forward(self, x_1, x_2, adjacency_1, incidence_2):
        r"""Forward pass.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_2 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        adjacency_0 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        incidence_2 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        x_up = F.elu(self.conv_1_to_1(x_1, adjacency_1))
        x_coboundary = F.elu(self.conv_2_to_1(x_2, incidence_2))
        return x_up + x_coboundary


class _CWNDefaultSecondConv(nn.Module):
    r"""
    Default implementation of the second convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the second convolutional step in CWN layer.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on (r-1)-cells (nodes in case r = 1).
    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).
    out_channels : int
        Dimension of output features on r-cells.
    """

    def __init__(self, in_channels_0, in_channels_1, out_channels) -> None:
        super().__init__()
        self.conv_0_to_1 = Conv(
            in_channels_0, out_channels
        )

    def forward(self, x_0, x_1, incidence_1_t):
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        incidence_1_t : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        return F.elu(self.conv_0_to_1(x_0, incidence_1_t))


class _CWNDefaultAggregate(nn.Module):
    r"""
    Default implementation of an aggregation step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the aggregation step in CWN layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the first convolutional step.
        y : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the second convolutional step.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Aggregated representations on the r-cells.
        """
        return x + y


class _CWNDefaultUpdate(nn.Module):
    r"""Default implementation of an update step in CWNLayer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.transform = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_prev=None):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            New representations on the r-cells obtained after the aggregation step.
        x_prev : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Original representations on the r-cells passed into the CWN layer.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        return F.elu(self.transform(x))



class CWN(torch.nn.Module):
    """Implementation of a specific version of CW network [1]_.

    A layer updates the cells that have cells below and above them: the
    edges with max_rank=2, the edges and the triangles with max_rank=3.
    A cell of rank r receives from the (r-1)-cells on its boundary, from
    the r-cells that share an (r+1)-cell with it and from those
    (r+1)-cells. The vertices and the cells of rank max_rank keep their
    projected input features; the readout carries every rank down to the
    vertices.

    Parameters
    ----------
    in_channels : list of int
        Dimension of input features, one per rank of the data.
    hidden_channels : int or list of int
        Dimension of hidden features: one per rank 0..max_rank, all
        equal, or a single int.
    n_layers : int
        Number of CWN layers.
    max_rank : int, optional
        Top rank the model reads (default: 2, up to the triangles). 3
        adds the tetrahedra of a 3-dimensional complex; None takes every
        rank of the data.
    **kwargs : optional
        Additional arguments CWNLayer.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: List[int],
        n_layers,
        task_level: str = 'graph',
        head_layers: int = 0,
        max_rank: Optional[int] = 2,
        **kwargs,
    ):
        super().__init__()
        self.task_level = task_level
        self.max_rank = resolve_max_rank(max_rank, in_channels)
        hidden_channels = per_rank(hidden_channels, self.max_rank + 1)

        for rank, channels in enumerate(hidden_channels):
            setattr(self, f"proj_{rank}", torch.nn.Linear(in_channels[rank], channels))

        # One stack of layers per updated rank: `layers` for the edges,
        # `layers_{r}` for the rank r >= 2 of a model with max_rank > r.
        self.layers = torch.nn.ModuleList(
            CWNLayer(
                in_channels=hidden_channels[:3],
                out_channels=hidden_channels[0]
            )
            for _ in range(n_layers)
        )
        for rank in range(2, self.max_rank):
            setattr(
                self,
                f"layers_{rank}",
                torch.nn.ModuleList(
                    CWNLayer(
                        in_channels=hidden_channels[rank - 1:rank + 2],
                        out_channels=hidden_channels[rank]
                    )
                    for _ in range(n_layers)
                ),
            )
        self.readout = PropagateSignalDown(max_dimension=self.max_rank + 1, hidden_channel=hidden_channels[0])
        # The triangles read the sum of their edges' final states, which
        # every max_rank updates.
        if task_level == 'triangle':
            self.triangle_readout = head(hidden_channels[0], kwargs['out_channels'], head_layers)
        # Candidate triangles need not be cells, so they read the vertices.
        if task_level == 'candidate':
            self.candidate_readout = head(hidden_channels[0], kwargs['out_channels'], head_layers)
        self.graph_readout = MLP(in_channels=hidden_channels[0], hidden_channels=hidden_channels[0], out_channels=kwargs['out_channels'], num_layers=2)

    def _reconstruct(self, batch, name):
        idx = batch[f"{name}_idx"]
        val = batch[f"{name}_val"].detach()
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

    def forward(
        self,
        batch
    ):
        """Forward computation through projection, convolutions, linear layers and average pooling.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch of `collate_simplicial` with, for every rank r up to
            max_rank, the features `x_{r}`, the boundary matrix
            `incidence_{r}` (r >= 1) and, for the updated ranks
            1..max_rank-1, the upper adjacency `adjacency_{r}`.

        Returns
        -------
        torch.Tensor
            One row per complex, or per vertex, boundary triangle or
            candidate triangle for the task levels below the complex.
        """
        ranks = range(self.max_rank + 1)
        x = [F.elu(getattr(self, f"proj_{rank}")(batch[f"x_{rank}"])) for rank in ranks]

        incidence = {
            rank: self._reconstruct(batch, f"incidence_{rank}") for rank in ranks[1:]
        }
        stacks = {1: self.layers}
        for rank in range(2, self.max_rank):
            stacks[rank] = getattr(self, f"layers_{rank}")
        adjacency = {
            rank: self._reconstruct(batch, f"adjacency_{rank}") for rank in stacks
        }

        for depth in range(len(self.layers)):
            # Every rank reads the states of the previous layer.
            updated = {
                rank: stack[depth](
                    x[rank - 1],
                    x[rank],
                    x[rank + 1],
                    adjacency[rank],
                    incidence[rank + 1],
                    incidence[rank].t(),
                )
                for rank, stack in stacks.items()
            }
            for rank, state in updated.items():
                x[rank] = state

        if self.task_level == 'triangle':
            return self.triangle_readout(triangle_rows(incidence[2].t() @ x[1], batch))

        model_out = {f"x_{rank}": x[rank] for rank in ranks}
        incidence_map = {f"incidence_{rank}": matrix for rank, matrix in incidence.items()}
        out = self.readout(model_out, incidence_map)
        if self.task_level == 'candidate':
            return self.candidate_readout(target_rows(out['x_0'], batch, 'candidate'))
        out = self.graph_readout(out['x_0'])

        # Node-level tasks keep the per-0-cell predictions of the readout
        if self.task_level == 'node':
            return out

        out = scatter_add(out, batch['batch_0'], dim=-2)

        return out
