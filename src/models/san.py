"""Simplicial Attention Network (SAN) Layer."""
from typing import Literal

import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn import GCNConv


class SANConv(torch.nn.Module):
    r"""Simplicial Attention Network (SAN) Convolution from [1]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_filters : int
        Number of simplicial filters.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Weight initialization method.

    References
    ----------
    .. [1] Giusti, Battiloro, Di Lorenzo, Sardellitti and Barbarossa.
        Simplicial attention neural networks (2022).
        https://arxiv.org/abs/2203.07485.
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_filters,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.initialization = initialization

        self.weight = Parameter(
            torch.Tensor(self.n_filters, self.in_channels, self.out_channels)
        )

        self.att_weight = Parameter(
            torch.Tensor(
                2 * self.out_channels * self.n_filters,
            )
        )

        self.reset_parameters()

    def attention(self, x_source, x_target=None):
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def forward(self, x_source, neighborhood):
        r"""Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells, which are the same source cells.

        In practice, this will update the features on the target cells [2]_.

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{u,(1 \rightarrow 2 \rightarrow 1)}  = ((L_{\uparrow,1} \odot \operatorname{att}(h_z^{t,(2)}, h_y^{t,(1)}))^u)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,u}\\
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{d,(1 \rightarrow 0 \rightarrow 1)}  = ((L_{\downarrow,1} \odot \operatorname{att}(h_z^{t,(0)}, h_y^{t,(1)}))^d)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,d}\\
            &🟥 \quad m^{p,(1 \rightarrow 1)}\_{y \rightarrow x} = ((1-wH_1)^p)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,p}\\
            &🟧 \quad m_{x}^{u,(1 \rightarrow 2 \rightarrow 1)}  = \sum_{y \in \mathcal{L}\_\uparrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{u,(1 \rightarrow 2 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{d,(1 \rightarrow 0 \rightarrow 1)}  = \sum_{y \in \mathcal{L}\downarrow(X)} m_{y \rightarrow \{z\} \rightarrow x}^{d,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟧 \quad m^{p,(1 \rightarrow 1)}\_{x}  = m^{p,(1 \rightarrow 1)}\_{x \rightarrow x}\\
            &🟩 \quad m_x^{(1)}  = \sum_{p=1}^P m_x^{p,(1 \rightarrow 1)} + \sum_{u=1}^{U} m_{x}^{u,(1 \rightarrow 2 \rightarrow 1)} + \sum_{d=1}^{D} m_{x}^{d,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟦 \quad h_x^{t+1, (1)} = \sigma(m_x^{(1)})
            \end{align*}

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        x_message = torch.matmul(x_source, self.weight)
        # Reshape required to re-use the attention function of parent Conv class
        # -> [num_nodes, out_channels * n_filters]
        x_message_reshaped = x_message.permute(1, 0, 2).reshape(
            -1, self.out_channels * self.n_filters
        )

        # SAN always requires attention
        # In SAN, neighborhood is defined by lower/upper laplacians; we only use them as masks
        # to keep only the relevant attention coeffs
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        attention_values = self.attention(x_message_reshaped)
        att_laplacian = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention_values,
            size=neighborhood.shape,
        )

        # Attention coeffs are normalized using softmax
        att_laplacian = torch.sparse.softmax(att_laplacian, dim=1).to_dense()
        # We need to compute the power of the attention laplacian according up to order p
        att_laplacian_power = [att_laplacian]
        for _ in range(1, self.n_filters):
            att_laplacian_power.append(
                torch.matmul(att_laplacian_power[-1], att_laplacian)
            )
        att_laplacian_power = torch.stack(att_laplacian_power)

        # When computing the final message on targets, we multiply the message by each power
        # of the attention laplacian and sum the results
        return torch.matmul(att_laplacian_power, x_message).sum(dim=0)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=1.414)

class SANLayer(torch.nn.Module):
    r"""Implementation of the Simplicial Attention Network (SAN) Layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_filters : int, default = 2
        Approximation order.

    Notes
    -----
    Architecture proposed for r-simplex (r>0) classification on simplicial complices.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_filters: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        #  Convolutions
        # Down convolutions, one for each filter order p
        self.conv_down = SANConv(in_channels, out_channels, n_filters)

        # Up convolutions, one for each filter order p
        self.conv_up = SANConv(in_channels, out_channels, n_filters)

        # Harmonic convolution
        self.conv_harmonic = GCNConv(in_channels, out_channels, add_self_loop=sFalse, normalize=False)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.conv_down.reset_parameters()
        self.conv_up.reset_parameters()
        self.conv_harmonic.reset_parameters()

    def forward(self, x, laplacian_up, laplacian_down, projection_mat):
        r"""Forward pass of the SAN Layer.

        .. math::
            \mathcal N = \{\mathcal N_1, \mathcal N_2,...,\mathcal N_{2p+1}\}
                = \{A_{\uparrow, r}, A_{\downarrow, r}, A_{\uparrow, r}^2, A_{\downarrow, r}^2,...,A_{\uparrow, r}^p, A_{\downarrow, r}^p, Q_r\},

        .. math::
            \begin{align*}
            &🟥\quad m_{(y \rightarrow x),k}^{(r)}
                = \alpha_k(h_x^t,h_y^t) = a_k(h_x^{t}, h_y^{t}) \cdot \psi_k^t(h_x^{t})\quad \forall \mathcal N_k \in \mathcal{N}\\
            &🟧\quad m_{x,k}^{(r)}
                = \bigoplus_{y \in \mathcal{N}_k(x)}  m^{(r)}_{(y \rightarrow x),k}\\
            &🟩\quad m_{x}^{(r)}
                = \bigotimes_{\mathcal{N}_k\in\mathcal N}m_{x,k}^{(r)}\\
            &🟦\quad h_x^{t+1,(r)}
                = \phi^{t}(h_x^t, m_{x}^{(r)})
            \end{align*}

        Parameters
        ----------
        x : torch.Tensor, shape = (..., n_cells, in_channels)
            Input tensor.
        laplacian_up : torch.Tensor
        laplacian_down : torch.Tensor
            The up- and down-laplacians of the simplicial complex.
        projection_mat : torch.Tensor
            The projection matrix used.

        Returns
        -------
        torch.Tensor, shape = (..., n_cells, out_channels)
            Output tensor.
        """
        # Compute the down and up convolutions
        z_down = self.conv_down(x, laplacian_down)
        z_up = self.conv_up(x, laplacian_up)
        # For the harmonic convolution, we use the precomputed projection matrix P as the neighborhood
        # with no attention
        z_har = torch.nn.functional.relu(self.conv_harmonic(projection_mat, x))

        # final output
        return z_down + z_up + z_har

class SAN(torch.nn.Module):
    """Simplicial Attention Network (SAN) implementation for binary edge classification.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    hidden_channels : int
        Dimension of hidden features.
    out_channels : int
        Dimension of output features.
    n_filters : int, default = 2
        Approximation order for simplicial filters.
    order_harmonic : int, default = 5
        Approximation order for harmonic convolution.
    epsilon_harmonic : float, default = 1e-1
        Epsilon value for harmonic convolution.
    n_layers : int, default = 2
        Number of message passing layers.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels=None,
        n_filters=2,
        order_harmonic=5,
        epsilon_harmonic=1e-1,
        n_layers=2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = (
            out_channels if out_channels is not None else hidden_channels
        )
        self.n_filters = n_filters
        self.order_harmonic = order_harmonic
        self.epsilon_harmonic = epsilon_harmonic

        if n_layers == 1:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    n_filters=self.n_filters,
                )
            ]
        else:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.hidden_channels,
                    n_filters=self.n_filters,
                )
            ]
            for _ in range(n_layers - 2):
                self.layers.append(
                    SANLayer(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        n_filters=self.n_filters,
                    )
                )
            self.layers.append(
                SANLayer(
                    in_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    n_filters=self.n_filters,
                )
            )
        self.layers = torch.nn.ModuleList(self.layers)

    def compute_projection_matrix(self, laplacian):
        """Compute the projection matrix.

        The matrix is used to calculate the harmonic component in SAN layers.

        Parameters
        ----------
        laplacian : torch.Tensor, shape = (n_edges, n_edges)
            Hodge laplacian of rank 1.

        Returns
        -------
        torch.Tensor, shape = (n_edges, n_edges)
            Projection matrix.
        """
        eye = torch.eye(laplacian.shape[0]).to(laplacian.device)
        projection_mat = eye - self.epsilon_harmonic * laplacian
        return torch.linalg.matrix_power(projection_mat, self.order_harmonic)

    def forward(self, batch):
        """Forward computation.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_nodes, channels_in)
            Node features.
        laplacian_up : torch.Tensor, shape = (n_edges, n_edges)
            Upper laplacian of rank 1.
        laplacian_down : torch.Tensor, shape = (n_edges, n_edges)
            Down laplacian of rank 1.

        Returns
        -------
        torch.Tensor, shape = (n_edges, out_channels)
            Final hidden representations of edges.
        """
        x = batch.x
        laplacian_up = batch['laplacian_up']
        laplacian_down = batch['laplacian_down']

        # Compute the projection matrix for the harmonic component
        laplacian = laplacian_up + laplacian_down
        projection_mat = self.compute_projection_matrix(laplacian)

        # Forward computation
        for layer in self.layers:
            x = layer(x, laplacian_up, laplacian_down, projection_mat)
        return x