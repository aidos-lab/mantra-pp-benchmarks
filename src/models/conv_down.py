"""Readout layer that propagates the signal from cells of a certain order to the cells of the lower order."""

import torch
from torch import nn
import torch_geometric
from torch_sparse import SparseTensor
import torch_sparse

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(
            self.weight, gain=gain
        )

    def forward(self, x_source, neighborhood: SparseTensor):
        row, col, val = neighborhood.coo()
        m, n = neighborhood.sparse_sizes()
        index = torch.stack([row, col], dim=0)

        x_message = x_source @ self.weight

        x_message_on_target = torch_sparse.spmm(index, val, m, n, x_message)

        return x_message_on_target


class PropagateSignalDown(nn.Module):
    r"""Propagate signal down readout layer.

    This readout layer propagates the signal from cells of a certain order to the cells of the lower order.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments. It should contain the following keys:
        - num_cell_dimensions (int): Highest order of cells considered by the model.
        - self.hidden_dim (int): Dimension of the cells representations.
        - readout_name (str): Readout name.
    """

    def __init__(self, max_dimension, hidden_channel):
        super().__init__()


        self.dimensions = range(max_dimension - 1, 0, -1)
        self.hidden_channel = hidden_channel

        for i in self.dimensions:
            setattr(
                self,
                f"agg_conv_{i}",
                Conv(
                    self.hidden_channel, self.hidden_channel
                ),
            )

            setattr(self, f"ln_{i}", torch.nn.LayerNorm(self.hidden_channel))

            setattr(
                self,
                f"projector_{i}",
                torch.nn.Linear(2 * self.hidden_channel, self.hidden_channel),
            )

    def forward(self, model_out: dict, batch):
        r"""Forward pass of the propagate signal down readout layer.

        The layer takes the embeddings of the cells of a certain order and applies a convolutional layer to them. Layer normalization is then applied to the features. The output is concatenated with the initial embeddings of the cells and the result is projected with the use of a linear layer to the dimensions of the cells of lower rank. The process is repeated until the nodes embeddings, which are the cells of rank 0, are reached.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        for i in self.dimensions:
            x_i = getattr(self, f"agg_conv_{i}")(
                model_out[f"x_{i}"], batch[f"incidence_{i}"]
            )
            x_i = getattr(self, f"ln_{i}")(x_i)
            model_out[f"x_{i - 1}"] = getattr(self, f"projector_{i}")(
                torch.cat([x_i, model_out[f"x_{i - 1}"]], dim=1)
            )

        return model_out