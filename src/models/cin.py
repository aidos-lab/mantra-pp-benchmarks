from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, Sequential
from torch_geometric.nn import JumpingKnowledge, global_add_pool
from torch_geometric.nn.inits import reset

from src.models.cells.mp.cell_mp import (
    CochainMessagePassing,
    CochainMessagePassingParams,
)
from src.models.cells.mp.complex import ComplexBatch


class CINCochainConv(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""

    def __init__(
        self,
        up_msg_size: int,
        down_msg_size: int,
        msg_up_nn: Callable,
        msg_down_nn: Callable,
        update_nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
    ):
        super(CINCochainConv, self).__init__(
            up_msg_size, down_msg_size, use_boundary_msg=False
        )
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.update_nn = update_nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        out_up, out_down, _ = self.propagate(
            cochain.up_index,
            cochain.down_index,
            None,
            x=cochain.x,
            up_attr=cochain.kwargs["up_attr"],
            down_attr=cochain.kwargs["down_attr"],
        )

        out_up += (1 + self.eps) * cochain.x
        out_down += (1 + self.eps) * cochain.x
        return self.update_nn(out_up + out_down)

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_down_nn)
        reset(self.update_nn)
        self.eps.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        if up_attr is not None:
            x = torch.cat([up_x_j, up_attr], dim=-1)
            return self.msg_up_nn(x)
        else:
            return self.msg_up_nn(up_x_j)

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        x = torch.cat([down_x_j, down_attr], dim=-1)
        return self.msg_down_nn(x)


class CINConv(torch.nn.Module):
    def __init__(
        self,
        up_msg_size: int,
        down_msg_size: int,
        msg_up_nn: Callable,
        msg_down_nn: Callable,
        update_nn: Callable,
        eps: float = 0.0,
        train_eps: bool = False,
        max_dim: int = 2,
    ):
        super(CINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim + 1):
            mp = CINCochainConv(
                up_msg_size,
                down_msg_size,
                msg_up_nn,
                msg_down_nn,
                update_nn,
                eps,
                train_eps,
            )
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim + 1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out


class CIN0(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(
        self,
        num_input_features,
        num_classes,
        num_layers,
        hidden,
        dropout_rate: float = 0.5,
        max_dim: int = 2,
        jump_mode=None,
        nonlinearity="relu",
        readout="sum",
    ):
        super(CIN0, self).__init__()

        self.max_dim = max_dim
        self.dropout_rate = dropout_rate
        self.jump_mode = jump_mode
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.pooling_fn = global_add_pool
        conv_nonlinearity = torch.nn.ReLU
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            conv_update = Sequential(
                Linear(layer_dim, hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                conv_nonlinearity(),
                BN(hidden),
            )
            conv_up = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim),
            )
            conv_down = Sequential(
                Linear(layer_dim * 2, layer_dim),
                conv_nonlinearity(),
                BN(layer_dim),
            )
            self.convs.append(
                CINConv(
                    layer_dim,
                    layer_dim,
                    conv_up,
                    conv_down,
                    conv_update,
                    train_eps=False,
                    max_dim=self.max_dim,
                )
            )
        self.jump = (
            JumpingKnowledge(jump_mode) if jump_mode is not None else None
        )
        if jump_mode == "cat":
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def pool_complex(self, xs, data):
        # All complexes have nodes so we can extract the batch size from cochains[0]
        batch_size = data.cochains[0].batch.max() + 1
        # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
        pooled_xs = torch.zeros(
            self.max_dim + 1,
            batch_size,
            xs[0].size(-1),
            device=batch_size.device,
        )
        for i in range(len(xs)):
            # It's very important that size is supplied.
            pooled_xs[i, :, :] = self.pooling_fn(
                xs[i], data.cochains[i].batch, size=batch_size
            )
        return pooled_xs

    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch):
        model_nonlinearity = F.relu
        xs, jump_xs = None, None
        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim)
            xs = conv(*params)
            data.set_xs(xs)

            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)
        pooled_xs = self.pool_complex(xs, data)
        x = pooled_xs.sum(dim=0)

        x = model_nonlinearity(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
