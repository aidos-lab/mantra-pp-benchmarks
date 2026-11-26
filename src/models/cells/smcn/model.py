"""Scalable Multi-Cellular Network (SMCN).

Eitan et al., "Topological Blindspots: Understanding and Extending
Topological Deep Learning Through the Lens of Expressivity", ICLR 2025
(arXiv:2408.05486). Higher-order message passing (HOMP) keeps one state
per cell and cannot tell apart, for instance, complexes that differ in
orientability. SMCN adds states on the pairs (vertex, h-cell): one copy
of the 1-skeleton per h-cell, in which that cell is marked.

The layers are the ones of the reference implementation
(https://github.com/yoavgelberg/SMCN, see `layers/`), and the diagram
follows its topological-property models
(`experiments/second_betti_number_prediction/model.py`, with the leading
CIN layers of `experiments/torus_dataset/model.py`):

1. a linear embedding of the features of ranks 0 to 2,
2. `n_cin_layers` HOMP (CIN) layers on the ranks 0 to 2,
3. the pair states: the vertex state, concatenated with an embedding of
   the marking of the pair,
4. `n_layers` subcomplex layers, each an incidence convolution, which
   sends the marked pairs of a vertex to all its pairs, and a
   convolution along the edges within every copy,
5. pooling of the pairs back to the vertices and to the h-cells,
6. sum pooling of every output rank over the complex, a linear layer and
   ReLU per rank, and the readout on their sum.

Deviations from the reference, for this repository: the input features
are embedded linearly (the reference embeds atom and bond types, or a
constant), the hidden width is `hidden_channels` (the reference scales
it by 1.5 and adds two subcomplex layers to the configured number), the
final linear layer is the shared `head`, and the node, triangle and
candidate heads read the vertex and triangle states before the pooling
of step 6. Updates whose state nothing reads are left out (`prune`),
which changes the parameter count but not the output.

The pairs make the model quadratic: a complex with n_0 vertices, n_1
edges and n_h h-cells has n_0 * n_h pair states and (h + 1) * n_h^2 +
2 * n_1 * n_h pair edges per subcomplex layer. With `conv_type='gin'`
the edges cost indices only (see `complex.py`); `'custom_gin'` builds
one message row per edge within the copies.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from src.models.cells.smcn.complex import build_complex
from src.models.cells.smcn.layers.homp import (
    AdjacencyConv,
    EmbeddingAdapter,
    IncidenceConv,
    PointwiseConv,
)
from src.models.cells.smcn.layers.merge_node import MergeNode
from src.models.cells.smcn.layers.subcomplex import (
    SubComplexBroadcastLow,
    SubComplexDistanceMarkingEmbed,
    SubComplexIncidenceConv,
    SubComplexLowConv,
    SubComplexPool,
)
from src.models.utils import head, target_rows, triangle_rows


def cin_layer(dim):
    """HOMP layer on the ranks 0 to 2 (`get_cin_layer` of the reference).

    Like every builder here, returns per state the module that writes it
    and the states that module reads, in the order the entries run.
    """
    return {
        "x_0": (
            MergeNode(
                [
                    AdjacencyConv(
                        input_rank=0,
                        bridge_rank=1,
                        embedding_dim=dim,
                        conv_type="custom_gin",
                        train_eps=False,
                    ),
                    PointwiseConv(rank=0, embedding_dim=dim),
                ],
                embedding_dim=dim,
            ),
            {"x_0", "x_1"},
        ),
        "x_1": (
            MergeNode(
                [
                    IncidenceConv(
                        input_rank=0,
                        output_rank=1,
                        embedding_dim=dim,
                        train_eps=False,
                    ),
                    AdjacencyConv(
                        input_rank=1,
                        bridge_rank=2,
                        embedding_dim=dim,
                        conv_type="custom_gin",
                        train_eps=False,
                    ),
                ],
                embedding_dim=dim,
            ),
            {"x_0", "x_1", "x_2"},
        ),
        "x_2": (
            MergeNode(
                [
                    IncidenceConv(
                        input_rank=1,
                        output_rank=2,
                        embedding_dim=dim,
                        train_eps=False,
                    ),
                    PointwiseConv(rank=2, embedding_dim=dim),
                ],
                embedding_dim=dim,
            ),
            {"x_1", "x_2"},
        ),
    }


def subcomplex_embed_layer(dim, high_rank, max_distance):
    """Initial pair states (`get_subcomplex_embed_layer` of the reference)."""
    return {
        f"x_0_{high_rank}": (
            MergeNode(
                [
                    SubComplexBroadcastLow(low_rank=0, high_rank=high_rank),
                    SubComplexDistanceMarkingEmbed(
                        low_rank=0,
                        high_rank=high_rank,
                        embed_dim=dim,
                        max_dist=max_distance,
                    ),
                ],
                embedding_dim=dim,
            ),
            {"x_0"},
        )
    }


def subcomplex_layer(dim, high_rank, conv_type):
    """Update of the pair states (`get_subcomplex_layer` of the reference)."""
    pairs = f"x_0_{high_rank}"
    return {
        pairs: (
            MergeNode(
                [
                    SubComplexIncidenceConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=dim,
                        output_channels=dim,
                        hidden_channels=dim,
                        train_eps=False,
                    ),
                    SubComplexLowConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=dim,
                        output_channels=dim,
                        hidden_channels=dim,
                        edge_dim=dim,
                        train_eps=False,
                        conv_type=conv_type,
                    ),
                ],
                embedding_dim=dim,
            ),
            # Every convolution but the plain GIN reads the edge states.
            {pairs} if conv_type == "gin" else {pairs, "x_1"},
        )
    }


def subcomplex_pooling_layer(high_rank, aggregation):
    """Pair states to cell states (`get_subcomplex_pooling_layer` of the reference)."""
    pairs = f"x_0_{high_rank}"
    return {
        "x_0": (
            SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=True,
                aggregation=aggregation,
            ),
            {pairs},
        ),
        f"x_{high_rank}": (
            SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=False,
                aggregation=aggregation,
            ),
            {pairs},
        ),
    }


def prune(layers, read):
    """Drop the entries of `layers` whose state nothing reads.

    The reference diagrams update every rank in every CIN layer, also
    where a later layer overwrites the state or no layer reads it again;
    those entries cost parameters and time without changing the output.

    Parameters
    ----------
    layers : list of dict
        Per layer, state name to `(module, states it reads)`.
    read : set of str
        States the readout reads.

    Returns
    -------
    list of torch.nn.ModuleDict
        The layers with the entries that reach the readout.
    """
    read = set(read)
    kept = []
    for entries in reversed(layers):
        live = []
        for name, (module, reads) in reversed(list(entries.items())):
            if name in read:
                read = (read - {name}) | reads
                live.append((name, module))
        kept.append(nn.ModuleDict(dict(reversed(live))))
    return kept[::-1]


class SMCN(nn.Module):
    """SMCN on the simplicial batches of `collate_simplicial`.

    Needs `x_0`, `x_1`, `x_2` and the incidence matrices up to rank
    `max(2, high_rank)` (the `incidence_sc` representation); their signs
    are ignored.

    Parameters
    ----------
    in_channels : list of int
        Feature widths of the ranks; the first three are read.
    hidden_channels : int
        Width of every cell and pair state.
    out_channels : int
        Prediction width, set from the task by `main.py`.
    n_layers : int
        Subcomplex layers. 0 drops the pairs: the HOMP baseline of the
        reference, `n_cin_layers` CIN layers.
    n_cin_layers : int
        CIN layers before the pairs. The edge features enter the vertex
        states with the first one, the triangle features with the second.
    high_rank : int
        Rank h of the pairs (vertex, h-cell), up to the dimension of the
        complexes.
    marking : str
        `'binary'`: whether the vertex lies in the h-cell (the reference
        setting for the Betti-number and cross-diameter tasks).
        `'distance'`: the hop distance from the vertex to the h-cell.
    max_distance : int
        Distances above it share one embedding.
    conv_type : str
        Convolution within the copies: `'gin'`, or `'custom_gin'`, which
        also reads the state of the edge of every message.
    subcomplex_pool : str
        `'mean'` or `'sum'` of the pairs of a cell in step 5. The
        reference passes its merge mode there, which selects the mean.
    pool_method : str
        Reduction over the cells of a complex in the graph readout.
    output_ranks : list of int, optional
        Ranks the graph readout pools; `[0, high_rank]` by default and
        `[0, 1, 2]` without subcomplex layers. With subcomplex layers
        they must include 0 or `high_rank`, the states the pairs write.
    residuals : bool
        Residual connections around the CIN and subcomplex layers.
    dropout : float
        Dropout after every layer but the input embedding, and before
        the final readout.
    task_level : str
        `'graph'`, or `'node'` / `'candidate'` (read the vertex states)
        or `'triangle'` (reads the triangle states, so the pairs need
        `high_rank=2`).
    head_layers : int
        Hidden layers of the readout (`head`).
    """

    def __init__(
        self,
        in_channels: List[int],
        hidden_channels: int,
        out_channels: int,
        n_layers: int = 4,
        n_cin_layers: int = 1,
        high_rank: int = 2,
        marking: str = "binary",
        max_distance: int = 10,
        conv_type: str = "gin",
        subcomplex_pool: str = "mean",
        pool_method: str = "sum",
        output_ranks: Optional[List[int]] = None,
        residuals: bool = False,
        dropout: float = 0.0,
        task_level: str = "graph",
        head_layers: int = 0,
    ):
        super().__init__()
        if n_layers < 0 or n_cin_layers < 0 or n_layers + n_cin_layers == 0:
            raise ValueError("SMCN needs at least one CIN or subcomplex layer")
        if high_rank < 1:
            raise ValueError(f"high_rank must be at least 1, got {high_rank}")
        if marking not in ("binary", "distance"):
            raise ValueError(f"marking must be 'binary' or 'distance', got {marking!r}")
        if subcomplex_pool not in ("mean", "sum"):
            raise ValueError(
                f"subcomplex_pool must be 'mean' or 'sum', got {subcomplex_pool!r}"
            )

        self.task_level = task_level
        self.pool_method = pool_method
        self.marking = marking
        # Rank whose pairs the forward pass builds; None for plain HOMP.
        self.high_rank = high_rank if n_layers > 0 else None
        # Only the plain GIN runs on an operator instead of an edge list.
        self.edge_states = conv_type != "gin"

        dim = hidden_channels
        has_pairs = n_layers > 0

        # States the readout reads.
        if task_level == "graph":
            available = {0, 1, 2, high_rank} if has_pairs else {0, 1, 2}
            if output_ranks is None:
                output_ranks = [0, high_rank] if has_pairs else [0, 1, 2]
            self.output_ranks = [int(rank) for rank in output_ranks]
            if not set(self.output_ranks) <= available:
                raise ValueError(
                    f"output_ranks {self.output_ranks} must be among {sorted(available)}"
                )
            read = {f"x_{rank}" for rank in self.output_ranks}
        elif task_level == "triangle":
            read = {"x_2"}
        else:
            read = {"x_0"}

        layers = [
            {
                f"x_{rank}": (
                    EmbeddingAdapter(in_channels[rank], dim, rank),
                    {f"x_{rank}"},
                )
                for rank in range(3)
            }
        ]
        residual = [False]
        for _ in range(n_cin_layers):
            layers.append(cin_layer(dim))
            residual.append(residuals)
        if has_pairs:
            layers.append(subcomplex_embed_layer(dim, high_rank, max_distance))
            residual.append(False)
            for _ in range(n_layers):
                layers.append(subcomplex_layer(dim, high_rank, conv_type))
                residual.append(residuals)
            layers.append(subcomplex_pooling_layer(high_rank, subcomplex_pool))
            residual.append(False)

        layers = prune(layers, read)
        if not all(len(layer) for layer in layers):
            raise ValueError(
                "the readout reads no state the subcomplex layers write: put 0 "
                "or high_rank in output_ranks, and use high_rank=2 for the "
                "triangle task"
            )

        self.layers = nn.ModuleList(layers)
        self.residual = residual
        self.dropouts = nn.ModuleList(
            nn.Dropout(0.0 if index == 0 else dropout)
            for index in range(len(layers))
        )

        if task_level == "graph":
            self.linear_heads = nn.ModuleList(
                nn.Sequential(nn.Linear(dim, 2 * dim), nn.ReLU())
                for _ in self.output_ranks
            )
            self.final_dropout = nn.Dropout(dropout)
            self.final_linear = head(2 * dim, out_channels, head_layers)
        else:
            self.final_linear = head(dim, out_channels, head_layers)

    def forward(self, batch) -> torch.Tensor:
        data = build_complex(batch, self.high_rank, self.marking, self.edge_states)

        # As in the reference, the entries of a layer run in order and
        # each sees the states the previous ones wrote.
        for layer, dropout, residual in zip(self.layers, self.dropouts, self.residual):
            for name, module in layer.items():
                h = dropout(module(data))
                setattr(data, name, getattr(data, name) + h if residual else h)

        if self.task_level == "triangle":
            return self.final_linear(triangle_rows(data.x_2, batch))
        if self.task_level != "graph":
            return self.final_linear(target_rows(data.x_0, batch, self.task_level))

        num_graphs = data.num_cells_0.numel()
        out = 0
        for rank, rank_head in zip(self.output_ranks, self.linear_heads):
            pooled = scatter(
                getattr(data, f"x_{rank}"),
                getattr(data, f"batch_{rank}"),
                dim=0,
                dim_size=num_graphs,
                reduce=self.pool_method,
            )
            out = out + rank_head(pooled)
        return self.final_linear(self.final_dropout(out))
