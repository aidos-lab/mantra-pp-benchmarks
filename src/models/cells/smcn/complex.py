"""Connectivity of a batch of simplicial complexes, as the SMCN layers read it.

Upstream SMCN (https://github.com/yoavgelberg/SMCN) precomputes this
connectivity per sample (`data/complex_data.py`, `data/utils.py`) and
stores it in the dataset. Here it is derived on the fly, for the whole
batch at once, from the incidence matrices that `collate_simplicial`
batches (`incidence_{r}_idx` / `incidence_{r}_val`), so SMCN trains on the
same cache as the other simplicial models. The attribute names and the
index layout are upstream's.

For a pair of ranks (0, h) the multi-cellular features live on the pairs
(vertex, h-cell) of the same complex. Their rows are ordered by vertex
and then by h-cell, as upstream: the pairs of one vertex are contiguous.

The edges between the pairs are what makes SMCN quadratic in the size of
a complex. Upstream stores them as edge lists; here the operators of the
GIN convolutions are `SparseTensor`s, for which PyG sums the neighbours
with a sparse product instead of one message row per edge. The sums are
the same.
"""

from types import SimpleNamespace

import torch
from torch_sparse import SparseTensor

# Distance upstream imputes to an unreachable pair; the distance-marking
# embedding maps every value above 1000 to its last row.
UNREACHABLE = 1001


def _ragged_arange(count):
    """`arange(count[0]), arange(count[1]), ...` concatenated.

    Returns the position in `count` every entry belongs to and the entry.
    """
    owner = torch.repeat_interleave(
        torch.arange(count.numel(), device=count.device), count
    )
    start = torch.cumsum(count, 0) - count
    return owner, torch.arange(owner.numel(), device=count.device) - start[owner]


def shared_cell_pairs(index, key, num_keys):
    """Ordered pairs `(index[a], index[b])`, `a != b`, with `key[a] == key[b]`.

    With the incidence pairs (r-cell `index`, r'-cell `key`) these are the
    r-cells that share an r'-cell, once per shared cell and direction,
    like upstream's `get_adjacency_from_incidence`. Also returns the
    shared cell (the bridge) of every pair.
    """
    order = torch.argsort(key)
    count = torch.bincount(key, minlength=num_keys)
    start = torch.cumsum(count, 0) - count
    first, offset = _ragged_arange(count[key])
    second = order[start[key[first]] + offset]
    keep = first != second
    first, second = first[keep], second[keep]
    return torch.stack([index[first], index[second]]), key[first]


def compose_incidence(lower, upper, num_mid, num_high):
    """Incidence of rank a in rank c from a-in-b (`lower`) and b-in-c (`upper`)."""
    order = torch.argsort(lower[1])
    count = torch.bincount(lower[1], minlength=num_mid)
    start = torch.cumsum(count, 0) - count
    pair, offset = _ragged_arange(count[upper[0]])
    low = lower[0][order[start[upper[0][pair]] + offset]]
    # An a-cell lies in a c-cell through several b-cells; keep it once.
    code = torch.unique(low * num_high + upper[1][pair])
    return torch.stack(
        [torch.div(code, num_high, rounding_mode="floor"), code % num_high]
    )


def _boundary_pairs(batch, rank):
    """Pairs ((rank-1)-cell, rank-cell) of the batched boundary matrix."""
    key = f"incidence_{rank}"
    if f"{key}_idx" not in batch:
        raise KeyError(
            f"SMCN needs `{key}` in the batch: add the incidence_sc "
            "representation and keep the key in select_attributes"
        )
    return batch[f"{key}_idx"][:, batch[f"{key}_val"] != 0]


def _distance_to_marked(marked, source, target, num_nodes):
    """Hop distance of every node to the nearest marked node."""
    distance = torch.full((num_nodes,), float(UNREACHABLE), device=source.device)
    frontier = torch.zeros(num_nodes, dtype=torch.bool, device=source.device)
    frontier[marked] = True
    distance[marked] = 0
    visited = frontier.clone()
    level = 0
    while frontier.any():
        level += 1
        reached = torch.zeros_like(frontier)
        reached[target[frontier[source]]] = True
        frontier = reached & ~visited
        visited |= frontier
        distance[frontier] = level
    return distance


def _operator(row_count, col, size):
    """Square operator with `row_count[i]` entries in row i, columns `col`."""
    rowptr = torch.zeros(size + 1, dtype=torch.long, device=col.device)
    torch.cumsum(row_count, 0, out=rowptr[1:])
    operator = SparseTensor(
        rowptr=rowptr, col=col, sparse_sizes=(size, size), is_sorted=True, trust_data=True
    )
    if col.is_cuda and torch.is_grad_enabled():
        # The backward pass of the sparse product on GPU sorts the entries
        # by column, and every convolution works on its own copy of the
        # operator: sort once, so that the copies inherit the result.
        operator.storage.row()
        operator.storage.csr2csc()
        operator.storage.colptr()
    return operator


def subcomplex_connectivity(
    incidence, adjacency, bridge, batch_low, num_high, marking, edge_states=False
):
    """Connectivity of the pairs (low cell, high cell) of every complex.

    Batched equivalent of upstream's `Utils.construct_subcomplex_features`
    for the layers ported here.

    Parameters
    ----------
    incidence : torch.Tensor, shape = (2, n)
        Pairs (low cell, high cell containing it).
    adjacency, bridge : torch.Tensor
        Adjacent low cells and the cell each pair shares.
    batch_low : torch.Tensor
        Complex of every low cell.
    num_high : torch.Tensor
        High cells of every complex.
    marking : str
        `'binary'` marks the incident pairs with 1; `'distance'` gives
        every pair the hop distance, along `adjacency`, from the low cell
        to the high cell (0 when incident).
    edge_states : bool
        Return the edges within the copies as an edge list with the
        bridge of every edge, for the convolutions that read the states
        of the bridges, instead of as an operator.

    Returns
    -------
    dict
        Upstream's attribute names without the `_{low}_{high}` suffix.
        The operators hold the targets in their rows.
    """
    num_low = batch_low.numel()
    high_offset = torch.cumsum(num_high, 0) - num_high
    repeats = num_high[batch_low]  # pairs of every low cell
    start = torch.cumsum(repeats, 0) - repeats  # the first of them
    low_cell, local_high = _ragged_arange(repeats)
    high_cell = high_offset[batch_low[low_cell]] + local_high
    num_pairs = low_cell.numel()

    low, high = incidence
    incident = torch.sort(start[low] + high - high_offset[batch_low[low]]).values

    # (v, t) -> (v, t') for every v in t and every t' of the complex:
    # each pair of v receives from the incident pairs of v.
    degree = torch.bincount(low_cell[incident], minlength=num_low)
    first = torch.cumsum(degree, 0) - degree
    row_count = degree[low_cell]
    row, offset = _ragged_arange(row_count)
    incidence_operator = _operator(
        row_count, incident[first[low_cell[row]] + offset], num_pairs
    )

    # (u, t) -> (v, t) for adjacent u, v and every t of the complex.
    order = torch.argsort(adjacency[1] * num_low + adjacency[0])
    source = adjacency[0][order]
    degree = torch.bincount(adjacency[1], minlength=num_low)
    first = torch.cumsum(degree, 0) - degree
    row_count = degree[low_cell]
    row, offset = _ragged_arange(row_count)
    neighbour = first[low_cell[row]] + offset
    col = start[source[neighbour]] + local_high[row]

    if marking == "binary":
        node_marking = torch.zeros(num_pairs, device=low.device)
        node_marking[incident] = 1.0
    elif marking == "distance":
        node_marking = _distance_to_marked(incident, col, row, num_pairs)
    else:
        raise ValueError(f"marking must be 'binary' or 'distance', got {marking!r}")

    pairs = {
        "x_node_marking_subcomplex": node_marking.view(-1, 1),
        "cells_high_repeats": repeats,
        "low_cell_indices_subcomplex": low_cell,
        "high_cell_indices_subcomplex": high_cell,
        "edge_index_incidence_subcomplex": incidence_operator,
    }
    if edge_states:
        pairs["edge_index_low_adjacency_subcomplex"] = torch.stack([col, row])
        pairs["bridge_index_low_adjacency_subcomplex"] = bridge[order][neighbour]
    else:
        pairs["edge_index_low_adjacency_subcomplex"] = _operator(
            row_count, col, num_pairs
        )
    return pairs


def build_complex(batch, high_rank=None, marking="binary", edge_states=False):
    """Inputs of the SMCN layers for a batch of `collate_simplicial`.

    Parameters
    ----------
    batch : torch_geometric.data.Data
        Batch with `x_r`, `batch_r` and `incidence_r_idx` / `_val`.
    high_rank : int, optional
        Rank h of the pairs (vertex, h-cell); None builds no pairs.
    marking : str
        Initial marking of the pairs, see `subcomplex_connectivity`.
    edge_states : bool
        Whether the convolution within the copies reads the edge states,
        see `subcomplex_connectivity`.

    Returns
    -------
    types.SimpleNamespace
        The features and counts of ranks 0 to 2, the incidences and
        adjacencies between consecutive ranks and, with `high_rank`, the
        connectivity of the pairs. The layers write their outputs to it.
    """
    num_graphs = int(batch["batch_0"].max()) + 1
    data = SimpleNamespace()

    ranks = {0, 1, 2} if high_rank is None else {0, 1, 2, high_rank}
    for rank in sorted(ranks):
        if rank <= 2:
            setattr(data, f"x_{rank}", batch[f"x_{rank}"])
        setattr(data, f"batch_{rank}", batch[f"batch_{rank}"])
        setattr(
            data,
            f"num_cells_{rank}",
            torch.bincount(batch[f"batch_{rank}"], minlength=num_graphs),
        )

    for low in (0, 1):
        high = low + 1
        incidence = _boundary_pairs(batch, high)
        adjacency, bridge = shared_cell_pairs(
            incidence[0], incidence[1], batch[f"batch_{high}"].numel()
        )
        setattr(data, f"edge_index_incidence_{low}_{high}", incidence)
        setattr(data, f"edge_index_adjacency_{low}_{high}", adjacency)
        setattr(data, f"bridge_index_adjacency_{low}_{high}", bridge)

    if high_rank is None:
        return data

    incidence = data.edge_index_incidence_0_1
    for rank in range(2, high_rank + 1):
        incidence = compose_incidence(
            incidence,
            data.edge_index_incidence_1_2 if rank == 2 else _boundary_pairs(batch, rank),
            batch[f"batch_{rank - 1}"].numel(),
            batch[f"batch_{rank}"].numel(),
        )
    setattr(data, f"edge_index_incidence_0_{high_rank}", incidence)

    pairs = subcomplex_connectivity(
        incidence,
        data.edge_index_adjacency_0_1,
        data.bridge_index_adjacency_0_1,
        data.batch_0,
        getattr(data, f"num_cells_{high_rank}"),
        marking,
        edge_states,
    )
    for name, value in pairs.items():
        setattr(data, f"{name}_0_{high_rank}", value)
    return data
