"""Every model must expose the embedding its linear head consumes.

The Pachner objectives act on that embedding, so
``forward(batch, return_embedding=True)`` has to return exactly the
logits the plain call returns, and the logits must be the head applied
to the embedding.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.cwn import CWN
from src.models.GCN import GCN
from src.models.MLP import MLP
from src.models.ResidualGatedGCN import ResidualGatedGCN
from src.models.sccnn import SCCNN

BATCH_SIZE = 2


def graph_batch():
    first = Data(
        x=torch.randn(4, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
    )
    second = Data(
        x=torch.randn(3, 3),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
    )
    return Batch.from_data_list([first, second])


def simplicial_batch(with_laplacians=False):
    """Two small complexes in the collated form the models expect."""
    batch = Data()
    batch.x_0 = torch.randn(7, 3)
    batch.x_1 = torch.randn(7, 3)
    batch.x_2 = torch.randn(3, 3)
    batch.batch_0 = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch.batch_1 = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    batch.batch_2 = torch.tensor([0, 0, 1])

    pair = torch.tensor([[0, 1], [1, 0]])
    values = torch.tensor([1.0, 1.0])
    incidence = torch.tensor([[0, 1], [0, 1]])
    signed = torch.tensor([1.0, -1.0])

    batch.adjacency_1_idx, batch.adjacency_1_val = pair, values
    batch.incidence_1_idx, batch.incidence_1_val = incidence, signed
    batch.incidence_2_idx, batch.incidence_2_val = incidence, signed

    if with_laplacians:
        for name in (
            "hodge_laplacian_0",
            "down_laplacian_1",
            "up_laplacian_1",
            "down_laplacian_2",
        ):
            batch[f"{name}_idx"] = pair
            batch[f"{name}_val"] = values

    return batch


def graph_models():
    return [
        GCN(
            in_channels=3,
            hidden_channels=8,
            out_channels=5,
            n_layers=2,
            residuals=True,
            batch_norm=True,
            dropout=0.0,
        ),
        ResidualGatedGCN(
            in_channels=3,
            hidden_channels=8,
            out_channels=5,
            n_layers=2,
            dropout=0.0,
        ),
        MLP(
            in_channels=3,
            hidden_channels=8,
            out_channels=5,
            n_layers=2,
            dropout=0.0,
            batch_norm=True,
            residuals=True,
        ),
    ]


@pytest.mark.parametrize("model", graph_models())
def test_graph_models_keep_their_logits(model):
    model.eval()
    batch = graph_batch()

    logits, embedding = model(batch, return_embedding=True)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    assert embedding.shape == (BATCH_SIZE, 8)
    assert torch.allclose(logits, model.readout.head(embedding))


def test_cwn_keeps_its_logits():
    torch.manual_seed(0)
    model = CWN(
        in_channels=[3, 3, 3],
        hidden_channels=[4, 4, 4],
        n_layers=1,
        out_channels=5,
    )
    model.eval()
    batch = simplicial_batch()

    logits, embedding = model(batch, return_embedding=True)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    # One block of pooled features per rank.
    assert embedding.shape == (BATCH_SIZE, 12)
    assert torch.allclose(logits, model.readout.head(embedding))


def test_sccnn_keeps_its_logits():
    torch.manual_seed(0)
    model = SCCNN(
        in_channels=[3, 3, 3],
        hidden_channels=[4, 4, 4],
        out_channels=5,
        conv_order=1,
        sc_order=2,
        n_layers=1,
    )
    model.eval()
    batch = simplicial_batch(with_laplacians=True)

    logits, embedding = model(batch, return_embedding=True)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    # One block of pooled features per rank.
    assert embedding.shape == (BATCH_SIZE, 12)
    assert torch.allclose(logits, model.readout.head(embedding))


@pytest.mark.parametrize("pool_method", ["sum", "mean"])
def test_graph_models_honour_pool_method(pool_method):
    torch.manual_seed(0)
    model = GCN(
        in_channels=3, hidden_channels=8, out_channels=5, n_layers=1,
        residuals=True, batch_norm=False, dropout=0.0, pool_method=pool_method,
    )
    model.eval()
    batch = graph_batch()
    _, embedding = model(batch, return_embedding=True)

    node_features = batch.x
    for layer in model.hidden_layers:
        node_features = layer(node_features, batch.edge_index)
    first = node_features[batch.batch == 0]
    expected = {"sum": first.sum(0), "mean": first.mean(0)}
    assert torch.allclose(embedding[0], expected[pool_method], atol=1e-6)


@pytest.mark.parametrize("cls", [CWN, SCCNN])
def test_simplicial_models_alternating_readout(cls):
    torch.manual_seed(0)
    kwargs = dict(in_channels=[3, 3, 3], hidden_channels=[4, 4, 4], n_layers=1, out_channels=5)
    if cls is SCCNN:
        kwargs.update(conv_order=1, sc_order=2)
    model = cls(rank_readout="alternating", **kwargs)
    model.eval()
    batch = simplicial_batch(with_laplacians=cls is SCCNN)

    logits, embedding = model(batch, return_embedding=True)

    assert torch.equal(logits, model(batch))
    # The alternating sum keeps the width of one rank.
    assert embedding.shape == (BATCH_SIZE, 4)
    assert torch.allclose(logits, model.readout.head(embedding))


def test_embeddings_separate_different_inputs():
    torch.manual_seed(0)
    model = graph_models()[0]
    model.eval()

    _, first = model(graph_batch(), return_embedding=True)
    _, second = model(graph_batch(), return_embedding=True)

    assert not torch.allclose(first, second)
