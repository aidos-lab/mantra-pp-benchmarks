"""Every model must expose an embedding without changing its logits.

The Pachner objectives read the embedding that feeds the classifier
head, so ``forward_with_embedding`` has to return exactly the logits
``forward`` returned before the split.
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

    logits, embedding = model.forward_with_embedding(batch)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    assert embedding.shape[0] == BATCH_SIZE
    assert embedding.shape[1] > 0
    assert torch.isfinite(embedding).all()


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

    logits, embedding = model.forward_with_embedding(batch)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    assert embedding.shape == (BATCH_SIZE, 4)


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

    logits, embedding = model.forward_with_embedding(batch)

    assert torch.equal(logits, model(batch))
    assert logits.shape == (BATCH_SIZE, 5)
    # One block of pooled features per rank.
    assert embedding.shape == (BATCH_SIZE, 12)


def test_embeddings_separate_different_inputs():
    torch.manual_seed(0)
    model = graph_models()[0]
    model.eval()

    _, first = model.forward_with_embedding(graph_batch())
    _, second = model.forward_with_embedding(graph_batch())

    assert not torch.allclose(first, second)
