from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.models.simplicial.sccnn import SCCNN
from src.transforms import SparseOperatorsToCOO
from src.utils.data_loading import SimplicialData, collate_simplicial

CONFIG = Path(__file__).parents[1] / "config/transform/sparse_operators.yaml"

# Row and column rank of each operator; the last two are not kept.
OPERATORS = {
    "hodge_laplacian_0": (0, 0),
    "down_laplacian_1": (1, 1),
    "up_laplacian_1": (1, 1),
    "down_laplacian_2": (2, 2),
    "up_laplacian_2": (2, 2),
    "incidence_1": (0, 1),
    "incidence_2": (1, 2),
    "coadjacency_1": (1, 1),
    "hodge_laplacian_3": (3, 3),
}
SIZES = [(3, 4, 2, 2), (5, 2, 3, 1), (4, 6, 5, 3)]


def _to_coo():
    return instantiate(OmegaConf.load(CONFIG))


def _sample(sizes, seed):
    g = torch.Generator().manual_seed(seed)
    data = Data(
        y=torch.tensor([float(seed)]),
        dimension=torch.tensor(3),
        triangle_mask=torch.rand(sizes[2], generator=g) < 0.5,
        candidate_index_2=torch.randint(sizes[0], (3, 2), generator=g),
    )
    for rank, n in enumerate(sizes):
        data[f"x_{rank}"] = torch.randn(n, 4, generator=g)
    for name, (row_rank, col_rank) in OPERATORS.items():
        shape = (sizes[row_rank], sizes[col_rank])
        dense = torch.randn(shape, generator=g)
        dense *= torch.rand(shape, generator=g) < 0.4
        data[name] = dense.to_sparse()
    return data


def _samples():
    return [_sample(sizes, seed) for seed, sizes in enumerate(SIZES)]


def test_kept_operators_become_index_and_values_and_the_rest_is_dropped():
    sample = _sample(SIZES[0], 0)
    out = _to_coo()(sample)

    assert isinstance(out, SimplicialData)
    for key in ("coadjacency_1", "hodge_laplacian_3", "x_3", "incidence_1"):
        assert key not in out
    dense = torch.sparse_coo_tensor(
        out.incidence_1_idx, out.incidence_1_val, (3, 4)
    ).to_dense()
    assert torch.equal(dense, sample.incidence_1.to_dense())
    assert torch.equal(out.x_2, sample.x_2)
    assert torch.equal(out.triangle_mask, sample.triangle_mask)


def test_operators_outside_the_kept_ranks_or_the_sample_raise():
    with pytest.raises(ValueError):
        SparseOperatorsToCOO(keep=("hodge_laplacian_3",), max_rank=2)
    with pytest.raises(KeyError):
        SparseOperatorsToCOO(keep=("adjacency_1",))(_sample(SIZES[0], 0))


def test_coo_batches_are_block_diagonal_with_per_rank_batch_vectors():
    samples = _samples()
    batch = collate_simplicial([_to_coo()(s) for s in samples])

    for name in _to_coo().keep:
        blocks = torch.block_diag(*(s[name].to_dense() for s in samples))
        batched = torch.sparse_coo_tensor(
            batch[f"{name}_idx"], batch[f"{name}_val"], blocks.shape
        ).to_dense()
        assert torch.equal(batched, blocks), name
    for rank in range(3):
        counts = torch.tensor([sizes[rank] for sizes in SIZES])
        expected = torch.arange(3).repeat_interleave(counts)
        assert torch.equal(batch[f"batch_{rank}"], expected)
    assert batch.y.tolist() == [0.0, 1.0, 2.0]
    offsets = torch.tensor([0, 3, 8]).repeat_interleave(2)
    candidates = torch.cat([s.candidate_index_2 for s in samples], dim=1)
    assert torch.equal(batch.candidate_index_2, candidates + offsets)


@pytest.mark.parametrize(
    "task_level,rows",
    [("graph", 3), ("node", 12), ("triangle", None), ("candidate", 6)],
)
def test_sccnn_outputs_match_the_sparse_path(task_level, rows):
    samples = _samples()
    torch.manual_seed(0)
    model = SCCNN(
        in_channels=[4, 4, 4], hidden_channels=[8, 8, 8], out_channels=2,
        sc_order=3, n_layers=2, task_level=task_level,
        head_layers=0 if task_level == "graph" else 2,
    ).eval()

    sparse = model(collate_simplicial(samples))
    coo = model(collate_simplicial([_to_coo()(s) for s in samples]))

    if rows is None:
        rows = sum(int(s.triangle_mask.sum()) for s in samples)
    assert sparse.shape == (rows, 2)
    assert torch.allclose(coo, sparse, atol=1e-6)


def test_coo_caches_load_with_weights_only(tmp_path):
    """`InMemoryDataset.save` stores `(data.to_dict(), slices, data_cls)`
    and loads it with `weights_only=True`."""
    sample = _to_coo()(_sample(SIZES[0], 0))
    torch.save((sample.to_dict(), {}, type(sample)), tmp_path / "data.pt")
    data, _, cls = torch.load(tmp_path / "data.pt", weights_only=True)
    assert torch.equal(
        cls.from_dict(data).incidence_2_idx, sample.incidence_2_idx
    )
