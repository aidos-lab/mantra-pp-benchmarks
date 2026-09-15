"""Tests for the per-batch ID permutation in src/utils/augment.py."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.utils.augment import permute_ids


def graph_batch():
    first = Data(x=torch.arange(8.0).view(4, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
    second = Data(x=torch.arange(10.0, 16.0).view(3, 2), edge_index=torch.empty(2, 0, dtype=torch.long))
    return Batch.from_data_list([first, second])


def rows(x):
    return {tuple(r.tolist()) for r in x}


class TestGraphs:
    def test_rows_are_permuted_within_each_graph(self):
        batch = graph_batch()
        before = batch.x.clone()
        generator = torch.Generator().manual_seed(0)

        permute_ids(batch, generator=generator)

        assert rows(batch.x[:4]) == rows(before[:4])
        assert rows(batch.x[4:]) == rows(before[4:])
        assert batch.x.shape == before.shape

    def test_only_leading_columns_move(self):
        batch = graph_batch()
        batch.x = torch.cat([batch.x, torch.arange(7.0).view(7, 1)], dim=-1)
        before = batch.x.clone()

        permute_ids(batch, n_columns=2, generator=torch.Generator().manual_seed(0))

        # The third column is a node attribute (e.g. node type) and stays.
        assert torch.equal(batch.x[:, 2], before[:, 2])
        assert rows(batch.x[:4, :2]) == rows(before[:4, :2])
        assert rows(batch.x[4:, :2]) == rows(before[4:, :2])

    def test_some_permutation_actually_happens(self):
        changed = False
        for seed in range(5):
            batch = graph_batch()
            before = batch.x.clone()
            permute_ids(batch, generator=torch.Generator().manual_seed(seed))
            changed |= not torch.equal(batch.x, before)
        assert changed


class TestComplexes:
    def triangle_pair(self):
        """Two triangles sharing edge (0,1): vertices 0..3, edges, faces."""
        batch = Data()
        batch.x_0 = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
        batch.batch_0 = torch.zeros(4, dtype=torch.long)
        edges = [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3)]
        faces = [(0, 1, 2), (0, 1, 3)]
        batch.x_1 = torch.stack([(batch.x_0[a] + batch.x_0[b]) / 2 for a, b in edges])
        batch.x_2 = torch.stack([batch.x_0[list(f)].mean(0) for f in faces])
        batch.batch_1 = torch.zeros(len(edges), dtype=torch.long)
        batch.batch_2 = torch.zeros(len(faces), dtype=torch.long)
        idx1, val1 = [], []
        for e, (a, b) in enumerate(edges):
            idx1 += [[a, e], [b, e]]
            val1 += [-1.0, 1.0]
        batch.incidence_1_idx = torch.tensor(idx1).t()
        batch.incidence_1_val = torch.tensor(val1)
        idx2, val2 = [], []
        for f, tri in enumerate(faces):
            for sign, (a, b) in zip((1.0, -1.0, 1.0), [(tri[1], tri[2]), (tri[0], tri[2]), (tri[0], tri[1])]):
                idx2.append([edges.index((a, b)), f])
                val2.append(sign)
        batch.incidence_2_idx = torch.tensor(idx2).t()
        batch.incidence_2_val = torch.tensor(val2)
        return batch, edges, faces

    def test_higher_ranks_are_barycentres_of_the_permuted_vertices(self):
        batch, edges, faces = self.triangle_pair()

        permute_ids(batch, generator=torch.Generator().manual_seed(1))

        x_0 = batch.x_0
        expected_1 = torch.stack([(x_0[a] + x_0[b]) / 2 for a, b in edges])
        expected_2 = torch.stack([x_0[list(f)].mean(0) for f in faces])
        assert torch.allclose(batch.x_1, expected_1)
        assert torch.allclose(batch.x_2, expected_2)

    def test_identity_permutation_reproduces_the_input_features(self):
        batch, _, _ = self.triangle_pair()
        x_1, x_2 = batch.x_1.clone(), batch.x_2.clone()
        # A single-vertex "complex" per vertex forces the identity.
        batch.batch_0 = torch.arange(4)

        permute_ids(batch)

        assert torch.allclose(batch.x_1, x_1)
        assert torch.allclose(batch.x_2, x_2)

    def test_n_columns_is_rejected_for_complexes(self):
        batch, _, _ = self.triangle_pair()
        with pytest.raises(ValueError, match="n_columns"):
            permute_ids(batch, n_columns=1)
