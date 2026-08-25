"""Tests for the PointNet++ model (src/models/PointNetPP.py).

FPS starts deterministically at each complex's first point, so a
forward pass is reproducible without seeding; the tests check shapes,
determinism, gradient flow, and the exact behavior of the geometry
helpers.
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.PointNetPP import (
    PointNetPP,
    farthest_point_indices,
    radius_edges,
)


def _point_batch():
    """Two 4-d lattice point sets of different sizes."""
    torch.manual_seed(0)
    clouds = [
        Data(x=torch.randint(-3, 4, (12, 4)).float()),
        Data(x=torch.randint(-3, 4, (7, 4)).float()),
    ]
    return Batch.from_data_list(clouds)


class TestPointNetPP:
    def test_output_shape(self):
        model = PointNetPP(in_channels=4, hidden_channels=16, out_channels=3)
        model.eval()
        out = model(_point_batch())
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_forward_is_deterministic(self):
        model = PointNetPP(in_channels=4, hidden_channels=16, out_channels=2)
        model.eval()
        assert torch.equal(model(_point_batch()), model(_point_batch()))

    def test_gradients_flow_to_all_parameters(self):
        model = PointNetPP(in_channels=4, hidden_channels=16, out_channels=1)
        model(_point_batch()).sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_single_sa_level(self):
        model = PointNetPP(
            in_channels=4, hidden_channels=16, out_channels=1,
            sa_ratios=[0.5], sa_radii=[3.0],
        )
        model.eval()
        out = model(_point_batch())
        assert out.shape == (2, 1)

    def test_tiny_clouds_survive_the_hierarchy(self):
        model = PointNetPP(in_channels=4, hidden_channels=8, out_channels=1)
        model.eval()
        batch = Batch.from_data_list([
            Data(x=torch.tensor([[0.0, 0, 0, 0], [1, 0, 0, 0]])),
            Data(x=torch.tensor([[0.0, 1, 0, 0]])),
        ])
        out = model(batch)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()

    def test_invalid_arguments_rejected(self):
        with pytest.raises(AssertionError):
            PointNetPP(in_channels=4, pool_method='max')
        with pytest.raises(AssertionError):
            PointNetPP(in_channels=4, sa_ratios=[0.5], sa_radii=[2.0, 4.0])


class TestGeometryHelpers:
    def test_fps_picks_the_farthest_point(self):
        pos = torch.tensor([[0.0, 0, 0, 0], [1.0, 0, 0, 0], [10.0, 0, 0, 0]])
        batch = torch.zeros(3, dtype=torch.long)
        idx = farthest_point_indices(pos, batch, ratio=2 / 3)
        assert idx.tolist() == [0, 2]

    def test_fps_returns_flat_per_graph_indices(self):
        pos = torch.tensor(
            [[0.0, 0, 0, 0], [3.0, 0, 0, 0], [0.0, 0, 0, 0], [5.0, 0, 0, 0]]
        )
        batch = torch.tensor([0, 0, 1, 1])
        idx = farthest_point_indices(pos, batch, ratio=0.5)
        assert idx.tolist() == [0, 2]

    def test_radius_edges_exact_edge_set(self):
        src = torch.tensor([[0.0, 0, 0, 0], [2.0, 0, 0, 0], [5.0, 0, 0, 0]])
        batch = torch.zeros(3, dtype=torch.long)
        dst = src[[0, 2]]
        dst_batch = torch.zeros(2, dtype=torch.long)
        edges = radius_edges(src, batch, dst, dst_batch, r=2.0)
        pairs = set(zip(edges[0].tolist(), edges[1].tolist()))
        assert pairs == {(0, 0), (1, 0), (2, 1)}
