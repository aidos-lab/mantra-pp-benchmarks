"""Tests for src/models/PointNetPP.py: batching, connectivity blindness,
degenerate sets and the node-level feature-propagation head."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.PointNetPP import PointNetPP


def _clouds():
    """Two 4-d lattice point sets of different sizes, origin first."""
    torch.manual_seed(0)
    clouds = [
        Data(x=torch.randint(-3, 4, (12, 4)).float()),
        Data(x=torch.randint(-3, 4, (7, 4)).float()),
    ]
    for cloud in clouds:
        cloud.x[0] = 0.0
    return clouds


def _batch(clouds=None):
    return Batch.from_data_list(_clouds() if clouds is None else clouds)


def _model(**overrides):
    kwargs = dict(
        in_channels=4, hidden_channels=16, out_channels=3, dropout=0.0
    )
    kwargs.update(overrides)
    torch.manual_seed(0)
    return PointNetPP(**kwargs).eval()


class TestPointNetPP:
    @pytest.mark.parametrize("task_level", ["graph", "node"])
    def test_batched_output_equals_per_set_outputs(self, task_level):
        """Grouping and interpolation must not cross the sets of a batch."""
        model = _model(task_level=task_level)
        clouds = _clouds()
        batched = model(_batch(clouds))
        single = torch.cat([model(_batch([c])) for c in clouds])
        assert batched.shape == ((19, 3) if task_level == "node" else (2, 3))
        assert torch.allclose(batched, single, atol=1e-5)

    def test_connectivity_blindness(self):
        model = _model()
        clouds = _clouds()
        wired = [
            Data(x=c.x, edge_index=torch.tensor([[0, 1], [1, 0]]))
            for c in clouds
        ]
        assert torch.equal(model(_batch(clouds)), model(_batch(wired)))

    @pytest.mark.parametrize("task_level", ["graph", "node"])
    def test_tiny_sets_survive_the_hierarchy(self, task_level):
        clouds = [
            Data(x=torch.tensor([[0.0, 0, 0, 0], [1, 0, 0, 0]])),
            Data(x=torch.tensor([[0.0, 0, 0, 0]])),
        ]
        out = _model(task_level=task_level)(_batch(clouds))
        assert out.shape == ((3, 3) if task_level == "node" else (2, 3))
        assert torch.isfinite(out).all()


class TestPointNetPPNodeHead:
    def test_node_output_sees_the_whole_set(self):
        """The set feature is propagated back to every point, so moving
        a point far from the others still changes their outputs."""
        model = _model(task_level="node")
        clouds = _clouds()
        moved = clouds[0].clone()
        moved.x[-1] += 10.0
        out = model(_batch([clouds[0]]))
        out_moved = model(_batch([moved]))
        assert not torch.allclose(out[0], out_moved[0])
