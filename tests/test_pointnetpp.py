"""Tests for src/models/PointNetPP.py: shapes, determinism, batching,
the node-level feature-propagation head and argument validation."""

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
    def test_output_shape(self):
        out = _model()(_batch())
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_forward_is_deterministic(self):
        model = _model()
        assert torch.equal(model(_batch()), model(_batch()))

    @pytest.mark.parametrize("task_level", ["graph", "node"])
    def test_batched_output_equals_per_set_outputs(self, task_level):
        """Grouping and interpolation must not cross the sets of a batch."""
        model = _model(task_level=task_level)
        clouds = _clouds()
        batched = model(_batch(clouds))
        single = torch.cat([model(_batch([c])) for c in clouds])
        assert torch.allclose(batched, single, atol=1e-5)

    @pytest.mark.parametrize("task_level", ["graph", "node"])
    def test_gradients_flow_to_all_parameters(self, task_level):
        model = _model(task_level=task_level).train()
        model(_batch()).sum().backward()
        missing = [n for n, p in model.named_parameters() if p.grad is None]
        assert missing == []

    def test_connectivity_blindness(self):
        model = _model()
        clouds = _clouds()
        wired = [
            Data(x=c.x, edge_index=torch.tensor([[0, 1], [1, 0]]))
            for c in clouds
        ]
        assert torch.equal(model(_batch(clouds)), model(_batch(wired)))

    def test_single_sa_level(self):
        out = _model(sa_ratios=[0.5], sa_radii=[3.0])(_batch())
        assert out.shape == (2, 3)

    def test_max_pooling(self):
        out = _model(pool_method="max")(_batch())
        assert out.shape == (2, 3)

    @pytest.mark.parametrize("task_level", ["graph", "node"])
    def test_tiny_sets_survive_the_hierarchy(self, task_level):
        clouds = [
            Data(x=torch.tensor([[0.0, 0, 0, 0], [1, 0, 0, 0]])),
            Data(x=torch.tensor([[0.0, 0, 0, 0]])),
        ]
        out = _model(task_level=task_level)(_batch(clouds))
        assert out.shape == ((3, 3) if task_level == "node" else (2, 3))
        assert torch.isfinite(out).all()

    def test_mismatched_levels_rejected(self):
        with pytest.raises(ValueError):
            PointNetPP(in_channels=4, sa_ratios=[0.5], sa_radii=[2.0, 4.0])


class TestPointNetPPNodeHead:
    def test_node_level_emits_one_row_per_point(self):
        out = _model(task_level="node")(_batch())
        assert out.shape == (19, 3)
        assert torch.isfinite(out).all()

    def test_fp_levels_mirror_sa_levels_plus_input(self):
        model = _model(
            task_level="node",
            sa_ratios=[0.5, 0.5, 0.5],
            sa_radii=[2.0, 3.0, 4.0],
        )
        assert len(model.fp_mlps) == 4
        assert len(_model().fp_mlps) == 0

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
