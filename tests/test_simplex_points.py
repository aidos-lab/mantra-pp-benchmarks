import pytest
import torch
from torch_geometric.data import Batch, Data

from src.models.dect import DECT
from src.models.point_cloud.pointnet import PointNetPP
from src.models.set.deep_sets import DeepSets
from src.models.utils import target_rows
from src.transforms import FaceIndex, SimplexPoints


def _cone(shift=0.0):
    """Cone over the boundary of a tetrahedron, origin first, with the
    two boundary triangles (2,3,4) and (1,2,3) as targets, in that order."""
    coords = torch.tensor(
        [[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, -1, -1]]
    )
    return Data(
        triangulation=[[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 4, 5], [1, 3, 4, 5]],
        coords=coords + shift,
        x=coords + shift,
        simplex_index_2=torch.tensor([[2, 1], [3, 2], [4, 3]]),
    )


def test_simplex_points_order_features_and_triangle_rows():
    data = SimplexPoints(max_rank=2)(_cone())
    # 5 vertices, 10 edges, 10 triangles; 3 coordinates and 3 ranks.
    assert data.x.shape == (25, 6)
    assert data.vertex_mask.tolist() == [True] * 5 + [False] * 20
    assert torch.equal(data.x[:5, :3], data.coords)
    assert data.x[:, 3:].argmax(1).tolist() == [0] * 5 + [1] * 10 + [2] * 10
    # Edge (1,2) comes first among the edges: origin and (1,0,0).
    assert data.x[5, :3].tolist() == [0.5, 0, 0]
    for row, triangle in zip(data.triangle_node_index, data.simplex_index_2.t()):
        assert torch.allclose(data.x[row, :3], data.coords[triangle].mean(0))


def test_simplex_points_without_triangles_keep_the_vertex_sum():
    data = SimplexPoints(max_rank=1)(_cone())
    assert "triangle_node_index" not in data
    batch = Batch.from_data_list([data, data.clone()])
    rows = target_rows(batch.x, batch, "triangle")
    assert torch.equal(rows, batch.x[batch.simplex_index_2].sum(0))


def test_triangle_rows_follow_the_batch_offsets():
    first, second = SimplexPoints(max_rank=2)(_cone()), SimplexPoints(max_rank=2)(_cone(1.0))
    batch = Batch.from_data_list([first, second])
    rows = target_rows(batch.x, batch, "triangle")
    assert torch.equal(rows, torch.cat([d.x[d.triangle_node_index] for d in (first, second)]))
    assert torch.equal(target_rows(batch.x, batch, "node"), torch.cat([first.x[:5], second.x[:5]]))


def test_face_index_shapes_and_batch_offsets():
    data = FaceIndex(max_rank=3)(_cone())
    assert [data[f"faces_index_{k}"].shape for k in (1, 2, 3)] == [(2, 10), (3, 10), (4, 4)]
    assert data.faces_index_3.t().tolist() == [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4]]
    batch = Batch.from_data_list([data, data.clone()])
    assert torch.equal(batch.faces_index_2[:, 10:], data.faces_index_2 + 5)


@pytest.mark.parametrize("max_rank, euler", [(0, 5), (1, -5), (2, 5), (3, 1)])
def test_dect_counts_the_simplices_below_the_last_threshold(max_rank, euler):
    batch = Batch.from_data_list([FaceIndex(max_rank=3)(_cone()), FaceIndex(max_rank=3)(_cone())])
    model = DECT(3, 1, num_directions=4, resolution=8, radius=4.0, scale=1e4, max_rank=max_rank)
    image = model.ect(batch)
    assert image.shape == (2, 4, 8)
    # Every vertex has norm below the radius, so all simplices count at
    # the last threshold and none at the first: 5 - 10 + 10 - 4.
    assert torch.allclose(image[:, :, -1], torch.full((2, 4), float(euler)))
    assert torch.allclose(image[:, :, 0], torch.zeros(2, 4))


@pytest.mark.parametrize("task_level, rows", [("graph", 2), ("node", 10), ("triangle", 4)])
def test_dect_output_rows_and_direction_gradient(task_level, rows):
    batch = Batch.from_data_list([FaceIndex(max_rank=3)(_cone()), FaceIndex(max_rank=3)(_cone(0.5))])
    model = DECT(3, 1, num_directions=4, resolution=8, radius=4.0, scale=8.0, max_rank=3, hidden_channels=16, task_level=task_level)
    out = model(batch)
    assert out.shape == (rows, 1)
    out.sum().backward()
    assert model.directions.grad.abs().sum() > 0


def test_point_models_read_simplex_points():
    batch = Batch.from_data_list([SimplexPoints(max_rank=2)(_cone()), SimplexPoints(max_rank=2)(_cone(1.0))])
    deepsets = DeepSets(6, hidden_channels=16, task_level="triangle")
    assert deepsets(batch).shape == (4, 1)
    pointnet = PointNetPP(6, hidden_channels=16, pos_channels=3, task_level="node")
    assert pointnet(batch).shape == (10, 1)


def test_pointnet_without_pos_channels_takes_every_column_as_position():
    torch.manual_seed(0)
    plain = PointNetPP(3, hidden_channels=16)
    torch.manual_seed(0)
    split = PointNetPP(3, hidden_channels=16, pos_channels=3)
    batch = Batch.from_data_list([_cone(), _cone(1.0)])
    assert torch.equal(plain(batch), split(batch))
