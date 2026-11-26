import torch
from torch_geometric.data import Batch, Data

from src.models.set.inv_feature_mlp import InvariantFeatureMLP
from src.transforms import HullVertices

CROSS = [[0, 0, 0, 0]] + [[s * (i == j) for j in range(4)] for i in range(4) for s in (1, -1)]


def _set(points, mask=None):
    data = Data(x=torch.tensor(points, dtype=torch.float32))
    if mask is not None:
        data.hull_mask = torch.tensor(mask)
    return data


def test_hull_vertices_marks_the_extreme_points():
    data = HullVertices()(Data(coords=torch.tensor(CROSS + [[0.25, 0.25, 0, 0]], dtype=torch.float32)))
    assert data.hull_mask.tolist() == [False] + [True] * 8 + [False]


def test_padding_does_not_change_a_set_prediction():
    torch.manual_seed(0)
    model = InvariantFeatureMLP(4, 3, max_points=12, hidden_channels=16).eval()
    alone = model(Batch.from_data_list([_set(CROSS)]))
    with_larger = model(Batch.from_data_list([_set(CROSS), _set(CROSS + [[2, 0, 0, 0], [0, 2, 0, 0]])]))
    assert torch.allclose(alone[0], with_larger[0], atol=1e-5)


def test_point_mask_hides_the_other_points():
    torch.manual_seed(0)
    model = InvariantFeatureMLP(4, 3, max_points=9, hidden_channels=16, point_mask="hull_mask").eval()
    mask = [False] + [True] * 8 + [False]
    a = model(Batch.from_data_list([_set(CROSS + [[0.25, 0.25, 0, 0]], mask)]))
    b = model(Batch.from_data_list([_set(CROSS + [[0, 0.5, 0, 0]], mask)]))
    assert torch.allclose(a, b)
    assert a.shape == (1, 3)


def test_node_level_predicts_every_point_in_batch_order():
    torch.manual_seed(0)
    model = InvariantFeatureMLP(4, 2, max_points=12, hidden_channels=16, task_level="node").eval()
    small, large = _set(CROSS), _set(CROSS + [[2, 0, 0, 0], [0, 2, 0, 0]])
    out = model(Batch.from_data_list([small, large]))
    assert out.shape == (9 + 11, 2)
    assert torch.allclose(out[:9], model(Batch.from_data_list([small])), atol=1e-5)
    assert torch.allclose(out[9:], model(Batch.from_data_list([large])), atol=1e-5)
