import torch
from torch_geometric.data import Data

from mantra.transforms import PropagateConvexComb

from src.transforms import VertexDegree

# Octahedron: 6 vertices of degree 4, 12 edges, 8 triangles.
OCTAHEDRON = [[1, 2, 3], [1, 3, 4], [1, 4, 5], [1, 2, 5], [2, 3, 6], [3, 4, 6], [4, 5, 6], [2, 5, 6]]
# Stellar subdivision of one face of the tetrahedron: vertices 1..3 have
# degree 4, the apexes 4 and 5 degree 3.
TETRA_STELLAR = [[1, 2, 4], [1, 3, 4], [2, 3, 4], [1, 2, 5], [2, 3, 5], [1, 3, 5]]


def test_degrees_follow_the_vertex_labels():
    data = VertexDegree()(Data(triangulation=TETRA_STELLAR))
    assert torch.equal(data.degree, torch.tensor([[4.0], [4.0], [4.0], [3.0], [3.0]]))


def test_euler_characteristic_from_the_degrees():
    # chi = sum_v (1 - deg v / 6) on a closed surface: 2 for the sphere.
    data = VertexDegree()(Data(triangulation=OCTAHEDRON))
    assert abs(float((1 - data.degree / 6).sum()) - 2.0) < 1e-5


def test_propagates_to_the_higher_ranks():
    data = Data(triangulation=OCTAHEDRON, dimension=2)
    data = PropagateConvexComb(source="degree")(VertexDegree()(data))
    assert data.x_0.shape == (6, 1) and data.x_1.shape == (12, 1) and data.x_2.shape == (8, 1)
    assert torch.allclose(data.x_2, torch.full((8, 1), 4.0))
