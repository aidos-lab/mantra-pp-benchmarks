from torch_geometric.data import Data

from mantra.transforms import MomentCurveEmbedding, PropagateConvexComb


def test_moment_curve_features_are_propagated_to_simplices():
    data = Data(
        triangulation=[[1, 2, 3]],
        n_vertices=3,
        dimension=2,
    )

    data = MomentCurveEmbedding()(data)
    data = PropagateConvexComb(source="moment_curve_embedding")(data)

    assert data.x_0.shape == (3, 5)
    assert data.x_1.shape == (3, 5)
    assert data.x_2.shape == (1, 5)
