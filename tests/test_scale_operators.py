import torch
from torch_geometric.data import Data

from src.transforms import ScaleOperators


def test_operators_get_unit_spectral_radius_and_incidences_stay():
    laplacian = torch.tensor([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])
    incidence = torch.tensor([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    data = ScaleOperators()(
        Data(hodge_laplacian_1=laplacian.to_sparse(), incidence_1=incidence.to_sparse())
    )

    scaled = data.hodge_laplacian_1.to_dense()
    assert torch.allclose(scaled, laplacian / 4)
    assert torch.linalg.eigvalsh(scaled).abs().max() <= 1
    assert torch.equal(data.incidence_1.to_dense(), incidence)
