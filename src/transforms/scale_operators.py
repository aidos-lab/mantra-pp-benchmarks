"""Scale the Laplacian and adjacency operators of a sample to spectral radius at most one."""

import re

import torch
from torch_geometric.transforms import BaseTransform

OPERATOR = re.compile(r"^(\w*laplacian|adjacency)_\d+$")


class ScaleOperators(BaseTransform):
    """Divide every Laplacian and adjacency operator by its largest absolute row sum.

    That Gershgorin bound keeps the spectral radius at most one, the
    convention of polynomial filters (`L / lambda_max` in ChebNet). The
    raw operators of the Calabi-Yau complexes have eigenvalues up to 14,
    which the Chebyshev convolutions of the SCCNN compound layer by layer
    until the sum readout is in the tens of thousands. Incidence matrices
    stay as they are.
    """

    def forward(self, data):
        for key in list(data.keys()):
            if OPERATOR.match(key):
                op = data[key].coalesce()
                idx, val = op.indices(), op.values()
                row_abs = torch.zeros(op.shape[0], dtype=val.dtype).scatter_add_(0, idx[0], val.abs())
                scaled = val / row_abs.max().clamp_min(1.0)
                data[key] = torch.sparse_coo_tensor(idx, scaled, op.shape).coalesce()
        return data
