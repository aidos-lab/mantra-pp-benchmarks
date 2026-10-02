"""Tests for the orientation-sensitive random-walk encoding."""

import random

import numpy as np
import torch
from torch_geometric.data import Data

from src.transforms.orientation_pe import OrientationPE, orientation_ratios

TETRAHEDRON = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]  # S^2
RP2 = [
    [1, 2, 3], [1, 2, 6], [1, 3, 5], [1, 4, 5], [1, 4, 6],
    [2, 3, 4], [2, 4, 5], [2, 5, 6], [3, 4, 6], [3, 5, 6],
]  # fmt: skip
TORUS = [sorted({i % 7 + 1, (i + 1) % 7 + 1, (i + 3) % 7 + 1}) for i in range(7)] + [
    sorted({i % 7 + 1, (i + 2) % 7 + 1, (i + 3) % 7 + 1}) for i in range(7)
]
S3 = [[v for v in range(1, 6) if v != i] for i in range(1, 6)]  # boundary of the 4-simplex


def test_orientable_closed_manifolds_give_one():
    for tri in (TETRAHEDRON, TORUS, S3):
        assert np.allclose(orientation_ratios(tri, 16), 1.0)


def test_non_orientable_decays_to_zero():
    rho = orientation_ratios(RP2, 16)
    assert np.allclose(rho[:, 0], 1.0)  # one step never comes back flipped
    assert np.abs(rho[:, -1]).max() < 1e-3


def test_independent_of_labels_and_orientations():
    perm = list(range(1, 7))
    random.Random(0).shuffle(perm)
    relabelled = [[perm[v - 1] for v in t] for t in RP2]  # also changes simplex orientations
    a = OrientationPE(num_scales=8)(Data(triangulation=RP2)).orientation_pe
    b = OrientationPE(num_scales=8)(Data(triangulation=relabelled)).orientation_pe
    assert torch.allclose(a, b[torch.tensor(perm) - 1], atol=1e-6)


def test_levels_and_append():
    data = Data(triangulation=TORUS, moment_curve_embedding=torch.zeros(7, 5))
    assert OrientationPE(4)(data.clone()).orientation_pe.shape == (7, 4)
    assert OrientationPE(4, level="top")(data.clone()).orientation_pe.shape == (14, 4)
    appended = OrientationPE(4, append_to="moment_curve_embedding")(data.clone())
    assert appended.moment_curve_embedding.shape == (7, 9)


def test_with_boundary_does_not_underflow():
    disk = TETRAHEDRON[:3]  # S^2 minus a triangle
    rho = orientation_ratios(disk, 20)
    assert np.isfinite(rho).all() and np.allclose(rho, 1.0)
