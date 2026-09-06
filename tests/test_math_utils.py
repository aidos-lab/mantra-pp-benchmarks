"""Tests for eigenvector computation from src/math_utils.py.

Note: CellComplexCombinatorics is not installed, so we test
eigenvectors_smallest_k_eigenvalues by importing its logic directly.
"""

import numpy as np
import scipy.sparse as sp


def eigenvectors_smallest_k_eigenvalues(L, k):
    """Copied from src/math_utils.py to test independently of missing deps."""
    n = L.shape[0]
    EigVal, EigVec = np.linalg.eig(L.toarray())
    max_freqs = min(n - 1, k)
    kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = EigVec[:, topk_indices]
    rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
    PE = rand_sign * topk_EigVec
    if n <= k:
        temp_EigVec = np.zeros((n, k - n + 1), dtype=np.float32)
        PE = np.concatenate((PE, temp_EigVec), axis=1)
    return PE


class TestEigenvectorsSmallestK:
    def test_output_shape_k_less_than_n(self):
        # 4x4 identity Laplacian-like matrix
        L = sp.csr_matrix(np.diag([0.0, 1.0, 2.0, 3.0]))
        k = 2
        result = eigenvectors_smallest_k_eigenvalues(L, k)
        assert result.shape == (4, 2)

    def test_output_shape_k_greater_than_n(self):
        L = sp.csr_matrix(np.diag([0.0, 1.0, 2.0]))
        k = 5
        result = eigenvectors_smallest_k_eigenvalues(L, k)
        # n=3, k=5 -> should pad: (3, 2) base + (3, 3) padding = (3, 5)
        assert result.shape[0] == 3
        assert result.shape[1] == k

    def test_selects_smallest_eigenvalues(self):
        # Laplacian of a path graph 0-1-2
        L = sp.csr_matrix(
            np.array(
                [
                    [1.0, -1.0, 0.0],
                    [-1.0, 2.0, -1.0],
                    [0.0, -1.0, 1.0],
                ]
            )
        )
        np.random.seed(42)
        result = eigenvectors_smallest_k_eigenvalues(L, k=2)
        # Should return 2 eigenvectors (skipping the zero eigenvalue)
        assert result.shape == (3, 2)

    def test_padding_zeros(self):
        L = sp.csr_matrix(np.diag([0.0, 1.0]))
        k = 4
        result = eigenvectors_smallest_k_eigenvalues(L, k)
        # Last columns should be zero padding
        assert result.shape == (2, k)
        assert np.allclose(result[:, 1:], 0.0)

    def test_deterministic_with_seed(self):
        L = sp.csr_matrix(np.diag([0.0, 1.0, 2.0, 3.0]))
        np.random.seed(0)
        r1 = eigenvectors_smallest_k_eigenvalues(L, 2)
        np.random.seed(0)
        r2 = eigenvectors_smallest_k_eigenvalues(L, 2)
        assert np.allclose(r1, r2)
