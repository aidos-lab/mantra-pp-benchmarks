import numpy as np
import scipy
import torch
from toponetx import SimplicialComplex, CellComplex

from CellComplexCombinatorics import hodge_laplacian_matrix


def eigenvectors_smallest_k_eigenvalues(L, k):
    n = L.shape[0]
    # select eigenvectors with smaller eigenvalues O(n + klogk)
    EigVal, EigVec = np.linalg.eig(L.toarray())
    max_freqs = min(n - 1, k)
    kpartition_indices = np.argpartition(EigVal, max_freqs)[: max_freqs + 1]
    topk_eigvals = EigVal[kpartition_indices]
    topk_indices = kpartition_indices[topk_eigvals.argsort()][1:]
    topk_EigVec = EigVec[:, topk_indices]
    # get random flip signs
    rand_sign = 2 * (np.random.rand(max_freqs) > 0.5) - 1.0
    PE = rand_sign * topk_EigVec
    # add paddings
    if n <= k:
        temp_EigVec = np.zeros((n, k - n + 1), dtype=np.float32)
        PE = np.concatenate((PE, temp_EigVec), axis=1)
    return PE


def compute_hodge_laplacian_matrix(
    t_complex: SimplicialComplex | CellComplex, dim: int, signed=True
):
    # Check type of t_complex
    if isinstance(t_complex, SimplicialComplex):
        return t_complex.hodge_laplacian_matrix(dim, signed=signed)
    elif isinstance(t_complex, CellComplex):
        return hodge_laplacian_matrix(t_complex, dim, signed=signed)


def normalize_laplacian(H: scipy.sparse.spmatrix):
    # We assume that the matrix is a scipy sparse matrix. We normalize by multiplying by D^{-1/2} where D is the
    # diagonal matrix of the Hodge Laplacian H.
    inverse_squared_diagonal = (scipy.sparse.diags(H.diagonal()).sqrt()).power(
        -1
    )
    return inverse_squared_diagonal @ H @ inverse_squared_diagonal
