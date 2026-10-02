"""Orientation-sensitive random-walk encoding: a signed analogue of RWPE.

Intuition: walk between adjacent top simplices carrying an "up" flag, and flip it whenever you
cross into a simplex whose (arbitrary) orientation disagrees with yours. On an orientable
manifold no walk ever comes back flipped; on a non-orientable one, long walks through a
crosscap come back flipped about half the time.

Definition, for a d-dimensional pseudomanifold with N top simplices (each oriented by its sorted
vertex order): B is the boundary matrix (facets x top simplices) and L = B^T B the Hodge
Laplacian of the top simplices. W = I - L / (2(d+1)) is a lazy signed walk (it stays with
probability 1/2) and U = |W| its unsigned version. For each top simplex t and walk length
s = 1, 2, 4, ..., 2^(k-1),

    rho_s(t) = (W^s)_tt / (U^s)_tt,

i.e. (same-sheet - opposite-sheet) / (same + opposite) return probability on the orientation
double cover. It is exactly 1 on orientable closed manifolds and tends to 0 on non-orientable
ones (at rate lambda_min(L) / (2(d+1))). It does not depend on the orientation chosen for each
simplex, on the vertex labels, nor on the size. Vertices take the mean over their top simplices.

The dimension k is the number of dyadic walk lengths (scales), not the number of steps:
k = 16 means walks of 1, 2, 4, ..., 32768 steps. Each scale is read off one eigendecomposition
of W and of U, O(N^3 + k N^2), so long walks cost nothing extra. Subdividing a complex shrinks
lambda_min(L) (about 8x per barycentric subdivision in 2D), so larger complexes need more scales.
"""

from typing import Optional

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform


def _return_mass(M: np.ndarray, steps: np.ndarray, scale: float) -> np.ndarray:
    """diag(M^s / scale^s) for each s in `steps`, for a symmetric M. Shape [N, len(steps)]."""
    lam, Q = np.linalg.eigh(M)
    return (Q**2) @ np.power(lam[:, None] / scale, steps[None, :])


def orientation_ratios(top_simplices, num_scales: int) -> np.ndarray:
    """rho_s for every top simplex (in the given order) and s = 1, 2, ..., 2^(num_scales-1)."""
    simplices = [tuple(sorted(s)) for s in top_simplices]
    d = len(simplices[0]) - 1
    facets, rows, cols, signs = {}, [], [], []
    for j, s in enumerate(simplices):
        for i in range(d + 1):
            rows.append(facets.setdefault(s[:i] + s[i + 1 :], len(facets)))
            cols.append(j)
            signs.append((-1.0) ** i)
    B = np.zeros((len(facets), len(simplices)))
    B[rows, cols] = signs
    W = np.eye(len(simplices)) - B.T @ B / (2 * (d + 1))
    U = np.abs(W)
    steps = 2.0 ** np.arange(num_scales)
    # Rescale both walks by U's largest eigenvalue: it is 1 for closed pseudomanifolds, and
    # below 1 with boundary, where long powers would otherwise underflow. The ratio is unchanged.
    scale = np.linalg.eigvalsh(U)[-1]
    return _return_mass(W, steps, scale) / _return_mass(U, steps, scale)


class OrientationPE(BaseTransform):
    """Adds `attr_name` with the orientation-sensitive random-walk encoding.

    Parameters
    ----------
    num_scales : int
        Number of dyadic walk lengths 1, 2, ..., 2^(num_scales-1) (the encoding dimension).
    attr_name : str
        Attribute to write.
    level : str
        "vertex": [n_vertices, num_scales], the mean over the top simplices of each vertex
        (for the 1-skeleton / Hasse diagram, or propagated with PropagateConvexComb).
        "top": [n_top_simplices, num_scales], in the node order of the DualGraph representation.
    append_to : str or None
        If set, concatenate onto this existing attribute instead (e.g. "moment_curve_embedding"
        or "random_walk_pe"), so the encoding can be added to any existing configuration.
    """

    def __init__(
        self,
        num_scales: int = 16,
        attr_name: str = "orientation_pe",
        level: str = "vertex",
        append_to: Optional[str] = None,
    ):
        super().__init__()
        assert level in ("vertex", "top"), f"Invalid level: {level}"
        self.num_scales = num_scales
        self.attr_name = attr_name
        self.level = level
        self.append_to = append_to

    def forward(self, data):
        # Same order as the DualGraph nodes: unique top simplices, sorted lexicographically.
        top = sorted({tuple(s) for s in data["triangulation"]})
        rho = orientation_ratios(top, self.num_scales)
        if self.level == "vertex":
            n = max(max(s) for s in top)
            total, count = np.zeros((n, self.num_scales)), np.zeros(n)
            for s, r in zip(top, rho):
                idx = np.asarray(s) - 1  # MANTRA triangulations are 1-indexed
                total[idx] += r
                count[idx] += 1
            rho = total / count[:, None]
        pe = torch.from_numpy(rho).to(torch.float32)
        if self.append_to is not None:
            data[self.append_to] = torch.cat([data[self.append_to].to(torch.float32), pe], dim=1)
        else:
            data[self.attr_name] = pe
        return data
