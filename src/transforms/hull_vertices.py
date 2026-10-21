"""Mark the points that are vertices of the convex hull of a point set."""

import numpy as np
import torch
from scipy.spatial import ConvexHull
from torch_geometric.transforms import BaseTransform


class HullVertices(BaseTransform):
    """Write a boolean mask of the extreme points of `source`.

    The lattice points of a reflexive polytope include its interior
    and boundary points; the polytope's vertices are the extreme points
    of their convex hull. Models that want the vertex matrix of the
    polytope, as in Ewert et al. (2024), read the mask.

    Parameters
    ----------
    source : str
        Attribute holding one coordinate row per point.
    target : str
        Boolean attribute to write, one entry per point.
    """

    def __init__(self, source: str = "coords", target: str = "hull_mask"):
        super().__init__()
        self.source = source
        self.target = target

    def forward(self, data):
        points = np.asarray(data[self.source], dtype=np.float64)
        mask = torch.zeros(points.shape[0], dtype=torch.bool)
        mask[ConvexHull(points).vertices] = True
        data[self.target] = mask
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source={self.source!r}, target={self.target!r})"
