from itertools import combinations

import torch
from torch_geometric.transforms import BaseTransform


class NoOpSelectFeatures(BaseTransform):
    """No-op stand-in for the `select_features` pipeline slot.

    `mantra.transforms.SelectFeatures` (representation='sc') requires its
    `src` attribute to already be a `Dict[int, Tensor]`, which none of the
    sibling repo's embedding transforms currently produce for the sc case;
    the working alternative is `PropagateVertexFeatures` below, which writes
    `x_0..x_d` directly onto the `Data` object as an earlier pipeline step.
    This class exists only so that step's output isn't re-derived: it
    satisfies `main.py`'s "exactly one select_features transform" contract
    (`arrange_transforms`) and its `dataset.transform.select_features.representation`
    dispatch check, without touching data that's already in its final form.
    """

    def __init__(self, representation: str = "sc"):
        super().__init__()
        self.representation = representation

    def forward(self, data):
        return data


class PropagateVertexFeatures(BaseTransform):
    """Barycenter-propagates a vertex-level feature tensor to every higher
    simplex of a triangulation, writing `x_0..x_d` directly onto `data`.

    Local re-implementation of `mantra.transforms.PropagateConvexComb`: that
    class assumes its source tensor is a numpy array (`np.mean(x[s-1,:],
    axis=0)`), but the sibling repo's own embedding transforms (e.g.
    `MomentCurveEmbedding`) now store a `torch.Tensor`, so the upstream class
    raises `TypeError` at pre_transform time. This version does the same
    barycenter averaging directly with tensor ops.
    """

    def __init__(self, source: str = "x"):
        super().__init__()
        self.source = source

    def forward(self, data):
        assert "triangulation" in data, "Data object is missing `triangulation`"
        assert self.source in data, f"Data object is missing source tensor `{self.source}`"

        x = getattr(data, self.source)
        triangulation = data.triangulation

        simplices = {tuple(s) for s in triangulation}
        max_dim = len(next(iter(simplices)))
        for simplex in triangulation:
            for dim in range(1, max_dim):
                simplices.update(combinations(simplex, r=dim))
        simplices = sorted(sorted(simplices), key=len)

        data.x_0 = x
        for dim in range(1, max_dim):
            simplices_ = [s for s in simplices if len(s) == dim + 1]
            # triangulation vertex indices are 1-indexed
            idx = torch.tensor([[v - 1 for v in s] for s in simplices_], dtype=torch.long)
            data[f"x_{dim}"] = x[idx].mean(dim=1)

        return data
