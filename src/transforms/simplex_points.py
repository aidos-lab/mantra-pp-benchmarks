"""The simplices of a triangulation as inputs of models without message
passing: one point per simplex, or the vertex indices of every rank."""

from itertools import combinations

import torch
from torch_geometric.transforms import BaseTransform


def faces_by_rank(triangulation, max_rank: int) -> list[list[tuple[int, ...]]]:
    """Faces of rank 0 to `max_rank` of the 1-indexed top simplices, each
    rank sorted lexicographically; a rank the complex lacks is empty."""
    return [
        sorted({face for simplex in triangulation for face in combinations(simplex, rank + 1)})
        for rank in range(max_rank + 1)
    ]


class SimplexPoints(BaseTransform):
    """Write one point per simplex of rank at most `max_rank` to `x`.

    A point is the barycentre of its simplex followed by a one-hot of the
    rank, so a point-set model sees the triangulation up to that rank
    without any edges. The vertices come first and in vertex order and
    carry `vertex_mask`, the higher simplices follow rank by rank; this is
    the node set of `mantra.representations.HasseDiagram` with
    `rank_feature`. With triangles among the points and `simplex_index_2`
    present, `triangle_node_index` holds the point of every target
    triangle, which `target_rows` reads instead of the three vertex rows.

    Parameters
    ----------
    max_rank : int
        Highest rank that becomes points; 0 keeps the vertices alone.
    source : str
        Attribute holding one coordinate row per vertex, in vertex order.
    """

    def __init__(self, max_rank: int, source: str = "coords"):
        super().__init__()
        self.max_rank = max_rank
        self.source = source

    def forward(self, data):
        coords = data[self.source]
        ranks = faces_by_rank(data.triangulation, self.max_rank)
        rows = []
        for rank, faces in enumerate(ranks):
            index = torch.tensor(faces, dtype=torch.long).reshape(-1, rank + 1) - 1
            one_hot = torch.zeros(len(faces), self.max_rank + 1)
            one_hot[:, rank] = 1
            rows.append(torch.cat([coords[index].mean(1), one_hot], dim=1))
        data.x = torch.cat(rows)
        data.vertex_mask = torch.arange(data.x.shape[0]) < len(ranks[0])
        if self.max_rank >= 2 and "simplex_index_2" in data:
            offset = len(ranks[0]) + len(ranks[1])
            row = {face: offset + i for i, face in enumerate(ranks[2])}
            triangles = (data.simplex_index_2.t() + 1).sort(dim=1).values.tolist()
            data.triangle_node_index = torch.tensor(
                [row[tuple(t)] for t in triangles], dtype=torch.long
            )
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_rank={self.max_rank}, source={self.source!r})"


class FaceIndex(BaseTransform):
    """Write the vertices of the simplices of every rank from 1 to
    `max_rank` as `faces_index_<rank>`, zero-based and of shape
    `(rank + 1, n_simplices)`.

    The name carries `index`, so PyG offsets the entries by the node
    counts when it collates a batch. A model that works on the vertex
    coordinates alone, such as `src.models.dect.DECT`, reads them.

    Parameters
    ----------
    max_rank : int
        Highest rank written.
    """

    def __init__(self, max_rank: int = 4):
        super().__init__()
        self.max_rank = max_rank

    def forward(self, data):
        ranks = faces_by_rank(data.triangulation, self.max_rank)
        for rank, faces in enumerate(ranks[1:], start=1):
            index = torch.tensor(faces, dtype=torch.long).reshape(-1, rank + 1)
            data[f"faces_index_{rank}"] = index.t().contiguous() - 1
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_rank={self.max_rank})"
