import re

from torch_geometric.transforms import BaseTransform

DEAD_KEYS = ("simplex_trie", "triangulation", "simplex", "intersection_numbers")
DEAD_PATTERNS = (re.compile(r"^(incidence|adjacency|\w*laplacian)_[3-9]$"),)
RANK_FEATURE_PATTERN = re.compile(r"^x_\d+$")


class DropCacheAttributes(BaseTransform):
    """Remove attributes that are dead after preprocessing from a sample.

    Appended to a `pre_transform` chain so that the cached samples hold
    only what the collate functions and models read: `DEAD_KEYS` are the
    Python object graphs the representations leave behind, which
    dominate cache size and load time, and the raw labels; `DEAD_PATTERNS`
    is the connectivity of ranks no model reads (CWN uses ranks 0 to 2,
    SCCNN operators up to rank 2). `extra_keys` names further attributes
    to drop, e.g. the source of `select_features` once it is copied to `x`.
    `graph=True` also drops the per-rank `x_k` a graph representation has
    propagated into `x`; left in the sample, they make `main` take the
    graph for a simplicial complex.
    """

    def __init__(self, extra_keys=(), graph=False):
        super().__init__()
        self.keys = set(DEAD_KEYS) | set(extra_keys)
        self.patterns = list(DEAD_PATTERNS) + ([RANK_FEATURE_PATTERN] if graph else [])

    def forward(self, data):
        for key in list(data.keys()):
            if key in self.keys or any(p.match(key) for p in self.patterns):
                del data[key]
        return data

    def __repr__(self):
        graph = RANK_FEATURE_PATTERN in self.patterns
        return f"{self.__class__.__name__}({sorted(self.keys)}, graph={graph})"
