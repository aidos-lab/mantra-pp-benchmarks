"""Per-rank effective resistance features for simplicial models.

``EffectiveResistanceEmbedding`` (mantra) writes a dict ``er`` mapping
rank k = 1..d to a 1-D vector of per-simplex effective resistances; the
simplicial models (SCCNN, CWN) instead want 2-D feature tensors
``x_k``. This transform reshapes ``er[k]`` into ``x_k = er[k][:, None]``
for k = 1..d. Rank 0 is NOT written: no effective resistance is defined
for vertices in this framework, so the pipeline assigns ``x_0``
separately (random vertex features via SimplexRandomTransform +
SelectFeatures, as in the existing sc sweeps).
"""

from torch_geometric.transforms import BaseTransform


class ERSimplicialFeatures(BaseTransform):
    def forward(self, data):
        assert "er" in data, "Run EffectiveResistanceEmbedding first"

        for k, values in data.er.items():
            data[f"x_{int(k)}"] = values.reshape(-1, 1).float()

        return data
