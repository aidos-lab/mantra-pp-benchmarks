import torch
from torch_geometric.data import Batch, Data

from baseline import VertexCount


def test_vertex_count_predicts_h11_as_class_index_or_value():
    batch = Batch.from_data_list(
        [Data(coords=torch.zeros(n, 4), n_vertices=n) for n in (11, 12, 13)]
    )
    # 13 - 5 = 8 is not a train class and falls back to index 0.
    assert VertexCount({6: 0, 7: 1})(batch).argmax(1).tolist() == [0, 1, 0]
    assert VertexCount({})(batch).squeeze(1).tolist() == [6.0, 7.0, 8.0]
