import sys
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets.graphormer_spd import GraphormerSPDPreTransform


def _make_data(edge_index, num_nodes=None, x=None):
    """Helper to build a Data object with optional node features."""
    data = Data(edge_index=edge_index)
    if x is not None:
        data.x = x
    if num_nodes is not None:
        data.num_nodes = num_nodes
    return data


class TestGraphormerSPDPreTransform(unittest.TestCase):

    # -- basic correctness --

    def test_linear_chain(self):
        """0 - 1 - 2 - 3: distances should equal hop counts."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        dist = data.spatial_pos.view(4, 4)
        self.assertEqual(dist[0, 0].item(), 0)
        self.assertEqual(dist[0, 1].item(), 1)
        self.assertEqual(dist[0, 2].item(), 2)
        self.assertEqual(dist[0, 3].item(), 3)
        # symmetry
        self.assertEqual(dist[3, 0].item(), 3)

    def test_self_distance_is_zero(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=2))

        dist = data.spatial_pos.view(2, 2)
        self.assertEqual(dist[0, 0].item(), 0)
        self.assertEqual(dist[1, 1].item(), 0)

    def test_complete_graph(self):
        """All pairs in K4 should have distance 1 (except diagonal)."""
        edges_src, edges_dst = [], []
        for i in range(4):
            for j in range(4):
                if i != j:
                    edges_src.append(i)
                    edges_dst.append(j)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        dist = data.spatial_pos.view(4, 4)
        for i in range(4):
            for j in range(4):
                expected = 0 if i == j else 1
                self.assertEqual(dist[i, j].item(), expected)

    # -- cutoff behavior --

    def test_cutoff_clamps_long_distances(self):
        """0 - 1 - 2 - 3 with max_distance=2: dist(0,3) should be unreachable."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=2, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        dist = data.spatial_pos.view(4, 4)
        unreachable = 3  # max_distance + 1
        self.assertEqual(dist[0, 3].item(), unreachable)
        self.assertEqual(dist[0, 2].item(), 2)  # exactly at cutoff

    # -- disconnected components --

    def test_disconnected_graph_marks_unreachable(self):
        """Two disconnected edges: 0-1, 2-3. Cross-component = unreachable."""
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        dist = data.spatial_pos.view(4, 4)
        unreachable = 17
        self.assertEqual(dist[0, 1].item(), 1)
        self.assertEqual(dist[0, 2].item(), unreachable)
        self.assertEqual(dist[0, 3].item(), unreachable)

    def test_isolated_node(self):
        """Node 2 has no edges — unreachable from 0 and 1."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = data.spatial_pos.view(3, 3)
        unreachable = 17
        self.assertEqual(dist[0, 2].item(), unreachable)
        self.assertEqual(dist[2, 2].item(), 0)  # self-distance still 0

    # -- directed graphs --

    def test_directed_asymmetric_distances(self):
        """Directed: 0->1->2. dist(0,2)=2 but dist(2,0)=unreachable."""
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=False)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = data.spatial_pos.view(3, 3)
        unreachable = 17
        self.assertEqual(dist[0, 2].item(), 2)
        self.assertEqual(dist[2, 0].item(), unreachable)

    # -- edge cases --

    def test_no_edges(self):
        """Graph with nodes but no edges — all off-diagonal = unreachable."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = data.spatial_pos.view(3, 3)
        unreachable = 5
        for i in range(3):
            for j in range(3):
                expected = 0 if i == j else unreachable
                self.assertEqual(dist[i, j].item(), expected)

    def test_zero_nodes(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=0))

        self.assertEqual(data.spatial_pos.numel(), 0)
        self.assertEqual(data.spatial_pos_n.item(), 0)

    def test_single_node_no_edges(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=1))

        dist = data.spatial_pos.view(1, 1)
        self.assertEqual(dist[0, 0].item(), 0)

    # -- output format --

    def test_output_is_flat_with_correct_size(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        self.assertEqual(data.spatial_pos.dim(), 1)
        self.assertEqual(data.spatial_pos.shape[0], 9)  # 3*3
        self.assertEqual(data.spatial_pos_n.item(), 3)
        self.assertEqual(data.spatial_pos.dtype, torch.long)

    def test_num_nodes_inferred_from_x(self):
        """When num_nodes is not set, infer from x."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.randn(3, 2)
        data = Data(edge_index=edge_index, x=x)
        # Don't set num_nodes explicitly
        del data.num_nodes

        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(data)

        self.assertEqual(data.spatial_pos.view(3, 3)[0, 1].item(), 1)

    def test_custom_attr_name(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, attr_name="my_spd")
        data = transform(_make_data(edge_index, num_nodes=2))

        self.assertTrue(hasattr(data, "my_spd"))
        self.assertFalse(hasattr(data, "spatial_pos"))

    def test_no_edge_index_returns_data_unchanged(self):
        """Data without edge_index should pass through untouched."""
        data = Data(x=torch.randn(3, 2))
        transform = GraphormerSPDPreTransform(max_distance=4)
        result = transform(data)

        self.assertFalse(hasattr(result, "spatial_pos"))


if __name__ == "__main__":
    unittest.main()
