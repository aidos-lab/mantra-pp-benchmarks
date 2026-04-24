import unittest

import torch
from torch_geometric.data import Data

from src.utils.graphormer_spd import GraphormerSPDPreTransform


def _make_data(edge_index, num_nodes=None, x=None):
    """Helper to build a Data object with optional node features."""
    data = Data(edge_index=edge_index)
    if x is not None:
        data.x = x
    if num_nodes is not None:
        data.num_nodes = num_nodes
    return data


def _dense_spd(data, max_distance, prefix="spd"):
    """Reconstruct an (n, n) distance matrix from sparse SPD attributes."""
    n = data.num_nodes
    unreachable = max_distance + 1
    dist = torch.full((n, n), unreachable, dtype=torch.long)
    if n > 0:
        dist.fill_diagonal_(0)
    ei = getattr(data, f"{prefix}_edge_index")
    ea = getattr(data, f"{prefix}_edge_attr")
    if ei.numel():
        dist[ei[0], ei[1]] = ea
    return dist


class TestGraphormerSPDPreTransform(unittest.TestCase):

    # -- basic correctness --

    def test_linear_chain(self):
        """0 - 1 - 2 - 3: distances should equal hop counts."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        dist = _dense_spd(data, 16)
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

        dist = _dense_spd(data, 16)
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

        dist = _dense_spd(data, 16)
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

        dist = _dense_spd(data, 2)
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

        dist = _dense_spd(data, 16)
        unreachable = 17
        self.assertEqual(dist[0, 1].item(), 1)
        self.assertEqual(dist[0, 2].item(), unreachable)
        self.assertEqual(dist[0, 3].item(), unreachable)

    def test_isolated_node(self):
        """Node 2 has no edges — unreachable from 0 and 1."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = _dense_spd(data, 16)
        unreachable = 17
        self.assertEqual(dist[0, 2].item(), unreachable)
        self.assertEqual(dist[2, 2].item(), 0)  # self-distance still 0

    # -- directed graphs --

    def test_directed_asymmetric_distances(self):
        """Directed: 0->1->2. dist(0,2)=2 but dist(2,0)=unreachable."""
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=False)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = _dense_spd(data, 16)
        unreachable = 17
        self.assertEqual(dist[0, 2].item(), 2)
        self.assertEqual(dist[2, 0].item(), unreachable)

    # -- edge cases --

    def test_no_edges(self):
        """Graph with nodes but no edges — all off-diagonal = unreachable."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        dist = _dense_spd(data, 4)
        unreachable = 5
        for i in range(3):
            for j in range(3):
                expected = 0 if i == j else unreachable
                self.assertEqual(dist[i, j].item(), expected)

    def test_zero_nodes(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=0))

        self.assertEqual(tuple(data.spd_edge_index.shape), (2, 0))
        self.assertEqual(data.spd_edge_attr.numel(), 0)

    def test_single_node_no_edges(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=1))

        dist = _dense_spd(data, 4)
        self.assertEqual(dist[0, 0].item(), 0)

    # -- output format --

    def test_output_is_coo_with_correct_shape(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        # Only the pair 0<->1 is reachable (node 2 is isolated).
        # Expected: 2 ordered pairs (0,1) and (1,0), both at d=1.
        self.assertEqual(data.spd_edge_index.dim(), 2)
        self.assertEqual(data.spd_edge_index.shape[0], 2)
        self.assertEqual(data.spd_edge_index.shape[1], 2)
        self.assertEqual(data.spd_edge_attr.shape[0], 2)
        self.assertEqual(data.spd_edge_index.dtype, torch.long)
        self.assertEqual(data.spd_edge_attr.dtype, torch.long)
        self.assertTrue(torch.equal(data.spd_edge_attr, torch.tensor([1, 1])))

    def test_num_nodes_inferred_from_x(self):
        """When num_nodes is not set, infer from x."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.randn(3, 2)
        data = Data(edge_index=edge_index, x=x)
        # Don't set num_nodes explicitly
        del data.num_nodes

        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(data)

        self.assertEqual(_dense_spd(data, 4)[0, 1].item(), 1)

    def test_custom_attr_prefix(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, attr_prefix="my_spd")
        data = transform(_make_data(edge_index, num_nodes=2))

        self.assertTrue(hasattr(data, "my_spd_edge_index"))
        self.assertTrue(hasattr(data, "my_spd_edge_attr"))
        self.assertFalse(hasattr(data, "spd_edge_index"))
        self.assertFalse(hasattr(data, "spatial_pos"))

    def test_no_edge_index_returns_data_unchanged(self):
        """Data without edge_index should pass through untouched."""
        data = Data(x=torch.randn(3, 2))
        transform = GraphormerSPDPreTransform(max_distance=4)
        result = transform(data)

        self.assertFalse(hasattr(result, "spd_edge_index"))
        self.assertFalse(hasattr(result, "spd_edge_attr"))
        self.assertFalse(hasattr(result, "spatial_pos"))


if __name__ == "__main__":
    unittest.main()
