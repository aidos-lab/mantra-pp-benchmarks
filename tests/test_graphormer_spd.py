"""Unit tests for the dense uint8 GraphormerSPDPreTransform.

Each test is written so the expected SPD values are hand-computable from
the input graph topology and you can verify the numbers by inspection.
"""

import unittest

import networkx as nx
import torch
from torch_geometric.data import Batch, Data

from src.utils.graphormer_spd import GraphormerSPDPreTransform, SPDData


def _make_data(edge_index, num_nodes=None, x=None):
    data = Data(edge_index=edge_index)
    if x is not None:
        data.x = x
    if num_nodes is not None:
        data.num_nodes = num_nodes
    return data


def _spd_matrix(data, attr="spd") -> torch.Tensor:
    """Return the (n, n) long distance matrix from a Data with flat uint8 SPD."""
    n = int(getattr(data, f"{attr}_n").item())
    flat = getattr(data, attr)
    assert flat.dtype == torch.uint8, f"expected uint8, got {flat.dtype}"
    assert flat.numel() == n * n, f"flat size {flat.numel()} != n*n {n * n}"
    return flat.view(n, n).to(torch.long)


# --------------------------------------------------------------------------- #
# Hand-verifiable correctness                                                 #
# --------------------------------------------------------------------------- #


class TestExactDistances(unittest.TestCase):

    def test_linear_chain_n4_full_matrix(self):
        """Chain 0-1-2-3. Full distance matrix is fixed and small."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        expected = torch.tensor(
            [[0, 1, 2, 3],
             [1, 0, 1, 2],
             [2, 1, 0, 1],
             [3, 2, 1, 0]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(_spd_matrix(data), expected))

    def test_cycle_n5_distances_from_node_0(self):
        """C5 (pentagon). Distances from node 0: [0, 1, 2, 2, 1]."""
        edges = []
        for i in range(5):
            edges.append((i, (i + 1) % 5))
            edges.append(((i + 1) % 5, i))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=5))

        D = _spd_matrix(data)
        self.assertTrue(torch.equal(D[0], torch.tensor([0, 1, 2, 2, 1])))
        # symmetric
        self.assertTrue(torch.equal(D, D.t()))

    def test_complete_graph_K4(self):
        edges = [(i, j) for i in range(4) for j in range(4) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        expected = torch.ones((4, 4), dtype=torch.long) - torch.eye(4, dtype=torch.long)
        self.assertTrue(torch.equal(_spd_matrix(data), expected))

    def test_two_disconnected_edges(self):
        """0-1, 2-3. Cross-component pairs encode as max_distance+1."""
        edge_index = torch.tensor(
            [[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        UN = 17
        expected = torch.tensor(
            [[0, 1, UN, UN],
             [1, 0, UN, UN],
             [UN, UN, 0, 1],
             [UN, UN, 1, 0]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(_spd_matrix(data), expected))

    def test_isolated_node(self):
        """Edge 0-1; node 2 isolated."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        UN = 17
        expected = torch.tensor(
            [[0, 1, UN],
             [1, 0, UN],
             [UN, UN, 0]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(_spd_matrix(data), expected))


# --------------------------------------------------------------------------- #
# Cutoff & encoding                                                           #
# --------------------------------------------------------------------------- #


class TestCutoffEncoding(unittest.TestCase):

    def test_cutoff_clamps_above_max_distance(self):
        """Chain 0-1-2-3 with max_distance=2; dist(0,3) actual=3 → unreachable."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=2, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=4))

        UN = 3  # max_distance + 1
        expected = torch.tensor(
            [[0, 1, 2, UN],
             [1, 0, 1, 2],
             [2, 1, 0, 1],
             [UN, 2, 1, 0]],
            dtype=torch.long,
        )
        self.assertTrue(torch.equal(_spd_matrix(data), expected))

    def test_distance_at_cutoff_is_kept(self):
        """Chain 0-1-2 with max_distance=2; dist(0,2) = 2 stays as 2."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=2, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        self.assertEqual(_spd_matrix(data)[0, 2].item(), 2)

    def test_max_distance_254_fits_uint8(self):
        """Boundary: max_distance=254 → unreachable=255, still fits in uint8."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=254, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=2))
        self.assertEqual(data.spd.dtype, torch.uint8)

    def test_max_distance_255_rejected(self):
        with self.assertRaises(ValueError):
            GraphormerSPDPreTransform(max_distance=255)

    def test_max_distance_zero_rejected(self):
        with self.assertRaises(ValueError):
            GraphormerSPDPreTransform(max_distance=0)


# --------------------------------------------------------------------------- #
# Directed                                                                    #
# --------------------------------------------------------------------------- #


class TestDirected(unittest.TestCase):

    def test_directed_chain_asymmetric(self):
        """0→1→2: dist(0,2)=2, dist(2,0)=unreachable."""
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=False)
        data = transform(_make_data(edge_index, num_nodes=3))

        D = _spd_matrix(data)
        UN = 17
        self.assertEqual(D[0, 2].item(), 2)
        self.assertEqual(D[2, 0].item(), UN)
        self.assertEqual(D[1, 0].item(), UN)


# --------------------------------------------------------------------------- #
# Edge cases                                                                  #
# --------------------------------------------------------------------------- #


class TestEdgeCases(unittest.TestCase):

    def test_no_edges(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=3))

        UN = 5
        expected = torch.tensor(
            [[0, UN, UN], [UN, 0, UN], [UN, UN, 0]], dtype=torch.long
        )
        self.assertTrue(torch.equal(_spd_matrix(data), expected))

    def test_zero_nodes(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=0))

        self.assertEqual(data.spd.numel(), 0)
        self.assertEqual(data.spd.dtype, torch.uint8)
        self.assertEqual(int(data.spd_n.item()), 0)
        self.assertIs(type(data), SPDData)

    def test_single_node(self):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=1))

        D = _spd_matrix(data)
        self.assertTrue(torch.equal(D, torch.zeros((1, 1), dtype=torch.long)))

    def test_no_edge_index_passes_through_unchanged(self):
        data = Data(x=torch.randn(3, 2))
        transform = GraphormerSPDPreTransform(max_distance=4)
        out = transform(data)

        self.assertFalse(hasattr(out, "spd"))
        self.assertFalse(hasattr(out, "spd_n"))


# --------------------------------------------------------------------------- #
# Output format                                                               #
# --------------------------------------------------------------------------- #


class TestOutputFormat(unittest.TestCase):

    def test_dtype_is_uint8(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4)
        data = transform(_make_data(edge_index, num_nodes=2))
        self.assertEqual(data.spd.dtype, torch.uint8)

    def test_flat_length_equals_n_squared(self):
        """For n=10, flat must be exactly 100 uint8 bytes — verifiable size."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4)
        data = transform(_make_data(edge_index, num_nodes=10))

        self.assertEqual(data.spd.numel(), 100)
        # uint8 → exactly 100 bytes.
        self.assertEqual(data.spd.element_size() * data.spd.numel(), 100)

    def test_size_attr_is_long_scalar_tensor(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4)
        data = transform(_make_data(edge_index, num_nodes=5))

        self.assertEqual(data.spd_n.dtype, torch.long)
        self.assertEqual(int(data.spd_n.item()), 5)

    def test_class_is_promoted_to_SPDData(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4)
        data = transform(_make_data(edge_index, num_nodes=2))
        self.assertIs(type(data), SPDData)

    def test_custom_attr_name(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, attr_name="my_spd")
        data = transform(_make_data(edge_index, num_nodes=2))

        self.assertTrue(hasattr(data, "my_spd"))
        self.assertTrue(hasattr(data, "my_spd_n"))
        self.assertFalse(hasattr(data, "spd"))

    def test_num_nodes_inferred_from_x_when_missing(self):
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        x = torch.randn(3, 2)
        data = Data(edge_index=edge_index, x=x)
        # Don't set num_nodes explicitly.
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(data)

        # n=3 should be inferred via x.size(0); flat length = 9.
        self.assertEqual(int(data.spd_n.item()), 3)
        self.assertEqual(data.spd.numel(), 9)


# --------------------------------------------------------------------------- #
# Batching                                                                    #
# --------------------------------------------------------------------------- #


class TestBatching(unittest.TestCase):

    def test_two_graph_batch_lengths(self):
        """Triangle (n=3) + chain n=4 → batch.spd len = 9 + 16 = 25; spd_n=[3,4]."""
        e1 = torch.tensor(
            [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
        )
        e2 = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        g1 = transform(_make_data(e1, num_nodes=3))
        g2 = transform(_make_data(e2, num_nodes=4))

        batch = Batch.from_data_list([g1, g2])

        self.assertEqual(batch.spd.numel(), 9 + 16)
        self.assertEqual(batch.spd.dtype, torch.uint8)
        self.assertTrue(torch.equal(batch.spd_n, torch.tensor([3, 4])))

    def test_batched_blocks_match_per_graph(self):
        """After batching, splitting by spd_n² recovers each per-graph matrix."""
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        transform = GraphormerSPDPreTransform(max_distance=16, undirected=True)
        g1 = transform(_make_data(edge_index, num_nodes=4))   # chain n=4

        e2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        g2 = transform(_make_data(e2, num_nodes=2))            # edge n=2

        D1 = _spd_matrix(g1).clone()
        D2 = _spd_matrix(g2).clone()

        batch = Batch.from_data_list([g1, g2])
        sizes_sq = (batch.spd_n * batch.spd_n).tolist()
        chunks = torch.split(batch.spd, sizes_sq)
        ns = batch.spd_n.tolist()

        rec1 = chunks[0].view(ns[0], ns[0]).to(torch.long)
        rec2 = chunks[1].view(ns[1], ns[1]).to(torch.long)

        self.assertTrue(torch.equal(rec1, D1))
        self.assertTrue(torch.equal(rec2, D2))


# --------------------------------------------------------------------------- #
# Parity with networkx (random graph)                                         #
# --------------------------------------------------------------------------- #


class TestNetworkXParity(unittest.TestCase):

    def test_random_graph_matches_networkx(self):
        """Independent reference: scipy-based transform must agree with nx BFS."""
        torch.manual_seed(0)
        # Build a deterministic small random graph.
        g_nx = nx.gnp_random_graph(20, 0.2, seed=0)
        edges = list(g_nx.edges())
        ei = []
        for u, v in edges:
            ei.append((u, v))
            ei.append((v, u))
        edge_index = torch.tensor(ei, dtype=torch.long).t().contiguous()
        n = g_nx.number_of_nodes()

        max_distance = 10
        transform = GraphormerSPDPreTransform(
            max_distance=max_distance, undirected=True
        )
        data = transform(_make_data(edge_index, num_nodes=n))
        D = _spd_matrix(data)

        unreachable = max_distance + 1
        for src, lengths in nx.all_pairs_shortest_path_length(g_nx):
            for tgt in range(n):
                d = lengths.get(tgt, None)
                if d is None or d > max_distance:
                    self.assertEqual(D[src, tgt].item(), unreachable)
                else:
                    self.assertEqual(D[src, tgt].item(), d)


# --------------------------------------------------------------------------- #
# Memory footprint                                                            #
# --------------------------------------------------------------------------- #


class TestMemoryFootprint(unittest.TestCase):

    def test_dense_uint8_size_is_n_squared_bytes(self):
        """For a triangulated-Hasse-style graph at n=200, SPD costs exactly 40000 bytes."""
        n = 200
        # K_n: every pair at distance 1 — worst-case dense SPD.
        rows, cols = [], []
        for i in range(n):
            for j in range(n):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data = transform(_make_data(edge_index, num_nodes=n))

        bytes_used = data.spd.element_size() * data.spd.numel()
        self.assertEqual(bytes_used, n * n)
        # Sanity: same n² regardless of edge density.
        self.assertEqual(bytes_used, 40000)


if __name__ == "__main__":
    unittest.main()
