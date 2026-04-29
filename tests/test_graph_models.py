import unittest

import torch
from torch_geometric.data import Batch, Data

from src.utils.graphormer_spd import GraphormerSPDPreTransform
from src.models.Graphormer import Graphormer
from src.models.ResidualGatedGCN import ResidualGatedGCN


def make_graph_batch() -> Batch:
    graph_a = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    graph_b = Data(
        x=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    return graph_a, graph_b


def make_batch_with_spd(max_distance=4) -> Batch:
    """Create a batch with precomputed SPD."""
    graph_a, graph_b = make_graph_batch()
    pre_transform = GraphormerSPDPreTransform(max_distance=max_distance, undirected=True)
    data_list = [pre_transform(graph_a), pre_transform(graph_b)]
    return Batch.from_data_list(data_list)


class GraphModelIntegrationTests(unittest.TestCase):
    def test_graphormer_forward_returns_graph_level_outputs(self):
        model = Graphormer(
            hidden_channels=8,
            n_layers=2,
            in_channels=2,
            out_channels=3,
            num_heads=2,
            dropout=0.0,
            max_degree=4,
            max_distance=4,
        )
        model.eval()

        batch = make_batch_with_spd(max_distance=4)
        output = model(batch)

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_graphormer_uses_precomputed_spd(self):
        model = Graphormer(
            hidden_channels=8,
            n_layers=1,
            in_channels=2,
            out_channels=2,
            num_heads=2,
            dropout=0.0,
            max_distance=4,
        )
        model.eval()

        batch = make_batch_with_spd(max_distance=4)
        output = model(batch)

        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_graphormer_skips_spd_when_spatial_encoding_is_disabled(self):
        model = Graphormer(
            hidden_channels=8,
            n_layers=1,
            in_channels=2,
            out_channels=2,
            num_heads=2,
            dropout=0.0,
            use_spatial_encoding=False,
        )
        model.eval()

        graph_a, graph_b = make_graph_batch()
        batch = Batch.from_data_list([graph_a, graph_b])
        output = model(batch)

        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_graphormer_raises_without_spd_when_spatial_encoding_enabled(self):
        model = Graphormer(
            hidden_channels=8,
            n_layers=1,
            in_channels=2,
            out_channels=2,
            num_heads=2,
            dropout=0.0,
            use_spatial_encoding=True,
        )
        model.eval()

        graph_a, graph_b = make_graph_batch()
        batch = Batch.from_data_list([graph_a, graph_b])

        with self.assertRaises(ValueError):
            model(batch)

    def test_residual_gated_gcn_forward_returns_correct_shape(self):
        model = ResidualGatedGCN(
            hidden_channels=8,
            n_layers=2,
            in_channels=2,
            out_channels=4,
            batch_norm=True,
        )
        model.eval()

        graph_a, graph_b = make_graph_batch()
        batch = Batch.from_data_list([graph_a, graph_b])
        output = model(batch)

        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())

    def test_residual_gated_gcn_forward_handles_edgeless_graphs(self):
        model = ResidualGatedGCN(
            hidden_channels=8,
            n_layers=2,
            in_channels=2,
            out_channels=4,
            batch_norm=False,
        )
        model.eval()

        graph_a, graph_b = make_graph_batch()
        batch = Batch.from_data_list([graph_a, graph_b])
        output = model(batch)

        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
