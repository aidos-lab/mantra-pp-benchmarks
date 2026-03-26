import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Batch, Data

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from datasets.graphormer_spd import GraphormerSPDPreTransform
from models.Graphormer import Graphormer
from models.ResidualGatedGCN import ResidualGatedGCN


def make_graph_batch() -> Batch:
    graph_a = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
    )
    graph_b = Data(
        x=torch.tensor([[0.5, 0.5], [0.25, 0.75]]),
        edge_index=torch.empty((2, 0), dtype=torch.long),
    )
    return Batch.from_data_list([graph_a, graph_b])


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

        output = model(make_graph_batch())

        self.assertEqual(output.shape, (2, 3))
        self.assertTrue(torch.isfinite(output).all())

    def test_graphormer_spd_respects_undirected_flag(self):
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        device = torch.device("cpu")
        undirected = Graphormer._compute_spd_single(
            edge_index, num_nodes=3, max_dist=4, device=device, undirected=True
        )
        directed = Graphormer._compute_spd_single(
            edge_index, num_nodes=3, max_dist=4, device=device, undirected=False
        )

        self.assertEqual(undirected[2, 0].item(), 2)
        self.assertEqual(directed[2, 0].item(), 5)

    def test_graphormer_uses_precomputed_spd_when_available(self):
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

        data_list = make_graph_batch().to_data_list()
        pre_transform = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        data_list = [pre_transform(data) for data in data_list]
        batch = Batch.from_data_list(data_list)

        with patch.object(
            model,
            "_compute_spd_batch",
            side_effect=AssertionError("SPD should be precomputed"),
        ):
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

        with patch.object(
            model,
            "_compute_spd_batch",
            side_effect=AssertionError("SPD should be skipped"),
        ):
            output = model(make_graph_batch())

        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_residual_gated_gcn_uses_configured_dropout(self):
        model = ResidualGatedGCN(
            hidden_channels=8,
            n_layers=2,
            in_channels=2,
            out_channels=4,
            batch_norm=False,
            dropout=0.25,
        )
        model.train()

        with patch("models.ResidualGatedGCN.F.dropout", wraps=torch.nn.functional.dropout) as mock_dropout:
            _ = model(make_graph_batch())

        _, kwargs = mock_dropout.call_args
        self.assertEqual(kwargs.get("p"), 0.25)

    def test_residual_gated_gcn_forward_handles_edgeless_graphs(self):
        model = ResidualGatedGCN(
            hidden_channels=8,
            n_layers=2,
            in_channels=2,
            out_channels=4,
            batch_norm=False,
        )
        model.eval()

        output = model(make_graph_batch())

        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
