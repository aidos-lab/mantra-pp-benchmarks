"""Tests for the node-level task path: masked supervision in the
training helpers, the node-level heads of the models, and the
task_level wiring of main.py."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import scatter

import main
from src.metrics.regression import RoundedAccuracy
from src.models.GCN import GCN
from src.models.Graphormer import Graphormer
from src.models.MLP import MLP
from src.models.ResidualGatedGCN import ResidualGatedGCN
from src.models.cwn import CWN
from src.models.sccnn import SCCNN
from src.utils.graphormer_spd import GraphormerSPDPreTransform
from src.utils.train_utils import (
    select_supervised,
    test as _test_entrypoint,
    train,
    validate,
)



def _cycle(n):
    edges = [[i, (i + 1) % n] for i in range(n)]
    edges += [[j, i] for i, j in edges]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _node_graph(n, seed):
    """Graph on `n` nodes with one target per node; node 0 is masked."""
    torch.manual_seed(seed)
    return Data(
        x=torch.randn(n, 2),
        edge_index=_cycle(n),
        y=torch.arange(n, dtype=torch.float).view(-1, 1) + 10 * seed,
        node_mask=torch.tensor([False] + [True] * (n - 1)),
    )


def _graphs():
    return [_node_graph(3, seed=1), _node_graph(2, seed=2)]


def _batch(graphs=None, spd=False):
    graphs = _graphs() if graphs is None else graphs
    if spd:
        pre = GraphormerSPDPreTransform(max_distance=4, undirected=True)
        graphs = [pre(g) for g in graphs]
    return Batch.from_data_list(graphs)


def _gcn(task_level, out_channels=1):
    torch.manual_seed(0)
    return GCN(in_channels=2, hidden_channels=8, out_channels=out_channels,
               n_layers=1, residuals=False, batch_norm=False, dropout=0.0,
               task_level=task_level)


def _rgg(task_level, out_channels=1):
    torch.manual_seed(0)
    return ResidualGatedGCN(hidden_channels=8, n_layers=1, in_channels=2,
                            out_channels=out_channels, batch_norm=False,
                            dropout=0.0, task_level=task_level)


def _mlp(task_level, out_channels=1):
    torch.manual_seed(0)
    return MLP(2, 8, out_channels, n_layers=1, dropout=0.0, batch_norm=False,
               residuals=False, task_level=task_level)


def _graphormer(task_level, out_channels=1):
    torch.manual_seed(0)
    return Graphormer(hidden_channels=8, n_layers=1, in_channels=2,
                      out_channels=out_channels, num_heads=2, dropout=0.0,
                      max_degree=4, max_distance=4, task_level=task_level)


GRAPH_MODELS = [
    pytest.param(_gcn, False, id="gcn"),
    pytest.param(_rgg, False, id="residual_gated_gcn"),
    pytest.param(_mlp, False, id="mlp"),
    pytest.param(_graphormer, True, id="graphormer"),
]


def _cwn_batch(seed=0):
    """Two simplicial complexes (3 + 4 nodes), as in tests/test_values.py."""
    torch.manual_seed(seed)
    batch = Data()
    batch.x_0 = torch.randn(7, 3)
    batch.x_1 = torch.randn(8, 3)
    batch.x_2 = torch.randn(3, 3)
    batch.batch_0 = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch.batch_1 = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])
    batch.batch_2 = torch.tensor([0, 0, 1])
    batch.adjacency_1_idx = torch.tensor([[0, 1], [1, 0]])
    batch.adjacency_1_val = torch.tensor([1.0, 1.0])
    batch.incidence_1_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_1_val = torch.tensor([1.0, -1.0])
    batch.incidence_2_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_2_val = torch.tensor([1.0, -1.0])
    return batch


def _sccnn_batch(seed=0):
    """Two simplicial complexes (3 + 4 nodes), as in tests/test_values.py."""
    torch.manual_seed(seed)
    batch = Data()
    batch.x_0 = torch.randn(7, 3)
    batch.x_1 = torch.randn(7, 3)
    batch.x_2 = torch.randn(3, 3)
    batch.batch_0 = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    batch.batch_1 = torch.tensor([0, 0, 0, 0, 1, 1, 1])
    batch.batch_2 = torch.tensor([0, 0, 1])
    for name in ("hodge_laplacian_0", "down_laplacian_1", "up_laplacian_1",
                 "down_laplacian_2"):
        batch[f"{name}_idx"] = torch.tensor([[0, 1], [1, 0]])
        batch[f"{name}_val"] = torch.tensor([1.0, 1.0])
    batch.incidence_1_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_1_val = torch.tensor([1.0, -1.0])
    batch.incidence_2_idx = torch.tensor([[0, 1], [0, 1]])
    batch.incidence_2_val = torch.tensor([1.0, -1.0])
    return batch


def _cwn(task_level, out_channels=1):
    torch.manual_seed(7)
    return CWN(in_channels=[3, 3, 3], hidden_channels=[4, 4, 4], n_layers=1,
               out_channels=out_channels, task_level=task_level)


def _sccnn(task_level, out_channels=1):
    torch.manual_seed(7)
    return SCCNN(in_channels=[3, 3, 3], hidden_channels=[4, 4, 4],
                 out_channels=out_channels, conv_order=1, sc_order=2,
                 n_layers=2, task_level=task_level)


class _FakeRun:
    name = "test-run"
    id = "test-run"

    def __init__(self):
        self.logged = []

    def log(self, payload):
        self.logged.append(payload)



class TestSelectSupervised:
    def test_masked_rows_are_selected(self):
        batch = Data(
            y=torch.tensor([[10.0], [1.0], [2.0], [3.0]]),
            node_mask=torch.tensor([False, True, True, False]),
        )
        y_hat = torch.tensor([[0.1], [0.2], [0.3], [0.4]])

        out_hat, out_y = select_supervised(y_hat, batch)

        assert torch.equal(out_hat, torch.tensor([[0.2], [0.3]]))
        assert torch.equal(out_y, torch.tensor([[1.0], [2.0]]))

    def test_graph_level_batch_passes_through(self):
        y = torch.tensor([[1.0], [2.0]])
        y_hat = torch.tensor([[0.5], [0.6]])

        out_hat, out_y = select_supervised(y_hat, Data(y=y))

        assert out_hat is y_hat
        assert out_y is y

    def test_graph_level_predictions_on_node_targets_raise(self):
        """A graph-level model on a node-level task must fail loudly
        instead of broadcasting the loss."""
        batch = Data(
            y=torch.zeros(5, 1), node_mask=torch.ones(5, dtype=torch.bool)
        )
        with pytest.raises(ValueError, match="node_mask"):
            select_supervised(torch.zeros(2, 1), batch)



class TestGraphModelNodeHeads:
    @pytest.mark.parametrize(("build", "spd"), GRAPH_MODELS)
    def test_node_level_emits_one_row_per_node(self, build, spd):
        model = build("node", out_channels=3)
        model.eval()
        out = model(_batch(spd=spd))
        assert out.shape == (5, 3)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(("build", "spd"), GRAPH_MODELS)
    def test_graph_output_is_mean_of_node_outputs(self, build, spd):
        """With mean pooling the readout is linear, so pooling before or
        after the final layer agrees: the graph-level prediction equals
        the per-graph mean of the node-level predictions."""
        node_model = build("node", out_channels=2)
        graph_model = build("graph", out_channels=2)
        graph_model.load_state_dict(node_model.state_dict())
        node_model.eval()
        graph_model.eval()

        batch = _batch(spd=spd)
        node_out = node_model(batch)
        graph_out = graph_model(batch)

        pooled = scatter(node_out, batch.batch, dim=0, reduce="mean")
        assert torch.allclose(graph_out, pooled, atol=1e-5)

    def test_graphormer_node_outputs_follow_the_batch_order(self):
        """Unpadding the dense batch must give the nodes in the order of
        `batch.x`: the batched outputs equal the per-graph outputs."""
        model = _graphormer("node", out_channels=2)
        model.eval()

        graphs = _graphs()
        batched = model(_batch(graphs, spd=True))
        single = torch.cat([model(_batch([g], spd=True)) for g in graphs])

        assert batched.shape == (5, 2)
        assert torch.allclose(batched, single, atol=1e-5)



class TestCWNNodeHead:
    def test_graph_output_is_scatter_add_of_node_output(self):
        """The graph path is exactly scatter_add over the node path."""
        model = _cwn("node", out_channels=2)
        model.eval()

        node_out = model(_cwn_batch(seed=3))
        assert node_out.shape == (7, 2)

        model.task_level = "graph"
        graph_out = model(_cwn_batch(seed=3))
        assert graph_out.shape == (2, 2)

        pooled = scatter(node_out, torch.tensor([0, 0, 0, 1, 1, 1, 1]),
                         dim=0, reduce="sum")
        assert torch.allclose(graph_out, pooled, atol=1e-6)

    def test_node_head_gradient_flow(self):
        model = _cwn("node")
        out = model(_cwn_batch(seed=1))
        nn.MSELoss()(out, torch.zeros_like(out)).backward()
        assert model.proj_0.weight.grad is not None
        assert model.proj_0.weight.grad.abs().sum() > 0


class TestSCCNNNodeHead:
    def test_node_output_shape(self):
        model = _sccnn("node", out_channels=3)
        model.eval()
        out = model(_sccnn_batch(seed=3))
        assert out.shape == (7, 3)
        assert torch.isfinite(out).all()

    def test_node_head_is_linear_on_rank0(self):
        """Node mode swaps the three SumReadouts for one Linear head."""
        model = _sccnn("node", out_channels=3)
        assert isinstance(model.node_readout, nn.Linear)
        assert model.node_readout.in_features == 4
        assert model.node_readout.out_features == 3
        assert not hasattr(model, "readout_0")

        graph_model = _sccnn("graph", out_channels=3)
        assert not hasattr(graph_model, "node_readout")
        assert hasattr(graph_model, "readout_0")

    def test_node_head_gradient_flow(self):
        model = _sccnn("node")
        out = model(_sccnn_batch(seed=1))
        nn.MSELoss()(out, torch.zeros_like(out)).backward()
        assert model.node_readout.weight.grad.abs().sum() > 0



class TestTrainingHelpersNodeLevel:
    cfg = SimpleNamespace(trainer=SimpleNamespace(device="cpu"))

    def _expected(self, model, batch):
        mask = batch.node_mask
        y_hat = model(batch)[mask]
        y = batch.y[mask]
        return nn.MSELoss()(y_hat, y).item(), (y_hat - y).abs().mean().item()

    def test_train_loss_and_metric_use_supervised_nodes_only(self):
        import torchmetrics

        model = _gcn("node")
        reference = copy.deepcopy(model)
        batch = _batch()
        expected_loss, expected_mae = self._expected(reference, batch)

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        metrics = train(
            model, optimizer, [batch], nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())], self.cfg,
        )

        assert metrics["train/loss"] == pytest.approx(expected_loss, rel=1e-5)
        assert metrics["train/mae"] == pytest.approx(expected_mae, rel=1e-5)

    def test_validate_uses_supervised_nodes_only(self):
        import torchmetrics

        model = _gcn("node")
        model.eval()
        batch = _batch()
        expected_loss, expected_mae = self._expected(model, batch)

        metrics = validate(
            model, [batch], nn.MSELoss(),
            [("mae", torchmetrics.MeanAbsoluteError())], self.cfg,
        )

        assert metrics["val/loss"] == pytest.approx(expected_loss, rel=1e-5)
        assert metrics["val/mae"] == pytest.approx(expected_mae, rel=1e-5)

    def test_regression_test_excludes_masked_rows(self):
        """A model echoing the targets is perfect on the supervised
        rows; the masked row (99.0) must not count."""

        class Echo(nn.Module):
            def forward(self, batch):
                return batch.y + 0.4 * (batch.y == 99.0)

        run = _FakeRun()
        cfg = SimpleNamespace(
            task=SimpleNamespace(task_type="regression"),
            trainer=SimpleNamespace(device="cpu"),
        )
        batches = [
            Data(y=torch.tensor([[2.0], [4.0]])),
            Data(
                y=torch.tensor([[99.0], [6.0], [8.0]]),
                node_mask=torch.tensor([False, True, True]),
            ),
        ]
        _test_entrypoint(
            Echo(), batches, [("rounded_accuracy", RoundedAccuracy())], {},
            run, cfg, epoch=0,
        )
        assert run.logged[0]["test/rounded_accuracy"] == pytest.approx(1.0)

    def test_classification_test_names_classes_from_the_mapping(
        self, monkeypatch
    ):
        """Node-level classification batches carry no per-sample
        `label`; the confusion matrix is built from label_to_index and
        the masked nodes are left out."""
        captured = {}

        def fake_confusion_matrix(**kwargs):
            captured.update(kwargs)
            return None

        monkeypatch.setattr(
            "src.utils.train_utils.wandb.plot.confusion_matrix",
            fake_confusion_matrix,
        )

        class Logits(nn.Module):
            def forward(self, batch):
                return nn.functional.one_hot(batch.y, 3).float()

        batch = Data(
            y=torch.tensor([2, 0, 1, 1]),
            node_mask=torch.tensor([False, True, True, True]),
        )
        run = _FakeRun()
        cfg = SimpleNamespace(
            task=SimpleNamespace(task_type="classification"),
            trainer=SimpleNamespace(device="cpu"),
        )

        _test_entrypoint(
            Logits(), [batch], [], {-20: 0, 2: 1, 4: 2}, run, cfg, epoch=0
        )

        assert captured["class_names"] == ["-20", "2", "4"]
        assert captured["y_true"] == [0, 1, 1]
        assert captured["preds"] == [0, 1, 1]


class TestCollateSimplicialNodeLevel:
    def test_node_targets_and_mask_are_concatenated(self):
        """The simplicial collate must carry the per-vertex `y` and
        `node_mask` through like `x_0`, aligned with `batch_0`."""
        from src.utils.data_loading import collate_simplicial

        def complex_(n_vertices, offset):
            n_edges = n_vertices
            incidence = torch.sparse_coo_tensor(
                torch.tensor([[0, 1], [0, 0]]), torch.tensor([1.0, -1.0]),
                (n_vertices, n_edges),
            )
            return Data(
                x_0=torch.randn(n_vertices, 2),
                x_1=torch.randn(n_edges, 2),
                incidence_1=incidence,
                dimension=torch.tensor(1),
                y=(torch.arange(n_vertices, dtype=torch.float) + offset)
                .view(-1, 1),
                node_mask=torch.tensor([False] + [True] * (n_vertices - 1)),
            )

        batch = collate_simplicial([complex_(3, 10), complex_(4, 20)])

        assert batch.y.shape == (7, 1)
        assert batch.y.flatten().tolist() == [10, 11, 12, 20, 21, 22, 23]
        assert batch.node_mask.tolist() == [False, True, True] + [False] + [True] * 3
        assert batch.batch_0.tolist() == [0, 0, 0, 1, 1, 1, 1]
        assert batch.x_0.shape == (7, 2)



def _create_model_cfg(model, task_level):
    return SimpleNamespace(
        model=model,
        task=SimpleNamespace(task_type="regression", task_level=task_level),
        dataset=SimpleNamespace(),
        trainer=SimpleNamespace(device="cpu"),
        compile=False,
    )


class TestCreateModelTaskLevel:
    @pytest.fixture(autouse=True)
    def _fake_instantiate(self, monkeypatch):
        def fake(model_cfg, **kwargs):
            return nn.Linear(1, model_cfg.out_channels)

        monkeypatch.setattr("main.hydra.utils.instantiate", fake)

    def test_task_level_is_set_from_the_task(self):
        cfg = _create_model_cfg(
            SimpleNamespace(in_channels=1, out_channels=1, task_level="graph"),
            "node",
        )
        main.create_model(cfg, num_features=1, label_to_index={})
        assert cfg.model.task_level == "node"
        assert cfg.model.out_channels == 1

    def test_graph_level_needs_no_switch(self):
        cfg = _create_model_cfg(
            SimpleNamespace(in_channels=1, out_channels=1), "graph"
        )
        main.create_model(cfg, num_features=1, label_to_index={})
        assert not hasattr(cfg.model, "task_level")

    def test_model_without_switch_rejects_node_level(self):
        cfg = _create_model_cfg(
            SimpleNamespace(in_channels=1, out_channels=1), "node"
        )
        with pytest.raises(ValueError, match="task_level"):
            main.create_model(cfg, num_features=1, label_to_index={})

    def test_unknown_task_level_rejected(self):
        """A typo must not silently select the node-level head."""
        cfg = _create_model_cfg(
            SimpleNamespace(in_channels=1, out_channels=1, task_level="graph"),
            "edge",
        )
        with pytest.raises(ValueError, match="task_level"):
            main.create_model(cfg, num_features=1, label_to_index={})


class TestCheckNodeLevelSample:
    node_cfg = SimpleNamespace(task=SimpleNamespace(task_level="node"))
    graph_cfg = SimpleNamespace(task=SimpleNamespace(task_level="graph"))

    def test_vertex_nodes_pass(self):
        sample = Data(x=torch.zeros(4, 2), edge_index=_cycle(4),
                      y=torch.zeros(4, 1))
        main.check_node_level_sample(self.node_cfg, sample)

    def test_simplicial_sample_counts_0_cells(self):
        sample = Data(x_0=torch.zeros(4, 2), x_1=torch.zeros(6, 2),
                      y=torch.zeros(4, 1))
        main.check_node_level_sample(self.node_cfg, sample)

    def test_mismatch_names_the_representation(self):
        # Dual graph of a 4-vertex triangulation: nodes are top simplices.
        sample = Data(x=torch.zeros(2, 2), edge_index=_cycle(2),
                      y=torch.zeros(4, 1))
        with pytest.raises(ValueError, match="one_skeleton"):
            main.check_node_level_sample(self.node_cfg, sample)

    def test_graph_level_is_not_checked(self):
        sample = Data(x=torch.zeros(2, 2), edge_index=_cycle(2),
                      y=torch.zeros(1, 1))
        main.check_node_level_sample(self.graph_cfg, sample)


try:
    from mantra.transforms import (
        AttributeToNodeClassTransform,
        AttributeToNodeRegressionTransform,
    )
except ImportError:  # mantra without aidos-lab/mantra#96
    AttributeToNodeClassTransform = AttributeToNodeRegressionTransform = None


@pytest.mark.skipif(
    AttributeToNodeRegressionTransform is None,
    reason="requires mantra with the node-level task transforms (#96)",
)
class TestTaskToTransformNodeLevel:
    @staticmethod
    def _cfg(task_type, mask_first=True):
        return SimpleNamespace(
            task=SimpleNamespace(
                name="ChernNode", task_type=task_type, task_level="node",
                mask_first=mask_first, target="c2",
            ),
            dataset=SimpleNamespace(dimension=4),
        )

    def test_node_regression(self):
        transform = main.task_to_transform(self._cfg("regression"))
        assert isinstance(transform, AttributeToNodeRegressionTransform)
        assert transform.source == "c2"
        assert transform.mask_first is True

        data = transform(Data(c2=[-20, 2, 4]))
        assert data.y.flatten().tolist() == [-20.0, 2.0, 4.0]
        assert data.node_mask.tolist() == [False, True, True]

    def test_node_classification(self):
        mapping = {-20: 0, 2: 1, 4: 2}
        transform = main.task_to_transform(
            self._cfg("classification", mask_first=False), mapping=mapping
        )
        assert isinstance(transform, AttributeToNodeClassTransform)
        assert transform.mask_first is False

        data = transform(Data(c2=[4, 2, -20]))
        assert data.y.tolist() == [2, 1, 0]
        assert data.node_mask.tolist() == [True, True, True]

    def test_node_classification_needs_a_mapping(self):
        with pytest.raises(ValueError, match="mapping"):
            main.task_to_transform(self._cfg("classification"))


def test_node_level_transforms_are_imported_lazily():
    """main.py must import against a mantra without #96; the node-level
    transforms are only resolved for node-level tasks."""
    assert not hasattr(main, "AttributeToNodeRegressionTransform")
