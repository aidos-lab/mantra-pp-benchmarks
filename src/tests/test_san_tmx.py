"""Tests for src/models/san_tmx.py — SANTMX, SANTMXLayer, SANConv."""

import pytest
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import Data

from src.models.san_tmx import SANConv, SANTMXLayer, SANTMX
from src.utils.data_loading import collate_simplicial


# ---------------------------------------------------------------------------
# Helpers — build minimal simplicial complex data
# ---------------------------------------------------------------------------

def _make_simplicial_complex(n_nodes=4, n_edges=5, n_triangles=2, feat_dim=8):
    """Create a single simplicial complex Data object with Laplacians.

    Builds a small triangulation (tetrahedron-like) with:
    - x_0: node features      (n_nodes, feat_dim)
    - x_1: edge features      (n_edges, feat_dim)
    - x_2: triangle features  (n_triangles, feat_dim)
    - up_laplacian_1:   (n_edges, n_edges) sparse
    - down_laplacian_1: (n_edges, n_edges) sparse
    - dimension: 2
    """
    data = Data()
    data.x_0 = torch.randn(n_nodes, feat_dim)
    data.x_1 = torch.randn(n_edges, feat_dim)
    data.x_2 = torch.randn(n_triangles, feat_dim)
    data.dimension = torch.tensor(2)
    data.y = torch.tensor([0])

    # Build a simple symmetric up-Laplacian (n_edges x n_edges)
    # Use a tridiagonal-like pattern for realism
    rows, cols, vals = [], [], []
    for i in range(n_edges):
        rows.append(i)
        cols.append(i)
        vals.append(2.0)
        if i + 1 < n_edges:
            rows.extend([i, i + 1])
            cols.extend([i + 1, i])
            vals.extend([-1.0, -1.0])
    up_lap = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]),
        torch.tensor(vals),
        (n_edges, n_edges),
    ).coalesce()
    data.up_laplacian_1 = up_lap

    # Build a simple symmetric down-Laplacian
    rows2, cols2, vals2 = [], [], []
    for i in range(n_edges):
        rows2.append(i)
        cols2.append(i)
        vals2.append(1.0)
        if i + 2 < n_edges:
            rows2.extend([i, i + 2])
            cols2.extend([i + 2, i])
            vals2.extend([-0.5, -0.5])
    down_lap = torch.sparse_coo_tensor(
        torch.tensor([rows2, cols2]),
        torch.tensor(vals2),
        (n_edges, n_edges),
    ).coalesce()
    data.down_laplacian_1 = down_lap

    return data


def _make_batch(num_graphs=2, **kwargs):
    """Create a collated simplicial batch using collate_simplicial."""
    sc_list = [_make_simplicial_complex(**kwargs) for _ in range(num_graphs)]
    return collate_simplicial(sc_list)


def _make_sparse_tensor(n, density=0.3):
    """Create a random symmetric SparseTensor of size (n, n)."""
    nnz = max(n, int(n * n * density))
    row = torch.randint(0, n, (nnz,))
    col = torch.randint(0, n, (nnz,))
    # Make symmetric
    row_sym = torch.cat([row, col])
    col_sym = torch.cat([col, row])
    val = torch.randn(row_sym.size(0))
    return SparseTensor(row=row_sym, col=col_sym, value=val, sparse_sizes=(n, n))


# ---------------------------------------------------------------------------
# SANConv
# ---------------------------------------------------------------------------

class TestSANConv:
    def test_output_shape(self):
        in_ch, out_ch, n_filters = 8, 16, 2
        n_cells = 10
        conv = SANConv(in_ch, out_ch, n_filters)
        conv.eval()

        x = torch.randn(n_cells, in_ch)
        neighborhood = _make_sparse_tensor(n_cells)

        out = conv(x, neighborhood)
        assert out.shape == (n_cells, out_ch)

    def test_output_is_finite(self):
        conv = SANConv(4, 8, 2)
        conv.eval()
        x = torch.randn(6, 4)
        nb = _make_sparse_tensor(6)
        out = conv(x, nb)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        conv = SANConv(4, 8, 3)
        x = torch.randn(5, 4, requires_grad=True)
        nb = _make_sparse_tensor(5)
        out = conv(x, nb)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for p in conv.parameters():
            assert p.grad is not None

    def test_single_filter(self):
        conv = SANConv(4, 4, n_filters=1)
        conv.eval()
        x = torch.randn(5, 4)
        nb = _make_sparse_tensor(5)
        out = conv(x, nb)
        assert out.shape == (5, 4)
        assert torch.isfinite(out).all()

    def test_different_in_out_channels(self):
        conv = SANConv(3, 7, 2)
        conv.eval()
        x = torch.randn(4, 3)
        nb = _make_sparse_tensor(4)
        out = conv(x, nb)
        assert out.shape == (4, 7)


# ---------------------------------------------------------------------------
# SANTMXLayer
# ---------------------------------------------------------------------------

class TestSANTMXLayer:
    def test_output_shape(self):
        layer = SANTMXLayer(8, 16, n_filters=2)
        layer.eval()
        n = 10
        x = torch.randn(n, 8)
        L_down = _make_sparse_tensor(n)
        L_up = _make_sparse_tensor(n)
        P = _make_sparse_tensor(n)
        out = layer(x, L_down, L_up, P)
        assert out.shape == (n, 16)

    def test_output_is_finite(self):
        layer = SANTMXLayer(4, 8, n_filters=2)
        layer.eval()
        n = 6
        x = torch.randn(n, 4)
        L_down = _make_sparse_tensor(n)
        L_up = _make_sparse_tensor(n)
        P = _make_sparse_tensor(n)
        out = layer(x, L_down, L_up, P)
        assert torch.isfinite(out).all()

    def test_gradients_flow_through_all_convolutions(self):
        layer = SANTMXLayer(4, 8, n_filters=2)
        n = 5
        x = torch.randn(n, 4, requires_grad=True)
        L_down = _make_sparse_tensor(n)
        L_up = _make_sparse_tensor(n)
        P = _make_sparse_tensor(n)
        out = layer(x, L_down, L_up, P)
        out.sum().backward()

        assert x.grad is not None
        # All three conv modules should have gradients
        for name in ['conv_down', 'conv_up', 'conv_harmonic']:
            conv = getattr(layer, name)
            for pname, p in conv.named_parameters():
                assert p.grad is not None, f"No gradient for {name}.{pname}"


# ---------------------------------------------------------------------------
# SANTMX (full model)
# ---------------------------------------------------------------------------

class TestSANTMX:
    def test_init_requires_more_than_2_layers(self):
        with pytest.raises(AssertionError):
            SANTMX(in_channels=8, hidden_channels=16, out_channels=3, n_layers=2)

    def test_forward_output_shape(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=5,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.eval()
        batch = _make_batch(num_graphs=3, feat_dim=8)
        out = model(batch)
        assert out.shape == (3, 5)

    def test_forward_output_is_finite(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.eval()
        batch = _make_batch(num_graphs=2, feat_dim=8)
        out = model(batch)
        assert torch.isfinite(out).all()

    def test_forward_single_graph(self):
        model = SANTMX(
            in_channels=4, hidden_channels=8, out_channels=2,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.eval()
        batch = _make_batch(num_graphs=1, feat_dim=4)
        out = model(batch)
        assert out.shape == (1, 2)
        assert torch.isfinite(out).all()

    def test_forward_larger_batch(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=4,
            n_layers=4, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.eval()
        batch = _make_batch(num_graphs=8, feat_dim=8)
        out = model(batch)
        assert out.shape == (8, 4)
        assert torch.isfinite(out).all()

    def test_gradients_flow_end_to_end(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        batch = _make_batch(num_graphs=2, feat_dim=8)
        out = model(batch)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_training_step_reduces_loss(self):
        """Verify that a few training steps reduce the loss (sanity check)."""
        torch.manual_seed(42)
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        batch = _make_batch(num_graphs=4, feat_dim=8)
        target = torch.randint(0, 3, (4,))

        initial_loss = None
        for step in range(10):
            optimizer.zero_grad()
            out = model(batch)
            loss = torch.nn.functional.cross_entropy(out, target)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        assert loss.item() < initial_loss, "Loss did not decrease after 10 steps"

    def test_different_kappa_values(self):
        for kappa in [1, 2, 3]:
            model = SANTMX(
                in_channels=8, hidden_channels=16, out_channels=3,
                n_layers=3, kappa=kappa, epsilon=0.1, k_proj=3,
            )
            model.eval()
            batch = _make_batch(num_graphs=2, feat_dim=8)
            out = model(batch)
            assert out.shape == (2, 3)
            assert torch.isfinite(out).all()

    def test_different_graph_sizes_in_batch(self):
        """Graphs in the same batch can have different numbers of simplices."""
        sc1 = _make_simplicial_complex(n_nodes=3, n_edges=4, n_triangles=1, feat_dim=8)
        sc2 = _make_simplicial_complex(n_nodes=5, n_edges=7, n_triangles=3, feat_dim=8)
        batch = collate_simplicial([sc1, sc2])

        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        model.eval()
        out = model(batch)
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_reconstruct_produces_correct_sparse_sizes(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        batch = _make_batch(num_graphs=2, n_edges=5, feat_dim=8)
        sp = model._reconstruct(batch, "up_laplacian_1")
        total_edges = batch['x_1'].size(0)
        assert sp.sparse_sizes() == (total_edges, total_edges)

    def test_harmonic_projector_shape(self):
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        batch = _make_batch(num_graphs=3, n_edges=5, feat_dim=8)
        up_lap = model._reconstruct(batch, "up_laplacian_1")
        down_lap = model._reconstruct(batch, "down_laplacian_1")
        hodge = up_lap + down_lap
        batch_1 = batch['batch_1']

        P = model._harmonic_projector(hodge, batch_1)
        total_edges = batch['x_1'].size(0)
        assert P.sparse_sizes() == (total_edges, total_edges)

    def test_harmonic_projector_block_diagonal(self):
        """P should be block-diagonal — no cross-graph entries."""
        model = SANTMX(
            in_channels=4, hidden_channels=8, out_channels=2,
            n_layers=3, kappa=2, epsilon=0.1, k_proj=3,
        )
        batch = _make_batch(num_graphs=2, n_edges=5, feat_dim=4)
        up_lap = model._reconstruct(batch, "up_laplacian_1")
        down_lap = model._reconstruct(batch, "down_laplacian_1")
        hodge = up_lap + down_lap
        batch_1 = batch['batch_1']

        P = model._harmonic_projector(hodge, batch_1)
        row, col, _ = P.coo()

        # For each entry in P, row and col should belong to the same graph
        graph_of_row = batch_1[row]
        graph_of_col = batch_1[col]
        assert torch.equal(graph_of_row, graph_of_col), "P has cross-graph entries"

    def test_unused_kwargs_accepted(self):
        """Config compatibility: alpha_leaky_relu, dropout, attention should be accepted."""
        model = SANTMX(
            in_channels=8, hidden_channels=16, out_channels=3,
            n_layers=3, alpha_leaky_relu=0.01, dropout=0.5, attention=False,
        )
        assert model is not None
