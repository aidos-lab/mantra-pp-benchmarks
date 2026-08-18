"""Value-level tests for src/transforms.py.

Covers CreateNodeRegressionLabels (fresh sample, cached sample, mask
options, source removal) and TensorizeAttributes (dtype handling,
nested parquet-style object arrays, ragged rejection, missing sources).

Style mirrors tests/test_values.py (plain pytest, small synthetic Data
objects, exact-value asserts).
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.transforms import CreateNodeRegressionLabels, TensorizeAttributes


# ===========================================================================
# 1. CreateNodeRegressionLabels
# ===========================================================================

class TestCreateNodeRegressionLabels:
    def _make_data(self, c2=(-12.0, 4.0, 4.0, 4.0)):
        return Data(
            x=torch.zeros(4, 2),
            c2=np.array(c2, dtype=np.float64),
        )

    def test_creates_column_vector_targets(self):
        data = CreateNodeRegressionLabels(source="c2")(self._make_data())
        assert data.y.shape == (4, 1)
        assert data.y.dtype == torch.float32
        expected = torch.tensor([[-12.0], [4.0], [4.0], [4.0]])
        assert torch.equal(data.y, expected)

    def test_masks_origin_vertex_by_default(self):
        """Vertex 0 is the polytope origin (no divisor) and is masked out."""
        data = CreateNodeRegressionLabels(source="c2")(self._make_data())
        assert data.node_mask.dtype == torch.bool
        assert data.node_mask.tolist() == [False, True, True, True]

    def test_mask_first_false_supervises_all_nodes(self):
        transform = CreateNodeRegressionLabels(source="c2", mask_first=False)
        data = transform(self._make_data())
        assert data.node_mask.tolist() == [True, True, True, True]

    def test_source_attribute_is_consumed(self):
        """The ragged numpy source is deleted so cache collation works."""
        data = CreateNodeRegressionLabels(source="c2")(self._make_data())
        assert "c2" not in data

    def test_cached_sample_passes_through(self):
        """A sample whose pre-transform already baked y/node_mask is untouched."""
        y = torch.tensor([[1.0], [2.0]])
        mask = torch.tensor([False, True])
        cached = Data(x=torch.zeros(2, 2), y=y, node_mask=mask)
        out = CreateNodeRegressionLabels(source="c2")(cached)
        # BaseTransform.__call__ shallow-copies the Data, so identity holds
        # for the tensors, not the container.
        assert out.y is y
        assert out.node_mask is mask

    def test_missing_source_and_missing_cache_raises(self):
        bare = Data(x=torch.zeros(2, 2))
        with pytest.raises(AssertionError):
            CreateNodeRegressionLabels(source="c2")(bare)

    def test_alternative_source_name(self):
        data = Data(x=torch.zeros(3, 2), kappa=np.array([1.0, 2.0, 3.0]))
        out = CreateNodeRegressionLabels(source="kappa")(data)
        assert torch.equal(out.y, torch.tensor([[1.0], [2.0], [3.0]]))
        assert "kappa" not in out

    def test_label_to_index_interface_is_empty_dict(self):
        """Interface compatibility with CreateLabels: no label indexing."""
        assert CreateNodeRegressionLabels(source="c2").label_to_index == {}


# ===========================================================================
# 2. TensorizeAttributes
# ===========================================================================

class TestTensorizeAttributes:
    def test_float64_array_becomes_float32_tensor(self):
        data = Data(c2=np.array([-12.0, 4.0, 8.0], dtype=np.float64))
        out = TensorizeAttributes("c2")(data)
        assert isinstance(out.c2, torch.Tensor)
        assert out.c2.dtype == torch.float32
        assert torch.equal(out.c2, torch.tensor([-12.0, 4.0, 8.0]))

    def test_integer_array_keeps_integer_dtype(self):
        data = Data(kappa=np.array([1, 2, 3], dtype=np.int64))
        out = TensorizeAttributes(["kappa"])(data)
        assert out.kappa.dtype == torch.int64
        assert torch.equal(out.kappa, torch.tensor([1, 2, 3]))

    def test_nested_object_array_is_stacked(self):
        """Parquet list<list<int>> columns arrive as object arrays of rows
        (e.g. the (nnz, 4) COO block of intersection_numbers)."""
        rows = np.empty(2, dtype=object)
        rows[0] = np.array([0, 1, 2, 5], dtype=np.int64)
        rows[1] = np.array([1, 2, 3, -7], dtype=np.int64)
        data = Data(intersection_numbers=rows)
        out = TensorizeAttributes("intersection_numbers")(data)
        assert out.intersection_numbers.shape == (2, 4)
        expected = torch.tensor([[0, 1, 2, 5], [1, 2, 3, -7]])
        assert torch.equal(out.intersection_numbers, expected)

    def test_ragged_rows_raise_value_error(self):
        rows = np.empty(2, dtype=object)
        rows[0] = np.array([1, 2, 3])
        rows[1] = np.array([1, 2, 3, 4])
        data = Data(bad=rows)
        with pytest.raises(ValueError, match="ragged"):
            TensorizeAttributes("bad")(data)

    def test_missing_source_is_skipped(self):
        data = Data(x=torch.zeros(2, 2))
        out = TensorizeAttributes(["c2", "kappa"])(data)
        # BaseTransform.__call__ shallow-copies the Data; the missing
        # sources must simply stay absent.
        assert "c2" not in out and "kappa" not in out
        assert out.x is data.x

    def test_existing_tensor_left_untouched(self):
        t = torch.tensor([1.0, 2.0])
        data = Data(c2=t)
        out = TensorizeAttributes("c2")(data)
        assert out.c2 is t

    def test_empty_object_array_becomes_empty_int_tensor(self):
        empty = np.empty(0, dtype=object)
        data = Data(inter=empty)
        out = TensorizeAttributes("inter")(data)
        assert out.inter.shape == (0,)
        assert out.inter.dtype == torch.int64

    def test_plain_list_input(self):
        """Non-numpy list attributes are tensorized too."""
        data = Data(c2=[1.5, 2.5])
        out = TensorizeAttributes("c2")(data)
        assert out.c2.dtype == torch.float32
        assert torch.equal(out.c2, torch.tensor([1.5, 2.5]))

    def test_string_sources_argument_is_wrapped(self):
        assert TensorizeAttributes("c2").sources == ["c2"]
        assert TensorizeAttributes(["a", "b"]).sources == ["a", "b"]
