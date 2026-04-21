"""Tests for the barycentric subdivision implementation."""

import pytest

from src.utils.barycentric_utils import (
    barycentric_subdivision_raw,
    subdivide_dataset_json,
)


class TestBarycentricSubdivisionRaw:
    """Tests for barycentric_subdivision_raw on individual triangulations."""

    def test_single_triangle(self):
        """A single triangle [1,2,3] should produce 6 triangles and 7 vertices."""
        tri = [[1, 2, 3]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 3 vertices + 3 edges + 1 face = 7 new vertices
        assert n_v == 7
        # 3! = 6 new triangles
        assert len(new_tri) == 6

    def test_single_tetrahedron(self):
        """A single tetrahedron [1,2,3,4] should produce 24 tetrahedra and 15 vertices."""
        tri = [[1, 2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 4 vertices + 6 edges + 4 faces + 1 tetrahedron = 15 new vertices
        assert n_v == 15
        # 4! = 24 new tetrahedra
        assert len(new_tri) == 24

    def test_two_triangles_shared_edge(self):
        """Two triangles sharing an edge should share sub-simplex vertices."""
        tri = [[1, 2, 3], [2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)

        # Vertices: {1}, {2}, {3}, {4} = 4
        # Edges: {1,2}, {1,3}, {2,3}, {2,4}, {3,4} = 5
        # Faces: {1,2,3}, {2,3,4} = 2
        # Total = 11
        assert n_v == 11
        # 2 triangles * 6 = 12 new triangles
        assert len(new_tri) == 12

    def test_vertices_are_one_indexed(self):
        """All vertex indices should be >= 1."""
        tri = [[1, 2, 3]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        for simplex in new_tri:
            for v in simplex:
                assert v >= 1

    def test_simplex_dimension_preserved(self):
        """Output simplices should have the same number of vertices as input."""
        # 2D: triangles (3 vertices each)
        tri_2d = [[1, 2, 3], [2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri_2d)
        for s in new_tri:
            assert len(s) == 3

        # 3D: tetrahedra (4 vertices each)
        tri_3d = [[1, 2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri_3d)
        for s in new_tri:
            assert len(s) == 4

    def test_no_duplicate_simplices(self):
        """There should be no duplicate simplices in the output."""
        tri = [[1, 2, 3], [2, 3, 4]]
        new_tri, _ = barycentric_subdivision_raw(tri)
        # Convert to frozensets for comparison (order within simplex doesn't matter
        # for the resulting simplicial complex, but our output has a canonical ordering)
        as_tuples = [tuple(s) for s in new_tri]
        assert len(as_tuples) == len(set(as_tuples))

    def test_double_subdivision(self):
        """Applying subdivision twice should produce correct counts."""
        tri = [[1, 2, 3]]
        # First subdivision: 6 triangles, 7 vertices
        new_tri, n_v = barycentric_subdivision_raw(tri)
        assert len(new_tri) == 6
        assert n_v == 7

        # Second subdivision: each of 6 triangles -> 6 new ones = 36
        new_tri2, n_v2 = barycentric_subdivision_raw(new_tri)
        assert len(new_tri2) == 36
        assert n_v2 > n_v

    def test_boundary_triangulation(self):
        """Test the boundary of a tetrahedron (4 triangles forming a sphere)."""
        # Boundary of tetrahedron [1,2,3,4] = 4 triangles
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        new_tri, n_v = barycentric_subdivision_raw(tri)
        # 4 vertices + 6 edges + 4 faces = 14 vertices
        assert n_v == 14
        # 4 * 6 = 24 new triangles
        assert len(new_tri) == 24


    def test_euler_characteristic_preserved_2d(self):
        """Barycentric subdivision should preserve the Euler characteristic for 2-manifolds."""
        from itertools import combinations as combs

        def euler_char(triangulation):
            vertices = set()
            edges = set()
            faces = set()
            for simplex in triangulation:
                s = tuple(sorted(simplex))
                faces.add(s)
                for e in combs(s, 2):
                    edges.add(e)
                for v in s:
                    vertices.add((v,))
            return len(vertices) - len(edges) + len(faces)

        # Boundary of tetrahedron = S^2, chi = 2
        tri = [[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]]
        chi_original = euler_char(tri)
        assert chi_original == 2

        new_tri, _ = barycentric_subdivision_raw(tri)
        chi_subdivided = euler_char(new_tri)
        assert chi_subdivided == chi_original

    def test_euler_characteristic_preserved_3d(self):
        """Barycentric subdivision should preserve Euler characteristic for 3-manifolds.

        For closed 3-manifolds, chi = 0.
        """
        from itertools import combinations as combs

        def euler_char_3d(triangulation):
            cells = {0: set(), 1: set(), 2: set(), 3: set()}
            for simplex in triangulation:
                s = tuple(sorted(simplex))
                cells[3].add(s)
                for f in combs(s, 3):
                    cells[2].add(f)
                for e in combs(s, 2):
                    cells[1].add(e)
                for v in s:
                    cells[0].add((v,))
            return sum((-1) ** k * len(cells[k]) for k in range(4))

        # Minimal S^3 triangulation (boundary of 4-simplex has 5 tetrahedra)
        # but that's not a 3-manifold triangulation per se. Use a known one:
        # The boundary of a 5-cell (4-simplex) with vertices 1..5
        # gives 5 tetrahedra forming S^3
        tri_s3 = [
            [1, 2, 3, 4],
            [1, 2, 3, 5],
            [1, 2, 4, 5],
            [1, 3, 4, 5],
            [2, 3, 4, 5],
        ]
        chi_original = euler_char_3d(tri_s3)
        assert chi_original == 0, f"Expected chi=0, got {chi_original}"

        new_tri, _ = barycentric_subdivision_raw(tri_s3)
        chi_subdivided = euler_char_3d(new_tri)
        assert chi_subdivided == chi_original


class TestSubdivideDatasetJson:
    """Tests for subdivide_dataset_json on dataset entries."""

    def _make_entry(self, triangulation, n_vertices, name="S^2"):
        return {
            "id": "test_0",
            "triangulation": triangulation,
            "dimension": 2,
            "n_vertices": n_vertices,
            "betti_numbers": [1, 0, 1],
            "torsion_coefficients": ["", "", ""],
            "name": name,
        }

    def test_preserves_metadata(self):
        """Subdivision should preserve all non-geometric metadata."""
        entry = self._make_entry([[1, 2, 3]], 3, name="T^2")
        result = subdivide_dataset_json([entry], n_subdivisions=1)
        assert len(result) == 1
        assert result[0]["name"] == "T^2"
        assert result[0]["betti_numbers"] == [1, 0, 1]
        assert result[0]["id"] == "test_0"

    def test_updates_n_vertices(self):
        """Subdivision should update n_vertices correctly."""
        entry = self._make_entry([[1, 2, 3]], 3)
        result = subdivide_dataset_json([entry], n_subdivisions=1)
        assert result[0]["n_vertices"] == 7

    def test_zero_subdivisions(self):
        """Zero subdivisions should return unchanged data."""
        entry = self._make_entry([[1, 2, 3]], 3)
        result = subdivide_dataset_json([entry], n_subdivisions=0)
        assert result[0]["triangulation"] == [[1, 2, 3]]
        assert result[0]["n_vertices"] == 3

    def test_multiple_entries(self):
        """Should handle multiple dataset entries."""
        entries = [
            self._make_entry([[1, 2, 3]], 3, name="S^2"),
            self._make_entry([[1, 2, 3], [2, 3, 4]], 4, name="T^2"),
        ]
        result = subdivide_dataset_json(entries, n_subdivisions=1)
        assert len(result) == 2
        assert result[0]["name"] == "S^2"
        assert result[1]["name"] == "T^2"
        assert len(result[0]["triangulation"]) == 6
        assert len(result[1]["triangulation"]) == 12
