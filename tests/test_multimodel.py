import pytest
import numpy as np
import polytope as pc
from opyrability import multimodel_rep
from shower import shower2x2

# --------------------------------------------------------------------------- #
# Multimodel representation tests.
#
# multimodel_rep returns [Region, AS_coords] where:
#   - Region is a polytope.Region of convex polytopes
#   - AS_coords is an ndarray of the vertices
#
# We test region construction using the analytical shower model.
# --------------------------------------------------------------------------- #


class TestMultimodelRep:
    """Tests for multimodel_rep polytopic region construction."""

    def test_returns_list_with_region(self):
        """Should return [Region, ndarray]."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        result = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], pc.Region)
        assert isinstance(result[1], np.ndarray)

    def test_region_dimension(self):
        AIS_bounds = np.array([[1, 10], [1, 10]])
        result = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        region = result[0]
        assert region.dim == 2

    def test_region_contains_known_point(self):
        """The AOS region should contain y=[10, 90] (from u=[5,5])."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        result = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        region = result[0]

        # Check using polytope containment
        point = np.array([[10.0, 90.0]])
        # A point is in a Region if it's in at least one polytope
        contained = any(p.contains(point.T) for p in region)
        assert contained

    def test_region_excludes_outside_point(self):
        """Points far outside the AOS should not be contained."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        result = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        region = result[0]

        point = np.array([[100.0, 200.0]])
        contained = any(p.contains(point.T) for p in region)
        assert not contained

    def test_identity_model_produces_square_region(self):
        """Identity model y=u should produce AOS = AIS."""
        def identity(u):
            return u

        bounds = np.array([[0, 10], [0, 10]])
        result = multimodel_rep(identity, bounds, [5, 5], plot=False)
        region = result[0]

        # Center point should be contained
        point = np.array([[5.0, 5.0]])
        contained = any(p.contains(point.T) for p in region)
        assert contained

    def test_simplices_vs_polyhedra(self):
        """Both tracing methods should produce valid regions."""
        AIS_bounds = np.array([[1, 10], [1, 10]])

        result_simp = multimodel_rep(shower2x2, AIS_bounds, [5, 5],
                                      polytopic_trace='simplices', plot=False)
        result_poly = multimodel_rep(shower2x2, AIS_bounds, [5, 5],
                                      polytopic_trace='polyhedra', plot=False)

        # Both should contain the same interior point
        point = np.array([[10.0, 90.0]])
        assert any(p.contains(point.T) for p in result_simp[0])
        assert any(p.contains(point.T) for p in result_poly[0])

    def test_vertices_cover_aos_range(self):
        """The vertex coordinates should span the expected output range."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        result = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        coords = result[1]

        # y[0] = u[0]+u[1], range: [2, 20]
        assert coords[:, 0].min() == pytest.approx(2.0, abs=0.1)
        assert coords[:, 0].max() == pytest.approx(20.0, abs=0.1)
