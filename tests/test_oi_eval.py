import pytest
import numpy as np
import polytope as pc
from opyrability import multimodel_rep, OI_eval, AIS2AOS_map
from shower import shower2x2, inv_shower2x2

# --------------------------------------------------------------------------- #
# Operability Index (OI) evaluation tests.
#
# We use geometric configurations where the expected OI can be computed
# analytically. The shower model provides convenient, well-understood
# polytopic regions.
# --------------------------------------------------------------------------- #


class TestOIKnownValues:
    """OI evaluation against analytically known values."""

    def test_shower_oi(self):
        """Reproduce the existing shower2x2 OI test."""
        DOS_bounds = np.array([[10, 20], [70, 100]])
        AIS_bounds = np.array([[1, 10], [1, 10]])

        AOS_region = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        OI = OI_eval(AOS_region, DOS_bounds, plot=False)

        assert OI == pytest.approx(60.23795007653283, abs=1e-7, rel=1e-7)

    def test_inverse_shower_oi(self):
        """Reproduce the existing inverse shower OI test."""
        AOS_bounds = np.array([[10, 20], [70, 100]])
        DIS_bounds = np.array([[0, 10.00], [0, 10.00]])

        AIS_region = multimodel_rep(inv_shower2x2, AOS_bounds, [5, 5],
                                     polytopic_trace='polyhedra', plot=False)
        OI = OI_eval(AIS_region, DIS_bounds, plot=False)

        assert OI == pytest.approx(40.0, abs=1e-7, rel=1e-7)


class TestOIEdgeCases:
    """Edge cases for OI evaluation."""

    def test_no_overlap_returns_zero(self):
        """When DOS is completely outside AOS, OI should be 0."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        AOS_region = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)

        # DOS far outside the shower's output range
        DOS_bounds = np.array([[500, 600], [500, 600]])
        OI = OI_eval(AOS_region, DOS_bounds, plot=False)

        assert OI == 0.0

    def test_full_containment(self):
        """When DOS is fully inside AOS, OI should be 100."""
        # Use a linear model: y = u (identity)
        def identity_model(u):
            return u

        AIS_bounds = np.array([[0, 10], [0, 10]])
        AOS_region = multimodel_rep(identity_model, AIS_bounds, [5, 5],
                                     plot=False)

        # DOS well inside AOS
        DOS_bounds = np.array([[2, 8], [2, 8]])
        OI = OI_eval(AOS_region, DOS_bounds, plot=False)

        assert OI == pytest.approx(100.0, abs=1.0)

    def test_oi_between_0_and_100(self):
        """OI should always be in [0, 100]."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        AOS_region = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)

        DOS_bounds = np.array([[5, 25], [60, 120]])
        OI = OI_eval(AOS_region, DOS_bounds, plot=False)

        assert 0.0 <= OI <= 100.0


class TestOICalculationMethods:
    """Test different OI calculation methods give consistent results."""

    def test_polytope_vs_robust(self):
        """Both calculation methods should give similar results."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        AOS_region = multimodel_rep(shower2x2, AIS_bounds, [5, 5], plot=False)
        DOS_bounds = np.array([[10, 20], [70, 100]])

        OI_polytope = OI_eval(AOS_region, DOS_bounds,
                               hypervol_calc='polytope', plot=False)
        OI_robust = OI_eval(AOS_region, DOS_bounds,
                             hypervol_calc='robust', plot=False)

        assert OI_polytope == pytest.approx(OI_robust, abs=5.0)
