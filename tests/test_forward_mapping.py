import pytest
import numpy as np
from opyrability import AIS2AOS_map
from shower import shower2x2, shower3x2

# --------------------------------------------------------------------------- #
# Forward mapping (AIS2AOS_map) tests using analytical shower models.
#
# shower2x2:
#   y[0] = u[0] + u[1]                             (total flow)
#   y[1] = (60*u[0] + 120*u[1]) / (u[0] + u[1])   (temperature)
#
# All expected values can be computed by hand.
# --------------------------------------------------------------------------- #


class TestAIS2AOSBasic:
    """Basic forward mapping with the 2x2 shower model."""

    def test_output_shapes_2x2(self):
        AIS_bounds = np.array([[1, 10], [1, 10]])
        resolution = [5, 5]
        AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, resolution, plot=False)
        assert AIS.shape == (5, 5, 2)
        assert AOS.shape == (5, 5, 2)

    def test_known_corner_values(self):
        """At u=[1,1]: y=[2, 90]. At u=[10,10]: y=[20, 90]."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        resolution = [2, 2]
        AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, resolution, plot=False)

        # Corner (0,0): u=[1,1] -> y=[2, 90]
        np.testing.assert_allclose(AOS[0, 0], [2.0, 90.0], atol=1e-10)
        # Corner (1,1): u=[10,10] -> y=[20, 90]
        np.testing.assert_allclose(AOS[1, 1], [20.0, 90.0], atol=1e-10)
        # Corner (0,1): u=[1,10] -> y=[11, (60+1200)/11 = 114.545...]
        np.testing.assert_allclose(AOS[0, 1, 0], 11.0, atol=1e-10)
        np.testing.assert_allclose(AOS[0, 1, 1], 1260.0 / 11.0, atol=1e-10)

    def test_equal_flows_give_90(self):
        """When u[0] == u[1], temperature is always 90."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        resolution = [5, 5]
        AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, resolution, plot=False)

        # Diagonal entries have u[0] == u[1]
        for i in range(5):
            np.testing.assert_allclose(AOS[i, i, 1], 90.0, atol=1e-10)

    def test_single_resolution(self):
        """Resolution [1,1] should give a single point."""
        AIS_bounds = np.array([[5, 5], [5, 5]])
        resolution = [1, 1]
        AIS, AOS = AIS2AOS_map(shower2x2, AIS_bounds, resolution, plot=False)
        np.testing.assert_allclose(AOS[0, 0], [10.0, 90.0], atol=1e-10)


class TestAIS2AOSWithEDS:
    """Forward mapping with Expected Disturbance Set (EDS)."""

    def test_with_eds_shapes(self):
        """shower3x2 has 2 control inputs + 1 disturbance -> 2 outputs."""
        AIS_bounds = np.array([[1, 10], [1, 10]])
        EDS_bounds = np.array([[0, 5]])
        AIS, AOS = AIS2AOS_map(shower3x2,
                                AIS_bounds, [3, 3],
                                EDS_bound=EDS_bounds,
                                EDS_resolution=[3],
                                plot=False)
        # Should have 3 AIS dims + 1 EDS dim + output dim
        assert AIS.shape[-1] == 3  # 2 inputs + 1 disturbance
        assert AOS.shape[-1] == 2  # 2 outputs

    def test_zero_disturbance_matches_nominal(self):
        """With d=0, shower3x2 should match shower2x2."""
        AIS_bounds = np.array([[1, 10], [1, 10]])

        # shower3x2 with d=0
        def shower3x2_d0(u):
            return shower3x2(np.array([u[0], u[1], 0.0]))

        AIS1, AOS1 = AIS2AOS_map(shower2x2,
                                   AIS_bounds, [3, 3], plot=False)
        AIS2, AOS2 = AIS2AOS_map(shower3x2_d0,
                                   AIS_bounds, [3, 3], plot=False)
        np.testing.assert_allclose(AOS1, AOS2, atol=1e-10)
