import pytest
import numpy as np
from opyrability import nlp_based_approach
from shower import shower2x2

# --------------------------------------------------------------------------- #
# Inverse mapping (nlp_based_approach) tests.
#
# For shower2x2, given a desired output y, the analytical inverse is:
#   u[0] = y[0] * (y[1] - 120) / (60 - 120)  =  y[0] * (y[1] - 120) / (-60)
#   u[1] = y[0] - u[0]
#
# We test that the NLP solver recovers inputs that produce the desired outputs.
# --------------------------------------------------------------------------- #


class TestNLPInverseBasic:
    """Inverse mapping with known analytical solutions."""

    @pytest.fixture
    def solver_params(self):
        return dict(
            u0=np.array([5.0, 5.0]),
            lb=np.array([0.0, 0.0]),
            ub=np.array([100.0, 100.0]),
            plot=False,
            warmstart=True,
        )

    def test_single_point_recovery(self, solver_params):
        """DOS containing a single point y=[10, 90] should recover u=[5, 5]."""
        DOS_bounds = np.array([[10.0, 10.0], [90.0, 90.0]])
        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, DOS_bounds, [1, 1],
            method='ipopt', ad=False, **solver_params
        )
        np.testing.assert_allclose(fDOS[0], [10.0, 90.0], atol=1e-4)
        np.testing.assert_allclose(fDIS[0], [5.0, 5.0], atol=1e-4)

    def test_fDOS_matches_DOS_grid(self, solver_params):
        """The feasible DOS should lie on (or very near) the DOS grid points."""
        DOS_bounds = np.array([[10.0, 15.0], [80.0, 100.0]])
        resolution = [3, 3]
        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, DOS_bounds, resolution,
            method='ipopt', ad=False, **solver_params
        )
        # Each fDOS point should match the model evaluated at fDIS
        for i in range(fDIS.shape[0]):
            y_check = shower2x2(fDIS[i])
            np.testing.assert_allclose(fDOS[i], y_check, atol=1e-4)

    def test_output_shapes(self, solver_params):
        DOS_bounds = np.array([[10.0, 20.0], [80.0, 100.0]])
        resolution = [4, 4]
        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, DOS_bounds, resolution,
            method='ipopt', ad=False, **solver_params
        )
        assert fDIS.shape == (16, 2)
        assert fDOS.shape == (16, 2)
        assert len(msg) == 16

    def test_bounds_respected(self, solver_params):
        """All recovered inputs should respect the bounds."""
        solver_params['lb'] = np.array([0.0, 0.0])
        solver_params['ub'] = np.array([15.0, 15.0])
        DOS_bounds = np.array([[10.0, 20.0], [80.0, 100.0]])

        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, DOS_bounds, [5, 5],
            method='ipopt', ad=False, **solver_params
        )
        assert np.all(fDIS >= -1e-6)  # small tolerance for solver
        assert np.all(fDIS <= 15.0 + 1e-6)


class TestNLPMethods:
    """Test different solver methods produce consistent results."""

    def setup_method(self):
        self.DOS_bounds = np.array([[10.0, 15.0], [85.0, 95.0]])
        self.resolution = [3, 3]
        self.params = dict(
            u0=np.array([5.0, 5.0]),
            lb=np.array([0.0, 0.0]),
            ub=np.array([100.0, 100.0]),
            plot=False, warmstart=True, ad=False,
        )

    def test_trust_constr(self):
        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, self.DOS_bounds, self.resolution,
            method='trust-constr', **self.params
        )
        for i in range(fDIS.shape[0]):
            y_check = shower2x2(fDIS[i])
            np.testing.assert_allclose(fDOS[i], y_check, atol=1e-3)

    def test_nelder_mead(self):
        fDIS, fDOS, msg = nlp_based_approach(
            shower2x2, self.DOS_bounds, self.resolution,
            method='Nelder-Mead', **self.params
        )
        for i in range(fDIS.shape[0]):
            y_check = shower2x2(fDIS[i])
            np.testing.assert_allclose(fDOS[i], y_check, atol=1e-3)


class TestNLPWarmstart:
    """Test warmstart on/off produces valid results."""

    def test_warmstart_on_off_both_valid(self):
        DOS_bounds = np.array([[10.0, 15.0], [85.0, 95.0]])
        params = dict(
            u0=np.array([5.0, 5.0]),
            lb=np.array([0.0, 0.0]),
            ub=np.array([100.0, 100.0]),
            plot=False, method='ipopt', ad=False,
        )

        fDIS_ws, fDOS_ws, _ = nlp_based_approach(
            shower2x2, DOS_bounds, [3, 3], warmstart=True, **params
        )
        fDIS_no, fDOS_no, _ = nlp_based_approach(
            shower2x2, DOS_bounds, [3, 3], warmstart=False, **params
        )

        # Both should produce valid forward-consistent results
        for i in range(fDIS_ws.shape[0]):
            np.testing.assert_allclose(fDOS_ws[i], shower2x2(fDIS_ws[i]), atol=1e-4)
            np.testing.assert_allclose(fDOS_no[i], shower2x2(fDIS_no[i]), atol=1e-4)
