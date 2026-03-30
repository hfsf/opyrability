import pytest
import numpy as np
from opyrability import create_grid, get_extreme_vertices

# --------------------------------------------------------------------------- #
# Utility function tests.
#
# create_grid expects list resolutions (not numpy arrays) and returns
# a multidimensional array of shape (*resolution, n_dims).
# --------------------------------------------------------------------------- #


class TestCreateGrid:
    """Tests for the create_grid utility."""

    def test_2d_grid_shape(self):
        bounds = np.array([[0, 10], [0, 10]])
        resolution = [5, 5]
        grid = create_grid(bounds, resolution)
        assert grid.shape == (5, 5, 2)

    def test_2d_grid_corners(self):
        bounds = np.array([[0, 10], [0, 20]])
        resolution = [2, 2]
        grid = create_grid(bounds, resolution)
        np.testing.assert_allclose(grid[0, 0], [0, 0])
        np.testing.assert_allclose(grid[1, 1], [10, 20])

    def test_3d_grid_shape(self):
        bounds = np.array([[0, 1], [0, 1], [0, 1]])
        resolution = [3, 3, 3]
        grid = create_grid(bounds, resolution)
        assert grid.shape == (3, 3, 3, 3)

    def test_1d_grid(self):
        bounds = np.array([[0, 10]])
        resolution = [5]
        grid = create_grid(bounds, resolution)
        assert grid.shape == (5, 1)
        np.testing.assert_allclose(grid[:, 0],
                                    np.linspace(0, 10, 5))

    def test_grid_bounds_respected(self):
        bounds = np.array([[-5, 5], [10, 20]])
        resolution = [10, 10]
        grid = create_grid(bounds, resolution)
        assert grid[..., 0].min() == pytest.approx(-5.0)
        assert grid[..., 0].max() == pytest.approx(5.0)
        assert grid[..., 1].min() == pytest.approx(10.0)
        assert grid[..., 1].max() == pytest.approx(20.0)


class TestGetExtremeVertices:
    """Tests for the get_extreme_vertices utility."""

    def test_2d_box(self):
        bounds = np.array([[0, 10], [0, 20]])
        vertices = get_extreme_vertices(bounds)
        assert vertices.shape[0] == 4
        assert vertices.shape[1] == 2

    def test_3d_box(self):
        bounds = np.array([[0, 1], [0, 1], [0, 1]])
        vertices = get_extreme_vertices(bounds)
        assert vertices.shape[0] == 8
        assert vertices.shape[1] == 3

    def test_vertices_on_bounds(self):
        bounds = np.array([[-1, 1], [-2, 2]])
        vertices = get_extreme_vertices(bounds)
        for v in vertices:
            assert v[0] in [-1, 1]
            assert v[1] in [-2, 2]
