"""Regression tests for Q-map pixel index correctness.

These tests verify that qmap["x"] and qmap["y"] contain absolute pixel
indices (0 to N-1) rather than beam-center offsets. The fix creates
separate pixel-index meshgrids independent of the beam center.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.simplemask.qmap import (
    compute_reflection_qmap,
    compute_transmission_qmap,
)


class TestTransmissionPixelIndices:
    """Tests for pixel indices in transmission Q-map."""

    @pytest.fixture()
    def default_params(self):
        """Standard detector geometry for testing."""
        return {
            "energy": 10.0,
            "center": (64.0, 128.0),
            "shape": (128, 256),
            "pix_dim": 0.075,
            "det_dist": 5000.0,
        }

    def test_x_is_pixel_index(self, default_params):
        """qmap["x"][0,:] should be arange(ncols), i.e. column indices."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        ncols = default_params["shape"][1]
        expected = np.arange(ncols, dtype=np.int32)
        np.testing.assert_array_equal(
            qmap["x"][0, :],
            expected,
            err_msg="x values in first row should be column indices 0..N-1",
        )

    def test_y_is_pixel_index(self, default_params):
        """qmap["y"][:,0] should be arange(nrows), i.e. row indices."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        nrows = default_params["shape"][0]
        expected = np.arange(nrows, dtype=np.int32)
        np.testing.assert_array_equal(
            qmap["y"][:, 0],
            expected,
            err_msg="y values in first column should be row indices 0..N-1",
        )

    def test_xy_independent_of_beam_center(self, default_params):
        """Pixel indices should be the same regardless of beam center."""
        compute_transmission_qmap.cache_clear()
        qmap_a, _ = compute_transmission_qmap(**default_params)

        # Different beam center
        params_b = {**default_params, "center": (32.0, 200.0)}
        compute_transmission_qmap.cache_clear()
        qmap_b, _ = compute_transmission_qmap(**params_b)

        np.testing.assert_array_equal(
            qmap_a["x"],
            qmap_b["x"],
            err_msg="x pixel indices should not depend on beam center",
        )
        np.testing.assert_array_equal(
            qmap_a["y"],
            qmap_b["y"],
            err_msg="y pixel indices should not depend on beam center",
        )

    def test_x_shape_matches_detector(self, default_params):
        """qmap["x"] shape should match detector shape."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        assert qmap["x"].shape == default_params["shape"]

    def test_y_shape_matches_detector(self, default_params):
        """qmap["y"] shape should match detector shape."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        assert qmap["y"].shape == default_params["shape"]

    def test_x_constant_along_rows(self, default_params):
        """Each column should have the same x value (column index)."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        # All rows should have the same x values
        for row_idx in range(default_params["shape"][0]):
            np.testing.assert_array_equal(qmap["x"][row_idx, :], qmap["x"][0, :])

    def test_y_constant_along_columns(self, default_params):
        """Each row should have the same y value (row index)."""
        compute_transmission_qmap.cache_clear()
        qmap, _ = compute_transmission_qmap(**default_params)
        # All columns should have the same y values
        for col_idx in range(default_params["shape"][1]):
            np.testing.assert_array_equal(qmap["y"][:, col_idx], qmap["y"][:, 0])


class TestReflectionPixelIndices:
    """Tests for pixel indices in reflection Q-map."""

    @pytest.fixture()
    def default_params(self):
        """Standard detector geometry for reflection testing."""
        return {
            "energy": 10.0,
            "center": (64.0, 128.0),
            "shape": (128, 256),
            "pix_dim": 0.075,
            "det_dist": 5000.0,
            "alpha_i_deg": 0.14,
            "orientation": "north",
        }

    def test_x_is_pixel_index(self, default_params):
        """qmap["x"][0,:] should be arange(ncols) for reflection geometry."""
        compute_reflection_qmap.cache_clear()
        qmap, _ = compute_reflection_qmap(**default_params)
        ncols = default_params["shape"][1]
        expected = np.arange(ncols, dtype=np.int32)
        np.testing.assert_array_equal(
            qmap["x"][0, :],
            expected,
            err_msg="x values in first row should be column indices 0..N-1",
        )

    def test_y_is_pixel_index(self, default_params):
        """qmap["y"][:,0] should be arange(nrows) for reflection geometry."""
        compute_reflection_qmap.cache_clear()
        qmap, _ = compute_reflection_qmap(**default_params)
        nrows = default_params["shape"][0]
        expected = np.arange(nrows, dtype=np.int32)
        np.testing.assert_array_equal(
            qmap["y"][:, 0],
            expected,
            err_msg="y values in first column should be row indices 0..N-1",
        )

    def test_xy_independent_of_beam_center(self, default_params):
        """Pixel indices should be the same regardless of beam center."""
        compute_reflection_qmap.cache_clear()
        qmap_a, _ = compute_reflection_qmap(**default_params)

        params_b = {**default_params, "center": (32.0, 200.0)}
        compute_reflection_qmap.cache_clear()
        qmap_b, _ = compute_reflection_qmap(**params_b)

        np.testing.assert_array_equal(
            qmap_a["x"],
            qmap_b["x"],
            err_msg="x pixel indices should not depend on beam center",
        )
        np.testing.assert_array_equal(
            qmap_a["y"],
            qmap_b["y"],
            err_msg="y pixel indices should not depend on beam center",
        )
