"""Regression tests for IEEE 754 float precision in partition binning.

These tests verify that generate_partition() correctly handles floating-point
bin-edge values that are sensitive to IEEE 754 representation errors.
The fix rounds both xmap and bin edges to 12 decimal places before digitize().
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.simplemask.utils import generate_partition


class TestLinearBinEdgePrecision:
    """Tests for linear binning precision at bin edges."""

    def test_uniform_edge_values_produce_single_bin(self):
        """Values exactly at a bin edge should all land in the same bin.

        When all pixel values equal a bin-edge value, digitize should assign
        them consistently to one bin, not split them due to float rounding.
        """
        nrows, ncols = 10, 10
        edge_val = 0.3  # A value that is an exact bin edge for linspace(0, 0.9, 4)
        xmap = np.full((nrows, ncols), edge_val, dtype=np.float64)
        mask = np.ones((nrows, ncols), dtype=np.uint8)

        result = generate_partition(
            map_name="q", mask=mask, xmap=xmap, num_pts=3, style="linear"
        )

        partition = result["partition"]
        # All unmasked pixels should be in the same bin
        unique_bins = np.unique(partition[mask > 0])
        assert len(unique_bins) == 1, (
            f"Uniform edge value {edge_val} was split across bins: {unique_bins}"
        )

    def test_arithmetic_edge_precision(self):
        """Classic 0.1 + 0.2 != 0.3 scenario should not cause bin misassignment.

        The value 0.1 + 0.2 = 0.30000000000000004 in IEEE 754. Without rounding,
        this value could be placed in a different bin than 0.3.
        """
        nrows, ncols = 10, 10
        mask = np.ones((nrows, ncols), dtype=np.uint8)

        # Create xmap where some pixels are exactly 0.3 and others are 0.1+0.2
        xmap = np.full((nrows, ncols), 0.3, dtype=np.float64)
        xmap[0, :] = 0.1 + 0.2  # This equals 0.30000000000000004

        result = generate_partition(
            map_name="q", mask=mask, xmap=xmap, num_pts=3, style="linear"
        )

        partition = result["partition"]
        # Row 0 (0.1+0.2) and all other rows (0.3) should be in the same bin
        bin_row0 = np.unique(partition[0, :])
        bin_others = np.unique(partition[1:, :])
        assert len(bin_row0) == 1
        assert len(bin_others) == 1
        assert bin_row0[0] == bin_others[0], (
            f"0.1+0.2 landed in bin {bin_row0[0]} but 0.3 in bin {bin_others[0]}"
        )


class TestLogBinEdgePrecision:
    """Tests for logarithmic binning precision at bin edges."""

    def test_uniform_edge_values_produce_single_bin(self):
        """Log-spaced bin edges should handle uniform values correctly."""
        nrows, ncols = 10, 10
        # Use a value that is a plausible log-space bin edge
        edge_val = 0.01  # Common Q value
        xmap = np.full((nrows, ncols), edge_val, dtype=np.float64)
        mask = np.ones((nrows, ncols), dtype=np.uint8)

        result = generate_partition(
            map_name="q", mask=mask, xmap=xmap, num_pts=3, style="logarithmic"
        )

        partition = result["partition"]
        unique_bins = np.unique(partition[mask > 0])
        assert len(unique_bins) == 1, (
            f"Uniform log edge value {edge_val} was split across bins: {unique_bins}"
        )


class TestBinEdgeBoundaryAssignment:
    """Tests for correct bin assignment at min/max boundaries."""

    def test_max_value_assigned_to_last_bin(self):
        """The maximum value in the data should be assigned to the last bin."""
        nrows, ncols = 10, 10
        mask = np.ones((nrows, ncols), dtype=np.uint8)
        xmap = np.linspace(0.01, 0.1, nrows * ncols).reshape(nrows, ncols)

        result = generate_partition(
            map_name="q", mask=mask, xmap=xmap, num_pts=5, style="linear"
        )

        partition = result["partition"]
        num_pts = result["num_pts"]
        # The pixel with max value should be in the last bin (num_pts)
        max_pos = np.unravel_index(np.argmax(xmap), xmap.shape)
        assert partition[max_pos] == num_pts, (
            f"Max value pixel is in bin {partition[max_pos]}, expected {num_pts}"
        )

    def test_min_value_assigned_to_first_bin(self):
        """The minimum value in the data should be assigned to the first bin."""
        nrows, ncols = 10, 10
        mask = np.ones((nrows, ncols), dtype=np.uint8)
        xmap = np.linspace(0.01, 0.1, nrows * ncols).reshape(nrows, ncols)

        result = generate_partition(
            map_name="q", mask=mask, xmap=xmap, num_pts=5, style="linear"
        )

        partition = result["partition"]
        # The pixel with min value should be in the first bin (1)
        min_pos = np.unravel_index(np.argmin(xmap), xmap.shape)
        assert partition[min_pos] == 1, (
            f"Min value pixel is in bin {partition[min_pos]}, expected 1"
        )
