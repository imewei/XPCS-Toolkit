"""Tests for JAX-native interpolation migration.

Tests for Technical Guidelines compliance:
- T040: No scipy.interpolate imports in module/
- T041: interpolate_g2_data() produces correct output
- T042: vectorized_background_subtraction() produces correct output
- T043: optimized_c2_sampling() produces correct output
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest


class TestNoScipyInterpolateImports:
    """T040: Verify no scipy.interpolate imports in module/ directory."""

    def test_no_scipy_interpolate_imports(self):
        """Verify grep finds no scipy.interpolate imports in module/."""
        module_path = Path(__file__).parent.parent.parent.parent / "xpcsviewer" / "module"

        # Use grep to search for scipy.interpolate imports
        result = subprocess.run(
            ["grep", "-r", "from scipy.interpolate", str(module_path)],
            capture_output=True,
            text=True,
        )

        # grep returns exit code 1 when no matches found (success for us)
        assert result.returncode == 1, (
            f"Found scipy.interpolate imports in module/:\n{result.stdout}"
        )

    def test_no_scipy_ndimage_zoom_imports(self):
        """Verify grep finds no direct scipy.ndimage imports in module/."""
        module_path = Path(__file__).parent.parent.parent.parent / "xpcsviewer" / "module"

        # Use grep to search for scipy.ndimage imports
        result = subprocess.run(
            ["grep", "-r", "from scipy.ndimage import", str(module_path)],
            capture_output=True,
            text=True,
        )

        # grep returns exit code 1 when no matches found (success for us)
        assert result.returncode == 1, (
            f"Found scipy.ndimage imports in module/:\n{result.stdout}"
        )


class TestG2Interpolation:
    """T041: Test vectorized_g2_interpolation produces correct output."""

    def test_g2_interpolation_basic(self):
        """Verify G2 interpolation produces reasonable output."""
        from xpcsviewer.module.g2mod import vectorized_g2_interpolation

        # Create test data with more points for stable cubic interpolation
        tel = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
        g2_data = np.column_stack([
            1.0 + 0.5 * np.exp(-tel / 0.1),  # q=0: fast decay
            1.0 + 0.3 * np.exp(-tel / 1.0),  # q=1: slow decay
        ])
        # Target points within the range of original data
        target_tel = np.array([0.002, 0.02, 0.2, 2.0])

        # Run interpolation
        result = vectorized_g2_interpolation(tel, g2_data, target_tel)

        # Check output shape
        assert result.shape == (4, 2), f"Expected shape (4, 2), got {result.shape}"

        # Check output is finite (no NaN/inf)
        assert np.all(np.isfinite(result)), "Interpolation produced non-finite values"

    def test_g2_interpolation_output_shape(self):
        """Verify G2 interpolation output shape is correct."""
        from xpcsviewer.module.g2mod import vectorized_g2_interpolation

        # Create test data
        tel = np.linspace(0.001, 10.0, 20)
        g2_data = np.column_stack([
            1.0 + 0.5 * np.exp(-tel / 0.1),
            1.0 + 0.3 * np.exp(-tel / 1.0),
            1.0 + 0.2 * np.exp(-tel / 5.0),
        ])
        target_tel = np.linspace(0.01, 5.0, 10)

        result = vectorized_g2_interpolation(tel, g2_data, target_tel)

        # Check shape matches target_tel length x number of q values
        assert result.shape == (10, 3)


class TestVectorizedBackgroundSubtraction:
    """T042: Test vectorized_background_subtraction produces correct output."""

    def test_background_subtraction_same_q(self):
        """Verify background subtraction with same q-values."""
        from xpcsviewer.module.saxs1d import vectorized_background_subtraction

        q = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        I_fg = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        I_bg = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        I_err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        foreground = (q, I_fg, I_err)
        background = (q, I_bg, I_err)

        result_q, result_I, result_err = vectorized_background_subtraction(
            foreground, background, weight=1.0
        )

        # Check q unchanged
        np.testing.assert_array_equal(result_q, q)

        # Check subtraction
        expected_I = I_fg - I_bg
        np.testing.assert_array_almost_equal(result_I, expected_I)

    def test_background_subtraction_different_q(self):
        """Verify background subtraction with different q-values requiring interpolation."""
        from xpcsviewer.module.saxs1d import vectorized_background_subtraction

        q_fg = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        I_fg = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
        I_err_fg = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        # Background with different q spacing (but overlapping range)
        q_bg = np.array([0.005, 0.015, 0.025, 0.035, 0.045, 0.055])
        I_bg = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        I_err_bg = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        foreground = (q_fg, I_fg, I_err_fg)
        background = (q_bg, I_bg, I_err_bg)

        result_q, result_I, result_err = vectorized_background_subtraction(
            foreground, background, weight=1.0
        )

        # Check output shape matches foreground
        assert len(result_q) == len(q_fg)
        assert len(result_I) == len(q_fg)

        # Check values are close to expected (I_fg - interpolated I_bg ≈ I_fg - 10)
        expected_approx = I_fg - 10.0
        np.testing.assert_array_almost_equal(result_I, expected_approx, decimal=1)


class TestOptimizedC2Sampling:
    """T043: Test optimized_c2_sampling with JAX-based zoom replacement."""

    def test_c2_sampling_bilinear(self):
        """Verify bilinear downsampling produces correct output size."""
        from xpcsviewer.module.twotime_utils import optimized_c2_sampling

        # Create test C2 matrix
        c2_matrix = np.random.rand(100, 100)
        target_size = 50

        result = optimized_c2_sampling(c2_matrix, target_size, method="bilinear")

        # Check output shape
        assert result.shape == (target_size, target_size)

    def test_c2_sampling_uniform(self):
        """Verify uniform downsampling produces correct output."""
        from xpcsviewer.module.twotime_utils import optimized_c2_sampling

        c2_matrix = np.arange(100).reshape(10, 10).astype(float)
        target_size = 5

        result = optimized_c2_sampling(c2_matrix, target_size, method="uniform")

        # Check output shape
        assert result.shape == (target_size, target_size)

    def test_c2_sampling_no_change_when_smaller(self):
        """Verify no change when target_size >= current_size."""
        from xpcsviewer.module.twotime_utils import optimized_c2_sampling

        c2_matrix = np.random.rand(50, 50)
        target_size = 100

        result = optimized_c2_sampling(c2_matrix, target_size, method="bilinear")

        # Should return original unchanged
        np.testing.assert_array_equal(result, c2_matrix)
