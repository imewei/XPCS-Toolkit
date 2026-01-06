"""Tests for NLSQ uncertainty band computation (T091).

Tests for compute_uncertainty_band() function that computes
prediction uncertainty via Jacobian and variance propagation (FR-016).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.visualization import compute_uncertainty_band


def linear_model(x, a, b):
    """Simple linear model for testing."""
    return a * x + b


def single_exp_model(x, tau, baseline, contrast):
    """Single exponential model for testing."""
    return baseline + contrast * np.exp(-2 * x / tau)


class TestComputeUncertaintyBand:
    """Tests for compute_uncertainty_band function."""

    def test_linear_model_band_shape(self) -> None:
        """Test output shapes match input."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])  # slope=2, intercept=1
        pcov = np.diag([0.1, 0.05])  # variances on slope and intercept

        y_fit, y_lower, y_upper = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.95
        )

        assert y_fit.shape == x.shape
        assert y_lower.shape == x.shape
        assert y_upper.shape == x.shape

    def test_band_ordering(self) -> None:
        """Test lower <= fit <= upper."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])

        y_fit, y_lower, y_upper = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.95
        )

        assert np.all(y_lower <= y_fit)
        assert np.all(y_fit <= y_upper)

    def test_band_width_increases_with_uncertainty(self) -> None:
        """Test band width increases with larger covariance."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])

        # Small uncertainty
        pcov_small = np.diag([0.01, 0.01])
        _, lower_small, upper_small = compute_uncertainty_band(
            linear_model, x, popt, pcov_small
        )
        width_small = np.mean(upper_small - lower_small)

        # Large uncertainty
        pcov_large = np.diag([1.0, 1.0])
        _, lower_large, upper_large = compute_uncertainty_band(
            linear_model, x, popt, pcov_large
        )
        width_large = np.mean(upper_large - lower_large)

        assert width_large > width_small

    def test_zero_covariance_gives_zero_width(self) -> None:
        """Test zero covariance gives zero band width."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.zeros((2, 2))

        y_fit, y_lower, y_upper = compute_uncertainty_band(linear_model, x, popt, pcov)

        np.testing.assert_allclose(y_lower, y_fit)
        np.testing.assert_allclose(y_upper, y_fit)

    def test_confidence_level_affects_width(self) -> None:
        """Test higher confidence gives wider bands."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])

        _, lower_90, upper_90 = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.90
        )
        width_90 = np.mean(upper_90 - lower_90)

        _, lower_99, upper_99 = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.99
        )
        width_99 = np.mean(upper_99 - lower_99)

        assert width_99 > width_90

    def test_single_exp_model(self) -> None:
        """Test uncertainty band for exponential model."""
        x = np.logspace(-3, 2, 50)
        popt = np.array([1.0, 1.0, 0.3])  # tau, baseline, contrast
        pcov = np.diag([0.01, 0.001, 0.01])

        y_fit, y_lower, y_upper = compute_uncertainty_band(
            single_exp_model, x, popt, pcov
        )

        # Check shapes
        assert y_fit.shape == x.shape
        assert np.all(y_lower <= y_fit)
        assert np.all(y_fit <= y_upper)

        # Check fit values match model
        expected = single_exp_model(x, *popt)
        np.testing.assert_allclose(y_fit, expected, rtol=1e-10)

    def test_single_point(self) -> None:
        """Test band computation for single point."""
        x = np.array([5.0])
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])

        y_fit, y_lower, y_upper = compute_uncertainty_band(linear_model, x, popt, pcov)

        assert y_fit.shape == (1,)
        assert y_lower[0] < y_fit[0] < y_upper[0]
