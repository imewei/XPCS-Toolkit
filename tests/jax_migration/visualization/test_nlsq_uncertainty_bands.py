"""Tests for NLSQ uncertainty band computation (T091).

Tests for compute_uncertainty_band() and compute_prediction_interval()
functions that compute prediction uncertainty via Jacobian and
variance propagation (FR-016, NLSQ 0.6.0).
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.visualization import (
    compute_prediction_interval,
    compute_uncertainty_band,
)


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


class TestComputePredictionInterval:
    """Tests for compute_prediction_interval function (NLSQ 0.6.0).

    Prediction intervals are wider than confidence intervals because they
    account for both parameter uncertainty AND observation noise.
    """

    def test_prediction_interval_shape(self) -> None:
        """Test output shapes match input."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])
        residuals = np.random.normal(0, 0.5, 50)

        y_fit, pi_lower, pi_upper = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals, confidence=0.95
        )

        assert y_fit.shape == x.shape
        assert pi_lower.shape == x.shape
        assert pi_upper.shape == x.shape

    def test_prediction_interval_ordering(self) -> None:
        """Test lower <= fit <= upper."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])
        residuals = np.random.normal(0, 0.5, 50)

        y_fit, pi_lower, pi_upper = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals
        )

        assert np.all(pi_lower <= y_fit)
        assert np.all(y_fit <= pi_upper)

    def test_prediction_interval_wider_than_confidence(self) -> None:
        """Test prediction interval is wider than confidence interval.

        PI accounts for observation noise, so it should always be >= CI.
        """
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])
        residuals = np.random.normal(0, 0.5, 50)

        # Confidence interval
        _, ci_lower, ci_upper = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.95
        )
        ci_width = np.mean(ci_upper - ci_lower)

        # Prediction interval
        _, pi_lower, pi_upper = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals, confidence=0.95
        )
        pi_width = np.mean(pi_upper - pi_lower)

        assert pi_width >= ci_width, (
            "Prediction interval should be >= confidence interval"
        )

    def test_prediction_interval_with_zero_residuals(self) -> None:
        """Test PI equals CI when residuals are zero."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])
        residuals = np.zeros(50)

        # With zero residuals, PI should approximately equal CI
        _, ci_lower, ci_upper = compute_uncertainty_band(
            linear_model, x, popt, pcov, confidence=0.95
        )

        _, pi_lower, pi_upper = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals, confidence=0.95
        )

        # Allow small tolerance due to t-distribution vs normal
        np.testing.assert_allclose(pi_lower, ci_lower, rtol=0.1)
        np.testing.assert_allclose(pi_upper, ci_upper, rtol=0.1)

    def test_prediction_interval_larger_residuals_wider_band(self) -> None:
        """Test larger residuals produce wider prediction intervals."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])

        # Small residuals
        residuals_small = np.random.normal(0, 0.1, 50)
        _, pi_lower_small, pi_upper_small = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals_small
        )
        width_small = np.mean(pi_upper_small - pi_lower_small)

        # Large residuals
        residuals_large = np.random.normal(0, 1.0, 50)
        _, pi_lower_large, pi_upper_large = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals_large
        )
        width_large = np.mean(pi_upper_large - pi_lower_large)

        assert width_large > width_small

    def test_prediction_interval_confidence_level(self) -> None:
        """Test higher confidence gives wider prediction intervals."""
        x = np.linspace(0, 10, 50)
        popt = np.array([2.0, 1.0])
        pcov = np.diag([0.1, 0.05])
        residuals = np.random.normal(0, 0.5, 50)

        _, pi_lower_90, pi_upper_90 = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals, confidence=0.90
        )
        width_90 = np.mean(pi_upper_90 - pi_lower_90)

        _, pi_lower_99, pi_upper_99 = compute_prediction_interval(
            linear_model, x, popt, pcov, residuals, confidence=0.99
        )
        width_99 = np.mean(pi_upper_99 - pi_lower_99)

        assert width_99 > width_90

    def test_prediction_interval_single_exp_model(self) -> None:
        """Test prediction interval for exponential model."""
        x = np.logspace(-3, 2, 50)
        popt = np.array([1.0, 1.0, 0.3])  # tau, baseline, contrast
        pcov = np.diag([0.01, 0.001, 0.01])

        # Generate realistic residuals from the model
        y_true = single_exp_model(x, *popt)
        y_noisy = y_true + np.random.normal(0, 0.01, len(x))
        residuals = y_noisy - y_true

        y_fit, pi_lower, pi_upper = compute_prediction_interval(
            single_exp_model, x, popt, pcov, residuals
        )

        # Check shapes
        assert y_fit.shape == x.shape
        assert np.all(pi_lower <= y_fit)
        assert np.all(y_fit <= pi_upper)

        # Check fit values match model
        expected = single_exp_model(x, *popt)
        np.testing.assert_allclose(y_fit, expected, rtol=1e-10)
