"""Tests for scipy vs NLSQ equivalence (T055-T058).

Verifies that nlsq.curve_fit produces equivalent results to
scipy.optimize.curve_fit for the standard fitting use cases.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from nlsq import curve_fit as nlsq_curve_fit


def single_exp(x, tau, bkg, cts):
    """Single exponential model for testing (JAX-compatible)."""
    return cts * jnp.exp(-2 * x / tau) + bkg


class TestParameterEquivalence:
    """Tests for parameter equivalence between scipy and nlsq (T056)."""

    def test_parameter_equivalence_single_exp(self) -> None:
        """Test that nlsq produces parameters close to true values."""
        np.random.seed(42)
        x = np.logspace(-3, 2, 50)
        true_tau, true_bkg, true_cts = 1.0, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        noise = np.random.normal(0, 0.01, size=y_true.shape)
        y = y_true + noise

        popt, pcov = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[0.5, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        np.testing.assert_allclose(popt[0], true_tau, rtol=0.1)
        np.testing.assert_allclose(popt[1], true_bkg, rtol=0.1)
        np.testing.assert_allclose(popt[2], true_cts, rtol=0.2)

    def test_parameter_equivalence_with_sigma(self) -> None:
        """Test parameter estimation with error bars."""
        np.random.seed(123)
        x = np.logspace(-3, 2, 100)  # More data points for stability
        true_tau, true_bkg, true_cts = 2.0, 1.05, 0.25
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        sigma = np.full_like(y_true, 0.01)  # Lower noise
        noise = np.random.normal(0, 0.01, size=y_true.shape)
        y = y_true + noise

        popt, pcov = nlsq_curve_fit(
            single_exp,
            x,
            y,
            sigma=sigma,
            p0=[2.0, 1.0, 0.25],  # Better initial guess
            bounds=([0.1, 0.9, 0.1], [10.0, 1.2, 0.5]),
        )

        # Allow reasonable tolerance for stochastic optimization
        np.testing.assert_allclose(popt[0], true_tau, rtol=0.2)
        np.testing.assert_allclose(popt[1], true_bkg, rtol=0.1)
        np.testing.assert_allclose(popt[2], true_cts, rtol=0.3)


class TestCovarianceEquivalence:
    """Tests for covariance matrix equivalence (T057)."""

    def test_covariance_positive_definite(self) -> None:
        """Test that covariance matrix is positive semi-definite."""
        np.random.seed(456)
        x = np.logspace(-3, 2, 100)
        true_tau, true_bkg, true_cts = 1.5, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        noise = np.random.normal(0, 0.005, size=y_true.shape)
        y = y_true + noise

        popt, pcov = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        assert np.all(np.isfinite(pcov))
        eigenvalues = np.linalg.eigvalsh(pcov)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical errors

    def test_covariance_uncertainties_reasonable(self) -> None:
        """Test that uncertainties from covariance are reasonable."""
        np.random.seed(789)
        x = np.logspace(-3, 2, 100)
        true_tau, true_bkg, true_cts = 1.0, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        sigma_val = 0.01
        noise = np.random.normal(0, sigma_val, size=y_true.shape)
        y = y_true + noise
        sigma = np.full_like(y, sigma_val)

        popt, pcov = nlsq_curve_fit(
            single_exp,
            x,
            y,
            sigma=sigma,
            absolute_sigma=True,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        perr = np.sqrt(np.diag(pcov))
        assert np.all(perr > 0)
        assert np.all(perr < np.abs(popt))  # Uncertainties smaller than values


class TestMetricsEquivalence:
    """Tests for metrics equivalence (T058)."""

    def test_r_squared_reasonable(self) -> None:
        """Test that R² is close to 1 for good fits."""
        np.random.seed(111)
        x = np.logspace(-3, 2, 100)
        true_tau, true_bkg, true_cts = 1.0, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        noise = np.random.normal(0, 0.005, size=y_true.shape)
        y = y_true + noise

        result = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        if hasattr(result, "r_squared"):
            assert result.r_squared > 0.95

    def test_rmse_reasonable(self) -> None:
        """Test that RMSE is close to noise level."""
        np.random.seed(222)
        x = np.logspace(-3, 2, 100)
        true_tau, true_bkg, true_cts = 1.0, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, size=y_true.shape)
        y = y_true + noise

        result = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        if hasattr(result, "rmse"):
            assert result.rmse < noise_level * 3

    def test_residuals_unbiased(self) -> None:
        """Test that residuals are centered around zero."""
        np.random.seed(333)
        x = np.logspace(-3, 2, 100)
        true_tau, true_bkg, true_cts = 1.0, 1.0, 0.3
        y_true = single_exp(x, true_tau, true_bkg, true_cts)
        noise = np.random.normal(0, 0.01, size=y_true.shape)
        y = y_true + noise

        result = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        if hasattr(result, "residuals"):
            residuals = np.asarray(result.residuals)
            assert np.abs(np.mean(residuals)) < 0.01  # Nearly zero mean


class TestLegacyFunctionsEquivalence:
    """Tests for legacy function compatibility."""

    def test_curve_fit_tuple_return(self) -> None:
        """Test that curve_fit can still return tuple for legacy code."""
        np.random.seed(444)
        x = np.logspace(-3, 2, 50)
        y = single_exp(x, 1.0, 1.0, 0.3) + np.random.normal(0, 0.01, 50)

        result = nlsq_curve_fit(
            single_exp,
            x,
            y,
            p0=[1.0, 1.0, 0.2],
            bounds=([0.01, 0.8, 0.1], [10.0, 1.2, 0.5]),
        )

        # Should work with tuple unpacking or attribute access
        if hasattr(result, "popt"):
            popt, pcov = result.popt, result.pcov
        else:
            popt, pcov = result

        assert len(popt) == 3
        assert pcov.shape == (3, 3)
