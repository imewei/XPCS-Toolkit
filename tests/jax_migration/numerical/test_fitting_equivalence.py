"""Tests for fitting equivalence between new and reference implementations (T054).

This module tests that the new Bayesian fitting module produces results
equivalent to the reference scipy.optimize.curve_fit implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.models import (
    double_exp_func,
    power_law_func,
    single_exp_func,
    stretched_exp_func,
)


def generate_single_exp_data(
    tau: float = 1.0,
    baseline: float = 1.0,
    contrast: float = 0.3,
    n_points: int = 50,
    noise: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic single exponential G2 data."""
    np.random.seed(seed)
    x = np.logspace(-3, 2, n_points)
    y_true = baseline + contrast * np.exp(-2 * x / tau)
    y = y_true + np.random.normal(0, noise, n_points)
    yerr = np.full(n_points, noise)
    return x, y, yerr


class TestModelFunctionEquivalence:
    """Test that model functions produce equivalent results."""

    def test_single_exp_func_matches_reference(self) -> None:
        """Test single exponential function matches reference implementation."""
        x = np.logspace(-3, 2, 50)
        tau, baseline, contrast = 1.0, 1.0, 0.3

        # New implementation
        y_new = single_exp_func(x, tau, baseline, contrast)

        # Reference: baseline + contrast * exp(-2 * x / tau)
        y_ref = baseline + contrast * np.exp(-2 * x / tau)

        np.testing.assert_allclose(y_new, y_ref, rtol=1e-10)

    def test_stretched_exp_func_matches_reference(self) -> None:
        """Test stretched exponential function matches reference."""
        x = np.logspace(-3, 2, 50)
        tau, baseline, contrast, beta = 1.0, 1.0, 0.3, 0.8

        # New implementation
        y_new = stretched_exp_func(x, tau, baseline, contrast, beta)

        # Reference: baseline + contrast * exp(-(2*x/tau)^beta)
        y_ref = baseline + contrast * np.exp(-np.power(2 * x / tau, beta))

        np.testing.assert_allclose(y_new, y_ref, rtol=1e-10)

    def test_double_exp_func_matches_reference(self) -> None:
        """Test double exponential function matches reference."""
        x = np.logspace(-3, 2, 50)
        tau1, tau2, baseline, contrast1, contrast2 = 0.1, 10.0, 1.0, 0.15, 0.15

        # New implementation
        y_new = double_exp_func(x, tau1, tau2, baseline, contrast1, contrast2)

        # Reference: baseline + c1*exp(-2x/tau1) + c2*exp(-2x/tau2)
        y_ref = (
            baseline
            + contrast1 * np.exp(-2 * x / tau1)
            + contrast2 * np.exp(-2 * x / tau2)
        )

        np.testing.assert_allclose(y_new, y_ref, rtol=1e-10)

    def test_power_law_func_matches_reference(self) -> None:
        """Test power law function matches reference."""
        q = np.array([0.01, 0.05, 0.1, 0.5, 1.0])
        tau0, alpha = 1.0, 2.0

        # New implementation
        tau_new = power_law_func(q, tau0, alpha)

        # Reference: tau0 * q^(-alpha)
        tau_ref = tau0 * np.power(q, -alpha)

        np.testing.assert_allclose(tau_new, tau_ref, rtol=1e-10)


class TestNLSQEquivalence:
    """Test NLSQ fitting equivalence with scipy.optimize.curve_fit."""

    def test_nlsq_matches_scipy_single_exp(self) -> None:
        """Test NLSQ fit matches scipy curve_fit for single exponential."""
        from scipy.optimize import curve_fit

        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data(tau=1.0, baseline=1.0, contrast=0.3)

        # Reference scipy implementation
        def ref_model(x, tau, baseline, contrast):
            return baseline + contrast * np.exp(-2 * x / tau)

        popt_scipy, _ = curve_fit(
            ref_model,
            x,
            y,
            p0=[1.0, 1.0, 0.3],
            sigma=yerr,
            bounds=([1e-6, 0.0, 0.0], [1e6, 2.0, 1.0]),
            absolute_sigma=True,
        )

        # New NLSQ implementation
        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Compare results - should be very close
        np.testing.assert_allclose(result.params["tau"], popt_scipy[0], rtol=0.1)
        np.testing.assert_allclose(result.params["baseline"], popt_scipy[1], rtol=0.1)
        np.testing.assert_allclose(result.params["contrast"], popt_scipy[2], rtol=0.1)

    def test_nlsq_recovers_true_parameters(self) -> None:
        """Test NLSQ fitting recovers true parameters."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        true_tau = 1.5
        true_baseline = 1.0
        true_contrast = 0.25

        x, y, yerr = generate_single_exp_data(
            tau=true_tau, baseline=true_baseline, contrast=true_contrast, seed=123
        )

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Check recovered values are close to true values
        assert np.abs(result.params["tau"] - true_tau) < 0.2
        assert np.abs(result.params["baseline"] - true_baseline) < 0.05
        assert np.abs(result.params["contrast"] - true_contrast) < 0.05


class TestChiSquaredEquivalence:
    """Test chi-squared calculation equivalence."""

    def test_chi_squared_calculation(self) -> None:
        """Test chi-squared calculation matches manual calculation."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Calculate chi-squared manually
        y_fit = single_exp_func(
            x,
            result.params["tau"],
            result.params["baseline"],
            result.params["contrast"],
        )
        residuals = (y - y_fit) / yerr
        chi_sq_manual = np.sum(residuals**2)

        # Allow some tolerance due to different calculation methods
        # (reduced chi-squared vs total chi-squared)
        assert result.chi_squared > 0
        # The chi_squared in result should be related to the residuals
        # Result stores reduced chi-squared
        dof = len(x) - 3  # 3 parameters
        reduced_chi_sq_manual = chi_sq_manual / dof
        np.testing.assert_allclose(result.chi_squared, reduced_chi_sq_manual, rtol=0.5)


class TestCovarianceEquivalence:
    """Test covariance matrix equivalence."""

    def test_covariance_shape(self) -> None:
        """Test covariance matrix has correct shape."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Covariance should be (n_params, n_params)
        assert result.covariance.shape == (3, 3)

    def test_covariance_is_symmetric(self) -> None:
        """Test covariance matrix is symmetric."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Covariance should be symmetric
        np.testing.assert_allclose(result.covariance, result.covariance.T, rtol=1e-6)

    def test_covariance_diagonal_is_positive(self) -> None:
        """Test covariance diagonal (variances) are positive."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # Diagonal elements (variances) should be positive
        assert np.all(np.diag(result.covariance) >= 0)
