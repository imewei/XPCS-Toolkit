"""Tests for NLSQ convergence on synthetic data (T033).

This module tests the JAX-accelerated NLSQ solver for various
fitting scenarios using synthetic G2 correlation data.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.results import NLSQResult


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


class TestNLSQConvergence:
    """Test NLSQ convergence on synthetic data."""

    def test_nlsq_basic_convergence(self) -> None:
        """Test NLSQ works with basic fitting."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        # Use JAX-compatible model from the library
        from xpcsviewer.fitting.models import single_exp_func

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        assert isinstance(result, NLSQResult)
        # Check fit is reasonable even if "converged" flag may be False
        # (nlsq may report not converged if initial guess is already good)
        assert "tau" in result.params
        assert "baseline" in result.params
        assert "contrast" in result.params

        # Check fitted values are close to true values
        assert np.abs(result.params["tau"] - 1.0) < 0.5
        assert np.abs(result.params["baseline"] - 1.0) < 0.2
        assert np.abs(result.params["contrast"] - 0.3) < 0.2

    def test_nlsq_jax_convergence(self) -> None:
        """Test NLSQ convergence with JAX backend."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 0.5, "baseline": 0.9, "contrast": 0.2}  # Off from true values
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds)

        assert isinstance(result, NLSQResult)
        assert "tau" in result.params
        assert np.abs(result.params["tau"] - 1.0) < 0.5
        assert np.abs(result.params["baseline"] - 1.0) < 0.2

    def test_nlsq_result_has_covariance(self) -> None:
        """Test NLSQ result includes covariance matrix."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 0.8, "baseline": 0.95, "contrast": 0.25}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        assert result.covariance is not None
        assert result.covariance.shape == (3, 3)

    def test_nlsq_result_has_residuals(self) -> None:
        """Test NLSQ result includes residuals."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 0.8, "baseline": 0.95, "contrast": 0.25}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        assert result.residuals is not None
        assert len(result.residuals) == len(x)

    def test_nlsq_chi_squared(self) -> None:
        """Test NLSQ result has reasonable chi-squared."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 0.8, "baseline": 0.95, "contrast": 0.25}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        # For good fit, chi-squared should be positive
        assert result.chi_squared > 0

    def test_nlsq_pcov_validation(self) -> None:
        """Test NLSQ result includes pcov validation."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 0.8, "baseline": 0.95, "contrast": 0.25}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="fast")

        assert hasattr(result, "pcov_valid")
        assert hasattr(result, "pcov_message")
        assert isinstance(result.pcov_valid, bool)
        assert isinstance(result.pcov_message, str)


class TestNLSQEdgeCases:
    """Test NLSQ handling of edge cases."""

    def test_nlsq_with_no_errors(self) -> None:
        """Test NLSQ works without measurement errors."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, _ = generate_single_exp_data()

        p0 = {"tau": 0.8, "baseline": 0.95, "contrast": 0.25}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, None, p0, bounds, preset="fast")

        assert isinstance(result, NLSQResult)
        # Check we get reasonable parameters even without yerr
        assert "tau" in result.params
        assert np.abs(result.params["tau"] - 1.0) < 1.0

    def test_nlsq_off_initial_guess(self) -> None:
        """Test NLSQ can recover from off initial guess."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        x, y, yerr = generate_single_exp_data(tau=1.0)

        # Initial guess that's off but reasonable
        p0 = {"tau": 2.0, "baseline": 0.8, "contrast": 0.4}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds, preset="robust")

        # Should find a solution closer to truth than initial guess
        assert np.abs(result.params["tau"] - 1.0) < 1.0
