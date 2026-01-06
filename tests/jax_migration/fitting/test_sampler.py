"""Tests for NUTS sampler with NLSQ warm-start (T035).

This module tests the NumPyro NUTS sampler functionality with
NLSQ warm-start initialization.
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if NumPyro is available
try:
    import arviz as az
    import jax
    import numpyro

    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False


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


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestSamplerWarmStart:
    """Tests for NUTS sampler with NLSQ warm-start."""

    def test_sampler_functions_can_be_imported(self) -> None:
        """Test sampler functions can be imported."""
        from xpcsviewer.fitting.sampler import (
            run_double_exp_fit,
            run_power_law_fit,
            run_single_exp_fit,
            run_stretched_exp_fit,
        )

        assert callable(run_single_exp_fit)
        assert callable(run_double_exp_fit)
        assert callable(run_stretched_exp_fit)
        assert callable(run_power_law_fit)

    def test_single_exp_fit_returns_fit_result(self) -> None:
        """Test single exponential fit returns FitResult."""
        from xpcsviewer.fitting.results import FitResult
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        # Use minimal samples for testing
        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        assert isinstance(result, FitResult)
        assert "tau" in result.samples
        assert "baseline" in result.samples
        assert "contrast" in result.samples

    def test_single_exp_fit_recovers_true_parameters(self) -> None:
        """Test single exponential fit recovers true parameters."""
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        true_tau = 1.0
        true_baseline = 1.0
        true_contrast = 0.3

        x, y, yerr = generate_single_exp_data(
            tau=true_tau, baseline=true_baseline, contrast=true_contrast
        )

        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=100,
            num_samples=200,
            num_chains=1,
            random_seed=42,
        )

        # Check posterior means are close to true values
        tau_mean = result.get_mean("tau")
        baseline_mean = result.get_mean("baseline")
        contrast_mean = result.get_mean("contrast")

        # Allow reasonable tolerance for MCMC estimation
        assert np.abs(tau_mean - true_tau) < 0.3
        assert np.abs(baseline_mean - true_baseline) < 0.1
        assert np.abs(contrast_mean - true_contrast) < 0.1

    def test_fit_result_has_diagnostics(self) -> None:
        """Test FitResult includes convergence diagnostics."""
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        assert result.diagnostics is not None
        assert "tau" in result.diagnostics.r_hat
        assert "tau" in result.diagnostics.ess_bulk
        assert isinstance(result.diagnostics.divergences, int)

    def test_fit_result_has_nlsq_init(self) -> None:
        """Test FitResult includes NLSQ warm-start values."""
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        assert result.nlsq_init is not None
        assert "tau" in result.nlsq_init
        assert "baseline" in result.nlsq_init
        assert "contrast" in result.nlsq_init

    def test_fit_result_has_arviz_data(self) -> None:
        """Test FitResult includes ArviZ InferenceData."""
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        assert result.arviz_data is not None
        assert hasattr(result.arviz_data, "posterior")

    def test_stretched_exp_fit_has_beta(self) -> None:
        """Test stretched exponential fit includes beta parameter."""
        from xpcsviewer.fitting.sampler import run_stretched_exp_fit

        # Generate stretched exponential data
        np.random.seed(42)
        x = np.logspace(-3, 2, 50)
        y_true = 1.0 + 0.3 * np.exp(-np.power(2 * x / 1.0, 0.8))
        y = y_true + np.random.normal(0, 0.01, 50)
        yerr = np.full(50, 0.01)

        result = run_stretched_exp_fit(
            x,
            y,
            yerr,
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            random_seed=42,
        )

        assert "beta" in result.samples
        beta_mean = result.get_mean("beta")
        # Beta should be in (0, 1)
        assert 0 < beta_mean < 1


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestSamplerConfig:
    """Tests for sampler configuration."""

    def test_custom_config_is_used(self) -> None:
        """Test custom sampler configuration is used."""
        from xpcsviewer.fitting.results import SamplerConfig
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        config = SamplerConfig(
            num_warmup=30,
            num_samples=60,
            num_chains=1,
            target_accept_prob=0.9,
            random_seed=123,
        )

        result = run_single_exp_fit(x, y, yerr, sampler_config=config)

        # Check samples have expected count (num_samples)
        assert result.samples["tau"].shape[0] == 60

    def test_kwargs_config_is_used(self) -> None:
        """Test kwargs-based configuration is used."""
        from xpcsviewer.fitting.sampler import run_single_exp_fit

        x, y, yerr = generate_single_exp_data()

        result = run_single_exp_fit(
            x,
            y,
            yerr,
            num_warmup=25,
            num_samples=50,
            num_chains=1,
            random_seed=456,
        )

        # Check samples have expected count
        assert result.samples["tau"].shape[0] == 50


@pytest.mark.skipif(not NUMPYRO_AVAILABLE, reason="NumPyro not installed")
class TestPublicAPI:
    """Tests for public API functions."""

    def test_fit_single_exp_api(self) -> None:
        """Test fit_single_exp public API."""
        from xpcsviewer.fitting import fit_single_exp

        x, y, yerr = generate_single_exp_data()

        result = fit_single_exp(
            x,
            y,
            yerr,
            num_warmup=30,
            num_samples=50,
            num_chains=1,
            random_seed=42,
        )

        assert "tau" in result.samples

    def test_fit_stretched_exp_api(self) -> None:
        """Test fit_stretched_exp public API."""
        from xpcsviewer.fitting import fit_stretched_exp

        np.random.seed(42)
        x = np.logspace(-3, 2, 50)
        y = 1.0 + 0.3 * np.exp(-np.power(2 * x / 1.0, 0.8))
        y += np.random.normal(0, 0.01, 50)
        yerr = np.full(50, 0.01)

        result = fit_stretched_exp(
            x,
            y,
            yerr,
            num_warmup=30,
            num_samples=50,
            num_chains=1,
            random_seed=42,
        )

        assert "beta" in result.samples

    def test_nlsq_fit_api(self) -> None:
        """Test nlsq_fit public API."""
        from xpcsviewer.fitting import nlsq_fit
        from xpcsviewer.fitting.models import single_exp_func

        x, y, yerr = generate_single_exp_data()

        p0 = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
        bounds = {
            "tau": (1e-6, 1e6),
            "baseline": (0.0, 2.0),
            "contrast": (0.0, 1.0),
        }

        result = nlsq_fit(single_exp_func, x, y, yerr, p0, bounds)

        assert "tau" in result.params
        assert result.converged
