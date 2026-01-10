"""Tests for FitResult dataclass and diagnostics (T036).

This module tests the fit result dataclasses and diagnostic
functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from xpcsviewer.fitting.results import (
    FitDiagnostics,
    FitResult,
    NLSQResult,
    SamplerConfig,
)


class TestSamplerConfig:
    """Tests for SamplerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SamplerConfig()

        assert config.num_warmup == 500
        assert config.num_samples == 1000
        assert config.num_chains == 4
        assert config.target_accept_prob == 0.8
        assert config.max_tree_depth == 10
        assert config.random_seed is None

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SamplerConfig(
            num_warmup=1000,
            num_samples=2000,
            num_chains=2,
            target_accept_prob=0.9,
            max_tree_depth=8,
            random_seed=42,
        )

        assert config.num_warmup == 1000
        assert config.num_samples == 2000
        assert config.num_chains == 2
        assert config.target_accept_prob == 0.9
        assert config.max_tree_depth == 8
        assert config.random_seed == 42


class TestFitDiagnostics:
    """Tests for FitDiagnostics dataclass."""

    def test_converged_good_diagnostics(self) -> None:
        """Test converged=True for good diagnostics."""
        diag = FitDiagnostics(
            r_hat={"tau": 1.001, "baseline": 1.002},
            ess_bulk={"tau": 1000, "baseline": 1200},
            ess_tail={"tau": 800, "baseline": 900},
            divergences=0,
            max_treedepth_reached=0,
        )

        assert diag.converged is True

    def test_not_converged_high_rhat(self) -> None:
        """Test converged=False for high r_hat."""
        diag = FitDiagnostics(
            r_hat={"tau": 1.05},  # > 1.01 threshold
            ess_bulk={"tau": 1000},
            ess_tail={"tau": 800},
            divergences=0,
        )

        assert diag.converged is False

    def test_not_converged_low_ess(self) -> None:
        """Test converged=False for low ESS."""
        diag = FitDiagnostics(
            r_hat={"tau": 1.001},
            ess_bulk={"tau": 200},  # < 400 threshold
            ess_tail={"tau": 800},
            divergences=0,
        )

        assert diag.converged is False

    def test_not_converged_divergences(self) -> None:
        """Test converged=False for divergences."""
        diag = FitDiagnostics(
            r_hat={"tau": 1.001},
            ess_bulk={"tau": 1000},
            ess_tail={"tau": 800},
            divergences=5,  # > 0
        )

        assert diag.converged is False


class TestNLSQResult:
    """Tests for NLSQResult dataclass."""

    def test_creation(self) -> None:
        """Test NLSQResult creation."""
        result = NLSQResult(
            params={"tau": 1.0, "baseline": 1.0},
            chi_squared=1.0,
            converged=True,
            pcov_valid=True,
            pcov_message="Covariance matrix is valid",
        )
        # Set properties via setters for backward compat
        result.covariance = np.eye(2)
        result.residuals = np.zeros(10)

        assert result.params["tau"] == 1.0
        assert result.converged is True
        assert result.pcov_valid is True

    def test_default_pcov_values(self) -> None:
        """Test default pcov validation values."""
        result = NLSQResult(
            params={"tau": 1.0},
            chi_squared=1.0,
            converged=True,
        )
        result.covariance = np.eye(1)
        result.residuals = np.zeros(10)

        assert result.pcov_valid is True
        assert result.pcov_message == ""


class TestFitResult:
    """Tests for FitResult dataclass."""

    @pytest.fixture
    def sample_result(self) -> FitResult:
        """Create sample FitResult for testing."""
        samples = {
            "tau": np.array([0.9, 1.0, 1.1, 1.0, 1.05]),
            "baseline": np.array([0.99, 1.0, 1.01, 1.0, 1.005]),
            "contrast": np.array([0.28, 0.3, 0.32, 0.31, 0.29]),
        }
        diagnostics = FitDiagnostics(
            r_hat={"tau": 1.001, "baseline": 1.002, "contrast": 1.003},
            ess_bulk={"tau": 1000, "baseline": 1200, "contrast": 1100},
            ess_tail={"tau": 800, "baseline": 900, "contrast": 850},
            divergences=0,
        )
        return FitResult(
            samples=samples,
            diagnostics=diagnostics,
            nlsq_init={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
        )

    def test_get_mean(self, sample_result: FitResult) -> None:
        """Test get_mean method."""
        tau_mean = sample_result.get_mean("tau")
        assert np.isclose(tau_mean, np.mean([0.9, 1.0, 1.1, 1.0, 1.05]))

    def test_get_std(self, sample_result: FitResult) -> None:
        """Test get_std method."""
        tau_std = sample_result.get_std("tau")
        assert tau_std > 0
        assert np.isclose(tau_std, np.std([0.9, 1.0, 1.1, 1.0, 1.05]))

    def test_get_hdi(self, sample_result: FitResult) -> None:
        """Test get_hdi method."""
        lower, upper = sample_result.get_hdi("tau", prob=0.94)
        assert lower < upper
        assert lower >= 0.9  # Min sample
        assert upper <= 1.1  # Max sample

    def test_get_samples(self, sample_result: FitResult) -> None:
        """Test get_samples method."""
        tau_samples = sample_result.get_samples("tau")
        assert len(tau_samples) == 5
        np.testing.assert_array_equal(tau_samples, [0.9, 1.0, 1.1, 1.0, 1.05])

    def test_get_nonexistent_param_raises(self, sample_result: FitResult) -> None:
        """Test KeyError for nonexistent parameter."""
        with pytest.raises(KeyError, match="not found"):
            sample_result.get_mean("nonexistent")

    def test_to_dict(self, sample_result: FitResult) -> None:
        """Test to_dict method."""
        d = sample_result.to_dict()

        assert "samples" in d
        assert "nlsq_init" in d
        assert "diagnostics" in d
        assert "tau" in d["samples"]
        assert isinstance(d["samples"]["tau"], list)  # Converted from array

    def test_predict_placeholder(self, sample_result: FitResult) -> None:
        """Test predict method returns placeholder."""
        x = np.linspace(0, 10, 20)
        mean, std = sample_result.predict(x)

        assert len(mean) == 20
        assert len(std) == 20
        # Placeholder returns zeros
        np.testing.assert_array_equal(mean, 0)
        np.testing.assert_array_equal(std, 0)


class TestFitResultVisualizationMethods:
    """Tests for FitResult visualization convenience methods."""

    def test_plot_posterior_predictive_exists(self) -> None:
        """Test plot_posterior_predictive method exists."""
        result = FitResult(samples={"tau": np.array([1.0])})
        assert hasattr(result, "plot_posterior_predictive")

    def test_generate_diagnostics_exists(self) -> None:
        """Test generate_diagnostics method exists."""
        result = FitResult(samples={"tau": np.array([1.0])})
        assert hasattr(result, "generate_diagnostics")


class TestNLSQResultVisualizationMethods:
    """Tests for NLSQResult visualization convenience methods."""

    def test_plot_method_exists(self) -> None:
        """Test plot method exists."""
        result = NLSQResult(
            params={"tau": 1.0},
            chi_squared=1.0,
            converged=True,
        )
        result.covariance = np.eye(1)
        result.residuals = np.zeros(10)
        assert hasattr(result, "plot")
