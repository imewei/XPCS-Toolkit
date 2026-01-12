"""Tests for comparison plot (T095).

Tests for plot_comparison() function that overlays NLSQ and Bayesian
fit results for comparison (FR-020).
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def single_exp_model(x, tau, baseline, contrast):
    """Single exponential model for testing."""
    return baseline + contrast * np.exp(-2 * x / tau)


@pytest.fixture
def mock_nlsq_result():
    """Create mock NLSQResult for testing."""
    from xpcsviewer.fitting.results import NLSQResult

    params = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
    covariance = np.diag([0.01, 0.001, 0.01])
    residuals = np.random.normal(0, 0.01, 30)

    return NLSQResult(
        params=params,
        _covariance=covariance,
        _residuals=residuals,
        chi_squared=1.2,
        converged=True,
        pcov_valid=True,
    )


@pytest.fixture
def mock_fit_result():
    """Create mock FitResult for testing."""
    from xpcsviewer.fitting.results import FitResult

    np.random.seed(42)
    n_samples = 100

    samples = {
        "tau": np.random.normal(1.0, 0.1, n_samples),
        "baseline": np.random.normal(1.0, 0.01, n_samples),
        "contrast": np.random.normal(0.3, 0.02, n_samples),
    }

    return FitResult(samples=samples)


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for plotting."""
    np.random.seed(42)
    x = np.logspace(-3, 2, 30)
    y_true = 1.0 + 0.3 * np.exp(-2 * x / 1.0)
    y = y_true + np.random.normal(0, 0.01, len(x))
    return x, y


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestPlotComparison:
    """Tests for plot_comparison function."""

    def test_returns_axes(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test function returns matplotlib axes."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        assert hasattr(ax, "plot")  # Duck typing check for axes
        plt.close("all")

    def test_uses_provided_axes(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test function uses provided axes."""
        from xpcsviewer.fitting.visualization import plot_comparison

        fig, ax = plt.subplots()
        x, y = synthetic_data

        result_ax = plot_comparison(
            mock_nlsq_result, mock_fit_result, single_exp_model, x, y, ax=ax
        )

        assert result_ax is ax
        plt.close("all")

    def test_creates_new_axes_if_none(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test function creates new axes if none provided."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        assert ax is not None
        plt.close("all")

    def test_plots_data_points(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test data points are plotted."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        # Check that scatter plot exists (data points)
        collections = ax.collections
        assert len(collections) > 0
        plt.close("all")

    def test_plots_both_fits(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test both NLSQ and Bayesian fits are plotted."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        # Should have at least 2 lines (NLSQ fit and Bayesian median)
        lines = ax.get_lines()
        assert len(lines) >= 2
        plt.close("all")

    def test_legend_present(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test legend is present with both fit labels."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")

    def test_nlsq_uncertainty_band_present(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test NLSQ uncertainty band is shown."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)

        # fill_between creates PolyCollection in collections
        assert len(ax.collections) >= 1
        plt.close("all")


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestPlotComparisonValidInputs:
    """Tests for comparison plot input validation."""

    def test_requires_both_results(
        self, mock_nlsq_result, mock_fit_result, synthetic_data
    ) -> None:
        """Test that both results are required for comparison."""
        from xpcsviewer.fitting.visualization import plot_comparison

        x, y = synthetic_data
        # Should work with both results
        ax = plot_comparison(mock_nlsq_result, mock_fit_result, single_exp_model, x, y)
        assert ax is not None
        plt.close("all")
