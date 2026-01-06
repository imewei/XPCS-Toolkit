"""Tests for posterior predictive plot with 95% CI (T094).

Tests for plot_posterior_predictive() function that creates
Bayesian fit visualization with credible intervals (FR-014).
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
class TestPlotPosteriorPredictive:
    """Tests for plot_posterior_predictive function."""

    def test_returns_axes(self, mock_fit_result, synthetic_data) -> None:
        """Test function returns matplotlib axes."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(mock_fit_result, single_exp_model, x, y)

        assert hasattr(ax, "plot")  # Duck typing check for axes
        plt.close("all")

    def test_uses_provided_axes(self, mock_fit_result, synthetic_data) -> None:
        """Test function uses provided axes."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        fig, ax = plt.subplots()
        x, y = synthetic_data

        result_ax = plot_posterior_predictive(
            mock_fit_result, single_exp_model, x, y, ax=ax
        )

        assert result_ax is ax
        plt.close("all")

    def test_creates_new_axes_if_none(self, mock_fit_result, synthetic_data) -> None:
        """Test function creates new axes if none provided."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(mock_fit_result, single_exp_model, x, y)

        assert ax is not None
        plt.close("all")

    def test_plots_data_points(self, mock_fit_result, synthetic_data) -> None:
        """Test data points are plotted."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(mock_fit_result, single_exp_model, x, y)

        # Check that scatter plot exists
        collections = ax.collections
        assert len(collections) > 0  # Scatter creates PathCollection
        plt.close("all")

    def test_plots_credible_interval(self, mock_fit_result, synthetic_data) -> None:
        """Test credible interval band is plotted."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(mock_fit_result, single_exp_model, x, y)

        # Check for fill_between (creates PolyCollection)
        # fill_between adds to collections
        assert len(ax.collections) >= 1
        plt.close("all")

    def test_custom_credible_level(self, mock_fit_result, synthetic_data) -> None:
        """Test custom credible level works."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(
            mock_fit_result, single_exp_model, x, y, credible_level=0.99
        )

        assert ax is not None
        plt.close("all")

    def test_custom_n_draws(self, mock_fit_result, synthetic_data) -> None:
        """Test custom n_draws parameter works."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(
            mock_fit_result, single_exp_model, x, y, n_draws=50
        )

        assert ax is not None
        plt.close("all")

    def test_legend_present(self, mock_fit_result, synthetic_data) -> None:
        """Test legend is present."""
        from xpcsviewer.fitting.visualization import plot_posterior_predictive

        x, y = synthetic_data
        ax = plot_posterior_predictive(mock_fit_result, single_exp_model, x, y)

        legend = ax.get_legend()
        assert legend is not None
        plt.close("all")


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestFitResultPlotMethod:
    """Tests for FitResult.plot_posterior_predictive convenience method."""

    def test_method_exists(self, mock_fit_result) -> None:
        """Test method exists on FitResult."""
        assert hasattr(mock_fit_result, "plot_posterior_predictive")

    def test_method_calls_visualization(self, mock_fit_result, synthetic_data) -> None:
        """Test convenience method works."""
        x, y = synthetic_data
        ax = mock_fit_result.plot_posterior_predictive(single_exp_model, x, y)

        assert ax is not None
        plt.close("all")
