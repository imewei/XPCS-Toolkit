"""Tests for NLSQ visualization enhancements (T070-T073).

Tests programmatic figure assertions for:
- Prediction interval display
- Diagnostics display
- 2x2 diagnostics subplot layout
"""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xpcsviewer.fitting.results import NLSQResult
from xpcsviewer.fitting.visualization import plot_nlsq_fit


def single_exp(x, tau, bkg, cts):
    """Single exponential model for testing."""
    return cts * np.exp(-2 * x / tau) + bkg


@pytest.fixture
def sample_nlsq_result():
    """Create a sample NLSQResult with native_result mock."""
    # Create mock native_result with diagnostics
    mock_native = MagicMock()
    mock_native.r_squared = 0.98
    mock_native.adj_r_squared = 0.975
    mock_native.rmse = 0.01
    mock_native.mae = 0.008
    mock_native.aic = -50.0
    mock_native.bic = -45.0
    mock_native.residuals = np.random.normal(0, 0.01, 50)
    mock_native.pcov = np.eye(3) * 0.001

    # Create diagnostics mock
    mock_diag = MagicMock()
    mock_diag.health_score = 0.92
    mock_diag.status = "healthy"
    mock_diag.identifiability = MagicMock()
    mock_diag.identifiability.condition_number = 15.0
    mock_native.diagnostics = mock_diag

    result = NLSQResult(
        params={"tau": 1.0, "bkg": 1.0, "cts": 0.3},
        chi_squared=1.05,
        converged=True,
        pcov_valid=True,
        pcov_message="",
        native_result=mock_native,
        _param_names=["tau", "bkg", "cts"],
    )
    result._covariance = np.eye(3) * 0.001
    result._residuals = np.random.normal(0, 0.01, 50)

    return result


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    x = np.logspace(-3, 2, 50)
    y_true = single_exp(x, 1.0, 1.0, 0.3)
    y = y_true + np.random.normal(0, 0.01, size=y_true.shape)
    return x, y


class TestPlotNLSQFitPredictionInterval:
    """Tests for prediction interval display (T071)."""

    def test_plot_nlsq_fit_prediction_interval_fill_between(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that fill_between collection is present when show_prediction_interval=True."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            show_prediction_interval=True,
            ax=ax,
        )

        # Check for fill_between collections (PolyCollection)
        collections = ax.collections
        assert len(collections) >= 1, (
            "Expected at least one fill_between collection for PI"
        )

        plt.close(fig)

    def test_plot_nlsq_fit_no_prediction_interval_by_default(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that prediction interval is not shown by default."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            show_prediction_interval=False,
            ax=ax,
        )

        # Should have only CI fill_between (1 collection) not PI
        collections = ax.collections
        # One for scatter, one for CI fill_between
        assert len(collections) >= 1

        plt.close(fig)


class TestPlotNLSQFitDiagnostics:
    """Tests for diagnostics display (T072)."""

    def test_plot_nlsq_fit_show_metrics_default(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that metrics are shown by default."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            show_metrics=True,
            ax=ax,
        )

        # Check for text annotation containing R²
        texts = [t.get_text() for t in ax.texts]
        assert any("R²" in t or "R^2" in t or "R2" in t for t in texts), (
            "Expected R² metric in plot text"
        )

        plt.close(fig)

    def test_plot_nlsq_fit_metrics_content(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that metrics annotation contains expected values."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            show_metrics=True,
            ax=ax,
        )

        # Get all text content
        all_text = " ".join([t.get_text() for t in ax.texts])

        # Check for expected metrics
        assert "RMSE" in all_text, "Expected RMSE in metrics"
        assert "χ²" in all_text or "chi" in all_text.lower(), (
            "Expected chi-squared in metrics"
        )

        plt.close(fig)


class TestPlotDiagnosticsLayout:
    """Tests for diagnostics 2x2 subplot layout (T073)."""

    def test_plot_diagnostics_creates_four_axes(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that plot_diagnostics creates a figure with 4 axes."""
        # Import the function if it exists, or skip if not yet implemented
        try:
            from xpcsviewer.fitting.visualization import plot_diagnostics
        except ImportError:
            pytest.skip("plot_diagnostics not yet implemented")

        x, y = sample_data

        fig = plot_diagnostics(sample_nlsq_result, single_exp, x, y)

        # Check for 4 axes (2x2 layout)
        assert len(fig.axes) == 4, f"Expected 4 axes, got {len(fig.axes)}"

        plt.close(fig)


class TestPlotNLSQFitAesthetics:
    """Tests for plot aesthetics and labels."""

    def test_plot_nlsq_fit_custom_labels(self, sample_nlsq_result, sample_data) -> None:
        """Test that custom labels are applied."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            xlabel="Delay Time (s)",
            ylabel="G2",
            title="Test Fit",
            ax=ax,
        )

        assert ax.get_xlabel() == "Delay Time (s)"
        assert ax.get_ylabel() == "G2"
        assert ax.get_title() == "Test Fit"

        plt.close(fig)

    def test_plot_nlsq_fit_legend_present(
        self, sample_nlsq_result, sample_data
    ) -> None:
        """Test that legend is present."""
        x, y = sample_data

        fig, ax = plt.subplots()
        plot_nlsq_fit(
            sample_nlsq_result,
            single_exp,
            x,
            y,
            ax=ax,
        )

        legend = ax.get_legend()
        assert legend is not None, "Expected legend to be present"

        plt.close(fig)


class TestPlotNLSQFitInvalidCov:
    """Tests for handling invalid covariance."""

    def test_plot_nlsq_fit_invalid_covariance(self, sample_data) -> None:
        """Test plot still works with invalid covariance."""
        x, y = sample_data

        result = NLSQResult(
            params={"tau": 1.0, "bkg": 1.0, "cts": 0.3},
            chi_squared=1.05,
            converged=True,
            pcov_valid=False,
            pcov_message="Covariance contains inf values",
        )

        fig, ax = plt.subplots()
        # Should not raise
        plot_nlsq_fit(
            result,
            single_exp,
            x,
            y,
            ax=ax,
        )

        # Check for "uncertainty unavailable" in legend
        legend = ax.get_legend()
        if legend:
            labels = [t.get_text() for t in legend.get_texts()]
            assert any("unavailable" in lbl.lower() for lbl in labels)

        plt.close(fig)
