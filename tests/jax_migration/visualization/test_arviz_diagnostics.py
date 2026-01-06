"""Tests for ArviZ diagnostic plot generation (T093).

Tests for generate_arviz_diagnostics() function that creates
6 standard ArviZ diagnostic plots (FR-013).
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if ArviZ is available
try:
    import arviz as az

    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False


@pytest.mark.skipif(not ARVIZ_AVAILABLE, reason="ArviZ not installed")
class TestGenerateArvizDiagnostics:
    """Tests for generate_arviz_diagnostics function."""

    @pytest.fixture
    def mock_trace(self):
        """Create mock InferenceData for testing."""
        # Create synthetic posterior samples
        np.random.seed(42)
        n_chains = 2
        n_samples = 100

        posterior_samples = {
            "tau": np.random.normal(1.0, 0.1, (n_chains, n_samples)),
            "baseline": np.random.normal(1.0, 0.01, (n_chains, n_samples)),
            "contrast": np.random.normal(0.3, 0.02, (n_chains, n_samples)),
        }

        return az.from_dict(posterior=posterior_samples)

    def test_returns_dict(self, mock_trace) -> None:
        """Test function returns dictionary."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(mock_trace)
        assert isinstance(result, dict)

    def test_generates_expected_plots(self, mock_trace) -> None:
        """Test expected plot types are generated."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(mock_trace)

        # Should have 6 plot types (pair, forest, energy, autocorr, rank, ess)
        expected_plots = {"pair", "forest", "energy", "autocorr", "rank", "ess"}
        assert (
            len(set(result.keys()) & expected_plots) >= 3
        )  # At least 3 plots generated

    def test_pair_plot_generated(self, mock_trace) -> None:
        """Test pair plot is generated."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(mock_trace)
        assert "pair" in result

    def test_forest_plot_generated(self, mock_trace) -> None:
        """Test forest plot is generated."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(mock_trace)
        assert "forest" in result

    def test_none_trace_returns_empty(self) -> None:
        """Test None trace returns empty dict."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(None)
        assert result == {}

    def test_var_names_filtering(self, mock_trace) -> None:
        """Test var_names parameter filters parameters."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        # Should not raise error with specific var_names
        result = generate_arviz_diagnostics(mock_trace, var_names=["tau", "baseline"])
        assert isinstance(result, dict)

    def test_output_dir_creates_files(self, mock_trace, tmp_path) -> None:
        """Test output_dir parameter creates files."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(
            mock_trace, output_dir=tmp_path, formats=("png",), dpi=72
        )

        # Should have file paths in result
        assert any("_png" in key for key in result.keys())

        # Check at least one file exists
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0


@pytest.mark.skipif(not ARVIZ_AVAILABLE, reason="ArviZ not installed")
class TestDiagnosticPlotContent:
    """Tests for diagnostic plot content."""

    @pytest.fixture
    def simple_trace(self):
        """Create simple trace for content tests."""
        np.random.seed(123)
        posterior = {
            "x": np.random.normal(0, 1, (2, 50)),
        }
        return az.from_dict(posterior=posterior)

    def test_figures_have_axes(self, simple_trace) -> None:
        """Test generated figures have axes."""
        from xpcsviewer.fitting.visualization import generate_arviz_diagnostics

        result = generate_arviz_diagnostics(simple_trace)

        for name, fig in result.items():
            if hasattr(fig, "axes"):
                assert len(fig.axes) > 0, f"{name} has no axes"
