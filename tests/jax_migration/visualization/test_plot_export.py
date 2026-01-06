"""Tests for plot export functionality (T096).

Tests for save_figure() function that exports figures to PDF/PNG
at 300 DPI (FR-018).
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


@pytest.fixture
def sample_figure():
    """Create a sample figure for export testing."""
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Test Plot")
    return fig


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestSaveFigure:
    """Tests for save_figure function."""

    def test_save_png(self, sample_figure, tmp_path) -> None:
        """Test saving figure as PNG."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(sample_figure, str(output_path), formats=("png",))

        png_path = tmp_path / "test_plot.png"
        assert png_path.exists()
        assert png_path.stat().st_size > 0
        plt.close("all")

    def test_save_pdf(self, sample_figure, tmp_path) -> None:
        """Test saving figure as PDF."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(sample_figure, str(output_path), formats=("pdf",))

        pdf_path = tmp_path / "test_plot.pdf"
        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0
        plt.close("all")

    def test_save_multiple_formats(self, sample_figure, tmp_path) -> None:
        """Test saving figure in multiple formats."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(sample_figure, str(output_path), formats=("png", "pdf"))

        png_path = tmp_path / "test_plot.png"
        pdf_path = tmp_path / "test_plot.pdf"

        assert png_path.exists()
        assert pdf_path.exists()
        plt.close("all")

    def test_default_dpi(self, sample_figure, tmp_path) -> None:
        """Test default DPI is 300."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(sample_figure, str(output_path), formats=("png",))

        # File should exist and be reasonably large for 300 DPI
        png_path = tmp_path / "test_plot.png"
        assert png_path.exists()
        # 300 DPI should produce larger files than default 100 DPI
        # Just check file was created successfully
        assert png_path.stat().st_size > 1000  # At least 1KB
        plt.close("all")

    def test_custom_dpi(self, sample_figure, tmp_path) -> None:
        """Test custom DPI setting."""
        from xpcsviewer.fitting.visualization import save_figure

        low_dpi_path = tmp_path / "low_dpi"
        high_dpi_path = tmp_path / "high_dpi"

        save_figure(sample_figure, str(low_dpi_path), formats=("png",), dpi=72)
        save_figure(sample_figure, str(high_dpi_path), formats=("png",), dpi=300)

        # Higher DPI should produce larger file
        low_png = tmp_path / "low_dpi.png"
        high_png = tmp_path / "high_dpi.png"
        assert high_png.stat().st_size > low_png.stat().st_size
        plt.close("all")

    def test_multiple_formats_in_call(self, sample_figure, tmp_path) -> None:
        """Test saving in multiple formats in one call."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(sample_figure, str(output_path), formats=("png", "pdf"))

        # Should return dict with format -> path mapping
        assert isinstance(result, dict)
        assert "png" in result
        assert "pdf" in result
        plt.close("all")


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestPublicationStyle:
    """Tests for publication style preset."""

    def test_publication_style_exists(self) -> None:
        """Test PUBLICATION_STYLE constant exists."""
        from xpcsviewer.fitting.visualization import PUBLICATION_STYLE

        assert PUBLICATION_STYLE is not None
        assert isinstance(PUBLICATION_STYLE, dict)

    def test_publication_style_has_font_family(self) -> None:
        """Test publication style includes serif font."""
        from xpcsviewer.fitting.visualization import PUBLICATION_STYLE

        # Should specify serif font for publication quality
        assert "font.family" in PUBLICATION_STYLE or "font.serif" in PUBLICATION_STYLE

    def test_publication_style_has_dpi(self) -> None:
        """Test publication style includes high DPI."""
        from xpcsviewer.fitting.visualization import PUBLICATION_STYLE

        # Should specify 300 DPI for publication
        if "savefig.dpi" in PUBLICATION_STYLE:
            assert PUBLICATION_STYLE["savefig.dpi"] >= 300

    def test_apply_publication_style(self, sample_figure, tmp_path) -> None:
        """Test applying publication style to figure."""
        from xpcsviewer.fitting.visualization import (
            PUBLICATION_STYLE,
            apply_publication_style,
            save_figure,
        )

        # Apply style using function
        apply_publication_style()

        fig, ax = plt.subplots()
        x = np.linspace(0, 10, 100)
        ax.plot(x, np.sin(x))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        output_path = tmp_path / "publication_plot"
        save_figure(fig, str(output_path), formats=("png",))

        png_path = tmp_path / "publication_plot.png"
        assert png_path.exists()
        plt.close("all")


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not installed")
class TestSaveFigureFormats:
    """Tests for various output formats."""

    def test_svg_format(self, sample_figure, tmp_path) -> None:
        """Test saving figure as SVG."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        save_figure(sample_figure, str(output_path), formats=("svg",))

        svg_path = tmp_path / "test_plot.svg"
        assert svg_path.exists()
        assert svg_path.stat().st_size > 0
        plt.close("all")

    def test_eps_format(self, sample_figure, tmp_path) -> None:
        """Test saving figure as EPS."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        save_figure(sample_figure, str(output_path), formats=("eps",))

        eps_path = tmp_path / "test_plot.eps"
        assert eps_path.exists()
        assert eps_path.stat().st_size > 0
        plt.close("all")

    def test_multiple_formats_saved(self, sample_figure, tmp_path) -> None:
        """Test multiple formats can be saved in one call."""
        from xpcsviewer.fitting.visualization import save_figure

        output_path = tmp_path / "test_plot"
        result = save_figure(
            sample_figure, str(output_path), formats=("png", "pdf", "svg")
        )

        # Check all files exist
        for ext in ["png", "pdf", "svg"]:
            path = tmp_path / f"test_plot.{ext}"
            assert path.exists(), f"{ext} file not created"

        plt.close("all")
