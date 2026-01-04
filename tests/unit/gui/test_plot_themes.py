"""Unit tests for plot_themes module."""

import pytest


class TestMatplotlibParams:
    """Tests for Matplotlib theme parameter generation."""

    def test_get_matplotlib_light_params(self, qtbot, monkeypatch, tmp_path):
        """Light theme should return light-colored matplotlib params."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()
        manager.set_theme("light")

        params = manager.get_matplotlib_params()

        # Light theme should have light backgrounds
        assert params["figure.facecolor"] == "#FFFFFF"
        assert params["axes.facecolor"] == "#FFFFFF"
        # Text should be dark on light background
        assert params["text.color"] == "#1D1D1F"

    def test_get_matplotlib_dark_params(self, qtbot, monkeypatch, tmp_path):
        """Dark theme should return dark-colored matplotlib params."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()
        manager.set_theme("dark")

        params = manager.get_matplotlib_params()

        # Dark theme should have dark backgrounds
        assert params["figure.facecolor"] == "#1C1C1E"
        assert params["axes.facecolor"] == "#1C1C1E"
        # Text should be light on dark background (uses text_primary from DARK_TOKENS)
        assert params["text.color"] == "#F5F5F7"

    def test_matplotlib_params_have_required_keys(self, qtbot, monkeypatch, tmp_path):
        """Matplotlib params should have all required rcParams keys."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()
        params = manager.get_matplotlib_params()

        required_keys = [
            "figure.facecolor",
            "axes.facecolor",
            "axes.edgecolor",
            "axes.labelcolor",
            "xtick.color",
            "ytick.color",
            "text.color",
            "grid.color",
            "legend.facecolor",
            "legend.edgecolor",
        ]

        for key in required_keys:
            assert key in params, f"Missing required key: {key}"

    def test_matplotlib_params_are_valid_colors(self, qtbot, monkeypatch, tmp_path):
        """All matplotlib param values should be valid hex colors."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()
        params = manager.get_matplotlib_params()

        for key, value in params.items():
            # All values should be hex colors
            assert isinstance(value, str), f"{key} is not a string"
            assert value.startswith("#"), f"{key} value '{value}' is not a hex color"
            assert len(value) in [4, 7, 9], f"{key} value '{value}' has invalid length"


class TestPyQtGraphIntegration:
    """Tests for PyQtGraph theme integration."""

    def test_apply_to_pyqtgraph_does_not_raise(self, qtbot, monkeypatch, tmp_path):
        """apply_to_pyqtgraph should not raise any errors."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()

        # Should not raise
        manager.apply_to_pyqtgraph()

    def test_apply_to_pyqtgraph_sets_config(self, qtbot, monkeypatch, tmp_path):
        """apply_to_pyqtgraph should set PyQtGraph config options."""
        monkeypatch.setenv("HOME", str(tmp_path))

        import pyqtgraph as pg

        from xpcsviewer.gui.theme.manager import ThemeManager

        manager = ThemeManager()
        manager.set_theme("dark")
        manager.apply_to_pyqtgraph()

        # Check that config was set (use getConfigOption for individual options)
        background = pg.getConfigOption("background")
        antialias = pg.getConfigOption("antialias")
        # Background should be set to a dark color
        assert background is not None
        # Antialias should be enabled
        assert antialias is True


class TestPlotThemesModule:
    """Tests for standalone plot_themes module functions."""

    def test_matplotlib_light_dict_exists(self):
        """MATPLOTLIB_LIGHT dict should exist and have proper structure."""
        from xpcsviewer.gui.theme.plot_themes import MATPLOTLIB_LIGHT

        assert isinstance(MATPLOTLIB_LIGHT, dict)
        assert "figure.facecolor" in MATPLOTLIB_LIGHT
        assert "axes.facecolor" in MATPLOTLIB_LIGHT

    def test_matplotlib_dark_dict_exists(self):
        """MATPLOTLIB_DARK dict should exist and have proper structure."""
        from xpcsviewer.gui.theme.plot_themes import MATPLOTLIB_DARK

        assert isinstance(MATPLOTLIB_DARK, dict)
        assert "figure.facecolor" in MATPLOTLIB_DARK
        assert "axes.facecolor" in MATPLOTLIB_DARK

    def test_light_dark_params_differ(self):
        """Light and dark params should have different values."""
        from xpcsviewer.gui.theme.plot_themes import MATPLOTLIB_DARK, MATPLOTLIB_LIGHT

        assert (
            MATPLOTLIB_LIGHT["figure.facecolor"] != MATPLOTLIB_DARK["figure.facecolor"]
        )
        assert MATPLOTLIB_LIGHT["text.color"] != MATPLOTLIB_DARK["text.color"]

    def test_get_plot_colors_light(self, qtbot, monkeypatch, tmp_path):
        """get_plot_colors should return colors dict for light theme."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.plot_themes import get_plot_colors

        colors = get_plot_colors("light")

        assert isinstance(colors, dict)
        assert "background" in colors
        assert "foreground" in colors
        assert "axis" in colors
        assert "grid" in colors

    def test_get_plot_colors_dark(self, qtbot, monkeypatch, tmp_path):
        """get_plot_colors should return colors dict for dark theme."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.plot_themes import get_plot_colors

        colors = get_plot_colors("dark")

        assert isinstance(colors, dict)
        assert "background" in colors
        # Dark theme background should be dark
        assert colors["background"].startswith("#1") or colors["background"].startswith(
            "#2"
        )

    def test_get_pyqtgraph_options_returns_dict(self, qtbot, monkeypatch, tmp_path):
        """get_pyqtgraph_options should return config dict."""
        monkeypatch.setenv("HOME", str(tmp_path))

        from xpcsviewer.gui.theme.plot_themes import get_pyqtgraph_options

        options = get_pyqtgraph_options("light")

        assert isinstance(options, dict)
        assert "background" in options
        assert "foreground" in options
