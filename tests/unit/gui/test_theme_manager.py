"""Unit tests for ThemeManager."""

import pytest

from xpcsviewer.gui.theme.manager import ThemeManager
from xpcsviewer.gui.theme.tokens import DARK_TOKENS, LIGHT_TOKENS


class TestThemeManagerBasics:
    """Basic ThemeManager tests."""

    def test_default_theme(self, qtbot, monkeypatch, tmp_path):
        """ThemeManager should default to system theme detection."""
        # Set up clean preferences directory
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        # Default mode should be system (from default preferences)
        # The actual theme depends on OS, so just check it's valid
        assert manager.get_current_theme() in ["light", "dark"]

    def test_set_light_theme(self, qtbot, monkeypatch, tmp_path):
        """Setting light theme should update current theme."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("light")
        assert manager.get_current_theme() == "light"
        assert manager.get_current_mode() == "light"

    def test_set_dark_theme(self, qtbot, monkeypatch, tmp_path):
        """Setting dark theme should update current theme."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("dark")
        assert manager.get_current_theme() == "dark"
        assert manager.get_current_mode() == "dark"

    def test_theme_changed_signal(self, qtbot, monkeypatch, tmp_path):
        """Theme changes should emit theme_changed signal."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        # Set to light first
        manager.set_theme("light")

        # Now switch to dark and check signal
        with qtbot.waitSignal(manager.theme_changed, timeout=1000) as blocker:
            manager.set_theme("dark")

        assert blocker.args == ["dark"]

    def test_no_signal_when_theme_unchanged(self, qtbot, monkeypatch, tmp_path):
        """No signal should be emitted if theme doesn't actually change."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        # Set to light
        manager.set_theme("light")

        # Set to light again - should not emit signal
        signal_received = False

        def on_signal(theme):
            nonlocal signal_received
            signal_received = True

        manager.theme_changed.connect(on_signal)
        manager.set_theme("light")

        # Process events
        qtbot.wait(100)
        assert not signal_received


class TestThemeManagerTokenAccess:
    """Tests for token access methods."""

    def test_get_color_valid_token(self, qtbot, monkeypatch, tmp_path):
        """get_color should return valid color for known tokens."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("light")
        color = manager.get_color("background_primary")
        assert color == LIGHT_TOKENS.colors.background_primary

    def test_get_color_dark_theme(self, qtbot, monkeypatch, tmp_path):
        """get_color should return dark theme colors when dark."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("dark")
        color = manager.get_color("background_primary")
        assert color == DARK_TOKENS.colors.background_primary

    def test_get_color_invalid_token(self, qtbot, monkeypatch, tmp_path):
        """get_color should raise KeyError for invalid tokens."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        with pytest.raises(KeyError):
            manager.get_color("invalid_token_name")

    def test_get_spacing_valid_size(self, qtbot, monkeypatch, tmp_path):
        """get_spacing should return valid spacing for known sizes."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        assert manager.get_spacing("sm") == 8
        assert manager.get_spacing("md") == 16
        assert manager.get_spacing("lg") == 24

    def test_get_spacing_invalid_size(self, qtbot, monkeypatch, tmp_path):
        """get_spacing should raise KeyError for invalid sizes."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        with pytest.raises(KeyError):
            manager.get_spacing("invalid_size")

    def test_get_tokens(self, qtbot, monkeypatch, tmp_path):
        """get_tokens should return current ThemeDefinition."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("light")
        tokens = manager.get_tokens()
        assert tokens.name == "light"

        manager.set_theme("dark")
        tokens = manager.get_tokens()
        assert tokens.name == "dark"


class TestThemeManagerMatplotlib:
    """Tests for Matplotlib integration."""

    def test_get_matplotlib_params(self, qtbot, monkeypatch, tmp_path):
        """get_matplotlib_params should return valid rcParams dict."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        params = manager.get_matplotlib_params()

        assert "figure.facecolor" in params
        assert "axes.facecolor" in params
        assert "text.color" in params
        assert "grid.color" in params

    def test_matplotlib_params_differ_by_theme(self, qtbot, monkeypatch, tmp_path):
        """Matplotlib params should be different for light vs dark."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        manager.set_theme("light")
        light_params = manager.get_matplotlib_params()

        manager.set_theme("dark")
        dark_params = manager.get_matplotlib_params()

        assert light_params["figure.facecolor"] != dark_params["figure.facecolor"]


class TestThemeManagerPyQtGraph:
    """Tests for PyQtGraph integration."""

    def test_apply_to_pyqtgraph(self, qtbot, monkeypatch, tmp_path):
        """apply_to_pyqtgraph should not raise errors."""
        monkeypatch.setenv("HOME", str(tmp_path))

        manager = ThemeManager()

        # Should not raise
        manager.apply_to_pyqtgraph()
