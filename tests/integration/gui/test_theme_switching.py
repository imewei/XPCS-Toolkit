"""Integration tests for theme switching functionality."""

import time

import pytest
from PySide6.QtWidgets import QApplication

from xpcsviewer.gui.state.preferences import (
    UserPreferences,
    get_preferences_path,
    load_preferences,
    save_preferences,
)
from xpcsviewer.gui.theme.manager import ThemeManager


class TestThemeSwitchingIntegration:
    """Integration tests for theme switching across the application."""

    @pytest.fixture
    def theme_manager(self, qtbot, tmp_path, monkeypatch):
        """Create a ThemeManager with temporary preferences."""
        prefs_file = tmp_path / ".xpcsviewer" / "preferences.json"
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.preferences.get_preferences_path",
            lambda: prefs_file,
        )
        manager = ThemeManager()
        return manager

    def test_switch_to_light_theme(self, theme_manager):
        """Switching to light theme should update current theme."""
        theme_manager.set_theme("light")

        assert theme_manager.get_current_theme() == "light"

    def test_switch_to_dark_theme(self, theme_manager):
        """Switching to dark theme should update current theme."""
        theme_manager.set_theme("dark")

        assert theme_manager.get_current_theme() == "dark"

    def test_theme_change_emits_signal(self, theme_manager, qtbot):
        """Changing theme should emit theme_changed signal."""
        theme_manager.set_theme("light")

        with qtbot.waitSignal(theme_manager.theme_changed, timeout=1000) as blocker:
            theme_manager.set_theme("dark")

        assert blocker.args[0] == "dark"

    def test_no_signal_when_theme_unchanged(self, theme_manager, qtbot):
        """Setting same theme should not emit signal."""
        theme_manager.set_theme("light")

        signal_received = []

        def record_signal(mode):
            signal_received.append(mode)

        theme_manager.theme_changed.connect(record_signal)

        # Set the same theme again
        theme_manager.set_theme("light")

        # Give time for signal to be emitted
        QApplication.processEvents()

        # No new signals should be recorded
        assert len(signal_received) == 0

    def test_theme_persistence(self, tmp_path, monkeypatch):
        """Theme preference should persist across manager instances."""
        prefs_file = tmp_path / ".xpcsviewer" / "preferences.json"
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.preferences.get_preferences_path",
            lambda: prefs_file,
        )

        # Create manager and set theme
        manager1 = ThemeManager()
        manager1.set_theme("dark")

        # Create new manager - should load saved preference
        manager2 = ThemeManager()

        assert manager2.get_current_theme() == "dark"

    def test_theme_switch_performance(self, theme_manager):
        """Theme switch should complete within reasonable time."""
        start_time = time.time()

        theme_manager.set_theme("light")
        theme_manager.set_theme("dark")
        theme_manager.set_theme("light")

        elapsed = time.time() - start_time

        # 3 theme switches should complete in under 5 seconds
        # (accounts for CI/testing environment overhead)
        assert elapsed < 5.0, f"Theme switching took {elapsed:.2f}s"

    def test_build_stylesheet_returns_content(self, theme_manager):
        """_build_stylesheet should return non-empty stylesheet."""
        theme_manager.set_theme("light")
        stylesheet = theme_manager._build_stylesheet()

        assert stylesheet is not None
        assert len(stylesheet) > 0

    def test_stylesheet_differs_between_themes(self, theme_manager):
        """Light and dark stylesheets should be different."""
        theme_manager.set_theme("light")
        light_stylesheet = theme_manager._build_stylesheet()

        theme_manager.set_theme("dark")
        dark_stylesheet = theme_manager._build_stylesheet()

        assert light_stylesheet != dark_stylesheet

    def test_get_color_returns_valid_value(self, theme_manager):
        """get_color should return valid color string."""
        theme_manager.set_theme("light")

        bg_color = theme_manager.get_color("background_primary")

        assert bg_color is not None
        assert bg_color.startswith("#")

    def test_colors_differ_between_themes(self, theme_manager):
        """Colors should differ between light and dark themes."""
        theme_manager.set_theme("light")
        light_bg = theme_manager.get_color("background_primary")

        theme_manager.set_theme("dark")
        dark_bg = theme_manager.get_color("background_primary")

        assert light_bg != dark_bg

    def test_get_tokens_returns_theme_definition(self, theme_manager):
        """get_tokens should return current theme definition."""
        theme_manager.set_theme("dark")

        tokens = theme_manager.get_tokens()

        assert tokens is not None
        assert hasattr(tokens, "colors")
        assert hasattr(tokens, "spacing")
        assert hasattr(tokens, "typography")


class TestThemePreferencesIntegration:
    """Integration tests for theme preferences persistence."""

    def test_preferences_save_and_load(self, tmp_path, monkeypatch):
        """Preferences should save and load correctly."""
        prefs_file = tmp_path / ".xpcsviewer" / "preferences.json"
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.preferences.get_preferences_path",
            lambda: prefs_file,
        )

        # Save preferences
        prefs = UserPreferences(theme="dark")
        result = save_preferences(prefs)
        assert result is True

        # Load preferences
        loaded = load_preferences()

        assert loaded.theme == "dark"

    def test_default_preferences_created(self, tmp_path, monkeypatch):
        """Default preferences should be created if file doesn't exist."""
        prefs_file = tmp_path / ".xpcsviewer" / "preferences.json"
        prefs_file.parent.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "xpcsviewer.gui.state.preferences.get_preferences_path",
            lambda: prefs_file,
        )

        # Load without existing file
        prefs = load_preferences()

        # Should get defaults
        assert prefs.theme == "system"
