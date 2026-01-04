"""
Theme manager for XPCS-TOOLKIT GUI.

This module provides the ThemeManager class that handles theme switching,
OS theme detection, and stylesheet application.
"""

import logging
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from xpcsviewer.gui.state.preferences import load_preferences, save_preferences
from xpcsviewer.gui.theme.tokens import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    ColorTokens,
    SpacingTokens,
    ThemeDefinition,
)

logger = logging.getLogger(__name__)

ThemeMode = Literal["light", "dark", "system"]


class ThemeManager(QObject):
    """
    Manages application theming including stylesheets and plot colors.

    Handles:
    - Loading and applying Qt stylesheets
    - Detecting OS theme preference
    - Persisting user theme choice
    - Emitting signals on theme change
    - Providing current theme tokens to other components
    """

    # Signal emitted when theme changes
    # Signature: theme_changed(theme_name: str)
    theme_changed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        """
        Initialize the ThemeManager.

        Args:
            parent: Parent QObject
        """
        super().__init__(parent)
        self._current_theme: str = "light"
        self._current_mode: ThemeMode = "system"
        self._tokens: ThemeDefinition = LIGHT_TOKENS
        self._styles_dir = Path(__file__).parent / "styles"

        # Load saved preference
        prefs = load_preferences()
        self._current_mode = prefs.theme

        # Apply initial theme
        self._apply_theme_from_mode()

    def get_current_theme(self) -> str:
        """
        Get the name of the currently active theme.

        Returns:
            Either "light" or "dark"
        """
        return self._current_theme

    def get_current_mode(self) -> ThemeMode:
        """
        Get the current theme mode setting.

        Returns:
            "light", "dark", or "system"
        """
        return self._current_mode

    def set_theme(self, mode: ThemeMode) -> None:
        """
        Set the application theme.

        Args:
            mode: One of "light", "dark", or "system"

        Side Effects:
            - Updates application stylesheet
            - Emits theme_changed signal
            - Persists preference to disk
        """
        self._current_mode = mode

        # Save preference
        prefs = load_preferences()
        prefs.theme = mode
        save_preferences(prefs)

        # Apply theme
        old_theme = self._current_theme
        self._apply_theme_from_mode()

        if old_theme != self._current_theme:
            self.theme_changed.emit(self._current_theme)
            logger.info(f"Theme changed to {self._current_theme}")

    def _apply_theme_from_mode(self) -> None:
        """Apply theme based on current mode setting."""
        if self._current_mode == "system":
            self._current_theme = self._detect_os_theme()
        else:
            self._current_theme = self._current_mode

        self._tokens = DARK_TOKENS if self._current_theme == "dark" else LIGHT_TOKENS
        self._apply_stylesheet()

    def _detect_os_theme(self) -> str:
        """
        Detect the operating system theme preference.

        Returns:
            "light" or "dark" based on OS setting
        """
        app = QGuiApplication.instance()
        if app is None:
            return "light"

        # Qt 6.5+ has colorScheme() on QStyleHints
        style_hints = app.styleHints()
        if hasattr(style_hints, "colorScheme"):
            from PySide6.QtCore import Qt

            scheme = style_hints.colorScheme()
            if scheme == Qt.ColorScheme.Dark:
                return "dark"
            return "light"

        # Fallback: check palette
        palette = app.palette()
        bg = palette.window().color()
        # If background is dark, assume dark mode
        if bg.lightness() < 128:
            return "dark"
        return "light"

    def _apply_stylesheet(self) -> None:
        """Load and apply the combined stylesheet."""
        app = QApplication.instance()
        if app is None:
            logger.warning("No QApplication instance for stylesheet")
            return

        stylesheet = self._build_stylesheet()
        app.setStyleSheet(stylesheet)
        logger.debug(f"Applied {self._current_theme} theme stylesheet")

    def _build_stylesheet(self) -> str:
        """
        Build combined stylesheet with token substitution.

        Returns:
            Combined and processed stylesheet string
        """
        # Load base stylesheet
        base_qss = self._load_qss_file("base.qss")

        # Load theme-specific overrides
        theme_qss = self._load_qss_file(f"{self._current_theme}.qss")

        # Combine stylesheets
        combined = base_qss + "\n" + theme_qss

        # Substitute tokens
        combined = self._substitute_tokens(combined)

        return combined

    def _load_qss_file(self, filename: str) -> str:
        """Load a QSS file from the styles directory."""
        file_path = self._styles_dir / filename
        if not file_path.exists():
            logger.debug(f"QSS file not found: {file_path}")
            return ""

        try:
            with open(file_path, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to load {filename}: {e}")
            return ""

    def _substitute_tokens(self, stylesheet: str) -> str:
        """
        Substitute @token references in stylesheet.

        Tokens are referenced as @token_name in QSS files.
        """
        colors = self._tokens.colors
        spacing = self._tokens.spacing
        typography = self._tokens.typography

        # Color tokens
        for field_name in ColorTokens.__dataclass_fields__:
            token = f"@{field_name}"
            value = getattr(colors, field_name)
            stylesheet = stylesheet.replace(token, value)

        # Spacing tokens
        for field_name in SpacingTokens.__dataclass_fields__:
            token = f"@spacing_{field_name}"
            value = str(getattr(spacing, field_name))
            stylesheet = stylesheet.replace(token, f"{value}px")

        # Typography tokens
        stylesheet = stylesheet.replace("@font_family", typography.font_family)
        stylesheet = stylesheet.replace("@size_xs", f"{typography.size_xs}pt")
        stylesheet = stylesheet.replace("@size_sm", f"{typography.size_sm}pt")
        stylesheet = stylesheet.replace("@size_base", f"{typography.size_base}pt")
        stylesheet = stylesheet.replace("@size_lg", f"{typography.size_lg}pt")
        stylesheet = stylesheet.replace("@size_xl", f"{typography.size_xl}pt")
        stylesheet = stylesheet.replace("@size_xxl", f"{typography.size_xxl}pt")

        return stylesheet

    def get_color(self, token_name: str) -> str:
        """
        Get a color value by token name.

        Args:
            token_name: Name of the color token

        Returns:
            Hex color value

        Raises:
            KeyError: If token_name is not valid
        """
        if not hasattr(self._tokens.colors, token_name):
            raise KeyError(f"Unknown color token: {token_name}")
        return getattr(self._tokens.colors, token_name)

    def get_spacing(self, size: str) -> int:
        """
        Get a spacing value by size name.

        Args:
            size: One of "xs", "sm", "md", "lg", "xl", "xxl"

        Returns:
            Spacing value in pixels

        Raises:
            KeyError: If size is not valid
        """
        if not hasattr(self._tokens.spacing, size):
            raise KeyError(f"Unknown spacing size: {size}")
        return getattr(self._tokens.spacing, size)

    def get_tokens(self) -> ThemeDefinition:
        """
        Get the current theme definition.

        Returns:
            Current ThemeDefinition
        """
        return self._tokens

    def get_matplotlib_params(self) -> dict[str, str]:
        """
        Get matplotlib rcParams for current theme.

        Returns:
            Parameters suitable for matplotlib.rcParams.update()
        """
        colors = self._tokens.colors
        return {
            "figure.facecolor": colors.plot_background,
            "axes.facecolor": colors.plot_background,
            "axes.edgecolor": colors.plot_axis,
            "axes.labelcolor": colors.text_primary,
            "xtick.color": colors.plot_axis,
            "ytick.color": colors.plot_axis,
            "text.color": colors.text_primary,
            "grid.color": colors.plot_grid,
            "legend.facecolor": colors.background_elevated,
            "legend.edgecolor": colors.border_subtle,
        }

    def apply_to_pyqtgraph(self) -> None:
        """
        Apply current theme to PyQtGraph global configuration.

        Side Effects:
            - Updates pg.setConfigOptions()
            - May require plot refresh to take effect
        """
        try:
            import pyqtgraph as pg
        except ImportError:
            logger.warning("PyQtGraph not available")
            return

        colors = self._tokens.colors
        is_dark = self._current_theme == "dark"

        pg.setConfigOptions(
            background=colors.plot_background,
            foreground=colors.plot_axis,
            antialias=True,
        )

        logger.debug(f"Applied {self._current_theme} theme to PyQtGraph")
