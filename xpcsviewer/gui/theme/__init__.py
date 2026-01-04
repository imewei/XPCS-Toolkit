"""
Theme system for XPCS-TOOLKIT GUI.

This module provides theming infrastructure including:
- Design tokens (colors, spacing, typography)
- ThemeManager for switching and persisting themes
- Plot theme adapters for Matplotlib and PyQtGraph
"""

from xpcsviewer.gui.theme.manager import ThemeManager
from xpcsviewer.gui.theme.plot_themes import (
    MATPLOTLIB_DARK,
    MATPLOTLIB_LIGHT,
    apply_matplotlib_theme,
    apply_pyqtgraph_theme,
    get_matplotlib_params,
    get_plot_colors,
    get_pyqtgraph_options,
)
from xpcsviewer.gui.theme.tokens import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    ColorTokens,
    SpacingTokens,
    ThemeDefinition,
    TypographyTokens,
)

__all__ = [
    "DARK_TOKENS",
    "LIGHT_TOKENS",
    "MATPLOTLIB_DARK",
    "MATPLOTLIB_LIGHT",
    "ColorTokens",
    "SpacingTokens",
    "ThemeDefinition",
    "ThemeManager",
    "TypographyTokens",
    "apply_matplotlib_theme",
    "apply_pyqtgraph_theme",
    "get_matplotlib_params",
    "get_plot_colors",
    "get_pyqtgraph_options",
]
