"""
Theme system for XPCS-TOOLKIT GUI.

This module provides theming infrastructure including:
- Design tokens (colors, spacing, typography)
- ThemeManager for switching and persisting themes
- Plot theme adapters for Matplotlib and PyQtGraph
"""

from xpcs_toolkit.gui.theme.manager import ThemeManager
from xpcs_toolkit.gui.theme.plot_themes import (
    MATPLOTLIB_DARK,
    MATPLOTLIB_LIGHT,
    apply_matplotlib_theme,
    apply_pyqtgraph_theme,
    get_matplotlib_params,
    get_plot_colors,
    get_pyqtgraph_options,
)
from xpcs_toolkit.gui.theme.tokens import (
    DARK_TOKENS,
    LIGHT_TOKENS,
    ColorTokens,
    SpacingTokens,
    ThemeDefinition,
    TypographyTokens,
)

__all__ = [
    "ThemeManager",
    "ColorTokens",
    "SpacingTokens",
    "TypographyTokens",
    "ThemeDefinition",
    "LIGHT_TOKENS",
    "DARK_TOKENS",
    "MATPLOTLIB_LIGHT",
    "MATPLOTLIB_DARK",
    "get_matplotlib_params",
    "get_plot_colors",
    "get_pyqtgraph_options",
    "apply_matplotlib_theme",
    "apply_pyqtgraph_theme",
]
