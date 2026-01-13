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
from xpcsviewer.gui.theme.style_helpers import (
    apply_destructive_buttons,
    apply_secondary_buttons,
    set_button_size,
    set_button_style,
    set_card_style,
    set_control_row,
    set_density,
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
    "apply_destructive_buttons",
    "apply_matplotlib_theme",
    "apply_pyqtgraph_theme",
    "apply_secondary_buttons",
    "get_matplotlib_params",
    "get_plot_colors",
    "get_pyqtgraph_options",
    "set_button_size",
    "set_button_style",
    "set_card_style",
    "set_control_row",
    "set_density",
]
