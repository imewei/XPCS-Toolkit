"""
Plot theme definitions for Matplotlib and PyQtGraph.

This module provides theme-aware color dictionaries for plotting libraries,
ensuring plots match the application theme.
"""

from typing import Literal

from xpcsviewer.gui.theme.tokens import DARK_TOKENS, LIGHT_TOKENS

ThemeName = Literal["light", "dark"]

# Matplotlib rcParams for light theme
MATPLOTLIB_LIGHT: dict[str, str] = {
    "figure.facecolor": LIGHT_TOKENS.colors.plot_background,
    "axes.facecolor": LIGHT_TOKENS.colors.plot_background,
    "axes.edgecolor": LIGHT_TOKENS.colors.plot_axis,
    "axes.labelcolor": LIGHT_TOKENS.colors.text_primary,
    "xtick.color": LIGHT_TOKENS.colors.plot_axis,
    "ytick.color": LIGHT_TOKENS.colors.plot_axis,
    "text.color": LIGHT_TOKENS.colors.text_primary,
    "grid.color": LIGHT_TOKENS.colors.plot_grid,
    "legend.facecolor": LIGHT_TOKENS.colors.background_elevated,
    "legend.edgecolor": LIGHT_TOKENS.colors.border_subtle,
    "savefig.facecolor": LIGHT_TOKENS.colors.plot_background,
    "savefig.edgecolor": LIGHT_TOKENS.colors.plot_background,
}

# Matplotlib rcParams for dark theme
MATPLOTLIB_DARK: dict[str, str] = {
    "figure.facecolor": DARK_TOKENS.colors.plot_background,
    "axes.facecolor": DARK_TOKENS.colors.plot_background,
    "axes.edgecolor": DARK_TOKENS.colors.plot_axis,
    "axes.labelcolor": DARK_TOKENS.colors.text_primary,
    "xtick.color": DARK_TOKENS.colors.plot_axis,
    "ytick.color": DARK_TOKENS.colors.plot_axis,
    "text.color": DARK_TOKENS.colors.text_primary,
    "grid.color": DARK_TOKENS.colors.plot_grid,
    "legend.facecolor": DARK_TOKENS.colors.background_elevated,
    "legend.edgecolor": DARK_TOKENS.colors.border_subtle,
    "savefig.facecolor": DARK_TOKENS.colors.plot_background,
    "savefig.edgecolor": DARK_TOKENS.colors.plot_background,
}


def get_matplotlib_params(theme: ThemeName) -> dict[str, str]:
    """
    Get matplotlib rcParams for the specified theme.

    Parameters
    ----------
    theme : ThemeName
        Either "light" or "dark"

    Returns
    -------
    dict[str, str]
        Dictionary of rcParams suitable for matplotlib.rcParams.update()
    """
    if theme == "dark":
        return MATPLOTLIB_DARK.copy()
    return MATPLOTLIB_LIGHT.copy()


def get_plot_colors(theme: ThemeName) -> dict[str, str]:
    """
    Get plot color dictionary for the specified theme.

    This provides a simplified color palette for use in custom plotting code.

    Parameters
    ----------
    theme : ThemeName
        Either "light" or "dark"

    Returns
    -------
    dict[str, str]
        Dictionary with keys: background, foreground, axis, grid, text
    """
    tokens = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    return {
        "background": tokens.colors.plot_background,
        "foreground": tokens.colors.text_primary,
        "axis": tokens.colors.plot_axis,
        "grid": tokens.colors.plot_grid,
        "text": tokens.colors.text_primary,
        "accent": tokens.colors.accent_primary,
    }


def get_pyqtgraph_options(theme: ThemeName) -> dict[str, str | bool]:
    """
    Get PyQtGraph config options for the specified theme.

    Parameters
    ----------
    theme : ThemeName
        Either "light" or "dark"

    Returns
    -------
    dict[str, str | bool]
        Dictionary suitable for pg.setConfigOptions(**options)
    """
    tokens = DARK_TOKENS if theme == "dark" else LIGHT_TOKENS
    return {
        "background": tokens.colors.plot_background,
        "foreground": tokens.colors.plot_axis,
        "antialias": True,
    }


def apply_matplotlib_theme(theme: ThemeName) -> None:
    """
    Apply theme to matplotlib globally.

    This updates matplotlib.rcParams with the theme colors.

    Parameters
    ----------
    theme : ThemeName
        Either "light" or "dark"
    """
    import matplotlib.pyplot as plt

    params = get_matplotlib_params(theme)
    plt.rcParams.update(params)


def apply_pyqtgraph_theme(theme: ThemeName) -> None:
    """
    Apply theme to PyQtGraph globally.

    This calls pg.setConfigOptions() with the theme colors.

    Parameters
    ----------
    theme : ThemeName
        Either "light" or "dark"
    """
    import pyqtgraph as pg

    options = get_pyqtgraph_options(theme)
    pg.setConfigOptions(**options)
