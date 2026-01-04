"""
Centralized plotting constants for consistent styling across all plot handlers.

This module consolidates color schemes, markers, and styling constants that were
previously duplicated across multiple modules (matplot_qt.py, g2mod.py, tauq.py).

Also provides theme-aware plotting colors via get_theme_colors().
"""

from typing import Literal

ThemeName = Literal["light", "dark"]

# Matplotlib default color cycle (10 colors) - the standard scientific palette
MATPLOTLIB_COLORS_HEX = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
)

# Same colors in RGB format for backends that require tuples
MATPLOTLIB_COLORS_RGB = (
    (31, 119, 180),  # blue
    (255, 127, 14),  # orange
    (44, 160, 44),  # green
    (214, 39, 40),  # red
    (148, 103, 189),  # purple
    (140, 86, 75),  # brown
    (227, 119, 194),  # pink
    (127, 127, 127),  # gray
    (188, 189, 34),  # olive
    (23, 190, 207),  # cyan
)

# Short color codes for simple plotting
BASIC_COLORS = ("b", "r", "g", "c", "m", "y", "k")

# Matplotlib marker styles
MATPLOTLIB_MARKERS = ["o", "v", "^", ">", "<", "s", "p", "h", "*", "+", "d", "x"]

# PyQtGraph equivalent markers
PYQTGRAPH_MARKERS = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d", "x"]

# Extended marker set for specialized use
EXTENDED_MARKERS = ("o", "v", "^", "<", ">", "8", "s", "p", "P", "*")


def get_color_marker(
    n: int, backend: str = "matplotlib", color_format: str = "hex"
) -> tuple[str | tuple[int, int, int], str]:
    """
    Get color and marker for plotting by index.

    Parameters
    ----------
    n : int
        Index for color/marker selection
    backend : str
        Backend type ("matplotlib" or "pyqtgraph")
    color_format : str
        Color format ("hex", "rgb", or "basic")

    Returns
    -------
    tuple
        (color, marker) for the given index
    """
    # Select color based on format
    color: str | tuple[int, int, int]
    if color_format == "hex":
        color = MATPLOTLIB_COLORS_HEX[n % len(MATPLOTLIB_COLORS_HEX)]
    elif color_format == "rgb":
        color = MATPLOTLIB_COLORS_RGB[n % len(MATPLOTLIB_COLORS_RGB)]
    elif color_format == "basic":
        color = BASIC_COLORS[n % len(BASIC_COLORS)]
    else:
        raise ValueError(f"Unknown color format: {color_format}")

    # Select marker based on backend
    if backend == "matplotlib":
        marker = MATPLOTLIB_MARKERS[n % len(MATPLOTLIB_MARKERS)]
    elif backend == "pyqtgraph":
        marker = PYQTGRAPH_MARKERS[n % len(PYQTGRAPH_MARKERS)]
    else:
        marker = EXTENDED_MARKERS[n % len(EXTENDED_MARKERS)]

    return color, marker


def get_color_cycle(
    backend: str = "matplotlib", color_format: str = "hex"
) -> tuple[str | tuple[int, int, int], ...]:
    """
    Get the full color cycle for a backend.

    Parameters
    ----------
    backend : str
        Backend type (affects return format)
    color_format : str
        Color format ("hex", "rgb", or "basic")

    Returns
    -------
    tuple
        Color cycle for the specified format
    """
    if color_format == "hex":
        return MATPLOTLIB_COLORS_HEX
    if color_format == "rgb":
        return MATPLOTLIB_COLORS_RGB
    if color_format == "basic":
        return BASIC_COLORS
    raise ValueError(f"Unknown color format: {color_format}")


def get_marker_cycle(backend: str = "matplotlib") -> list[str] | tuple[str, ...]:
    """
    Get the marker cycle for a backend.

    Parameters
    ----------
    backend : str
        Backend type ("matplotlib", "pyqtgraph", or "extended")

    Returns
    -------
    tuple
        Marker cycle for the specified backend
    """
    if backend == "matplotlib":
        return MATPLOTLIB_MARKERS
    if backend == "pyqtgraph":
        return PYQTGRAPH_MARKERS
    if backend == "extended":
        return EXTENDED_MARKERS
    raise ValueError(f"Unknown backend: {backend}")


def get_theme_colors(theme: ThemeName | None = None) -> dict[str, str]:
    """
    Get plot colors for the specified theme.

    If theme is None, attempts to get the current theme from ThemeManager.
    Falls back to light theme if theme system is unavailable.

    Parameters
    ----------
    theme : ThemeName or None
        Either "light", "dark", or None (auto-detect from ThemeManager)

    Returns
    -------
    dict[str, str]
        Dictionary with keys: background, foreground, axis, grid, text, accent
    """
    if theme is None:
        # Try to get from theme manager
        try:
            from xpcsviewer.gui.theme.plot_themes import get_plot_colors

            # Try to get current theme from singleton or default
            try:
                from PySide6.QtWidgets import QApplication

                app = QApplication.instance()
                if app is not None:
                    # Find ThemeManager in top-level widgets
                    for widget in app.topLevelWidgets():
                        if hasattr(widget, "theme_manager"):
                            current_theme = widget.theme_manager.get_current_theme()
                            return get_plot_colors(current_theme)
            except Exception:
                pass
            # Default to light if no manager found
            return get_plot_colors("light")
        except ImportError:
            # Fall back to hardcoded light theme colors
            return {
                "background": "#FFFFFF",
                "foreground": "#1D1D1F",
                "axis": "#3C3C43",
                "grid": "#E5E5EA",
                "text": "#1D1D1F",
                "accent": "#007AFF",
            }

    # Direct theme specified
    try:
        from xpcsviewer.gui.theme.plot_themes import get_plot_colors

        return get_plot_colors(theme)
    except ImportError:
        # Fall back to hardcoded colors
        if theme == "dark":
            return {
                "background": "#1C1C1E",
                "foreground": "#F5F5F7",
                "axis": "#98989D",
                "grid": "#38383A",
                "text": "#F5F5F7",
                "accent": "#0A84FF",
            }
        return {
            "background": "#FFFFFF",
            "foreground": "#1D1D1F",
            "axis": "#3C3C43",
            "grid": "#E5E5EA",
            "text": "#1D1D1F",
            "accent": "#007AFF",
        }
