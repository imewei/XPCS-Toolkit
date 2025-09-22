"""
Centralized plotting constants for consistent styling across all plot handlers.

This module consolidates color schemes, markers, and styling constants that were
previously duplicated across multiple modules (matplot_qt.py, g2mod.py, tauq.py).
"""

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
