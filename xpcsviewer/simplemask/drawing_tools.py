"""Drawing tool configuration and utilities for SimpleMask.

This module provides tool definitions, default configurations, and utility
functions for the mask drawing system.
"""

from dataclasses import dataclass
from typing import Literal

# Qt imports via compatibility layer
from xpcsviewer.gui.qt_compat import Qt


@dataclass
class DrawingTool:
    """Configuration for a drawing tool.

    Attributes:
        name: Tool display name
        tool_type: ROI type for kernel add_drawing()
        icon: Icon name (for future icon support)
        shortcut: Keyboard shortcut
        tooltip: Tooltip text
        default_mode: Default mask mode (exclusive/inclusive)
    """

    name: str
    tool_type: Literal["Rectangle", "Circle", "Polygon", "Line", "Ellipse"]
    icon: str
    shortcut: str | None
    tooltip: str
    default_mode: Literal["exclusive", "inclusive"] = "exclusive"
    cursor: Qt.CursorShape = Qt.CursorShape.CrossCursor


# Drawing tool definitions
DRAWING_TOOLS = {
    "rectangle": DrawingTool(
        name="Rectangle",
        tool_type="Rectangle",
        icon="rectangle",
        shortcut="R",
        tooltip="Draw rectangular mask region (R)",
        default_mode="exclusive",
    ),
    "circle": DrawingTool(
        name="Circle",
        tool_type="Circle",
        icon="circle",
        shortcut="C",
        tooltip="Draw circular mask region (C)",
        default_mode="exclusive",
    ),
    "polygon": DrawingTool(
        name="Polygon",
        tool_type="Polygon",
        icon="polygon",
        shortcut="P",
        tooltip="Draw polygon mask region (P)",
        default_mode="exclusive",
    ),
    "line": DrawingTool(
        name="Line",
        tool_type="Line",
        icon="line",
        shortcut="L",
        tooltip="Draw line mask region (L)",
        default_mode="exclusive",
    ),
    "ellipse": DrawingTool(
        name="Ellipse",
        tool_type="Ellipse",
        icon="ellipse",
        shortcut="E",
        tooltip="Draw elliptical mask region (E)",
        default_mode="exclusive",
    ),
}


# Eraser tool (uses inclusive mode to remove from mask)
ERASER_TOOL = DrawingTool(
    name="Eraser",
    tool_type="Rectangle",
    icon="eraser",
    shortcut="X",
    tooltip="Erase mask region (X) - removes pixels from mask",
    default_mode="inclusive",
)


# Drawing tool colors by mode
TOOL_COLORS = {
    "exclusive": "r",  # Red for exclusion (masking)
    "inclusive": "g",  # Green for inclusion (erasing)
}


# Default drawing parameters
DEFAULT_DRAW_PARAMS = {
    "radius": 60,
    "width": 3,
    "num_edges": None,  # Random 6-10 for polygon
    "movable": True,
}


def get_tool_color(mode: str) -> str:
    """Get the pen color for a mask mode.

    Args:
        mode: "exclusive" or "inclusive"

    Returns:
        Color string for pen
    """
    return TOOL_COLORS.get(mode, "r")


def get_shortcut_map() -> dict[str, str]:
    """Get mapping of keyboard shortcuts to tool names.

    Returns:
        Dictionary mapping shortcut keys to tool names
    """
    shortcuts = {}
    for name, tool in DRAWING_TOOLS.items():
        if tool.shortcut:
            shortcuts[tool.shortcut] = name
    if ERASER_TOOL.shortcut:
        shortcuts[ERASER_TOOL.shortcut] = "eraser"
    return shortcuts
