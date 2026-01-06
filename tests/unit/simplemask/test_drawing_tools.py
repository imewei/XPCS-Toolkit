"""Unit tests for SimpleMask drawing tools configuration.

Tests the drawing tool definitions, utilities, and configuration module.
"""

import pytest
from PySide6.QtCore import Qt

from xpcsviewer.simplemask.drawing_tools import (
    DEFAULT_DRAW_PARAMS,
    DRAWING_TOOLS,
    ERASER_TOOL,
    TOOL_COLORS,
    DrawingTool,
    get_shortcut_map,
    get_tool_color,
)


class TestDrawingToolDataclass:
    """Tests for DrawingTool dataclass."""

    def test_drawing_tool_creation(self):
        """DrawingTool should be creatable with required fields."""
        tool = DrawingTool(
            name="Test",
            tool_type="Rectangle",
            icon="test",
            shortcut="T",
            tooltip="Test tool",
        )

        assert tool.name == "Test"
        assert tool.tool_type == "Rectangle"
        assert tool.icon == "test"
        assert tool.shortcut == "T"
        assert tool.tooltip == "Test tool"

    def test_drawing_tool_default_mode(self):
        """DrawingTool should default to exclusive mode."""
        tool = DrawingTool(
            name="Test",
            tool_type="Rectangle",
            icon="test",
            shortcut=None,
            tooltip="Test",
        )

        assert tool.default_mode == "exclusive"

    def test_drawing_tool_default_cursor(self):
        """DrawingTool should default to CrossCursor."""
        tool = DrawingTool(
            name="Test",
            tool_type="Rectangle",
            icon="test",
            shortcut=None,
            tooltip="Test",
        )

        assert tool.cursor == Qt.CursorShape.CrossCursor

    def test_drawing_tool_inclusive_mode(self):
        """DrawingTool should accept inclusive mode."""
        tool = DrawingTool(
            name="Eraser",
            tool_type="Rectangle",
            icon="eraser",
            shortcut="X",
            tooltip="Erase",
            default_mode="inclusive",
        )

        assert tool.default_mode == "inclusive"


class TestDrawingToolsDefinitions:
    """Tests for DRAWING_TOOLS dictionary."""

    def test_drawing_tools_not_empty(self):
        """DRAWING_TOOLS should contain tool definitions."""
        assert len(DRAWING_TOOLS) > 0

    def test_rectangle_tool_exists(self):
        """Rectangle tool should be defined."""
        assert "rectangle" in DRAWING_TOOLS
        tool = DRAWING_TOOLS["rectangle"]
        assert tool.name == "Rectangle"
        assert tool.tool_type == "Rectangle"

    def test_circle_tool_exists(self):
        """Circle tool should be defined."""
        assert "circle" in DRAWING_TOOLS
        tool = DRAWING_TOOLS["circle"]
        assert tool.name == "Circle"
        assert tool.tool_type == "Circle"

    def test_polygon_tool_exists(self):
        """Polygon tool should be defined."""
        assert "polygon" in DRAWING_TOOLS
        tool = DRAWING_TOOLS["polygon"]
        assert tool.name == "Polygon"
        assert tool.tool_type == "Polygon"

    def test_line_tool_exists(self):
        """Line tool should be defined."""
        assert "line" in DRAWING_TOOLS
        tool = DRAWING_TOOLS["line"]
        assert tool.name == "Line"
        assert tool.tool_type == "Line"

    def test_ellipse_tool_exists(self):
        """Ellipse tool should be defined."""
        assert "ellipse" in DRAWING_TOOLS
        tool = DRAWING_TOOLS["ellipse"]
        assert tool.name == "Ellipse"
        assert tool.tool_type == "Ellipse"

    def test_all_tools_have_shortcuts(self):
        """All drawing tools should have keyboard shortcuts."""
        for tool_key, tool in DRAWING_TOOLS.items():
            assert tool.shortcut is not None, f"{tool_key} missing shortcut"

    def test_all_tools_have_tooltips(self):
        """All drawing tools should have tooltips."""
        for tool_key, tool in DRAWING_TOOLS.items():
            assert tool.tooltip, f"{tool_key} missing tooltip"

    def test_all_tools_exclusive_mode(self):
        """All drawing tools should default to exclusive mode."""
        for tool_key, tool in DRAWING_TOOLS.items():
            assert tool.default_mode == "exclusive", f"{tool_key} not exclusive"


class TestEraserTool:
    """Tests for ERASER_TOOL definition."""

    def test_eraser_tool_exists(self):
        """ERASER_TOOL should be defined."""
        assert ERASER_TOOL is not None

    def test_eraser_tool_name(self):
        """Eraser tool should have correct name."""
        assert ERASER_TOOL.name == "Eraser"

    def test_eraser_tool_type(self):
        """Eraser tool should use Rectangle type."""
        assert ERASER_TOOL.tool_type == "Rectangle"

    def test_eraser_inclusive_mode(self):
        """Eraser tool should use inclusive mode."""
        assert ERASER_TOOL.default_mode == "inclusive"

    def test_eraser_has_shortcut(self):
        """Eraser tool should have keyboard shortcut."""
        assert ERASER_TOOL.shortcut is not None
        assert ERASER_TOOL.shortcut == "X"


class TestToolColors:
    """Tests for tool color configuration."""

    def test_tool_colors_defined(self):
        """TOOL_COLORS should be defined."""
        assert TOOL_COLORS is not None
        assert len(TOOL_COLORS) > 0

    def test_exclusive_color(self):
        """Exclusive mode should have red color."""
        assert "exclusive" in TOOL_COLORS
        assert TOOL_COLORS["exclusive"] == "r"

    def test_inclusive_color(self):
        """Inclusive mode should have green color."""
        assert "inclusive" in TOOL_COLORS
        assert TOOL_COLORS["inclusive"] == "g"


class TestGetToolColor:
    """Tests for get_tool_color function."""

    def test_get_exclusive_color(self):
        """get_tool_color should return red for exclusive."""
        color = get_tool_color("exclusive")
        assert color == "r"

    def test_get_inclusive_color(self):
        """get_tool_color should return green for inclusive."""
        color = get_tool_color("inclusive")
        assert color == "g"

    def test_get_unknown_mode_returns_default(self):
        """get_tool_color should return red for unknown mode."""
        color = get_tool_color("unknown")
        assert color == "r"

    def test_get_empty_mode_returns_default(self):
        """get_tool_color should return red for empty mode."""
        color = get_tool_color("")
        assert color == "r"


class TestGetShortcutMap:
    """Tests for get_shortcut_map function."""

    def test_shortcut_map_not_empty(self):
        """Shortcut map should not be empty."""
        shortcuts = get_shortcut_map()
        assert len(shortcuts) > 0

    def test_shortcut_map_contains_drawing_tools(self):
        """Shortcut map should contain all drawing tool shortcuts."""
        shortcuts = get_shortcut_map()

        for tool_key, tool in DRAWING_TOOLS.items():
            if tool.shortcut:
                assert tool.shortcut in shortcuts
                assert shortcuts[tool.shortcut] == tool_key

    def test_shortcut_map_contains_eraser(self):
        """Shortcut map should contain eraser shortcut."""
        shortcuts = get_shortcut_map()

        if ERASER_TOOL.shortcut:
            assert ERASER_TOOL.shortcut in shortcuts
            assert shortcuts[ERASER_TOOL.shortcut] == "eraser"

    def test_shortcut_keys_are_strings(self):
        """All shortcut keys should be strings."""
        shortcuts = get_shortcut_map()

        for key in shortcuts:
            assert isinstance(key, str)

    def test_shortcut_values_are_strings(self):
        """All shortcut values should be tool name strings."""
        shortcuts = get_shortcut_map()

        for value in shortcuts.values():
            assert isinstance(value, str)


class TestDefaultDrawParams:
    """Tests for default drawing parameters."""

    def test_default_params_defined(self):
        """DEFAULT_DRAW_PARAMS should be defined."""
        assert DEFAULT_DRAW_PARAMS is not None

    def test_default_radius(self):
        """Default radius should be positive."""
        assert "radius" in DEFAULT_DRAW_PARAMS
        assert DEFAULT_DRAW_PARAMS["radius"] > 0

    def test_default_width(self):
        """Default width should be positive."""
        assert "width" in DEFAULT_DRAW_PARAMS
        assert DEFAULT_DRAW_PARAMS["width"] > 0

    def test_default_movable(self):
        """Default movable should be True."""
        assert "movable" in DEFAULT_DRAW_PARAMS
        assert DEFAULT_DRAW_PARAMS["movable"] is True


class TestToolTypeValidity:
    """Tests that tool types match expected ROI types."""

    VALID_TOOL_TYPES = {"Rectangle", "Circle", "Polygon", "Line", "Ellipse"}

    def test_all_drawing_tools_have_valid_types(self):
        """All drawing tools should have valid tool types."""
        for tool_key, tool in DRAWING_TOOLS.items():
            assert tool.tool_type in self.VALID_TOOL_TYPES, (
                f"{tool_key} has invalid tool_type: {tool.tool_type}"
            )

    def test_eraser_has_valid_type(self):
        """Eraser should have valid tool type."""
        assert ERASER_TOOL.tool_type in self.VALID_TOOL_TYPES
