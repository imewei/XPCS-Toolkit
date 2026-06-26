"""Unit tests for Enhanced ROI Scale Handles (Feature 1).

Tests the additional scale handles added to Rectangle, Ellipse, and Circle ROIs
in simplemask_kernel.add_drawing() method.

Scientific testing approach:
- Verify handle count matches specification
- Test handle positions are geometrically correct
- Verify scaling behavior from different handles
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

# Tests that require Qt GUI - marked for conditional execution
pytestmark = pytest.mark.gui


class TestRectangleScaleHandles:
    """Tests for Rectangle ROI scale handles (8 handles: 4 corners + 4 midpoints)."""

    @pytest.fixture
    def mock_kernel(self, qapp):
        """Create a minimal SimpleMaskKernel mock for testing."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        kernel.hdl = MagicMock()
        kernel.hdl.add_item = MagicMock(return_value="roi_0")
        kernel.hdl.roi_items = {}
        kernel.hdl.scene = MagicMock()
        kernel.hdl.scene.itemsBoundingRect = MagicMock(return_value=MagicMock())
        kernel.shape = (512, 512)
        kernel.metadata = {"bcx": 256, "bcy": 256}
        return kernel

    def test_rectangle_has_eight_handles(self, mock_kernel):
        """Rectangle ROI should have 8 scale handles."""
        roi = mock_kernel.add_drawing(
            sl_type="Rectangle",
            sl_mode="exclusive",
        )

        assert roi is not None
        # PyQtGraph RectROI has handles in a handles list
        # Count custom scale handles (excluding rotation handles)
        assert len(roi.handles) >= 8, (
            f"Expected at least 8 handles, got {len(roi.handles)}"
        )

    def test_rectangle_corner_handles_exist(self, mock_kernel):
        """Rectangle should have handles at all 4 corners."""
        roi = mock_kernel.add_drawing(
            sl_type="Rectangle",
            sl_mode="exclusive",
        )

        assert roi is not None
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]

        # Check corner positions are present (normalized 0-1 coordinates)
        expected_corners = [
            (0, 0),  # bottom-left
            (0, 1),  # top-left
            (1, 0),  # bottom-right
            (1, 1),  # top-right
        ]

        for corner in expected_corners:
            found = any(
                np.allclose([h.x(), h.y()], corner, atol=0.01) for h in handle_positions
            )
            assert found, f"Corner handle at {corner} not found"

    def test_rectangle_midpoint_handles_exist(self, mock_kernel):
        """Rectangle should have handles at all 4 midpoints."""
        roi = mock_kernel.add_drawing(
            sl_type="Rectangle",
            sl_mode="exclusive",
        )

        assert roi is not None
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]

        # Check midpoint positions (normalized 0-1 coordinates)
        expected_midpoints = [
            (0.5, 0),  # bottom-mid
            (0.5, 1),  # top-mid
            (0, 0.5),  # left-mid
            (1, 0.5),  # right-mid
        ]

        for midpoint in expected_midpoints:
            found = any(
                np.allclose([h.x(), h.y()], midpoint, atol=0.01)
                for h in handle_positions
            )
            assert found, f"Midpoint handle at {midpoint} not found"

    def test_rectangle_default_size_updated(self, mock_kernel):
        """Rectangle should have updated default size [200, 150]."""
        roi = mock_kernel.add_drawing(
            sl_type="Rectangle",
            sl_mode="exclusive",
        )

        assert roi is not None
        size = roi.size()
        # Default size should be [200, 150] as specified in the plan
        assert size[0] == 200, f"Expected width 200, got {size[0]}"
        assert size[1] == 150, f"Expected height 150, got {size[1]}"


class TestEllipseScaleHandles:
    """Tests for Ellipse ROI scale handles (8 handles: 4 midpoints + 4 corners)."""

    @pytest.fixture
    def mock_kernel(self, qapp):
        """Create a minimal SimpleMaskKernel mock for testing."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        kernel.hdl = MagicMock()
        kernel.hdl.add_item = MagicMock(return_value="roi_0")
        kernel.hdl.roi_items = {}
        kernel.hdl.scene = MagicMock()
        kernel.shape = (512, 512)
        kernel.metadata = {"bcx": 256, "bcy": 256}
        return kernel

    def test_ellipse_has_eight_handles(self, mock_kernel):
        """Ellipse ROI should have 8 scale handles."""
        roi = mock_kernel.add_drawing(
            sl_type="Ellipse",
            sl_mode="exclusive",
        )

        assert roi is not None
        assert len(roi.handles) >= 8, (
            f"Expected at least 8 handles, got {len(roi.handles)}"
        )

    def test_ellipse_midpoint_handles_exist(self, mock_kernel):
        """Ellipse should have handles at 4 midpoints."""
        roi = mock_kernel.add_drawing(
            sl_type="Ellipse",
            sl_mode="exclusive",
        )

        assert roi is not None
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]

        # Original midpoint positions
        expected_midpoints = [
            (0.5, 0),  # bottom
            (0.5, 1),  # top
            (0, 0.5),  # left
            (1, 0.5),  # right
        ]

        for midpoint in expected_midpoints:
            found = any(
                np.allclose([h.x(), h.y()], midpoint, atol=0.01)
                for h in handle_positions
            )
            assert found, f"Midpoint handle at {midpoint} not found"

    def test_ellipse_corner_handles_geometry(self, mock_kernel):
        """Ellipse corner handles should be at geometrically correct positions."""
        roi = mock_kernel.add_drawing(
            sl_type="Ellipse",
            sl_mode="exclusive",
        )

        assert roi is not None
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]

        # Corner positions adjusted for ellipse (≈ 0.1464, 0.8536)
        # These are cos(45°)/2 = 0.3536, offset from center gives 0.5 ± 0.3536
        expected_corners = [
            (0.1464, 0.1464),
            (0.1464, 0.8536),
            (0.8536, 0.1464),
            (0.8536, 0.8536),
        ]

        for corner in expected_corners:
            found = any(
                np.allclose([h.x(), h.y()], corner, atol=0.02) for h in handle_positions
            )
            assert found, f"Corner handle at {corner} not found"


class TestCircleScaleHandles:
    """Tests for Circle ROI scale handles (2 handles for uniform scaling)."""

    @pytest.fixture
    def mock_kernel(self, qapp):
        """Create a minimal SimpleMaskKernel mock for testing."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        kernel.hdl = MagicMock()
        kernel.hdl.add_item = MagicMock(return_value="roi_0")
        kernel.hdl.roi_items = {}
        kernel.hdl.scene = MagicMock()
        kernel.shape = (512, 512)
        kernel.metadata = {"bcx": 256, "bcy": 256}
        return kernel

    def test_circle_has_two_handles(self, mock_kernel):
        """Circle ROI should have 2 scale handles for uniform scaling."""
        roi = mock_kernel.add_drawing(
            sl_type="Circle",
            second_point=(256, 280),  # Required for Circle
            sl_mode="exclusive",
        )

        assert roi is not None
        # CircleROI starts with default handles, we add 2 more
        # Check that at least 2 handles exist at the expected positions
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]
        assert len(handle_positions) >= 2, (
            f"Expected at least 2 handles, got {len(handle_positions)}"
        )

    def test_circle_handles_are_opposite(self, mock_kernel):
        """Circle handles should be at opposite sides (top and bottom)."""
        roi = mock_kernel.add_drawing(
            sl_type="Circle",
            second_point=(256, 280),
            sl_mode="exclusive",
        )

        assert roi is not None
        handle_positions = [h["pos"] for h in roi.handles if "pos" in h]

        # Handles at (0.5, 0) and (0.5, 1) - opposite vertical positions
        expected_handles = [
            (0.5, 0),  # bottom
            (0.5, 1),  # top
        ]

        for pos in expected_handles:
            found = any(
                np.allclose([h.x(), h.y()], pos, atol=0.01) for h in handle_positions
            )
            assert found, f"Handle at {pos} not found"

    def test_circle_handles_pivot_at_center(self, mock_kernel):
        """Circle handles should pivot around center (0.5, 0.5)."""
        roi = mock_kernel.add_drawing(
            sl_type="Circle",
            second_point=(256, 280),
            sl_mode="exclusive",
        )

        assert roi is not None
        # Check that handle centers are at (0.5, 0.5)
        for h in roi.handles:
            if "center" in h:
                center = h["center"]
                np.testing.assert_allclose(
                    [center.x(), center.y()],
                    [0.5, 0.5],
                    atol=0.01,
                    err_msg="Handle center should be at (0.5, 0.5)",
                )

    def test_circle_without_second_point_uses_default_radius(self, mock_kernel):
        """Circle without second_point should use default radius of 10."""
        roi = mock_kernel.add_drawing(
            sl_type="Circle",
            second_point=None,
            sl_mode="exclusive",
        )

        # Circle with no second_point still creates ROI with default radius
        assert roi is not None
        # Default radius is 10, so size should be diameter (20, 20)
        size = roi.size()
        assert size[0] == 20.0, f"Expected diameter 20, got {size[0]}"


class TestHandleScalingBehavior:
    """Integration tests for scale handle behavior during interaction."""

    @pytest.fixture
    def mock_kernel(self, qapp):
        """Create a minimal SimpleMaskKernel mock for testing."""
        from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel

        kernel = SimpleMaskKernel()
        kernel.hdl = MagicMock()
        kernel.hdl.add_item = MagicMock(return_value="roi_0")
        kernel.hdl.roi_items = {}
        kernel.hdl.scene = MagicMock()
        kernel.shape = (512, 512)
        kernel.metadata = {"bcx": 256, "bcy": 256}
        return kernel

    @pytest.mark.parametrize(
        "sl_type,expected_min_handles",
        [
            ("Rectangle", 8),
            ("Ellipse", 8),
            ("Circle", 2),
        ],
    )
    def test_roi_type_handle_counts(self, mock_kernel, sl_type, expected_min_handles):
        """Verify each ROI type has the minimum expected handles."""
        # Circle needs second_point
        second_point = (256, 280) if sl_type == "Circle" else None

        roi = mock_kernel.add_drawing(
            sl_type=sl_type,
            second_point=second_point,
            sl_mode="exclusive",
        )

        if roi is not None:
            handle_count = len([h for h in roi.handles if "pos" in h])
            assert handle_count >= expected_min_handles, (
                f"{sl_type} expected at least {expected_min_handles} handles, got {handle_count}"
            )

    def test_all_handles_are_interactive(self, mock_kernel):
        """All added handles should be interactive (not static markers)."""
        roi = mock_kernel.add_drawing(
            sl_type="Rectangle",
            sl_mode="exclusive",
        )

        assert roi is not None
        # PyQtGraph handles are interactive by default when added via addScaleHandle
        for handle_info in roi.handles:
            # Verify handle is not marked as non-interactive
            assert handle_info.get("type") != "none", "Handle should be interactive"


class TestEllipseCornerPositionsScientific:
    """Scientific validation tests for ellipse handle positions."""

    @pytest.mark.scientific
    def test_ellipse_corner_positions_mathematically_correct(self):
        """Verify ellipse corner handle positions are mathematically derived from 45°."""
        # The corner positions should be at 45° angles on the ellipse
        # cos(45°) = sin(45°) ≈ 0.707
        # Normalized: (1 - 0.707) / 2 ≈ 0.1464 and (1 + 0.707) / 2 ≈ 0.8536

        expected_value = (1 - np.cos(np.pi / 4)) / 2
        np.testing.assert_allclose(expected_value, 0.1464, atol=0.001)

        complement = 1 - expected_value
        np.testing.assert_allclose(complement, 0.8536, atol=0.001)
