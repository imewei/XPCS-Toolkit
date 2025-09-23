#!/usr/bin/env python3
"""
Test suite for the centralized plot constants module.
Tests the functionality and consistency of the consolidated plotting constants.
"""

import sys
import time
import unittest

try:
    from xpcs_toolkit.plothandler.plot_constants import (
        BASIC_COLORS,
        EXTENDED_MARKERS,
        MATPLOTLIB_COLORS_HEX,
        MATPLOTLIB_COLORS_RGB,
        get_color_cycle,
        get_color_marker,
        get_marker_cycle,
    )

    PLOT_CONSTANTS_AVAILABLE = True
except ImportError:
    PLOT_CONSTANTS_AVAILABLE = False


@unittest.skipUnless(PLOT_CONSTANTS_AVAILABLE, "Plot constants module not available")
class TestPlotConstants(unittest.TestCase):
    """Test suite for centralized plot constants."""

    def test_constants_exist_and_non_empty(self):
        """Test that all expected constants exist and are non-empty."""
        constants = [
            BASIC_COLORS,
            EXTENDED_MARKERS,
            MATPLOTLIB_COLORS_HEX,
            MATPLOTLIB_COLORS_RGB,
        ]

        for constant in constants:
            self.assertIsInstance(constant, (list, tuple))
            self.assertGreater(len(constant), 0, "Constant should not be empty")

    def test_color_consistency(self):
        """Test that hex and RGB color lists have consistent lengths."""
        self.assertEqual(
            len(MATPLOTLIB_COLORS_HEX),
            len(MATPLOTLIB_COLORS_RGB),
            "HEX and RGB color lists should have same length",
        )

    def test_hex_color_format(self):
        """Test that all hex colors are properly formatted."""
        for i, color in enumerate(MATPLOTLIB_COLORS_HEX):
            self.assertIsInstance(color, str, f"Color {i} should be string")
            self.assertTrue(
                color.startswith("#") and len(color) == 7,
                f"Color {i} ({color}) should be valid hex format #RRGGBB",
            )
            # Check that characters after # are valid hex
            hex_part = color[1:]
            try:
                int(hex_part, 16)
            except ValueError:
                self.fail(f"Color {i} ({color}) contains invalid hex characters")

    def test_rgb_color_format(self):
        """Test that all RGB colors are properly formatted."""
        for i, color in enumerate(MATPLOTLIB_COLORS_RGB):
            self.assertIsInstance(
                color, (list, tuple), f"RGB color {i} should be list/tuple"
            )
            self.assertEqual(len(color), 3, f"RGB color {i} should have 3 components")

            for j, component in enumerate(color):
                self.assertIsInstance(
                    component,
                    (int, float),
                    f"RGB color {i} component {j} should be numeric",
                )
                self.assertGreaterEqual(
                    component, 0, f"RGB color {i} component {j} should be non-negative"
                )
                self.assertLessEqual(
                    component, 255, f"RGB color {i} component {j} should be <= 255"
                )

    def test_marker_format(self):
        """Test that all markers are valid strings."""
        for i, marker in enumerate(EXTENDED_MARKERS):
            self.assertIsInstance(marker, str, f"Marker {i} should be string")
            self.assertGreater(len(marker), 0, f"Marker {i} should not be empty string")

    def test_get_color_marker_function(self):
        """Test the get_color_marker function."""
        # Test valid indices
        for i in range(5):
            color, marker = get_color_marker(i, "matplotlib")
            self.assertIsInstance(color, str)
            self.assertIsInstance(marker, str)
            self.assertTrue(color.startswith("#"))

        # Test cycling behavior
        color1, _marker1 = get_color_marker(0, "matplotlib")
        color2, _marker2 = get_color_marker(len(MATPLOTLIB_COLORS_HEX), "matplotlib")
        self.assertEqual(color1, color2, "Colors should cycle properly")

        # Test marker cycling with correct marker set for matplotlib backend
        from xpcs_toolkit.plothandler.plot_constants import MATPLOTLIB_MARKERS

        marker1_cycle = get_color_marker(0, "matplotlib")[1]
        marker2_cycle = get_color_marker(len(MATPLOTLIB_MARKERS), "matplotlib")[1]
        self.assertEqual(marker1_cycle, marker2_cycle, "Markers should cycle properly")

    def test_get_color_cycle_function(self):
        """Test the get_color_cycle function."""
        # Test hex format
        hex_colors = get_color_cycle("matplotlib", "hex")
        self.assertIsInstance(hex_colors, (list, tuple))
        self.assertEqual(hex_colors, MATPLOTLIB_COLORS_HEX)

        # Test RGB format
        rgb_colors = get_color_cycle("matplotlib", "rgb")
        self.assertIsInstance(rgb_colors, (list, tuple))
        self.assertEqual(rgb_colors, MATPLOTLIB_COLORS_RGB)

        # Test invalid format
        with self.assertRaises(ValueError):
            get_color_cycle("matplotlib", "invalid_format")

        # Test invalid library - currently backend parameter is not validated
        # This test documents current behavior, not ideal behavior
        colors = get_color_cycle("invalid_library", "hex")
        self.assertEqual(colors, MATPLOTLIB_COLORS_HEX)  # Backend ignored currently

    def test_get_marker_cycle_function(self):
        """Test the get_marker_cycle function."""
        from xpcs_toolkit.plothandler.plot_constants import MATPLOTLIB_MARKERS

        # Test matplotlib backend
        markers = get_marker_cycle("matplotlib")
        self.assertIsInstance(markers, (list, tuple))
        self.assertEqual(markers, MATPLOTLIB_MARKERS)

        # Test extended backend
        extended_markers = get_marker_cycle("extended")
        self.assertIsInstance(extended_markers, (list, tuple))
        self.assertEqual(extended_markers, EXTENDED_MARKERS)

        # Test invalid library
        with self.assertRaises(ValueError):
            get_marker_cycle("invalid_library")

    def test_backwards_compatibility(self):
        """Test that the module provides backwards compatibility for common use cases."""
        # Test that we can get colors and markers as expected by existing code
        color, marker = get_color_marker(0, "matplotlib")
        self.assertIsInstance(color, str)
        self.assertIsInstance(marker, str)

        # Test that basic colors are accessible (single letter codes)
        self.assertIn("r", BASIC_COLORS)  # red
        self.assertIn("b", BASIC_COLORS)  # blue
        self.assertIn("g", BASIC_COLORS)  # green

    def test_performance_constants_access(self):
        """Test that accessing constants is performant (no heavy computation)."""

        # Time multiple accesses to ensure they're fast
        start_time = time.perf_counter()
        for i in range(1000):
            _ = MATPLOTLIB_COLORS_HEX[i % len(MATPLOTLIB_COLORS_HEX)]
            _ = MATPLOTLIB_COLORS_RGB[i % len(MATPLOTLIB_COLORS_RGB)]
            _ = EXTENDED_MARKERS[i % len(EXTENDED_MARKERS)]
        end_time = time.perf_counter()

        # Should be very fast (less than 50ms for 1000 accesses in CI, 10ms locally)
        elapsed_time = end_time - start_time
        # Use more lenient threshold for CI environments
        import os

        max_time = 0.05 if os.environ.get("CI") else 0.01
        self.assertLess(
            elapsed_time,
            max_time,
            f"Constants access too slow: {elapsed_time:.3f}s for 1000 accesses (max: {max_time:.3f}s)",
        )

    def test_scientific_plot_compatibility(self):
        """Test that constants work well for scientific plotting scenarios."""
        # Test that we have enough colors/markers for typical scientific plots
        self.assertGreaterEqual(
            len(MATPLOTLIB_COLORS_HEX),
            10,
            "Should have at least 10 colors for scientific plots",
        )
        self.assertGreaterEqual(
            len(EXTENDED_MARKERS),
            10,
            "Should have at least 10 markers for scientific plots",
        )

        # Test that colors are distinguishable (no duplicates)
        unique_hex_colors = set(MATPLOTLIB_COLORS_HEX)
        self.assertEqual(
            len(unique_hex_colors),
            len(MATPLOTLIB_COLORS_HEX),
            "All hex colors should be unique",
        )

        unique_markers = set(EXTENDED_MARKERS)
        self.assertEqual(
            len(unique_markers), len(EXTENDED_MARKERS), "All markers should be unique"
        )

    def test_memory_efficiency(self):
        """Test that constants don't use excessive memory."""

        # Check size of constant lists
        hex_size = sys.getsizeof(MATPLOTLIB_COLORS_HEX)
        rgb_size = sys.getsizeof(MATPLOTLIB_COLORS_RGB)
        marker_size = sys.getsizeof(EXTENDED_MARKERS)

        # Should be reasonable sizes (less than 10KB each)
        self.assertLess(hex_size, 10240, "HEX colors list too large")
        self.assertLess(rgb_size, 10240, "RGB colors list too large")
        self.assertLess(marker_size, 10240, "Markers list too large")


if __name__ == "__main__":
    unittest.main()
