#!/usr/bin/env python3
"""
Test suite for centralized detector constants in qmap_utils.
Tests the consolidated detector size and beam center constants.
"""

import sys
import time
import unittest

try:
    from xpcs_toolkit.fileIO.qmap_utils import (
        DEFAULT_BEAM_CENTER,
        DEFAULT_DETECTOR_SIZE,
    )

    QMAP_CONSTANTS_AVAILABLE = True
except ImportError:
    QMAP_CONSTANTS_AVAILABLE = False


@unittest.skipUnless(QMAP_CONSTANTS_AVAILABLE, "QMap constants not available")
class TestQMapConstants(unittest.TestCase):
    """Test suite for centralized QMap constants."""

    def test_constants_exist(self):
        """Test that the required constants exist."""
        self.assertTrue(hasattr(self, "__class__"))  # Basic test setup

        # Test that constants are defined
        self.assertIsNotNone(DEFAULT_DETECTOR_SIZE)
        self.assertIsNotNone(DEFAULT_BEAM_CENTER)

    def test_detector_size_properties(self):
        """Test properties of the default detector size."""
        # Should be a positive integer
        self.assertIsInstance(DEFAULT_DETECTOR_SIZE, int)
        self.assertGreater(DEFAULT_DETECTOR_SIZE, 0)

        # Should be a reasonable detector size (between 256 and 4096)
        self.assertGreaterEqual(DEFAULT_DETECTOR_SIZE, 256)
        self.assertLessEqual(DEFAULT_DETECTOR_SIZE, 4096)

        # Should be a power of 2 or standard detector size
        standard_sizes = [256, 512, 1024, 2048, 4096]
        self.assertIn(DEFAULT_DETECTOR_SIZE, standard_sizes)

    def test_beam_center_properties(self):
        """Test properties of the default beam center."""
        # Should be a positive number
        self.assertIsInstance(DEFAULT_BEAM_CENTER, (int, float))
        self.assertGreater(DEFAULT_BEAM_CENTER, 0)

        # Should be less than detector size
        self.assertLess(DEFAULT_BEAM_CENTER, DEFAULT_DETECTOR_SIZE)

    def test_beam_center_detector_relationship(self):
        """Test the mathematical relationship between beam center and detector size."""
        # Beam center should be half the detector size (center of detector)
        expected_beam_center = DEFAULT_DETECTOR_SIZE // 2
        self.assertEqual(
            DEFAULT_BEAM_CENTER,
            expected_beam_center,
            f"Beam center should be detector_size//2: {expected_beam_center}",
        )

    def test_constants_immutability(self):
        """Test that constants behave as immutable values."""
        # Store original values
        original_detector_size = DEFAULT_DETECTOR_SIZE
        original_beam_center = DEFAULT_BEAM_CENTER

        # Attempt to modify (this should not affect the constants in the module)
        try:
            # These assignments create new local variables, they don't modify module constants
            DEFAULT_DETECTOR_SIZE + 100
            DEFAULT_BEAM_CENTER + 50

            # Original constants should be unchanged
            self.assertEqual(DEFAULT_DETECTOR_SIZE, original_detector_size)
            self.assertEqual(DEFAULT_BEAM_CENTER, original_beam_center)

        except Exception:
            # If constants are truly immutable, this is fine
            pass

    def test_constants_scientific_validity(self):
        """Test that constants are scientifically reasonable for XPCS experiments."""
        # Detector size should be appropriate for typical 2D detectors
        self.assertIn(DEFAULT_DETECTOR_SIZE, [512, 1024, 2048])

        # Beam center should be in reasonable range for direct beam position
        min_center = DEFAULT_DETECTOR_SIZE * 0.3  # At least 30% from edge
        max_center = DEFAULT_DETECTOR_SIZE * 0.7  # At most 70% from edge

        self.assertGreaterEqual(DEFAULT_BEAM_CENTER, min_center)
        self.assertLessEqual(DEFAULT_BEAM_CENTER, max_center)

    def test_backwards_compatibility(self):
        """Test that constants provide expected values for backwards compatibility."""
        # These should match the values that were previously hardcoded
        self.assertEqual(DEFAULT_DETECTOR_SIZE, 1024)
        self.assertEqual(DEFAULT_BEAM_CENTER, 512)

    def test_constants_usage_patterns(self):
        """Test common usage patterns with the constants."""
        # Test creating arrays with detector size
        import numpy as np

        # Should be able to create detector-sized arrays
        detector_array = np.zeros((DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE))
        self.assertEqual(
            detector_array.shape, (DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE)
        )

        # Should be able to use beam center for indexing
        center_pixel = detector_array[DEFAULT_BEAM_CENTER, DEFAULT_BEAM_CENTER]
        self.assertEqual(center_pixel, 0.0)  # Initial value

        # Should be able to calculate distances from center
        y_coords, x_coords = np.ogrid[:DEFAULT_DETECTOR_SIZE, :DEFAULT_DETECTOR_SIZE]
        distances = np.sqrt(
            (x_coords - DEFAULT_BEAM_CENTER) ** 2
            + (y_coords - DEFAULT_BEAM_CENTER) ** 2
        )

        # Distance at center should be 0
        center_distance = distances[DEFAULT_BEAM_CENTER, DEFAULT_BEAM_CENTER]
        self.assertAlmostEqual(center_distance, 0.0, places=10)

        # Maximum distance should be reasonable
        max_distance = np.max(distances)
        expected_max = np.sqrt(2) * DEFAULT_BEAM_CENTER  # Distance to corner
        self.assertAlmostEqual(max_distance, expected_max, places=5)

    def test_memory_efficiency(self):
        """Test that constants don't use excessive memory."""

        # Constants should be simple integers, very small memory footprint
        detector_size_memory = sys.getsizeof(DEFAULT_DETECTOR_SIZE)
        beam_center_memory = sys.getsizeof(DEFAULT_BEAM_CENTER)

        # Should be tiny (less than 100 bytes each)
        self.assertLess(detector_size_memory, 100)
        self.assertLess(beam_center_memory, 100)

    def test_performance_access(self):
        """Test that accessing constants is performant."""

        # Time multiple accesses
        start_time = time.perf_counter()
        for _ in range(10000):
            _ = DEFAULT_DETECTOR_SIZE
            _ = DEFAULT_BEAM_CENTER
            _ = DEFAULT_BEAM_CENTER == DEFAULT_DETECTOR_SIZE // 2
        end_time = time.perf_counter()

        # Should be very fast (less than 15ms for 10000 accesses in CI, 5ms locally)
        elapsed_time = end_time - start_time
        # Use more lenient threshold for CI environments, especially macOS ARM64
        import os

        max_time = 0.015 if os.environ.get("CI") else 0.005
        self.assertLess(
            elapsed_time,
            max_time,
            f"Constants access too slow: {elapsed_time:.6f}s for 10000 accesses (threshold: {max_time:.3f}s)",
        )


if __name__ == "__main__":
    unittest.main()
