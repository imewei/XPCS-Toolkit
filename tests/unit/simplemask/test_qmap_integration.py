"""Unit tests for Q-map integration in SimpleMask window.

Tests the Q-map generation, display, and geometry parameter handling
in SimpleMaskWindow.
"""

import numpy as np
import pytest

from xpcsviewer.simplemask import SimpleMaskWindow
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel


class TestQmapGeneration:
    """Tests for Q-map computation from geometry parameters."""

    def test_kernel_computes_qmap_on_data_load(self):
        """Kernel should compute Q-map when data is loaded."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)

        assert kernel.qmap is not None
        assert "q" in kernel.qmap

    def test_kernel_qmap_shape_matches_detector(self):
        """Q-map should have same shape as detector."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((256, 512), dtype=np.float64)
        metadata = {
            "bcx": 256,
            "bcy": 128,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)

        assert kernel.qmap["q"].shape == (256, 512)

    def test_kernel_recomputes_qmap_on_parameter_update(self):
        """Q-map should be recomputed when geometry changes."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)
        original_q = kernel.qmap["q"].copy()

        # Change detector distance
        kernel.update_parameters({"det_dist": 10000.0})

        # Q-map should be different
        assert not np.allclose(kernel.qmap["q"], original_q)


class TestWindowGeometrySpinboxes:
    """Tests for geometry spinboxes in SimpleMask window."""

    def test_geometry_spinboxes_exist(self, qapp, qtbot):
        """Window should have geometry parameter spinboxes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "spin_bcx")
        assert hasattr(window, "spin_bcy")
        assert hasattr(window, "spin_det_dist")
        assert hasattr(window, "spin_pix_dim")
        assert hasattr(window, "spin_energy")

    def test_spinboxes_populated_from_metadata(self, qapp, qtbot):
        """Spinboxes should be populated from loaded metadata."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 123.45,
            "bcy": 67.89,
            "det_dist": 7500.0,
            "pix_dim": 0.055,
            "energy": 12.5,
        }

        window.load_from_viewer(detector_image, metadata)

        assert window.spin_bcx.value() == pytest.approx(123.45)
        assert window.spin_bcy.value() == pytest.approx(67.89)
        assert window.spin_det_dist.value() == pytest.approx(7500.0)
        assert window.spin_pix_dim.value() == pytest.approx(0.055)
        assert window.spin_energy.value() == pytest.approx(12.5)


class TestQmapButton:
    """Tests for Generate Q-Map toolbar action."""

    def test_generate_qmap_action_exists(self, qapp, qtbot):
        """Window should have Generate Q-Map toolbar action."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "action_generate_qmap")

    def test_toggle_qmap_action_exists(self, qapp, qtbot):
        """Window should have Show Q-Map toggle action in toolbar."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "action_toggle_qmap")


class TestQmapOverlay:
    """Tests for Q-map overlay display."""

    def test_qmap_overlay_toggle_flag(self, qapp, qtbot):
        """Window should track Q-map overlay state."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "_show_qmap_overlay")
        assert window._show_qmap_overlay is False

    def test_qmap_display_requires_data(self, qapp, qtbot):
        """Q-map toggle should require loaded data."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Try to show Q-map without data
        initial_state = window._show_qmap_overlay
        window._on_toggle_qmap()

        # Should not change state without data
        assert window._show_qmap_overlay == initial_state


class TestQmapExportSignal:
    """Tests for qmap_exported signal."""

    def test_qmap_exported_signal_exists(self, qapp, qtbot):
        """qmap_exported signal should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "qmap_exported")

    def test_export_partition_to_viewer_method_exists(self, qapp, qtbot):
        """export_partition_to_viewer method should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "export_partition_to_viewer")
        assert callable(window.export_partition_to_viewer)


class TestQmapValues:
    """Tests for Q-map value computation."""

    def test_q_values_positive(self):
        """Q values should be positive (or zero at beam center)."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)

        # Q values should be >= 0
        assert np.all(kernel.qmap["q"] >= 0)

    def test_q_minimum_near_beam_center(self):
        """Minimum Q should be near beam center."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        bcx, bcy = 100, 50
        metadata = {
            "bcx": bcx,
            "bcy": bcy,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)

        q_array = kernel.qmap["q"]
        min_q_idx = np.unravel_index(np.argmin(q_array), q_array.shape)

        # Minimum should be near beam center (within 2 pixels)
        assert abs(min_q_idx[0] - bcy) <= 2
        assert abs(min_q_idx[1] - bcx) <= 2

    def test_qmap_units_returned(self):
        """Q-map units should be returned."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)

        assert kernel.qmap_unit is not None
        assert "q" in kernel.qmap_unit
