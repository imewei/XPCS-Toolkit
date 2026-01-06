"""Unit tests for SimpleMask partition (Q-binning) functionality.

Tests the partition computation, display, and export functionality
in SimpleMaskWindow and SimpleMaskKernel.
"""

import numpy as np
import pytest

from xpcsviewer.simplemask import SimpleMaskWindow
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel


class TestPartitionComputation:
    """Tests for partition computation in kernel."""

    def test_compute_partition_returns_dict(self):
        """compute_partition should return dictionary."""
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
        partition = kernel.compute_partition()

        assert isinstance(partition, dict)

    def test_partition_contains_dynamic_roi_map(self):
        """Partition should contain dynamic_roi_map."""
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
        partition = kernel.compute_partition()

        assert "dynamic_roi_map" in partition

    def test_partition_contains_static_roi_map(self):
        """Partition should contain static_roi_map."""
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
        partition = kernel.compute_partition()

        assert "static_roi_map" in partition

    def test_partition_roi_maps_have_correct_shape(self):
        """ROI maps should have same shape as detector."""
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
        partition = kernel.compute_partition()

        assert partition["dynamic_roi_map"].shape == (100, 200)
        assert partition["static_roi_map"].shape == (100, 200)

    def test_partition_respects_dq_num(self):
        """Number of dynamic Q-bins should match dq_num."""
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
        partition = kernel.compute_partition(dq_num=5, dp_num=1)

        # Number of unique Q-bins should be <= dq_num
        # (may be less if some bins have no pixels)
        unique_bins = np.unique(partition["dynamic_roi_map"])
        unique_bins = unique_bins[unique_bins > 0]  # Exclude zero/masked
        assert len(unique_bins) <= 5


class TestPartitionMetadata:
    """Tests for partition metadata."""

    def test_partition_contains_beam_center(self):
        """Partition should contain beam center."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 123.0,
            "bcy": 45.0,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }

        kernel.read_data(detector_image, metadata)
        partition = kernel.compute_partition()

        assert "beam_center_x" in partition
        assert "beam_center_y" in partition
        assert partition["beam_center_x"] == pytest.approx(123.0)
        assert partition["beam_center_y"] == pytest.approx(45.0)

    def test_partition_contains_mask(self):
        """Partition should contain mask array."""
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
        partition = kernel.compute_partition()

        assert "mask" in partition
        assert partition["mask"].shape == (100, 200)


class TestWindowPartitionControls:
    """Tests for partition controls in SimpleMask window."""

    def test_partition_spinboxes_exist(self, qapp, qtbot):
        """Window should have partition parameter spinboxes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "spin_dq_num")
        assert hasattr(window, "spin_sq_num")
        assert hasattr(window, "spin_dp_num")
        assert hasattr(window, "spin_sp_num")

    def test_compute_partition_button_exists(self, qapp, qtbot):
        """Window should have Compute Partition button."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "btn_compute_partition")

    def test_toggle_partition_button_exists(self, qapp, qtbot):
        """Window should have Show Partition toggle button."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "btn_toggle_partition")

    def test_export_partition_button_exists(self, qapp, qtbot):
        """Window should have Export to Viewer button."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "btn_export_partition")


class TestPartitionOverlay:
    """Tests for partition overlay display."""

    def test_partition_overlay_flag(self, qapp, qtbot):
        """Window should track partition overlay state."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "_show_partition_overlay")
        assert window._show_partition_overlay is False

    def test_partition_display_requires_data(self, qapp, qtbot):
        """Partition toggle should require computed partition."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Try to show partition without data
        initial_state = window._show_partition_overlay
        window._on_toggle_partition()

        # Should not change state without partition
        assert window._show_partition_overlay == initial_state


class TestPartitionExport:
    """Tests for partition export functionality."""

    def test_export_partition_to_viewer_exists(self, qapp, qtbot):
        """export_partition_to_viewer method should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "export_partition_to_viewer")
        assert callable(window.export_partition_to_viewer)

    def test_qmap_exported_signal_emits_partition(self, qapp, qtbot):
        """qmap_exported signal should emit partition dict."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        window.load_from_viewer(detector_image, metadata)

        # Compute partition
        window.kernel.compute_partition()

        received_partitions = []
        window.qmap_exported.connect(lambda p: received_partitions.append(p))

        window.export_partition_to_viewer()

        assert len(received_partitions) == 1
        assert isinstance(received_partitions[0], dict)


class TestPartitionStorage:
    """Tests for partition storage in kernel."""

    def test_kernel_stores_partition(self):
        """Kernel should store computed partition."""
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
        kernel.compute_partition()

        assert kernel.new_partition is not None

    def test_kernel_partition_initially_none(self):
        """Kernel partition should be None before computation."""
        kernel = SimpleMaskKernel()
        assert kernel.new_partition is None
