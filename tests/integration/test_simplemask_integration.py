"""Integration tests for SimpleMask to XPCS Viewer data flow.

Tests the mask and partition export/import functionality between
SimpleMaskWindow and XpcsViewer.
"""

import numpy as np
import pytest
from PySide6 import QtWidgets

from xpcsviewer.simplemask import SimpleMaskWindow


class TestMaskSignalExport:
    """Tests for mask_exported signal functionality."""

    def test_mask_exported_signal_emits_array(self, qapp, qtbot):
        """mask_exported signal should emit numpy array."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        # Load test data
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        window.load_from_viewer(detector_image, metadata)

        # Capture signal
        received_masks = []

        def capture_mask(mask):
            received_masks.append(mask)

        window.mask_exported.connect(capture_mask)

        # Export mask
        window.export_mask_to_viewer()

        assert len(received_masks) == 1
        assert isinstance(received_masks[0], np.ndarray)

    def test_mask_exported_signal_emits_correct_shape(self, qapp, qtbot):
        """Exported mask should have same shape as detector."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        detector_image = np.ones((256, 512), dtype=np.float64)
        metadata = {
            "bcx": 256,
            "bcy": 128,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        window.load_from_viewer(detector_image, metadata)

        received_masks = []
        window.mask_exported.connect(lambda m: received_masks.append(m))

        window.export_mask_to_viewer()

        assert received_masks[0].shape == (256, 512)

    def test_mask_exported_not_emitted_without_data(self, qapp, qtbot):
        """mask_exported should not emit when no data loaded."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        received_masks = []
        window.mask_exported.connect(lambda m: received_masks.append(m))

        # Try to export without data
        window.export_mask_to_viewer()

        assert len(received_masks) == 0


class TestPartitionSignalExport:
    """Tests for qmap_exported signal functionality."""

    def test_qmap_exported_signal_exists(self, qapp, qtbot):
        """qmap_exported signal should exist on window."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "qmap_exported")

    def test_partition_export_with_data(self, qapp, qtbot):
        """export_partition_to_viewer should emit partition dict."""
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

        # Compute partition first
        if window.kernel is not None:
            window.kernel.compute_partition()

        received_partitions = []
        window.qmap_exported.connect(lambda p: received_partitions.append(p))

        window.export_partition_to_viewer()

        # Should emit if partition was computed
        if window.kernel is not None and window.kernel.new_partition is not None:
            assert len(received_partitions) == 1
            assert isinstance(received_partitions[0], dict)


class TestMaskExportContent:
    """Tests for mask content during export."""

    def test_exported_mask_is_boolean(self, qapp, qtbot):
        """Exported mask should be boolean array."""
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

        received_masks = []
        window.mask_exported.connect(lambda m: received_masks.append(m))

        window.export_mask_to_viewer()

        assert received_masks[0].dtype == bool

    def test_exported_mask_default_is_all_true(self, qapp, qtbot):
        """Default mask (no edits) should be all True."""
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

        received_masks = []
        window.mask_exported.connect(lambda m: received_masks.append(m))

        window.export_mask_to_viewer()

        # Default mask should keep all pixels
        assert np.all(received_masks[0])


class TestApplyToViewerButton:
    """Tests for Apply to Viewer button in toolbar."""

    def test_apply_to_viewer_toolbar_action_exists(self, qapp, qtbot):
        """Toolbar should have Apply to Viewer action."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "action_apply_viewer")
        assert window.action_apply_viewer is not None

    def test_apply_to_viewer_menu_action_exists(self, qapp, qtbot):
        """File menu should have Apply to Viewer action."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "action_apply_to_viewer")
        assert window.action_apply_to_viewer is not None


class TestStatusBarUpdates:
    """Tests for status bar messages during export."""

    def test_export_updates_status_bar(self, qapp, qtbot):
        """Exporting mask should update status bar."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        window.load_from_viewer(detector_image, metadata)

        window.export_mask_to_viewer()

        # Status bar should show export message
        status_text = window.status_bar.currentMessage()
        assert "export" in status_text.lower() or "viewer" in status_text.lower()
