"""GUI tests for SimpleMask window integration.

Tests the SimpleMask window launch, data loading, and basic interactions
from the XPCS Viewer.
"""

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from xpcsviewer.simplemask import SimpleMaskWindow


class TestSimpleMaskWindowCreation:
    """Tests for SimpleMask window creation and initialization."""

    def test_window_creation(self, qapp, qtbot):
        """SimpleMask window should be created successfully."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert window is not None
        assert window.windowTitle() == "Mask Editor"

    def test_window_minimum_size(self, qapp, qtbot):
        """SimpleMask window should have minimum size set."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        assert window.minimumWidth() >= 800
        assert window.minimumHeight() >= 600

    def test_window_has_image_view(self, qapp, qtbot):
        """SimpleMask window should have ImageView widget."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "image_view")
        assert window.image_view is not None

    def test_window_has_status_bar(self, qapp, qtbot):
        """SimpleMask window should have status bar."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "status_bar")
        assert window.status_bar is not None


class TestSimpleMaskDataLoading:
    """Tests for loading detector data into SimpleMask."""

    def test_load_none_data(self, qapp, qtbot):
        """Loading None should show empty canvas message."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        window.load_from_viewer(None, None)

        # Check info label contains message about no data
        assert (
            "No data" in window.info_label.text()
            or "no" in window.info_label.text().lower()
        )

    def test_load_valid_detector_image(self, qapp, qtbot):
        """Loading valid detector image should display it."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Create synthetic detector image
        detector_image = np.random.poisson(100, (512, 1024)).astype(np.float64)
        metadata = {
            "bcx": 512,
            "bcy": 256,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
            "shape": (512, 1024),
        }

        window.load_from_viewer(detector_image, metadata)

        # Check that kernel was initialized
        assert window.kernel is not None
        # Check that image was loaded
        assert window._detector_image is not None
        assert window._detector_image.shape == (512, 1024)

    def test_load_updates_info_label(self, qapp, qtbot):
        """Loading data should update info label with image info."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        detector_image = np.ones((256, 512), dtype=np.float64) * 100
        metadata = {
            "bcx": 256,
            "bcy": 128,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
            "shape": (256, 512),
        }

        window.load_from_viewer(detector_image, metadata)

        # Info label should contain dimensions
        info_text = window.info_label.text()
        assert "512" in info_text  # width
        assert "256" in info_text  # height


class TestSimpleMaskUnsavedChanges:
    """Tests for unsaved changes tracking."""

    def test_initial_no_unsaved_changes(self, qapp, qtbot):
        """New window should have no unsaved changes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert not window.has_unsaved_changes()

    def test_mark_unsaved_changes(self, qapp, qtbot):
        """Marking unsaved should update state and title."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        window.mark_unsaved()

        assert window.has_unsaved_changes()
        assert window.windowTitle().endswith("*")

    def test_mark_saved_clears_flag(self, qapp, qtbot):
        """Marking saved should clear unsaved state."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        window.mark_unsaved()
        window.mark_saved()

        assert not window.has_unsaved_changes()
        assert not window.windowTitle().endswith("*")


class TestSimpleMaskSignals:
    """Tests for SimpleMask signal emission."""

    def test_mask_exported_signal_exists(self, qapp, qtbot):
        """mask_exported signal should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "mask_exported")

    def test_qmap_exported_signal_exists(self, qapp, qtbot):
        """qmap_exported signal should exist."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)

        assert hasattr(window, "qmap_exported")


class TestSimpleMaskFromViewer:
    """Tests for launching SimpleMask from XPCS Viewer."""

    def test_mask_editor_tab_exists(self, gui_main_window, qtbot):
        """Mask Editor tab should exist in XPCS Viewer."""
        # Find the Mask Editor tab
        tab_widget = gui_main_window.tabWidget
        mask_editor_index = -1

        for i in range(tab_widget.count()):
            if "mask" in tab_widget.tabText(i).lower():
                mask_editor_index = i
                break

        assert mask_editor_index >= 0, "Mask Editor tab not found"

    def test_open_simplemask_creates_window(self, gui_main_window, qtbot):
        """open_simplemask should create SimpleMask window."""
        # Ensure no window exists
        gui_main_window._simplemask_window = None

        # Call open_simplemask
        gui_main_window.open_simplemask()
        qtbot.wait(100)

        # Check window was created
        assert gui_main_window._simplemask_window is not None
        assert isinstance(gui_main_window._simplemask_window, SimpleMaskWindow)

    def test_open_simplemask_shows_window(self, gui_main_window, qtbot):
        """open_simplemask should show the window."""
        gui_main_window._simplemask_window = None

        gui_main_window.open_simplemask()
        qtbot.wait(100)

        assert gui_main_window._simplemask_window.isVisible()

    def test_open_simplemask_reuses_window(self, gui_main_window, qtbot):
        """Calling open_simplemask twice should reuse the window."""
        gui_main_window._simplemask_window = None

        gui_main_window.open_simplemask()
        qtbot.wait(50)
        first_window = gui_main_window._simplemask_window

        gui_main_window.open_simplemask()
        qtbot.wait(50)
        second_window = gui_main_window._simplemask_window

        assert first_window is second_window

    def test_simplemask_window_parent_reference(self, gui_main_window, qtbot):
        """SimpleMask window should have reference to parent viewer."""
        gui_main_window._simplemask_window = None

        gui_main_window.open_simplemask()
        qtbot.wait(100)

        assert gui_main_window._simplemask_window.parent_viewer is gui_main_window


class TestSimpleMaskCloseEvent:
    """Tests for SimpleMask window close behavior."""

    def test_close_without_unsaved_changes(self, qapp, qtbot):
        """Window should close without prompt when no unsaved changes."""
        window = SimpleMaskWindow()
        qtbot.addWidget(window)
        window.show()
        qtbot.wait(50)

        # Close should succeed without prompt
        window.close()
        qtbot.wait(50)

        assert not window.isVisible()
