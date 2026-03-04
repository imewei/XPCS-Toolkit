"""Tests for main window functionality and tab management.

This module tests the main XpcsViewer window, including initialization,
tab switching, menu actions, and status updates.
"""

from unittest.mock import patch

import pytest
from PySide6 import QtCore, QtWidgets

from xpcsviewer.xpcs_viewer import XpcsViewer, tab_mapping


class TestMainWindow:
    """Test suite for main window functionality."""

    @pytest.mark.gui
    def test_main_window_initialization(self, gui_main_window, qtbot):
        """Test main window initializes correctly."""
        window = gui_main_window

        # Check window is visible and has correct title
        assert window.isVisible()
        assert "XPCS" in window.windowTitle()

        # Check main UI elements exist
        assert hasattr(window, "centralWidget")
        assert hasattr(window, "statusbar")

        # Verify tab widget exists and has expected tabs
        tab_widget = window.findChild(QtWidgets.QTabWidget)
        assert tab_widget is not None
        assert tab_widget.count() >= len(tab_mapping)

    @pytest.mark.gui
    def test_status_bar_updates(self, gui_main_window, qtbot):
        """Test status bar message updates."""
        window = gui_main_window
        status_bar = window.statusBar()

        # Mock status update
        test_message = "Test status message"
        window.statusBar().showMessage(test_message)
        qtbot.wait(50)

        # Verify message is displayed
        assert test_message in status_bar.currentMessage()


class TestTabManagement:
    """Test suite for tab-specific functionality."""

    @pytest.mark.gui
    def test_saxs_2d_tab(self, gui_main_window, qtbot, gui_test_helpers):
        """Test SAXS 2D analysis tab functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to SAXS 2D tab
        saxs_2d_index = 0  # First tab is typically SAXS 2D
        gui_test_helpers.click_tab(qtbot, tab_widget, saxs_2d_index)

        # Look for image view or plot widget
        current_widget = tab_widget.currentWidget()

        # Check for expected widgets (image display, controls, etc.)
        plot_widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(plot_widgets) > 0

    @pytest.mark.gui
    def test_g2_analysis_tab(self, gui_main_window, qtbot, gui_test_helpers):
        """Test G2 correlation analysis tab functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to G2 tab (typically tab index 4)
        g2_tab_index = 4
        if g2_tab_index < tab_widget.count():
            gui_test_helpers.click_tab(qtbot, tab_widget, g2_tab_index)

            current_widget = tab_widget.currentWidget()

            # Look for G2-specific elements
            plot_widgets = current_widget.findChildren(QtWidgets.QWidget)
            assert len(plot_widgets) > 0

    @pytest.mark.gui
    def test_stability_tab(self, gui_main_window, qtbot, gui_test_helpers):
        """Test stability analysis tab functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to stability tab
        stability_tab_index = 2  # Based on tab_mapping
        if stability_tab_index < tab_widget.count():
            gui_test_helpers.click_tab(qtbot, tab_widget, stability_tab_index)

            current_widget = tab_widget.currentWidget()
            assert current_widget is not None

    @pytest.mark.gui
    def test_twotime_tab(self, gui_main_window, qtbot, gui_test_helpers):
        """Test two-time correlation tab functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to two-time tab
        twotime_tab_index = 6  # Based on tab_mapping
        if twotime_tab_index < tab_widget.count():
            gui_test_helpers.click_tab(qtbot, tab_widget, twotime_tab_index)

            current_widget = tab_widget.currentWidget()
            assert current_widget is not None


class TestWindowStateManagement:
    """Test suite for window state persistence and management."""

    @pytest.mark.gui
    def test_tab_state_persistence(self, gui_main_window, qtbot, gui_test_helpers):
        """Test that tab states are maintained during switching."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Record initial tab
        initial_tab = tab_widget.currentIndex()

        # Switch to different tab
        new_tab = (initial_tab + 1) % tab_widget.count()
        gui_test_helpers.click_tab(qtbot, tab_widget, new_tab)

        # Verify tab changed
        assert tab_widget.currentIndex() == new_tab

        # Switch back
        gui_test_helpers.click_tab(qtbot, tab_widget, initial_tab)
        assert tab_widget.currentIndex() == initial_tab

    @pytest.mark.gui
    def test_window_close_behavior(self, qapp, qtbot, mock_viewer_kernel):
        """Test window closing behavior."""
        # Create a temporary window for close testing
        with patch("xpcsviewer.xpcs_viewer.ViewerKernel") as mock_vk_class:
            mock_vk_class.return_value = mock_viewer_kernel
            window = XpcsViewer(path=None)
            window.vk = mock_viewer_kernel
            qtbot.addWidget(window)
            window.show()
            qtbot.wait(50)

            # Test close
            window.close()
            qtbot.wait(50)

            # Verify window is closed
            assert not window.isVisible()

    @pytest.mark.gui
    def test_multiple_window_instances(self, qapp, qtbot, mock_viewer_kernel):
        """Test behavior with multiple window instances."""
        windows = []

        try:
            # Create multiple windows
            for _i in range(2):
                with patch("xpcsviewer.xpcs_viewer.ViewerKernel") as mock_vk_class:
                    mock_vk_class.return_value = mock_viewer_kernel
                    window = XpcsViewer(path=None)
                    window.vk = mock_viewer_kernel
                    qtbot.addWidget(window)
                    window.show()
                    windows.append(window)
                    qtbot.wait(50)

            # Verify both windows are visible
            for window in windows:
                assert window.isVisible()

        finally:
            # Clean up
            for window in windows:
                window.close()


class TestSignalsAndSlots:
    """Test suite for Qt signals and slots integration."""

    @pytest.mark.gui
    def test_tab_change_signal(self, gui_main_window, qtbot):
        """Test tab change signal emission."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        if tab_widget and tab_widget.count() > 1:
            # Connect to signal
            signal_received = []

            def on_tab_changed(index):
                signal_received.append(index)

            tab_widget.currentChanged.connect(on_tab_changed)

            # Ensure we start from a different tab, then switch to target
            target = 1 if tab_widget.currentIndex() != 1 else 0
            tab_widget.setCurrentIndex(target)
            qtbot.wait(100)

            # Verify signal was emitted
            assert len(signal_received) > 0
            assert signal_received[-1] == target


class TestKeyboardShortcuts:
    """Test suite for keyboard shortcuts and accessibility."""

    @pytest.mark.gui
    def test_escape_key_handling(self, gui_main_window, qtbot):
        """Test escape key handling."""
        window = gui_main_window

        # Send escape key
        qtbot.keyClick(window, QtCore.Qt.Key.Key_Escape)
        qtbot.wait(50)

        # Window should still be visible (escape shouldn't close it)
        assert window.isVisible()


# ============================================================================
# Integration Tests
# ============================================================================


class TestMainWindowIntegration:
    """Integration tests for main window with other components."""

    @pytest.mark.gui
    @pytest.mark.slow
    def test_window_with_mock_data(self, gui_main_window, qtbot, mock_xpcs_file):
        """Test main window with mock data integration."""
        window = gui_main_window

        # Mock file loading
        with patch.object(window.vk, "load_file") as mock_load:
            mock_load.return_value = True

            # Simulate file selection
            if hasattr(window, "load_file"):
                window.load_file(mock_xpcs_file.fname)
                qtbot.wait(200)

    @pytest.mark.gui
    def test_async_operations_gui_update(self, gui_main_window, qtbot):
        """Test that async operations update GUI properly."""
        window = gui_main_window

        # Mock async operation
        if hasattr(window, "async_vk"):
            # This would test actual async behavior if implemented
            pass

        # For now, just verify the window remains responsive
        assert window.isVisible()
        qtbot.wait(100)
        assert window.isVisible()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
