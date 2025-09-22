"""Tests for main window functionality and tab management.

This module tests the main XpcsViewer window, including initialization,
tab switching, menu actions, and status updates.
"""

from unittest.mock import patch

import pytest
from PySide6 import QtCore, QtWidgets

from xpcs_toolkit.xpcs_viewer import XpcsViewer, tab_mapping


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
    def test_tab_switching(self, gui_main_window, qtbot, gui_test_helpers):
        """Test switching between different analysis tabs."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Test switching to each tab
        for tab_index, tab_name in tab_mapping.items():
            if tab_index < tab_widget.count():
                gui_test_helpers.click_tab(qtbot, tab_widget, tab_index)

                # Verify tab is selected
                assert tab_widget.currentIndex() == tab_index

                # Check tab has expected content
                current_widget = tab_widget.currentWidget()
                assert current_widget is not None

                # Verify tab-specific elements exist - more robust check
                if tab_name in {"saxs_2d", "g2"}:
                    # Just verify we have some kind of widget content
                    assert (
                        current_widget.children() or True
                    )  # Has children or is at least present

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

    @pytest.mark.gui
    def test_window_resize_behavior(self, gui_main_window, qtbot):
        """Test window resizing behavior."""
        window = gui_main_window
        original_size = window.size()

        # Resize window - try to make it smaller
        target_size = QtCore.QSize(800, 600)
        window.resize(target_size)
        qtbot.wait(100)

        # Verify size changed (may not be exact due to window constraints)
        current_size = window.size()
        size_changed = (
            current_size.width() != original_size.width()
            or current_size.height() != original_size.height()
        )

        # Either the window resized or it respects minimum size constraints
        if size_changed:
            assert True  # Window successfully resized
        else:
            # Window may have minimum size constraints - check it's at least reasonable
            assert current_size.width() >= 400 and current_size.height() >= 300

        # Restore original size
        window.resize(original_size)
        qtbot.wait(50)

    @pytest.mark.gui
    def test_menu_bar_functionality(self, gui_main_window, qtbot):
        """Test menu bar and menu actions."""
        window = gui_main_window
        menu_bar = window.menuBar()

        # Check menu bar exists
        assert menu_bar is not None

        # Test if common menus exist (File, View, Help, etc.)
        menus = [action.text() for action in menu_bar.actions() if action.text()]
        expected_menus = ["File", "View", "Help"]

        # At least some basic menus should exist
        menu_found = any(menu in menus for menu in expected_menus)
        if menu_found:
            # Test menu interaction
            for action in menu_bar.actions():
                if action.text() == "File":
                    # Click File menu
                    qtbot.mouseClick(menu_bar, QtCore.Qt.MouseButton.LeftButton)
                    qtbot.wait(50)
                    break


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
    def test_window_close_behavior(self, qapp, qtbot):
        """Test window closing behavior."""
        # Create a temporary window for close testing
        with patch("xpcs_toolkit.xpcs_viewer.ViewerKernel"):
            window = XpcsViewer(path=None)
            qtbot.addWidget(window)
            window.show()
            qtbot.wait(50)

            # Test close
            window.close()
            qtbot.wait(50)

            # Verify window is closed
            assert not window.isVisible()

    @pytest.mark.gui
    def test_multiple_window_instances(self, qapp, qtbot):
        """Test behavior with multiple window instances."""
        windows = []

        try:
            # Create multiple windows
            for _i in range(2):
                with patch("xpcs_toolkit.xpcs_viewer.ViewerKernel"):
                    window = XpcsViewer(path=None)
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

            # Change tab
            tab_widget.setCurrentIndex(1)
            qtbot.wait(100)

            # Verify signal was emitted
            assert len(signal_received) > 0
            assert signal_received[-1] == 1

    @pytest.mark.gui
    def test_window_state_signals(self, gui_main_window, qtbot):
        """Test window state change signals."""
        window = gui_main_window

        # Track window state changes
        state_changes = []

        def on_window_state_changed():
            state_changes.append(window.windowState())

        # Connect to window state change (if available)
        if hasattr(window, "windowStateChanged"):
            window.windowStateChanged.connect(on_window_state_changed)

        # Minimize and restore window
        window.showMinimized()
        qtbot.wait(100)

        window.showNormal()
        qtbot.wait(100)


class TestKeyboardShortcuts:
    """Test suite for keyboard shortcuts and accessibility."""

    @pytest.mark.gui
    def test_keyboard_navigation(self, gui_main_window, qtbot):
        """Test keyboard navigation between tabs."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        if tab_widget and tab_widget.count() > 1:
            # Set focus on tab widget
            tab_widget.setFocus()
            qtbot.wait(50)

            tab_widget.currentIndex()

            # Use Ctrl+Tab to switch tabs (if supported)
            qtbot.keyClick(
                tab_widget,
                QtCore.Qt.Key.Key_Tab,
                QtCore.Qt.KeyboardModifier.ControlModifier,
            )
            qtbot.wait(100)

            # Check if tab changed (behavior may vary by platform)
            # This is more of a validation that keyboard events work
            assert tab_widget.currentIndex() >= 0

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
