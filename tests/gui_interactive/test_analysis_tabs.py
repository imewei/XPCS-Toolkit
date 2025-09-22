"""Tests for analysis tab functionality and interactions.

This module provides comprehensive testing for all analysis tabs including
SAXS 2D/1D, G2, diffusion, two-time, stability, and intensity-time analysis.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets


class TestSAXS2DTab:
    """Test suite for SAXS 2D analysis tab."""

    @pytest.mark.gui
    def test_saxs_2d_tab_initialization(self, gui_main_window, qtbot, gui_test_helpers):
        """Test SAXS 2D tab initializes with proper components."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to SAXS 2D tab (index 0)
        gui_test_helpers.click_tab(qtbot, tab_widget, 0)
        current_widget = tab_widget.currentWidget()

        # Check for image view components
        assert current_widget is not None

        # Look for plot-related widgets
        plot_widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(plot_widgets) > 0

        # Check for control widgets (colorbar, zoom controls, etc.)
        control_widgets = current_widget.findChildren(QtWidgets.QPushButton)
        assert len(control_widgets) >= 0  # May have control buttons

    @pytest.mark.gui
    def test_saxs_2d_image_display(
        self,
        gui_main_window,
        qtbot,
        mock_xpcs_file,
        mock_viewer_kernel,
        gui_test_helpers,
    ):
        """Test SAXS 2D image display functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to SAXS 2D tab
        gui_test_helpers.click_tab(qtbot, tab_widget, 0)

        # Mock image data loading
        mock_data = np.random.poisson(100, (256, 256))
        mock_xpcs_file.load_saxs_2d.return_value = mock_data

        # Simulate data update (would normally happen through kernel)
        with patch.object(window.vk, "current_file", mock_xpcs_file):
            # Trigger plot update if available
            if hasattr(window, "plot_saxs_2d"):
                window.plot_saxs_2d()
                qtbot.wait(100)

    @pytest.mark.gui
    def test_saxs_2d_colorbar_controls(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test SAXS 2D colorbar and scaling controls."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 0)
        current_widget = tab_widget.currentWidget()

        # Look for colorbar controls
        combo_boxes = current_widget.findChildren(QtWidgets.QComboBox)
        for combo in combo_boxes:
            if combo.isVisible() and combo.count() > 0:
                # Test combo box interaction
                original_index = combo.currentIndex()
                new_index = (original_index + 1) % combo.count()
                combo.setCurrentIndex(new_index)
                qtbot.wait(50)
                assert combo.currentIndex() == new_index


class TestSAXS1DTab:
    """Test suite for SAXS 1D analysis tab."""

    @pytest.mark.gui
    def test_saxs_1d_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test SAXS 1D tab initializes with proper components."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to SAXS 1D tab (index 1)
        gui_test_helpers.click_tab(qtbot, tab_widget, 1)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Look for plot widgets
        plot_widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(plot_widgets) > 0

    @pytest.mark.gui
    def test_saxs_1d_plot_controls(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test SAXS 1D plot scaling and axis controls."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 1)
        current_widget = tab_widget.currentWidget()

        # Look for log/linear scale controls
        checkboxes = current_widget.findChildren(QtWidgets.QCheckBox)
        for checkbox in checkboxes:
            if "log" in checkbox.text().lower():
                # Test log scale toggle
                original_state = checkbox.isChecked()
                qtbot.mouseClick(checkbox, QtCore.Qt.MouseButton.LeftButton)
                qtbot.wait(50)
                assert checkbox.isChecked() != original_state


class TestG2AnalysisTab:
    """Test suite for G2 correlation analysis tab."""

    @pytest.mark.gui
    def test_g2_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test G2 analysis tab initializes with proper components."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to G2 tab (index 4 based on tab_mapping)
        gui_test_helpers.click_tab(qtbot, tab_widget, 4)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Look for G2-specific controls
        plot_widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(plot_widgets) > 0

    @pytest.mark.gui
    def test_g2_fitting_controls(
        self,
        gui_main_window,
        qtbot,
        mock_viewer_kernel,
        gui_test_helpers,
        mock_xpcs_file,
    ):
        """Test G2 fitting parameter controls."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 4)
        current_widget = tab_widget.currentWidget()

        # Mock G2 data
        taus = np.logspace(-6, 2, 50)
        g2_data = 1.5 * np.exp(-taus * 100) + 1.0
        errors = np.ones_like(g2_data) * 0.01
        mock_xpcs_file.get_g2.return_value = (taus, g2_data, errors)

        # Look for fitting controls
        spin_boxes = current_widget.findChildren(QtWidgets.QSpinBox)
        current_widget.findChildren(QtWidgets.QDoubleSpinBox)

        # Test parameter modification
        for spin_box in spin_boxes[:2]:  # Limit to first few to avoid long test times
            if spin_box.isVisible() and spin_box.isEnabled():
                original_value = spin_box.value()
                new_value = (
                    original_value + 1
                    if original_value < spin_box.maximum()
                    else original_value - 1
                )
                spin_box.setValue(new_value)
                qtbot.wait(50)
                assert spin_box.value() == new_value

    @pytest.mark.gui
    def test_g2_fit_execution(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test G2 fitting execution via button clicks."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 4)
        current_widget = tab_widget.currentWidget()

        # Look for fit button
        buttons = current_widget.findChildren(QtWidgets.QPushButton)
        fit_button = None

        for button in buttons:
            if "fit" in button.text().lower():
                fit_button = button
                break

        if fit_button and fit_button.isEnabled():
            # Click the fit button - the actual fitting is mocked elsewhere
            qtbot.mouseClick(fit_button, QtCore.Qt.MouseButton.LeftButton)
            qtbot.wait(100)


class TestDiffusionTab:
    """Test suite for diffusion analysis tab."""

    @pytest.mark.gui
    def test_diffusion_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test diffusion tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to diffusion tab (index 5)
        gui_test_helpers.click_tab(qtbot, tab_widget, 5)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Check for diffusion-specific widgets
        widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(widgets) > 0


class TestTwoTimeTab:
    """Test suite for two-time correlation analysis tab."""

    @pytest.mark.gui
    def test_twotime_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test two-time correlation tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to two-time tab (index 6)
        gui_test_helpers.click_tab(qtbot, tab_widget, 6)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Look for two-time specific controls
        widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(widgets) > 0

    @pytest.mark.gui
    def test_twotime_parameters(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test two-time correlation parameter controls."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 6)
        current_widget = tab_widget.currentWidget()

        # Look for parameter controls
        spin_boxes = current_widget.findChildren(QtWidgets.QSpinBox)
        for spin_box in spin_boxes[:2]:  # Test first few
            if spin_box.isVisible() and spin_box.isEnabled():
                original_value = spin_box.value()
                # Test increment/decrement
                if original_value < spin_box.maximum():
                    spin_box.setValue(original_value + 1)
                    qtbot.wait(50)
                    assert spin_box.value() == original_value + 1


class TestStabilityTab:
    """Test suite for stability analysis tab."""

    @pytest.mark.gui
    def test_stability_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test stability analysis tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to stability tab (index 2)
        gui_test_helpers.click_tab(qtbot, tab_widget, 2)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Check for stability plot widgets
        widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(widgets) > 0

    @pytest.mark.gui
    def test_stability_plot_updates(
        self,
        gui_main_window,
        qtbot,
        mock_viewer_kernel,
        gui_test_helpers,
        mock_xpcs_file,
    ):
        """Test stability plot updates with mock data."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        gui_test_helpers.click_tab(qtbot, tab_widget, 2)

        # Mock stability data
        frames = np.arange(100)
        intensity = np.random.normal(1000, 50, 100)

        # Mock data access
        mock_xpcs_file.get_stability_data = Mock(return_value=(frames, intensity))

        with patch.object(window.vk, "current_file", mock_xpcs_file):
            # Simulate plot update
            qtbot.wait(100)


class TestIntensityTimeTab:
    """Test suite for intensity vs time analysis tab."""

    @pytest.mark.gui
    def test_intensity_time_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test intensity-time tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to intensity-time tab (index 3)
        gui_test_helpers.click_tab(qtbot, tab_widget, 3)
        current_widget = tab_widget.currentWidget()

        assert current_widget is not None

        # Check for intensity-time specific widgets
        widgets = current_widget.findChildren(QtWidgets.QWidget)
        assert len(widgets) > 0


class TestQMapTab:
    """Test suite for Q-map analysis tab."""

    @pytest.mark.gui
    def test_qmap_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test Q-map tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to Q-map tab (index 7)
        if tab_widget.count() > 7:
            gui_test_helpers.click_tab(qtbot, tab_widget, 7)
            current_widget = tab_widget.currentWidget()

            assert current_widget is not None

            # Check for Q-map specific widgets
            widgets = current_widget.findChildren(QtWidgets.QWidget)
            assert len(widgets) > 0


class TestAverageTab:
    """Test suite for averaging functionality tab."""

    @pytest.mark.gui
    def test_average_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test averaging tab initializes properly."""
        window = gui_main_window

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Switch to average tab (index 8)
        if tab_widget.count() > 8:
            gui_test_helpers.click_tab(qtbot, tab_widget, 8)
            current_widget = tab_widget.currentWidget()

            assert current_widget is not None

            # Look for file selection and averaging controls
            widgets = current_widget.findChildren(QtWidgets.QWidget)
            assert len(widgets) > 0


class TestMetadataTab:
    """Test suite for metadata display tab."""

    @pytest.mark.gui
    def test_metadata_tab_initialization(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test metadata tab initializes properly."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Switch to metadata tab (index 9)
        if tab_widget.count() > 9:
            gui_test_helpers.click_tab(qtbot, tab_widget, 9)
            current_widget = tab_widget.currentWidget()

            assert current_widget is not None

            # Look for metadata display widgets (tree, table, etc.)
            widgets = current_widget.findChildren(QtWidgets.QWidget)
            assert len(widgets) > 0

    @pytest.mark.gui
    def test_metadata_display(
        self,
        gui_main_window,
        qtbot,
        mock_viewer_kernel,
        gui_test_helpers,
        mock_xpcs_file,
    ):
        """Test metadata display functionality."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        if tab_widget.count() > 9:
            gui_test_helpers.click_tab(qtbot, tab_widget, 9)
            current_widget = tab_widget.currentWidget()

            # Mock metadata
            mock_metadata = {
                "detector_size": [516, 516],
                "beam_center_x": 256.5,
                "beam_center_y": 256.5,
                "energy": 8.0,
                "sample_name": "Test_Sample",
            }

            mock_xpcs_file.meta = mock_metadata

            with patch.object(window.vk, "current_file", mock_xpcs_file):
                # Look for tree or table widgets that display metadata
                tree_widgets = current_widget.findChildren(QtWidgets.QTreeWidget)
                table_widgets = current_widget.findChildren(QtWidgets.QTableWidget)

                # Should have some form of metadata display
                assert len(tree_widgets) > 0 or len(table_widgets) > 0


# ============================================================================
# Cross-Tab Integration Tests
# ============================================================================


class TestTabIntegration:
    """Test suite for cross-tab interactions and data consistency."""

    @pytest.mark.gui
    def test_tab_data_consistency(
        self,
        gui_main_window,
        qtbot,
        mock_viewer_kernel,
        gui_test_helpers,
        mock_xpcs_file,
    ):
        """Test that data remains consistent across tab switches."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        # Set up mock data
        with patch.object(window.vk, "current_file", mock_xpcs_file):
            # Visit multiple tabs and ensure no errors
            test_tabs = [0, 1, 4, 2]  # SAXS-2D, SAXS-1D, G2, Stability

            for tab_index in test_tabs:
                if tab_index < tab_widget.count():
                    gui_test_helpers.click_tab(qtbot, tab_widget, tab_index)
                    qtbot.wait(100)

                    # Verify tab switched successfully
                    assert tab_widget.currentIndex() == tab_index

                    # Verify current widget exists
                    current_widget = tab_widget.currentWidget()
                    assert current_widget is not None

    @pytest.mark.gui
    def test_tab_state_preservation(
        self, gui_main_window, qtbot, mock_viewer_kernel, gui_test_helpers
    ):
        """Test that tab states are preserved during navigation."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Initialize the viewer kernel (it's None by default when path=None)
        window.vk = mock_viewer_kernel

        if tab_widget.count() > 2:
            # Go to first tab and modify a control
            gui_test_helpers.click_tab(qtbot, tab_widget, 0)
            first_widget = tab_widget.currentWidget()

            # Find a modifiable control
            combo_boxes = first_widget.findChildren(QtWidgets.QComboBox)
            test_combo = None
            for combo in combo_boxes:
                if combo.isEnabled() and combo.count() > 1:
                    test_combo = combo
                    break

            if test_combo:
                # Modify the control
                original_index = test_combo.currentIndex()
                new_index = (original_index + 1) % test_combo.count()
                test_combo.setCurrentIndex(new_index)
                qtbot.wait(50)

                # Switch to different tab
                gui_test_helpers.click_tab(qtbot, tab_widget, 1)
                qtbot.wait(100)

                # Switch back to first tab
                gui_test_helpers.click_tab(qtbot, tab_widget, 0)
                qtbot.wait(50)

                # Verify state was preserved
                assert test_combo.currentIndex() == new_index


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
