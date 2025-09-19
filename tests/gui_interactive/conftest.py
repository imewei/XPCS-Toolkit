"""Pytest configuration for GUI interactive tests.

This module provides fixtures and configuration specifically for GUI testing
using pytest-qt with PySide6 components.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    # Mock h5py for basic testing
    class MockH5pyFile:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def create_dataset(self, *args, **kwargs): pass
        def create_group(self, *args, **kwargs): return self
        def __getitem__(self, key): return self
        def __setitem__(self, key, value): pass

    class MockH5py:
        File = MockH5pyFile

    h5py = MockH5py()
from PySide6 import QtCore, QtGui, QtWidgets

# Import pytest-qt for GUI testing
pytest_qt = pytest.importorskip("pytestqt", reason="PySide6/Qt tests require pytest-qt")

from xpcs_toolkit.viewer_kernel import ViewerKernel
from xpcs_toolkit.xpcs_file import XpcsFile

# Local imports
from xpcs_toolkit.xpcs_viewer import XpcsViewer

# ============================================================================
# GUI Test Configuration
# ============================================================================


def pytest_configure(config):
    """Configure GUI-specific test settings."""
    # Ensure offscreen platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Suppress Qt warnings during GUI tests
    os.environ.setdefault("PYXPCS_SUPPRESS_QT_WARNINGS", "1")

    # Configure for GUI testing
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_Use96Dpi, True
    )


def pytest_runtest_setup(item):
    """Setup for each GUI test."""
    if item.get_closest_marker("gui"):
        # Ensure we have a display for GUI tests
        if (
            not os.environ.get("DISPLAY")
            and os.environ.get("QT_QPA_PLATFORM") != "offscreen"
        ):
            pytest.skip("GUI tests require display or offscreen platform")


# ============================================================================
# Qt Application Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def qapp():
    """Create QApplication instance for GUI testing session."""
    # Check if QApplication already exists
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
        app.setQuitOnLastWindowClosed(False)

    yield app

    # Clean up - don't quit the app as it may be shared


@pytest.fixture
def qtbot_wait_time():
    """Configure wait time for qtbot operations."""
    return 100  # milliseconds


# ============================================================================
# Mock Data Fixtures for GUI Testing
# ============================================================================


@pytest.fixture
def mock_xpcs_data():
    """Create synthetic XPCS data for GUI testing."""
    # Create minimal but realistic XPCS dataset structure
    data = {
        "entry/metadata/detector_size": np.array([516, 516]),
        "entry/metadata/beam_center_x": 256.5,
        "entry/metadata/beam_center_y": 256.5,
        "entry/metadata/detector_distance": 1.5,
        "entry/metadata/energy": 8.0,
        "entry/metadata/sample_name": "GUI_Test_Sample",
        # Analysis results
        "entry/analysis/g2/taus": np.logspace(-6, 2, 50),
        "entry/analysis/g2/g2": 1.5 * np.exp(-np.logspace(-6, 2, 50) * 100) + 1.0,
        "entry/analysis/g2/g2_err": np.ones(50) * 0.01,
        # SAXS data
        "entry/analysis/saxs_2d/data": np.random.poisson(100, (516, 516)),
        "entry/analysis/saxs_1d/q": np.linspace(0.001, 0.1, 100),
        "entry/analysis/saxs_1d/I": np.exp(-np.linspace(0.001, 0.1, 100) * 20) * 1000
        + 10,
        "entry/analysis/saxs_1d/I_err": np.sqrt(
            np.exp(-np.linspace(0.001, 0.1, 100) * 20) * 1000 + 10
        ),
        # Stability data
        "entry/analysis/stability/frame_num": np.arange(1000),
        "entry/analysis/stability/intensity": np.random.normal(1000, 50, 1000),
        # Two-time correlation
        "entry/analysis/twotime/C2": np.random.normal(1.0, 0.1, (50, 50)),
        "entry/analysis/twotime/taus": np.logspace(-6, 2, 50),
    }
    return data


@pytest.fixture
def mock_hdf5_file(tmp_path, mock_xpcs_data):
    """Create a temporary HDF5 file with mock XPCS data."""
    hdf_path = tmp_path / "test_data.hdf5"

    with h5py.File(hdf_path, "w") as f:
        for key, value in mock_xpcs_data.items():
            f.create_dataset(key, data=value)

    return str(hdf_path)


@pytest.fixture
def mock_xpcs_file(mock_hdf5_file):
    """Create a mock XpcsFile instance."""
    with patch("xpcs_toolkit.xpcs_file.XpcsFile.__init__", return_value=None):
        mock_file = XpcsFile.__new__(XpcsFile)
        mock_file.fname = mock_hdf5_file
        mock_file.ftype = "nexus"
        mock_file.fpath = Path(mock_hdf5_file).parent
        mock_file.meta = {
            "detector_size": np.array([516, 516]),
            "beam_center_x": 256.5,
            "beam_center_y": 256.5,
            "detector_distance": 1.5,
            "energy": 8.0,
            "sample_name": "GUI_Test_Sample",
        }

        # Mock common methods
        mock_file.load_saxs_2d = Mock(return_value=np.random.poisson(100, (516, 516)))
        mock_file.load_qmap = Mock(return_value={"dqmap": np.full((516, 516), 0.05)})
        mock_file.get_g2 = Mock(
            return_value=(
                np.logspace(-6, 2, 50),
                1.5 * np.exp(-np.logspace(-6, 2, 50) * 100) + 1.0,
                np.ones(50) * 0.01,
            )
        )

        return mock_file


@pytest.fixture
def mock_viewer_kernel(mock_xpcs_file):
    """Create a mock ViewerKernel instance."""
    with patch("xpcs_toolkit.viewer_kernel.ViewerKernel.__init__", return_value=None):
        mock_kernel = ViewerKernel.__new__(ViewerKernel)
        mock_kernel.flist = [mock_xpcs_file]
        mock_kernel.current_file = mock_xpcs_file
        mock_kernel.path = str(Path(mock_xpcs_file.fname).parent)
        mock_kernel.target = [mock_xpcs_file]  # Add target attribute for GUI tests
        mock_kernel.source = []  # Add source attribute for GUI tests

        # Mock common methods
        mock_kernel.update_file_list = Mock()
        mock_kernel.load_file = Mock()
        mock_kernel.set_current_file = Mock()
        mock_kernel.build = Mock()  # Add build method to prevent FileNotFoundError

        return mock_kernel


# ============================================================================
# GUI Widget Fixtures
# ============================================================================


@pytest.fixture
def gui_main_window(qapp, qtbot, mock_viewer_kernel):
    """Create main XPCS Viewer window for testing."""
    # Mock the initialization to avoid actual file system operations
    with patch("xpcs_toolkit.xpcs_viewer.ViewerKernel") as mock_vk_class:
        mock_vk_class.return_value = mock_viewer_kernel

        # Create the main window
        window = XpcsViewer(path=None)
        qtbot.addWidget(window)

        # Show window for interaction testing
        window.show()
        qtbot.wait(100)  # Wait for window to be shown

        yield window

        # Clean up
        window.close()


@pytest.fixture
def gui_plot_widget():
    """Create a plot widget for testing."""
    import pyqtgraph as pg

    widget = pg.PlotWidget()
    return widget


@pytest.fixture
def gui_parameter_tree():
    """Create a parameter tree widget for testing."""
    from pyqtgraph.parametertree import Parameter, ParameterTree

    # Create test parameters
    params = [
        {
            "name": "Test Group",
            "type": "group",
            "children": [
                {"name": "Value", "type": "float", "value": 1.0},
                {"name": "Enable", "type": "bool", "value": True},
                {
                    "name": "Choice",
                    "type": "list",
                    "values": ["Option 1", "Option 2"],
                    "value": "Option 1",
                },
            ],
        },
    ]

    parameter = Parameter.create(name="Test Parameters", type="group", children=params)
    tree = ParameterTree()
    tree.setParameters(parameter, showTop=False)

    return tree, parameter


# ============================================================================
# Test Helpers and Utilities
# ============================================================================


@pytest.fixture
def gui_test_helpers():
    """Provide helper functions for GUI testing."""

    class GuiTestHelpers:
        @staticmethod
        def click_tab(qtbot, tab_widget, tab_index):
            """Click on a specific tab."""
            tab_widget.setCurrentIndex(tab_index)
            qtbot.wait(50)

        @staticmethod
        def set_parameter(qtbot, parameter, name, value):
            """Set parameter value in parameter tree."""
            param = parameter.child(name)
            param.setValue(value)
            qtbot.wait(50)

        @staticmethod
        def simulate_file_drop(qtbot, widget, file_path):
            """Simulate file drop on widget."""
            # Create mime data
            mime_data = QtCore.QMimeData()
            urls = [QtCore.QUrl.fromLocalFile(file_path)]
            mime_data.setUrls(urls)

            # Create drag enter event
            drag_enter = QtGui.QDragEnterEvent(
                widget.rect().center(),
                QtCore.Qt.DropAction.CopyAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

            # Create drop event
            drop = QtGui.QDropEvent(
                widget.rect().center(),
                QtCore.Qt.DropAction.CopyAction,
                mime_data,
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.KeyboardModifier.NoModifier,
            )

            # Send events
            QtWidgets.QApplication.sendEvent(widget, drag_enter)
            QtWidgets.QApplication.sendEvent(widget, drop)
            qtbot.wait(100)

        @staticmethod
        def wait_for_signal(qtbot, signal, timeout=5000):
            """Wait for a Qt signal to be emitted."""
            with qtbot.waitSignal(signal, timeout=timeout):
                pass

    return GuiTestHelpers()


# ============================================================================
# Error Simulation Fixtures
# ============================================================================


@pytest.fixture
def gui_error_simulator():
    """Provide error simulation utilities for testing error handling."""

    class ErrorSimulator:
        @staticmethod
        def simulate_file_load_error(mock_kernel):
            """Simulate file loading error."""
            mock_kernel.load_file.side_effect = Exception("Mock file load error")

        @staticmethod
        def simulate_analysis_error(mock_file):
            """Simulate analysis computation error."""
            mock_file.get_g2.side_effect = Exception("Mock analysis error")

        @staticmethod
        def simulate_memory_error(mock_file):
            """Simulate memory error."""
            mock_file.load_saxs_2d.side_effect = MemoryError("Mock memory error")

        @staticmethod
        def simulate_plot_error(plot_widget):
            """Simulate plotting error."""
            original_plot = plot_widget.plot

            def error_plot(*args, **kwargs):
                raise Exception("Mock plot error")

            plot_widget.plot = error_plot
            return original_plot

    return ErrorSimulator()


# ============================================================================
# Performance Testing Fixtures
# ============================================================================


@pytest.fixture
def gui_performance_monitor():
    """Monitor GUI performance during tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = []

        def start_timing(self):
            """Start timing measurement."""
            import time

            self.start_time = time.time()

        def end_timing(self, operation_name):
            """End timing measurement."""
            if self.start_time is not None:
                import time

                elapsed = time.time() - self.start_time
                self.measurements.append((operation_name, elapsed))
                return elapsed
            return None

        def get_measurements(self):
            """Get all timing measurements."""
            return self.measurements.copy()

    return PerformanceMonitor()


# ============================================================================
# Advanced GUI Testing Fixtures
# ============================================================================


@pytest.fixture
def gui_widget_inspector():
    """Inspector for analyzing widget properties and states."""

    class WidgetInspector:
        @staticmethod
        def get_all_widgets(parent):
            """Recursively get all widgets from parent."""
            widgets = []
            for child in parent.findChildren(QtWidgets.QWidget):
                widgets.append(child)
            return widgets

        @staticmethod
        def find_widget_by_text(parent, text, widget_type=None):
            """Find widget containing specific text."""
            for child in parent.findChildren(QtWidgets.QWidget):
                if (hasattr(child, "text") and text in child.text()) or (
                    hasattr(child, "windowTitle") and text in child.windowTitle()
                ):
                    if widget_type is None or isinstance(child, widget_type):
                        return child
            return None

        @staticmethod
        def get_widget_properties(widget):
            """Get comprehensive widget properties."""
            props = {
                "class": widget.__class__.__name__,
                "object_name": widget.objectName(),
                "visible": widget.isVisible(),
                "enabled": widget.isEnabled(),
                "size": (widget.width(), widget.height()),
                "position": (widget.x(), widget.y()),
            }

            if hasattr(widget, "text"):
                props["text"] = widget.text()
            if hasattr(widget, "value"):
                props["value"] = widget.value()
            if hasattr(widget, "isChecked"):
                props["checked"] = widget.isChecked()

            return props

    return WidgetInspector()


@pytest.fixture
def gui_interaction_recorder():
    """Record and replay GUI interactions."""

    class InteractionRecorder:
        def __init__(self):
            self.interactions = []

        def record_click(self, widget, button=QtCore.Qt.MouseButton.LeftButton):
            """Record a mouse click."""
            self.interactions.append(
                {
                    "type": "click",
                    "widget": widget.objectName() or str(widget),
                    "button": button,
                    "position": widget.rect().center(),
                }
            )

        def record_key_press(
            self, widget, key, modifier=QtCore.Qt.KeyboardModifier.NoModifier
        ):
            """Record a key press."""
            self.interactions.append(
                {
                    "type": "keypress",
                    "widget": widget.objectName() or str(widget),
                    "key": key,
                    "modifier": modifier,
                }
            )

        def record_text_input(self, widget, text):
            """Record text input."""
            self.interactions.append(
                {
                    "type": "text",
                    "widget": widget.objectName() or str(widget),
                    "text": text,
                }
            )

        def replay_interactions(self, qtbot, parent_widget):
            """Replay recorded interactions."""
            for interaction in self.interactions:
                if interaction["type"] == "click":
                    # Find widget by name or recreate
                    widget = parent_widget.findChild(
                        QtWidgets.QWidget, interaction["widget"]
                    )
                    if widget:
                        qtbot.mouseClick(widget, interaction["button"])
                elif interaction["type"] == "keypress":
                    widget = parent_widget.findChild(
                        QtWidgets.QWidget, interaction["widget"]
                    )
                    if widget:
                        qtbot.keyClick(
                            widget, interaction["key"], interaction["modifier"]
                        )
                elif interaction["type"] == "text":
                    widget = parent_widget.findChild(
                        QtWidgets.QWidget, interaction["widget"]
                    )
                    if widget and hasattr(widget, "setText"):
                        widget.setText(interaction["text"])
                qtbot.wait(50)

        def get_interaction_log(self):
            """Get recorded interaction log."""
            return self.interactions.copy()

    return InteractionRecorder()


@pytest.fixture
def gui_state_validator():
    """Validate GUI state consistency."""

    class StateValidator:
        @staticmethod
        def validate_tab_consistency(tab_widget):
            """Validate tab widget consistency."""
            issues = []

            for i in range(tab_widget.count()):
                tab_widget.setCurrentIndex(i)
                widget = tab_widget.currentWidget()

                if widget is None:
                    issues.append(f"Tab {i} has no widget")
                elif not widget.isEnabled():
                    issues.append(f"Tab {i} widget is disabled")

            return issues

        @staticmethod
        def validate_plot_widgets(parent):
            """Validate plot widget states."""
            issues = []
            plot_widgets = parent.findChildren(QtWidgets.QWidget)

            for widget in plot_widgets:
                if "plot" in widget.__class__.__name__.lower():
                    if not widget.isVisible():
                        issues.append(
                            f"Plot widget {widget.__class__.__name__} is not visible"
                        )
                    if widget.size().width() < 10 or widget.size().height() < 10:
                        issues.append(
                            f"Plot widget {widget.__class__.__name__} has invalid size"
                        )

            return issues

        @staticmethod
        def validate_control_states(parent, expected_controls):
            """Validate that expected control widgets exist and are properly configured."""
            issues = []
            found_controls = {}

            for control_type, control_names in expected_controls.items():
                found_controls[control_type] = []
                widgets = parent.findChildren(eval(f"QtWidgets.{control_type}"))

                for widget in widgets:
                    name = widget.objectName()
                    if name in control_names:
                        found_controls[control_type].append(name)

                        # Check if control is properly initialized
                        if not widget.isEnabled() and control_names[name].get(
                            "should_be_enabled", True
                        ):
                            issues.append(f"{control_type} {name} should be enabled")

                # Check for missing controls
                for name in control_names:
                    if name not in found_controls[control_type]:
                        issues.append(f"Missing {control_type}: {name}")

            return issues, found_controls

    return StateValidator()


@pytest.fixture
def gui_accessibility_helper():
    """Helper for testing accessibility features."""

    class AccessibilityHelper:
        @staticmethod
        def check_keyboard_navigation(qtbot, parent_widget):
            """Test keyboard navigation through focusable widgets."""
            focusable_widgets = []

            # Find all focusable widgets
            for widget in parent_widget.findChildren(QtWidgets.QWidget):
                if widget.focusPolicy() != QtCore.Qt.FocusPolicy.NoFocus:
                    focusable_widgets.append(widget)

            results = []
            for i, widget in enumerate(focusable_widgets):
                # Set focus
                widget.setFocus()
                qtbot.wait(50)

                # Check if focus was set
                has_focus = widget.hasFocus()
                results.append(
                    {
                        "widget": widget.__class__.__name__,
                        "object_name": widget.objectName(),
                        "focus_successful": has_focus,
                        "tab_order": i,
                    }
                )

                # Test Tab key navigation
                qtbot.keyClick(widget, QtCore.Qt.Key.Key_Tab)
                qtbot.wait(50)

            return results

        @staticmethod
        def check_tooltips(parent_widget):
            """Check that important widgets have tooltips."""
            widgets_without_tooltips = []

            important_widget_types = [
                QtWidgets.QPushButton,
                QtWidgets.QSpinBox,
                QtWidgets.QDoubleSpinBox,
                QtWidgets.QComboBox,
            ]

            for widget_type in important_widget_types:
                widgets = parent_widget.findChildren(widget_type)
                for widget in widgets:
                    if not widget.toolTip():
                        widgets_without_tooltips.append(
                            {
                                "type": widget.__class__.__name__,
                                "name": widget.objectName() or "unnamed",
                                "text": getattr(widget, "text", lambda: "")(),
                            }
                        )

            return widgets_without_tooltips

        @staticmethod
        def check_minimum_sizes(parent_widget):
            """Check that widgets have appropriate minimum sizes."""
            size_issues = []

            for widget in parent_widget.findChildren(QtWidgets.QWidget):
                min_size = widget.minimumSize()
                current_size = widget.size()

                if (
                    current_size.width() < min_size.width()
                    or current_size.height() < min_size.height()
                ):
                    size_issues.append(
                        {
                            "widget": widget.__class__.__name__,
                            "name": widget.objectName(),
                            "current_size": (
                                current_size.width(),
                                current_size.height(),
                            ),
                            "minimum_size": (min_size.width(), min_size.height()),
                        }
                    )

            return size_issues

    return AccessibilityHelper()
