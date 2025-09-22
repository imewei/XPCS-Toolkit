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
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_dataset(self, *args, **kwargs):
            pass

        def create_group(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def __setitem__(self, key, value):
            pass

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
        mock_file.atype = ["Multitau"]  # Add analysis type attribute
        mock_file._saxs_data_loaded = False  # Add missing internal attribute
        mock_file._use_lazy_loading = False  # Add missing lazy loading attribute
        mock_file.label = "Mock_Test_File"  # Add missing label attribute
        mock_file.meta = {
            "detector_size": np.array([516, 516]),
            "beam_center_x": 256.5,
            "beam_center_y": 256.5,
            "detector_distance": 1.5,
            "energy": 8.0,
            "sample_name": "GUI_Test_Sample",
        }

        # Mock qmap object with required attributes
        mock_qmap = Mock()
        mock_qmap.bcx = 256.5
        mock_qmap.bcy = 256.5
        mock_file.qmap = mock_qmap

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

        # Add SAXS data properties to avoid lazy loading issues
        mock_file.saxs_2d = np.random.poisson(100, (516, 516))
        mock_file.saxs_2d_log = np.log10(mock_file.saxs_2d + 1)

        # Add saxs_1d data structure that the code expects
        q_vals = np.logspace(-3, 0, 50)
        i_vals = np.random.poisson(1000, 50).astype(float)
        mock_file.saxs_1d = {
            "labels": ["q (nm⁻¹)", "I(q) (counts)"],
            "data": np.column_stack([q_vals, i_vals]),  # Proper 2D array for plotting
        }

        # Ensure consistent array data for SAXS methods
        q_values = np.logspace(-3, 0, 50)
        i_values = np.random.poisson(1000, 50).astype(float)

        # Add missing methods that tests expect (ensuring consistent array sizes)
        mock_file.load_saxs_1d = Mock(
            return_value=(
                q_values,  # q values - consistent size
                i_values,  # I(q) values - consistent size
            )
        )

        mock_file.get_saxs1d_data = Mock(
            return_value=(
                q_values,  # q values - consistent size
                i_values,  # I(q) values - consistent size
                "q (nm⁻¹)",  # xlabel
                "I(q) (counts)",  # ylabel
            )
        )

        # Add fitting methods that GUI tests expect
        mock_file.fit_g2 = Mock(
            return_value={
                "success": True,
                "params": [1.5, 100.0, 1.0],
                "errors": [0.1, 10.0, 0.05],
                "r_squared": 0.95,
            }
        )

        # Add comprehensive missing attributes for GUI workflows
        mock_file.saxs_2d_log_data = Mock()
        mock_file.fit_summary = {"q_val": [1, 2, 3], "fit_val": [0.1, 0.2, 0.3]}
        mock_file.Int_t = np.random.random(
            50
        )  # Make it subscriptable array instead of Mock
        mock_file.hdf_info = {"sample_name": "GUI_Test", "detector": "Lambda"}
        mock_file.tau = np.logspace(-6, 2, 50)
        mock_file.g2_t0 = 1.0
        mock_file.t_el = mock_file.tau * mock_file.g2_t0
        mock_file._cache_prefix = "test_cache"  # Cache prefix for diffusion plots
        mock_file.c2_processed_bins = np.random.random((100, 100))  # Two-time data

        # Add G2 correlation data attributes for multitau analysis
        mock_file.g2 = np.random.random((50, 10)) + 1.0  # G2 values > 1
        mock_file.g2_err = np.random.random((50, 10)) * 0.1

        # Add two-time correlation attributes as fallback
        mock_file.c2_g2 = np.random.random((100, 100))

        # Add qmap attributes for Q-space mapping - always override mock methods
        mock_file.qmap.get_qbin_in_qrange = Mock(
            return_value=(
                list(range(5)),  # qindex_selected
                np.linspace(0.001, 0.1, 5),  # qvalues
            )
        )
        mock_file.qmap.get_qbin_label = Mock(side_effect=lambda i: f"q_{i}")
        mock_file.qmap.dqmap = np.random.random((516, 516))
        mock_file.dqmap = mock_file.qmap.dqmap

        return mock_file


@pytest.fixture
def temp_hdf5_files(tmp_path):
    """Create temporary HDF5 files for testing."""
    files = []

    for i in range(3):
        file_path = tmp_path / f"test_file_{i}.hdf5"

        # Create minimal HDF5 file structure for testing
        try:
            import h5py

            with h5py.File(file_path, "w") as f:
                f.create_dataset(
                    "entry/data/intensities", data=np.random.poisson(100, (100, 100))
                )
                f.create_dataset("entry/metadata/detector_size", data=[100, 100])
                f.attrs["sample_name"] = f"Test_Sample_{i}"
        except ImportError:
            # Create dummy file if h5py not available
            file_path.write_text(f"dummy hdf5 file {i}")

        files.append(str(file_path))

    return files


@pytest.fixture
def mock_viewer_kernel(mock_xpcs_file):
    """Create a mock ViewerKernel instance."""
    with patch("xpcs_toolkit.viewer_kernel.ViewerKernel.__init__", return_value=None):
        mock_kernel = ViewerKernel.__new__(ViewerKernel)
        mock_kernel.flist = [mock_xpcs_file]
        mock_kernel.current_file = mock_xpcs_file  # This will allow tests to patch it
        mock_kernel.path = str(Path(mock_xpcs_file.fname).parent)
        # Import ListDataModel for proper Qt model compatibility
        from xpcs_toolkit.helper.listmodel import ListDataModel

        # Create proper Qt models instead of Python lists
        mock_kernel.target = ListDataModel([mock_xpcs_file])
        mock_kernel.source = ListDataModel([])
        mock_kernel.source_search = ListDataModel([])  # Add missing attribute
        mock_kernel.cache = {}  # Add cache attribute for FileLocator compatibility
        mock_kernel.timestamp = "test_timestamp_123"  # Add missing timestamp attribute
        mock_kernel._current_dset_cache = {}  # Add missing current dataset cache
        # Create a real Qt model for avg_worker
        from PySide6.QtCore import QAbstractTableModel, QModelIndex

        class MockAvgWorkerModel(QAbstractTableModel):
            def rowCount(self, parent=QModelIndex()):
                return 0

            def columnCount(self, parent=QModelIndex()):
                return 0

            def data(self, index, role):
                return None

        mock_kernel.avg_worker = MockAvgWorkerModel()
        mock_kernel.meta = {"saxs1d_bkg_xf": None}  # Add missing meta attribute

        # Mock common methods
        mock_kernel.update_file_list = Mock()
        mock_kernel.load_file = Mock()
        mock_kernel.load_files = Mock()  # Add missing method for multiple file loading
        mock_kernel.set_current_file = Mock()
        mock_kernel.build = Mock()  # Add build method to prevent FileNotFoundError
        mock_kernel.get_xf_list = Mock(
            return_value=[mock_xpcs_file]
        )  # Return non-empty list
        mock_kernel._get_cached_dataset = Mock(
            return_value=mock_xpcs_file
        )  # Add missing cache method

        # Mock all plotting operations to prevent timeouts during tab switching
        # Return proper data structure: (qd, tel) where tel is list of time arrays
        mock_kernel.plot_g2 = Mock(
            return_value=(
                np.array([1, 2, 3]),  # qd - q values
                [
                    np.logspace(-6, 2, 50),
                    np.logspace(-6, 2, 50),
                ],  # tel - list of time arrays
            )
        )
        mock_kernel.plot_saxs2d = Mock()
        mock_kernel.plot_saxs1d = Mock()
        mock_kernel.plot_qmap = Mock()
        mock_kernel.plot_tauq = Mock()
        mock_kernel.plot_tauq_pre = Mock()
        mock_kernel.plot_twotime = Mock(return_value=["label1", "label2"])
        mock_kernel.plot_intt = Mock()
        mock_kernel.plot_stability = Mock()
        mock_kernel.plot_diffusion = Mock()

        return mock_kernel


# ============================================================================
# GUI Widget Fixtures
# ============================================================================


@pytest.fixture
def gui_main_window(qapp, qtbot, mock_viewer_kernel):
    """Create main XPCS Viewer window for testing."""
    # Fix PyQtGraph compatibility issues with PySide6
    original_buildTree = None
    original_getViewBox = None
    try:
        # Fix DataTreeWidget issue
        import pyqtgraph.widgets.DataTreeWidget

        original_buildTree = pyqtgraph.widgets.DataTreeWidget.DataTreeWidget.buildTree

        def fixed_buildTree(self, data, parent, name="", hideRoot=False, path=()):
            """Fixed buildTree method that handles PySide6 compatibility."""
            if data is None:
                # Skip building tree for None data to avoid AttributeError
                return None
            return original_buildTree(self, data, parent, name, hideRoot, path)

        pyqtgraph.widgets.DataTreeWidget.DataTreeWidget.buildTree = fixed_buildTree

        # Fix GraphicsItem getViewBox issue
        import pyqtgraph.graphicsItems.GraphicsItem

        original_getViewBox = (
            pyqtgraph.graphicsItems.GraphicsItem.GraphicsItem.getViewBox
        )

        def fixed_getViewBox(self):
            """Fixed getViewBox method that handles PySide6 compatibility."""
            try:
                return original_getViewBox(self)
            except AttributeError as e:
                if "'QWidgetItem' object has no attribute 'parentItem'" in str(e):
                    # Return None for incompatible items to avoid crash
                    return None
                raise

        pyqtgraph.graphicsItems.GraphicsItem.GraphicsItem.getViewBox = fixed_getViewBox
    except ImportError:
        pass  # PyQtGraph not available

    # Mock the initialization to avoid actual file system operations
    with patch("xpcs_toolkit.xpcs_viewer.ViewerKernel") as mock_vk_class:
        mock_vk_class.return_value = mock_viewer_kernel

        # Create the main window
        window = XpcsViewer(path=None)
        qtbot.addWidget(window)

        # Manually set the viewer kernel since we're mocking it
        window.vk = mock_viewer_kernel

        # Mock expensive GUI operations that cause timeouts during tab switching
        # Be selective - mock only the operations that cause hangs, not core functionality
        window.update_plot_sync = Mock()  # Prevent expensive plot synchronization
        window.init_diffusion = Mock()  # Prevent diffusion plot initialization
        # Don't mock plot_g2 - it's needed for fitting workflow
        window.plot_saxs2d = Mock()
        window.plot_saxs1d = Mock()
        window.plot_qmap = Mock()
        window.plot_twotime = Mock()
        window.plot_stability = Mock()
        window.plot_intt = Mock()
        # Don't mock refit_g2 - it's needed for fitting workflow

        # Show window for interaction testing
        window.show()
        qtbot.wait(100)  # Wait for window to be shown

        yield window

        # Comprehensive cleanup to prevent test state pollution
        try:
            # Restore original PyQtGraph methods if they were patched
            if original_buildTree is not None:
                try:
                    import pyqtgraph.widgets.DataTreeWidget

                    pyqtgraph.widgets.DataTreeWidget.DataTreeWidget.buildTree = (
                        original_buildTree
                    )
                except ImportError:
                    pass

            if original_getViewBox is not None:
                try:
                    import pyqtgraph.graphicsItems.GraphicsItem

                    pyqtgraph.graphicsItems.GraphicsItem.GraphicsItem.getViewBox = (
                        original_getViewBox
                    )
                except ImportError:
                    pass

            # Clear all child widgets and their connections
            for child in window.findChildren(QtWidgets.QWidget):
                child.setParent(None)
                child.deleteLater()

            # Close and delete the window
            window.close()
            window.setParent(None)
            window.deleteLater()

            # Force Qt event processing to complete deletions
            qtbot.wait(50)

            # Clear Qt application state
            app = QtWidgets.QApplication.instance()
            if app:
                app.processEvents()
                # Clear any remaining widgets
                for widget in app.allWidgets():
                    if widget and not widget.parent():
                        widget.close()
                        widget.deleteLater()
                app.processEvents()

        except Exception as e:
            # Don't let cleanup failures break tests
            print(f"Warning: GUI cleanup failed: {e}")

        # Clear any global Qt caches or state
        import gc

        gc.collect()  # Force garbage collection of Qt objects


@pytest.fixture
def gui_plot_widget():
    """Create a plot widget for testing."""
    import pyqtgraph as pg

    widget = pg.PlotWidget()
    widget.show()  # Make widget visible for tests
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
            """Validate tab widget consistency without triggering expensive operations."""
            issues = []

            # Store current tab to restore later
            tab_widget.currentIndex()

            for i in range(tab_widget.count()):
                # Don't set current index - just check if widget exists
                widget = tab_widget.widget(
                    i
                )  # Use widget(i) instead of setting current

                if widget is None:
                    issues.append(f"Tab {i} has no widget")
                elif not widget.isEnabled():
                    issues.append(f"Tab {i} widget is disabled")

            # Restore original tab without triggering updates
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
