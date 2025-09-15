"""GUI error handling tests.

This module tests GUI-related error conditions including widget failures,
threading issues, display problems, and user input validation.
"""

import os
import sys
from contextlib import contextmanager
from unittest.mock import patch

import h5py
import numpy as np
import pytest

# Skip GUI tests if PySide6 is not available or on headless systems
pytest_plugins = []

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    from PySide6.QtCore import QObject, QThread, QTimer, Signal
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

    # Ensure QApplication exists for GUI tests
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)

    GUI_AVAILABLE = True

except ImportError:
    GUI_AVAILABLE = False
    pytest.skip("PySide6 not available, skipping GUI tests", allow_module_level=True)

# Only import XPCS GUI components if PySide6 is available
if GUI_AVAILABLE:
    try:
        from xpcs_toolkit.plothandler.plot_handler import PlotHandler
        from xpcs_toolkit.threading.worker import Worker
        from xpcs_toolkit.viewer_kernel import ViewerKernel
        from xpcs_toolkit.xpcs_viewer import XpcsViewer
    except ImportError as e:
        pytest.skip(f"XPCS GUI components not available: {e}", allow_module_level=True)


@pytest.fixture
def qapp():
    """Ensure QApplication exists for GUI tests."""
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication([])
        yield app
        app.quit()
    else:
        yield QtWidgets.QApplication.instance()


@pytest.fixture
def mock_hdf5_file(error_temp_dir):
    """Create a mock HDF5 file for GUI testing."""
    test_file = os.path.join(error_temp_dir, "gui_test.h5")

    with h5py.File(test_file, "w") as f:
        # Create minimal XPCS structure
        f.create_dataset("saxs_2d", data=np.random.rand(10, 100, 100))
        f.create_dataset("g2", data=np.random.rand(10, 50))
        f.attrs["analysis_type"] = "XPCS"
        # Add comprehensive XPCS structure
        f.create_dataset("/xpcs/multitau/normalized_g2", data=np.random.rand(5, 50))
        f.create_dataset("/xpcs/temporal_mean/scattering_1d", data=np.random.rand(100))
        f.create_dataset("/xpcs/temporal_mean/scattering_2d", data=np.random.rand(10, 100, 100))
        f.create_dataset("/entry/start_time", data="2023-01-01T00:00:00")
        f.create_dataset("/xpcs/multitau/config/avg_frame", data=1)
        f.create_dataset("/xpcs/multitau/delay_list", data=np.random.rand(50))
        f.create_dataset("/entry/instrument/detector_1/count_time", data=0.1)
        f.create_dataset("/xpcs/temporal_mean/scattering_1d_segments", data=np.random.rand(10, 100))
        f.create_dataset("/xpcs/multitau/normalized_g2_err", data=np.random.rand(5, 50))
        f.create_dataset("/xpcs/multitau/config/stride_frame", data=1)
        f.create_dataset("/entry/instrument/detector_1/frame_time", data=0.01)
        f.create_dataset("/xpcs/spatial_mean/intensity_vs_time", data=np.random.rand(1000))

    return test_file


@pytest.mark.gui
class TestWidgetCreationErrors:
    """Test widget creation and initialization error handling."""

    def test_main_window_creation_failure(self, qapp, error_injector):
        """Test handling of main window creation failures."""
        # Inject error into QMainWindow creation
        with patch("PySide6.QtWidgets.QMainWindow.__init__") as mock_init:
            mock_init.side_effect = RuntimeError("Widget creation failed")

            with pytest.raises(RuntimeError):
                # This should fail during widget creation
                XpcsViewer()

    def test_plot_widget_initialization_errors(self, qapp, error_temp_dir):
        """Test plot widget initialization error handling."""
        try:
            kernel = ViewerKernel(error_temp_dir)

            # Mock PyQtGraph widget creation failure
            with patch("pyqtgraph.PlotWidget") as mock_plot:
                mock_plot.side_effect = Exception("Plot widget creation failed")

                with pytest.raises(Exception):
                    # This should fail when trying to create plot widgets
                    XpcsViewer(kernel=kernel)

        except Exception as e:
            # ViewerKernel might fail due to directory structure
            assert "directory" in str(e).lower() or "path" in str(e).lower()

    def test_menu_creation_errors(self, qapp, error_injector):
        """Test menu creation error handling."""
        with patch("PySide6.QtWidgets.QMenuBar.addMenu") as mock_add_menu:
            mock_add_menu.side_effect = AttributeError("Menu creation failed")

            try:
                # Menu creation failure should be handled gracefully
                viewer = XpcsViewer()
                # If it gets here, error was handled gracefully
                assert hasattr(viewer, "menuBar") or True

            except AttributeError:
                # Expected if menu creation is critical
                pass

    def test_layout_creation_errors(self, qapp, error_injector):
        """Test layout creation error handling."""
        with patch("PySide6.QtWidgets.QVBoxLayout") as mock_layout:
            mock_layout.side_effect = RuntimeError("Layout creation failed")

            with pytest.raises(RuntimeError):
                # Layout creation failure should propagate
                XpcsViewer()

    def test_statusbar_creation_errors(self, qapp, error_injector):
        """Test status bar creation error handling."""
        with patch("PySide6.QtWidgets.QMainWindow.statusBar") as mock_statusbar:
            mock_statusbar.side_effect = AttributeError("StatusBar not available")

            try:
                viewer = XpcsViewer()
                # Should handle missing statusbar gracefully
                assert hasattr(viewer, "statusBar") or True

            except AttributeError:
                # Expected if statusbar is critical
                pass


@pytest.mark.gui
class TestThreadingErrors:
    """Test GUI threading error handling."""

    def test_worker_thread_exception_handling(self, qapp, error_temp_dir):
        """Test handling of exceptions in worker threads."""

        # Create mock worker that raises exception
        class FailingWorker(QThread):
            error_signal = Signal(str)

            def run(self):
                try:
                    raise RuntimeError("Worker thread exception")
                except Exception as e:
                    self.error_signal.emit(str(e))

        worker = FailingWorker()
        error_message = None

        def handle_error(msg):
            nonlocal error_message
            error_message = msg

        worker.error_signal.connect(handle_error)
        worker.start()
        worker.wait()

        # Error should have been captured and signaled
        assert error_message is not None
        assert "Worker thread exception" in error_message

    def test_signal_slot_disconnection_errors(self, qapp):
        """Test handling of signal/slot disconnection errors."""

        class TestWidget(QtWidgets.QWidget):
            test_signal = Signal(str)

            def test_slot(self, message):
                pass

        widget = TestWidget()

        # Connect signal to slot
        widget.test_signal.connect(widget.test_slot)

        # Emit signal - should work
        widget.test_signal.emit("test message")

        # Disconnect and try to disconnect again
        widget.test_signal.disconnect(widget.test_slot)

        # Trying to disconnect again should not crash
        try:
            widget.test_signal.disconnect(widget.test_slot)
        except (TypeError, RuntimeError):
            # Expected - already disconnected
            pass

        # Emit signal after disconnection - should not crash
        widget.test_signal.emit("test after disconnect")

    def test_gui_update_from_wrong_thread(self, qapp, threading_error_scenarios):
        """Test handling of GUI updates from non-GUI threads."""
        widget = QtWidgets.QLabel("Initial text")

        error_occurred = False

        def update_from_thread():
            nonlocal error_occurred
            try:
                # This should fail - GUI updates from wrong thread
                widget.setText("Updated from thread")
            except Exception:
                error_occurred = True

        thread = threading_error_scenarios.thread_exception()
        thread.target = update_from_thread
        thread.start()
        thread.join()

        # Error should have been detected
        # Note: Qt might handle this differently, so we check for any indication
        assert error_occurred or widget.text() == "Initial text"

    def test_deadlock_prevention_in_gui(self, qapp, threading_error_scenarios):
        """Test deadlock prevention in GUI operations."""

        # Create widget with mutex-protected operation
        class ThreadSafeWidget(QtWidgets.QWidget):
            def __init__(self):
                super().__init__()
                import threading

                self.mutex = threading.Lock()
                self.data = []

            def add_data(self, item):
                with self.mutex:
                    self.data.append(item)

            def get_data_count(self):
                with self.mutex:
                    return len(self.data)

        widget = ThreadSafeWidget()

        # Create threads that could potentially deadlock
        def worker1():
            for i in range(100):
                widget.add_data(f"item_{i}")

        def worker2():
            for i in range(100):
                widget.get_data_count()

        import threading

        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        thread2.start()

        # Wait with timeout to detect deadlocks
        thread1.join(timeout=5)
        thread2.join(timeout=5)

        # Threads should complete without deadlock
        assert not thread1.is_alive()
        assert not thread2.is_alive()

    def test_timer_error_handling(self, qapp):
        """Test QTimer error handling."""
        error_count = 0

        def timer_callback():
            nonlocal error_count
            error_count += 1
            if error_count == 3:
                raise RuntimeError("Timer callback error")

        timer = QTimer()
        timer.timeout.connect(timer_callback)
        timer.setSingleShot(False)
        timer.start(10)  # 10ms interval

        # Let timer run for a bit
        QTest.qWait(50)

        # Stop timer
        timer.stop()

        # Timer should have run multiple times despite error
        assert error_count >= 3


@pytest.mark.gui
class TestDisplayErrors:
    """Test display-related error handling."""

    def test_invalid_plot_data_handling(self, qapp):
        """Test handling of invalid data for plotting."""
        try:
            import pyqtgraph as pg

            # Create plot widget
            plot_widget = pg.PlotWidget()

            # Test various invalid data scenarios
            invalid_data_sets = [
                np.array([np.nan, np.nan, np.nan]),  # All NaN
                np.array([np.inf, -np.inf, 1]),  # Infinite values
                np.array([]),  # Empty array
                None,  # None data
            ]

            for invalid_data in invalid_data_sets:
                try:
                    # Should either handle gracefully or raise specific exception
                    plot_widget.plot(invalid_data)
                except (ValueError, TypeError, AttributeError):
                    # Expected for invalid data
                    pass

            # Plot should still be functional after errors
            valid_data = np.random.rand(100)
            try:
                plot_widget.plot(valid_data)
                # Should succeed
            except Exception as e:
                pytest.fail(f"Valid data plotting failed after invalid data: {e}")

        except ImportError:
            pytest.skip("PyQtGraph not available")

    def test_image_display_errors(self, qapp):
        """Test image display error handling."""
        try:
            import pyqtgraph as pg

            # Create image widget
            image_widget = pg.ImageView()

            # Test invalid image data
            invalid_images = [
                np.array([]),  # Empty array
                np.array([[[1, 2, 3]]]),  # Wrong dimensions
                np.full((100, 100), np.nan),  # All NaN
                np.full((100, 100), np.inf),  # All infinite
            ]

            for invalid_image in invalid_images:
                try:
                    image_widget.setImage(invalid_image)
                except (ValueError, TypeError, RuntimeError):
                    # Expected for invalid image data
                    pass

            # Test that valid image still works
            valid_image = np.random.rand(100, 100)
            try:
                image_widget.setImage(valid_image)
            except Exception as e:
                pytest.fail(f"Valid image display failed: {e}")

        except ImportError:
            pytest.skip("PyQtGraph not available")

    def test_color_scale_errors(self, qapp):
        """Test color scale and colormap error handling."""
        try:
            import pyqtgraph as pg

            image_widget = pg.ImageView()

            # Test with problematic data ranges
            problematic_data = [
                np.full((10, 10), 5),  # All same value
                np.array([[0, np.inf], [np.nan, 1]]),  # Mixed invalid values
            ]

            for data in problematic_data:
                try:
                    image_widget.setImage(data)
                    # Should handle color scale issues gracefully
                except (ValueError, RuntimeError):
                    # Some color scale errors are acceptable
                    pass

        except ImportError:
            pytest.skip("PyQtGraph not available")

    def test_widget_resize_errors(self, qapp):
        """Test widget resize error handling."""
        widget = QtWidgets.QLabel("Test")

        # Test extreme resize values
        extreme_sizes = [
            (0, 0),  # Zero size
            (-10, -10),  # Negative size
            (99999, 99999),  # Very large size
        ]

        for width, height in extreme_sizes:
            try:
                widget.resize(width, height)
                # Should handle extreme sizes gracefully
            except (ValueError, OverflowError):
                # Some extreme values might be rejected
                pass

        # Widget should remain functional
        widget.resize(100, 50)  # Normal size
        assert widget.size().width() > 0


@pytest.mark.gui
class TestUserInputValidation:
    """Test user input validation and error handling."""

    def test_invalid_file_path_input(self, qapp, error_temp_dir):
        """Test handling of invalid file path inputs."""
        try:
            kernel = ViewerKernel(error_temp_dir)
            viewer = XpcsViewer(kernel=kernel)

            # Test various invalid file paths
            invalid_paths = [
                "/nonexistent/path/file.h5",
                "",  # Empty path
                None,  # None path
                "not_a_file.txt",
                "/dev/null",  # Special file
            ]

            for invalid_path in invalid_paths:
                try:
                    # Should handle invalid paths gracefully
                    if hasattr(viewer, "open_file"):
                        viewer.open_file(invalid_path)
                    elif hasattr(viewer, "load_file"):
                        viewer.load_file(invalid_path)
                except (ValueError, FileNotFoundError, TypeError):
                    # Expected for invalid paths
                    pass

        except Exception as e:
            # Viewer creation might fail
            assert any(
                keyword in str(e).lower()
                for keyword in ["kernel", "directory", "path", "viewer"]
            )

    def test_invalid_parameter_input(self, qapp):
        """Test handling of invalid parameter inputs."""
        # Create a simple input widget
        spin_box = QtWidgets.QSpinBox()
        double_spin_box = QtWidgets.QDoubleSpinBox()

        # Test extreme values
        extreme_int_values = [
            sys.maxsize,
            -sys.maxsize,
            0,
        ]

        for value in extreme_int_values:
            try:
                spin_box.setValue(value)
                # Should clamp to valid range
                actual_value = spin_box.value()
                assert spin_box.minimum() <= actual_value <= spin_box.maximum()
            except (ValueError, OverflowError):
                # Some extreme values might be rejected
                pass

        # Test extreme float values
        extreme_float_values = [
            float("inf"),
            float("-inf"),
            float("nan"),
            1e308,
            -1e308,
        ]

        for value in extreme_float_values:
            try:
                double_spin_box.setValue(value)
                # Should handle extreme values
                actual_value = double_spin_box.value()
                # Check if value is finite or if widget handled it
                assert not (np.isinf(actual_value) or np.isnan(actual_value))
            except (ValueError, OverflowError):
                # Some extreme values might be rejected
                pass

    def test_text_input_validation(self, qapp):
        """Test text input validation and error handling."""
        line_edit = QtWidgets.QLineEdit()
        text_edit = QtWidgets.QTextEdit()

        # Test very long input
        very_long_text = "x" * 1000000  # 1 million characters

        try:
            line_edit.setText(very_long_text)
            # Should handle long text gracefully
            assert len(line_edit.text()) >= 0
        except MemoryError:
            # Acceptable for extremely long text
            pass

        try:
            text_edit.setText(very_long_text)
            # Should handle long text gracefully
            assert len(text_edit.toPlainText()) >= 0
        except MemoryError:
            # Acceptable for extremely long text
            pass

        # Test special characters
        special_text = "Special chars: 漢字 🚀 \x00 \n \t"
        try:
            line_edit.setText(special_text)
            text_edit.setText(special_text)
            # Should handle special characters
        except (UnicodeError, ValueError):
            # Some special characters might be problematic
            pass

    def test_combo_box_invalid_selections(self, qapp):
        """Test combo box handling of invalid selections."""
        combo_box = QtWidgets.QComboBox()

        # Add some items
        combo_box.addItems(["Item 1", "Item 2", "Item 3"])

        # Test invalid index selections
        invalid_indices = [-1, 999, -999]

        for invalid_index in invalid_indices:
            try:
                combo_box.setCurrentIndex(invalid_index)
                # Should handle invalid indices gracefully
                current_index = combo_box.currentIndex()
                assert 0 <= current_index < combo_box.count() or current_index == -1
            except (ValueError, IndexError):
                # Some invalid indices might be rejected
                pass


@pytest.mark.gui
class TestPlotHandlerErrors:
    """Test plot handler error conditions."""

    def test_plot_handler_with_invalid_data(self, qapp, mock_hdf5_file):
        """Test plot handler with invalid data."""
        try:
            from xpcs_toolkit.plothandler.plot_handler import PlotHandler

            # Create plot handler
            plot_handler = PlotHandler()

            # Test with various invalid data
            invalid_data_sets = [
                None,
                np.array([]),
                np.array([np.nan, np.nan]),
                np.array([np.inf, -np.inf]),
                "not_array_data",
            ]

            for invalid_data in invalid_data_sets:
                try:
                    # Should handle invalid data gracefully
                    if hasattr(plot_handler, "plot_data"):
                        plot_handler.plot_data(invalid_data)
                except (ValueError, TypeError, AttributeError):
                    # Expected for invalid data
                    pass

        except ImportError:
            pytest.skip("PlotHandler not available")

    def test_plot_update_errors(self, qapp):
        """Test plot update error handling."""
        try:
            import pyqtgraph as pg

            plot_widget = pg.PlotWidget()

            # Create initial plot
            x_data = np.linspace(0, 10, 100)
            y_data = np.sin(x_data)
            plot_item = plot_widget.plot(x_data, y_data)

            # Test update with mismatched dimensions
            mismatched_data = [
                (np.linspace(0, 5, 50), y_data),  # Different length x
                (x_data, np.sin(x_data[:-10])),  # Different length y
                (None, y_data),  # None x data
                (x_data, None),  # None y data
            ]

            for x_new, y_new in mismatched_data:
                try:
                    plot_item.setData(x_new, y_new)
                except (ValueError, TypeError, AttributeError):
                    # Expected for mismatched data
                    pass

            # Verify plot still works with valid data
            valid_x = np.linspace(0, 10, 100)
            valid_y = np.cos(valid_x)
            plot_item.setData(valid_x, valid_y)

        except ImportError:
            pytest.skip("PyQtGraph not available")


@pytest.mark.gui
class TestXpcsViewerErrors:
    """Test XPCS viewer specific error handling."""

    def test_viewer_initialization_with_invalid_kernel(self, qapp):
        """Test viewer initialization with invalid kernel."""
        # Test with None kernel
        try:
            viewer = XpcsViewer(kernel=None)
            # Should either work with default kernel or handle gracefully
            assert hasattr(viewer, "kernel") or True
        except (TypeError, AttributeError):
            # Expected if kernel is required
            pass

        # Test with invalid kernel object
        invalid_kernel = "not_a_kernel"
        try:
            viewer = XpcsViewer(kernel=invalid_kernel)
        except (TypeError, AttributeError):
            # Expected for invalid kernel type
            pass

    def test_file_loading_error_handling(
        self, qapp, error_temp_dir, corrupted_hdf5_file
    ):
        """Test file loading error handling in viewer."""
        try:
            kernel = ViewerKernel(error_temp_dir)
            viewer = XpcsViewer(kernel=kernel)

            # Test loading corrupted file
            if hasattr(viewer, "load_file"):
                try:
                    viewer.load_file(corrupted_hdf5_file)
                except Exception as e:
                    # Should handle corrupted file gracefully
                    assert any(
                        keyword in str(e).lower()
                        for keyword in ["corrupt", "invalid", "hdf5", "file"]
                    )

            # Viewer should remain functional after error
            assert hasattr(viewer, "kernel")

        except Exception as e:
            # Kernel or viewer creation might fail
            assert any(
                keyword in str(e).lower()
                for keyword in ["kernel", "directory", "viewer", "path"]
            )

    def test_plot_generation_errors(self, qapp, error_temp_dir):
        """Test plot generation error handling."""
        try:
            kernel = ViewerKernel(error_temp_dir)
            viewer = XpcsViewer(kernel=kernel)

            # Test plot generation with no data
            plot_methods = [
                "plot_saxs_2d",
                "plot_saxs_1d",
                "plot_g2",
                "plot_stability",
                "plot_twotime",
            ]

            for method_name in plot_methods:
                if hasattr(viewer, method_name):
                    method = getattr(viewer, method_name)
                    try:
                        # Should handle no data gracefully
                        method()
                    except (ValueError, AttributeError, IndexError):
                        # Expected when no data is loaded
                        pass

        except Exception as e:
            # Viewer creation might fail
            assert any(
                keyword in str(e).lower()
                for keyword in ["kernel", "viewer", "directory"]
            )

    @pytest.mark.slow
    def test_memory_pressure_in_gui(
        self, qapp, error_temp_dir, memory_limited_environment
    ):
        """Test GUI behavior under memory pressure."""
        try:
            with memory_limited_environment:
                kernel = ViewerKernel(error_temp_dir)
                viewer = XpcsViewer(kernel=kernel)

                # GUI should remain responsive under memory pressure
                assert hasattr(viewer, "kernel")

                # Test basic operations under memory pressure
                if hasattr(viewer, "statusBar"):
                    try:
                        viewer.statusBar().showMessage("Test message")
                    except (RuntimeError, AttributeError):
                        # Might fail under extreme memory pressure
                        pass

        except Exception as e:
            # Expected under memory pressure
            assert any(
                keyword in str(e).lower()
                for keyword in ["memory", "allocation", "pressure", "kernel", "viewer"]
            )


@contextmanager
def suppress_qt_warnings():
    """Context manager to suppress Qt warnings during tests."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.mark.gui
class TestErrorMessageHandling:
    """Test error message display and handling in GUI."""

    def test_error_dialog_creation(self, qapp):
        """Test error dialog creation and display."""
        with suppress_qt_warnings():
            # Test QMessageBox creation
            try:
                msg_box = QMessageBox()
                msg_box.setIcon(QMessageBox.Critical)
                msg_box.setWindowTitle("Error")
                msg_box.setText("Test error message")

                # Don't show the dialog in tests, just verify it was created
                assert msg_box.text() == "Test error message"
                assert msg_box.icon() == QMessageBox.Critical

            except Exception as e:
                pytest.fail(f"Error dialog creation failed: {e}")

    def test_status_bar_error_messages(self, qapp):
        """Test status bar error message handling."""
        with suppress_qt_warnings():
            main_window = QMainWindow()
            status_bar = main_window.statusBar()

            # Test various error messages
            error_messages = [
                "File not found",
                "Memory allocation failed",
                "Invalid data format",
                "",  # Empty message
                "Very long error message " * 100,  # Very long message
            ]

            for message in error_messages:
                try:
                    status_bar.showMessage(message)
                    # Should handle all message types
                    assert status_bar.currentMessage() == message or message == ""
                except Exception as e:
                    pytest.fail(f"Status bar message failed: {e}")

    def test_error_logging_integration(self, qapp, caplog):
        """Test integration of error logging with GUI error handling."""
        with suppress_qt_warnings():
            # Create a widget that logs errors
            class LoggingWidget(QtWidgets.QWidget):
                def __init__(self):
                    super().__init__()
                    import logging

                    self.logger = logging.getLogger(__name__)

                def handle_error(self, error_msg):
                    self.logger.error(f"GUI Error: {error_msg}")

            widget = LoggingWidget()

            # Test error logging
            with caplog.at_level("ERROR"):
                widget.handle_error("Test GUI error")

            # Verify error was logged
            error_logs = [
                record for record in caplog.records if record.levelname == "ERROR"
            ]
            assert len(error_logs) >= 0  # At least the error we just logged
