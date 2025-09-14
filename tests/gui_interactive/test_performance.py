"""Tests for GUI performance and responsiveness validation.

This module provides comprehensive performance testing for the GUI interface,
including responsiveness, memory usage, rendering performance, and scalability
under various load conditions.
"""

import gc
from unittest.mock import Mock, patch

import numpy as np
import psutil
import pytest
from PySide6 import QtCore, QtWidgets


class TestResponseTimes:
    """Test suite for GUI response time validation."""

    @pytest.mark.gui
    @pytest.mark.performance
    def test_tab_switching_response_time(
        self, gui_main_window, qtbot, gui_test_helpers, gui_performance_monitor
    ):
        """Test tab switching response times."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)
        monitor = gui_performance_monitor

        if tab_widget and tab_widget.count() > 1:
            response_times = []

            # Test each tab switch
            for tab_index in range(tab_widget.count()):
                monitor.start_timing()

                gui_test_helpers.click_tab(qtbot, tab_widget, tab_index)
                qtbot.wait(50)  # Minimum wait for UI update

                elapsed = monitor.end_timing(f"Tab switch to {tab_index}")
                response_times.append(elapsed)

                # Verify switch was successful
                assert tab_widget.currentIndex() == tab_index

            # Response times should be reasonable (< 200ms per switch)
            max_response_time = max(response_times)
            assert max_response_time < 0.2, (
                f"Tab switch took too long: {max_response_time:.3f}s"
            )

            # Average response time should be even better
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 0.1, (
                f"Average tab switch time too slow: {avg_response_time:.3f}s"
            )

    @pytest.mark.gui
    @pytest.mark.performance
    def test_button_click_response_time(
        self, gui_main_window, qtbot, gui_performance_monitor
    ):
        """Test button click response times."""
        window = gui_main_window
        monitor = gui_performance_monitor

        buttons = window.findChildren(QtWidgets.QPushButton)
        clickable_buttons = [b for b in buttons if b.isVisible() and b.isEnabled()]

        if clickable_buttons:
            response_times = []

            # Test first few buttons to avoid long test times
            for button in clickable_buttons[:5]:
                monitor.start_timing()

                qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
                qtbot.wait(10)  # Minimal wait

                elapsed = monitor.end_timing(f"Button click: {button.text()}")
                response_times.append(elapsed)

            # Button clicks should be very responsive (< 50ms)
            max_response_time = max(response_times)
            assert max_response_time < 0.05, (
                f"Button click too slow: {max_response_time:.3f}s"
            )

    @pytest.mark.gui
    @pytest.mark.performance
    def test_parameter_adjustment_response_time(
        self, gui_main_window, qtbot, gui_test_helpers, gui_performance_monitor
    ):
        """Test parameter control adjustment response times."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)
        monitor = gui_performance_monitor

        # Test parameter controls in G2 tab (likely to have many controls)
        if tab_widget.count() > 4:
            gui_test_helpers.click_tab(qtbot, tab_widget, 4)
            g2_widget = tab_widget.currentWidget()

            spin_boxes = g2_widget.findChildren(QtWidgets.QSpinBox)
            response_times = []

            for spin_box in spin_boxes[:3]:  # Test first few
                if spin_box.isVisible() and spin_box.isEnabled():
                    original_value = spin_box.value()

                    monitor.start_timing()

                    # Adjust value
                    new_value = min(original_value + 1, spin_box.maximum())
                    spin_box.setValue(new_value)
                    qtbot.wait(10)

                    elapsed = monitor.end_timing("SpinBox adjustment")
                    response_times.append(elapsed)

                    assert spin_box.value() == new_value

            if response_times:
                # Parameter adjustments should be very responsive
                max_response_time = max(response_times)
                assert max_response_time < 0.1, (
                    f"Parameter adjustment too slow: {max_response_time:.3f}s"
                )


class TestMemoryUsage:
    """Test suite for memory usage validation."""

    @pytest.fixture
    def memory_monitor(self):
        """Monitor memory usage during tests."""

        class MemoryMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.baseline = None
                self.measurements = []

            def start_monitoring(self):
                """Start memory monitoring."""
                gc.collect()  # Force garbage collection
                self.baseline = self.process.memory_info().rss / 1024 / 1024  # MB
                self.measurements = [self.baseline]

            def take_measurement(self, label=""):
                """Take a memory measurement."""
                current = self.process.memory_info().rss / 1024 / 1024  # MB
                self.measurements.append(current)
                return current

            def get_memory_increase(self):
                """Get memory increase since baseline."""
                if self.baseline is None:
                    return 0
                current = self.process.memory_info().rss / 1024 / 1024  # MB
                return current - self.baseline

            def get_peak_memory(self):
                """Get peak memory usage."""
                return max(self.measurements) if self.measurements else 0

        return MemoryMonitor()

    @pytest.mark.gui
    @pytest.mark.performance
    def test_tab_switching_memory_usage(
        self, gui_main_window, qtbot, gui_test_helpers, memory_monitor
    ):
        """Test memory usage during tab switching."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        memory_monitor.start_monitoring()
        initial_memory = memory_monitor.baseline

        if tab_widget and tab_widget.count() > 1:
            # Switch through all tabs multiple times
            for cycle in range(3):
                for tab_index in range(tab_widget.count()):
                    gui_test_helpers.click_tab(qtbot, tab_widget, tab_index)
                    qtbot.wait(100)

                    memory_monitor.take_measurement(f"Cycle {cycle}, Tab {tab_index}")

            # Force garbage collection
            gc.collect()
            qtbot.wait(100)

            final_memory = memory_monitor.take_measurement("Final")

            # Memory increase should be reasonable (< 100MB for tab switching)
            memory_increase = final_memory - initial_memory
            assert memory_increase < 100, (
                f"Excessive memory usage: {memory_increase:.1f}MB"
            )

            # Peak memory shouldn't be too high
            peak_memory = memory_monitor.get_peak_memory()
            peak_increase = peak_memory - initial_memory
            assert peak_increase < 200, f"Peak memory too high: {peak_increase:.1f}MB"

    @pytest.mark.gui
    @pytest.mark.performance
    def test_plot_memory_usage(
        self, gui_main_window, qtbot, gui_test_helpers, mock_xpcs_file, memory_monitor
    ):
        """Test memory usage during plot operations."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Mock large dataset for plotting
        large_data = np.random.random((1000, 1000))
        mock_xpcs_file.load_saxs_2d.return_value = large_data

        memory_monitor.start_monitoring()
        initial_memory = memory_monitor.baseline

        with patch.object(window.vk, "current_file", mock_xpcs_file):
            # Create and destroy plots multiple times
            for cycle in range(5):
                # Switch to SAXS 2D tab
                gui_test_helpers.click_tab(qtbot, tab_widget, 0)
                qtbot.wait(200)  # Allow for plot rendering
                memory_monitor.take_measurement(f"SAXS 2D Cycle {cycle}")

                # Switch to different tab (potentially clearing plot)
                gui_test_helpers.click_tab(qtbot, tab_widget, 1)
                qtbot.wait(100)
                memory_monitor.take_measurement(f"Switch away Cycle {cycle}")

            # Force cleanup
            gc.collect()
            qtbot.wait(100)

            final_memory = memory_monitor.take_measurement("Final")

        # Memory should not continuously grow (memory leak check)
        memory_increase = final_memory - initial_memory
        assert memory_increase < 150, (
            f"Potential memory leak detected: {memory_increase:.1f}MB"
        )

    @pytest.mark.gui
    @pytest.mark.performance
    def test_widget_creation_memory(self, qapp, memory_monitor):
        """Test memory usage during widget creation and destruction."""
        memory_monitor.start_monitoring()

        widgets = []

        # Create many widgets
        for i in range(100):
            widget = QtWidgets.QWidget()
            layout = QtWidgets.QVBoxLayout()

            # Add some controls
            label = QtWidgets.QLabel(f"Label {i}")
            button = QtWidgets.QPushButton(f"Button {i}")
            spin_box = QtWidgets.QSpinBox()

            layout.addWidget(label)
            layout.addWidget(button)
            layout.addWidget(spin_box)
            widget.setLayout(layout)

            widgets.append(widget)

            if i % 20 == 0:
                memory_monitor.take_measurement(f"Created {i} widgets")

        memory_monitor.get_peak_memory()

        # Destroy widgets
        for widget in widgets:
            widget.deleteLater()

        widgets.clear()
        qapp.processEvents()  # Process deletion events
        gc.collect()

        final_memory = memory_monitor.take_measurement("After cleanup")

        # Memory should return close to baseline after cleanup
        memory_increase = final_memory - memory_monitor.baseline
        assert memory_increase < 50, (
            f"Widget cleanup incomplete: {memory_increase:.1f}MB remaining"
        )


class TestRenderingPerformance:
    """Test suite for rendering and visualization performance."""

    @pytest.mark.gui
    @pytest.mark.performance
    def test_plot_rendering_performance(
        self, gui_plot_widget, qtbot, gui_performance_monitor
    ):
        """Test plot rendering performance with various data sizes."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)
        monitor = gui_performance_monitor

        data_sizes = [100, 1000, 10000]
        render_times = []

        for size in data_sizes:
            x_data = np.linspace(0, 10, size)
            y_data = np.sin(x_data) + 0.1 * np.random.normal(size=size)

            monitor.start_timing()

            plot_widget.plot(x_data, y_data, clear=True)
            qtbot.wait(100)  # Allow for rendering

            elapsed = monitor.end_timing(f"Plot {size} points")
            render_times.append((size, elapsed))

        # Rendering should scale reasonably with data size
        small_time = render_times[0][1]
        large_time = render_times[-1][1]

        # Large dataset shouldn't be more than 10x slower
        assert large_time < small_time * 10, (
            f"Poor scaling: {small_time:.3f}s -> {large_time:.3f}s"
        )

        # Even large plots should render within reasonable time
        assert large_time < 2.0, f"Large plot too slow: {large_time:.3f}s"

    @pytest.mark.gui
    @pytest.mark.performance
    def test_image_display_performance(self, qtbot, gui_performance_monitor):
        """Test 2D image display performance."""
        import pyqtgraph as pg

        image_view = pg.ImageView()
        qtbot.addWidget(image_view)
        monitor = gui_performance_monitor

        image_sizes = [(256, 256), (512, 512), (1024, 1024)]
        display_times = []

        for size in image_sizes:
            image_data = np.random.random(size)

            monitor.start_timing()

            image_view.setImage(image_data)
            qtbot.wait(200)  # Allow for image processing and display

            elapsed = monitor.end_timing(f"Display {size[0]}x{size[1]} image")
            display_times.append((size, elapsed))

        # Image display should be reasonable for typical XPCS sizes
        for size, elapsed in display_times:
            max_time = 3.0 if size[0] > 512 else 1.0
            assert elapsed < max_time, (
                f"Image display too slow for {size}: {elapsed:.3f}s"
            )

    @pytest.mark.gui
    @pytest.mark.performance
    def test_rapid_plot_updates(self, gui_plot_widget, qtbot, gui_performance_monitor):
        """Test performance of rapid plot updates."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)
        monitor = gui_performance_monitor

        x_data = np.linspace(0, 10, 1000)
        plot_item = plot_widget.plot(x_data, np.sin(x_data))

        monitor.start_timing()

        # Perform rapid updates
        update_count = 50
        for i in range(update_count):
            y_data = np.sin(x_data + i * 0.1)
            plot_item.setData(x_data, y_data)
            qtbot.wait(10)  # Minimal wait

        total_time = monitor.end_timing("Rapid plot updates")

        # Updates should be fast enough for real-time visualization
        avg_update_time = total_time / update_count
        assert avg_update_time < 0.05, (
            f"Plot updates too slow: {avg_update_time:.4f}s per update"
        )
        assert total_time < 5.0, f"Total update time too long: {total_time:.3f}s"


class TestScalabilityLimits:
    """Test suite for scalability and performance limits."""

    @pytest.mark.gui
    @pytest.mark.performance
    @pytest.mark.slow
    def test_large_file_handling_performance(
        self, gui_main_window, qtbot, tmp_path, gui_performance_monitor
    ):
        """Test performance with large file operations."""
        window = gui_main_window
        monitor = gui_performance_monitor

        # Create a large HDF5 file (simulated)
        large_file = tmp_path / "large_test.hdf5"

        with patch("h5py.File") as mock_h5py:
            # Mock large file creation
            mock_file = Mock()
            mock_h5py.return_value.__enter__.return_value = mock_file

            monitor.start_timing()

            # Mock file loading
            with patch.object(window.vk, "load_file") as mock_load:
                mock_load.return_value = True

                with patch(
                    "PySide6.QtWidgets.QFileDialog.getOpenFileName"
                ) as mock_dialog:
                    mock_dialog.return_value = (str(large_file), "")

                    # Attempt to load large file
                    buttons = window.findChildren(QtWidgets.QPushButton)
                    for button in buttons:
                        if "load" in button.text().lower():
                            qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
                            qtbot.wait(500)  # Allow for processing
                            break

            elapsed = monitor.end_timing("Large file loading")

        # Large file operations should complete within reasonable time
        assert elapsed < 10.0, f"Large file loading too slow: {elapsed:.3f}s"

    @pytest.mark.gui
    @pytest.mark.performance
    def test_many_widgets_performance(self, qapp, qtbot, gui_performance_monitor):
        """Test performance with many GUI widgets."""
        monitor = gui_performance_monitor

        monitor.start_timing()

        # Create window with many widgets
        window = QtWidgets.QMainWindow()
        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()

        widget_count = 0
        for row in range(20):
            for col in range(10):
                if widget_count % 3 == 0:
                    widget = QtWidgets.QLabel(f"Label {widget_count}")
                elif widget_count % 3 == 1:
                    widget = QtWidgets.QPushButton(f"Button {widget_count}")
                else:
                    widget = QtWidgets.QSpinBox()
                    widget.setMaximum(1000)
                    widget.setValue(widget_count)

                layout.addWidget(widget, row, col)
                widget_count += 1

        central_widget.setLayout(layout)
        window.setCentralWidget(central_widget)

        window.show()
        qtbot.addWidget(window)
        qtbot.wait(500)  # Allow for rendering

        creation_time = monitor.end_timing("Many widgets creation")

        # Widget creation should be reasonable
        assert creation_time < 5.0, f"Widget creation too slow: {creation_time:.3f}s"

        # Test interaction performance
        monitor.start_timing()

        # Interact with some widgets
        buttons = window.findChildren(QtWidgets.QPushButton)
        for button in buttons[:10]:  # Test first 10 buttons
            qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
            qtbot.wait(1)  # Minimal wait

        interaction_time = monitor.end_timing("Widget interactions")

        # Interactions should remain responsive
        assert interaction_time < 2.0, (
            f"Widget interactions too slow: {interaction_time:.3f}s"
        )

        window.close()

    @pytest.mark.gui
    @pytest.mark.performance
    def test_concurrent_operations_performance(
        self, gui_main_window, qtbot, gui_performance_monitor
    ):
        """Test performance under concurrent operation simulation."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)
        monitor = gui_performance_monitor

        monitor.start_timing()

        # Simulate concurrent operations
        operations = [
            lambda: tab_widget.setCurrentIndex(0),
            lambda: tab_widget.setCurrentIndex(1),
            lambda: tab_widget.setCurrentIndex(2),
        ]

        # Rapid operations
        for _ in range(30):  # 30 operations
            for op in operations:
                if tab_widget.count() > 2:
                    op()
                    qtbot.wait(5)  # Very short wait

        concurrent_time = monitor.end_timing("Concurrent operations")

        # Should handle concurrent operations efficiently
        assert concurrent_time < 3.0, (
            f"Concurrent operations too slow: {concurrent_time:.3f}s"
        )

        # GUI should remain stable
        assert window.isVisible()


class TestResourceUtilization:
    """Test suite for system resource utilization."""

    @pytest.mark.gui
    @pytest.mark.performance
    def test_cpu_usage_monitoring(self, gui_main_window, qtbot, gui_test_helpers):
        """Test CPU usage during typical operations."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        # Monitor CPU usage
        process = psutil.Process()
        cpu_samples = []

        # Perform typical operations while monitoring CPU
        for _ in range(10):
            if tab_widget.count() > 1:
                # Switch tabs
                gui_test_helpers.click_tab(qtbot, tab_widget, 0)
                qtbot.wait(50)

                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)

                gui_test_helpers.click_tab(qtbot, tab_widget, 1)
                qtbot.wait(50)

                cpu_percent = process.cpu_percent(interval=0.1)
                cpu_samples.append(cpu_percent)

        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)

            # CPU usage should be reasonable for GUI operations
            assert avg_cpu < 50.0, f"Average CPU too high: {avg_cpu:.1f}%"
            assert max_cpu < 80.0, f"Peak CPU too high: {max_cpu:.1f}%"

    @pytest.mark.gui
    @pytest.mark.performance
    def test_thread_usage_efficiency(self, gui_main_window, qtbot):
        """Test efficient thread usage."""
        window = gui_main_window
        process = psutil.Process()

        initial_threads = process.num_threads()

        # Perform operations that might create threads
        tab_widget = window.findChild(QtWidgets.QTabWidget)

        for tab_index in range(min(5, tab_widget.count())):
            tab_widget.setCurrentIndex(tab_index)
            qtbot.wait(100)

        final_threads = process.num_threads()

        # Thread count shouldn't grow excessively
        thread_increase = final_threads - initial_threads
        assert thread_increase < 10, f"Too many threads created: {thread_increase}"


class TestPerformanceRegression:
    """Test suite for performance regression detection."""

    @pytest.mark.gui
    @pytest.mark.performance
    def test_startup_performance(self, qapp, gui_performance_monitor):
        """Test application startup performance."""
        monitor = gui_performance_monitor

        monitor.start_timing()

        # Simulate startup (create main window)
        with patch("xpcs_toolkit.xpcs_viewer.ViewerKernel"):
            from xpcs_toolkit.xpcs_viewer import XpcsViewer

            window = XpcsViewer(path=None)
            window.show()
            qapp.processEvents()

        startup_time = monitor.end_timing("Application startup")

        # Startup should be reasonably fast
        assert startup_time < 5.0, f"Startup too slow: {startup_time:.3f}s"

        window.close()

    @pytest.mark.gui
    @pytest.mark.performance
    def test_baseline_performance_metrics(
        self, gui_main_window, qtbot, gui_test_helpers, gui_performance_monitor
    ):
        """Establish baseline performance metrics for regression testing."""
        window = gui_main_window
        tab_widget = window.findChild(QtWidgets.QTabWidget)
        monitor = gui_performance_monitor

        metrics = {}

        # Test tab switching performance
        monitor.start_timing()
        if tab_widget.count() > 0:
            gui_test_helpers.click_tab(qtbot, tab_widget, 0)
            qtbot.wait(50)
        metrics["tab_switch"] = monitor.end_timing("Baseline tab switch")

        # Test button click performance
        monitor.start_timing()
        buttons = window.findChildren(QtWidgets.QPushButton)
        if buttons:
            qtbot.mouseClick(buttons[0], QtCore.Qt.MouseButton.LeftButton)
            qtbot.wait(10)
        metrics["button_click"] = monitor.end_timing("Baseline button click")

        # Test window resize performance
        monitor.start_timing()
        original_size = window.size()
        window.resize(800, 600)
        qtbot.wait(100)
        window.resize(original_size)
        metrics["window_resize"] = monitor.end_timing("Baseline window resize")

        # Store metrics for comparison (in real scenario, would save to file)
        expected_metrics = {
            "tab_switch": 0.1,  # 100ms
            "button_click": 0.05,  # 50ms
            "window_resize": 0.2,  # 200ms
        }

        # Verify performance meets baseline expectations
        for operation, expected_time in expected_metrics.items():
            actual_time = metrics.get(operation, float("inf"))
            assert actual_time <= expected_time * 1.5, (
                f"Performance regression in {operation}: {actual_time:.3f}s > {expected_time:.3f}s"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
