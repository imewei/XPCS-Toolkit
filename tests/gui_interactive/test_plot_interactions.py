"""Tests for plot widget interactions and visualization components.

This module tests PyQtGraph and Matplotlib integration, plot interactions,
zooming, panning, data selection, and visualization updates.
"""

import os

# Set Qt API to PySide6 before importing matplotlib
os.environ.setdefault("QT_API", "PySide6")

import matplotlib

# Set matplotlib backend to PySide6-compatible qtagg
matplotlib.use("qtagg")

import numpy as np
import pyqtgraph as pg
import pytest

# Import from the PySide6-compatible backend
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6 import QtCore, QtWidgets


class TestPyQtGraphIntegration:
    """Test suite for PyQtGraph widget interactions."""

    @pytest.mark.gui
    def test_pyqtgraph_plot_creation(self, gui_plot_widget, qtbot):
        """Test PyQtGraph plot widget creation and basic setup."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Verify widget is created properly
        assert plot_widget is not None
        assert isinstance(plot_widget, pg.PlotWidget)

        # Test basic plot functionality
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        plot_item = plot_widget.plot(x_data, y_data)

        assert plot_item is not None
        assert len(plot_widget.listDataItems()) == 1

    @pytest.mark.gui
    def test_plot_data_update(self, gui_plot_widget, qtbot):
        """Test dynamic plot data updates."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Create initial plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        plot_item = plot_widget.plot(x_data, y_data)

        initial_data_count = len(plot_widget.listDataItems())

        # Update with new data
        new_y_data = np.cos(x_data)
        plot_item.setData(x_data, new_y_data)
        qtbot.wait(50)

        # Verify data was updated (same number of items)
        assert len(plot_widget.listDataItems()) == initial_data_count

        # Add a second plot
        plot_widget.plot(x_data, new_y_data * 0.5, pen="r")
        qtbot.wait(50)

        # Verify second plot was added
        assert len(plot_widget.listDataItems()) == initial_data_count + 1

    @pytest.mark.gui
    def test_plot_legend_functionality(self, gui_plot_widget, qtbot):
        """Test plot legend creation and interaction."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Create plots with labels
        x_data = np.linspace(0, 10, 100)
        y1_data = np.sin(x_data)
        y2_data = np.cos(x_data)

        plot_widget.plot(x_data, y1_data, pen="b", name="sin(x)")
        plot_widget.plot(x_data, y2_data, pen="r", name="cos(x)")

        # Add legend
        legend = plot_widget.addLegend()
        qtbot.wait(100)

        # Verify legend was created
        assert legend is not None

    @pytest.mark.gui
    def test_image_view_functionality(self, qtbot):
        """Test ImageView widget for 2D data display."""
        # Create ImageView widget
        image_view = pg.ImageView()
        qtbot.addWidget(image_view)

        # Create test 2D data
        test_data = np.random.random((100, 100))
        image_view.setImage(test_data)
        qtbot.wait(100)

        # Verify image was set
        assert image_view.image is not None
        assert image_view.image.shape == test_data.shape

        # Test histogram interaction
        hist_widget = image_view.getHistogramWidget()
        assert hist_widget is not None

        # Test region selection
        roi = image_view.getRoiPlot()
        assert roi is not None


class TestMatplotlibIntegration:
    """Test suite for Matplotlib canvas integration."""

    @pytest.fixture
    def matplotlib_canvas(self, qapp):
        """Create a Matplotlib canvas for testing."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(5, 4), dpi=100)
        canvas = FigureCanvasQTAgg(figure)
        axes = figure.add_subplot(111)
        canvas.show()  # Make canvas visible for tests

        return canvas, figure, axes

    @pytest.mark.gui
    def test_matplotlib_canvas_creation(self, matplotlib_canvas, qtbot):
        """Test Matplotlib canvas creation and basic plotting."""
        canvas, _figure, axes = matplotlib_canvas
        qtbot.addWidget(canvas)

        # Create test plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        axes.plot(x_data, y_data, label="sin(x)")
        axes.set_xlabel("x")
        axes.set_ylabel("y")
        axes.legend()

        # Draw the canvas
        canvas.draw()
        qtbot.wait(100)

        # Verify plot was created
        assert len(axes.lines) == 1
        assert axes.get_xlabel() == "x"
        assert axes.get_ylabel() == "y"

    @pytest.mark.gui
    def test_matplotlib_canvas_interaction(self, matplotlib_canvas, qtbot):
        """Test Matplotlib canvas mouse interactions."""
        canvas, _figure, axes = matplotlib_canvas
        qtbot.addWidget(canvas)

        # Create test plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        axes.plot(x_data, y_data)
        canvas.draw()

        # Test mouse click on canvas
        center_pos = canvas.rect().center()
        qtbot.mouseClick(canvas, QtCore.Qt.MouseButton.LeftButton, pos=center_pos)
        qtbot.wait(50)

        # Canvas should handle the click without errors
        assert canvas.isVisible()

    @pytest.mark.gui
    def test_matplotlib_plot_updates(self, matplotlib_canvas, qtbot):
        """Test dynamic Matplotlib plot updates."""
        canvas, _figure, axes = matplotlib_canvas
        qtbot.addWidget(canvas)

        # Initial plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        (line,) = axes.plot(x_data, y_data)
        canvas.draw()
        qtbot.wait(50)

        initial_ydata = line.get_ydata()

        # Update plot data
        new_y_data = np.cos(x_data)
        line.set_ydata(new_y_data)
        canvas.draw()
        qtbot.wait(50)

        # Verify data was updated
        updated_ydata = line.get_ydata()
        assert not np.array_equal(initial_ydata, updated_ydata)


class TestPlotCustomization:
    """Test suite for plot customization and styling."""

    @pytest.mark.gui
    def test_plot_axis_labels(self, gui_plot_widget, qtbot):
        """Test plot axis label customization."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Set axis labels
        plot_widget.setLabel("left", "Y Axis")
        plot_widget.setLabel("bottom", "X Axis")

        # Create test plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        plot_widget.plot(x_data, y_data)

        qtbot.wait(50)

        # Verify labels were set
        left_label = plot_widget.getAxis("left").label
        bottom_label = plot_widget.getAxis("bottom").label

        assert "Y Axis" in left_label.toPlainText()
        assert "X Axis" in bottom_label.toPlainText()

    @pytest.mark.gui
    def test_plot_grid_functionality(self, gui_plot_widget, qtbot):
        """Test plot grid on/off functionality."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Enable grid
        plot_widget.showGrid(x=True, y=True)

        # Create test plot
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        plot_widget.plot(x_data, y_data)

        qtbot.wait(50)

        # Grid should be visible (tested implicitly through no errors)
        assert plot_widget.isVisible()

    @pytest.mark.gui
    def test_plot_color_schemes(self, gui_plot_widget, qtbot):
        """Test different plot color schemes."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Test different pen colors
        colors = ["b", "r", "g", "k", "y"]
        x_data = np.linspace(0, 10, 100)

        for i, color in enumerate(colors):
            y_data = np.sin(x_data + i * 0.5)
            plot_widget.plot(x_data, y_data, pen=color)

        qtbot.wait(100)

        # Verify all plots were added
        assert len(plot_widget.listDataItems()) == len(colors)


class TestPlotPerformance:
    """Test suite for plot performance and responsiveness."""

    @pytest.mark.gui
    @pytest.mark.slow
    def test_large_dataset_plotting(
        self, gui_plot_widget, qtbot, gui_performance_monitor
    ):
        """Test plotting performance with large datasets."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Create large dataset
        n_points = 10000
        x_data = np.linspace(0, 100, n_points)
        y_data = np.sin(x_data) + 0.1 * np.random.normal(size=n_points)

        # Time the plotting operation
        gui_performance_monitor.start_timing()
        plot_widget.plot(x_data, y_data)
        qtbot.wait(100)
        elapsed_time = gui_performance_monitor.end_timing("Large dataset plot")

        # Plot should complete within reasonable time (5 seconds)
        assert elapsed_time < 5.0

        # Verify data was plotted
        assert len(plot_widget.listDataItems()) == 1

    @pytest.mark.gui
    def test_rapid_plot_updates(self, gui_plot_widget, qtbot, gui_performance_monitor):
        """Test rapid plot data updates."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        plot_item = plot_widget.plot(x_data, y_data)

        # Time rapid updates
        gui_performance_monitor.start_timing()

        n_updates = 10
        for i in range(n_updates):
            new_y_data = np.sin(x_data + i * 0.1)
            plot_item.setData(x_data, new_y_data)
            qtbot.wait(10)  # Small delay between updates

        elapsed_time = gui_performance_monitor.end_timing("Rapid plot updates")

        # Updates should be responsive (< 2 seconds for 10 updates)
        assert elapsed_time < 2.0


class TestPlotErrorHandling:
    """Test suite for plot error handling and edge cases."""

    @pytest.mark.gui
    def test_empty_data_plotting(self, gui_plot_widget, qtbot):
        """Test plotting with empty data arrays."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Test empty arrays
        try:
            plot_widget.plot([], [])
            qtbot.wait(50)
            # Should not crash
            assert plot_widget.isVisible()
        except Exception as e:
            # Empty data should be handled gracefully
            assert "empty" in str(e).lower() or "size" in str(e).lower()

    @pytest.mark.gui
    def test_nan_data_handling(self, gui_plot_widget, qtbot):
        """Test plotting with NaN values in data."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Create data with NaN values
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(x_data)
        y_data[50:60] = np.nan  # Insert NaN values

        # Should handle NaN gracefully
        plot_item = plot_widget.plot(x_data, y_data)
        qtbot.wait(50)

        assert plot_item is not None
        assert len(plot_widget.listDataItems()) == 1

    @pytest.mark.gui
    def test_mismatched_data_arrays(self, gui_plot_widget, qtbot):
        """Test plotting with mismatched array sizes."""
        plot_widget = gui_plot_widget
        qtbot.addWidget(plot_widget)

        # Create mismatched arrays
        x_data = np.linspace(0, 10, 100)
        y_data = np.sin(np.linspace(0, 10, 50))  # Different size

        # Should handle size mismatch
        try:
            plot_widget.plot(x_data, y_data)
            qtbot.wait(50)
            # May succeed with automatic handling or fail gracefully
        except (ValueError, IndexError, Exception) as e:
            # Expected for size mismatch - PyQtGraph may raise various exception types
            assert any(
                keyword in str(e).lower()
                for keyword in ["size", "length", "shape", "same"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
