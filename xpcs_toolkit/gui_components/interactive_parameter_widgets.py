"""
Interactive Parameter Analysis Widgets

This module provides interactive tools for exploring parameter sensitivity,
confidence intervals, and model comparison in G2 fitting analysis.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QSlider, QDoubleSpinBox, QSpinBox, QCheckBox,
    QPushButton, QComboBox, QTextEdit, QProgressBar, QFrame, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QFont, QPalette, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import stats

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ParameterAnalysisWidget(QWidget):
    """
    Interactive parameter analysis widget for exploring sensitivity and correlations.

    Features:
    - Parameter sensitivity analysis
    - Real-time parameter adjustment
    - Correlation matrix visualization
    - Confidence contour plots
    """

    # Signals
    parameter_changed = Signal(str, float)  # parameter_name, value
    sensitivity_analysis_requested = Signal(dict)
    correlation_analysis_updated = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.parameters = {}
        self.parameter_bounds = {}
        self.sensitivity_data = {}
        self.correlation_matrix = None

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Parameter adjustment tab
        self.setup_parameter_adjustment_tab()

        # Sensitivity analysis tab
        self.setup_sensitivity_analysis_tab()

        # Correlation analysis tab
        self.setup_correlation_analysis_tab()

    def setup_parameter_adjustment_tab(self):
        """Setup the parameter adjustment tab."""
        param_widget = QWidget()
        layout = QVBoxLayout(param_widget)

        # Parameter controls section
        controls_group = QGroupBox("Parameter Controls")
        controls_layout = QVBoxLayout(controls_group)

        # Scrollable area for parameter sliders
        self.parameter_controls_layout = QVBoxLayout()
        controls_layout.addLayout(self.parameter_controls_layout)

        # Real-time update checkbox
        self.realtime_update = QCheckBox("Real-time Update")
        self.realtime_update.setChecked(True)
        controls_layout.addWidget(self.realtime_update)

        layout.addWidget(controls_group)

        # Live plot section
        plot_group = QGroupBox("Live Parameter Effects")
        plot_layout = QVBoxLayout(plot_group)

        self.parameter_effect_plot = PlotWidget(title="Parameter Effect on Fit")
        self.parameter_effect_plot.setLabel('left', 'G2')
        self.parameter_effect_plot.setLabel('bottom', 'Time (s)')
        self.parameter_effect_plot.showGrid(x=True, y=True, alpha=0.3)
        self.parameter_effect_plot.setBackground('w')
        plot_layout.addWidget(self.parameter_effect_plot)

        layout.addWidget(plot_group)

        self.tab_widget.addTab(param_widget, "Parameter Adjustment")

    def setup_sensitivity_analysis_tab(self):
        """Setup the sensitivity analysis tab."""
        sens_widget = QWidget()
        layout = QVBoxLayout(sens_widget)

        # Controls section
        controls_frame = QFrame()
        controls_layout = QHBoxLayout(controls_frame)

        # Sensitivity method selection
        controls_layout.addWidget(QLabel("Method:"))
        self.sensitivity_method = QComboBox()
        self.sensitivity_method.addItems([
            "Local Sensitivity",
            "Global Sensitivity (Sobol)",
            "Morris Method",
            "Finite Difference"
        ])
        controls_layout.addWidget(self.sensitivity_method)

        # Number of samples
        controls_layout.addWidget(QLabel("Samples:"))
        self.sensitivity_samples = QSpinBox()
        self.sensitivity_samples.setRange(100, 10000)
        self.sensitivity_samples.setValue(1000)
        controls_layout.addWidget(self.sensitivity_samples)

        # Run analysis button
        self.run_sensitivity_button = QPushButton("Run Analysis")
        self.run_sensitivity_button.clicked.connect(self.run_sensitivity_analysis)
        controls_layout.addWidget(self.run_sensitivity_button)

        controls_layout.addStretch()
        layout.addWidget(controls_frame)

        # Results section
        results_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(results_splitter)

        # Sensitivity plot
        self.sensitivity_plot = PlotWidget(title="Parameter Sensitivity")
        self.sensitivity_plot.setLabel('left', 'Sensitivity Index')
        self.sensitivity_plot.setLabel('bottom', 'Parameters')
        self.sensitivity_plot.setBackground('w')
        results_splitter.addWidget(self.sensitivity_plot)

        # Sensitivity table
        self.sensitivity_table = QTableWidget()
        self.sensitivity_table.setColumnCount(3)
        self.sensitivity_table.setHorizontalHeaderLabels(['Parameter', 'Sensitivity', 'Rank'])
        self.sensitivity_table.horizontalHeader().setStretchLastSection(True)
        results_splitter.addWidget(self.sensitivity_table)

        results_splitter.setSizes([300, 200])

        self.tab_widget.addTab(sens_widget, "Sensitivity Analysis")

    def setup_correlation_analysis_tab(self):
        """Setup the correlation analysis tab."""
        corr_widget = QWidget()
        layout = QVBoxLayout(corr_widget)

        # Correlation matrix plot
        self.correlation_plot = PlotWidget(title="Parameter Correlation Matrix")
        self.correlation_plot.setBackground('w')
        layout.addWidget(self.correlation_plot)

        # Correlation details
        details_frame = QFrame()
        details_layout = QHBoxLayout(details_frame)

        # High correlations table
        high_corr_group = QGroupBox("High Correlations (|r| > 0.7)")
        high_corr_layout = QVBoxLayout(high_corr_group)

        self.high_correlation_table = QTableWidget()
        self.high_correlation_table.setColumnCount(3)
        self.high_correlation_table.setHorizontalHeaderLabels(['Parameter 1', 'Parameter 2', 'Correlation'])
        self.high_correlation_table.horizontalHeader().setStretchLastSection(True)
        high_corr_layout.addWidget(self.high_correlation_table)

        details_layout.addWidget(high_corr_group)

        # Correlation statistics
        stats_group = QGroupBox("Correlation Statistics")
        stats_layout = QVBoxLayout(stats_group)

        self.max_correlation_label = QLabel("Max |correlation|: --")
        self.mean_correlation_label = QLabel("Mean |correlation|: --")
        self.condition_number_label = QLabel("Condition number: --")

        stats_layout.addWidget(self.max_correlation_label)
        stats_layout.addWidget(self.mean_correlation_label)
        stats_layout.addWidget(self.condition_number_label)

        details_layout.addWidget(stats_group)

        layout.addWidget(details_frame)

        self.tab_widget.addTab(corr_widget, "Correlation Analysis")

    def setup_connections(self):
        """Setup signal connections."""
        self.realtime_update.toggled.connect(self.toggle_realtime_updates)

    def add_parameter_control(self, param_name, value, bounds, step=None):
        """Add a parameter control slider and spinbox."""
        if param_name in self.parameters:
            return  # Already exists

        # Create control group
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)

        # Parameter label
        label = QLabel(f"{param_name}:")
        label.setMinimumWidth(80)
        control_layout.addWidget(label)

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)  # Will be scaled to actual bounds
        slider.setValue(self.value_to_slider(value, bounds))
        control_layout.addWidget(slider)

        # SpinBox
        spinbox = QDoubleSpinBox()
        spinbox.setRange(bounds[0], bounds[1])
        spinbox.setValue(value)
        if step is not None:
            spinbox.setSingleStep(step)
        else:
            spinbox.setSingleStep((bounds[1] - bounds[0]) / 100)
        spinbox.setDecimals(6)
        control_layout.addWidget(spinbox)

        # Connect signals
        slider.valueChanged.connect(
            lambda v: self.slider_changed(param_name, v, bounds, spinbox)
        )
        spinbox.valueChanged.connect(
            lambda v: self.spinbox_changed(param_name, v, bounds, slider)
        )

        # Store references
        self.parameters[param_name] = {
            'value': value,
            'bounds': bounds,
            'slider': slider,
            'spinbox': spinbox,
            'frame': control_frame
        }
        self.parameter_bounds[param_name] = bounds

        # Add to layout
        self.parameter_controls_layout.addWidget(control_frame)

    def remove_parameter_control(self, param_name):
        """Remove a parameter control."""
        if param_name in self.parameters:
            frame = self.parameters[param_name]['frame']
            self.parameter_controls_layout.removeWidget(frame)
            frame.deleteLater()
            del self.parameters[param_name]
            del self.parameter_bounds[param_name]

    def value_to_slider(self, value, bounds):
        """Convert parameter value to slider position."""
        return int(1000 * (value - bounds[0]) / (bounds[1] - bounds[0]))

    def slider_to_value(self, slider_pos, bounds):
        """Convert slider position to parameter value."""
        return bounds[0] + (bounds[1] - bounds[0]) * slider_pos / 1000

    def slider_changed(self, param_name, slider_pos, bounds, spinbox):
        """Handle slider value change."""
        value = self.slider_to_value(slider_pos, bounds)
        spinbox.blockSignals(True)
        spinbox.setValue(value)
        spinbox.blockSignals(False)

        self.parameters[param_name]['value'] = value
        self.parameter_changed.emit(param_name, value)

        if self.realtime_update.isChecked():
            self.update_parameter_effect(param_name, value)

    def spinbox_changed(self, param_name, value, bounds, slider):
        """Handle spinbox value change."""
        slider_pos = self.value_to_slider(value, bounds)
        slider.blockSignals(True)
        slider.setValue(slider_pos)
        slider.blockSignals(False)

        self.parameters[param_name]['value'] = value
        self.parameter_changed.emit(param_name, value)

        if self.realtime_update.isChecked():
            self.update_parameter_effect(param_name, value)

    def toggle_realtime_updates(self, enabled):
        """Toggle real-time parameter updates."""
        if enabled:
            self.logger.info("Real-time parameter updates enabled")
        else:
            self.logger.info("Real-time parameter updates disabled")

    def update_parameter_effect(self, param_name, value):
        """Update the parameter effect plot."""
        # This would be connected to the fitting engine to show
        # how changing a parameter affects the fit curve
        pass

    def run_sensitivity_analysis(self):
        """Run parameter sensitivity analysis."""
        method = self.sensitivity_method.currentText()
        samples = self.sensitivity_samples.value()

        analysis_params = {
            'method': method,
            'samples': samples,
            'parameters': self.parameter_bounds
        }

        self.sensitivity_analysis_requested.emit(analysis_params)

    def update_sensitivity_results(self, sensitivity_data):
        """Update sensitivity analysis results."""
        self.sensitivity_data = sensitivity_data

        # Update sensitivity plot
        self.sensitivity_plot.clear()

        if 'parameter_names' in sensitivity_data and 'sensitivity_indices' in sensitivity_data:
            param_names = sensitivity_data['parameter_names']
            indices = sensitivity_data['sensitivity_indices']

            # Create bar plot
            x_pos = np.arange(len(param_names))
            bargraph = pg.BarGraphItem(x=x_pos, height=indices, width=0.6, brush='b')
            self.sensitivity_plot.addItem(bargraph)

            # Set x-axis labels
            ax = self.sensitivity_plot.getPlotItem().getAxis('bottom')
            ax.setTicks([[(i, name) for i, name in enumerate(param_names)]])

        # Update sensitivity table
        self.update_sensitivity_table(sensitivity_data)

    def update_sensitivity_table(self, sensitivity_data):
        """Update the sensitivity table."""
        if 'parameter_names' not in sensitivity_data or 'sensitivity_indices' not in sensitivity_data:
            return

        param_names = sensitivity_data['parameter_names']
        indices = sensitivity_data['sensitivity_indices']

        # Sort by sensitivity (descending)
        sorted_indices = np.argsort(indices)[::-1]

        self.sensitivity_table.setRowCount(len(param_names))

        for i, idx in enumerate(sorted_indices):
            param_item = QTableWidgetItem(param_names[idx])
            sens_item = QTableWidgetItem(f"{indices[idx]:.4f}")
            rank_item = QTableWidgetItem(str(i + 1))

            self.sensitivity_table.setItem(i, 0, param_item)
            self.sensitivity_table.setItem(i, 1, sens_item)
            self.sensitivity_table.setItem(i, 2, rank_item)

    def update_correlation_matrix(self, correlation_matrix, parameter_names):
        """Update the correlation matrix visualization."""
        self.correlation_matrix = correlation_matrix

        # Create correlation heatmap
        self.correlation_plot.clear()

        # Create image item for heatmap
        img = pg.ImageItem(correlation_matrix)
        self.correlation_plot.addItem(img)

        # Set up color map
        colormap = pg.colormap.getFromMatplotlib('RdBu_r')
        img.setColorMap(colormap)
        img.setLevels([-1, 1])

        # Set axis labels
        ax = self.correlation_plot.getPlotItem().getAxis('bottom')
        ax.setTicks([[(i, name) for i, name in enumerate(parameter_names)]])
        ay = self.correlation_plot.getPlotItem().getAxis('left')
        ay.setTicks([[(i, name) for i, name in enumerate(parameter_names)]])

        # Update high correlations table
        self.update_high_correlations_table(correlation_matrix, parameter_names)

        # Update correlation statistics
        self.update_correlation_statistics(correlation_matrix)

    def update_high_correlations_table(self, corr_matrix, param_names):
        """Update the high correlations table."""
        high_correlations = []

        n = len(param_names)
        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.7:
                    high_correlations.append((param_names[i], param_names[j], corr))

        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        self.high_correlation_table.setRowCount(len(high_correlations))

        for i, (param1, param2, corr) in enumerate(high_correlations):
            param1_item = QTableWidgetItem(param1)
            param2_item = QTableWidgetItem(param2)
            corr_item = QTableWidgetItem(f"{corr:.4f}")

            # Color code based on correlation strength
            if abs(corr) > 0.9:
                color = QColor(255, 0, 0)  # Red for very high
            elif abs(corr) > 0.8:
                color = QColor(255, 165, 0)  # Orange for high
            else:
                color = QColor(255, 255, 0)  # Yellow for moderate

            corr_item.setBackground(color)

            self.high_correlation_table.setItem(i, 0, param1_item)
            self.high_correlation_table.setItem(i, 1, param2_item)
            self.high_correlation_table.setItem(i, 2, corr_item)

    def update_correlation_statistics(self, corr_matrix):
        """Update correlation statistics."""
        # Get upper triangle (excluding diagonal)
        n = corr_matrix.shape[0]
        upper_triangle = []
        for i in range(n):
            for j in range(i + 1, n):
                upper_triangle.append(abs(corr_matrix[i, j]))

        if upper_triangle:
            max_corr = max(upper_triangle)
            mean_corr = np.mean(upper_triangle)
        else:
            max_corr = mean_corr = 0

        # Condition number
        try:
            cond_num = np.linalg.cond(corr_matrix)
        except:
            cond_num = float('inf')

        self.max_correlation_label.setText(f"Max |correlation|: {max_corr:.4f}")
        self.mean_correlation_label.setText(f"Mean |correlation|: {mean_corr:.4f}")
        self.condition_number_label.setText(f"Condition number: {cond_num:.2e}")

        # Color code condition number
        if cond_num > 1e12:
            color = "red"
        elif cond_num > 1e6:
            color = "orange"
        else:
            color = "green"

        self.condition_number_label.setStyleSheet(f"QLabel {{ color: {color}; }}")

    def get_current_parameters(self):
        """Get current parameter values."""
        return {name: data['value'] for name, data in self.parameters.items()}

    def set_parameter_value(self, param_name, value):
        """Set a parameter value programmatically."""
        if param_name in self.parameters:
            param_data = self.parameters[param_name]
            bounds = param_data['bounds']

            # Update both slider and spinbox
            param_data['slider'].setValue(self.value_to_slider(value, bounds))
            param_data['spinbox'].setValue(value)
            param_data['value'] = value


class ConfidenceIntervalWidget(QWidget):
    """
    Widget for visualizing confidence intervals and uncertainty quantification.

    Features:
    - Confidence band visualization
    - Bootstrap distribution plots
    - Prediction interval calculation
    - Uncertainty propagation analysis
    """

    # Signals
    confidence_level_changed = Signal(float)
    bootstrap_completed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.confidence_data = {}
        self.bootstrap_results = {}

        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Configuration section
        config_group = QGroupBox("Confidence Interval Configuration")
        config_layout = QGridLayout(config_group)

        # Confidence level
        config_layout.addWidget(QLabel("Confidence Level:"), 0, 0)
        self.confidence_level_spin = QDoubleSpinBox()
        self.confidence_level_spin.setRange(0.50, 0.99)
        self.confidence_level_spin.setValue(0.95)
        self.confidence_level_spin.setSingleStep(0.01)
        self.confidence_level_spin.setDecimals(2)
        config_layout.addWidget(self.confidence_level_spin, 0, 1)

        # Interval type
        config_layout.addWidget(QLabel("Interval Type:"), 0, 2)
        self.interval_type = QComboBox()
        self.interval_type.addItems([
            "Confidence Bands",
            "Prediction Intervals",
            "Both"
        ])
        config_layout.addWidget(self.interval_type, 0, 3)

        # Bootstrap settings
        config_layout.addWidget(QLabel("Bootstrap Samples:"), 1, 0)
        self.bootstrap_samples_spin = QSpinBox()
        self.bootstrap_samples_spin.setRange(50, 10000)
        self.bootstrap_samples_spin.setValue(1000)
        config_layout.addWidget(self.bootstrap_samples_spin, 1, 1)

        # Calculate button
        self.calculate_button = QPushButton("Calculate Intervals")
        self.calculate_button.clicked.connect(self.calculate_confidence_intervals)
        config_layout.addWidget(self.calculate_button, 1, 2, 1, 2)

        layout.addWidget(config_group)

        # Visualization section
        viz_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(viz_splitter)

        # Main confidence plot
        self.confidence_plot = PlotWidget(title="Confidence Intervals")
        self.confidence_plot.setLabel('left', 'G2')
        self.confidence_plot.setLabel('bottom', 'Time (s)')
        self.confidence_plot.showGrid(x=True, y=True, alpha=0.3)
        self.confidence_plot.setBackground('w')
        viz_splitter.addWidget(self.confidence_plot)

        # Bootstrap distribution plot
        self.bootstrap_plot = PlotWidget(title="Bootstrap Distributions")
        self.bootstrap_plot.setLabel('left', 'Frequency')
        self.bootstrap_plot.setLabel('bottom', 'Parameter Value')
        self.bootstrap_plot.setBackground('w')
        viz_splitter.addWidget(self.bootstrap_plot)

        viz_splitter.setSizes([400, 300])

        # Statistics section
        stats_group = QGroupBox("Interval Statistics")
        stats_layout = QGridLayout(stats_group)

        self.coverage_label = QLabel("Coverage: --")
        self.width_label = QLabel("Mean Width: --")
        self.asymmetry_label = QLabel("Asymmetry: --")

        stats_layout.addWidget(self.coverage_label, 0, 0)
        stats_layout.addWidget(self.width_label, 0, 1)
        stats_layout.addWidget(self.asymmetry_label, 0, 2)

        layout.addWidget(stats_group)

    def setup_connections(self):
        """Setup signal connections."""
        self.confidence_level_spin.valueChanged.connect(
            lambda v: self.confidence_level_changed.emit(v)
        )

    def calculate_confidence_intervals(self):
        """Calculate confidence intervals."""
        confidence_level = self.confidence_level_spin.value()
        interval_type = self.interval_type.currentText()
        bootstrap_samples = self.bootstrap_samples_spin.value()

        # Emit signal to request calculation
        params = {
            'confidence_level': confidence_level,
            'interval_type': interval_type,
            'bootstrap_samples': bootstrap_samples
        }

        self.logger.info(f"Calculating confidence intervals: {params}")

    def update_confidence_visualization(self, fit_data, confidence_data):
        """Update the confidence interval visualization."""
        self.confidence_plot.clear()

        if 'x_data' not in fit_data or 'y_fit' not in fit_data:
            return

        x_data = fit_data['x_data']
        y_fit = fit_data['y_fit']
        y_data = fit_data.get('y_data', None)

        # Plot original data if available
        if y_data is not None:
            self.confidence_plot.plot(
                x_data, y_data,
                pen=None, symbol='o', symbolSize=4,
                symbolBrush=pg.mkBrush(100, 100, 100, 150),
                name='Data'
            )

        # Plot fit line
        self.confidence_plot.plot(
            x_data, y_fit,
            pen=pg.mkPen(color='blue', width=2),
            name='Fit'
        )

        # Plot confidence bands
        if 'confidence_lower' in confidence_data and 'confidence_upper' in confidence_data:
            conf_lower = confidence_data['confidence_lower']
            conf_upper = confidence_data['confidence_upper']

            # Create fill between curves
            self.confidence_plot.plot(
                x_data, conf_lower,
                pen=pg.mkPen(color='red', style=Qt.DashLine),
                name='Confidence Bounds'
            )
            self.confidence_plot.plot(
                x_data, conf_upper,
                pen=pg.mkPen(color='red', style=Qt.DashLine)
            )

            # Fill area
            fill = pg.FillBetweenItem(
                pg.PlotCurveItem(x_data, conf_lower),
                pg.PlotCurveItem(x_data, conf_upper),
                brush=pg.mkBrush(255, 0, 0, 50)
            )
            self.confidence_plot.addItem(fill)

        # Plot prediction intervals if available
        if 'prediction_lower' in confidence_data and 'prediction_upper' in confidence_data:
            pred_lower = confidence_data['prediction_lower']
            pred_upper = confidence_data['prediction_upper']

            self.confidence_plot.plot(
                x_data, pred_lower,
                pen=pg.mkPen(color='green', style=Qt.DotLine),
                name='Prediction Bounds'
            )
            self.confidence_plot.plot(
                x_data, pred_upper,
                pen=pg.mkPen(color='green', style=Qt.DotLine)
            )

    def update_bootstrap_visualization(self, bootstrap_results):
        """Update bootstrap distribution visualization."""
        self.bootstrap_plot.clear()
        self.bootstrap_results = bootstrap_results

        if 'parameter_distributions' not in bootstrap_results:
            return

        param_distributions = bootstrap_results['parameter_distributions']
        param_names = list(param_distributions.keys())

        # Plot histogram for each parameter
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (param_name, distribution) in enumerate(param_distributions.items()):
            color = colors[i % len(colors)]

            # Create histogram
            hist, bin_edges = np.histogram(distribution, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            self.bootstrap_plot.plot(
                bin_centers, hist,
                pen=pg.mkPen(color=color, width=2),
                brush=pg.mkBrush(color + (50,)),
                fillLevel=0,
                name=param_name
            )

        # Add legend
        self.bootstrap_plot.addLegend()

    def update_interval_statistics(self, confidence_data):
        """Update interval statistics display."""
        if 'coverage' in confidence_data:
            coverage = confidence_data['coverage']
            self.coverage_label.setText(f"Coverage: {coverage:.1%}")

        if 'mean_width' in confidence_data:
            mean_width = confidence_data['mean_width']
            self.width_label.setText(f"Mean Width: {mean_width:.4f}")

        if 'asymmetry' in confidence_data:
            asymmetry = confidence_data['asymmetry']
            self.asymmetry_label.setText(f"Asymmetry: {asymmetry:.4f}")