"""
Diagnostic Visualization Widgets for Robust Fitting

This module provides real-time diagnostic visualization components including
residual analysis, confidence intervals, bootstrap distributions, and
comprehensive fit quality assessment.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer, QThread, pyqtSignal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QGroupBox, QLabel, QFrame, QSplitter, QScrollArea, QTextEdit,
    QPushButton, QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox
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


class RealTimeDiagnosticWidget(QWidget):
    """
    Real-time diagnostic visualization widget that updates during fitting.

    Features:
    - Live residual plots
    - Parameter convergence tracking
    - Goodness of fit metrics
    - Outlier detection visualization
    - Performance monitoring
    """

    # Signals
    diagnostics_updated = Signal(dict)
    outliers_detected = Signal(list)
    convergence_achieved = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.fitting_data = {}
        self.convergence_history = []
        self.outlier_threshold = 2.5

        self.setup_ui()
        self.setup_plot_configurations()

    def setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)

        # Create splitter for flexible layout
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # Top section: Live plots
        self.setup_live_plots_section(splitter)

        # Bottom section: Metrics and status
        self.setup_metrics_section(splitter)

        # Set splitter proportions
        splitter.setSizes([400, 200])

    def setup_live_plots_section(self, parent):
        """Setup the live plotting section."""
        plot_widget = QWidget()
        layout = QGridLayout(plot_widget)

        # Residual plot
        self.residual_plot = PlotWidget(title="Residuals vs Fitted Values")
        self.residual_plot.setLabel('left', 'Residuals')
        self.residual_plot.setLabel('bottom', 'Fitted Values')
        self.residual_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.residual_plot, 0, 0)

        # Q-Q plot for normality testing
        self.qq_plot = PlotWidget(title="Q-Q Plot (Normality Test)")
        self.qq_plot.setLabel('left', 'Sample Quantiles')
        self.qq_plot.setLabel('bottom', 'Theoretical Quantiles')
        self.qq_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.qq_plot, 0, 1)

        # Parameter convergence plot
        self.convergence_plot = PlotWidget(title="Parameter Convergence")
        self.convergence_plot.setLabel('left', 'Parameter Value')
        self.convergence_plot.setLabel('bottom', 'Iteration')
        self.convergence_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.convergence_plot, 1, 0)

        # Fit quality metrics over time
        self.quality_plot = PlotWidget(title="Fit Quality Evolution")
        self.quality_plot.setLabel('left', 'R² / χ² / AIC')
        self.quality_plot.setLabel('bottom', 'Iteration')
        self.quality_plot.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.quality_plot, 1, 1)

        parent.addWidget(plot_widget)

    def setup_metrics_section(self, parent):
        """Setup the metrics and status section."""
        metrics_widget = QWidget()
        layout = QHBoxLayout(metrics_widget)

        # Goodness of fit metrics
        self.setup_goodness_metrics(layout)

        # Outlier detection panel
        self.setup_outlier_panel(layout)

        # Statistical tests
        self.setup_statistical_tests(layout)

        parent.addWidget(metrics_widget)

    def setup_goodness_metrics(self, layout):
        """Setup goodness of fit metrics display."""
        metrics_group = QGroupBox("Fit Quality Metrics")
        metrics_layout = QGridLayout(metrics_group)

        # R-squared
        self.r_squared_label = QLabel("R²:")
        self.r_squared_value = QLabel("--")
        self.r_squared_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.r_squared_label, 0, 0)
        metrics_layout.addWidget(self.r_squared_value, 0, 1)

        # Adjusted R-squared
        self.adj_r_squared_label = QLabel("Adj. R²:")
        self.adj_r_squared_value = QLabel("--")
        self.adj_r_squared_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.adj_r_squared_label, 1, 0)
        metrics_layout.addWidget(self.adj_r_squared_value, 1, 1)

        # Chi-squared
        self.chi_squared_label = QLabel("χ²:")
        self.chi_squared_value = QLabel("--")
        self.chi_squared_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.chi_squared_label, 2, 0)
        metrics_layout.addWidget(self.chi_squared_value, 2, 1)

        # AIC/BIC
        self.aic_label = QLabel("AIC:")
        self.aic_value = QLabel("--")
        self.aic_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.aic_label, 3, 0)
        metrics_layout.addWidget(self.aic_value, 3, 1)

        self.bic_label = QLabel("BIC:")
        self.bic_value = QLabel("--")
        self.bic_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.bic_label, 4, 0)
        metrics_layout.addWidget(self.bic_value, 4, 1)

        # RMSE
        self.rmse_label = QLabel("RMSE:")
        self.rmse_value = QLabel("--")
        self.rmse_value.setStyleSheet("QLabel { font-weight: bold; }")
        metrics_layout.addWidget(self.rmse_label, 5, 0)
        metrics_layout.addWidget(self.rmse_value, 5, 1)

        layout.addWidget(metrics_group)

    def setup_outlier_panel(self, layout):
        """Setup outlier detection panel."""
        outlier_group = QGroupBox("Outlier Analysis")
        outlier_layout = QVBoxLayout(outlier_group)

        # Outlier count and threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Threshold (σ):"))

        self.outlier_threshold_spin = QDoubleSpinBox()
        self.outlier_threshold_spin.setRange(1.0, 5.0)
        self.outlier_threshold_spin.setValue(2.5)
        self.outlier_threshold_spin.setSingleStep(0.1)
        self.outlier_threshold_spin.valueChanged.connect(self.update_outlier_threshold)
        threshold_layout.addWidget(self.outlier_threshold_spin)

        outlier_layout.addLayout(threshold_layout)

        # Outlier statistics
        self.outlier_count_label = QLabel("Outliers Detected: 0")
        self.outlier_count_label.setStyleSheet("QLabel { font-weight: bold; }")
        outlier_layout.addWidget(self.outlier_count_label)

        self.outlier_percentage_label = QLabel("Percentage: 0.0%")
        outlier_layout.addWidget(self.outlier_percentage_label)

        # Outlier indices (scrollable)
        self.outlier_text = QTextEdit()
        self.outlier_text.setMaximumHeight(60)
        self.outlier_text.setPlaceholderText("Outlier indices will appear here...")
        outlier_layout.addWidget(self.outlier_text)

        layout.addWidget(outlier_group)

    def setup_statistical_tests(self, layout):
        """Setup statistical tests panel."""
        tests_group = QGroupBox("Statistical Tests")
        tests_layout = QVBoxLayout(tests_group)

        # Shapiro-Wilk test for normality
        self.shapiro_label = QLabel("Shapiro-Wilk (Normality):")
        self.shapiro_value = QLabel("--")
        tests_layout.addWidget(self.shapiro_label)
        tests_layout.addWidget(self.shapiro_value)

        # Durbin-Watson test for autocorrelation
        self.durbin_watson_label = QLabel("Durbin-Watson (Autocorr.):")
        self.durbin_watson_value = QLabel("--")
        tests_layout.addWidget(self.durbin_watson_label)
        tests_layout.addWidget(self.durbin_watson_value)

        # Breusch-Pagan test for heteroscedasticity
        self.breusch_pagan_label = QLabel("Breusch-Pagan (Heterosc.):")
        self.breusch_pagan_value = QLabel("--")
        tests_layout.addWidget(self.breusch_pagan_label)
        tests_layout.addWidget(self.breusch_pagan_value)

        # Overall diagnostic status
        self.diagnostic_status = QLabel("Status: Ready")
        self.diagnostic_status.setStyleSheet("QLabel { font-weight: bold; color: blue; }")
        tests_layout.addWidget(self.diagnostic_status)

        layout.addWidget(tests_group)

    def setup_plot_configurations(self):
        """Configure plot appearance and behavior."""
        # Set plot backgrounds and styles
        plots = [self.residual_plot, self.qq_plot, self.convergence_plot, self.quality_plot]

        for plot in plots:
            plot.setBackground('w')
            plot.getPlotItem().getAxis('left').setPen(pg.mkPen(color='black', width=1))
            plot.getPlotItem().getAxis('bottom').setPen(pg.mkPen(color='black', width=1))

        # Initialize plot data containers
        self.residual_data = {'x': [], 'y': []}
        self.qq_data = {'theoretical': [], 'sample': []}
        self.convergence_data = {'iterations': [], 'parameters': {}}
        self.quality_data = {'iterations': [], 'r_squared': [], 'chi_squared': [], 'aic': []}

    def update_residual_plot(self, fitted_values, residuals, outlier_mask=None):
        """Update the residual plot."""
        self.residual_plot.clear()

        if outlier_mask is not None:
            # Plot normal points
            normal_points = ~outlier_mask
            if np.any(normal_points):
                self.residual_plot.plot(
                    fitted_values[normal_points],
                    residuals[normal_points],
                    pen=None,
                    symbol='o',
                    symbolSize=5,
                    symbolBrush=pg.mkBrush(0, 0, 255, 150),
                    name='Normal Points'
                )

            # Plot outlier points
            if np.any(outlier_mask):
                self.residual_plot.plot(
                    fitted_values[outlier_mask],
                    residuals[outlier_mask],
                    pen=None,
                    symbol='o',
                    symbolSize=7,
                    symbolBrush=pg.mkBrush(255, 0, 0, 180),
                    name='Outliers'
                )
        else:
            # Plot all points as normal
            self.residual_plot.plot(
                fitted_values,
                residuals,
                pen=None,
                symbol='o',
                symbolSize=5,
                symbolBrush=pg.mkBrush(0, 0, 255, 150)
            )

        # Add zero line
        x_range = [np.min(fitted_values), np.max(fitted_values)]
        self.residual_plot.plot(x_range, [0, 0], pen=pg.mkPen(color='red', style=Qt.DashLine))

    def update_qq_plot(self, residuals):
        """Update the Q-Q plot for normality testing."""
        self.qq_plot.clear()

        # Calculate quantiles
        sorted_residuals = np.sort(residuals)
        n = len(sorted_residuals)
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))

        # Plot Q-Q points
        self.qq_plot.plot(
            theoretical_quantiles,
            sorted_residuals,
            pen=None,
            symbol='o',
            symbolSize=4,
            symbolBrush=pg.mkBrush(0, 100, 0, 150)
        )

        # Add reference line
        min_val = min(np.min(theoretical_quantiles), np.min(sorted_residuals))
        max_val = max(np.max(theoretical_quantiles), np.max(sorted_residuals))
        self.qq_plot.plot([min_val, max_val], [min_val, max_val],
                         pen=pg.mkPen(color='red', style=Qt.DashLine))

    def update_convergence_plot(self, iteration, parameters):
        """Update parameter convergence plot."""
        if iteration == 0:
            self.convergence_plot.clear()
            self.convergence_data = {'iterations': [], 'parameters': {}}

        self.convergence_data['iterations'].append(iteration)

        # Colors for different parameters
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']

        for i, (param_name, value) in enumerate(parameters.items()):
            if param_name not in self.convergence_data['parameters']:
                self.convergence_data['parameters'][param_name] = []

            self.convergence_data['parameters'][param_name].append(value)

            # Update plot
            color = colors[i % len(colors)]
            self.convergence_plot.plot(
                self.convergence_data['iterations'],
                self.convergence_data['parameters'][param_name],
                pen=pg.mkPen(color=color, width=2),
                name=param_name
            )

    def update_quality_plot(self, iteration, metrics):
        """Update fit quality metrics plot."""
        if iteration == 0:
            self.quality_plot.clear()
            self.quality_data = {'iterations': [], 'r_squared': [], 'chi_squared': [], 'aic': []}

        self.quality_data['iterations'].append(iteration)
        self.quality_data['r_squared'].append(metrics.get('r_squared', 0))
        self.quality_data['chi_squared'].append(metrics.get('chi_squared', 0))
        self.quality_data['aic'].append(metrics.get('aic', 0))

        # Plot R-squared (scale 0-1)
        self.quality_plot.plot(
            self.quality_data['iterations'],
            self.quality_data['r_squared'],
            pen=pg.mkPen(color='blue', width=2),
            name='R²'
        )

    def update_outlier_threshold(self, threshold):
        """Update outlier detection threshold."""
        self.outlier_threshold = threshold
        # Recompute outliers if data is available
        if hasattr(self, 'current_residuals') and len(self.current_residuals) > 0:
            self.detect_outliers(self.current_residuals)

    def detect_outliers(self, residuals):
        """Detect outliers based on current threshold."""
        self.current_residuals = residuals

        # Calculate standardized residuals
        std_residuals = np.abs(residuals) / np.std(residuals)
        outlier_mask = std_residuals > self.outlier_threshold

        # Update outlier information
        num_outliers = np.sum(outlier_mask)
        total_points = len(residuals)
        percentage = (num_outliers / total_points) * 100 if total_points > 0 else 0

        self.outlier_count_label.setText(f"Outliers Detected: {num_outliers}")
        self.outlier_percentage_label.setText(f"Percentage: {percentage:.1f}%")

        # Update outlier indices text
        if num_outliers > 0:
            outlier_indices = np.where(outlier_mask)[0]
            indices_text = ', '.join(map(str, outlier_indices[:20]))  # Show first 20
            if num_outliers > 20:
                indices_text += f", ... ({num_outliers - 20} more)"
            self.outlier_text.setText(indices_text)
        else:
            self.outlier_text.setText("No outliers detected")

        # Emit signal
        self.outliers_detected.emit(outlier_indices.tolist() if num_outliers > 0 else [])

        return outlier_mask

    def update_goodness_metrics(self, metrics):
        """Update goodness of fit metrics display."""
        def format_metric(value, decimals=4):
            return f"{value:.{decimals}f}" if value is not None else "--"

        self.r_squared_value.setText(format_metric(metrics.get('r_squared')))
        self.adj_r_squared_value.setText(format_metric(metrics.get('adj_r_squared')))
        self.chi_squared_value.setText(format_metric(metrics.get('chi_squared')))
        self.aic_value.setText(format_metric(metrics.get('aic')))
        self.bic_value.setText(format_metric(metrics.get('bic')))
        self.rmse_value.setText(format_metric(metrics.get('rmse')))

        # Color-code values based on quality
        r_squared = metrics.get('r_squared', 0)
        if r_squared > 0.95:
            color = "green"
        elif r_squared > 0.85:
            color = "orange"
        else:
            color = "red"

        self.r_squared_value.setStyleSheet(f"QLabel {{ font-weight: bold; color: {color}; }}")

    def update_statistical_tests(self, residuals):
        """Update statistical test results."""
        try:
            # Shapiro-Wilk test for normality
            if len(residuals) >= 3:
                stat, p_value = stats.shapiro(residuals)
                status = "Normal" if p_value > 0.05 else "Non-normal"
                self.shapiro_value.setText(f"p={p_value:.4f} ({status})")
                self.shapiro_value.setStyleSheet(
                    f"QLabel {{ color: {'green' if p_value > 0.05 else 'red'}; }}"
                )

            # Durbin-Watson test for autocorrelation
            if len(residuals) >= 4:
                dw_stat = self.durbin_watson_statistic(residuals)
                interpretation = self.interpret_durbin_watson(dw_stat)
                self.durbin_watson_value.setText(f"DW={dw_stat:.3f} ({interpretation})")

            # Overall status
            if hasattr(self, 'current_metrics'):
                r_squared = self.current_metrics.get('r_squared', 0)
                if r_squared > 0.9:
                    status = "Excellent fit"
                    color = "green"
                elif r_squared > 0.8:
                    status = "Good fit"
                    color = "blue"
                elif r_squared > 0.6:
                    status = "Fair fit"
                    color = "orange"
                else:
                    status = "Poor fit"
                    color = "red"

                self.diagnostic_status.setText(f"Status: {status}")
                self.diagnostic_status.setStyleSheet(f"QLabel {{ font-weight: bold; color: {color}; }}")

        except Exception as e:
            self.logger.warning(f"Statistical tests update failed: {e}")

    def durbin_watson_statistic(self, residuals):
        """Calculate Durbin-Watson statistic."""
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)

    def interpret_durbin_watson(self, dw_stat):
        """Interpret Durbin-Watson statistic."""
        if dw_stat < 1.5:
            return "Positive autocorr."
        elif dw_stat > 2.5:
            return "Negative autocorr."
        else:
            return "No autocorr."

    def update_diagnostics(self, fitted_values, residuals, parameters, metrics, iteration=0):
        """Comprehensive diagnostic update."""
        try:
            # Store current metrics for other methods
            self.current_metrics = metrics

            # Detect outliers
            outlier_mask = self.detect_outliers(residuals)

            # Update all plots
            self.update_residual_plot(fitted_values, residuals, outlier_mask)
            self.update_qq_plot(residuals)
            self.update_convergence_plot(iteration, parameters)
            self.update_quality_plot(iteration, metrics)

            # Update metrics and tests
            self.update_goodness_metrics(metrics)
            self.update_statistical_tests(residuals)

            # Emit diagnostic update signal
            diagnostic_data = {
                'fitted_values': fitted_values,
                'residuals': residuals,
                'outlier_mask': outlier_mask,
                'parameters': parameters,
                'metrics': metrics,
                'iteration': iteration
            }
            self.diagnostics_updated.emit(diagnostic_data)

        except Exception as e:
            self.logger.error(f"Diagnostic update failed: {e}")

    def clear_diagnostics(self):
        """Clear all diagnostic displays."""
        # Clear plots
        self.residual_plot.clear()
        self.qq_plot.clear()
        self.convergence_plot.clear()
        self.quality_plot.clear()

        # Reset metrics
        for widget in [self.r_squared_value, self.adj_r_squared_value,
                      self.chi_squared_value, self.aic_value, self.bic_value, self.rmse_value]:
            widget.setText("--")
            widget.setStyleSheet("QLabel { font-weight: bold; }")

        # Reset outlier information
        self.outlier_count_label.setText("Outliers Detected: 0")
        self.outlier_percentage_label.setText("Percentage: 0.0%")
        self.outlier_text.clear()

        # Reset statistical tests
        self.shapiro_value.setText("--")
        self.durbin_watson_value.setText("--")
        self.breusch_pagan_value.setText("--")
        self.diagnostic_status.setText("Status: Ready")
        self.diagnostic_status.setStyleSheet("QLabel { font-weight: bold; color: blue; }")

        # Clear data containers
        self.residual_data = {'x': [], 'y': []}
        self.qq_data = {'theoretical': [], 'sample': []}
        self.convergence_data = {'iterations': [], 'parameters': {}}
        self.quality_data = {'iterations': [], 'r_squared': [], 'chi_squared': [], 'aic': []}


class DiagnosticDashboard(QWidget):
    """
    Comprehensive diagnostic dashboard combining multiple diagnostic widgets.

    Features:
    - Real-time diagnostic visualization
    - Parameter analysis tools
    - Model comparison interface
    - Export capabilities
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.setup_ui()

    def setup_ui(self):
        """Initialize the dashboard UI."""
        layout = QVBoxLayout(self)

        # Create tab widget for different diagnostic views
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Real-time diagnostics tab
        self.realtime_widget = RealTimeDiagnosticWidget()
        self.tab_widget.addTab(self.realtime_widget, "Real-time Diagnostics")

        # Model comparison tab (placeholder for now)
        self.comparison_widget = QWidget()
        self.tab_widget.addTab(self.comparison_widget, "Model Comparison")

        # Export controls
        self.setup_export_controls(layout)

    def setup_export_controls(self, layout):
        """Setup export controls."""
        export_frame = QFrame()
        export_frame.setFrameStyle(QFrame.StyledPanel)
        export_layout = QHBoxLayout(export_frame)

        export_button = QPushButton("Export Diagnostics")
        export_button.clicked.connect(self.export_diagnostics)
        export_layout.addWidget(export_button)

        export_layout.addStretch()
        layout.addWidget(export_frame)

    def export_diagnostics(self):
        """Export diagnostic results."""
        # Implementation for exporting diagnostic data and plots
        self.logger.info("Exporting diagnostic results...")

    def update_diagnostics(self, *args, **kwargs):
        """Update all diagnostic components."""
        self.realtime_widget.update_diagnostics(*args, **kwargs)

    def clear_diagnostics(self):
        """Clear all diagnostic displays."""
        self.realtime_widget.clear_diagnostics()