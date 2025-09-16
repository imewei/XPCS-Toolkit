"""
Robust Fitting GUI Integration for XPCS Toolkit

This module integrates all robust fitting components into the main XPCS viewer,
providing a comprehensive interface for advanced G2 analysis with robust fitting,
diagnostic visualization, and uncertainty quantification.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QGroupBox, QPushButton, QProgressBar, QLabel, QFrame,
    QMessageBox, QFileDialog, QApplication
)
from PySide6.QtGui import QFont, QColor

from .robust_fitting_controls import RobustFittingControlPanel
from .diagnostic_widgets import DiagnosticDashboard, RealTimeDiagnosticWidget
from .interactive_parameter_widgets import ParameterAnalysisWidget, ConfidenceIntervalWidget
from .enhanced_plotting import EnhancedG2PlotWidget, G2PlotControlWidget
from ..helper.fitting import RobustOptimizerWithDiagnostics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RobustFittingWorker(QThread):
    """
    Worker thread for performing robust fitting operations.

    Signals:
    --------
    progress_updated : Signal(int, str)
        Progress percentage and status message
    diagnostics_updated : Signal(dict)
        Real-time diagnostic data
    fitting_completed : Signal(dict)
        Final fitting results
    fitting_failed : Signal(str)
        Error message if fitting fails
    """

    progress_updated = Signal(int, str)
    diagnostics_updated = Signal(dict)
    fitting_completed = Signal(dict)
    fitting_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.fitting_params = {}
        self.data = {}
        self.robust_optimizer = None
        self.is_cancelled = False

    def setup_fitting(self, data, fitting_params):
        """Setup fitting parameters and data."""
        self.data = data
        self.fitting_params = fitting_params
        self.is_cancelled = False

        # Initialize robust optimizer with diagnostics
        self.robust_optimizer = RobustOptimizerWithDiagnostics(
            max_iterations=fitting_params.get('max_iterations', 10000),
            tolerance_factor=fitting_params.get('tolerance_factor', 1.0),
            enable_caching=fitting_params.get('enable_caching', True),
            performance_tracking=fitting_params.get('enable_performance_tracking', True)
        )

    def run(self):
        """Execute the fitting process."""
        try:
            self.progress_updated.emit(10, "Preparing data...")

            # Extract data
            x_data = self.data['x_data']
            y_data = self.data['y_data']
            y_err = self.data.get('y_err', None)
            bounds = self.data.get('bounds', None)
            p0 = self.data.get('p0', None)

            self.progress_updated.emit(20, "Initializing optimizer...")

            if self.is_cancelled:
                return

            # Define fitting function based on parameters
            fitting_function = self.data.get('fitting_function')
            if fitting_function is None:
                self.fitting_failed.emit("No fitting function provided")
                return

            self.progress_updated.emit(30, "Starting optimization...")

            # Perform robust fitting with diagnostics
            result = self.robust_optimizer.optimize_with_full_diagnostics(
                fitting_function, x_data, y_data, p0=p0, bounds=bounds, sigma=y_err,
                progress_callback=self.progress_callback,
                diagnostics_callback=self.diagnostics_callback
            )

            if self.is_cancelled:
                return

            self.progress_updated.emit(90, "Finalizing results...")

            # Calculate additional metrics
            if result.get('success', False):
                fitted_values = fitting_function(x_data, *result['popt'])
                residuals = y_data - fitted_values

                # Calculate goodness of fit metrics
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Add to result
                result.update({
                    'fitted_values': fitted_values,
                    'residuals': residuals,
                    'r_squared': r_squared,
                    'rmse': np.sqrt(np.mean(residuals ** 2)),
                    'x_data': x_data,
                    'y_data': y_data
                })

                self.progress_updated.emit(100, "Fitting completed successfully")
                self.fitting_completed.emit(result)
            else:
                error_msg = result.get('message', 'Unknown fitting error')
                self.fitting_failed.emit(error_msg)

        except Exception as e:
            self.logger.error(f"Robust fitting failed: {e}")
            self.fitting_failed.emit(str(e))

    def progress_callback(self, iteration, total_iterations, current_cost):
        """Callback for progress updates during optimization."""
        if self.is_cancelled:
            return False

        progress = 30 + int(60 * iteration / total_iterations)
        self.progress_updated.emit(progress, f"Iteration {iteration}/{total_iterations}")
        return True

    def diagnostics_callback(self, diagnostics_data):
        """Callback for real-time diagnostics updates."""
        if self.is_cancelled:
            return

        self.diagnostics_updated.emit(diagnostics_data)

    def cancel_fitting(self):
        """Cancel the fitting process."""
        self.is_cancelled = True
        if self.robust_optimizer:
            # Implementation depends on how the optimizer handles cancellation
            pass


class RobustFittingIntegrationWidget(QWidget):
    """
    Main integration widget combining all robust fitting components.

    This widget provides a comprehensive interface for robust G2 fitting with:
    - Advanced parameter controls
    - Real-time diagnostic visualization
    - Interactive parameter analysis
    - Enhanced plotting with uncertainty visualization
    """

    # Signals for communication with main XPCS viewer
    fitting_results_ready = Signal(dict)
    plot_update_requested = Signal(dict)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.fitting_worker = None
        self.current_results = {}
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the comprehensive user interface."""
        layout = QVBoxLayout(self)

        # Create main splitter for flexible layout
        main_splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(main_splitter)

        # Left panel: Controls and parameter analysis
        self.setup_control_panel(main_splitter)

        # Center panel: Enhanced plotting
        self.setup_plotting_panel(main_splitter)

        # Right panel: Diagnostics
        self.setup_diagnostics_panel(main_splitter)

        # Set splitter proportions
        main_splitter.setSizes([300, 500, 400])

        # Bottom panel: Status and progress
        self.setup_status_panel(layout)

    def setup_control_panel(self, parent):
        """Setup the control panel with robust fitting controls."""
        control_widget = QWidget()
        layout = QVBoxLayout(control_widget)

        # Robust fitting controls
        self.robust_controls = RobustFittingControlPanel()
        layout.addWidget(self.robust_controls)

        # Parameter analysis widget
        self.parameter_analysis = ParameterAnalysisWidget()
        layout.addWidget(self.parameter_analysis)

        parent.addWidget(control_widget)

    def setup_plotting_panel(self, parent):
        """Setup the enhanced plotting panel."""
        plot_widget = QWidget()
        layout = QVBoxLayout(plot_widget)

        # Plot controls
        self.plot_controls = G2PlotControlWidget()
        layout.addWidget(self.plot_controls)

        # Enhanced G2 plot
        self.enhanced_plot = EnhancedG2PlotWidget()
        self.enhanced_plot.setMinimumHeight(400)
        layout.addWidget(self.enhanced_plot)

        # Confidence interval widget
        self.confidence_widget = ConfidenceIntervalWidget()
        layout.addWidget(self.confidence_widget)

        parent.addWidget(plot_widget)

    def setup_diagnostics_panel(self, parent):
        """Setup the diagnostics panel."""
        diag_widget = QWidget()
        layout = QVBoxLayout(diag_widget)

        # Diagnostic dashboard
        self.diagnostic_dashboard = DiagnosticDashboard()
        layout.addWidget(self.diagnostic_dashboard)

        parent.addWidget(diag_widget)

    def setup_status_panel(self, layout):
        """Setup status and progress panel."""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)

        # Status label
        self.status_label = QLabel("Ready for robust fitting")
        status_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        # Action buttons
        self.start_fitting_button = QPushButton("Start Robust Fitting")
        self.start_fitting_button.clicked.connect(self.start_robust_fitting)
        status_layout.addWidget(self.start_fitting_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_fitting)
        self.cancel_button.setVisible(False)
        status_layout.addWidget(self.cancel_button)

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        status_layout.addWidget(self.export_button)

        layout.addWidget(status_frame)

    def setup_connections(self):
        """Setup signal connections between components."""
        # Robust controls connections
        self.robust_controls.fit_requested.connect(self.start_robust_fitting)
        self.robust_controls.validation_completed.connect(self.on_validation_completed)

        # Plot controls connections
        self.plot_controls.settings_changed.connect(self.on_plot_settings_changed)
        self.plot_controls.outlier_detection_changed.connect(self.on_outlier_detection_changed)

        # Enhanced plot connections
        self.enhanced_plot.outlier_selection_changed.connect(self.on_outlier_selection_changed)

        # Parameter analysis connections
        self.parameter_analysis.parameter_changed.connect(self.on_parameter_changed)
        self.parameter_analysis.sensitivity_analysis_requested.connect(self.run_sensitivity_analysis)

        # Confidence interval connections
        self.confidence_widget.confidence_level_changed.connect(self.on_confidence_level_changed)

    def start_robust_fitting(self, parameters=None):
        """Start the robust fitting process."""
        try:
            # Get parameters from controls if not provided
            if parameters is None:
                parameters = self.robust_controls.get_all_parameters()

            # Validate that we have data to fit
            if not hasattr(self, 'fitting_data') or not self.fitting_data:
                QMessageBox.warning(self, "No Data", "No G2 data available for fitting.")
                return

            # Validate parameters
            if not self.validate_fitting_setup(parameters):
                return

            # Prepare fitting worker
            self.fitting_worker = RobustFittingWorker()
            self.fitting_worker.setup_fitting(self.fitting_data, parameters)

            # Connect worker signals
            self.fitting_worker.progress_updated.connect(self.on_fitting_progress)
            self.fitting_worker.diagnostics_updated.connect(self.on_diagnostics_updated)
            self.fitting_worker.fitting_completed.connect(self.on_fitting_completed)
            self.fitting_worker.fitting_failed.connect(self.on_fitting_failed)

            # Update UI for fitting state
            self.start_fitting_button.setVisible(False)
            self.cancel_button.setVisible(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.export_button.setEnabled(False)

            # Start fitting
            self.fitting_worker.start()

            self.logger.info("Started robust fitting with parameters: %s", parameters)

        except Exception as e:
            self.logger.error(f"Failed to start robust fitting: {e}")
            QMessageBox.critical(self, "Fitting Error", f"Failed to start fitting:\n{e}")

    def cancel_fitting(self):
        """Cancel the current fitting process."""
        if self.fitting_worker and self.fitting_worker.isRunning():
            self.fitting_worker.cancel_fitting()
            self.fitting_worker.wait(5000)  # Wait up to 5 seconds

        self.reset_fitting_ui()
        self.status_label.setText("Fitting cancelled")

    def validate_fitting_setup(self, parameters):
        """Validate that fitting can proceed with current setup."""
        # Check data completeness
        required_keys = ['x_data', 'y_data', 'fitting_function']
        missing_keys = [key for key in required_keys if key not in self.fitting_data]

        if missing_keys:
            QMessageBox.warning(
                self, "Incomplete Data",
                f"Missing required data: {', '.join(missing_keys)}"
            )
            return False

        # Check data quality
        x_data = self.fitting_data['x_data']
        y_data = self.fitting_data['y_data']

        if len(x_data) != len(y_data):
            QMessageBox.warning(self, "Data Mismatch", "X and Y data arrays have different lengths.")
            return False

        if len(x_data) < 10:
            QMessageBox.warning(self, "Insufficient Data", "Need at least 10 data points for robust fitting.")
            return False

        # Check for NaN or infinite values
        if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
            QMessageBox.warning(self, "Invalid Data", "Data contains NaN or infinite values.")
            return False

        return True

    def set_fitting_data(self, x_data, y_data, y_err=None, bounds=None, p0=None, fitting_function=None):
        """Set the data and parameters for fitting."""
        self.fitting_data = {
            'x_data': np.array(x_data),
            'y_data': np.array(y_data),
            'y_err': np.array(y_err) if y_err is not None else None,
            'bounds': bounds,
            'p0': p0,
            'fitting_function': fitting_function
        }

        # Update parameter analysis with initial values
        if p0 is not None and bounds is not None:
            param_names = ['tau', 'bkg', 'cts']  # Adjust based on fitting function
            for i, (name, value) in enumerate(zip(param_names, p0)):
                if i < len(bounds[0]) and i < len(bounds[1]):
                    param_bounds = (bounds[0][i], bounds[1][i])
                    self.parameter_analysis.add_parameter_control(name, value, param_bounds)

        self.logger.info(f"Set fitting data: {len(x_data)} points")

    def on_fitting_progress(self, progress, message):
        """Handle fitting progress updates."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(message)

    def on_diagnostics_updated(self, diagnostics_data):
        """Handle real-time diagnostics updates."""
        try:
            # Update diagnostic dashboard
            self.diagnostic_dashboard.update_diagnostics(
                diagnostics_data.get('fitted_values', []),
                diagnostics_data.get('residuals', []),
                diagnostics_data.get('parameters', {}),
                diagnostics_data.get('metrics', {}),
                diagnostics_data.get('iteration', 0)
            )

            # Update parameter analysis if convergence data available
            if 'parameters' in diagnostics_data:
                for param_name, value in diagnostics_data['parameters'].items():
                    self.parameter_analysis.set_parameter_value(param_name, value)

        except Exception as e:
            self.logger.warning(f"Error updating diagnostics: {e}")

    def on_fitting_completed(self, results):
        """Handle successful fitting completion."""
        self.current_results = results
        self.reset_fitting_ui()

        # Update status
        r_squared = results.get('r_squared', 0)
        self.status_label.setText(f"Fitting completed successfully (R² = {r_squared:.4f})")

        # Update enhanced plot
        self.update_enhanced_plot(results)

        # Update diagnostic dashboard with final results
        if 'residuals' in results and 'fitted_values' in results:
            self.diagnostic_dashboard.update_diagnostics(
                results['fitted_values'],
                results['residuals'],
                dict(zip(['tau', 'bkg', 'cts'], results.get('popt', []))),
                {
                    'r_squared': results.get('r_squared', 0),
                    'rmse': results.get('rmse', 0),
                    'chi_squared': results.get('chi_squared', 0)
                }
            )

        # Enable export
        self.export_button.setEnabled(True)

        # Emit signal for main viewer
        self.fitting_results_ready.emit(results)

        self.logger.info("Robust fitting completed successfully")

    def on_fitting_failed(self, error_message):
        """Handle fitting failure."""
        self.reset_fitting_ui()
        self.status_label.setText(f"Fitting failed: {error_message}")

        QMessageBox.critical(self, "Fitting Failed", f"Robust fitting failed:\n{error_message}")

        self.logger.error(f"Robust fitting failed: {error_message}")

    def reset_fitting_ui(self):
        """Reset UI to ready state."""
        self.start_fitting_button.setVisible(True)
        self.cancel_button.setVisible(False)
        self.progress_bar.setVisible(False)

    def update_enhanced_plot(self, results):
        """Update the enhanced plot with fitting results."""
        try:
            if 'x_data' not in results or 'y_data' not in results:
                return

            # Prepare fit curve data
            x_data = results['x_data']
            y_data = results['y_data']
            fitted_values = results.get('fitted_values', [])

            # Create fit result dict
            fit_result = {
                'x_fit': x_data,
                'y_fit': fitted_values
            } if len(fitted_values) > 0 else None

            # Prepare uncertainty data if available
            uncertainty_data = {}
            if 'confidence_intervals' in results:
                ci = results['confidence_intervals']
                uncertainty_data['confidence_lower'] = ci.get('lower', [])
                uncertainty_data['confidence_upper'] = ci.get('upper', [])

            # Detect outliers
            outlier_mask = None
            if 'residuals' in results:
                residuals = results['residuals']
                z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
                outlier_mask = z_scores > 2.5

            # Update plot
            self.enhanced_plot.plot_g2_with_uncertainty(
                x_data, y_data,
                y_err=self.fitting_data.get('y_err'),
                fit_result=fit_result,
                uncertainty_data=uncertainty_data,
                outlier_mask=outlier_mask,
                label='G2 Data'
            )

        except Exception as e:
            self.logger.error(f"Error updating enhanced plot: {e}")

    def on_validation_completed(self, success, message):
        """Handle parameter validation results."""
        if success:
            self.status_label.setText("Parameters validated successfully")
        else:
            self.status_label.setText(f"Validation warning: {message}")

    def on_plot_settings_changed(self, settings):
        """Handle plot settings changes."""
        uncertainty_settings = settings.get('uncertainty', {})
        self.enhanced_plot.set_uncertainty_display_options(**uncertainty_settings)

    def on_outlier_detection_changed(self, method, threshold):
        """Handle outlier detection parameter changes."""
        if hasattr(self.enhanced_plot, 'current_data') and self.enhanced_plot.current_data:
            self.enhanced_plot.update_outlier_detection(threshold, method)

    def on_outlier_selection_changed(self, outlier_indices):
        """Handle outlier selection changes."""
        if outlier_indices:
            self.status_label.setText(f"Detected {len(outlier_indices)} outliers")
        else:
            self.status_label.setText("No outliers detected")

    def on_parameter_changed(self, param_name, value):
        """Handle parameter changes from interactive analysis."""
        # Update live plot if real-time updates are enabled
        pass

    def run_sensitivity_analysis(self, analysis_params):
        """Run parameter sensitivity analysis."""
        # Implementation would depend on the specific sensitivity analysis method
        self.logger.info(f"Running sensitivity analysis: {analysis_params}")

    def on_confidence_level_changed(self, confidence_level):
        """Handle confidence level changes."""
        self.logger.info(f"Confidence level changed to: {confidence_level}")

    def export_results(self):
        """Export fitting results and diagnostics."""
        if not self.current_results:
            QMessageBox.warning(self, "No Results", "No fitting results to export.")
            return

        try:
            # Get export file path
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export Robust Fitting Results",
                "robust_fitting_results.npz",
                "NumPy Archive (*.npz);;JSON (*.json);;CSV (*.csv)"
            )

            if not file_path:
                return

            # Prepare export data
            export_data = {
                'fitting_results': self.current_results,
                'fitting_parameters': self.robust_controls.get_all_parameters(),
                'plot_data': self.enhanced_plot.export_plot_data(),
                'diagnostic_data': {}  # Would include diagnostic dashboard data
            }

            # Export based on file extension
            if file_path.endswith('.npz'):
                np.savez_compressed(file_path, **export_data)
            elif file_path.endswith('.json'):
                import json
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2, default=str)
            elif file_path.endswith('.csv'):
                # Export main results to CSV
                import pandas as pd
                results_df = pd.DataFrame({
                    'x_data': self.current_results['x_data'],
                    'y_data': self.current_results['y_data'],
                    'fitted_values': self.current_results.get('fitted_values', []),
                    'residuals': self.current_results.get('residuals', [])
                })
                results_df.to_csv(file_path, index=False)

            self.status_label.setText(f"Results exported to {file_path}")
            self.logger.info(f"Exported results to: {file_path}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export results:\n{e}")

    def get_current_results(self):
        """Get current fitting results."""
        return self.current_results

    def clear_results(self):
        """Clear current results and reset interface."""
        self.current_results = {}
        self.enhanced_plot.clear()
        self.diagnostic_dashboard.clear_diagnostics()
        self.parameter_analysis.get_current_parameters().clear()
        self.export_button.setEnabled(False)
        self.status_label.setText("Ready for robust fitting")