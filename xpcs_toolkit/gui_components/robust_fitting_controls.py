"""
Robust Fitting Control Panel for G2 Analysis

This module provides an intuitive control panel for configuring robust fitting
parameters, method selection, and diagnostic options for G2 correlation analysis.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QComboBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton,
    QLabel, QFrame, QTabWidget, QSlider, QProgressBar, QTextEdit
)
from PySide6.QtGui import QFont, QPalette, QColor

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RobustFittingControlPanel(QWidget):
    """
    Comprehensive control panel for robust fitting configuration.

    Features:
    - Optimization method selection with smart defaults
    - Advanced parameter controls with progressive disclosure
    - Real-time validation and feedback
    - Integration with diagnostic visualization
    - Bootstrap and confidence interval configuration
    """

    # Signals for communication with main GUI
    fitting_parameters_changed = Signal(dict)
    optimization_method_changed = Signal(str)
    diagnostic_mode_changed = Signal(bool)
    bootstrap_parameters_changed = Signal(dict)
    fit_requested = Signal(dict)
    validation_completed = Signal(bool, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.setup_ui()
        self.setup_connections()
        self.setup_validation_timer()
        self.load_default_parameters()

    def setup_ui(self):
        """Initialize the user interface components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(8, 8, 8, 8)

        # Main tab widget for organizing controls
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Basic Controls Tab
        self.setup_basic_controls_tab()

        # Advanced Controls Tab
        self.setup_advanced_controls_tab()

        # Diagnostics Tab
        self.setup_diagnostics_tab()

        # Bootstrap Tab
        self.setup_bootstrap_tab()

        # Status and action controls
        self.setup_status_controls()
        layout.addWidget(self.status_frame)

    def setup_basic_controls_tab(self):
        """Setup the basic fitting controls tab."""
        basic_widget = QWidget()
        layout = QVBoxLayout(basic_widget)

        # Optimization Method Selection
        method_group = QGroupBox("Optimization Method")
        method_layout = QGridLayout(method_group)

        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Auto (TRF → LM → DE)",
            "Trust Region Reflective (TRF)",
            "Levenberg-Marquardt (LM)",
            "Differential Evolution (DE)",
            "Robust (with outlier detection)",
            "Bayesian (MCMC)"
        ])
        self.method_combo.setCurrentText("Auto (TRF → LM → DE)")
        method_layout.addWidget(QLabel("Method:"), 0, 0)
        method_layout.addWidget(self.method_combo, 0, 1, 1, 2)

        # Method description
        self.method_description = QLabel()
        self.method_description.setWordWrap(True)
        self.method_description.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        method_layout.addWidget(self.method_description, 1, 0, 1, 3)

        layout.addWidget(method_group)

        # Tolerance Settings
        tolerance_group = QGroupBox("Convergence Tolerances")
        tolerance_layout = QGridLayout(tolerance_group)

        self.tolerance_factor = QDoubleSpinBox()
        self.tolerance_factor.setRange(0.1, 10.0)
        self.tolerance_factor.setValue(1.0)
        self.tolerance_factor.setSingleStep(0.1)
        self.tolerance_factor.setDecimals(2)
        tolerance_layout.addWidget(QLabel("Tolerance Factor:"), 0, 0)
        tolerance_layout.addWidget(self.tolerance_factor, 0, 1)

        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(100, 50000)
        self.max_iterations.setValue(10000)
        self.max_iterations.setSingleStep(1000)
        tolerance_layout.addWidget(QLabel("Max Iterations:"), 0, 2)
        tolerance_layout.addWidget(self.max_iterations, 0, 3)

        layout.addWidget(tolerance_group)

        # Robustness Options
        robust_group = QGroupBox("Robustness Options")
        robust_layout = QGridLayout(robust_group)

        self.enable_outlier_detection = QCheckBox("Enable Outlier Detection")
        self.enable_outlier_detection.setChecked(True)
        robust_layout.addWidget(self.enable_outlier_detection, 0, 0, 1, 2)

        self.outlier_threshold = QDoubleSpinBox()
        self.outlier_threshold.setRange(1.0, 5.0)
        self.outlier_threshold.setValue(2.5)
        self.outlier_threshold.setSingleStep(0.1)
        self.outlier_threshold.setDecimals(1)
        robust_layout.addWidget(QLabel("Outlier Threshold (σ):"), 1, 0)
        robust_layout.addWidget(self.outlier_threshold, 1, 1)

        self.enable_weight_adjustment = QCheckBox("Adaptive Weight Adjustment")
        self.enable_weight_adjustment.setChecked(True)
        robust_layout.addWidget(self.enable_weight_adjustment, 1, 2, 1, 2)

        layout.addWidget(robust_group)

        self.tab_widget.addTab(basic_widget, "Basic")

    def setup_advanced_controls_tab(self):
        """Setup the advanced controls tab."""
        advanced_widget = QWidget()
        layout = QVBoxLayout(advanced_widget)

        # Strategy Configuration
        strategy_group = QGroupBox("Strategy Configuration")
        strategy_layout = QGridLayout(strategy_group)

        self.enable_performance_tracking = QCheckBox("Performance Tracking")
        self.enable_performance_tracking.setChecked(True)
        strategy_layout.addWidget(self.enable_performance_tracking, 0, 0)

        self.enable_caching = QCheckBox("Enable Result Caching")
        self.enable_caching.setChecked(True)
        strategy_layout.addWidget(self.enable_caching, 0, 1)

        self.strategy_timeout = QSpinBox()
        self.strategy_timeout.setRange(10, 300)
        self.strategy_timeout.setValue(60)
        self.strategy_timeout.setSuffix(" seconds")
        strategy_layout.addWidget(QLabel("Strategy Timeout:"), 1, 0)
        strategy_layout.addWidget(self.strategy_timeout, 1, 1)

        layout.addWidget(strategy_group)

        # Parameter Estimation
        estimation_group = QGroupBox("Initial Parameter Estimation")
        estimation_layout = QGridLayout(estimation_group)

        self.parameter_estimation_method = QComboBox()
        self.parameter_estimation_method.addItems([
            "Intelligent Heuristics",
            "Linear Regression",
            "Grid Search",
            "Random Sampling",
            "User Provided"
        ])
        estimation_layout.addWidget(QLabel("Method:"), 0, 0)
        estimation_layout.addWidget(self.parameter_estimation_method, 0, 1)

        self.estimation_samples = QSpinBox()
        self.estimation_samples.setRange(10, 1000)
        self.estimation_samples.setValue(100)
        estimation_layout.addWidget(QLabel("Samples:"), 0, 2)
        estimation_layout.addWidget(self.estimation_samples, 0, 3)

        layout.addWidget(estimation_group)

        # Error Handling
        error_group = QGroupBox("Error Handling")
        error_layout = QGridLayout(error_group)

        self.fallback_on_failure = QCheckBox("Enable Fallback Methods")
        self.fallback_on_failure.setChecked(True)
        error_layout.addWidget(self.fallback_on_failure, 0, 0)

        self.retry_on_boundary = QCheckBox("Retry on Boundary Convergence")
        self.retry_on_boundary.setChecked(True)
        error_layout.addWidget(self.retry_on_boundary, 0, 1)

        self.max_retries = QSpinBox()
        self.max_retries.setRange(0, 10)
        self.max_retries.setValue(3)
        error_layout.addWidget(QLabel("Max Retries:"), 1, 0)
        error_layout.addWidget(self.max_retries, 1, 1)

        layout.addWidget(error_group)

        self.tab_widget.addTab(advanced_widget, "Advanced")

    def setup_diagnostics_tab(self):
        """Setup the diagnostics configuration tab."""
        diag_widget = QWidget()
        layout = QVBoxLayout(diag_widget)

        # Diagnostic Options
        diag_group = QGroupBox("Diagnostic Options")
        diag_layout = QGridLayout(diag_group)

        self.enable_diagnostics = QCheckBox("Enable Comprehensive Diagnostics")
        self.enable_diagnostics.setChecked(True)
        diag_layout.addWidget(self.enable_diagnostics, 0, 0, 1, 2)

        self.residual_analysis = QCheckBox("Residual Analysis")
        self.residual_analysis.setChecked(True)
        diag_layout.addWidget(self.residual_analysis, 1, 0)

        self.parameter_correlation = QCheckBox("Parameter Correlation")
        self.parameter_correlation.setChecked(True)
        diag_layout.addWidget(self.parameter_correlation, 1, 1)

        self.goodness_of_fit = QCheckBox("Goodness of Fit Tests")
        self.goodness_of_fit.setChecked(True)
        diag_layout.addWidget(self.goodness_of_fit, 2, 0)

        self.model_comparison = QCheckBox("Model Comparison")
        self.model_comparison.setChecked(False)
        diag_layout.addWidget(self.model_comparison, 2, 1)

        layout.addWidget(diag_group)

        # Visualization Options
        viz_group = QGroupBox("Visualization Options")
        viz_layout = QGridLayout(viz_group)

        self.show_confidence_bands = QCheckBox("Confidence Bands")
        self.show_confidence_bands.setChecked(True)
        viz_layout.addWidget(self.show_confidence_bands, 0, 0)

        self.show_prediction_intervals = QCheckBox("Prediction Intervals")
        self.show_prediction_intervals.setChecked(False)
        viz_layout.addWidget(self.show_prediction_intervals, 0, 1)

        self.highlight_outliers = QCheckBox("Highlight Outliers")
        self.highlight_outliers.setChecked(True)
        viz_layout.addWidget(self.highlight_outliers, 1, 0)

        self.show_residual_plots = QCheckBox("Residual Plots")
        self.show_residual_plots.setChecked(True)
        viz_layout.addWidget(self.show_residual_plots, 1, 1)

        self.confidence_level = QDoubleSpinBox()
        self.confidence_level.setRange(0.50, 0.99)
        self.confidence_level.setValue(0.95)
        self.confidence_level.setSingleStep(0.01)
        self.confidence_level.setDecimals(2)
        viz_layout.addWidget(QLabel("Confidence Level:"), 2, 0)
        viz_layout.addWidget(self.confidence_level, 2, 1)

        layout.addWidget(viz_group)

        self.tab_widget.addTab(diag_widget, "Diagnostics")

    def setup_bootstrap_tab(self):
        """Setup the bootstrap configuration tab."""
        bootstrap_widget = QWidget()
        layout = QVBoxLayout(bootstrap_widget)

        # Bootstrap Settings
        bootstrap_group = QGroupBox("Bootstrap Configuration")
        bootstrap_layout = QGridLayout(bootstrap_group)

        self.enable_bootstrap = QCheckBox("Enable Bootstrap Analysis")
        self.enable_bootstrap.setChecked(False)
        bootstrap_layout.addWidget(self.enable_bootstrap, 0, 0, 1, 2)

        self.bootstrap_samples = QSpinBox()
        self.bootstrap_samples.setRange(50, 10000)
        self.bootstrap_samples.setValue(1000)
        self.bootstrap_samples.setSingleStep(100)
        bootstrap_layout.addWidget(QLabel("Bootstrap Samples:"), 1, 0)
        bootstrap_layout.addWidget(self.bootstrap_samples, 1, 1)

        self.bootstrap_method = QComboBox()
        self.bootstrap_method.addItems([
            "Residual Bootstrap",
            "Parametric Bootstrap",
            "Non-parametric Bootstrap",
            "Wild Bootstrap"
        ])
        bootstrap_layout.addWidget(QLabel("Bootstrap Method:"), 1, 2)
        bootstrap_layout.addWidget(self.bootstrap_method, 1, 3)

        self.bootstrap_confidence_level = QDoubleSpinBox()
        self.bootstrap_confidence_level.setRange(0.50, 0.99)
        self.bootstrap_confidence_level.setValue(0.95)
        self.bootstrap_confidence_level.setSingleStep(0.01)
        self.bootstrap_confidence_level.setDecimals(2)
        bootstrap_layout.addWidget(QLabel("Confidence Level:"), 2, 0)
        bootstrap_layout.addWidget(self.bootstrap_confidence_level, 2, 1)

        self.parallel_bootstrap = QCheckBox("Parallel Processing")
        self.parallel_bootstrap.setChecked(True)
        bootstrap_layout.addWidget(self.parallel_bootstrap, 2, 2, 1, 2)

        layout.addWidget(bootstrap_group)

        # Cross-Validation Settings
        cv_group = QGroupBox("Cross-Validation")
        cv_layout = QGridLayout(cv_group)

        self.enable_cross_validation = QCheckBox("Enable Cross-Validation")
        self.enable_cross_validation.setChecked(False)
        cv_layout.addWidget(self.enable_cross_validation, 0, 0, 1, 2)

        self.cv_folds = QSpinBox()
        self.cv_folds.setRange(2, 20)
        self.cv_folds.setValue(5)
        cv_layout.addWidget(QLabel("CV Folds:"), 1, 0)
        cv_layout.addWidget(self.cv_folds, 1, 1)

        self.cv_scoring = QComboBox()
        self.cv_scoring.addItems([
            "R² Score",
            "Mean Squared Error",
            "Mean Absolute Error",
            "Log-Likelihood"
        ])
        cv_layout.addWidget(QLabel("Scoring Metric:"), 1, 2)
        cv_layout.addWidget(self.cv_scoring, 1, 3)

        layout.addWidget(cv_group)

        self.tab_widget.addTab(bootstrap_widget, "Bootstrap")

    def setup_status_controls(self):
        """Setup status display and action controls."""
        self.status_frame = QFrame()
        self.status_frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(self.status_frame)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QLabel("Ready for fitting")
        self.status_text.setStyleSheet("QLabel { color: green; }")
        layout.addWidget(self.status_text)

        # Action buttons
        button_layout = QHBoxLayout()

        self.validate_button = QPushButton("Validate Parameters")
        self.validate_button.clicked.connect(self.validate_parameters)
        button_layout.addWidget(self.validate_button)

        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_to_defaults)
        button_layout.addWidget(self.reset_button)

        self.apply_button = QPushButton("Apply & Fit")
        self.apply_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.apply_button.clicked.connect(self.apply_settings)
        button_layout.addWidget(self.apply_button)

        layout.addLayout(button_layout)

    def setup_connections(self):
        """Setup signal connections."""
        # Update method description when method changes
        self.method_combo.currentTextChanged.connect(self.update_method_description)

        # Enable/disable dependent controls
        self.enable_outlier_detection.toggled.connect(
            self.outlier_threshold.setEnabled
        )
        self.enable_bootstrap.toggled.connect(self.toggle_bootstrap_controls)
        self.enable_cross_validation.toggled.connect(self.toggle_cv_controls)
        self.enable_diagnostics.toggled.connect(self.toggle_diagnostic_controls)

        # Parameter change notifications
        for widget in [self.tolerance_factor, self.max_iterations, self.outlier_threshold,
                      self.confidence_level, self.bootstrap_samples]:
            widget.valueChanged.connect(self.on_parameter_changed)

        for widget in [self.method_combo, self.parameter_estimation_method,
                      self.bootstrap_method, self.cv_scoring]:
            widget.currentTextChanged.connect(self.on_parameter_changed)

        for widget in [self.enable_outlier_detection, self.enable_weight_adjustment,
                      self.enable_performance_tracking, self.enable_caching,
                      self.enable_diagnostics, self.enable_bootstrap]:
            widget.toggled.connect(self.on_parameter_changed)

    def setup_validation_timer(self):
        """Setup timer for real-time validation."""
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self.validate_parameters_delayed)

    def load_default_parameters(self):
        """Load default parameter values."""
        self.update_method_description()
        self.toggle_bootstrap_controls(False)
        self.toggle_cv_controls(False)

    def update_method_description(self):
        """Update the method description text."""
        method = self.method_combo.currentText()
        descriptions = {
            "Auto (TRF → LM → DE)": "Automatically selects best method: starts with Trust Region Reflective, falls back to Levenberg-Marquardt, then Differential Evolution if needed.",
            "Trust Region Reflective (TRF)": "Robust method for bounded optimization problems with excellent convergence properties.",
            "Levenberg-Marquardt (LM)": "Fast gradient-based method for unconstrained problems, excellent for well-conditioned fits.",
            "Differential Evolution (DE)": "Global optimization method, best for difficult problems with multiple local minima.",
            "Robust (with outlier detection)": "Automatically detects and handles outliers using statistical methods.",
            "Bayesian (MCMC)": "Markov Chain Monte Carlo sampling for full posterior distributions and uncertainty quantification."
        }
        self.method_description.setText(descriptions.get(method, ""))

    def toggle_bootstrap_controls(self, enabled):
        """Enable/disable bootstrap-related controls."""
        controls = [self.bootstrap_samples, self.bootstrap_method,
                   self.bootstrap_confidence_level, self.parallel_bootstrap]
        for control in controls:
            control.setEnabled(enabled)

    def toggle_cv_controls(self, enabled):
        """Enable/disable cross-validation controls."""
        controls = [self.cv_folds, self.cv_scoring]
        for control in controls:
            control.setEnabled(enabled)

    def toggle_diagnostic_controls(self, enabled):
        """Enable/disable diagnostic controls."""
        controls = [self.residual_analysis, self.parameter_correlation,
                   self.goodness_of_fit, self.model_comparison,
                   self.show_confidence_bands, self.show_prediction_intervals,
                   self.highlight_outliers, self.show_residual_plots]
        for control in controls:
            control.setEnabled(enabled)

    def on_parameter_changed(self):
        """Handle parameter changes with validation delay."""
        self.validation_timer.start(500)  # 500ms delay

    def validate_parameters_delayed(self):
        """Perform delayed validation."""
        self.validate_parameters()

    def validate_parameters(self):
        """Validate current parameter settings."""
        try:
            # Basic validation
            issues = []

            # Check tolerance factor
            if self.tolerance_factor.value() < 0.1:
                issues.append("Tolerance factor too small (may cause convergence issues)")
            elif self.tolerance_factor.value() > 5.0:
                issues.append("Tolerance factor too large (may reduce precision)")

            # Check max iterations
            if self.max_iterations.value() < 1000:
                issues.append("Max iterations may be too low for complex fits")

            # Check bootstrap samples
            if self.enable_bootstrap.isChecked() and self.bootstrap_samples.value() < 100:
                issues.append("Bootstrap samples too low for reliable statistics")

            # Check confidence level
            if self.confidence_level.value() < 0.8:
                issues.append("Low confidence level may miss important uncertainties")

            # Update status
            if issues:
                self.status_text.setText(f"Validation warnings: {'; '.join(issues[:2])}")
                self.status_text.setStyleSheet("QLabel { color: orange; }")
                self.validation_completed.emit(False, '; '.join(issues))
            else:
                self.status_text.setText("Parameters validated successfully")
                self.status_text.setStyleSheet("QLabel { color: green; }")
                self.validation_completed.emit(True, "")

        except Exception as e:
            self.logger.error(f"Parameter validation failed: {e}")
            self.status_text.setText(f"Validation error: {e}")
            self.status_text.setStyleSheet("QLabel { color: red; }")
            self.validation_completed.emit(False, str(e))

    def reset_to_defaults(self):
        """Reset all parameters to default values."""
        self.method_combo.setCurrentText("Auto (TRF → LM → DE)")
        self.tolerance_factor.setValue(1.0)
        self.max_iterations.setValue(10000)
        self.enable_outlier_detection.setChecked(True)
        self.outlier_threshold.setValue(2.5)
        self.enable_weight_adjustment.setChecked(True)
        self.enable_performance_tracking.setChecked(True)
        self.enable_caching.setChecked(True)
        self.enable_diagnostics.setChecked(True)
        self.enable_bootstrap.setChecked(False)
        self.bootstrap_samples.setValue(1000)
        self.confidence_level.setValue(0.95)

        self.logger.info("Reset parameters to defaults")

    def apply_settings(self):
        """Apply current settings and emit fit request."""
        self.validate_parameters()
        parameters = self.get_all_parameters()
        self.fit_requested.emit(parameters)

    def get_all_parameters(self):
        """Get all current parameter values as a dictionary."""
        return {
            # Basic parameters
            'optimization_method': self.method_combo.currentText(),
            'tolerance_factor': self.tolerance_factor.value(),
            'max_iterations': self.max_iterations.value(),
            'enable_outlier_detection': self.enable_outlier_detection.isChecked(),
            'outlier_threshold': self.outlier_threshold.value(),
            'enable_weight_adjustment': self.enable_weight_adjustment.isChecked(),

            # Advanced parameters
            'enable_performance_tracking': self.enable_performance_tracking.isChecked(),
            'enable_caching': self.enable_caching.isChecked(),
            'strategy_timeout': self.strategy_timeout.value(),
            'parameter_estimation_method': self.parameter_estimation_method.currentText(),
            'estimation_samples': self.estimation_samples.value(),
            'fallback_on_failure': self.fallback_on_failure.isChecked(),
            'retry_on_boundary': self.retry_on_boundary.isChecked(),
            'max_retries': self.max_retries.value(),

            # Diagnostic parameters
            'enable_diagnostics': self.enable_diagnostics.isChecked(),
            'residual_analysis': self.residual_analysis.isChecked(),
            'parameter_correlation': self.parameter_correlation.isChecked(),
            'goodness_of_fit': self.goodness_of_fit.isChecked(),
            'model_comparison': self.model_comparison.isChecked(),
            'show_confidence_bands': self.show_confidence_bands.isChecked(),
            'show_prediction_intervals': self.show_prediction_intervals.isChecked(),
            'highlight_outliers': self.highlight_outliers.isChecked(),
            'show_residual_plots': self.show_residual_plots.isChecked(),
            'confidence_level': self.confidence_level.value(),

            # Bootstrap parameters
            'enable_bootstrap': self.enable_bootstrap.isChecked(),
            'bootstrap_samples': self.bootstrap_samples.value(),
            'bootstrap_method': self.bootstrap_method.currentText(),
            'bootstrap_confidence_level': self.bootstrap_confidence_level.value(),
            'parallel_bootstrap': self.parallel_bootstrap.isChecked(),

            # Cross-validation parameters
            'enable_cross_validation': self.enable_cross_validation.isChecked(),
            'cv_folds': self.cv_folds.value(),
            'cv_scoring': self.cv_scoring.currentText(),
        }

    def set_parameters(self, parameters):
        """Set parameters from a dictionary."""
        param_map = {
            'optimization_method': self.method_combo.setCurrentText,
            'tolerance_factor': self.tolerance_factor.setValue,
            'max_iterations': self.max_iterations.setValue,
            'enable_outlier_detection': self.enable_outlier_detection.setChecked,
            'outlier_threshold': self.outlier_threshold.setValue,
            'enable_weight_adjustment': self.enable_weight_adjustment.setChecked,
            'enable_performance_tracking': self.enable_performance_tracking.setChecked,
            'enable_caching': self.enable_caching.setChecked,
            'enable_diagnostics': self.enable_diagnostics.setChecked,
            'confidence_level': self.confidence_level.setValue,
            'enable_bootstrap': self.enable_bootstrap.setChecked,
            'bootstrap_samples': self.bootstrap_samples.setValue,
        }

        for param_name, value in parameters.items():
            if param_name in param_map:
                try:
                    param_map[param_name](value)
                except Exception as e:
                    self.logger.warning(f"Failed to set parameter {param_name}: {e}")

    def show_progress(self, value, maximum=100):
        """Show progress during fitting operations."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)

        if value >= maximum:
            QTimer.singleShot(2000, lambda: self.progress_bar.setVisible(False))

    def update_status(self, message, color="black"):
        """Update status message."""
        self.status_text.setText(message)
        self.status_text.setStyleSheet(f"QLabel {{ color: {color}; }}")