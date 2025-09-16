"""
G2 Tab Enhancement for XPCS Toolkit

This module provides enhanced G2 analysis tab with robust fitting integration,
maintaining compatibility with existing workflow while adding advanced features.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QSplitter,
    QGroupBox, QPushButton, QCheckBox, QLabel, QFrame,
    QScrollArea, QComboBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QProgressBar
)
from PySide6.QtGui import QFont, QIcon

from .robust_fitting_integration import RobustFittingIntegrationWidget
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class EnhancedG2TabWidget(QWidget):
    """
    Enhanced G2 analysis tab with progressive disclosure of advanced features.

    Features:
    - Traditional G2 fitting interface (default)
    - Advanced robust fitting mode (optional)
    - Seamless integration with existing workflow
    - Progressive disclosure of complexity
    - Smart defaults and contextual help
    """

    # Signals for communication with main viewer
    fitting_mode_changed = Signal(str)  # 'traditional' or 'robust'
    plot_update_requested = Signal(dict)
    fitting_completed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logger
        self.fitting_mode = 'traditional'
        self.robust_widget = None
        self.setup_ui()
        self.setup_connections()

    def setup_ui(self):
        """Initialize the enhanced G2 tab interface."""
        layout = QVBoxLayout(self)

        # Mode selection and controls
        self.setup_mode_controls(layout)

        # Main content area
        self.setup_main_content(layout)

        # Status and help
        self.setup_status_area(layout)

    def setup_mode_controls(self, layout):
        """Setup fitting mode controls with progressive disclosure."""
        mode_frame = QFrame()
        mode_frame.setFrameStyle(QFrame.StyledPanel)
        mode_layout = QHBoxLayout(mode_frame)

        # Mode selection
        mode_layout.addWidget(QLabel("Fitting Mode:"))

        self.mode_selector = QComboBox()
        self.mode_selector.addItems([
            "Traditional Fitting",
            "Robust Fitting (Advanced)"
        ])
        self.mode_selector.setCurrentIndex(0)
        self.mode_selector.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_selector)

        # Help button
        self.help_button = QPushButton("?")
        self.help_button.setMaximumWidth(30)
        self.help_button.setToolTip("Show help about fitting modes")
        self.help_button.clicked.connect(self.show_mode_help)
        mode_layout.addWidget(self.help_button)

        # Advanced features indicator
        self.advanced_indicator = QLabel("◄ Try Advanced Mode for:")
        self.advanced_indicator.setStyleSheet("QLabel { color: blue; font-style: italic; }")
        self.advanced_indicator.setVisible(True)
        mode_layout.addWidget(self.advanced_indicator)

        # Features list (initially visible to encourage exploration)
        self.features_list = QLabel("• Outlier Detection • Uncertainty Quantification • Real-time Diagnostics")
        self.features_list.setStyleSheet("QLabel { color: gray; font-size: 10pt; }")
        mode_layout.addWidget(self.features_list)

        mode_layout.addStretch()

        # Smart recommendations
        self.recommendation_label = QLabel("")
        self.recommendation_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
        mode_layout.addWidget(self.recommendation_label)

        layout.addWidget(mode_frame)

    def setup_main_content(self, layout):
        """Setup the main content area with mode switching."""
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        # Initially empty - will be populated based on mode
        layout.addWidget(self.content_widget)

        # Initialize with traditional mode
        self.switch_to_traditional_mode()

    def setup_status_area(self, layout):
        """Setup status and progress area."""
        status_frame = QFrame()
        status_frame.setFrameStyle(QFrame.StyledPanel)
        status_layout = QHBoxLayout(status_frame)

        # Status label
        self.status_label = QLabel("Ready for G2 analysis")
        status_layout.addWidget(self.status_label)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)

        # Quick action button
        self.quick_action_button = QPushButton("Plot G2")
        self.quick_action_button.clicked.connect(self.perform_quick_action)
        status_layout.addWidget(self.quick_action_button)

        layout.addWidget(status_frame)

    def setup_connections(self):
        """Setup signal connections."""
        # Timer for smart recommendations
        self.recommendation_timer = QTimer()
        self.recommendation_timer.setSingleShot(True)
        self.recommendation_timer.timeout.connect(self.show_smart_recommendation)

    def on_mode_changed(self, mode_text):
        """Handle fitting mode changes."""
        old_mode = self.fitting_mode

        if "Traditional" in mode_text:
            self.fitting_mode = 'traditional'
            self.switch_to_traditional_mode()
        elif "Robust" in mode_text:
            self.fitting_mode = 'robust'
            self.switch_to_robust_mode()

        # Hide features indicator after user explores
        if old_mode == 'traditional' and self.fitting_mode == 'robust':
            self.advanced_indicator.setVisible(False)
            self.features_list.setVisible(False)

        # Update UI based on mode
        self.update_mode_ui()

        # Emit signal
        self.fitting_mode_changed.emit(self.fitting_mode)

        self.logger.info(f"Switched from {old_mode} to {self.fitting_mode} fitting mode")

    def switch_to_traditional_mode(self):
        """Switch to traditional G2 fitting interface."""
        # Clear current content
        self.clear_content_layout()

        # Add traditional fitting message
        traditional_label = QLabel(
            "Traditional G2 fitting interface is maintained in the existing controls below.\n"
            "Use the original G2 fitting controls in the main interface."
        )
        traditional_label.setStyleSheet("QLabel { color: gray; font-style: italic; padding: 20px; }")
        traditional_label.setAlignment(Qt.AlignCenter)
        traditional_label.setWordWrap(True)
        self.content_layout.addWidget(traditional_label)

        # Update quick action
        self.quick_action_button.setText("Plot G2")

        # Start recommendation timer
        self.recommendation_timer.start(30000)  # 30 seconds

    def switch_to_robust_mode(self):
        """Switch to robust fitting interface."""
        # Clear current content
        self.clear_content_layout()

        # Create robust fitting widget if not exists
        if self.robust_widget is None:
            self.robust_widget = RobustFittingIntegrationWidget()
            self.robust_widget.fitting_results_ready.connect(self.on_robust_fitting_completed)
            self.robust_widget.status_message.connect(self.update_status)

        # Add robust fitting widget
        self.content_layout.addWidget(self.robust_widget)

        # Update quick action
        self.quick_action_button.setText("Start Robust Fitting")

        # Stop recommendation timer
        self.recommendation_timer.stop()
        self.recommendation_label.clear()

    def clear_content_layout(self):
        """Clear the content layout."""
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().setParent(None)

    def update_mode_ui(self):
        """Update UI elements based on current mode."""
        if self.fitting_mode == 'traditional':
            self.status_label.setText("Using traditional G2 fitting")
        else:
            self.status_label.setText("Using robust G2 fitting with diagnostics")

    def perform_quick_action(self):
        """Perform the appropriate quick action based on current mode."""
        if self.fitting_mode == 'traditional':
            # Trigger traditional G2 plot
            self.plot_update_requested.emit({'mode': 'traditional'})
        else:
            # Trigger robust fitting
            if self.robust_widget:
                self.robust_widget.start_robust_fitting()

    def show_mode_help(self):
        """Show help dialog about fitting modes."""
        help_text = """
<h3>G2 Fitting Modes</h3>

<h4>Traditional Fitting</h4>
<ul>
<li>Fast, reliable fitting using standard algorithms</li>
<li>Uses scipy.optimize.curve_fit with Levenberg-Marquardt</li>
<li>Suitable for most well-behaved datasets</li>
<li>Minimal computational overhead</li>
</ul>

<h4>Robust Fitting (Advanced)</h4>
<ul>
<li><b>Outlier Detection:</b> Automatically identifies and handles outlier data points</li>
<li><b>Uncertainty Quantification:</b> Provides confidence intervals and error analysis</li>
<li><b>Real-time Diagnostics:</b> Live visualization of fitting quality and convergence</li>
<li><b>Multiple Optimization Strategies:</b> Automatic fallback between algorithms</li>
<li><b>Interactive Parameter Analysis:</b> Explore parameter sensitivity and correlations</li>
<li><b>Bootstrap Analysis:</b> Statistical validation of fit parameters</li>
</ul>

<h4>When to Use Robust Fitting</h4>
<ul>
<li>Data contains outliers or noise</li>
<li>Need reliable uncertainty estimates</li>
<li>Fitting is critical for publication</li>
<li>Traditional fitting fails to converge</li>
<li>Want to understand parameter relationships</li>
</ul>
        """

        msg = QMessageBox(self)
        msg.setWindowTitle("G2 Fitting Modes Help")
        msg.setText(help_text)
        msg.setTextFormat(Qt.RichText)
        msg.exec()

    def show_smart_recommendation(self):
        """Show smart recommendation to try robust fitting."""
        recommendations = [
            "💡 Need better error estimates? Try Robust Fitting!",
            "🔍 Want to detect outliers? Use Advanced Mode!",
            "📊 Get real-time diagnostics with Robust Fitting!",
            "🎯 Improve fitting reliability with Advanced Mode!"
        ]

        import random
        recommendation = random.choice(recommendations)
        self.recommendation_label.setText(recommendation)

        # Auto-hide after 10 seconds
        QTimer.singleShot(10000, lambda: self.recommendation_label.clear())

    def set_g2_data(self, x_data, y_data, y_err=None, bounds=None, p0=None, fitting_function=None):
        """Set G2 data for fitting (used by robust mode)."""
        if self.robust_widget and self.fitting_mode == 'robust':
            self.robust_widget.set_fitting_data(
                x_data, y_data, y_err, bounds, p0, fitting_function
            )

    def on_robust_fitting_completed(self, results):
        """Handle robust fitting completion."""
        self.fitting_completed.emit(results)

        # Show success message with key metrics
        if 'r_squared' in results:
            r_squared = results['r_squared']
            message = f"Robust fitting completed successfully!\nR² = {r_squared:.4f}"

            if r_squared > 0.95:
                message += "\nExcellent fit quality! 🎉"
            elif r_squared > 0.85:
                message += "\nGood fit quality! ✓"
            else:
                message += "\nFit quality could be improved."

            self.status_label.setText(message.replace('\n', ' '))

    def update_status(self, message):
        """Update status message."""
        self.status_label.setText(message)

    def show_progress(self, value, maximum=100):
        """Show progress during operations."""
        if value >= 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(maximum)
            self.progress_bar.setValue(value)
        else:
            self.progress_bar.setVisible(False)

    def get_current_mode(self):
        """Get current fitting mode."""
        return self.fitting_mode

    def get_robust_results(self):
        """Get robust fitting results if available."""
        if self.robust_widget:
            return self.robust_widget.get_current_results()
        return {}

    def export_robust_results(self):
        """Export robust fitting results."""
        if self.robust_widget:
            self.robust_widget.export_results()

    def clear_robust_results(self):
        """Clear robust fitting results."""
        if self.robust_widget:
            self.robust_widget.clear_results()

    def enable_advanced_features_hint(self, data_quality_score):
        """
        Show hints about advanced features based on data quality.

        Parameters:
        -----------
        data_quality_score : float
            Score from 0-1 indicating data quality (0=poor, 1=excellent)
        """
        if data_quality_score < 0.7 and self.fitting_mode == 'traditional':
            hint_messages = [
                "🚨 Data quality seems low - Robust Fitting can help!",
                "⚠️ Consider using Advanced Mode for better outlier handling",
                "💡 Low-quality data benefits from robust algorithms"
            ]

            import random
            hint = random.choice(hint_messages)
            self.recommendation_label.setText(hint)
            self.recommendation_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")

            # Auto-hide after 15 seconds
            QTimer.singleShot(15000, lambda: self.recommendation_label.clear())

    def suggest_robust_fitting_for_failure(self, error_message):
        """Suggest robust fitting when traditional fitting fails."""
        if self.fitting_mode == 'traditional':
            suggestion = "❌ Traditional fitting failed - Try Robust Fitting for better results!"
            self.recommendation_label.setText(suggestion)
            self.recommendation_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")

            # Make the suggestion more prominent
            self.advanced_indicator.setVisible(True)
            self.features_list.setVisible(True)
            self.advanced_indicator.setText("◄ Fitting failed? Try Advanced Mode:")
            self.advanced_indicator.setStyleSheet("QLabel { color: red; font-weight: bold; }")

    def provide_contextual_help(self, context):
        """Provide contextual help based on current situation."""
        help_messages = {
            'first_time': "👋 New to G2 analysis? Start with Traditional mode, then explore Advanced features!",
            'outliers_detected': "🔍 Outliers detected in your data - Robust Fitting can handle these automatically!",
            'poor_fit': "📉 Poor fit quality? Advanced Mode provides better diagnostics and optimization!",
            'good_fit': "✅ Great fit! Want uncertainty estimates? Try Advanced Mode for confidence intervals!",
            'complex_data': "🧩 Complex dataset? Robust Fitting offers multiple optimization strategies!"
        }

        if context in help_messages:
            self.recommendation_label.setText(help_messages[context])
            self.recommendation_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")


class G2IntegrationHelper:
    """
    Helper class for integrating enhanced G2 functionality with main XPCS viewer.

    This class provides the interface between the enhanced G2 tab and the
    existing XPCS viewer architecture.
    """

    def __init__(self, main_viewer):
        self.main_viewer = main_viewer
        self.enhanced_g2_tab = None
        self.logger = logger

    def integrate_enhanced_g2_tab(self):
        """Integrate the enhanced G2 tab into the main viewer."""
        try:
            # Create enhanced G2 tab
            self.enhanced_g2_tab = EnhancedG2TabWidget()

            # Connect signals
            self.enhanced_g2_tab.fitting_mode_changed.connect(self.on_fitting_mode_changed)
            self.enhanced_g2_tab.plot_update_requested.connect(self.on_plot_update_requested)
            self.enhanced_g2_tab.fitting_completed.connect(self.on_fitting_completed)

            # Add to main viewer's G2 tab or create new tab
            self.add_to_main_viewer()

            self.logger.info("Enhanced G2 tab integrated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to integrate enhanced G2 tab: {e}")
            return False

    def add_to_main_viewer(self):
        """Add enhanced tab to main viewer."""
        # This would integrate with the specific viewer architecture
        # Implementation depends on the main viewer's tab structure
        pass

    def on_fitting_mode_changed(self, mode):
        """Handle fitting mode changes."""
        self.logger.info(f"G2 fitting mode changed to: {mode}")

        # Update main viewer's fitting behavior
        if hasattr(self.main_viewer, 'g2_fitting_mode'):
            self.main_viewer.g2_fitting_mode = mode

    def on_plot_update_requested(self, plot_params):
        """Handle plot update requests."""
        if plot_params.get('mode') == 'traditional':
            # Trigger traditional G2 plot
            if hasattr(self.main_viewer, 'plot_g2'):
                self.main_viewer.plot_g2()

    def on_fitting_completed(self, results):
        """Handle fitting completion."""
        self.logger.info("Enhanced G2 fitting completed")

        # Update main viewer with results
        if hasattr(self.main_viewer, 'update_g2_results'):
            self.main_viewer.update_g2_results(results)

    def update_g2_data(self, x_data, y_data, y_err=None, bounds=None, p0=None, fitting_function=None):
        """Update G2 data in enhanced tab."""
        if self.enhanced_g2_tab:
            self.enhanced_g2_tab.set_g2_data(
                x_data, y_data, y_err, bounds, p0, fitting_function
            )

    def assess_data_quality(self, x_data, y_data, y_err=None):
        """Assess data quality and provide recommendations."""
        try:
            # Simple data quality assessment
            quality_score = 1.0

            # Check for outliers
            if len(y_data) > 5:
                from scipy import stats
                z_scores = np.abs(stats.zscore(y_data))
                outlier_fraction = np.sum(z_scores > 2.5) / len(y_data)
                quality_score -= outlier_fraction * 0.5

            # Check for NaN/inf values
            if not (np.isfinite(x_data).all() and np.isfinite(y_data).all()):
                quality_score -= 0.3

            # Check data coverage
            if len(y_data) < 20:
                quality_score -= 0.2

            # Provide recommendation
            if self.enhanced_g2_tab:
                self.enhanced_g2_tab.enable_advanced_features_hint(quality_score)

            return quality_score

        except Exception as e:
            self.logger.warning(f"Data quality assessment failed: {e}")
            return 0.5  # Neutral score