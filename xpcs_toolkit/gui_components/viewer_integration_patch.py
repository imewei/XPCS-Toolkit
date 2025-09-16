"""
XPCS Viewer Integration Patch for Robust Fitting

This module provides the integration patch for adding robust fitting capabilities
to the existing XPCS viewer without breaking existing functionality.

Usage:
------
To integrate robust fitting into the XPCS viewer, add this to xpcs_viewer.py:

    from .gui_components.viewer_integration_patch import RobustFittingPatch

    # In XpcsViewer.__init__():
    self.robust_fitting_patch = RobustFittingPatch(self)
    self.robust_fitting_patch.apply_patch()

This will add a "Robust Fitting" button to the G2 tab that opens the advanced
interface while preserving all existing functionality.
"""

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QPushButton, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QCheckBox, QGroupBox, QTabWidget, QWidget, QSplitter, QMessageBox
)
from PySide6.QtGui import QFont, QIcon

from .g2_tab_enhancement import EnhancedG2TabWidget, G2IntegrationHelper
from .robust_fitting_integration import RobustFittingIntegrationWidget
from ..helper.fitting import single_exp, double_exp
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RobustFittingDialog(QDialog):
    """
    Dialog for robust G2 fitting that can be opened from the main interface.

    This provides a non-intrusive way to add robust fitting capabilities
    without modifying the existing tab structure.
    """

    fitting_completed = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced G2 Fitting with Robust Optimization")
        self.setModal(False)  # Allow interaction with main window
        self.resize(1200, 800)

        self.setup_ui()
        self.robust_widget = None

    def setup_ui(self):
        """Setup dialog UI."""
        layout = QVBoxLayout(self)

        # Header
        header_label = QLabel("Advanced G2 Fitting with Robust Optimization")
        header_label.setFont(QFont("Arial", 14, QFont.Bold))
        header_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(header_label)

        # Description
        desc_label = QLabel(
            "This interface provides advanced G2 fitting with outlier detection, "
            "uncertainty quantification, and comprehensive diagnostics."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("QLabel { color: gray; font-style: italic; margin: 10px; }")
        layout.addWidget(desc_label)

        # Main robust fitting widget
        self.robust_widget = RobustFittingIntegrationWidget()
        layout.addWidget(self.robust_widget)

        # Connect signals
        self.robust_widget.fitting_results_ready.connect(self.fitting_completed)

        # Buttons
        button_layout = QHBoxLayout()

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        button_layout.addStretch()

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        button_layout.addWidget(self.export_button)

        layout.addLayout(button_layout)

        # Connect to enable export when fitting completes
        self.fitting_completed.connect(lambda: self.export_button.setEnabled(True))

    def set_fitting_data(self, x_data, y_data, y_err=None, bounds=None, p0=None, fitting_function=None):
        """Set data for robust fitting."""
        if self.robust_widget:
            self.robust_widget.set_fitting_data(
                x_data, y_data, y_err, bounds, p0, fitting_function
            )

    def export_results(self):
        """Export robust fitting results."""
        if self.robust_widget:
            self.robust_widget.export_results()


class RobustFittingPatch:
    """
    Integration patch for adding robust fitting to existing XPCS viewer.

    This class provides a minimal, non-intrusive way to add robust fitting
    capabilities to the existing XPCS viewer interface.
    """

    def __init__(self, xpcs_viewer):
        """
        Initialize the patch.

        Parameters:
        -----------
        xpcs_viewer : XpcsViewer
            The main XPCS viewer instance
        """
        self.viewer = xpcs_viewer
        self.logger = logger
        self.robust_fitting_dialog = None
        self.current_g2_data = {}

    def apply_patch(self):
        """Apply the robust fitting patch to the viewer."""
        try:
            # Add robust fitting button to G2 tab
            self.add_robust_fitting_button()

            # Patch the plot_g2 method to capture data
            self.patch_plot_g2_method()

            # Add menu option (if applicable)
            self.add_menu_option()

            self.logger.info("Robust fitting patch applied successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply robust fitting patch: {e}")
            return False

    def add_robust_fitting_button(self):
        """Add robust fitting button to the G2 tab."""
        try:
            # Find the G2 tab - this depends on the specific UI structure
            # The button will be added to the fitting controls area

            # Create the button
            self.robust_fitting_button = QPushButton("🔬 Advanced Fitting")
            self.robust_fitting_button.setToolTip(
                "Open advanced robust fitting interface with outlier detection "
                "and uncertainty quantification"
            )
            self.robust_fitting_button.clicked.connect(self.open_robust_fitting_dialog)

            # Add to the G2 fitting controls area
            # This will depend on the specific layout of the G2 tab
            if hasattr(self.viewer, 'groupBox_2'):  # G2 fitting groupbox
                layout = self.viewer.groupBox_2.layout()
                if layout:
                    # Add button to the layout
                    layout.addWidget(self.robust_fitting_button, layout.rowCount(), 0, 1, 2)

            self.logger.info("Added robust fitting button to G2 tab")

        except Exception as e:
            self.logger.warning(f"Could not add robust fitting button: {e}")

    def patch_plot_g2_method(self):
        """Patch the plot_g2 method to capture G2 data for robust fitting."""
        try:
            # Store original method
            original_plot_g2 = self.viewer.plot_g2

            def patched_plot_g2(*args, **kwargs):
                """Patched plot_g2 method that captures data."""
                try:
                    # Call original method
                    result = original_plot_g2(*args, **kwargs)

                    # Capture G2 data after successful plot
                    self.capture_g2_data()

                    return result

                except Exception as e:
                    self.logger.error(f"Error in patched plot_g2: {e}")
                    # Fall back to original behavior
                    return original_plot_g2(*args, **kwargs)

            # Replace the method
            self.viewer.plot_g2 = patched_plot_g2

            self.logger.info("Patched plot_g2 method for data capture")

        except Exception as e:
            self.logger.warning(f"Could not patch plot_g2 method: {e}")

    def capture_g2_data(self):
        """Capture current G2 data for robust fitting."""
        try:
            # Get selected files
            rows = self.viewer.get_selected_rows()
            if not rows:
                return

            # Get current G2 parameters
            p = self.viewer.check_g2_number()
            bounds, fit_flag, fit_func = self.viewer.check_g2_fitting_number()

            # Get file list
            xf_list = self.viewer.vk.get_xf_list(rows=rows, filter_atype="Multitau")
            if not xf_list:
                return

            # Extract G2 data from first file for now
            xf = xf_list[0]
            q_range = (p[0], p[1])
            t_range = (p[2], p[3])

            # Get G2 data
            try:
                q, tel, g2, g2_err, labels = xf.get_g2_data(qrange=q_range, trange=t_range)

                if q is not None and tel is not None and g2 is not None:
                    # Use first q-value's data for robust fitting demo
                    x_data = tel[0] if isinstance(tel, list) else tel
                    y_data = g2[0][:, 0] if g2[0].ndim > 1 else g2[0]
                    y_err = g2_err[0][:, 0] if g2_err[0] is not None and g2_err[0].ndim > 1 else g2_err[0]

                    # Determine fitting function
                    fitting_function = single_exp if fit_func == "single" else double_exp

                    # Store data
                    self.current_g2_data = {
                        'x_data': x_data,
                        'y_data': y_data,
                        'y_err': y_err,
                        'bounds': bounds,
                        'fit_flag': fit_flag,
                        'fitting_function': fitting_function,
                        'q_value': q[0][0] if isinstance(q, list) and len(q[0]) > 0 else None
                    }

                    # Enable robust fitting button
                    if hasattr(self, 'robust_fitting_button'):
                        self.robust_fitting_button.setEnabled(True)
                        self.robust_fitting_button.setToolTip(
                            f"Open advanced fitting for {len(y_data)} data points"
                        )

                    self.logger.info(f"Captured G2 data: {len(x_data)} points")

            except Exception as e:
                self.logger.warning(f"Could not extract G2 data: {e}")

        except Exception as e:
            self.logger.warning(f"Could not capture G2 data: {e}")

    def open_robust_fitting_dialog(self):
        """Open the robust fitting dialog."""
        try:
            if not self.current_g2_data:
                QMessageBox.warning(
                    self.viewer,
                    "No Data",
                    "Please plot G2 data first to enable robust fitting."
                )
                return

            # Create dialog if it doesn't exist
            if self.robust_fitting_dialog is None:
                self.robust_fitting_dialog = RobustFittingDialog(self.viewer)
                self.robust_fitting_dialog.fitting_completed.connect(
                    self.on_robust_fitting_completed
                )

            # Set current data
            self.robust_fitting_dialog.set_fitting_data(**self.current_g2_data)

            # Show dialog
            self.robust_fitting_dialog.show()
            self.robust_fitting_dialog.raise_()
            self.robust_fitting_dialog.activateWindow()

            self.logger.info("Opened robust fitting dialog")

        except Exception as e:
            self.logger.error(f"Failed to open robust fitting dialog: {e}")
            QMessageBox.critical(
                self.viewer,
                "Error",
                f"Failed to open robust fitting interface:\n{e}"
            )

    def on_robust_fitting_completed(self, results):
        """Handle robust fitting completion."""
        try:
            # Display results summary
            r_squared = results.get('r_squared', 0)
            rmse = results.get('rmse', 0)

            message = f"Robust fitting completed!\n\nR² = {r_squared:.4f}\nRMSE = {rmse:.6f}"

            if 'popt' in results:
                popt = results['popt']
                if len(popt) >= 3:
                    message += f"\n\nFitted parameters:"
                    message += f"\nτ = {popt[0]:.6f}"
                    message += f"\nBackground = {popt[1]:.6f}"
                    message += f"\nContrast = {popt[2]:.6f}"

            QMessageBox.information(self.viewer, "Robust Fitting Results", message)

            # Update status
            if hasattr(self.viewer, 'statusbar'):
                self.viewer.statusbar.showMessage(
                    f"Robust fitting completed (R² = {r_squared:.4f})", 5000
                )

            self.logger.info("Robust fitting completed successfully")

        except Exception as e:
            self.logger.error(f"Error handling robust fitting results: {e}")

    def add_menu_option(self):
        """Add robust fitting option to menu (if applicable)."""
        try:
            # This would add a menu option if the viewer has a menu bar
            # Implementation depends on the specific menu structure
            pass
        except Exception as e:
            self.logger.warning(f"Could not add menu option: {e}")

    def get_integration_status(self):
        """Get the status of the integration."""
        status = {
            'patch_applied': hasattr(self, 'robust_fitting_button'),
            'button_enabled': (
                hasattr(self, 'robust_fitting_button') and
                self.robust_fitting_button.isEnabled()
            ),
            'data_available': bool(self.current_g2_data),
            'dialog_open': (
                self.robust_fitting_dialog is not None and
                self.robust_fitting_dialog.isVisible()
            )
        }
        return status


# Utility function for easy integration
def integrate_robust_fitting(xpcs_viewer):
    """
    Utility function to easily integrate robust fitting into XPCS viewer.

    Usage:
    ------
    from xpcs_toolkit.gui_components.viewer_integration_patch import integrate_robust_fitting

    # In XpcsViewer.__init__():
    integrate_robust_fitting(self)
    """
    try:
        patch = RobustFittingPatch(xpcs_viewer)
        success = patch.apply_patch()

        if success:
            # Store patch reference for later use
            xpcs_viewer._robust_fitting_patch = patch
            logger.info("Robust fitting integration successful")
            return patch
        else:
            logger.error("Robust fitting integration failed")
            return None

    except Exception as e:
        logger.error(f"Robust fitting integration error: {e}")
        return None


# Example integration code for the main viewer
INTEGRATION_EXAMPLE = '''
# Add this to xpcs_viewer.py in the XpcsViewer.__init__ method:

def __init__(self, path=None, label_style=None):
    super().__init__()
    self.setupUi(self)

    # ... existing initialization code ...

    # Add robust fitting integration (add this near the end)
    try:
        from .gui_components.viewer_integration_patch import integrate_robust_fitting
        self.robust_fitting_patch = integrate_robust_fitting(self)
        if self.robust_fitting_patch:
            logger.info("Robust fitting integration enabled")
        else:
            logger.warning("Robust fitting integration failed")
    except ImportError:
        logger.info("Robust fitting components not available")
    except Exception as e:
        logger.warning(f"Could not integrate robust fitting: {e}")
'''