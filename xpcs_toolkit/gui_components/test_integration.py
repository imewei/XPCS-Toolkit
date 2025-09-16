"""
Test Script for Robust Fitting GUI Integration

This script provides comprehensive testing of the robust fitting GUI components
to ensure proper integration and functionality.
"""

import sys
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtCore import QTimer

# Import our GUI components
from .robust_fitting_controls import RobustFittingControlPanel
from .diagnostic_widgets import DiagnosticDashboard, RealTimeDiagnosticWidget
from .interactive_parameter_widgets import ParameterAnalysisWidget, ConfidenceIntervalWidget
from .enhanced_plotting import EnhancedG2PlotWidget, G2PlotControlWidget
from .robust_fitting_integration import RobustFittingIntegrationWidget
from .g2_tab_enhancement import EnhancedG2TabWidget
from ..helper.fitting import single_exp, RobustOptimizerWithDiagnostics
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_test_g2_data(noise_level=0.05, outlier_fraction=0.1, n_points=100):
    """Generate synthetic G2 data for testing."""
    # Time points (logarithmic scale)
    t_min, t_max = 1e-6, 1e-1
    tau = np.logspace(np.log10(t_min), np.log10(t_max), n_points)

    # True parameters
    true_tau = 1e-3
    true_bkg = 1.0
    true_cts = 0.5

    # Generate clean G2 data
    g2_clean = single_exp(tau, true_tau, true_bkg, true_cts)

    # Add noise
    noise = np.random.normal(0, noise_level, n_points)
    g2_noisy = g2_clean + noise

    # Add outliers
    n_outliers = int(outlier_fraction * n_points)
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    outlier_values = np.random.uniform(0.5, 2.0, n_outliers)
    g2_noisy[outlier_indices] = outlier_values

    # Error bars (simulate realistic uncertainties)
    g2_err = np.abs(g2_noisy) * 0.1 + 0.01

    return tau, g2_noisy, g2_err, true_tau, true_bkg, true_cts


class TestMainWindow(QMainWindow):
    """Test main window for GUI components."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("XPCS Robust Fitting GUI Test")
        self.setGeometry(100, 100, 1400, 900)

        # Generate test data
        self.x_data, self.y_data, self.y_err, self.true_tau, self.true_bkg, self.true_cts = generate_test_g2_data()

        self.setup_ui()

    def setup_ui(self):
        """Setup test UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create enhanced G2 tab
        self.enhanced_g2_tab = EnhancedG2TabWidget()
        layout.addWidget(self.enhanced_g2_tab)

        # Set test data
        bounds = ([1e-6, 0.5, 0.01], [1e-1, 1.5, 1.0])
        p0 = [1e-3, 1.0, 0.5]

        self.enhanced_g2_tab.set_g2_data(
            self.x_data, self.y_data, self.y_err, bounds, p0, single_exp
        )


class ComponentTester:
    """Test individual GUI components."""

    def __init__(self):
        self.logger = logger
        self.test_results = {}

    def test_robust_fitting_controls(self):
        """Test robust fitting control panel."""
        try:
            controls = RobustFittingControlPanel()
            controls.show()

            # Test parameter validation
            controls.validate_parameters()

            # Test parameter retrieval
            params = controls.get_all_parameters()
            assert isinstance(params, dict)
            assert 'optimization_method' in params

            self.test_results['robust_controls'] = 'PASS'
            logger.info("Robust fitting controls test: PASS")

        except Exception as e:
            self.test_results['robust_controls'] = f'FAIL: {e}'
            logger.error(f"Robust fitting controls test: FAIL - {e}")

    def test_diagnostic_widgets(self):
        """Test diagnostic visualization widgets."""
        try:
            # Test real-time diagnostic widget
            diag_widget = RealTimeDiagnosticWidget()
            diag_widget.show()

            # Generate test diagnostic data
            x_data, y_data, y_err, _, _, _ = generate_test_g2_data()
            fitted_values = y_data + np.random.normal(0, 0.01, len(y_data))
            residuals = y_data - fitted_values

            parameters = {'tau': 1e-3, 'bkg': 1.0, 'cts': 0.5}
            metrics = {'r_squared': 0.95, 'chi_squared': 1.2, 'rmse': 0.02}

            # Test diagnostic update
            diag_widget.update_diagnostics(fitted_values, residuals, parameters, metrics)

            # Test dashboard
            dashboard = DiagnosticDashboard()
            dashboard.show()

            self.test_results['diagnostic_widgets'] = 'PASS'
            logger.info("Diagnostic widgets test: PASS")

        except Exception as e:
            self.test_results['diagnostic_widgets'] = f'FAIL: {e}'
            logger.error(f"Diagnostic widgets test: FAIL - {e}")

    def test_parameter_analysis(self):
        """Test parameter analysis widgets."""
        try:
            # Test parameter analysis widget
            param_widget = ParameterAnalysisWidget()
            param_widget.show()

            # Add test parameters
            param_widget.add_parameter_control('tau', 1e-3, (1e-6, 1e-1))
            param_widget.add_parameter_control('bkg', 1.0, (0.5, 1.5))
            param_widget.add_parameter_control('cts', 0.5, (0.01, 1.0))

            # Test confidence interval widget
            conf_widget = ConfidenceIntervalWidget()
            conf_widget.show()

            self.test_results['parameter_analysis'] = 'PASS'
            logger.info("Parameter analysis test: PASS")

        except Exception as e:
            self.test_results['parameter_analysis'] = f'FAIL: {e}'
            logger.error(f"Parameter analysis test: FAIL - {e}")

    def test_enhanced_plotting(self):
        """Test enhanced plotting components."""
        try:
            # Test enhanced G2 plot widget
            plot_widget = EnhancedG2PlotWidget()
            plot_widget.show()

            # Generate test data
            x_data, y_data, y_err, _, _, _ = generate_test_g2_data()

            # Create test fit result
            fit_result = {
                'x_fit': x_data,
                'y_fit': single_exp(x_data, 1e-3, 1.0, 0.5)
            }

            # Create test uncertainty data
            uncertainty_data = {
                'confidence_lower': fit_result['y_fit'] * 0.95,
                'confidence_upper': fit_result['y_fit'] * 1.05
            }

            # Test plotting
            plot_widget.plot_g2_with_uncertainty(
                x_data, y_data, y_err, fit_result, uncertainty_data
            )

            # Test plot controls
            controls = G2PlotControlWidget()
            controls.show()

            self.test_results['enhanced_plotting'] = 'PASS'
            logger.info("Enhanced plotting test: PASS")

        except Exception as e:
            self.test_results['enhanced_plotting'] = f'FAIL: {e}'
            logger.error(f"Enhanced plotting test: FAIL - {e}")

    def test_integration_widget(self):
        """Test main integration widget."""
        try:
            # Test robust fitting integration widget
            integration_widget = RobustFittingIntegrationWidget()
            integration_widget.show()

            # Set test data
            x_data, y_data, y_err, _, _, _ = generate_test_g2_data()
            bounds = ([1e-6, 0.5, 0.01], [1e-1, 1.5, 1.0])
            p0 = [1e-3, 1.0, 0.5]

            integration_widget.set_fitting_data(
                x_data, y_data, y_err, bounds, p0, single_exp
            )

            self.test_results['integration_widget'] = 'PASS'
            logger.info("Integration widget test: PASS")

        except Exception as e:
            self.test_results['integration_widget'] = f'FAIL: {e}'
            logger.error(f"Integration widget test: FAIL - {e}")

    def test_enhanced_g2_tab(self):
        """Test enhanced G2 tab."""
        try:
            # Test enhanced G2 tab widget
            g2_tab = EnhancedG2TabWidget()
            g2_tab.show()

            # Set test data
            x_data, y_data, y_err, _, _, _ = generate_test_g2_data()
            bounds = ([1e-6, 0.5, 0.01], [1e-1, 1.5, 1.0])
            p0 = [1e-3, 1.0, 0.5]

            g2_tab.set_g2_data(x_data, y_data, y_err, bounds, p0, single_exp)

            self.test_results['enhanced_g2_tab'] = 'PASS'
            logger.info("Enhanced G2 tab test: PASS")

        except Exception as e:
            self.test_results['enhanced_g2_tab'] = f'FAIL: {e}'
            logger.error(f"Enhanced G2 tab test: FAIL - {e}")

    def run_all_tests(self):
        """Run all component tests."""
        logger.info("Starting GUI component tests...")

        test_methods = [
            self.test_robust_fitting_controls,
            self.test_diagnostic_widgets,
            self.test_parameter_analysis,
            self.test_enhanced_plotting,
            self.test_integration_widget,
            self.test_enhanced_g2_tab
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                test_name = test_method.__name__
                self.test_results[test_name] = f'FAIL: {e}'
                logger.error(f"{test_name}: FAIL - {e}")

        self.print_test_results()

    def print_test_results(self):
        """Print test results summary."""
        logger.info("=== GUI Component Test Results ===")

        passed = 0
        failed = 0

        for test_name, result in self.test_results.items():
            status = "PASS" if result == "PASS" else "FAIL"
            logger.info(f"{test_name}: {status}")

            if status == "PASS":
                passed += 1
            else:
                failed += 1

        total = passed + failed
        logger.info(f"\nSummary: {passed}/{total} tests passed ({failed} failed)")

        if failed == 0:
            logger.info("🎉 All GUI component tests passed!")
        else:
            logger.warning(f"⚠️ {failed} tests failed. Check logs for details.")


def test_robust_optimizer_integration():
    """Test robust optimizer integration."""
    try:
        logger.info("Testing robust optimizer integration...")

        # Generate test data
        x_data, y_data, y_err, true_tau, true_bkg, true_cts = generate_test_g2_data()

        # Test robust optimizer
        optimizer = RobustOptimizerWithDiagnostics(max_iterations=1000)

        # Define bounds and initial parameters
        bounds = ([1e-6, 0.5, 0.01], [1e-1, 1.5, 1.0])
        p0 = [1e-3, 1.0, 0.5]

        # Run optimization
        result = optimizer.optimize_with_full_diagnostics(
            single_exp, x_data, y_data, p0=p0, bounds=bounds, sigma=y_err
        )

        if result.get('success', False):
            popt = result['popt']
            logger.info(f"Optimization successful!")
            logger.info(f"True parameters: tau={true_tau:.6f}, bkg={true_bkg:.3f}, cts={true_cts:.3f}")
            logger.info(f"Fitted parameters: tau={popt[0]:.6f}, bkg={popt[1]:.3f}, cts={popt[2]:.3f}")

            # Check parameter accuracy
            tau_error = abs(popt[0] - true_tau) / true_tau
            bkg_error = abs(popt[1] - true_bkg) / true_bkg
            cts_error = abs(popt[2] - true_cts) / true_cts

            logger.info(f"Parameter errors: tau={tau_error:.2%}, bkg={bkg_error:.2%}, cts={cts_error:.2%}")

            if tau_error < 0.1 and bkg_error < 0.05 and cts_error < 0.1:
                logger.info("✅ Robust optimizer integration test: PASS")
                return True
            else:
                logger.warning("⚠️ Parameter accuracy below threshold")
                return False
        else:
            logger.error("❌ Optimization failed")
            return False

    except Exception as e:
        logger.error(f"❌ Robust optimizer integration test: FAIL - {e}")
        return False


def main():
    """Main test function."""
    app = QApplication(sys.argv)

    logger.info("Starting XPCS Robust Fitting GUI Integration Tests")

    # Test robust optimizer integration
    optimizer_test_passed = test_robust_optimizer_integration()

    # Test GUI components
    tester = ComponentTester()
    tester.run_all_tests()

    # Show test main window
    main_window = TestMainWindow()
    main_window.show()

    logger.info("Test window displayed. Interact with the GUI to test functionality.")
    logger.info("Close the window to complete the test.")

    # Run the application
    app.exec()

    # Final summary
    total_tests = len(tester.test_results) + 1  # +1 for optimizer test
    passed_tests = sum(1 for result in tester.test_results.values() if result == 'PASS')
    if optimizer_test_passed:
        passed_tests += 1

    logger.info(f"\n🏁 Final Test Summary: {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        logger.info("🎉 All tests passed! GUI integration is working correctly.")
        return 0
    else:
        logger.warning("⚠️ Some tests failed. Review the logs and fix issues before deployment.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)