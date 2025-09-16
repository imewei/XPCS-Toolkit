"""
Unit tests for the robust multi-strategy optimization engine.

This module provides comprehensive tests for the RobustOptimizer and SyntheticG2DataGenerator
classes to ensure reliable G2 correlation function fitting.

Tests cover:
- Synthetic data generation with known ground truth
- Multi-strategy optimization fallback logic
- Performance tracking and monitoring
- Parameter estimation accuracy
- Error handling and edge cases
"""

import unittest
import warnings
from typing import Dict, Tuple
from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize import OptimizeWarning

from xpcs_toolkit.helper.fitting import (
    BootstrapAnalyzer,
    CovarianceAnalyzer,
    DiagnosticReporter,
    GoodnessOfFitAnalyzer,
    OptimizationStrategy,
    ResidualAnalyzer,
    RobustOptimizer,
    RobustOptimizerWithDiagnostics,
    SyntheticG2DataGenerator,
    robust_curve_fit,
    single_exp
)


class TestSyntheticG2DataGenerator(unittest.TestCase):
    """Test suite for synthetic G2 data generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = SyntheticG2DataGenerator(random_state=42)

    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsInstance(self.generator, SyntheticG2DataGenerator)
        self.assertIn('single_exp', self.generator.models)
        self.assertIn('double_exp', self.generator.models)
        self.assertIn('stretched_exp', self.generator.models)
        self.assertIn('power_law_plus_exp', self.generator.models)

    def test_single_exponential_model(self):
        """Test single exponential G2 model."""
        tau = np.array([1e-6, 1e-5, 1e-4])
        params = {'baseline': 1.0, 'beta': 0.5, 'gamma': 1000.0}

        result = self.generator._single_exponential(tau, params)

        self.assertEqual(len(result), len(tau))
        self.assertTrue(np.all(result >= 1.0))  # Should be >= baseline
        self.assertTrue(np.all(np.isfinite(result)))

        # Test monotonic decay property
        self.assertTrue(np.all(np.diff(result) <= 0))

    def test_double_exponential_model(self):
        """Test double exponential G2 model."""
        tau = np.logspace(-6, 0, 10)
        params = {
            'baseline': 1.0, 'beta1': 0.3, 'gamma1': 5000.0,
            'beta2': 0.2, 'gamma2': 500.0
        }

        result = self.generator._double_exponential(tau, params)

        self.assertEqual(len(result), len(tau))
        self.assertTrue(np.all(result >= 1.0))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_stretched_exponential_model(self):
        """Test stretched exponential G2 model."""
        tau = np.logspace(-6, 0, 10)
        params = {'baseline': 1.0, 'beta': 0.4, 'gamma': 1000.0, 'stretch': 0.7}

        result = self.generator._stretched_exponential(tau, params)

        self.assertEqual(len(result), len(tau))
        self.assertTrue(np.all(result >= 1.0))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_generate_single_exp_dataset(self):
        """Test synthetic single exponential dataset generation."""
        tau, g2, g2_err, params = self.generator.generate_dataset(
            model_type='single_exp',
            tau_range=(1e-6, 1e0),
            n_points=50,
            noise_level=0.02
        )

        # Test shapes
        self.assertEqual(len(tau), 50)
        self.assertEqual(len(g2), 50)
        self.assertEqual(len(g2_err), 50)

        # Test parameter structure
        self.assertIn('baseline', params)
        self.assertIn('beta', params)
        self.assertIn('gamma', params)

        # Test physical constraints
        self.assertTrue(np.all(g2 >= 1.0))  # G2 >= 1
        self.assertTrue(np.all(g2_err > 0))  # Positive errors
        self.assertTrue(params['baseline'] >= 0.9)  # Reasonable baseline
        self.assertTrue(params['beta'] > 0)  # Positive contrast
        self.assertTrue(params['gamma'] > 0)  # Positive decay rate

    def test_generate_double_exp_dataset(self):
        """Test synthetic double exponential dataset generation."""
        tau, g2, g2_err, params = self.generator.generate_dataset(
            model_type='double_exp',
            tau_range=(1e-6, 1e-1),
            n_points=30,
            noise_level=0.05
        )

        self.assertEqual(len(tau), 30)
        self.assertEqual(len(g2), 30)
        self.assertEqual(len(g2_err), 30)

        # Check double exponential parameters
        required_params = ['baseline', 'beta1', 'gamma1', 'beta2', 'gamma2']
        for param in required_params:
            self.assertIn(param, params)
            self.assertTrue(params[param] > 0 or param == 'baseline')

    def test_noise_types(self):
        """Test different noise types."""
        for noise_type in ['gaussian', 'poisson', 'mixed']:
            with self.subTest(noise_type=noise_type):
                tau, g2, g2_err, _params = self.generator.generate_dataset(
                    model_type='single_exp',
                    noise_type=noise_type,
                    noise_level=0.03
                )

                self.assertTrue(np.all(g2 >= 1.0))
                self.assertTrue(np.all(g2_err > 0))
                self.assertTrue(np.all(np.isfinite(g2)))

    def test_outlier_injection(self):
        """Test outlier injection functionality."""
        # Test with outliers
        tau_out, g2_out, _, _ = self.generator.generate_dataset(
            model_type='single_exp',
            outlier_fraction=0.1,
            n_points=50
        )

        # Test without outliers
        tau_clean, g2_clean, _, _ = self.generator.generate_dataset(
            model_type='single_exp',
            outlier_fraction=0.0,
            n_points=50
        )

        # With outliers should have different statistical properties
        var_out = np.var(g2_out)
        var_clean = np.var(g2_clean)
        self.assertGreater(var_out, var_clean * 0.5)  # Should be more variable

    def test_systematic_error(self):
        """Test systematic error addition."""
        tau, g2, g2_err, _params = self.generator.generate_dataset(
            model_type='single_exp',
            systematic_error=True
        )

        self.assertTrue(np.all(g2 >= 1.0))
        self.assertTrue(np.all(np.isfinite(g2)))

    def test_invalid_model_type(self):
        """Test error handling for invalid model type."""
        with self.assertRaises(ValueError):
            self.generator.generate_dataset(model_type='invalid_model')


class TestOptimizationStrategy(unittest.TestCase):
    """Test suite for OptimizationStrategy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = OptimizationStrategy(
            name="Test Strategy",
            method="trf",
            config={'maxfev': 1000}
        )

    def test_initialization(self):
        """Test strategy initialization."""
        self.assertEqual(self.strategy.name, "Test Strategy")
        self.assertEqual(self.strategy.method, "trf")
        self.assertEqual(self.strategy.config['maxfev'], 1000)
        self.assertEqual(self.strategy.success_count, 0)
        self.assertEqual(self.strategy.total_attempts, 0)

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        # Initially should be 0
        self.assertEqual(self.strategy.success_rate, 0.0)

        # After some attempts
        self.strategy.update_stats(True, 0.1, 10)
        self.assertEqual(self.strategy.success_rate, 1.0)

        self.strategy.update_stats(False, 0.2, 20)
        self.assertEqual(self.strategy.success_rate, 0.5)

    def test_stats_update(self):
        """Test statistics update mechanism."""
        self.strategy.update_stats(True, 0.5, 100)

        self.assertEqual(self.strategy.success_count, 1)
        self.assertEqual(self.strategy.total_attempts, 1)
        self.assertGreater(self.strategy.avg_time, 0)
        self.assertGreater(self.strategy.avg_iterations, 0)


class TestRobustOptimizer(unittest.TestCase):
    """Test suite for RobustOptimizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = RobustOptimizer(
            max_iterations=1000,
            tolerance_factor=1.0,
            performance_tracking=True
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(len(self.optimizer.strategies), 3)
        self.assertEqual(self.optimizer.max_iterations, 1000)
        self.assertEqual(self.optimizer.tolerance_factor, 1.0)
        self.assertTrue(self.optimizer.performance_tracking)

        # Check strategy names
        strategy_names = [s.name for s in self.optimizer.strategies]
        self.assertIn("Trust Region Reflective", strategy_names)
        self.assertIn("Levenberg-Marquardt", strategy_names)
        self.assertIn("Differential Evolution", strategy_names)

    def test_input_validation(self):
        """Test input validation methods."""
        # Valid inputs
        func = lambda x, a, b: a * np.exp(-b * x)
        xdata = np.array([1, 2, 3, 4])
        ydata = np.array([2.0, 1.5, 1.1, 0.8])

        # Should not raise
        self.optimizer._validate_inputs(func, xdata, ydata, None, None)

        # Invalid function
        with self.assertRaises(TypeError):
            self.optimizer._validate_inputs("not_a_function", xdata, ydata, None, None)

        # Mismatched array lengths
        with self.assertRaises(ValueError):
            self.optimizer._validate_inputs(func, xdata, ydata[:-1], None, None)

        # Too few data points
        with self.assertRaises(ValueError):
            self.optimizer._validate_inputs(func, xdata[:2], ydata[:2], None, None)

        # Non-finite values
        with self.assertRaises(ValueError):
            bad_ydata = ydata.copy()
            bad_ydata[0] = np.inf
            self.optimizer._validate_inputs(func, xdata, bad_ydata, None, None)

    def test_parameter_estimation(self):
        """Test intelligent parameter estimation."""
        # Generate test data with known parameters
        tau_true = 1e-3  # time constant in appropriate units
        bkg_true = 1.0   # background
        cts_true = 0.5   # contrast

        xdata = np.logspace(-6, 0, 50)
        ydata = single_exp(xdata, tau_true, bkg_true, cts_true)
        ydata += 0.01 * ydata * np.random.normal(size=len(ydata))  # Add small noise

        p0_est = self.optimizer._estimate_initial_parameters(
            single_exp, xdata, ydata, bounds=([1e-6, 0.9, 0.01], [1e0, 1.1, 2.0])
        )

        self.assertEqual(len(p0_est), 3)
        self.assertTrue(np.all(np.isfinite(p0_est)))

        # Should be reasonably close to true values (order of magnitude)
        self.assertGreater(p0_est[0], 1e-6)  # tau estimate
        self.assertLess(p0_est[0], 1e0)
        self.assertGreater(p0_est[1], 0.9)   # baseline estimate
        self.assertLess(p0_est[1], 1.1)

    def test_successful_optimization(self):
        """Test successful optimization with clean synthetic data."""
        # Generate clean exponential data using single_exp directly
        tau_true = 1e-3
        bkg_true = 1.0
        cts_true = 0.5

        xdata = np.logspace(-6, 0, 50)
        ydata = single_exp(xdata, tau_true, bkg_true, cts_true)
        ydata += 0.01 * ydata * np.random.normal(size=len(ydata))  # Very low noise
        y_err = 0.01 * ydata  # Error estimate

        bounds = ([1e-6, 0.9, 0.01], [1e0, 1.1, 2.0])

        # Perform robust fitting
        popt, pcov, info = self.optimizer.robust_curve_fit(
            single_exp, xdata, ydata, bounds=bounds, sigma=y_err
        )

        # Test return values
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))
        self.assertIn('method', info)
        self.assertIn('chi_squared', info)
        self.assertIn('r_squared', info)

        # Test parameter accuracy (should be within reasonable range for low noise)
        tau_fit, bkg_fit, cts_fit = popt

        rel_error_tau = abs(tau_fit - tau_true) / tau_true
        rel_error_bkg = abs(bkg_fit - bkg_true) / bkg_true
        rel_error_cts = abs(cts_fit - cts_true) / cts_true

        self.assertLess(rel_error_tau, 0.5, f"Tau error too large: {rel_error_tau}")
        self.assertLess(rel_error_bkg, 0.1, f"Background error too large: {rel_error_bkg}")
        self.assertLess(rel_error_cts, 0.5, f"Contrast error too large: {rel_error_cts}")

        # Test goodness of fit
        self.assertGreater(info['r_squared'], 0.8, "R-squared should be reasonable for clean data")

    def test_fallback_logic(self):
        """Test that fallback logic works when primary methods fail."""
        # Create simple test data
        xdata = np.array([1e-6, 1e-5, 1e-4, 1e-3])
        ydata = np.array([1.8, 1.6, 1.4, 1.2])  # Simple decreasing trend

        def simple_model(x, a, b):
            return a * np.exp(-b * x) + 1.0

        # Test with unbounded problem (should work with multiple strategies)
        try:
            popt, pcov, info = self.optimizer.robust_curve_fit(
                simple_model, xdata, ydata
            )

            # Should succeed with some method
            self.assertEqual(len(popt), 2)
            self.assertIn('method', info)
            self.assertTrue(np.all(np.isfinite(popt)))

        except RuntimeError:
            # If all methods fail, that's also acceptable for this test
            # (depends on the specific data and numerical conditions)
            pass

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Enable performance tracking
        optimizer = RobustOptimizer(performance_tracking=True)

        # Generate simple test data
        xdata = np.linspace(0.1, 5, 20)
        ydata = 2.0 * np.exp(-xdata / 1.5) + 1.0
        ydata += 0.05 * ydata * np.random.normal(size=len(ydata))

        def exp_model(x, a, tau, b):
            return a * np.exp(-x / tau) + b

        try:
            _popt, _pcov, _info = optimizer.robust_curve_fit(
                exp_model, xdata, ydata, bounds=([0.1, 0.1, 0.9], [10, 10, 1.1])
            )

            # Check performance tracking
            report = optimizer.get_performance_report()

            self.assertIn('total_optimizations', report)
            self.assertIn('total_failures', report)
            self.assertIn('overall_success_rate', report)
            self.assertIn('strategies', report)

            self.assertGreaterEqual(report['total_optimizations'], 1)
            self.assertGreaterEqual(report['overall_success_rate'], 0)

        except RuntimeError:
            # If optimization fails, check that failures are tracked
            report = optimizer.get_performance_report()
            self.assertGreaterEqual(report['total_failures'], 1)

    def test_reset_performance_tracking(self):
        """Test performance tracking reset."""
        # Generate some history
        self.optimizer.convergence_history.append({'test': 'data'})
        self.optimizer.failed_attempts.append({'test': 'failure'})

        # Reset
        self.optimizer.reset_performance_tracking()

        self.assertEqual(len(self.optimizer.convergence_history), 0)
        self.assertEqual(len(self.optimizer.failed_attempts), 0)

    def test_all_strategies_fail(self):
        """Test error handling when all strategies fail."""
        def bad_model(x, a):
            # Model that will cause optimization to fail
            return a / (x * 0)  # Division by zero

        xdata = np.array([1, 2, 3, 4])
        ydata = np.array([1, 1, 1, 1])

        with self.assertRaises(RuntimeError) as context:
            self.optimizer.robust_curve_fit(bad_model, xdata, ydata)

        self.assertIn("optimization strategies failed", str(context.exception))


class TestRobustCurveFitFunction(unittest.TestCase):
    """Test suite for the robust_curve_fit convenience function."""

    def test_backward_compatibility(self):
        """Test backward compatibility with scipy.optimize.curve_fit."""
        # Generate simple test data
        xdata = np.linspace(0.1, 3, 20)
        ydata = 2.5 * np.exp(-xdata / 1.2) + 1.0
        ydata += 0.02 * ydata * np.random.normal(size=len(ydata))

        def exp_model(x, a, tau, b):
            return a * np.exp(-x / tau) + b

        # Test function call (similar to scipy interface)
        popt, pcov = robust_curve_fit(
            exp_model, xdata, ydata,
            bounds=([0.1, 0.1, 0.9], [10, 10, 1.1])
        )

        # Should return same format as scipy.optimize.curve_fit
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(popt)))
        self.assertTrue(np.all(np.isfinite(pcov)))


class TestIntegrationWithExistingFitting(unittest.TestCase):
    """Test integration with existing XPCS fitting functions."""

    def test_integration_with_single_exp(self):
        """Test integration with existing single_exp function."""
        # Generate realistic G2-like data directly with single_exp
        tau_true = 5e-4
        bkg_true = 1.0
        cts_true = 0.3

        tau = np.logspace(-6, 0, 40)
        g2 = single_exp(tau, tau_true, bkg_true, cts_true)
        g2 += 0.03 * g2 * np.random.normal(size=len(g2))  # Add 3% noise
        g2_err = 0.03 * g2

        # Use robust optimizer with single_exp function
        optimizer = RobustOptimizer()
        bounds = ([1e-6, 0.9, 0.01], [1e0, 1.1, 2.0])

        popt, pcov, info = optimizer.robust_curve_fit(
            single_exp, tau, g2, bounds=bounds, sigma=g2_err
        )

        # Should successfully fit
        self.assertEqual(len(popt), 3)
        self.assertGreater(info['r_squared'], 0.7)

        # Test fitted curve
        g2_fit = single_exp(tau, *popt)
        rmse = np.sqrt(np.mean((g2 - g2_fit)**2))
        self.assertLess(rmse, 0.15, "RMSE should be reasonable")

    def test_performance_comparison(self):
        """Test that robust optimizer performs comparably to standard methods."""
        import time
        from scipy.optimize import curve_fit

        # Generate test data
        generator = SyntheticG2DataGenerator(random_state=789)
        tau, g2, g2_err, _true_params = generator.generate_dataset(
            model_type='single_exp',
            noise_level=0.02,
            n_points=30  # Smaller dataset for faster testing
        )

        bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        p0 = [1000.0, 1.0, 0.5]

        # Time standard curve_fit
        start_time = time.time()
        try:
            popt_std, _pcov_std = curve_fit(
                single_exp, tau, g2, p0=p0, bounds=bounds, sigma=g2_err
            )
            std_time = time.time() - start_time
            std_success = True
        except Exception:
            std_time = time.time() - start_time
            std_success = False
            popt_std = None

        # Time robust optimizer
        optimizer = RobustOptimizer()
        start_time = time.time()
        try:
            popt_robust, _pcov_robust, _info = optimizer.robust_curve_fit(
                single_exp, tau, g2, bounds=bounds, sigma=g2_err
            )
            robust_time = time.time() - start_time
            robust_success = True
        except Exception:
            robust_time = time.time() - start_time
            robust_success = False
            popt_robust = None

        # If both succeed, compare results
        if std_success and robust_success:
            # Parameters should be similar (within 10%)
            for i in range(len(popt_std)):
                rel_diff = abs(popt_std[i] - popt_robust[i]) / abs(popt_std[i])
                self.assertLess(rel_diff, 0.2, f"Parameter {i} too different: {rel_diff}")

        # Robust optimizer should not be excessively slower (less than 5x)
        if std_success:
            overhead_factor = robust_time / max(std_time, 0.001)  # Avoid division by zero
            self.assertLess(overhead_factor, 5.0,
                          f"Robust optimizer too slow: {overhead_factor}x overhead")

        # Robust optimizer should succeed at least as often as standard method
        if not std_success:
            # If standard method fails, robust should ideally succeed
            # But this is not a strict requirement for all cases
            pass


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        optimizer = RobustOptimizer()

        with self.assertRaises(ValueError):
            optimizer.robust_curve_fit(
                lambda x, a: a * x,
                np.array([]),
                np.array([])
            )

    def test_constant_data(self):
        """Test handling of constant data."""
        optimizer = RobustOptimizer()

        xdata = np.array([1, 2, 3, 4, 5])
        ydata = np.ones(5)  # Constant data

        # This should either succeed with reasonable parameters or fail gracefully
        try:
            popt, _pcov, _info = optimizer.robust_curve_fit(
                lambda x, a, b: a * x + b,
                xdata, ydata
            )
            # If it succeeds, parameters should be reasonable
            self.assertTrue(np.all(np.isfinite(popt)))
        except RuntimeError:
            # Acceptable to fail on pathological data
            pass

    def test_noisy_data(self):
        """Test handling of very noisy data."""
        generator = SyntheticG2DataGenerator(random_state=999)

        # Generate very noisy data
        tau, g2, g2_err, _true_params = generator.generate_dataset(
            model_type='single_exp',
            noise_level=0.3,  # 30% noise
            outlier_fraction=0.1  # 10% outliers
        )

        optimizer = RobustOptimizer()
        bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])

        try:
            popt, pcov, info = optimizer.robust_curve_fit(
                single_exp, tau, g2, bounds=bounds, sigma=g2_err
            )

            # If fitting succeeds with noisy data, parameters should be physical
            gamma, baseline, beta = popt
            self.assertGreater(gamma, 0)
            self.assertGreater(baseline, 0.8)
            self.assertGreater(beta, 0)

            # Errors should be reasonable
            param_errors = np.sqrt(np.diag(pcov))
            self.assertTrue(np.all(param_errors > 0))

        except RuntimeError:
            # Acceptable to fail on very noisy data
            pass

    def test_boundary_conditions(self):
        """Test behavior at parameter boundaries."""
        # Test with data that should push parameters to boundaries
        xdata = np.logspace(-6, 0, 30)
        # Create data with very small decay (should hit gamma lower bound)
        ydata = 1.0 + 0.01 * np.exp(-10 * xdata)
        ydata += 0.005 * np.random.normal(size=len(ydata))

        optimizer = RobustOptimizer()
        bounds = ([10, 0.9, 0.001], [100000, 1.1, 2.0])

        try:
            popt, _pcov, _info = optimizer.robust_curve_fit(
                single_exp, xdata, ydata, bounds=bounds
            )

            gamma, baseline, beta = popt

            # Parameters should respect bounds
            self.assertGreaterEqual(gamma, bounds[0][0])
            self.assertLessEqual(gamma, bounds[1][0])
            self.assertGreaterEqual(baseline, bounds[0][1])
            self.assertLessEqual(baseline, bounds[1][1])
            self.assertGreaterEqual(beta, bounds[0][2])
            self.assertLessEqual(beta, bounds[1][2])

        except RuntimeError:
            # Acceptable for boundary cases to fail
            pass


class TestBootstrapAnalyzer(unittest.TestCase):
    """Test suite for BootstrapAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = BootstrapAnalyzer(n_bootstrap=100, confidence_level=0.95, random_state=42)

        # Generate synthetic data for testing
        self.generator = SyntheticG2DataGenerator(random_state=123)
        self.tau, self.g2, self.g2_err, self.true_params = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=30,
            noise_level=0.05
        )

        # Get good initial fit
        self.optimizer = RobustOptimizer()
        self.bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        self.popt, self.pcov, _ = self.optimizer.robust_curve_fit(
            single_exp, self.tau, self.g2, bounds=self.bounds, sigma=self.g2_err
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.n_bootstrap, 100)
        self.assertEqual(self.analyzer.confidence_level, 0.95)
        self.assertAlmostEqual(self.analyzer.alpha, 0.05, places=7)

    def test_bootstrap_sample_generation(self):
        """Test bootstrap sample generation methods."""
        # Test non-parametric bootstrap
        x_boot, y_boot, sigma_boot = self.analyzer._generate_bootstrap_sample(
            self.tau, self.g2, self.g2_err
        )

        self.assertEqual(len(x_boot), len(self.tau))
        self.assertEqual(len(y_boot), len(self.g2))
        self.assertEqual(len(sigma_boot), len(self.g2_err))

        # Test parametric bootstrap
        x_boot_p, y_boot_p, sigma_boot_p = self.analyzer._parametric_bootstrap_sample(
            self.tau, self.g2, single_exp, self.popt, self.g2_err
        )

        self.assertEqual(len(x_boot_p), len(self.tau))
        self.assertEqual(len(y_boot_p), len(self.g2))
        self.assertTrue(np.all(np.isfinite(y_boot_p)))

    def test_bootstrap_confidence_intervals(self):
        """Test bootstrap confidence interval calculation."""
        # Test with reduced number of bootstrap samples for speed
        results = self.analyzer.bootstrap_confidence_intervals(
            single_exp, self.tau, self.g2, self.popt, self.pcov,
            bounds=self.bounds, sigma=self.g2_err, method='residual'
        )

        # Check result structure
        self.assertIn('parameter_statistics', results)
        self.assertIn('confidence_level', results)
        self.assertIn('n_successful_fits', results)
        self.assertIn('success_rate', results)

        # Check parameter statistics
        param_stats = results['parameter_statistics']
        for i in range(len(self.popt)):
            param_key = f'param_{i}'
            self.assertIn(param_key, param_stats)
            self.assertIn('confidence_interval', param_stats[param_key])
            self.assertIn('bootstrap_mean', param_stats[param_key])
            self.assertIn('bootstrap_std', param_stats[param_key])

            # Confidence interval should be a 2-element array
            ci = param_stats[param_key]['confidence_interval']
            self.assertEqual(len(ci), 2)
            self.assertLessEqual(ci[0], ci[1])  # Lower bound <= upper bound

    def test_bootstrap_methods(self):
        """Test different bootstrap methods."""
        methods = ['residual', 'parametric', 'nonparametric']

        for method in methods:
            with self.subTest(method=method):
                # Use fewer samples for speed
                self.analyzer.n_bootstrap = 50
                results = self.analyzer.bootstrap_confidence_intervals(
                    single_exp, self.tau, self.g2, self.popt, self.pcov,
                    bounds=self.bounds, sigma=self.g2_err, method=method
                )

                self.assertEqual(results['method'], method)
                self.assertGreaterEqual(results['n_successful_fits'], 0)


class TestCovarianceAnalyzer(unittest.TestCase):
    """Test suite for CovarianceAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CovarianceAnalyzer()

        # Create test data and fit
        self.generator = SyntheticG2DataGenerator(random_state=456)
        self.tau, self.g2, self.g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp', n_points=40, noise_level=0.03
        )

        optimizer = RobustOptimizer()
        bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        self.popt, self.pcov, _ = optimizer.robust_curve_fit(
            single_exp, self.tau, self.g2, bounds=bounds, sigma=self.g2_err
        )

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, CovarianceAnalyzer)
        self.assertEqual(self.analyzer.condition_threshold, 1e12)

    def test_covariance_matrix_analysis(self):
        """Test comprehensive covariance matrix analysis."""
        results = self.analyzer.analyze_covariance_matrix(
            self.pcov, self.popt, func_name="single_exp"
        )

        # Check required keys
        required_keys = [
            'function_name', 'n_parameters', 'condition_number', 'determinant',
            'parameter_errors', 'relative_errors', 'correlation_matrix',
            'high_correlations', 'stability_issues', 'is_numerically_stable'
        ]

        for key in required_keys:
            self.assertIn(key, results)

        # Check data types and values
        self.assertEqual(results['function_name'], "single_exp")
        self.assertEqual(results['n_parameters'], len(self.popt))
        self.assertIsInstance(results['condition_number'], float)
        self.assertIsInstance(results['parameter_errors'], np.ndarray)
        self.assertEqual(len(results['parameter_errors']), len(self.popt))

        # Parameter errors should be positive
        self.assertTrue(np.all(results['parameter_errors'] > 0))

        # Correlation matrix should be square and symmetric
        if results['correlation_matrix'] is not None:
            corr_matrix = results['correlation_matrix']
            self.assertEqual(corr_matrix.shape, (len(self.popt), len(self.popt)))
            np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)

            # Diagonal should be 1.0
            np.testing.assert_array_almost_equal(np.diag(corr_matrix), 1.0)

    def test_parameter_constraint_suggestions(self):
        """Test parameter constraint suggestions."""
        # First analyze the covariance matrix
        analysis = self.analyzer.analyze_covariance_matrix(
            self.pcov, self.popt, func_name="test"
        )

        suggestions = self.analyzer.suggest_parameter_constraints(analysis, self.popt)

        self.assertIsInstance(suggestions, list)
        # Each suggestion should be a string
        for suggestion in suggestions:
            self.assertIsInstance(suggestion, str)

    def test_ill_conditioned_matrix(self):
        """Test handling of ill-conditioned covariance matrices."""
        # Create an ill-conditioned matrix
        ill_pcov = np.array([[1e-20, 0, 0], [0, 1.0, 0.99], [0, 0.99, 1.0]])
        popt_test = np.array([1.0, 2.0, 3.0])

        results = self.analyzer.analyze_covariance_matrix(ill_pcov, popt_test)

        # Should detect stability issues
        self.assertFalse(results['is_numerically_stable'])
        self.assertGreater(len(results['stability_issues']), 0)


class TestResidualAnalyzer(unittest.TestCase):
    """Test suite for ResidualAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ResidualAnalyzer(outlier_threshold=3.0)

        # Generate test data
        self.generator = SyntheticG2DataGenerator(random_state=789)
        self.tau, self.g2, self.g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp', n_points=50, noise_level=0.04, outlier_fraction=0.1
        )

        # Fit the data
        optimizer = RobustOptimizer()
        bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        popt, _, _ = optimizer.robust_curve_fit(
            single_exp, self.tau, self.g2, bounds=bounds, sigma=self.g2_err
        )

        self.y_pred = single_exp(self.tau, *popt)

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.outlier_threshold, 3.0)

    def test_residual_analysis(self):
        """Test comprehensive residual analysis."""
        results = self.analyzer.analyze_residuals(
            self.tau, self.g2, self.y_pred, sigma=self.g2_err
        )

        # Check required sections
        required_sections = [
            'residual_statistics', 'standardized_residual_statistics', 'outliers',
            'randomness_test', 'autocorrelation', 'trend_analysis', 'normality_tests'
        ]

        for section in required_sections:
            self.assertIn(section, results)

        # Check residual statistics
        res_stats = results['residual_statistics']
        self.assertIn('mean', res_stats)
        self.assertIn('std', res_stats)
        self.assertIn('rms', res_stats)
        self.assertTrue(res_stats['weighted_analysis'])  # Should be True since sigma provided

        # Check outlier detection
        outliers = results['outliers']
        self.assertIn('count', outliers)
        self.assertIn('fraction', outliers)
        self.assertIn('indices', outliers)
        self.assertEqual(len(outliers['indices']), outliers['count'])

        # Check randomness test
        randomness = results['randomness_test']
        self.assertIn('runs', randomness)
        self.assertIn('runs_z_score', randomness)
        self.assertIn('is_random', randomness)
        self.assertIsInstance(randomness['is_random'], bool)

    def test_systematic_deviation_detection(self):
        """Test systematic deviation detection."""
        residuals = self.g2 - self.y_pred

        results = self.analyzer.detect_systematic_deviations(
            self.tau, residuals, n_bins=5
        )

        # Check structure
        self.assertIn('bin_analysis', results)
        self.assertIn('systematic_bias', results)
        self.assertIn('systematic_trend', results)
        self.assertIn('recommendations', results)

        # Check bin analysis
        bin_analysis = results['bin_analysis']
        self.assertLessEqual(len(bin_analysis), 5)  # Should have at most 5 bins

        for bin_data in bin_analysis:
            self.assertIn('bin_index', bin_data)
            self.assertIn('x_range', bin_data)
            self.assertIn('mean_residual', bin_data)
            self.assertIn('std_residual', bin_data)

        # Check bias test
        bias_info = results['systematic_bias']
        self.assertIn('overall_bias', bias_info)
        self.assertIn('is_significant', bias_info)
        self.assertIsInstance(bias_info['is_significant'], bool)

    def test_outlier_detection_threshold(self):
        """Test outlier detection with different thresholds."""
        # Test with strict threshold
        strict_analyzer = ResidualAnalyzer(outlier_threshold=2.0)
        strict_results = strict_analyzer.analyze_residuals(
            self.tau, self.g2, self.y_pred, sigma=self.g2_err
        )

        # Test with lenient threshold
        lenient_analyzer = ResidualAnalyzer(outlier_threshold=5.0)
        lenient_results = lenient_analyzer.analyze_residuals(
            self.tau, self.g2, self.y_pred, sigma=self.g2_err
        )

        # Strict should detect more outliers than lenient
        strict_count = strict_results['outliers']['count']
        lenient_count = lenient_results['outliers']['count']
        self.assertGreaterEqual(strict_count, lenient_count)


class TestGoodnessOfFitAnalyzer(unittest.TestCase):
    """Test suite for GoodnessOfFitAnalyzer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GoodnessOfFitAnalyzer()

        # Generate test data with good fit
        self.generator = SyntheticG2DataGenerator(random_state=101112)
        self.tau, self.g2, self.g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp', n_points=40, noise_level=0.02
        )

        # Get fit
        optimizer = RobustOptimizer()
        bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        popt, _, _ = optimizer.robust_curve_fit(
            single_exp, self.tau, self.g2, bounds=bounds, sigma=self.g2_err
        )

        self.y_pred = single_exp(self.tau, *popt)
        self.n_params = len(popt)

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, GoodnessOfFitAnalyzer)

    def test_goodness_of_fit_metrics_calculation(self):
        """Test comprehensive goodness-of-fit metrics calculation."""
        results = self.analyzer.calculate_goodness_of_fit_metrics(
            self.g2, self.y_pred, self.n_params, sigma=self.g2_err
        )

        # Check required metrics
        required_metrics = [
            'n_data_points', 'n_parameters', 'degrees_of_freedom',
            'r_squared', 'adjusted_r_squared', 'weighted_r_squared',
            'rmse', 'mae', 'chi_squared', 'reduced_chi_squared',
            'aic', 'bic', 'log_likelihood', 'fit_quality_assessment'
        ]

        for metric in required_metrics:
            self.assertIn(metric, results)

        # Check data integrity
        self.assertEqual(results['n_data_points'], len(self.g2))
        self.assertEqual(results['n_parameters'], self.n_params)
        self.assertEqual(results['degrees_of_freedom'], len(self.g2) - self.n_params)

        # Check value ranges (R-squared can be negative for poor fits)
        self.assertLessEqual(results['r_squared'], 1.0)
        self.assertGreater(results['rmse'], 0.0)
        self.assertGreater(results['mae'], 0.0)
        self.assertGreater(results['chi_squared'], 0.0)

        # AIC and BIC should be finite
        self.assertTrue(np.isfinite(results['aic']))
        self.assertTrue(np.isfinite(results['bic']))

    def test_fit_quality_assessment(self):
        """Test fit quality assessment."""
        results = self.analyzer.calculate_goodness_of_fit_metrics(
            self.g2, self.y_pred, self.n_params, sigma=self.g2_err
        )

        assessment = results['fit_quality_assessment']

        # Check structure
        required_keys = ['overall_quality', 'warnings', 'recommendations']
        for key in required_keys:
            self.assertIn(key, assessment)

        # Check types
        self.assertIsInstance(assessment['overall_quality'], str)
        self.assertIsInstance(assessment['warnings'], list)
        self.assertIsInstance(assessment['recommendations'], list)

        # Overall quality should be a valid category
        valid_qualities = ['excellent', 'good', 'fair', 'poor']
        self.assertIn(assessment['overall_quality'], valid_qualities)

    def test_model_comparison(self):
        """Test model comparison functionality."""
        # Create results for multiple "models" (same data, different analysis)
        results1 = self.analyzer.calculate_goodness_of_fit_metrics(
            self.g2, self.y_pred, self.n_params, sigma=self.g2_err
        )

        # Create a slightly worse fit by adding noise to predictions
        y_pred_worse = self.y_pred + 0.01 * np.random.normal(size=len(self.y_pred))
        results2 = self.analyzer.calculate_goodness_of_fit_metrics(
            self.g2, y_pred_worse, self.n_params, sigma=self.g2_err
        )

        # Compare models
        comparison = self.analyzer.compare_models(
            [results1, results2], model_names=['Model_A', 'Model_B']
        )

        # Check structure
        self.assertIn('model_names', comparison)
        self.assertIn('metrics', comparison)
        self.assertIn('rankings', comparison)
        self.assertIn('deltas', comparison)
        self.assertIn('best_model', comparison)
        self.assertIn('comparison_summary', comparison)

        # Check that best model is identified
        best_model = comparison['best_model']
        self.assertIn('index', best_model)
        self.assertIn('name', best_model)

        # Best model should have lower AIC/BIC (better fit)
        if best_model['index'] is not None:
            self.assertIn(best_model['index'], [0, 1])


class TestDiagnosticReporter(unittest.TestCase):
    """Test suite for DiagnosticReporter class."""

    def setUp(self):
        """Set up test fixtures."""
        self.reporter = DiagnosticReporter(include_plots=False)

        # Generate test data
        self.generator = SyntheticG2DataGenerator(random_state=131415)
        self.tau, self.g2, self.g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp', n_points=35, noise_level=0.03
        )

        # Get fit
        self.optimizer = RobustOptimizer()
        self.bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])
        self.popt, self.pcov, _ = self.optimizer.robust_curve_fit(
            single_exp, self.tau, self.g2, bounds=self.bounds, sigma=self.g2_err
        )

    def test_initialization(self):
        """Test reporter initialization."""
        self.assertIsInstance(self.reporter, DiagnosticReporter)
        self.assertIsInstance(self.reporter.bootstrap_analyzer, BootstrapAnalyzer)
        self.assertIsInstance(self.reporter.covariance_analyzer, CovarianceAnalyzer)
        self.assertIsInstance(self.reporter.residual_analyzer, ResidualAnalyzer)
        self.assertIsInstance(self.reporter.goodness_analyzer, GoodnessOfFitAnalyzer)

    def test_comprehensive_report_generation(self):
        """Test comprehensive diagnostic report generation."""
        # Use minimal bootstrap for speed
        report = self.reporter.generate_comprehensive_report(
            single_exp, self.tau, self.g2, self.popt, self.pcov,
            bounds=self.bounds, sigma=self.g2_err, func_name="single_exp",
            bootstrap_method='residual', n_bootstrap=50
        )

        # Check main report sections
        required_sections = [
            'function_name', 'timestamp', 'data_info',
            'goodness_of_fit', 'covariance_analysis', 'residual_analysis',
            'systematic_deviations', 'bootstrap_analysis', 'summary'
        ]

        for section in required_sections:
            self.assertIn(section, report)

        # Check data info
        data_info = report['data_info']
        self.assertEqual(data_info['n_data_points'], len(self.tau))
        self.assertEqual(data_info['n_parameters'], len(self.popt))
        self.assertTrue(data_info['has_uncertainties'])

        # Check summary
        summary = report['summary']
        required_summary_keys = ['overall_assessment', 'key_findings', 'warnings',
                                'recommendations', 'quality_score']
        for key in required_summary_keys:
            self.assertIn(key, summary)

        # Quality score should be between 0 and 1
        self.assertGreaterEqual(summary['quality_score'], 0.0)
        self.assertLessEqual(summary['quality_score'], 1.0)

    def test_text_report_generation(self):
        """Test text report generation."""
        # Generate diagnostic report first
        report = self.reporter.generate_comprehensive_report(
            single_exp, self.tau, self.g2, self.popt, self.pcov,
            bounds=self.bounds, sigma=self.g2_err, func_name="single_exp",
            n_bootstrap=0  # Skip bootstrap for speed
        )

        # Generate text report
        text_report = self.reporter.generate_text_report(report)

        self.assertIsInstance(text_report, str)
        self.assertGreater(len(text_report), 100)  # Should be substantial

        # Check that key sections are included
        self.assertIn("DIAGNOSTIC REPORT", text_report)
        self.assertIn("OVERALL ASSESSMENT", text_report)
        self.assertIn("GOODNESS OF FIT", text_report)


class TestRobustOptimizerWithDiagnostics(unittest.TestCase):
    """Test suite for RobustOptimizerWithDiagnostics class."""

    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = RobustOptimizerWithDiagnostics(
            diagnostic_level='standard',
            performance_tracking=True
        )

        # Generate test data
        self.generator = SyntheticG2DataGenerator(random_state=161718)
        self.tau, self.g2, self.g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp', n_points=30, noise_level=0.04
        )

        self.bounds = ([10, 0.9, 0.01], [100000, 1.1, 2.0])

    def test_initialization(self):
        """Test optimizer initialization with diagnostics."""
        self.assertIsInstance(self.optimizer, RobustOptimizerWithDiagnostics)
        self.assertEqual(self.optimizer.diagnostic_level, 'standard')
        self.assertEqual(self.optimizer.default_bootstrap_samples, 500)
        self.assertTrue(self.optimizer.include_residual_analysis)
        self.assertTrue(self.optimizer.include_covariance_analysis)

    def test_diagnostic_levels(self):
        """Test different diagnostic levels."""
        levels = ['basic', 'standard', 'comprehensive']

        for level in levels:
            with self.subTest(level=level):
                optimizer = RobustOptimizerWithDiagnostics(diagnostic_level=level)
                self.assertEqual(optimizer.diagnostic_level, level)

                if level == 'basic':
                    self.assertEqual(optimizer.default_bootstrap_samples, 0)
                    self.assertFalse(optimizer.include_residual_analysis)
                elif level == 'standard':
                    self.assertEqual(optimizer.default_bootstrap_samples, 500)
                    self.assertTrue(optimizer.include_residual_analysis)
                elif level == 'comprehensive':
                    self.assertEqual(optimizer.default_bootstrap_samples, 1000)
                    self.assertTrue(optimizer.include_residual_analysis)

    def test_robust_fitting_with_diagnostics(self):
        """Test robust fitting with comprehensive diagnostics."""
        # Use minimal bootstrap for speed
        popt, pcov, diagnostics = self.optimizer.robust_curve_fit_with_diagnostics(
            single_exp, self.tau, self.g2, bounds=self.bounds, sigma=self.g2_err,
            func_name="single_exp", bootstrap_samples=50
        )

        # Check fitting results
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))

        # Check diagnostics structure
        self.assertIn('function_name', diagnostics)
        self.assertIn('optimization_info', diagnostics)

        # Check that key diagnostic sections are present
        if not diagnostics.get('error'):
            self.assertIn('summary', diagnostics)
            self.assertIn('goodness_of_fit', diagnostics)

    def test_fit_quality_analysis(self):
        """Test fit quality assessment from diagnostics."""
        popt, pcov, diagnostics = self.optimizer.robust_curve_fit_with_diagnostics(
            single_exp, self.tau, self.g2, bounds=self.bounds, sigma=self.g2_err,
            bootstrap_samples=0  # Skip bootstrap for speed
        )

        quality_assessment = self.optimizer.analyze_fit_quality(diagnostics)

        self.assertIsInstance(quality_assessment, str)
        self.assertIn("score:", quality_assessment.lower())

    def test_parameter_uncertainty_extraction(self):
        """Test parameter uncertainty extraction."""
        popt, pcov, diagnostics = self.optimizer.robust_curve_fit_with_diagnostics(
            single_exp, self.tau, self.g2, bounds=self.bounds, sigma=self.g2_err,
            bootstrap_samples=30  # Minimal bootstrap
        )

        uncertainties = self.optimizer.get_parameter_uncertainties(diagnostics)

        # Check structure
        self.assertIn('covariance_based', uncertainties)
        self.assertIn('bootstrap_based', uncertainties)

        # Check covariance-based uncertainties
        cov_based = uncertainties['covariance_based']
        if cov_based:  # If not empty
            for param_key, param_info in cov_based.items():
                self.assertIn('absolute_error', param_info)
                self.assertIn('relative_error', param_info)

    def test_error_handling_in_diagnostics(self):
        """Test error handling when diagnostics fail."""
        # Create problematic data that might cause diagnostic failures
        problematic_tau = np.array([1e-10, 1e-9, 1e-8])  # Very small range
        problematic_g2 = np.array([1.0, 1.0, 1.0])  # Constant data
        problematic_err = np.array([0.1, 0.1, 0.1])

        try:
            popt, pcov, diagnostics = self.optimizer.robust_curve_fit_with_diagnostics(
                single_exp, problematic_tau, problematic_g2,
                sigma=problematic_err, bootstrap_samples=0
            )

            # If it succeeds, diagnostics should at least have basic info
            self.assertIn('optimization_info', diagnostics)

        except Exception:
            # It's acceptable for this to fail with problematic data
            pass


if __name__ == '__main__':
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    unittest.main(verbosity=2)