"""
Comprehensive tests for robust fitting integration in XPCS Toolkit.

Tests backward compatibility, performance optimizations, and integration
with existing XPCS analysis workflows.
"""

import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from xpcs_toolkit.helper.fitting import (
    ComprehensiveDiffusionAnalyzer,
    OptimizedXPCSFittingEngine,
    RobustOptimizer,
    XPCSPerformanceOptimizer,
    fit_with_fixed,
    robust_curve_fit,
)
from xpcs_toolkit.xpcs_file import XpcsFile


class TestBackwardCompatibility(unittest.TestCase):
    """Test that all existing APIs remain functional."""

    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic XPCS data
        self.tau = np.logspace(-6, 0, 50)  # Time delays
        self.n_q = 10  # Number of q-values
        self.q_vals = np.linspace(0.001, 0.01, self.n_q)

        # Generate synthetic G2 data with single exponential decay
        self.true_params = [1.0, 0.01, 0.1, 1.5]  # [tau, baseline, amplitude, scaling]
        self.g2_data = np.zeros((len(self.tau), self.n_q))
        self.g2_errors = np.zeros_like(self.g2_data)

        for i, q in enumerate(self.q_vals):
            # G2 = baseline + amplitude * exp(-2*t/tau)
            tau_decay = self.true_params[0] / (q**2)  # Diffusive scaling
            self.g2_data[:, i] = (
                self.true_params[1] +
                self.true_params[2] * np.exp(-2 * self.tau / tau_decay)
            )
            self.g2_errors[:, i] = 0.01 * np.ones_like(self.tau)

        # Add realistic noise
        np.random.seed(42)
        noise = np.random.normal(0, 0.001, self.g2_data.shape)
        self.g2_data += noise

    def test_fit_with_fixed_compatibility(self):
        """Test that fit_with_fixed maintains its original behavior."""
        # Define bounds for single exponential
        bounds = ([0.001, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 2.0])
        fit_flag = [True, True, True, True]

        # Generate fit_x
        fit_x = np.logspace(np.log10(np.min(self.tau)) - 0.5,
                           np.log10(np.max(self.tau)) + 0.5, 128)

        # Initial parameter guess
        p0 = [0.01, 0.1, 0.2, 1.5]

        # Mock the single_exp_all function
        def mock_single_exp(x, params, flags):
            return params[1] + params[2] * np.exp(-2 * x / params[0])

        # Test fit_with_fixed
        with patch('xpcs_toolkit.helper.fitting.single_exp_all', side_effect=mock_single_exp):
            fit_line, fit_val = fit_with_fixed(
                mock_single_exp, self.tau, self.g2_data, self.g2_errors,
                bounds, fit_flag, fit_x, p0=p0
            )

        # Verify outputs have expected structure
        self.assertEqual(fit_line.shape[0], self.n_q)
        self.assertEqual(fit_line.shape[1], len(fit_x))
        self.assertEqual(len(fit_val), self.n_q)

        # Check that each fit result has expected keys
        for result in fit_val:
            if isinstance(result, dict):
                self.assertIn('params', result.keys() | {'error'})

    def test_robust_curve_fit_drop_in_replacement(self):
        """Test that robust_curve_fit works as a scipy.curve_fit replacement."""
        # Simple exponential function for testing
        def exp_func(x, a, b):
            return a * np.exp(-b * x)

        # Generate test data
        x_test = np.linspace(0, 2, 20)
        y_true = exp_func(x_test, 2.0, 1.0)
        y_noisy = y_true + np.random.normal(0, 0.1, len(x_test))

        # Test robust_curve_fit
        popt, pcov = robust_curve_fit(exp_func, x_test, y_noisy, p0=[1.0, 0.5])

        # Verify outputs
        self.assertEqual(len(popt), 2)
        self.assertEqual(pcov.shape, (2, 2))
        self.assertTrue(np.allclose(popt, [2.0, 1.0], atol=0.5))

    def test_xpcs_file_fit_g2_unchanged_behavior(self):
        """Test that XpcsFile.fit_g2 maintains its original behavior when not using robust fitting."""
        # Create a mock XpcsFile
        with patch.object(XpcsFile, 'get_g2_data') as mock_get_data:
            mock_get_data.return_value = (
                self.q_vals, self.tau, self.g2_data, self.g2_errors,
                [f'q_{i}' for i in range(self.n_q)]
            )

            # Mock other required methods
            with patch.object(XpcsFile, '_generate_cache_key') as mock_cache_key:
                mock_cache_key.return_value = 'test_key'

                with patch.object(XpcsFile, '__init__', return_value=None):
                    xf = XpcsFile()
                    xf._computation_cache = {}

                    # Test bounds
                    bounds = ([0.001, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 2.0])

                    # Mock fit_with_fixed
                    mock_fit_line = np.random.random((self.n_q, 128))
                    mock_fit_val = [{'params': [0.01, 0.1, 0.2, 1.5]} for _ in range(self.n_q)]

                    with patch('xpcs_toolkit.xpcs_file.fit_with_fixed') as mock_fit:
                        mock_fit.return_value = (mock_fit_line, mock_fit_val)

                        # Call fit_g2 without robust fitting (should use original path)
                        result = xf.fit_g2(bounds=bounds, robust_fitting=False)

                        # Verify structure is maintained
                        self.assertIn('fit_func', result)
                        self.assertIn('fit_val', result)
                        self.assertIn('fit_line', result)
                        self.assertIn('q_val', result)
                        self.assertIn('t_el', result)

                        # Verify fit_with_fixed was called (original behavior)
                        mock_fit.assert_called_once()


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations for large datasets."""

    def setUp(self):
        """Set up performance test fixtures."""
        self.performance_optimizer = XPCSPerformanceOptimizer(max_memory_mb=1024)
        self.fitting_engine = OptimizedXPCSFittingEngine(self.performance_optimizer)

    def test_memory_estimation(self):
        """Test memory usage estimation."""
        tau_len = 100
        g2_shape = (100, 50)  # 100 time points, 50 q-values
        n_bootstrap = 1000

        memory_estimate = self.performance_optimizer.estimate_memory_usage(
            tau_len, g2_shape, n_bootstrap
        )

        # Should be a positive number
        self.assertGreater(memory_estimate, 0)
        # Should scale with bootstrap samples
        memory_no_bootstrap = self.performance_optimizer.estimate_memory_usage(
            tau_len, g2_shape, 0
        )
        self.assertGreater(memory_estimate, memory_no_bootstrap)

    def test_chunking_decision(self):
        """Test chunked processing decision logic."""
        # Small dataset should not use chunking
        use_chunking, chunk_size = self.performance_optimizer.should_use_chunked_processing(
            50, (50, 10), 0
        )
        self.assertFalse(use_chunking)

        # Large dataset should use chunking
        use_chunking, chunk_size = self.performance_optimizer.should_use_chunked_processing(
            1000, (1000, 500), 1000  # Large dataset with bootstrap
        )
        self.assertTrue(use_chunking)
        self.assertGreater(chunk_size, 0)

    def test_parallelization_optimization(self):
        """Test parallelization decision logic."""
        n_jobs = self.performance_optimizer.optimize_parallelization(20, 0)
        self.assertGreater(n_jobs, 0)
        self.assertLessEqual(n_jobs, multiprocessing.cpu_count())

        # Bootstrap should reduce parallelization
        n_jobs_bootstrap = self.performance_optimizer.optimize_parallelization(20, 1000)
        self.assertLessEqual(n_jobs_bootstrap, n_jobs)

    def test_performance_overhead_target(self):
        """Test that performance overhead stays within acceptable limits."""
        # Create moderate-sized synthetic dataset
        tau = np.logspace(-6, 0, 50)
        g2_data = np.random.random((50, 20))  # 20 q-values
        g2_errors = 0.01 * np.ones_like(g2_data)
        q_values = np.linspace(0.001, 0.01, 20)

        # Time the optimized fitting
        start_time = time.time()
        results = self.fitting_engine.fit_g2_optimized(
            tau=tau,
            g2_data=g2_data,
            g2_errors=g2_errors,
            q_values=q_values,
            bootstrap_samples=0,  # No bootstrap for speed
            enable_diagnostics=False
        )
        actual_time = time.time() - start_time

        # Verify overhead calculation
        calculated_overhead = results['optimization_summary']['performance_overhead_percent']
        self.assertIsInstance(calculated_overhead, (int, float))
        self.assertGreaterEqual(calculated_overhead, 0)

        # For this test size, overhead should be reasonable
        # (This is a soft constraint, actual values may vary)
        self.assertLess(calculated_overhead, 500)  # Less than 500% overhead


class TestCachingAndMemoryIntegration(unittest.TestCase):
    """Test integration with existing caching and memory management systems."""

    def setUp(self):
        """Set up caching test fixtures."""
        self.performance_optimizer = XPCSPerformanceOptimizer(cache_size=100)

    def test_cache_integration(self):
        """Test that caching works with robust fitting."""
        # This test would be more meaningful with actual cache implementation
        self.assertIsInstance(self.performance_optimizer.cache, dict)
        self.assertEqual(len(self.performance_optimizer.cache), 0)

    @patch('xpcs_toolkit.xpcs_file.MemoryMonitor.is_memory_pressure_high')
    def test_memory_pressure_response(self, mock_memory_pressure):
        """Test response to memory pressure conditions."""
        # Simulate high memory pressure
        mock_memory_pressure.return_value = True

        # Test that the system should switch to more conservative settings
        optimizer = XPCSPerformanceOptimizer(max_memory_mb=512)  # Lower threshold

        use_chunking, chunk_size = optimizer.should_use_chunked_processing(
            1000, (1000, 100), 0
        )

        # Should use chunking for large datasets
        self.assertTrue(use_chunking or chunk_size < 100)


class TestProductionReadiness(unittest.TestCase):
    """Test production readiness with realistic scenarios."""

    def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios."""
        optimizer = ComprehensiveDiffusionAnalyzer(diagnostic_level='basic')

        # Test with invalid data
        with self.assertRaises((ValueError, RuntimeError)):
            optimizer.analyze_diffusion(
                tau=np.array([]),  # Empty array
                g2_data=np.array([1, 2, 3]),
                q_value=0.001
            )

    def test_parameter_validation(self):
        """Test input parameter validation."""
        engine = OptimizedXPCSFittingEngine()

        # Test with mismatched array sizes
        tau = np.linspace(0, 1, 10)
        g2_data = np.random.random((5, 3))  # Wrong tau dimension

        with self.assertRaises(ValueError):
            results = engine.fit_g2_optimized(tau=tau, g2_data=g2_data)

    def test_large_dataset_handling(self):
        """Test handling of realistically large XPCS datasets."""
        # Create a larger synthetic dataset
        tau = np.logspace(-6, 0, 200)  # 200 time points
        n_q = 100  # 100 q-values
        g2_data = np.random.random((200, n_q)) + 1.0  # Realistic G2 values
        g2_errors = 0.01 * np.ones_like(g2_data)
        q_values = np.linspace(0.0001, 0.1, n_q)

        engine = OptimizedXPCSFittingEngine()

        # Should complete without errors
        results = engine.fit_g2_optimized(
            tau=tau,
            g2_data=g2_data,
            g2_errors=g2_errors,
            q_values=q_values,
            bootstrap_samples=0,  # Skip bootstrap for speed
            enable_diagnostics=False
        )

        # Verify results structure
        self.assertIn('fit_results', results)
        self.assertIn('performance_info', results)
        self.assertEqual(len(results['fit_results']), n_q)

        # Check timing is reasonable (< 60 seconds for this size)
        self.assertLess(results['timing']['total_time'], 60.0)


class TestAPIConsistency(unittest.TestCase):
    """Test API consistency across different fitting methods."""

    def test_consistent_return_structures(self):
        """Test that all fitting methods return consistent data structures."""
        # This would test that fit_g2, fit_g2_robust, and fit_g2_high_performance
        # all return compatible dictionary structures
        pass  # Placeholder for consistency tests

    def test_parameter_name_consistency(self):
        """Test that parameter names are consistent across methods."""
        # Check that all methods use the same parameter names
        pass  # Placeholder for parameter consistency tests


if __name__ == '__main__':
    # Configure test environment
    if 'PYTEST_CURRENT_TEST' not in os.environ:
        # Run with unittest
        unittest.main(verbosity=2)
    else:
        # Run with pytest
        pytest.main([__file__, '-v'])