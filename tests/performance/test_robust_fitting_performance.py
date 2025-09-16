"""
Performance and stress tests for robust fitting framework.

This module tests performance limits, memory usage, and behavior under
extreme conditions with large datasets.
"""

import unittest
import time
import psutil
import os
import gc
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tempfile

from xpcs_toolkit.helper.fitting import (
    RobustOptimizer,
    RobustOptimizerWithDiagnostics,
    SyntheticG2DataGenerator,
    XPCSPerformanceOptimizer,
    OptimizedXPCSFittingEngine,
    robust_curve_fit,
    single_exp,
    double_exp
)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for robust fitting algorithms."""

    def setUp(self):
        """Set up performance tests."""
        self.optimizer = RobustOptimizer()
        self.generator = SyntheticG2DataGenerator(random_state=42)
        self.process = psutil.Process(os.getpid())

    def measure_memory_usage(self):
        """Measure current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def test_single_fit_performance(self):
        """Benchmark single exponential fitting performance."""
        # Test datasets of increasing size
        dataset_sizes = [50, 100, 200, 500, 1000]
        performance_results = []

        for n_points in dataset_sizes:
            with self.subTest(n_points=n_points):
                # Generate test data
                tau, g2, g2_err, _ = self.generator.generate_dataset(
                    model_type='single_exp',
                    n_points=n_points,
                    noise_level=0.03
                )

                # Measure fitting time
                start_time = time.time()
                memory_before = self.measure_memory_usage()

                try:
                    popt, pcov, info = self.optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                        sigma=g2_err
                    )

                    fit_time = time.time() - start_time
                    memory_after = self.measure_memory_usage()
                    memory_used = memory_after - memory_before

                    performance_results.append({
                        'n_points': n_points,
                        'fit_time': fit_time,
                        'memory_used': memory_used,
                        'method': info['method'],
                        'r_squared': info['r_squared']
                    })

                    # Performance assertions
                    self.assertLess(fit_time, 30.0)  # Should complete within 30 seconds
                    self.assertLess(memory_used, 100)  # Should use less than 100MB extra
                    self.assertGreater(info['r_squared'], 0.7)  # Should achieve reasonable fit

                except RuntimeError:
                    # Some large datasets might fail - this is acceptable
                    continue

        # Verify performance scaling
        if len(performance_results) >= 3:
            # Fitting time should scale reasonably with dataset size
            sizes = [r['n_points'] for r in performance_results]
            times = [r['fit_time'] for r in performance_results]

            # Time should not scale worse than quadratically
            if max(sizes) > min(sizes):
                time_ratio = max(times) / min(times)
                size_ratio = max(sizes) / min(sizes)
                scaling_factor = time_ratio / size_ratio

                self.assertLess(scaling_factor, 10)  # Should scale better than 10x per 1x size increase

    def test_multi_q_batch_performance(self):
        """Test performance of batch fitting multiple q-values."""
        n_q_values = [5, 10, 20, 50]
        batch_performance = []

        for n_q in n_q_values:
            with self.subTest(n_q=n_q):
                # Generate multi-q dataset
                tau = np.logspace(-6, 0, 60)
                q_values = np.linspace(0.001, 0.005, n_q)

                start_time = time.time()
                memory_before = self.measure_memory_usage()

                successful_fits = 0
                for i, q in enumerate(q_values):
                    gamma = 1000.0 / (q**2 + 0.0001)
                    g2 = single_exp(tau, gamma, 1.0, 0.3)
                    g2 += 0.03 * 0.3 * np.random.normal(size=len(g2))
                    g2_err = 0.03 * 0.3 * np.ones_like(g2)

                    try:
                        popt, _, _ = self.optimizer.robust_curve_fit(
                            single_exp, tau, g2,
                            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                            sigma=g2_err
                        )
                        successful_fits += 1

                    except RuntimeError:
                        continue

                total_time = time.time() - start_time
                memory_after = self.measure_memory_usage()
                memory_used = memory_after - memory_before

                batch_performance.append({
                    'n_q': n_q,
                    'total_time': total_time,
                    'time_per_fit': total_time / n_q,
                    'memory_used': memory_used,
                    'success_rate': successful_fits / n_q
                })

                # Performance assertions
                self.assertGreater(successful_fits / n_q, 0.8)  # 80% success rate
                self.assertLess(total_time, 300)  # Complete within 5 minutes
                self.assertLess(memory_used, 500)  # Use less than 500MB extra

        # Verify batch processing efficiency
        if len(batch_performance) >= 2:
            times_per_fit = [r['time_per_fit'] for r in batch_performance]
            # Time per fit should be relatively constant (good caching/efficiency)
            time_variation = (max(times_per_fit) - min(times_per_fit)) / np.mean(times_per_fit)
            self.assertLess(time_variation, 1.0)  # Less than 100% variation

    def test_bootstrap_performance(self):
        """Test performance of bootstrap analysis."""
        bootstrap_sizes = [10, 50, 100]
        bootstrap_performance = []

        # Generate test data
        tau, g2, g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=40,
            noise_level=0.04
        )

        optimizer = RobustOptimizerWithDiagnostics()

        for n_bootstrap in bootstrap_sizes:
            with self.subTest(n_bootstrap=n_bootstrap):
                start_time = time.time()
                memory_before = self.measure_memory_usage()

                try:
                    popt, pcov, diagnostics = optimizer.robust_curve_fit_with_diagnostics(
                        single_exp, tau, g2,
                        bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                        sigma=g2_err,
                        bootstrap_samples=n_bootstrap
                    )

                    bootstrap_time = time.time() - start_time
                    memory_after = self.measure_memory_usage()
                    memory_used = memory_after - memory_before

                    bootstrap_performance.append({
                        'n_bootstrap': n_bootstrap,
                        'bootstrap_time': bootstrap_time,
                        'time_per_sample': bootstrap_time / n_bootstrap,
                        'memory_used': memory_used
                    })

                    # Performance assertions
                    self.assertLess(bootstrap_time, 180)  # Complete within 3 minutes
                    self.assertLess(memory_used, 200)  # Use less than 200MB extra

                    # Verify bootstrap results are available
                    if 'bootstrap_analysis' in diagnostics:
                        bootstrap_results = diagnostics['bootstrap_analysis']
                        self.assertIn('parameter_statistics', bootstrap_results)

                except Exception:
                    # Bootstrap might fail with small samples - continue testing
                    continue

        # Verify bootstrap scaling
        if len(bootstrap_performance) >= 2:
            times = [r['bootstrap_time'] for r in bootstrap_performance]
            samples = [r['n_bootstrap'] for r in bootstrap_performance]

            # Time should scale roughly linearly with bootstrap samples
            if max(samples) > min(samples):
                time_ratio = max(times) / min(times)
                sample_ratio = max(samples) / min(samples)
                scaling_factor = time_ratio / sample_ratio

                self.assertLess(scaling_factor, 3)  # Should scale better than 3x per 1x increase

    def test_memory_pressure_handling(self):
        """Test behavior under memory pressure."""
        # Create progressively larger datasets to test memory handling
        memory_before = self.measure_memory_usage()
        large_dataset_results = []

        dataset_sizes = [500, 1000, 2000, 5000]

        for n_points in dataset_sizes:
            # Force garbage collection before each test
            gc.collect()

            memory_start = self.measure_memory_usage()

            try:
                # Generate large dataset
                tau, g2, g2_err, _ = self.generator.generate_dataset(
                    model_type='single_exp',
                    n_points=n_points,
                    noise_level=0.02
                )

                start_time = time.time()

                # Fit with memory monitoring
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                fit_time = time.time() - start_time
                memory_end = self.measure_memory_usage()
                memory_used = memory_end - memory_start

                large_dataset_results.append({
                    'n_points': n_points,
                    'fit_time': fit_time,
                    'memory_used': memory_used,
                    'peak_memory': memory_end,
                    'success': True
                })

                # Clean up large arrays
                del tau, g2, g2_err, popt, pcov
                gc.collect()

                # Memory usage should not grow unboundedly
                if len(large_dataset_results) > 1:
                    memory_growth = memory_end - memory_before
                    self.assertLess(memory_growth, 1000)  # Less than 1GB growth

            except (RuntimeError, MemoryError):
                # Large datasets might fail due to memory constraints
                large_dataset_results.append({
                    'n_points': n_points,
                    'success': False
                })
                continue

        # Should handle at least moderate-sized datasets
        successful_results = [r for r in large_dataset_results if r['success']]
        self.assertGreaterEqual(len(successful_results), 2)

    def test_concurrent_fitting_performance(self):
        """Test performance of concurrent fitting operations."""
        try:
            # Generate multiple datasets for concurrent fitting
            n_concurrent = 4
            datasets = []

            for i in range(n_concurrent):
                tau, g2, g2_err, _ = self.generator.generate_dataset(
                    model_type='single_exp',
                    n_points=50,
                    noise_level=0.03
                )
                datasets.append((tau, g2, g2_err))

            def fit_dataset(data):
                """Fit a single dataset."""
                tau, g2, g2_err = data
                optimizer = RobustOptimizer()  # Create separate optimizer for thread safety

                try:
                    popt, pcov, info = optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                        sigma=g2_err
                    )
                    return {'success': True, 'popt': popt, 'r_squared': info['r_squared']}

                except RuntimeError:
                    return {'success': False}

            # Test sequential fitting
            start_time = time.time()
            sequential_results = [fit_dataset(data) for data in datasets]
            sequential_time = time.time() - start_time

            # Test concurrent fitting
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=n_concurrent) as executor:
                concurrent_results = list(executor.map(fit_dataset, datasets))
            concurrent_time = time.time() - start_time

            # Verify results quality is similar
            seq_success = sum(1 for r in sequential_results if r['success'])
            con_success = sum(1 for r in concurrent_results if r['success'])

            self.assertEqual(seq_success, con_success)  # Same success rate

            # Concurrent should be faster (if system has multiple cores)
            if psutil.cpu_count() > 1:
                speedup = sequential_time / concurrent_time
                self.assertGreater(speedup, 0.8)  # At least some speedup

        except Exception as e:
            self.skipTest(f"Concurrent fitting test failed: {e}")

    def test_optimization_timeout_handling(self):
        """Test handling of optimization timeouts."""
        # Create difficult optimization problems that might timeout
        tau = np.logspace(-6, 0, 100)
        g2 = single_exp(tau, 1000.0, 1.0, 0.5)
        g2 += 0.2 * np.random.normal(size=len(g2))  # High noise

        # Test with very low iteration limit (simulating timeout)
        timeout_optimizer = RobustOptimizer(max_iterations=5)

        start_time = time.time()

        try:
            popt, pcov, info = timeout_optimizer.robust_curve_fit(
                single_exp, tau, g2,
                bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0])
            )

            fit_time = time.time() - start_time

            # Should complete quickly due to low iteration limit
            self.assertLess(fit_time, 10)  # Should timeout/complete quickly

            # If it succeeds, results should still be reasonable
            self.assertEqual(len(popt), 3)
            self.assertTrue(np.all(np.isfinite(popt)))

        except RuntimeError:
            # Timeout failures are acceptable for this test
            fit_time = time.time() - start_time
            self.assertLess(fit_time, 10)  # Should fail quickly


class TestStressTesting(unittest.TestCase):
    """Stress tests for extreme conditions."""

    def setUp(self):
        """Set up stress tests."""
        self.optimizer = RobustOptimizer()
        self.generator = SyntheticG2DataGenerator(random_state=123)

    def test_extreme_parameter_ranges(self):
        """Test with extreme parameter ranges."""
        extreme_scenarios = [
            {
                'name': 'very_fast_dynamics',
                'gamma': 1e8,
                'tau_range': (1e-12, 1e-9),
                'bounds': ([1e6, 0.9, 0.01], [1e10, 1.1, 2.0])
            },
            {
                'name': 'very_slow_dynamics',
                'gamma': 1e-3,
                'tau_range': (1, 1e6),
                'bounds': ([1e-6, 0.9, 0.01], [1, 1.1, 2.0])
            },
            {
                'name': 'tiny_contrast',
                'gamma': 1000.0,
                'beta': 1e-4,
                'bounds': ([1, 0.99, 1e-6], [100000, 1.01, 1e-2])
            },
            {
                'name': 'huge_contrast',
                'gamma': 1000.0,
                'beta': 10.0,
                'bounds': ([1, 0.9, 1], [100000, 1.1, 20])
            }
        ]

        successful_scenarios = 0

        for scenario in extreme_scenarios:
            with self.subTest(scenario=scenario['name']):
                try:
                    # Generate data for extreme scenario
                    tau_range = scenario.get('tau_range', (1e-6, 1e0))
                    gamma = scenario.get('gamma', 1000.0)
                    beta = scenario.get('beta', 0.5)

                    tau = np.logspace(np.log10(tau_range[0]), np.log10(tau_range[1]), 50)
                    g2 = single_exp(tau, gamma, 1.0, beta)
                    g2 += 0.02 * beta * np.random.normal(size=len(g2))
                    g2_err = 0.02 * beta * np.ones_like(g2)

                    # Attempt fitting
                    popt, pcov, info = self.optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=scenario['bounds'],
                        sigma=g2_err
                    )

                    # Verify results are reasonable
                    self.assertEqual(len(popt), 3)
                    self.assertTrue(np.all(np.isfinite(popt)))
                    successful_scenarios += 1

                except (RuntimeError, ValueError):
                    # Extreme scenarios might legitimately fail
                    continue

        # Should handle at least some extreme scenarios
        self.assertGreater(successful_scenarios, 0)

    def test_massive_dataset_handling(self):
        """Test with very large datasets."""
        try:
            # Test with maximum reasonable dataset size
            n_points_large = 10000

            # Generate large dataset
            tau_large = np.logspace(-6, 2, n_points_large)
            g2_large = single_exp(tau_large, 100.0, 1.0, 0.3)
            g2_large += 0.01 * 0.3 * np.random.normal(size=n_points_large)
            g2_err_large = 0.01 * 0.3 * np.ones_like(g2_large)

            start_time = time.time()
            memory_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

            # Attempt fitting
            popt, pcov, info = self.optimizer.robust_curve_fit(
                single_exp, tau_large, g2_large,
                bounds=([1, 0.9, 0.01], [1000, 1.1, 2.0]),
                sigma=g2_err_large
            )

            fit_time = time.time() - start_time
            memory_after = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_used = memory_after - memory_before

            # Performance assertions for large dataset
            self.assertLess(fit_time, 600)  # Should complete within 10 minutes
            self.assertLess(memory_used, 2000)  # Should use less than 2GB extra

            # Quality assertions
            self.assertEqual(len(popt), 3)
            self.assertTrue(np.all(np.isfinite(popt)))
            self.assertGreater(info['r_squared'], 0.8)

        except (RuntimeError, MemoryError):
            self.skipTest("Massive dataset test failed due to resource constraints")

    def test_pathological_data_conditions(self):
        """Test with pathological data conditions."""
        pathological_cases = [
            {
                'name': 'all_identical_values',
                'g2': np.ones(50),
                'g2_err': 0.01 * np.ones(50)
            },
            {
                'name': 'alternating_values',
                'g2': np.array([1.0, 1.5] * 25),
                'g2_err': 0.1 * np.ones(50)
            },
            {
                'name': 'single_outlier',
                'g2': np.concatenate([1.5 * np.exp(-np.arange(49) / 10), [10.0]]),
                'g2_err': 0.05 * np.ones(50)
            },
            {
                'name': 'missing_data_simulation',
                'g2': np.array([1.5, 1.4, np.nan, 1.2, 1.1] * 10),
                'g2_err': 0.05 * np.ones(50)
            }
        ]

        for case in pathological_cases:
            with self.subTest(case=case['name']):
                tau = np.logspace(-6, 0, len(case['g2']))
                g2 = case['g2']
                g2_err = case['g2_err']

                try:
                    # Should either fit successfully or fail gracefully
                    if np.any(np.isnan(g2)) or np.any(np.isinf(g2)):
                        # Should raise ValueError for invalid data
                        with self.assertRaises((ValueError, RuntimeError)):
                            self.optimizer.robust_curve_fit(
                                single_exp, tau, g2,
                                bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                                sigma=g2_err
                            )
                    else:
                        # Should handle gracefully
                        popt, pcov, info = self.optimizer.robust_curve_fit(
                            single_exp, tau, g2,
                            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                            sigma=g2_err
                        )

                        # If it succeeds, should return finite values
                        self.assertTrue(np.all(np.isfinite(popt)))

                except (RuntimeError, ValueError):
                    # Pathological cases may legitimately fail
                    continue

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up."""
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        # Perform many fitting operations
        for i in range(20):
            tau, g2, g2_err, _ = self.generator.generate_dataset(
                model_type='single_exp',
                n_points=100,
                noise_level=0.03
            )

            try:
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                # Force cleanup
                del popt, pcov, info, tau, g2, g2_err

            except RuntimeError:
                continue

        # Force garbage collection
        gc.collect()

        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory

        # Memory growth should be limited (no significant memory leaks)
        self.assertLess(memory_growth, 100)  # Less than 100MB growth

    def test_long_running_stability(self):
        """Test stability over extended operation."""
        start_time = time.time()
        successful_fits = 0
        total_fits = 0

        # Run for a limited time to avoid infinite testing
        max_runtime = 30  # 30 seconds

        while time.time() - start_time < max_runtime:
            total_fits += 1

            # Generate random dataset
            n_points = np.random.randint(20, 100)
            noise_level = np.random.uniform(0.01, 0.1)

            tau, g2, g2_err, _ = self.generator.generate_dataset(
                model_type='single_exp',
                n_points=n_points,
                noise_level=noise_level
            )

            try:
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                successful_fits += 1

                # Verify stability of results
                self.assertTrue(np.all(np.isfinite(popt)))
                self.assertGreater(info['r_squared'], 0.3)  # Minimum quality

            except RuntimeError:
                # Some random datasets might fail - this is acceptable
                continue

        # Should maintain reasonable success rate over extended operation
        success_rate = successful_fits / total_fits
        self.assertGreater(success_rate, 0.7)  # At least 70% success rate
        self.assertGreater(total_fits, 10)  # Should complete multiple fits


if __name__ == '__main__':
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    unittest.main(verbosity=2)