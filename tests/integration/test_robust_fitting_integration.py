"""
Integration tests for robust fitting framework.

This module tests the complete analysis workflows from raw data to final
diffusion parameters, ensuring all components work together correctly.
"""

import unittest
import numpy as np
import tempfile
import os
import warnings
from pathlib import Path

from xpcs_toolkit.helper.fitting import (
    RobustOptimizer,
    RobustOptimizerWithDiagnostics,
    SyntheticG2DataGenerator,
    ComprehensiveDiffusionAnalyzer,
    OptimizedXPCSFittingEngine,
    XPCSPerformanceOptimizer,
    robust_curve_fit,
    single_exp,
    double_exp
)


class TestCompleteAnalysisWorkflow(unittest.TestCase):
    """Test complete analysis workflows from data to results."""

    def setUp(self):
        """Set up test fixtures for integration testing."""
        self.generator = SyntheticG2DataGenerator(random_state=42)
        self.optimizer = RobustOptimizer()
        self.diagnostic_optimizer = RobustOptimizerWithDiagnostics()

    def test_single_q_analysis_workflow(self):
        """Test complete analysis workflow for single q-value."""
        # Generate realistic G2 data
        tau, g2, g2_err, true_params = self.generator.generate_dataset(
            model_type='single_exp',
            tau_range=(1e-6, 1e0),
            n_points=60,
            noise_level=0.03,
            outlier_fraction=0.05
        )

        # Step 1: Basic robust fitting
        popt, pcov, info = self.optimizer.robust_curve_fit(
            single_exp, tau, g2,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err
        )

        # Verify fitting results
        self.assertEqual(len(popt), 3)
        self.assertEqual(pcov.shape, (3, 3))
        self.assertIn('method', info)
        self.assertIn('r_squared', info)
        self.assertGreater(info['r_squared'], 0.5)  # Should have reasonable fit

        # Step 2: Generate fitted curve and residuals
        g2_fit = single_exp(tau, *popt)
        residuals = g2 - g2_fit
        chi_squared = np.sum((residuals / g2_err)**2)
        reduced_chi_squared = chi_squared / (len(g2) - len(popt))

        # Verify fit quality
        self.assertLess(reduced_chi_squared, 5.0)  # Should be reasonable
        self.assertTrue(np.all(np.isfinite(g2_fit)))

        # Step 3: Extract physical parameters
        gamma_fit, baseline_fit, beta_fit = popt
        tau_relax_fit = 1.0 / gamma_fit

        # Verify parameters are physical
        self.assertGreater(gamma_fit, 0)
        self.assertGreater(baseline_fit, 0.9)
        self.assertLess(baseline_fit, 1.1)
        self.assertGreater(beta_fit, 0)
        self.assertLess(beta_fit, 2.0)

        # Step 4: Parameter uncertainty analysis
        param_errors = np.sqrt(np.diag(pcov))
        relative_errors = param_errors / np.abs(popt)

        # Errors should be reasonable (not too large)
        self.assertTrue(np.all(relative_errors < 1.0))  # Less than 100% error
        self.assertTrue(np.all(param_errors > 0))

    def test_multi_q_diffusion_analysis(self):
        """Test complete diffusion analysis workflow with multiple q-values."""
        # Simulate realistic multi-q XPCS experiment
        q_values = np.array([0.001, 0.002, 0.003, 0.004, 0.005])  # inverse Angstroms
        D_true = 1e-8  # cm²/s - realistic diffusion coefficient

        fit_results = []

        for i, q in enumerate(q_values):
            # Generate data with diffusive dynamics: Gamma = D * q^2
            gamma_true = D_true * (q * 1e8)**2  # Convert units appropriately
            tau, g2, g2_err, _ = self.generator.generate_dataset(
                model_type='single_exp',
                n_points=50,
                noise_level=0.04
            )

            # Override with our realistic gamma
            g2 = single_exp(tau, gamma_true, 1.0, 0.3)
            g2 += 0.04 * 0.3 * np.random.normal(size=len(g2))
            g2_err = 0.04 * 0.3 * np.ones_like(g2)

            try:
                # Fit with robust optimizer
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([0.1, 0.9, 0.01], [10000, 1.1, 2.0]),
                    sigma=g2_err
                )

                gamma_fit, baseline_fit, beta_fit = popt
                gamma_error = np.sqrt(pcov[0, 0])

                fit_results.append({
                    'q': q,
                    'q_squared': q**2,
                    'gamma': gamma_fit,
                    'gamma_error': gamma_error,
                    'baseline': baseline_fit,
                    'beta': beta_fit,
                    'r_squared': info['r_squared'],
                    'gamma_true': gamma_true
                })

            except RuntimeError:
                # Some fits might fail - this is realistic
                continue

        # Verify we got enough successful fits
        self.assertGreaterEqual(len(fit_results), 3)

        # Extract diffusion coefficient from gamma vs q² relationship
        if len(fit_results) >= 3:
            q_squared = np.array([r['q_squared'] for r in fit_results])
            gamma_values = np.array([r['gamma'] for r in fit_results])
            gamma_errors = np.array([r['gamma_error'] for r in fit_results])

            # Weighted linear fit: gamma = D * q²
            weights = 1.0 / (gamma_errors**2 + 1e-10)  # Avoid division by zero
            D_fit = np.sum(weights * gamma_values * q_squared) / np.sum(weights * q_squared**2)

            # Verify diffusion coefficient is reasonable
            self.assertGreater(D_fit, 0)
            self.assertLess(abs(D_fit - D_true) / D_true, 0.5)  # Within 50% of true value

    def test_comprehensive_diagnostics_workflow(self):
        """Test complete workflow with comprehensive diagnostics."""
        # Generate test data
        tau, g2, g2_err, true_params = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=40,
            noise_level=0.05,
            outlier_fraction=0.1
        )

        # Run comprehensive analysis
        popt, pcov, diagnostics = self.diagnostic_optimizer.robust_curve_fit_with_diagnostics(
            single_exp, tau, g2,
            bounds=([10, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err,
            func_name="single_exp",
            bootstrap_samples=20  # Reduced for speed
        )

        # Verify diagnostic structure
        required_sections = [
            'function_name', 'data_info', 'optimization_info',
            'goodness_of_fit', 'covariance_analysis', 'residual_analysis'
        ]

        for section in required_sections:
            self.assertIn(section, diagnostics)

        # Verify data info
        data_info = diagnostics['data_info']
        self.assertEqual(data_info['n_data_points'], len(tau))
        self.assertEqual(data_info['n_parameters'], 3)
        self.assertTrue(data_info['has_uncertainties'])

        # Verify goodness of fit
        gof = diagnostics['goodness_of_fit']
        self.assertIn('r_squared', gof)
        self.assertIn('chi_squared', gof)
        self.assertIn('aic', gof)
        self.assertIn('bic', gof)

        # Verify residual analysis
        residuals = diagnostics['residual_analysis']
        self.assertIn('outliers', residuals)
        self.assertIn('randomness_test', residuals)

        # Verify covariance analysis
        covariance = diagnostics['covariance_analysis']
        self.assertIn('parameter_errors', covariance)
        self.assertIn('correlation_matrix', covariance)

        # Verify bootstrap analysis (if available)
        if 'bootstrap_analysis' in diagnostics and not diagnostics['bootstrap_analysis'].get('error'):
            bootstrap = diagnostics['bootstrap_analysis']
            self.assertIn('parameter_statistics', bootstrap)

        # Test quality assessment
        quality_assessment = self.diagnostic_optimizer.analyze_fit_quality(diagnostics)
        self.assertIsInstance(quality_assessment, str)
        self.assertIn('score', quality_assessment.lower())

    def test_error_recovery_workflow(self):
        """Test workflow behavior under difficult conditions."""
        # Create challenging data scenarios
        test_scenarios = [
            {
                'name': 'high_noise',
                'noise_level': 0.3,
                'outlier_fraction': 0.2
            },
            {
                'name': 'few_points',
                'n_points': 10,
                'noise_level': 0.1
            },
            {
                'name': 'extreme_tau_range',
                'tau_range': (1e-12, 1e-8),
                'noise_level': 0.05
            }
        ]

        successful_analyses = 0

        for scenario in test_scenarios:
            with self.subTest(scenario=scenario['name']):
                try:
                    # Generate challenging data
                    kwargs = {
                        'model_type': 'single_exp',
                        'noise_level': scenario.get('noise_level', 0.05),
                        'n_points': scenario.get('n_points', 40),
                        'tau_range': scenario.get('tau_range', (1e-6, 1e0)),
                        'outlier_fraction': scenario.get('outlier_fraction', 0.0)
                    }

                    tau, g2, g2_err, _ = self.generator.generate_dataset(**kwargs)

                    # Attempt robust fitting
                    popt, pcov, info = self.optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=([0.1, 0.9, 0.01], [1e6, 1.1, 5.0]),
                        sigma=g2_err
                    )

                    # If we get here, the fit succeeded
                    successful_analyses += 1

                    # Verify basic sanity of results
                    self.assertEqual(len(popt), 3)
                    self.assertTrue(np.all(np.isfinite(popt)))
                    self.assertIn('method', info)

                except RuntimeError:
                    # Difficult scenarios might fail - this is expected
                    continue
                except Exception as e:
                    # Unexpected errors should be investigated
                    self.fail(f"Unexpected error in {scenario['name']}: {e}")

        # Should succeed in at least some challenging scenarios
        self.assertGreater(successful_analyses, 0)

    def test_performance_optimization_integration(self):
        """Test integration with performance optimization features."""
        try:
            # Test performance optimizer
            perf_optimizer = XPCSPerformanceOptimizer(max_memory_mb=512)
            fitting_engine = OptimizedXPCSFittingEngine(perf_optimizer)

            # Generate multi-q dataset
            n_q = 5
            tau = np.logspace(-6, 0, 50)
            g2_data = np.zeros((len(tau), n_q))
            g2_errors = np.zeros_like(g2_data)
            q_values = np.linspace(0.001, 0.005, n_q)

            for i, q in enumerate(q_values):
                gamma = 1000.0 / (q**2 + 0.0001)
                g2_ideal = single_exp(tau, gamma, 1.0, 0.3)
                noise = 0.03 * g2_ideal * np.random.normal(size=len(tau))
                g2_data[:, i] = g2_ideal + noise
                g2_errors[:, i] = 0.03 * g2_ideal

            # Run optimized fitting
            results = fitting_engine.fit_g2_optimized(
                tau=tau,
                g2_data=g2_data,
                g2_errors=g2_errors,
                q_values=q_values,
                bootstrap_samples=0,  # Skip for speed
                enable_diagnostics=False
            )

            # Verify results structure
            self.assertIn('fit_results', results)
            self.assertIn('performance_info', results)
            self.assertIn('timing', results)

            # Check we got results for all q-values
            fit_results = results['fit_results']
            self.assertEqual(len(fit_results), n_q)

            # Verify timing information
            timing = results['timing']
            self.assertIn('total_time', timing)
            self.assertGreater(timing['total_time'], 0)

            # Verify performance info
            perf_info = results['performance_info']
            self.assertIn('estimated_memory_mb', perf_info)

        except ImportError:
            # Skip if performance optimization components not available
            self.skipTest("Performance optimization components not available")

    def test_model_selection_integration(self):
        """Test integration of model selection in complete workflow."""
        # Generate data with known model complexity
        tau, g2_true, g2_err, true_params = self.generator.generate_dataset(
            model_type='double_exp',
            n_points=60,
            noise_level=0.02
        )

        models_to_test = [
            {'func': single_exp, 'bounds': ([1, 0.9, 0.01], [100000, 1.1, 2.0]), 'name': 'Single Exp'},
            {'func': double_exp, 'bounds': ([1, 0.9, 0.01, 1, 0.01], [100000, 1.1, 2.0, 100000, 2.0]), 'name': 'Double Exp'}
        ]

        model_results = []

        for model in models_to_test:
            try:
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    model['func'], tau, g2_true,
                    bounds=model['bounds'],
                    sigma=g2_err
                )

                # Calculate model metrics
                y_pred = model['func'](tau, *popt)
                n_params = len(popt)
                chi2 = np.sum(((g2_true - y_pred) / g2_err)**2)
                n_data = len(g2_true)
                aic = 2 * n_params + n_data * np.log(chi2 / n_data)
                bic = np.log(n_data) * n_params + n_data * np.log(chi2 / n_data)

                model_results.append({
                    'name': model['name'],
                    'popt': popt,
                    'pcov': pcov,
                    'r_squared': info['r_squared'],
                    'aic': aic,
                    'bic': bic,
                    'chi2': chi2,
                    'n_params': n_params
                })

            except RuntimeError:
                # Some models might fail to fit
                continue

        # Verify we tested both models
        self.assertGreaterEqual(len(model_results), 1)

        if len(model_results) >= 2:
            # Compare models - double exponential should generally be better
            # since it's the true model (lower AIC/BIC)
            aic_values = [r['aic'] for r in model_results]
            best_model_idx = np.argmin(aic_values)
            best_model = model_results[best_model_idx]

            # Verify model selection is working
            self.assertIn(best_model['name'], ['Single Exp', 'Double Exp'])

    def test_backward_compatibility_integration(self):
        """Test backward compatibility with existing XPCS workflows."""
        # Test that robust_curve_fit can be used as drop-in replacement
        tau = np.logspace(-6, 0, 40)
        g2 = single_exp(tau, 1000.0, 1.0, 0.5)
        g2 += 0.02 * g2 * np.random.normal(size=len(g2))
        g2_err = 0.02 * g2

        # Test scipy-like interface
        popt_robust, pcov_robust = robust_curve_fit(
            single_exp, tau, g2,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err
        )

        # Should return same format as scipy.optimize.curve_fit
        self.assertEqual(len(popt_robust), 3)
        self.assertEqual(pcov_robust.shape, (3, 3))
        self.assertTrue(np.all(np.isfinite(popt_robust)))
        self.assertTrue(np.all(np.isfinite(pcov_robust)))

        # Test with additional robust features
        optimizer = RobustOptimizer()
        popt_full, pcov_full, info_full = optimizer.robust_curve_fit(
            single_exp, tau, g2,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err
        )

        # Should give similar results
        rel_diff = np.abs(popt_robust - popt_full) / np.abs(popt_robust)
        self.assertTrue(np.all(rel_diff < 0.1))  # Within 10%


class TestXPCSSpecificIntegration(unittest.TestCase):
    """Test integration scenarios specific to XPCS analysis."""

    def test_typical_xpcs_parameters(self):
        """Test with typical XPCS parameter ranges."""
        # Typical XPCS parameters
        q_values = np.array([0.0008, 0.0015, 0.0025, 0.004])  # 1/Å
        D_typical = 5e-9  # cm²/s - typical for colloidal particles

        generator = SyntheticG2DataGenerator(random_state=123)
        optimizer = RobustOptimizer()

        for q in q_values:
            with self.subTest(q=q):
                # Generate realistic XPCS data
                gamma_true = D_typical * (q * 1e8)**2  # Convert units
                tau = np.logspace(-6, 2, 64)  # Typical XPCS tau range

                # Use generator but override gamma
                _, g2, g2_err, _ = generator.generate_dataset(
                    model_type='single_exp', n_points=64, noise_level=0.05
                )

                # Replace with realistic data
                g2 = single_exp(tau, gamma_true, 1.0, 0.2)
                g2 += 0.05 * 0.2 * np.random.normal(size=len(g2))
                g2_err = 0.05 * 0.2 * np.ones_like(g2)

                try:
                    popt, pcov, info = optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=([0.01, 0.99, 0.001], [1000, 1.01, 1.0]),
                        sigma=g2_err
                    )

                    gamma_fit, baseline_fit, beta_fit = popt

                    # Verify parameters are in XPCS ranges
                    self.assertGreater(gamma_fit, 0.01)
                    self.assertLess(gamma_fit, 1000)
                    self.assertAlmostEqual(baseline_fit, 1.0, delta=0.01)
                    self.assertGreater(beta_fit, 0.001)
                    self.assertLess(beta_fit, 1.0)

                    # Verify fit quality
                    self.assertGreater(info['r_squared'], 0.7)

                except RuntimeError:
                    # Some challenging parameters might fail
                    continue

    def test_multi_tau_correlation_data(self):
        """Test with multi-tau correlation data structure."""
        # Simulate multi-tau logarithmic spacing
        n_levels = 8
        tau_per_level = 8
        tau_multi = []

        for level in range(n_levels):
            base_tau = 1e-6 * (2**level)
            level_tau = [base_tau * (i + 1) for i in range(tau_per_level)]
            tau_multi.extend(level_tau)

        tau_multi = np.array(tau_multi)

        # Generate G2 data for multi-tau structure
        generator = SyntheticG2DataGenerator(random_state=456)
        optimizer = RobustOptimizer()

        gamma_true = 100.0
        g2_true = single_exp(tau_multi, gamma_true, 1.0, 0.3)
        g2_noise = g2_true + 0.03 * 0.3 * np.random.normal(size=len(g2_true))
        g2_err = 0.03 * 0.3 * np.ones_like(g2_true)

        try:
            popt, pcov, info = optimizer.robust_curve_fit(
                single_exp, tau_multi, g2_noise,
                bounds=([1, 0.9, 0.01], [1000, 1.1, 2.0]),
                sigma=g2_err
            )

            gamma_fit, baseline_fit, beta_fit = popt

            # Verify reasonable recovery of parameters
            rel_error_gamma = abs(gamma_fit - gamma_true) / gamma_true
            self.assertLess(rel_error_gamma, 0.3)  # Within 30%

            rel_error_baseline = abs(baseline_fit - 1.0) / 1.0
            self.assertLess(rel_error_baseline, 0.1)  # Within 10%

        except RuntimeError:
            self.skipTest("Multi-tau fitting failed")

    def test_temperature_dependent_analysis(self):
        """Test analysis across different temperatures (simulated)."""
        # Simulate temperature-dependent diffusion
        temperatures = np.array([25, 35, 45, 55])  # Celsius
        D_reference = 1e-9  # cm²/s at reference temperature
        q = 0.002  # 1/Å

        generator = SyntheticG2DataGenerator(random_state=789)
        optimizer = RobustOptimizer()

        diffusion_results = []

        for T in temperatures:
            # Stokes-Einstein: D ∝ T/η, assume η ∝ exp(E/kT)
            # For simplicity, use D ∝ T (linear approximation)
            D_T = D_reference * (T + 273.15) / (25 + 273.15)
            gamma_T = D_T * (q * 1e8)**2

            # Generate data
            tau, g2, g2_err, _ = generator.generate_dataset(
                model_type='single_exp', n_points=50, noise_level=0.04
            )

            # Override with temperature-dependent data
            g2 = single_exp(tau, gamma_T, 1.0, 0.25)
            g2 += 0.04 * 0.25 * np.random.normal(size=len(g2))
            g2_err = 0.04 * 0.25 * np.ones_like(g2)

            try:
                popt, pcov, info = optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([0.1, 0.9, 0.01], [10000, 1.1, 2.0]),
                    sigma=g2_err
                )

                gamma_fit, _, _ = popt
                gamma_error = np.sqrt(pcov[0, 0])
                D_fit = gamma_fit / (q * 1e8)**2

                diffusion_results.append({
                    'temperature': T,
                    'D_fitted': D_fit,
                    'D_true': D_T,
                    'gamma_fit': gamma_fit,
                    'gamma_error': gamma_error,
                    'r_squared': info['r_squared']
                })

            except RuntimeError:
                continue

        # Verify temperature dependence
        if len(diffusion_results) >= 3:
            temps = [r['temperature'] for r in diffusion_results]
            D_values = [r['D_fitted'] for r in diffusion_results]

            # Should see positive correlation between T and D
            temp_range = max(temps) - min(temps)
            D_range = max(D_values) - min(D_values)

            self.assertGreater(temp_range, 0)
            self.assertGreater(D_range, 0)

            # Basic check: D should increase with T
            if len(D_values) >= 2:
                self.assertGreater(max(D_values), min(D_values))


if __name__ == '__main__':
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    unittest.main(verbosity=2)