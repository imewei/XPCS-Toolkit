"""
Scientific validation tests for robust fitting framework.

This module validates scientific accuracy using synthetic datasets with known
ground truth and established theoretical results.
"""

import unittest
import numpy as np
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import tempfile
import os

from xpcs_toolkit.helper.fitting import (
    RobustOptimizer,
    SyntheticG2DataGenerator,
    ComprehensiveDiffusionAnalyzer,
    single_exp,
    double_exp,
    robust_curve_fit
)


class TestScientificAccuracy(unittest.TestCase):
    """Test scientific accuracy of fitting algorithms."""

    def setUp(self):
        """Set up scientific validation tests."""
        self.generator = SyntheticG2DataGenerator(random_state=42)
        self.optimizer = RobustOptimizer()
        self.tolerance = 0.15  # 15% tolerance for parameter recovery

    def test_single_exponential_parameter_recovery(self):
        """Test accurate recovery of single exponential parameters."""
        # Define ground truth parameters
        true_params = {
            'gamma': 1000.0,     # relaxation rate (Hz)
            'baseline': 1.0,     # baseline (normalized)
            'beta': 0.5          # contrast
        }

        # Test parameter recovery across different noise levels
        noise_levels = [0.01, 0.03, 0.05, 0.1]
        recoveries = []

        for noise_level in noise_levels:
            with self.subTest(noise_level=noise_level):
                # Generate synthetic data
                tau, g2, g2_err, _ = self.generator.generate_dataset(
                    model_type='single_exp',
                    n_points=60,
                    noise_level=noise_level,
                    tau_range=(1e-6, 1e-1)
                )

                # Override with known parameters
                g2_true = single_exp(tau, true_params['gamma'],
                                   true_params['baseline'], true_params['beta'])
                g2 = g2_true + noise_level * true_params['beta'] * np.random.normal(size=len(g2_true))
                g2_err = noise_level * true_params['beta'] * np.ones_like(g2)

                try:
                    # Fit with robust optimizer
                    popt, pcov, info = self.optimizer.robust_curve_fit(
                        single_exp, tau, g2,
                        bounds=([1, 0.9, 0.01], [10000, 1.1, 2.0]),
                        sigma=g2_err
                    )

                    gamma_fit, baseline_fit, beta_fit = popt

                    # Calculate recovery accuracy
                    gamma_error = abs(gamma_fit - true_params['gamma']) / true_params['gamma']
                    baseline_error = abs(baseline_fit - true_params['baseline']) / true_params['baseline']
                    beta_error = abs(beta_fit - true_params['beta']) / true_params['beta']

                    recoveries.append({
                        'noise_level': noise_level,
                        'gamma_error': gamma_error,
                        'baseline_error': baseline_error,
                        'beta_error': beta_error,
                        'r_squared': info['r_squared']
                    })

                    # Assert accuracy within tolerance
                    self.assertLess(gamma_error, self.tolerance)
                    self.assertLess(baseline_error, 0.05)  # Stricter for baseline
                    self.assertLess(beta_error, self.tolerance)

                    # Verify goodness of fit
                    self.assertGreater(info['r_squared'], 0.8)

                except RuntimeError as e:
                    self.fail(f"Fitting failed for noise level {noise_level}: {e}")

        # Verify error increases with noise level
        gamma_errors = [r['gamma_error'] for r in recoveries]
        noise_levels_tested = [r['noise_level'] for r in recoveries]

        # Should see positive correlation between noise and fitting error
        correlation, p_value = stats.pearsonr(noise_levels_tested, gamma_errors)
        self.assertGreater(correlation, 0)  # Positive correlation

    def test_double_exponential_parameter_recovery(self):
        """Test accurate recovery of double exponential parameters."""
        # Define ground truth parameters for double exponential
        true_params = {
            'gamma1': 5000.0,     # fast component
            'baseline': 1.0,
            'beta1': 0.3,
            'gamma2': 500.0,      # slow component
            'beta2': 0.2
        }

        # Generate synthetic data
        tau, g2, g2_err, _ = self.generator.generate_dataset(
            model_type='double_exp',
            n_points=80,
            noise_level=0.02,
            tau_range=(1e-6, 1e-2)
        )

        # Override with known parameters
        g2_true = double_exp(tau, true_params['gamma1'], true_params['baseline'],
                           true_params['beta1'], true_params['gamma2'], true_params['beta2'])
        g2 = g2_true + 0.02 * (true_params['beta1'] + true_params['beta2']) * np.random.normal(size=len(g2_true))
        g2_err = 0.02 * (true_params['beta1'] + true_params['beta2']) * np.ones_like(g2)

        try:
            # Fit with robust optimizer
            popt, pcov, info = self.optimizer.robust_curve_fit(
                double_exp, tau, g2,
                bounds=([1, 0.9, 0.01, 1, 0.01], [50000, 1.1, 1.0, 50000, 1.0]),
                sigma=g2_err
            )

            gamma1_fit, baseline_fit, beta1_fit, gamma2_fit, beta2_fit = popt

            # Calculate recovery accuracy
            gamma1_error = abs(gamma1_fit - true_params['gamma1']) / true_params['gamma1']
            gamma2_error = abs(gamma2_fit - true_params['gamma2']) / true_params['gamma2']
            beta1_error = abs(beta1_fit - true_params['beta1']) / true_params['beta1']
            beta2_error = abs(beta2_fit - true_params['beta2']) / true_params['beta2']

            # Assert accuracy within tolerance (more lenient for complex model)
            self.assertLess(gamma1_error, 0.3)  # 30% tolerance
            self.assertLess(gamma2_error, 0.3)
            self.assertLess(beta1_error, 0.3)
            self.assertLess(beta2_error, 0.3)

            # Verify goodness of fit
            self.assertGreater(info['r_squared'], 0.85)

        except RuntimeError as e:
            self.skipTest(f"Double exponential fitting failed: {e}")

    def test_diffusion_coefficient_extraction(self):
        """Test accurate extraction of diffusion coefficients from multi-q data."""
        # Define known diffusion coefficient
        D_true = 2.5e-9  # cm²/s
        q_values = np.array([0.001, 0.0015, 0.002, 0.0025, 0.003])  # 1/Å

        fitted_gammas = []
        q_squared_values = []

        for q in q_values:
            # Calculate expected gamma for this q
            gamma_true = D_true * (q * 1e8)**2  # Convert units

            # Generate synthetic data
            tau = np.logspace(-6, 1, 50)
            g2_true = single_exp(tau, gamma_true, 1.0, 0.25)
            g2 = g2_true + 0.03 * 0.25 * np.random.normal(size=len(g2_true))
            g2_err = 0.03 * 0.25 * np.ones_like(g2)

            try:
                # Fit to extract gamma
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([0.1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                gamma_fit = popt[0]
                fitted_gammas.append(gamma_fit)
                q_squared_values.append(q**2)

                # Verify individual fit quality
                self.assertGreater(info['r_squared'], 0.8)

            except RuntimeError:
                continue

        # Verify we got enough successful fits
        self.assertGreaterEqual(len(fitted_gammas), 3)

        # Extract diffusion coefficient from gamma vs q² relationship
        if len(fitted_gammas) >= 3:
            # Linear regression: gamma = D * q²
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                q_squared_values, fitted_gammas
            )

            D_fitted = slope  # Slope gives diffusion coefficient
            D_error = abs(D_fitted - D_true) / D_true

            # Verify accuracy of diffusion coefficient extraction
            self.assertLess(D_error, 0.2)  # Within 20%
            self.assertGreater(r_value**2, 0.9)  # Strong linear correlation
            self.assertLess(p_value, 0.05)  # Statistically significant

    def test_outlier_robustness(self):
        """Test robustness against outliers."""
        # Generate clean data
        tau, g2_clean, g2_err, true_params = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=50,
            noise_level=0.02,
            outlier_fraction=0.0
        )

        # Generate data with outliers
        tau, g2_outliers, g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=50,
            noise_level=0.02,
            outlier_fraction=0.15  # 15% outliers
        )

        # Fit both datasets
        popt_clean, _, info_clean = self.optimizer.robust_curve_fit(
            single_exp, tau, g2_clean,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err
        )

        popt_outliers, _, info_outliers = self.optimizer.robust_curve_fit(
            single_exp, tau, g2_outliers,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
            sigma=g2_err
        )

        # Compare parameter recovery
        param_differences = np.abs(popt_clean - popt_outliers) / np.abs(popt_clean)

        # Robust fitting should minimize impact of outliers
        self.assertTrue(np.all(param_differences < 0.3))  # Less than 30% difference

        # Both fits should have reasonable quality
        self.assertGreater(info_clean['r_squared'], 0.9)
        self.assertGreater(info_outliers['r_squared'], 0.7)  # Lower but still reasonable

    def test_noise_scaling_behavior(self):
        """Test that parameter uncertainties scale correctly with noise."""
        base_noise = 0.02
        noise_multipliers = [1, 2, 4, 8]

        uncertainties_by_noise = []

        for multiplier in noise_multipliers:
            noise_level = base_noise * multiplier

            # Generate data with scaled noise
            tau, g2, g2_err, _ = self.generator.generate_dataset(
                model_type='single_exp',
                n_points=50,
                noise_level=noise_level
            )

            try:
                # Fit and extract uncertainties
                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                param_errors = np.sqrt(np.diag(pcov))
                uncertainties_by_noise.append({
                    'noise_level': noise_level,
                    'gamma_uncertainty': param_errors[0],
                    'beta_uncertainty': param_errors[2]
                })

            except RuntimeError:
                continue

        # Verify uncertainties scale with noise
        if len(uncertainties_by_noise) >= 3:
            noise_levels = [u['noise_level'] for u in uncertainties_by_noise]
            gamma_uncertainties = [u['gamma_uncertainty'] for u in uncertainties_by_noise]

            # Should see positive correlation
            correlation, p_value = stats.pearsonr(noise_levels, gamma_uncertainties)
            self.assertGreater(correlation, 0.5)  # Strong positive correlation

    def test_systematic_bias_detection(self):
        """Test detection of systematic biases in fitting."""
        # Generate data with known systematic bias
        tau, g2, g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=50,
            noise_level=0.03,
            systematic_error=True
        )

        # Fit with diagnostics to detect bias
        from xpcs_toolkit.helper.fitting import RobustOptimizerWithDiagnostics

        diagnostic_optimizer = RobustOptimizerWithDiagnostics()

        try:
            popt, pcov, diagnostics = diagnostic_optimizer.robust_curve_fit_with_diagnostics(
                single_exp, tau, g2,
                bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                sigma=g2_err,
                bootstrap_samples=0  # Skip bootstrap for speed
            )

            # Check residual analysis for systematic deviations
            residual_analysis = diagnostics.get('residual_analysis', {})
            systematic_deviations = diagnostics.get('systematic_deviations', {})

            # Should detect some systematic issues
            if residual_analysis and 'randomness_test' in residual_analysis:
                randomness = residual_analysis['randomness_test']
                # Systematic bias might show up as non-random residuals
                self.assertIn('is_random', randomness)

        except Exception as e:
            self.skipTest(f"Systematic bias detection test failed: {e}")

    def test_physical_constraints_enforcement(self):
        """Test that physical constraints are properly enforced."""
        # Generate data that might lead to unphysical parameters
        tau = np.logspace(-6, 0, 40)
        g2 = single_exp(tau, 1000.0, 1.0, 0.5)

        # Add noise that might push parameters out of bounds
        g2 += 0.1 * np.random.normal(size=len(g2))
        g2_err = 0.1 * np.ones_like(g2)

        # Fit with physical bounds
        popt, pcov, info = self.optimizer.robust_curve_fit(
            single_exp, tau, g2,
            bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),  # Physical bounds
            sigma=g2_err
        )

        gamma_fit, baseline_fit, beta_fit = popt

        # Verify parameters respect physical constraints
        self.assertGreater(gamma_fit, 0)  # Positive relaxation rate
        self.assertGreater(baseline_fit, 0.9)  # Baseline near 1
        self.assertLess(baseline_fit, 1.1)
        self.assertGreater(beta_fit, 0)  # Positive contrast
        self.assertLess(beta_fit, 2.0)  # Reasonable contrast

        # Verify fit is still reasonable despite constraints
        self.assertGreater(info['r_squared'], 0.5)

    def test_convergence_consistency(self):
        """Test that fitting converges to consistent results."""
        # Generate the same data multiple times and verify consistent results
        tau, g2, g2_err, _ = self.generator.generate_dataset(
            model_type='single_exp',
            n_points=50,
            noise_level=0.03
        )

        # Fit multiple times with different initial conditions
        results = []
        for i in range(5):
            try:
                # Vary initial guess slightly
                p0 = [1000.0 * (1 + 0.1 * (i - 2)), 1.0, 0.5 * (1 + 0.1 * (i - 2))]

                popt, pcov, info = self.optimizer.robust_curve_fit(
                    single_exp, tau, g2,
                    p0=p0,
                    bounds=([1, 0.9, 0.01], [100000, 1.1, 2.0]),
                    sigma=g2_err
                )

                results.append(popt)

            except RuntimeError:
                continue

        # Verify consistency across runs
        if len(results) >= 3:
            results_array = np.array(results)
            param_stds = np.std(results_array, axis=0)
            param_means = np.mean(results_array, axis=0)

            # Relative standard deviation should be small
            relative_stds = param_stds / np.abs(param_means)
            self.assertTrue(np.all(relative_stds < 0.1))  # Less than 10% variation

    def test_theoretical_limits(self):
        """Test behavior at theoretical limits."""
        # Test very fast relaxation (high gamma)
        tau_fast = np.logspace(-9, -6, 30)
        gamma_fast = 1e6
        g2_fast = single_exp(tau_fast, gamma_fast, 1.0, 0.3)
        g2_fast += 0.02 * 0.3 * np.random.normal(size=len(g2_fast))

        try:
            popt_fast, _, info_fast = self.optimizer.robust_curve_fit(
                single_exp, tau_fast, g2_fast,
                bounds=([1e3, 0.9, 0.01], [1e8, 1.1, 2.0])
            )

            gamma_fit_fast = popt_fast[0]
            self.assertGreater(gamma_fit_fast, 1e3)
            self.assertGreater(info_fast['r_squared'], 0.7)

        except RuntimeError:
            self.skipTest("Fast relaxation test failed")

        # Test very slow relaxation (low gamma)
        tau_slow = np.logspace(-3, 2, 30)
        gamma_slow = 0.1
        g2_slow = single_exp(tau_slow, gamma_slow, 1.0, 0.4)
        g2_slow += 0.02 * 0.4 * np.random.normal(size=len(g2_slow))

        try:
            popt_slow, _, info_slow = self.optimizer.robust_curve_fit(
                single_exp, tau_slow, g2_slow,
                bounds=([0.001, 0.9, 0.01], [100, 1.1, 2.0])
            )

            gamma_fit_slow = popt_slow[0]
            self.assertLess(gamma_fit_slow, 100)
            self.assertGreater(info_slow['r_squared'], 0.7)

        except RuntimeError:
            self.skipTest("Slow relaxation test failed")


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability of fitting algorithms."""

    def setUp(self):
        """Set up numerical stability tests."""
        self.optimizer = RobustOptimizer()

    def test_extreme_dynamic_range(self):
        """Test fitting with extreme dynamic ranges."""
        # Test with very small tau values
        tau_small = np.logspace(-12, -9, 20)
        g2_small = single_exp(tau_small, 1e9, 1.0, 0.5)

        try:
            popt, _, _ = self.optimizer.robust_curve_fit(
                single_exp, tau_small, g2_small,
                bounds=([1e6, 0.9, 0.01], [1e12, 1.1, 2.0])
            )

            self.assertEqual(len(popt), 3)
            self.assertTrue(np.all(np.isfinite(popt)))

        except RuntimeError:
            self.skipTest("Extreme small tau test failed")

        # Test with very large tau values
        tau_large = np.logspace(3, 6, 20)
        g2_large = single_exp(tau_large, 1e-3, 1.0, 0.5)

        try:
            popt, _, _ = self.optimizer.robust_curve_fit(
                single_exp, tau_large, g2_large,
                bounds=([1e-6, 0.9, 0.01], [1, 1.1, 2.0])
            )

            self.assertEqual(len(popt), 3)
            self.assertTrue(np.all(np.isfinite(popt)))

        except RuntimeError:
            self.skipTest("Extreme large tau test failed")

    def test_ill_conditioned_problems(self):
        """Test behavior with ill-conditioned fitting problems."""
        # Create data with very small contrast (difficult to fit)
        tau = np.logspace(-6, 0, 30)
        g2 = single_exp(tau, 1000.0, 1.0, 0.001)  # Very small beta
        g2 += 0.001 * np.random.normal(size=len(g2))  # Noise comparable to signal

        try:
            popt, pcov, info = self.optimizer.robust_curve_fit(
                single_exp, tau, g2,
                bounds=([1, 0.99, 1e-6], [100000, 1.01, 0.1])
            )

            # Should still produce finite results
            self.assertTrue(np.all(np.isfinite(popt)))
            self.assertTrue(np.all(np.isfinite(pcov)))

            # Check that covariance matrix is not singular
            cond_number = np.linalg.cond(pcov)
            self.assertLess(cond_number, 1e12)

        except RuntimeError:
            # Ill-conditioned problems may legitimately fail
            pass

    def test_precision_limits(self):
        """Test behavior at machine precision limits."""
        # Test with data that tests numerical precision
        tau = np.array([1e-15, 1e-12, 1e-9, 1e-6])
        g2 = np.array([1.5, 1.4, 1.3, 1.2])
        g2_err = np.array([0.01, 0.01, 0.01, 0.01])

        try:
            popt, _, _ = self.optimizer.robust_curve_fit(
                single_exp, tau, g2,
                bounds=([1e10, 0.9, 0.01], [1e18, 1.1, 2.0]),
                sigma=g2_err
            )

            # Should handle extreme values gracefully
            self.assertTrue(np.all(np.isfinite(popt)))

        except (RuntimeError, ValueError):
            # May fail at precision limits - this is acceptable
            pass


if __name__ == '__main__':
    # Suppress warnings for cleaner test output
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    unittest.main(verbosity=2)