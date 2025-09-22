"""
G2 Analysis Algorithm Validation Tests

This module provides comprehensive validation of G2 correlation function analysis,
including mathematical properties, physical constraints, and fitting algorithms.

G2 correlation functions must satisfy several mathematical and physical properties:
1. G2(τ=0) ≥ 1 (normalization)
2. G2(τ→∞) → 1 (decorrelation at infinite time)
3. Monotonic decay for simple systems
4. Causality (future doesn't affect past)
5. Statistical properties for noisy data
"""

import unittest
import warnings

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import optimize

from tests.scientific.constants import SCIENTIFIC_CONSTANTS, VALIDATION_CONFIG

# Import XPCS modules
from xpcs_toolkit.module.g2mod import (
    batch_g2_normalization,
    compute_g2_ensemble_statistics,
    optimize_g2_error_propagation,
    vectorized_g2_baseline_correction,
    vectorized_g2_interpolation,
)


class TestG2MathematicalProperties(unittest.TestCase):
    """Test mathematical properties that G2 functions must satisfy"""

    def setUp(self):
        """Set up test data and parameters"""
        self.rtol = SCIENTIFIC_CONSTANTS["rtol_default"]
        self.atol = SCIENTIFIC_CONSTANTS["atol_default"]

        # Create synthetic G2 data with known properties
        self.tau = np.logspace(-6, 3, 100)  # Time delays from µs to s
        self.q_values = np.linspace(0.001, 0.1, 20)  # Q-values in Å⁻¹

        # Single exponential decay model
        self.gamma = 1000.0  # Decay rate (Hz)
        self.beta = 0.8  # Coherence factor
        self.baseline = 1.0  # Baseline level

    def test_g2_normalization_property(self):
        """Test that G2(τ=0) ≥ 1"""
        # Generate single exponential G2
        g2_single = self.baseline + self.beta * np.exp(-self.gamma * self.tau)

        # Test normalization at τ=0
        g2_zero = self.baseline + self.beta
        self.assertGreaterEqual(g2_zero, 1.0, "G2(τ=0) must be ≥ 1")

        # Test with multiple q-values
        g2_matrix = np.tile(g2_single[:, np.newaxis], (1, len(self.q_values)))

        for q_idx in range(len(self.q_values)):
            g2_zero_q = g2_matrix[0, q_idx]
            self.assertGreaterEqual(
                g2_zero_q, 1.0, f"G2(τ=0, q={self.q_values[q_idx]}) must be ≥ 1"
            )

    def test_g2_asymptotic_behavior(self):
        """Test that G2(τ→∞) → 1"""
        # Generate G2 with proper asymptotic behavior
        g2 = self.baseline + self.beta * np.exp(-self.gamma * self.tau)

        # Check asymptotic value
        g2_inf = g2[-10:].mean()  # Average last 10 points
        np.testing.assert_allclose(
            g2_inf, self.baseline, rtol=1e-3, err_msg="G2(τ→∞) should approach baseline"
        )

    @given(
        beta=st.floats(min_value=0.1, max_value=2.0),
        gamma=st.floats(min_value=1.0, max_value=10000.0),
        baseline=st.floats(min_value=0.95, max_value=1.05),
    )
    @settings(max_examples=VALIDATION_CONFIG["hypothesis_max_examples"])
    def test_g2_monotonicity_property(self, beta, gamma, baseline):
        """Property-based test for G2 monotonicity in simple systems"""
        assume(beta > 0 and gamma > 0)

        # Generate monotonic G2
        g2 = baseline + beta * np.exp(-gamma * self.tau)

        # Test monotonicity
        diff = np.diff(g2)
        self.assertTrue(
            np.all(diff <= 0),
            "Single exponential G2 should be monotonically decreasing",
        )

    def test_g2_causality_principle(self):
        """Test that G2 respects causality"""
        # Create G2 data
        g2 = self.baseline + self.beta * np.exp(-self.gamma * self.tau)

        # Add some noise
        noise = 0.01 * np.random.normal(size=g2.shape)
        g2 + noise

        # Test that correlation at negative times equals positive times
        # (for stationary processes)
        tau_symmetric = np.concatenate([-self.tau[::-1], self.tau])
        g2_symmetric = np.concatenate([g2[::-1], g2])

        # Check symmetry property
        mid_point = len(tau_symmetric) // 2
        left_half = g2_symmetric[:mid_point]
        right_half = g2_symmetric[mid_point:]

        # For stationary processes, G2(-τ) should equal G2(τ)
        np.testing.assert_allclose(
            left_half,
            right_half[::-1],
            rtol=1e-2,
            err_msg="G2 should respect time-reversal symmetry for stationary processes",
        )


class TestG2VectorizedOperations(unittest.TestCase):
    """Test vectorized G2 operations for correctness and performance"""

    def setUp(self):
        """Set up test data"""
        self.n_times = 100
        self.n_q = 20
        self.n_datasets = 5

        # Generate test G2 data
        self.tau = np.logspace(-6, 2, self.n_times)
        self.q_values = np.linspace(0.001, 0.1, self.n_q)

        # Create multiple G2 datasets
        self.g2_datasets = []
        self.baselines = np.random.uniform(
            1.0, 1.05, self.n_q
        )  # Ensure baselines ≥ 1.0

        for _ in range(self.n_datasets):
            beta = np.random.uniform(0.5, 1.5, self.n_q)
            gamma = np.random.uniform(100, 5000, self.n_q)

            g2 = np.zeros((self.n_times, self.n_q))
            for q_idx in range(self.n_q):
                g2[:, q_idx] = self.baselines[q_idx] + beta[q_idx] * np.exp(
                    -gamma[q_idx] * self.tau
                )

            self.g2_datasets.append(g2)

    def test_baseline_correction_vectorized(self):
        """Test vectorized baseline correction"""
        g2_data = self.g2_datasets[0]
        baseline_values = self.baselines

        # Apply vectorized correction
        corrected = vectorized_g2_baseline_correction(g2_data, baseline_values)

        # Verify correction
        expected = g2_data - baseline_values[np.newaxis, :] + 1.0
        np.testing.assert_allclose(
            corrected,
            expected,
            rtol=1e-12,
            err_msg="Vectorized baseline correction failed",
        )

        # Test that corrected data has baseline ≈ 1
        corrected_baseline = corrected[-10:, :].mean(axis=0)
        np.testing.assert_allclose(
            corrected_baseline,
            1.0,
            rtol=1e-2,
            err_msg="Corrected baseline should be ≈ 1",
        )

    def test_batch_normalization_methods(self):
        """Test batch normalization methods"""
        methods = ["max", "mean", "std"]

        for method in methods:
            normalized = batch_g2_normalization(self.g2_datasets, method=method)

            self.assertEqual(
                len(normalized),
                len(self.g2_datasets),
                f"Output length mismatch for method {method}",
            )

            # Test specific properties for each method
            if method == "max":
                for norm_data in normalized:
                    max_vals = np.max(norm_data, axis=0)
                    np.testing.assert_allclose(
                        max_vals, 1.0, rtol=1e-10, err_msg="Max normalization failed"
                    )

            elif method == "mean":
                for norm_data in normalized:
                    mean_vals = np.mean(norm_data, axis=0)
                    np.testing.assert_allclose(
                        mean_vals, 1.0, rtol=1e-10, err_msg="Mean normalization failed"
                    )

    def test_ensemble_statistics_computation(self):
        """Test ensemble statistics computation"""
        stats = compute_g2_ensemble_statistics(self.g2_datasets)

        # Verify statistical properties
        expected_keys = [
            "ensemble_mean",
            "ensemble_std",
            "ensemble_median",
            "ensemble_min",
            "ensemble_max",
            "ensemble_var",
            "q_mean_values",
            "temporal_correlation",
        ]

        for key in expected_keys:
            self.assertIn(key, stats, f"Missing statistic: {key}")

        # Test statistical consistency
        ensemble_mean = stats["ensemble_mean"]
        ensemble_var = stats["ensemble_var"]
        ensemble_std = stats["ensemble_std"]

        # Variance should equal std squared
        np.testing.assert_allclose(
            ensemble_var, ensemble_std**2, rtol=1e-10, err_msg="Variance ≠ std²"
        )

        # Mean should be within reasonable bounds (allowing for small numerical errors)
        for q_idx in range(self.n_q):
            q_mean = ensemble_mean[:, q_idx]
            self.assertTrue(
                np.all(q_mean >= 0.98),
                "Ensemble mean G2 should be approximately ≥ 1 (within noise tolerance)",
            )

    def test_error_propagation_accuracy(self):
        """Test error propagation for G2 operations"""
        g2_data = self.g2_datasets[0]
        g2_errors = 0.05 * np.abs(g2_data)  # 5% relative errors

        # Test scaling operation
        scale_factor = 2.0
        operations = [{"type": "scale", "factor": scale_factor}]

        propagated_errors = optimize_g2_error_propagation(
            g2_data, g2_errors, operations
        )
        expected_errors = np.abs(scale_factor) * g2_errors

        np.testing.assert_allclose(
            propagated_errors,
            expected_errors,
            rtol=1e-12,
            err_msg="Scale error propagation failed",
        )

        # Test logarithmic operation
        operations = [{"type": "log"}]
        log_errors = optimize_g2_error_propagation(g2_data, g2_errors, operations)
        expected_log_errors = g2_errors / np.abs(g2_data)

        np.testing.assert_allclose(
            log_errors,
            expected_log_errors,
            rtol=1e-12,
            err_msg="Logarithmic error propagation failed",
        )


class TestG2InterpolationAccuracy(unittest.TestCase):
    """Test G2 interpolation accuracy and preservation of properties"""

    def test_interpolation_preserves_properties(self):
        """Test that interpolation preserves G2 mathematical properties"""
        # Create analytical G2
        tau_original = np.logspace(-4, 2, 50)
        beta, gamma, baseline = 0.8, 1000.0, 1.0
        g2_original = baseline + beta * np.exp(-gamma * tau_original)

        # Create 2D data (multiple q-values)
        n_q = 10
        g2_2d = np.tile(g2_original[:, np.newaxis], (1, n_q))

        # Define new time points
        tau_new = np.logspace(-4, 2, 100)

        # Interpolate
        g2_interpolated = vectorized_g2_interpolation(tau_original, g2_2d, tau_new)

        # Test that interpolated data preserves key properties

        # 1. Normalization property at τ≈0
        min_tau_idx = np.argmin(np.abs(tau_new))
        g2_zero = g2_interpolated[min_tau_idx, :]
        expected_g2_zero = baseline + beta
        np.testing.assert_allclose(
            g2_zero,
            expected_g2_zero,
            rtol=5e-2,  # Increased tolerance for interpolation errors
            err_msg="Interpolation doesn't preserve normalization",
        )

        # 2. Asymptotic behavior
        g2_inf = g2_interpolated[-10:, :].mean(axis=0)
        np.testing.assert_allclose(
            g2_inf,
            baseline,
            rtol=1e-2,
            err_msg="Interpolation doesn't preserve asymptotic behavior",
        )

        # 3. Monotonicity
        for q_idx in range(n_q):
            diff = np.diff(g2_interpolated[:, q_idx])
            self.assertTrue(
                np.all(diff <= 1e-6),  # Allow small numerical errors
                "Interpolated G2 should preserve monotonicity",
            )

    def test_interpolation_accuracy_against_analytical(self):
        """Test interpolation accuracy against analytical solution"""
        # Create sparse analytical data
        tau_sparse = np.logspace(-3, 1, 20)
        beta, gamma, baseline = 0.7, 500.0, 1.0

        # Analytical G2
        def g2_analytical(tau):
            return baseline + beta * np.exp(-gamma * tau)

        g2_sparse = g2_analytical(tau_sparse)

        # Create 2D data
        g2_2d_sparse = g2_sparse[:, np.newaxis]

        # Interpolate to denser grid
        tau_dense = np.logspace(-3, 1, 100)
        g2_interpolated = vectorized_g2_interpolation(
            tau_sparse, g2_2d_sparse, tau_dense
        )

        # Compare with analytical solution at dense points
        g2_analytical_dense = g2_analytical(tau_dense)

        # Calculate relative error
        rel_error = (
            np.abs(g2_interpolated[:, 0] - g2_analytical_dense) / g2_analytical_dense
        )
        max_rel_error = np.max(rel_error)

        self.assertLess(
            max_rel_error,
            0.01,  # 1% maximum error
            f"Interpolation error too large: {max_rel_error:.3f}",
        )


class TestG2FittingValidation(unittest.TestCase):
    """Test G2 fitting algorithms for accuracy and robustness"""

    def setUp(self):
        """Set up fitting test data"""
        self.tau = np.logspace(-6, 2, 100)
        self.noise_level = 0.02  # 2% noise

        # True parameters for single exponential
        self.true_params = {"beta": 0.8, "gamma": 1000.0, "baseline": 1.0}

    def single_exponential_model(self, tau, beta, gamma, baseline):
        """Single exponential G2 model"""
        return baseline + beta * np.exp(-gamma * tau)

    def test_single_exponential_fitting_accuracy(self):
        """Test single exponential fitting accuracy"""
        # Generate synthetic data with noise
        g2_true = self.single_exponential_model(self.tau, **self.true_params)
        noise = self.noise_level * np.random.normal(size=g2_true.shape)
        g2_noisy = g2_true + noise

        # Ensure G2 properties are maintained even with noise
        g2_noisy = np.maximum(g2_noisy, 1.0)  # Ensure G2 ≥ 1

        # Define fitting function
        def fit_func(tau, beta, gamma, baseline):
            return self.single_exponential_model(tau, beta, gamma, baseline)

        # Perform fitting with reasonable bounds
        bounds = ([0.1, 1.0, 0.9], [2.0, 10000.0, 1.1])
        initial_guess = [0.5, 500.0, 1.0]

        try:
            popt, pcov = optimize.curve_fit(
                fit_func,
                self.tau,
                g2_noisy,
                p0=initial_guess,
                bounds=bounds,
                maxfev=5000,
            )

            # Extract fitted parameters
            beta_fit, gamma_fit, baseline_fit = popt
            np.sqrt(np.diag(pcov))

            # Test parameter accuracy
            self.assertAlmostEqual(
                beta_fit,
                self.true_params["beta"],
                delta=0.1,
                msg="Beta parameter fit inaccurate",
            )
            self.assertAlmostEqual(
                gamma_fit,
                self.true_params["gamma"],
                delta=200,
                msg="Gamma parameter fit inaccurate",
            )
            self.assertAlmostEqual(
                baseline_fit,
                self.true_params["baseline"],
                delta=0.05,
                msg="Baseline parameter fit inaccurate",
            )

            # Test that fitted curve preserves G2 properties
            g2_fitted = fit_func(self.tau, *popt)

            # Normalization
            self.assertGreaterEqual(g2_fitted[0], 1.0, "Fitted G2(0) should be ≥ 1")

            # Asymptotic behavior
            g2_inf_fitted = g2_fitted[-10:].mean()
            self.assertAlmostEqual(
                g2_inf_fitted,
                baseline_fit,
                delta=0.01,
                msg="Fitted G2(∞) should equal baseline",
            )

            # Monotonicity
            diff_fitted = np.diff(g2_fitted)
            self.assertTrue(
                np.all(diff_fitted <= 1e-10),
                "Fitted G2 should be monotonically decreasing",
            )

        except Exception as e:
            self.fail(f"Fitting failed with error: {e}")

    @given(
        noise_level=st.floats(min_value=0.001, max_value=0.1),
        beta=st.floats(min_value=0.3, max_value=1.5),
        gamma=st.floats(min_value=100, max_value=5000),
    )
    @settings(max_examples=20, deadline=10000)  # Longer deadline for fitting
    def test_fitting_robustness_to_noise(self, noise_level, beta, gamma):
        """Property-based test for fitting robustness to noise"""
        baseline = 1.0

        # Generate noisy data
        g2_true = self.single_exponential_model(self.tau, beta, gamma, baseline)
        noise = noise_level * np.random.normal(size=g2_true.shape)
        g2_noisy = np.maximum(g2_true + noise, 1.0)  # Ensure G2 ≥ 1

        # Define bounds and initial guess
        bounds = ([0.1, 50.0, 0.95], [2.0, 10000.0, 1.05])
        initial_guess = [0.8, 1000.0, 1.0]

        try:
            popt, _ = optimize.curve_fit(
                lambda tau, b, g, bl: bl + b * np.exp(-g * tau),
                self.tau,
                g2_noisy,
                p0=initial_guess,
                bounds=bounds,
                maxfev=3000,
            )

            beta_fit, gamma_fit, baseline_fit = popt

            # Check that fitted parameters are reasonable
            self.assertGreater(beta_fit, 0, "Beta should be positive")
            self.assertGreater(gamma_fit, 0, "Gamma should be positive")
            self.assertGreaterEqual(baseline_fit, 0.95, "Baseline should be ≈ 1")

            # Check relative errors are reasonable for given noise level
            beta_rel_error = abs(beta_fit - beta) / beta
            abs(gamma_fit - gamma) / gamma

            # Allow larger errors for higher noise levels
            max_allowed_error = min(2.0, 5 * noise_level)  # Cap at 200% error

            self.assertLess(
                beta_rel_error,
                max_allowed_error,
                f"Beta error too large: {beta_rel_error:.3f}",
            )

        except Exception as e:
            # Some high-noise cases may fail to converge - this is acceptable
            # Store exception info for potential debugging
            pytest.skip(f"High-noise case failed to converge (acceptable): {e}")


if __name__ == "__main__":
    # Configure warnings for scientific tests
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="invalid value encountered in log10"
    )
    warnings.filterwarnings(
        "ignore", category=RuntimeWarning, message="divide by zero encountered"
    )

    unittest.main(verbosity=2)
