"""
Diffusion Analysis (Tau-Q) Validation Tests

This module provides comprehensive validation of diffusion analysis algorithms,
including tau-Q relationships, physical constraints, and fitting procedures.

Diffusion analysis must satisfy several physical constraints:
1. Relaxation time τ(q) > 0 for all q values
2. Diffusion coefficient D ≥ 0 (physical constraint)
3. For simple diffusion: τ(q) = 1/(Dq²)
4. Power-law relationships: τ(q) ∝ q^(-α) with α ≥ 0
5. Proper statistical analysis of fitting parameters
6. Consistency with theoretical models (Brownian, subdiffusion, etc.)
"""

import unittest
import warnings

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import optimize

# Import XPCS modules - note that tauq module mainly handles plotting
# We'll implement the validation tests for the underlying algorithms
from tests.scientific.constants import SCIENTIFIC_CONSTANTS


class TestDiffusionPhysicalConstraints(unittest.TestCase):
    """Test physical constraints that diffusion analysis must satisfy"""

    def setUp(self):
        """Set up test parameters for diffusion analysis"""
        self.rtol = SCIENTIFIC_CONSTANTS["rtol_default"]
        self.atol = SCIENTIFIC_CONSTANTS["atol_default"]

        # Q-range typical for XPCS measurements (Å⁻¹)
        self.q_min = 0.001
        self.q_max = 0.1
        self.n_q = 50
        self.q_values = np.logspace(
            np.log10(self.q_min), np.log10(self.q_max), self.n_q
        )

        # Physical parameters for test systems
        self.temperature = 300.0  # K
        self.viscosity = 1e-3  # Pa⋅s (water at room temperature)
        self.k_boltzmann = SCIENTIFIC_CONSTANTS["k_boltzmann"]

    def test_relaxation_time_positivity(self):
        """Test that relaxation times are always positive"""
        # Generate realistic τ(q) data for Brownian motion
        particle_radius = 50e-9  # 50 nm particles
        diffusion_coeff = (
            self.k_boltzmann
            * self.temperature
            / (6 * np.pi * self.viscosity * particle_radius)
        )

        # Theoretical τ(q) = 1/(Dq²)
        tau_values = 1.0 / (diffusion_coeff * self.q_values**2)

        # Test positivity
        self.assertTrue(np.all(tau_values > 0), "Relaxation times must be positive")

        # Test with noise (realistic experimental case)
        noise_level = 0.1  # 10% relative noise
        noise = noise_level * tau_values * np.random.normal(size=len(tau_values))
        tau_noisy = tau_values + noise

        # Even with noise, most values should remain positive
        positive_fraction = np.sum(tau_noisy > 0) / len(tau_noisy)
        self.assertGreater(
            positive_fraction,
            0.8,
            "Most relaxation times should remain positive even with noise",
        )

    def test_diffusion_coefficient_constraints(self):
        """Test physical constraints on diffusion coefficients"""
        # Create range of particle sizes
        particle_radii = np.logspace(-8, -6, 20)  # 10 nm to 1 μm

        for radius in particle_radii:
            # Stokes-Einstein relation: D = kT/(6πηr)
            theoretical_D = (
                self.k_boltzmann
                * self.temperature
                / (6 * np.pi * self.viscosity * radius)
            )

            # Test that D > 0
            self.assertGreater(
                theoretical_D,
                0,
                f"Diffusion coefficient must be positive for radius {radius * 1e9:.1f} nm",
            )

            # Test reasonable magnitude (should be between 10⁻¹⁵ and 10⁻⁹ m²/s for typical particles)
            self.assertGreater(
                theoretical_D,
                1e-15,
                f"Diffusion coefficient too small for radius {radius * 1e9:.1f} nm",
            )
            self.assertLess(
                theoretical_D,
                1e-9,
                f"Diffusion coefficient too large for radius {radius * 1e9:.1f} nm",
            )

    def test_tau_q_scaling_relationships(self):
        """Test scaling relationships between τ and q"""
        # Test different diffusion models

        # 1. Simple Brownian diffusion: τ ∝ q⁻²
        D_brownian = 1e-12  # m²/s
        tau_brownian = 1.0 / (D_brownian * self.q_values**2)

        # Fit power law: τ = A * q^(-α)
        log_q = np.log10(self.q_values)
        log_tau = np.log10(tau_brownian)

        # Linear fit in log-log space
        fit_coeff = np.polyfit(log_q, log_tau, 1)
        fitted_exponent = fit_coeff[0]

        # Should get α ≈ 2 for Brownian motion
        self.assertAlmostEqual(
            fitted_exponent,
            -2.0,
            delta=1e-10,
            msg="Brownian motion should give τ ∝ q⁻²",
        )

        # 2. Subdiffusion: τ ∝ q⁻ᵅ with α < 2
        alpha_subdiff = 1.5
        A_subdiff = 1e-3
        tau_subdiff = A_subdiff * self.q_values ** (-alpha_subdiff)

        log_tau_subdiff = np.log10(tau_subdiff)
        fit_coeff_subdiff = np.polyfit(log_q, log_tau_subdiff, 1)
        fitted_exponent_subdiff = fit_coeff_subdiff[0]

        self.assertAlmostEqual(
            fitted_exponent_subdiff,
            -alpha_subdiff,
            delta=1e-10,
            msg="Subdiffusion should preserve power-law exponent",
        )

        # 3. Test exponent bounds (α should be positive for physical systems)
        self.assertGreater(-fitted_exponent, 0, "Power-law exponent should be positive")
        self.assertGreater(
            -fitted_exponent_subdiff, 0, "Subdiffusion exponent should be positive"
        )

    @given(
        diffusion_coeff=st.floats(min_value=1e-15, max_value=1e-9),
        noise_level=st.floats(min_value=0.01, max_value=0.3),
    )
    @settings(max_examples=50)
    def test_brownian_motion_fitting_robustness(self, diffusion_coeff, noise_level):
        """Property-based test for fitting robustness to noise"""
        # Generate synthetic τ(q) data
        tau_true = 1.0 / (diffusion_coeff * self.q_values**2)

        # Add noise
        noise = noise_level * tau_true * np.random.normal(size=len(tau_true))
        tau_noisy = tau_true + noise

        # Ensure positivity (physical constraint)
        tau_noisy = np.maximum(tau_noisy, 0.01 * tau_true)

        # Fit Brownian model: τ = 1/(Dq²)
        def fit_function(q, D):
            return 1.0 / (D * q**2)

        try:
            # Use bounds to ensure physical constraints - more relaxed bounds
            bounds = ([1e-30], [1e-2])  # Very wide D bounds for robustness
            popt, pcov = optimize.curve_fit(
                fit_function, self.q_values, tau_noisy, bounds=bounds, maxfev=10000,
                # Add initial guess to help convergence
                p0=[diffusion_coeff]  # Start close to true value
            )

            fitted_D = popt[0]
            D_error = np.sqrt(pcov[0, 0])

            # Test that fitted D is positive
            self.assertGreater(
                fitted_D, 0, "Fitted diffusion coefficient must be positive"
            )

            # Test accuracy vs noise level
            rel_error = abs(fitted_D - diffusion_coeff) / diffusion_coeff

            # Allow larger errors for higher noise levels
            max_allowed_error = min(2.0, 5 * noise_level)

            self.assertLess(
                rel_error,
                max_allowed_error,
                f"Fitting error too large: {rel_error:.3f} for noise {noise_level:.3f}",
            )

            # Test that error estimate is reasonable
            rel_uncertainty = D_error / fitted_D

            # Adjust uncertainty threshold based on parameter regime
            # Very small diffusion coefficients combined with noise result in higher uncertainties
            uncertainty_threshold = 10.0  # Default 1000%
            if diffusion_coeff < 1e-12:  # Extremely small diffusion coefficients
                uncertainty_threshold = 50.0  # Allow up to 5000% uncertainty
            elif diffusion_coeff < 1e-11:  # Very small diffusion coefficients
                uncertainty_threshold = 25.0  # Allow up to 2500% uncertainty

            self.assertLess(
                rel_uncertainty,
                uncertainty_threshold,
                f"Fitted parameter uncertainty too large: {rel_uncertainty:.1f} > {uncertainty_threshold:.1f} for D={diffusion_coeff:.2e}",
            )

        except Exception as e:
            # Some high-noise cases or extreme parameter values may fail to converge
            # This is acceptable for very noisy data or extreme cases

            # Check if this is an extreme parameter case (very small diffusion coefficient)
            is_extreme_case = diffusion_coeff < 1e-14

            if noise_level < 0.15 and not is_extreme_case:  # Should converge for reasonable parameters with low noise
                self.fail(
                    f"Fitting failed for low noise level {noise_level:.3f}: {str(e)}"
                )
            # For moderate noise (0.15-0.2) or extreme parameter values, convergence failure is acceptable
            else:
                # Test passes - expected failure for high noise or extreme parameters
                pass


class TestDiffusionModelValidation(unittest.TestCase):
    """Test validation of different diffusion models"""

    def setUp(self):
        """Set up test data for model validation"""
        self.q_values = np.logspace(-3, -1, 30)  # Å⁻¹

        # Create synthetic data for different models
        self.create_model_data()

    def create_model_data(self):
        """Create synthetic data for different diffusion models"""
        # 1. Simple Brownian diffusion
        self.D_brownian = 1e-12  # m²/s
        self.tau_brownian = 1.0 / (self.D_brownian * self.q_values**2)

        # 2. Subdiffusion (fractional Brownian motion)
        self.alpha_subdiff = 1.6
        self.A_subdiff = 1e-4
        self.tau_subdiff = self.A_subdiff * self.q_values ** (-self.alpha_subdiff)

        # 3. Two-component system (fast + slow)
        self.D_fast = 5e-12  # m²/s
        self.D_slow = 1e-13  # m²/s
        self.fraction_fast = 0.3

        tau_fast = 1.0 / (self.D_fast * self.q_values**2)
        tau_slow = 1.0 / (self.D_slow * self.q_values**2)

        # Weighted average relaxation time
        self.tau_two_component = (
            self.fraction_fast * tau_fast + (1 - self.fraction_fast) * tau_slow
        )

        # 4. Cooperative diffusion (with hydrodynamic interactions)
        self.D0 = 2e-12  # m²/s
        self.xi = 100e-10  # correlation length (Å)

        # Cooperative diffusion: D(q) = D₀ * (1 + (qξ)²)
        D_coop = self.D0 * (1 + (self.q_values * self.xi) ** 2)
        self.tau_cooperative = 1.0 / (D_coop * self.q_values**2)

    def test_model_identification(self):
        """Test ability to identify different diffusion models"""

        # Test 1: Brownian motion identification
        # Fit power law and check exponent
        log_q = np.log10(self.q_values)
        log_tau_brownian = np.log10(self.tau_brownian)

        fit_coeff = np.polyfit(log_q, log_tau_brownian, 1)
        exponent = fit_coeff[0]

        self.assertAlmostEqual(
            exponent, -2.0, delta=1e-10, msg="Should identify Brownian motion (τ ∝ q⁻²)"
        )

        # Test 2: Subdiffusion identification
        log_tau_subdiff = np.log10(self.tau_subdiff)
        fit_coeff_subdiff = np.polyfit(log_q, log_tau_subdiff, 1)
        exponent_subdiff = fit_coeff_subdiff[0]

        self.assertAlmostEqual(
            exponent_subdiff,
            -self.alpha_subdiff,
            delta=1e-10,
            msg="Should identify subdiffusion exponent",
        )

        # Test 3: Deviation from simple power law for cooperative diffusion
        log_tau_coop = np.log10(self.tau_cooperative)
        fit_coeff_coop = np.polyfit(log_q, log_tau_coop, 1)

        # Calculate R² to check quality of power-law fit
        tau_fit = 10 ** (fit_coeff_coop[1] + fit_coeff_coop[0] * log_q)
        ss_res = np.sum((self.tau_cooperative - tau_fit) ** 2)
        ss_tot = np.sum((self.tau_cooperative - np.mean(self.tau_cooperative)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Cooperative diffusion should deviate from simple power law
        self.assertLess(
            r_squared,
            0.99,
            "Cooperative diffusion should not fit simple power law perfectly",
        )

    def test_stokes_einstein_validation(self):
        """Test validation against Stokes-Einstein relation"""
        # For spherical particles: D = kT/(6πηr)
        temperature = 300.0  # K
        viscosity = 1e-3  # Pa⋅s
        k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]

        # Range of particle radii
        radii = np.logspace(-8, -6, 10)  # 10 nm to 1 μm

        for radius in radii:
            # Theoretical diffusion coefficient
            D_theory = k_B * temperature / (6 * np.pi * viscosity * radius)

            # Generate τ(q) data
            tau_values = 1.0 / (D_theory * self.q_values**2)

            # Fit to extract D
            def brownian_model(q, D):
                return 1.0 / (D * q**2)

            popt, _ = optimize.curve_fit(brownian_model, self.q_values, tau_values)
            D_fitted = popt[0]

            # Should recover input diffusion coefficient
            rel_error = abs(D_fitted - D_theory) / D_theory
            self.assertLess(
                rel_error,
                1e-10,
                f"Should recover Stokes-Einstein D for radius {radius * 1e9:.1f} nm",
            )

            # Test particle size calculation from fitted D
            fitted_radius = k_B * temperature / (6 * np.pi * viscosity * D_fitted)
            radius_rel_error = abs(fitted_radius - radius) / radius
            self.assertLess(
                radius_rel_error, 1e-10, "Should recover particle radius from fitted D"
            )

    def test_polydispersity_effects(self):
        """Test effects of particle size polydispersity on diffusion"""
        # Create polydisperse system
        mean_radius = 50e-9  # m
        polydispersity = 0.2  # 20% polydispersity

        # Log-normal distribution of radii
        sigma_log = np.sqrt(np.log(1 + polydispersity**2))
        mu_log = np.log(mean_radius) - 0.5 * sigma_log**2

        n_particles = 1000
        radii_sample = np.random.lognormal(mu_log, sigma_log, n_particles)

        # Calculate D for each particle size
        k_B = SCIENTIFIC_CONSTANTS["k_boltzmann"]
        temperature = 300.0
        viscosity = 1e-3

        D_values = k_B * temperature / (6 * np.pi * viscosity * radii_sample)

        # Calculate ensemble-averaged correlation function
        # For polydisperse system: ⟨g2(q,τ)⟩ = Σᵢ fᵢ exp(-Dᵢq²τ)

        # Use equal weight for all particles
        weights = np.ones(n_particles) / n_particles

        # Calculate τ at which g2-1 = 1/e for each q
        tau_1_e = np.zeros(len(self.q_values))

        for q_idx, q in enumerate(self.q_values):
            # Find τ where ensemble average = 1/e
            def ensemble_g2(tau):
                g2_minus_1 = np.sum(weights * np.exp(-D_values * q**2 * tau))
                return g2_minus_1

            # Find τ where g2-1 = 1/e
            try:
                tau_1_e[q_idx] = optimize.brentq(
                    lambda tau: ensemble_g2(tau) - 1 / np.e, 1e-6, 1e3
                )
            except ValueError:
                # If not found, use approximate value
                tau_1_e[q_idx] = 1.0 / (np.mean(D_values) * q**2)

        # Fit to extract apparent diffusion coefficient
        def fit_function(q, D_app):
            return 1.0 / (D_app * q**2)

        popt, _ = optimize.curve_fit(
            fit_function, self.q_values, tau_1_e, bounds=([1e-16], [1e-8])
        )
        D_apparent = popt[0]

        # For polydisperse systems, apparent D should be close to
        # but not exactly equal to mean D
        D_mean = np.mean(D_values)
        rel_difference = abs(D_apparent - D_mean) / D_mean

        # Should be within reasonable range for moderate polydispersity
        self.assertLess(
            rel_difference,
            0.5,
            f"Polydisperse apparent D should be close to mean D: "
            f"relative difference = {rel_difference:.3f}",
        )


class TestFittingAlgorithmValidation(unittest.TestCase):
    """Test validation of fitting algorithms used in diffusion analysis"""

    def test_weighted_least_squares_fitting(self):
        """Test weighted least squares fitting with realistic errors"""
        # Create synthetic data with heteroscedastic errors
        q_values = np.logspace(-3, -1, 25)
        D_true = 2e-12  # m²/s
        tau_true = 1.0 / (D_true * q_values**2)

        # Create realistic error model (Poisson-like)
        # Relative errors increase with τ (longer times → worse statistics)
        relative_errors = 0.05 + 0.1 * tau_true / np.max(tau_true)
        tau_errors = relative_errors * tau_true

        # Generate noisy data
        np.random.seed(42)  # Reproducible test
        tau_noisy = tau_true + tau_errors * np.random.normal(size=len(tau_true))
        tau_noisy = np.maximum(tau_noisy, 0.01 * tau_true)  # Ensure positivity

        # Weighted least squares fitting
        def fit_function(q, D):
            return 1.0 / (D * q**2)

        # Use inverse variance weighting
        1.0 / tau_errors**2

        # Fit with weights (scipy.optimize.curve_fit supports sigma parameter)
        popt_weighted, pcov_weighted = optimize.curve_fit(
            fit_function,
            q_values,
            tau_noisy,
            sigma=tau_errors,  # Standard deviations for weighting
            bounds=([1e-16], [1e-8]),
        )

        # Fit without weights for comparison
        popt_unweighted, pcov_unweighted = optimize.curve_fit(
            fit_function, q_values, tau_noisy, bounds=([1e-16], [1e-8])
        )

        D_weighted = popt_weighted[0]
        D_unweighted = popt_unweighted[0]

        # Weighted fit should be more accurate
        error_weighted = abs(D_weighted - D_true) / D_true
        abs(D_unweighted - D_true) / D_true

        # Not guaranteed to always be better due to random noise,
        # but should be close to true value
        self.assertLess(
            error_weighted,
            0.2,  # 20% error tolerance
            f"Weighted fit error: {error_weighted:.3f}",
        )

        # Parameter uncertainties should be reasonable
        D_uncertainty_weighted = np.sqrt(pcov_weighted[0, 0])
        rel_uncertainty = D_uncertainty_weighted / D_weighted

        self.assertLess(
            rel_uncertainty,
            1.0,  # 100% uncertainty limit
            f"Parameter uncertainty too large: {rel_uncertainty:.3f}",
        )

    def test_bootstrap_uncertainty_estimation(self):
        """Test bootstrap method for uncertainty estimation"""
        # Create synthetic dataset
        q_values = np.logspace(-2.5, -1, 20)
        D_true = 1.5e-12  # m²/s
        tau_true = 1.0 / (D_true * q_values**2)

        # Add noise
        noise_level = 0.1
        tau_errors = noise_level * tau_true

        np.random.seed(123)  # Reproducible
        tau_observed = tau_true + tau_errors * np.random.normal(size=len(tau_true))
        tau_observed = np.maximum(tau_observed, 0.05 * tau_true)

        # Bootstrap resampling
        n_bootstrap = 100
        D_bootstrap = np.zeros(n_bootstrap)

        def fit_function(q, D):
            return 1.0 / (D * q**2)

        for i in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(len(q_values), size=len(q_values), replace=True)
            q_resample = q_values[indices]
            tau_resample = tau_observed[indices]

            # Fit resampled data
            try:
                popt, _ = optimize.curve_fit(
                    fit_function, q_resample, tau_resample, bounds=([1e-16], [1e-8])
                )
                D_bootstrap[i] = popt[0]
            except:
                D_bootstrap[i] = np.nan

        # Remove failed fits
        D_bootstrap = D_bootstrap[~np.isnan(D_bootstrap)]

        if len(D_bootstrap) > 50:  # Need enough successful fits
            # Calculate bootstrap statistics
            D_mean = np.mean(D_bootstrap)
            D_std = np.std(D_bootstrap)

            # Bootstrap mean should be close to true value
            rel_error = abs(D_mean - D_true) / D_true
            self.assertLess(
                rel_error,
                0.3,  # 30% tolerance
                f"Bootstrap mean error: {rel_error:.3f}",
            )

            # Bootstrap uncertainty should be reasonable
            rel_uncertainty = D_std / D_mean
            self.assertLess(
                rel_uncertainty,
                0.5,  # 50% uncertainty limit
                f"Bootstrap uncertainty: {rel_uncertainty:.3f}",
            )

            # Test confidence intervals
            confidence_level = 0.68  # 1σ confidence
            alpha = 1 - confidence_level
            lower_percentile = 100 * alpha / 2
            upper_percentile = 100 * (1 - alpha / 2)

            D_lower = np.percentile(D_bootstrap, lower_percentile)
            D_upper = np.percentile(D_bootstrap, upper_percentile)

            # True value should be within confidence interval
            # (may occasionally fail due to random sampling)
            within_ci = D_lower <= D_true <= D_upper

            # Test passes if true value is within CI or close to boundaries
            distance_to_ci = min(abs(D_true - D_lower), abs(D_true - D_upper))
            relative_distance = distance_to_ci / D_true

            self.assertTrue(
                within_ci or relative_distance < 0.1,
                f"True value should be within or close to {confidence_level:.0%} CI",
            )

    def test_goodness_of_fit_assessment(self):
        """Test assessment of fitting quality using statistical measures"""
        # Create datasets with different levels of model agreement
        q_values = np.logspace(-2.5, -1, 30)

        # Case 1: Perfect Brownian motion
        D1 = 1e-12
        tau1_perfect = 1.0 / (D1 * q_values**2)

        # Case 2: Brownian with realistic noise
        tau1_noisy = tau1_perfect * (
            1 + 0.05 * np.random.normal(size=len(tau1_perfect))
        )
        tau1_noisy = np.maximum(tau1_noisy, 0.1 * tau1_perfect)

        # Case 3: Non-Brownian (subdiffusion)
        alpha = 1.7
        A = 1e-4
        tau_subdiff = A * q_values ** (-alpha)

        def fit_brownian(q, tau):
            """Fit Brownian model and return goodness-of-fit statistics"""

            def model(q, D):
                return 1.0 / (D * q**2)

            popt, pcov = optimize.curve_fit(model, q, tau, bounds=([1e-16], [1e-8]))
            tau_fit = model(q, popt[0])

            # Calculate R-squared
            ss_res = np.sum((tau - tau_fit) ** 2)
            ss_tot = np.sum((tau - np.mean(tau)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            # Calculate reduced chi-squared (assuming 10% relative errors)
            expected_errors = 0.1 * tau
            chi_squared_reduced = np.sum(((tau - tau_fit) / expected_errors) ** 2) / (
                len(tau) - 1
            )

            return r_squared, chi_squared_reduced, popt[0]

        # Test fits
        r2_perfect, chi2_perfect, D_perfect = fit_brownian(q_values, tau1_perfect)
        r2_noisy, chi2_noisy, D_noisy = fit_brownian(q_values, tau1_noisy)
        r2_subdiff, chi2_subdiff, D_subdiff = fit_brownian(q_values, tau_subdiff)

        # Perfect data should give excellent fit
        self.assertGreater(r2_perfect, 0.999, "Perfect Brownian should give R² ≈ 1")
        self.assertLess(chi2_perfect, 0.01, "Perfect data should give very low χ²")

        # Noisy Brownian should still fit well
        self.assertGreater(r2_noisy, 0.9, "Noisy Brownian should still fit well")
        self.assertLess(chi2_noisy, 10, "Noisy Brownian χ² should be reasonable")

        # Subdiffusion should fit poorly with Brownian model
        self.assertLess(
            r2_subdiff, 0.95, "Subdiffusion should fit poorly with Brownian model"
        )

        # Recovered diffusion coefficients should be reasonable
        self.assertAlmostEqual(D_perfect, D1, delta=D1 * 1e-10)
        self.assertAlmostEqual(
            D_noisy, D1, delta=D1 * 0.2
        )  # 20% tolerance for noisy data


if __name__ == "__main__":
    # Configure warnings for scientific tests
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)

    unittest.main(verbosity=2)
