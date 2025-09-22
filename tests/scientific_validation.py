#!/usr/bin/env python3
"""
Scientific Accuracy Validation Tests for XPCS Toolkit Core Algorithms.

This module provides comprehensive validation tests for the scientific accuracy
of core XPCS algorithms against known theoretical results and benchmarks.
"""

import json
import sys
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy import optimize, signal, stats
    from scipy.special import factorial, gamma

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


@dataclass
class ValidationResult:
    """Result of a scientific validation test."""

    test_name: str
    passed: bool
    theoretical_value: float
    computed_value: float
    relative_error: float
    absolute_error: float
    tolerance: float
    details: dict[str, Any]


@dataclass
class BenchmarkData:
    """Benchmark dataset for validation."""

    name: str
    description: str
    input_data: np.ndarray
    expected_output: np.ndarray
    parameters: dict[str, Any]
    tolerance: float


class ScientificValidationFramework:
    """Framework for validating scientific accuracy of XPCS algorithms."""

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.validation_results: list[ValidationResult] = []
        self.benchmark_datasets: list[BenchmarkData] = []

    def validate_correlation_functions(self) -> list[ValidationResult]:
        """Validate correlation function calculations against theoretical results."""
        results = []

        # Test 1: White noise should give flat G2
        white_noise = np.random.randn(10000)
        g2_white = self._calculate_g2_correlation(white_noise)

        theoretical_g2_white = 1.0  # For white noise, G2 should approach 1
        if len(g2_white) > 10:
            computed_g2_white = np.mean(g2_white[5:])  # Skip initial points
            error = abs(computed_g2_white - theoretical_g2_white)

            result = ValidationResult(
                test_name="white_noise_g2_flatness",
                passed=bool(error < 0.1),  # Relaxed tolerance for statistical noise
                theoretical_value=float(theoretical_g2_white),
                computed_value=float(computed_g2_white),
                relative_error=float(error / theoretical_g2_white),
                absolute_error=float(error),
                tolerance=0.1,
                details={"description": "G2 of white noise should approach 1"},
            )
            results.append(result)

        # Test 2: Exponential decay validation
        tau_values = np.arange(1, 51)
        decay_rate = 0.1
        theoretical_g2_exp = 1.0 + np.exp(-decay_rate * tau_values)

        # Generate synthetic exponential correlation data
        synthetic_signal = self._generate_exponential_correlation_signal(
            10000, decay_rate
        )
        computed_g2_exp = self._calculate_g2_correlation(synthetic_signal)[
            : len(tau_values)
        ]

        if len(computed_g2_exp) >= len(theoretical_g2_exp):
            mse = np.mean((computed_g2_exp - theoretical_g2_exp) ** 2)

            result = ValidationResult(
                test_name="exponential_decay_g2",
                passed=bool(mse < 0.01),
                theoretical_value=float(np.mean(theoretical_g2_exp)),
                computed_value=float(np.mean(computed_g2_exp)),
                relative_error=float(mse / np.mean(theoretical_g2_exp**2)),
                absolute_error=float(mse),
                tolerance=0.01,
                details={
                    "description": "G2 exponential decay validation",
                    "decay_rate": decay_rate,
                    "mse": float(mse),
                },
            )
            results.append(result)

        # Test 3: G2(0) >= G2(tau) for all tau > 0 (monotonicity)
        test_signal = self._generate_realistic_speckle_signal(5000)
        g2_monotonic = self._calculate_g2_correlation(test_signal)

        if len(g2_monotonic) > 10:
            monotonic_violations = np.sum(g2_monotonic[1:10] > g2_monotonic[0])

            result = ValidationResult(
                test_name="g2_monotonicity",
                passed=bool(
                    monotonic_violations <= 2
                ),  # Allow some statistical fluctuation
                theoretical_value=0.0,  # No violations expected
                computed_value=float(monotonic_violations),
                relative_error=float(monotonic_violations) / 10,
                absolute_error=float(monotonic_violations),
                tolerance=2.0,
                details={
                    "description": "G2 should be monotonically decreasing",
                    "violations": int(monotonic_violations),
                    "total_points": 10,
                },
            )
            results.append(result)

        return results

    def validate_fitting_algorithms(self) -> list[ValidationResult]:
        """Validate fitting algorithm accuracy against known functions."""
        results = []

        # Test 1: Single exponential fitting
        x_data = np.linspace(0, 5, 100)
        true_amplitude = 2.5
        true_decay = 0.3
        true_offset = 1.0

        y_true = true_amplitude * np.exp(-true_decay * x_data) + true_offset
        y_noisy = y_true + np.random.normal(0, 0.05, len(y_true))

        fitted_params = self._fit_single_exponential(x_data, y_noisy)

        if fitted_params is not None:
            amplitude_error = (
                abs(fitted_params["amplitude"] - true_amplitude) / true_amplitude
            )
            decay_error = abs(fitted_params["decay_rate"] - true_decay) / true_decay

            result = ValidationResult(
                test_name="single_exponential_fitting",
                passed=amplitude_error < 0.1 and decay_error < 0.1,
                theoretical_value=true_amplitude,
                computed_value=fitted_params["amplitude"],
                relative_error=amplitude_error,
                absolute_error=abs(fitted_params["amplitude"] - true_amplitude),
                tolerance=0.1,
                details={
                    "decay_rate_error": decay_error,
                    "true_decay": true_decay,
                    "fitted_decay": fitted_params["decay_rate"],
                    "r_squared": fitted_params.get("r_squared", 0),
                },
            )
            results.append(result)

        # Test 2: Double exponential fitting
        true_a1, true_a2 = 1.5, 0.8
        true_tau1, true_tau2 = 0.5, 2.0

        y_double = (
            true_a1 * np.exp(-x_data / true_tau1)
            + true_a2 * np.exp(-x_data / true_tau2)
            + true_offset
        )
        y_double_noisy = y_double + np.random.normal(0, 0.05, len(y_double))

        fitted_double = self._fit_double_exponential(x_data, y_double_noisy)

        if fitted_double is not None:
            # Check if either combination of parameters matches (order may vary)
            param_sets = [
                (true_a1, true_tau1, true_a2, true_tau2),
                (true_a2, true_tau2, true_a1, true_tau1),
            ]

            best_error = float("inf")
            for true_set in param_sets:
                error = (
                    abs(fitted_double["a1"] - true_set[0]) / true_set[0]
                    + abs(fitted_double["tau1"] - true_set[1]) / true_set[1]
                    + abs(fitted_double["a2"] - true_set[2]) / true_set[2]
                    + abs(fitted_double["tau2"] - true_set[3]) / true_set[3]
                ) / 4
                best_error = min(best_error, error)

            result = ValidationResult(
                test_name="double_exponential_fitting",
                passed=best_error < 0.2,  # More lenient for complex fitting
                theoretical_value=true_a1,
                computed_value=fitted_double["a1"],
                relative_error=best_error,
                absolute_error=best_error * true_a1,
                tolerance=0.2,
                details={
                    "fitted_params": fitted_double,
                    "true_params": {
                        "a1": true_a1,
                        "tau1": true_tau1,
                        "a2": true_a2,
                        "tau2": true_tau2,
                    },
                    "best_relative_error": best_error,
                },
            )
            results.append(result)

        return results

    def validate_statistical_properties(self) -> list[ValidationResult]:
        """Validate statistical calculations against theoretical distributions."""
        results = []

        # Test 1: Gamma distribution validation (speckle statistics)
        shape_param = 2.0
        scale_param = 1.5
        sample_size = 10000

        # Generate gamma-distributed data
        gamma_data = np.random.gamma(shape_param, scale_param, sample_size)

        # Calculate sample statistics
        sample_mean = np.mean(gamma_data)
        sample_var = np.var(gamma_data)

        # Theoretical values
        theoretical_mean = shape_param * scale_param
        theoretical_var = shape_param * scale_param**2

        mean_error = abs(sample_mean - theoretical_mean) / theoretical_mean
        var_error = abs(sample_var - theoretical_var) / theoretical_var

        result = ValidationResult(
            test_name="gamma_distribution_statistics",
            passed=mean_error < 0.05 and var_error < 0.1,
            theoretical_value=theoretical_mean,
            computed_value=sample_mean,
            relative_error=mean_error,
            absolute_error=abs(sample_mean - theoretical_mean),
            tolerance=0.05,
            details={
                "variance_error": var_error,
                "theoretical_var": theoretical_var,
                "sample_var": sample_var,
                "sample_size": sample_size,
            },
        )
        results.append(result)

        # Test 2: Contrast calculation validation
        intensities = np.random.gamma(1.0, 10.0, 5000)  # Single speckle mode
        theoretical_contrast = 1.0  # For fully developed speckle

        computed_contrast = np.std(intensities) / np.mean(intensities)
        contrast_error = (
            abs(computed_contrast - theoretical_contrast) / theoretical_contrast
        )

        result = ValidationResult(
            test_name="speckle_contrast_validation",
            passed=contrast_error < 0.1,
            theoretical_value=theoretical_contrast,
            computed_value=computed_contrast,
            relative_error=contrast_error,
            absolute_error=abs(computed_contrast - theoretical_contrast),
            tolerance=0.1,
            details={
                "description": "Fully developed speckle should have contrast = 1",
                "intensity_mean": np.mean(intensities),
                "intensity_std": np.std(intensities),
            },
        )
        results.append(result)

        return results

    def validate_fourier_transforms(self) -> list[ValidationResult]:
        """Validate Fourier transform calculations."""
        results = []

        # Test 1: FFT of known analytical function
        N = 1024
        t = np.linspace(0, 1, N, endpoint=False)
        frequency = 50.0
        amplitude = 2.0

        # Create pure sinusoid
        signal = amplitude * np.sin(2 * np.pi * frequency * t)
        fft_result = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, t[1] - t[0])

        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft_result))
        detected_frequency = abs(freqs[peak_idx])

        freq_error = abs(detected_frequency - frequency) / frequency

        result = ValidationResult(
            test_name="fft_frequency_detection",
            passed=freq_error < 0.01,
            theoretical_value=frequency,
            computed_value=detected_frequency,
            relative_error=freq_error,
            absolute_error=abs(detected_frequency - frequency),
            tolerance=0.01,
            details={
                "signal_amplitude": amplitude,
                "fft_peak_amplitude": np.abs(fft_result[peak_idx]),
                "peak_index": peak_idx,
            },
        )
        results.append(result)

        # Test 2: Parseval's theorem validation
        random_signal = np.random.randn(512)
        time_energy = np.sum(random_signal**2)
        freq_energy = np.sum(np.abs(np.fft.fft(random_signal)) ** 2) / len(
            random_signal
        )

        parseval_error = abs(time_energy - freq_energy) / time_energy

        result = ValidationResult(
            test_name="parseval_theorem_validation",
            passed=parseval_error < 1e-10,
            theoretical_value=time_energy,
            computed_value=freq_energy,
            relative_error=parseval_error,
            absolute_error=abs(time_energy - freq_energy),
            tolerance=1e-10,
            details={
                "description": "Energy conservation in Fourier transform",
                "signal_length": len(random_signal),
            },
        )
        results.append(result)

        return results

    def validate_q_space_calculations(self) -> list[ValidationResult]:
        """Validate q-space and scattering vector calculations."""
        results = []

        # Test 1: Q-vector magnitude calculation
        wavelength = 1.24e-10  # meters (10 keV X-rays)
        pixel_size = 75e-6  # meters
        detector_distance = 2.0  # meters
        beam_center = (512, 512)

        # Calculate q for a pixel at known position
        pixel_x, pixel_y = 612, 612  # 100 pixels from center

        # Theoretical calculation
        dx = (pixel_x - beam_center[0]) * pixel_size
        dy = (pixel_y - beam_center[1]) * pixel_size
        r = np.sqrt(dx**2 + dy**2)
        scattering_angle = np.arctan(r / detector_distance)
        theoretical_q = 4 * np.pi * np.sin(scattering_angle / 2) / wavelength

        # Mock calculation (replace with actual implementation)
        computed_q = self._calculate_q_magnitude(
            pixel_x, pixel_y, beam_center, pixel_size, detector_distance, wavelength
        )

        q_error = abs(computed_q - theoretical_q) / theoretical_q

        result = ValidationResult(
            test_name="q_vector_magnitude_calculation",
            passed=q_error < 1e-6,
            theoretical_value=theoretical_q,
            computed_value=computed_q,
            relative_error=q_error,
            absolute_error=abs(computed_q - theoretical_q),
            tolerance=1e-6,
            details={
                "wavelength": wavelength,
                "pixel_size": pixel_size,
                "detector_distance": detector_distance,
                "scattering_angle_rad": scattering_angle,
                "pixel_position": (pixel_x, pixel_y),
            },
        )
        results.append(result)

        return results

    def validate_intensity_normalization(self) -> list[ValidationResult]:
        """Validate intensity normalization procedures."""
        results = []

        # Test 1: Background subtraction
        signal_level = 1000.0
        background_level = 50.0
        noise_level = 10.0

        raw_intensities = (
            signal_level + background_level + np.random.normal(0, noise_level, 1000)
        )

        corrected_intensities = self._subtract_background(
            raw_intensities, background_level
        )

        theoretical_mean = signal_level
        computed_mean = np.mean(corrected_intensities)
        normalization_error = abs(computed_mean - theoretical_mean) / theoretical_mean

        result = ValidationResult(
            test_name="background_subtraction_accuracy",
            passed=normalization_error < 0.05,
            theoretical_value=theoretical_mean,
            computed_value=computed_mean,
            relative_error=normalization_error,
            absolute_error=abs(computed_mean - theoretical_mean),
            tolerance=0.05,
            details={
                "signal_level": signal_level,
                "background_level": background_level,
                "noise_level": noise_level,
            },
        )
        results.append(result)

        return results

    def run_comprehensive_validation(self) -> dict[str, list[ValidationResult]]:
        """Run all validation tests and return comprehensive results."""
        validation_suite = {
            "correlation_functions": self.validate_correlation_functions(),
            "fitting_algorithms": self.validate_fitting_algorithms(),
            "statistical_properties": self.validate_statistical_properties(),
            "fourier_transforms": self.validate_fourier_transforms(),
            "q_space_calculations": self.validate_q_space_calculations(),
            "intensity_normalization": self.validate_intensity_normalization(),
        }

        # Store all results
        self.validation_results = []
        for category_results in validation_suite.values():
            self.validation_results.extend(category_results)

        return validation_suite

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            self.run_comprehensive_validation()

        report = ["Scientific Accuracy Validation Report", "=" * 60, ""]

        # Summary statistics
        total_tests = len(self.validation_results)
        passed_tests = sum(1 for r in self.validation_results if r.passed)
        failed_tests = total_tests - passed_tests

        report.extend(
            [
                "SUMMARY:",
                f"  Total Tests: {total_tests}",
                f"  Passed: {passed_tests}",
                f"  Failed: {failed_tests}",
                f"  Success Rate: {passed_tests / total_tests * 100:.1f}%",
                "",
            ]
        )

        # Group results by category
        categories = {}
        for result in self.validation_results:
            category = (
                result.test_name.split("_")[0] if "_" in result.test_name else "general"
            )
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Report by category
        for category, results in categories.items():
            report.append(f"{category.upper().replace('_', ' ')} VALIDATION:")
            report.append("-" * 40)

            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                report.append(f"  {status}: {result.test_name}")
                report.append(f"    Theoretical: {result.theoretical_value:.6g}")
                report.append(f"    Computed:    {result.computed_value:.6g}")
                report.append(f"    Rel. Error:  {result.relative_error:.2e}")
                report.append(f"    Tolerance:   {result.tolerance:.2e}")

                if not result.passed:
                    report.append(f"    >>> FAILURE DETAILS: {result.details}")

                report.append("")

            category_passed = sum(1 for r in results if r.passed)
            report.append(
                f"  Category Success: {category_passed}/{len(results)} "
                f"({category_passed / len(results) * 100:.1f}%)"
            )
            report.append("")

        # Overall assessment
        if passed_tests == total_tests:
            report.append("🎉 ALL VALIDATION TESTS PASSED!")
            report.append("   Scientific accuracy is verified across all algorithms.")
        elif passed_tests / total_tests >= 0.9:
            report.append("✅ EXCELLENT: >90% of validation tests passed.")
            report.append("   Minor issues detected but overall accuracy is high.")
        elif passed_tests / total_tests >= 0.75:
            report.append("⚠️  GOOD: 75-90% of validation tests passed.")
            report.append("   Some accuracy issues require investigation.")
        else:
            report.append("🚨 ATTENTION REQUIRED: <75% validation success.")
            report.append("   Significant accuracy issues detected.")

        return "\n".join(report)

    # Helper methods for calculations (replace with actual XPCS implementations)

    def _calculate_g2_correlation(self, intensities: np.ndarray) -> np.ndarray:
        """Calculate G2 correlation function (mock implementation)."""
        n = len(intensities)
        max_lag = min(n // 4, 100)
        g2 = np.zeros(max_lag)

        mean_intensity = np.mean(intensities)

        for lag in range(max_lag):
            if lag == 0:
                g2[lag] = np.mean(intensities**2) / (mean_intensity**2)
            else:
                correlation = np.mean(intensities[:-lag] * intensities[lag:])
                g2[lag] = correlation / (mean_intensity**2)

        return g2

    def _generate_exponential_correlation_signal(
        self, n_points: int, decay_rate: float
    ) -> np.ndarray:
        """Generate signal with exponential correlation."""
        # AR(1) process with specified decay
        alpha = np.exp(-decay_rate)
        noise = np.random.randn(n_points)
        signal = np.zeros(n_points)
        signal[0] = noise[0]

        for i in range(1, n_points):
            signal[i] = alpha * signal[i - 1] + np.sqrt(1 - alpha**2) * noise[i]

        # Convert to positive intensities
        signal = np.exp(signal - np.mean(signal))
        return signal

    def _generate_realistic_speckle_signal(self, n_points: int) -> np.ndarray:
        """Generate realistic speckle intensity signal."""
        # Gamma-distributed intensities with exponential correlation
        base_signal = self._generate_exponential_correlation_signal(n_points, 0.1)
        # Transform to gamma distribution
        uniform_signal = stats.norm.cdf(base_signal)
        gamma_signal = stats.gamma.ppf(uniform_signal, a=1.0, scale=10.0)
        return gamma_signal

    def _fit_single_exponential(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> dict[str, float] | None:
        """Fit single exponential function (mock implementation)."""
        if not SCIPY_AVAILABLE:
            return None

        try:

            def exp_func(x, a, b, c):
                return a * np.exp(-b * x) + c

            # Initial guess
            p0 = [np.max(y_data) - np.min(y_data), 0.5, np.min(y_data)]

            popt, _pcov = optimize.curve_fit(exp_func, x_data, y_data, p0=p0)

            # Calculate R-squared
            y_pred = exp_func(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)

            return {
                "amplitude": popt[0],
                "decay_rate": popt[1],
                "offset": popt[2],
                "r_squared": r_squared,
            }
        except:
            return None

    def _fit_double_exponential(
        self, x_data: np.ndarray, y_data: np.ndarray
    ) -> dict[str, float] | None:
        """Fit double exponential function (mock implementation)."""
        if not SCIPY_AVAILABLE:
            return None

        try:

            def double_exp_func(x, a1, tau1, a2, tau2, c):
                return a1 * np.exp(-x / tau1) + a2 * np.exp(-x / tau2) + c

            # Initial guess
            p0 = [0.5, 1.0, 0.5, 3.0, np.min(y_data)]

            popt, _pcov = optimize.curve_fit(double_exp_func, x_data, y_data, p0=p0)

            return {
                "a1": popt[0],
                "tau1": popt[1],
                "a2": popt[2],
                "tau2": popt[3],
                "offset": popt[4],
            }
        except:
            return None

    def _calculate_q_magnitude(
        self,
        pixel_x: int,
        pixel_y: int,
        beam_center: tuple[int, int],
        pixel_size: float,
        detector_distance: float,
        wavelength: float,
    ) -> float:
        """Calculate q-vector magnitude (mock implementation)."""
        dx = (pixel_x - beam_center[0]) * pixel_size
        dy = (pixel_y - beam_center[1]) * pixel_size
        r = np.sqrt(dx**2 + dy**2)
        scattering_angle = np.arctan(r / detector_distance)
        q_magnitude = 4 * np.pi * np.sin(scattering_angle / 2) / wavelength
        return q_magnitude

    def _subtract_background(
        self, intensities: np.ndarray, background: float
    ) -> np.ndarray:
        """Subtract background from intensities (mock implementation)."""
        return intensities - background


class ScientificValidationTestSuite(unittest.TestCase):
    """Unit test suite for scientific validation framework."""

    def setUp(self):
        self.validator = ScientificValidationFramework(tolerance=1e-6)

    def test_validation_framework_initialization(self):
        """Test that the validation framework initializes correctly."""
        self.assertEqual(self.validator.tolerance, 1e-6)
        self.assertEqual(len(self.validator.validation_results), 0)

    def test_correlation_function_validation(self):
        """Test correlation function validation methods."""
        results = self.validator.validate_correlation_functions()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIsInstance(result, ValidationResult)
            self.assertIsInstance(result.passed, bool)

    def test_fitting_algorithm_validation(self):
        """Test fitting algorithm validation methods."""
        results = self.validator.validate_fitting_algorithms()
        self.assertIsInstance(results, list)

        for result in results:
            self.assertIsInstance(result, ValidationResult)

    def test_statistical_property_validation(self):
        """Test statistical property validation methods."""
        results = self.validator.validate_statistical_properties()
        self.assertIsInstance(results, list)

        for result in results:
            self.assertIsInstance(result, ValidationResult)

    def test_fourier_transform_validation(self):
        """Test Fourier transform validation methods."""
        results = self.validator.validate_fourier_transforms()
        self.assertIsInstance(results, list)

        # At least Parseval's theorem should always pass
        parseval_results = [r for r in results if "parseval" in r.test_name]
        if parseval_results:
            self.assertTrue(any(r.passed for r in parseval_results))

    def test_comprehensive_validation_execution(self):
        """Test that comprehensive validation runs without errors."""
        validation_suite = self.validator.run_comprehensive_validation()

        self.assertIsInstance(validation_suite, dict)
        self.assertGreater(len(validation_suite), 0)

        # Check that we have results
        total_results = sum(len(results) for results in validation_suite.values())
        self.assertGreater(total_results, 0)

    def test_validation_report_generation(self):
        """Test validation report generation."""
        report = self.validator.generate_validation_report()

        self.assertIsInstance(report, str)
        self.assertIn("Scientific Accuracy Validation Report", report)
        self.assertIn("SUMMARY:", report)

    def test_helper_method_functionality(self):
        """Test that helper methods work correctly."""
        # Test G2 calculation
        test_signal = np.random.randn(100) + 10
        g2_result = self.validator._calculate_g2_correlation(test_signal)
        self.assertGreater(len(g2_result), 0)
        self.assertTrue(np.all(np.isfinite(g2_result)))

        # Test q-magnitude calculation
        q_mag = self.validator._calculate_q_magnitude(
            100, 100, (50, 50), 75e-6, 1.0, 1e-10
        )
        self.assertGreater(q_mag, 0)
        self.assertTrue(np.isfinite(q_mag))


if __name__ == "__main__":
    """Command-line interface for scientific validation."""
    import argparse

    parser = argparse.ArgumentParser(description="XPCS Scientific Accuracy Validation")
    parser.add_argument(
        "--tolerance", type=float, default=1e-6, help="Validation tolerance"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "correlation",
            "fitting",
            "statistics",
            "fourier",
            "qspace",
            "normalization",
        ],
        help="Run specific validation category",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed validation report"
    )
    parser.add_argument(
        "--json-output", type=str, metavar="FILE", help="Save results as JSON file"
    )

    args = parser.parse_args()

    validator = ScientificValidationFramework(tolerance=args.tolerance)

    if args.category:
        # Run specific category
        category_map = {
            "correlation": validator.validate_correlation_functions,
            "fitting": validator.validate_fitting_algorithms,
            "statistics": validator.validate_statistical_properties,
            "fourier": validator.validate_fourier_transforms,
            "qspace": validator.validate_q_space_calculations,
            "normalization": validator.validate_intensity_normalization,
        }

        results = category_map[args.category]()

        print(f"{args.category.upper()} VALIDATION RESULTS:")
        print("-" * 40)
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"{status}: {result.test_name}")
            print(f"  Error: {result.relative_error:.2e}")

    elif args.report or len(sys.argv) == 1:
        # Generate full report
        report = validator.generate_validation_report()
        print(report)

        if args.json_output:
            # Save results as JSON with proper type conversion
            def convert_value(value):
                """Convert numpy types to JSON-serializable types."""
                if hasattr(value, "item"):  # numpy scalar
                    return value.item()
                if isinstance(value, np.ndarray):
                    return value.tolist()
                if isinstance(value, (np.bool_, bool)):
                    return bool(value)
                if isinstance(value, (np.integer, int)):
                    return int(value)
                if isinstance(value, (np.floating, float)):
                    return float(value)
                if isinstance(value, dict):
                    return {k: convert_value(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [convert_value(v) for v in value]
                return value

            validation_data = {
                "results": [
                    {
                        "test_name": r.test_name,
                        "passed": bool(r.passed),
                        "theoretical_value": float(r.theoretical_value),
                        "computed_value": float(r.computed_value),
                        "relative_error": float(r.relative_error),
                        "absolute_error": float(r.absolute_error),
                        "tolerance": float(r.tolerance),
                        "details": convert_value(r.details),
                    }
                    for r in validator.validation_results
                ],
                "summary": {
                    "total_tests": len(validator.validation_results),
                    "passed_tests": sum(
                        1 for r in validator.validation_results if r.passed
                    ),
                    "tolerance_used": args.tolerance,
                },
            }

            with open(args.json_output, "w") as f:
                json.dump(validation_data, f, indent=2)
            print(f"\nResults saved to: {args.json_output}")

    else:
        # Run comprehensive validation with summary
        validation_suite = validator.run_comprehensive_validation()

        total_tests = sum(len(results) for results in validation_suite.values())
        total_passed = sum(
            sum(1 for r in results if r.passed) for results in validation_suite.values()
        )

        print("Scientific Validation Summary:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {total_passed}")
        print(f"  Success Rate: {total_passed / total_tests * 100:.1f}%")

        for category, results in validation_suite.items():
            passed = sum(1 for r in results if r.passed)
            print(f"  {category}: {passed}/{len(results)}")
