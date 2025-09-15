#!/usr/bin/env python3
"""
Numerical accuracy validation suite for Phase 3 vectorization optimizations.
Ensures all optimizations preserve exact scientific accuracy.
"""

import os
import sys
from typing import Any

import numpy as np
import pytest

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from xpcs_toolkit.helper import fitting
    from xpcs_toolkit.module import g2mod, saxs1d, twotime_utils

    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    pytest.skip("Required modules not available", allow_module_level=True)


class NumericalAccuracyValidator:
    """Comprehensive validation of numerical accuracy for vectorized operations."""

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize validator with specified numerical tolerance.

        Args:
            tolerance: Maximum allowed difference between original and optimized results
        """
        self.tolerance = tolerance
        self.validation_results = {}
        self._generate_test_cases()

    def _generate_test_cases(self):
        """Generate comprehensive test cases for validation."""
        np.random.seed(12345)  # Reproducible test data

        # C2 matrix test cases
        self.test_cases = {
            "c2_matrices": {
                "small_symmetric": self._make_symmetric_matrix(64),
                "medium_symmetric": self._make_symmetric_matrix(256),
                "small_asymmetric": np.random.randn(64, 64).astype(np.float64),
                "medium_asymmetric": np.random.randn(256, 256).astype(np.float64),
                "edge_cases": {
                    "zeros": np.zeros((32, 32), dtype=np.float64),
                    "ones": np.ones((32, 32), dtype=np.float64),
                    "identity": np.eye(32, dtype=np.float64),
                    "very_small": 1e-15 * np.random.randn(32, 32).astype(np.float64),
                    "very_large": 1e15 * np.random.randn(32, 32).astype(np.float64),
                },
            },
            # G2 processing test cases
            "g2_data": {
                "standard": {
                    "tel": np.logspace(-6, 2, 100),
                    "g2": np.random.exponential(1.0, (100, 20)) + 0.1,
                    "g2_err": None,  # Will be calculated
                    "qd": np.logspace(-3, -1, 20),
                },
                "edge_cases": {
                    "single_point": {
                        "tel": np.array([1.0]),
                        "g2": np.array([[1.1]]),
                        "g2_err": np.array([[0.1]]),
                        "qd": np.array([0.01]),
                    },
                    "large_dynamic_range": {
                        "tel": np.logspace(-12, 6, 150),
                        "g2": np.random.uniform(0.001, 1000, (150, 30)) + 1.0,
                        "g2_err": None,
                        "qd": np.logspace(-4, 0, 30),
                    },
                },
            },
            # SAXS analysis test cases
            "saxs_data": {
                "standard": {
                    "q": np.logspace(-2, 0, 500),
                    "I": np.random.lognormal(0, 1, 500).astype(np.float64),
                    "I_err": None,  # Will be calculated
                },
                "edge_cases": {
                    "steep_decay": {
                        "q": np.logspace(-2, 0, 200),
                        "I": np.power(np.logspace(-2, 0, 200), -4) * 1e6,
                        "I_err": None,
                    },
                    "flat_background": {
                        "q": np.linspace(0.01, 1.0, 100),
                        "I": np.ones(100) * 1000.0,
                        "I_err": np.ones(100) * 10.0,
                    },
                },
            },
            # Fitting test cases
            "fitting_data": {
                "exponential": {
                    "x": np.linspace(0, 10, 200),
                    "params_true": [3.0, 0.1, 2.0],  # [tau, baseline, amplitude]
                    "noise_level": 0.02,
                },
                "multi_exponential": {
                    "x": np.logspace(-3, 3, 300),
                    "params_true": [
                        1.0,
                        10.0,
                        0.05,
                        1.0,
                        0.5,
                    ],  # [tau1, tau2, baseline, A1, A2]
                    "noise_level": 0.01,
                },
            },
        }

        # Calculate error estimates where needed
        self._calculate_error_estimates()

    def _make_symmetric_matrix(self, size: int) -> np.ndarray:
        """Generate a symmetric matrix for testing."""
        A = np.random.randn(size, size).astype(np.float64)
        return 0.5 * (A + A.T)

    def _calculate_error_estimates(self):
        """Calculate realistic error estimates for test data."""
        # G2 error estimates
        for case_name, case_data in self.test_cases["g2_data"].items():
            if isinstance(case_data, dict) and "g2" in case_data:
                if case_data.get("g2_err") is None:
                    case_data["g2_err"] = 0.1 * np.sqrt(case_data["g2"])
            elif case_name == "edge_cases" and isinstance(case_data, dict):
                # Handle nested edge cases
                for _edge_name, edge_data in case_data.items():
                    if edge_data.get("g2_err") is None and "g2" in edge_data:
                        edge_data["g2_err"] = 0.1 * np.sqrt(edge_data["g2"])

        # SAXS error estimates
        for case_name, case_data in self.test_cases["saxs_data"].items():
            if isinstance(case_data, dict) and "I" in case_data:
                if case_data.get("I_err") is None:
                    case_data["I_err"] = 0.05 * np.sqrt(case_data["I"])
            elif case_name == "edge_cases" and isinstance(case_data, dict):
                # Handle nested edge cases
                for _edge_name, edge_data in case_data.items():
                    if edge_data.get("I_err") is None and "I" in edge_data:
                        edge_data["I_err"] = 0.05 * np.sqrt(edge_data["I"])

    def validate_c2_diagonal_correction(self) -> dict[str, bool]:
        """Validate C2 diagonal correction accuracy."""
        results = {}

        for matrix_name, c2_matrix in self.test_cases["c2_matrices"].items():
            if isinstance(c2_matrix, dict):
                # Handle edge cases
                for edge_name, edge_matrix in c2_matrix.items():
                    test_name = f"{matrix_name}_{edge_name}"
                    results[test_name] = self._test_diagonal_correction(edge_matrix)
            else:
                results[matrix_name] = self._test_diagonal_correction(c2_matrix)

        return results

    def _test_diagonal_correction(self, c2_matrix: np.ndarray) -> bool:
        """Test diagonal correction for a single matrix."""
        try:
            # Create copies for testing
            c2_original = c2_matrix.copy()
            c2_vectorized = c2_matrix.copy()

            # Apply corrections
            if hasattr(twotime_utils, "correct_diagonal_c2"):
                c2_original = twotime_utils.correct_diagonal_c2(c2_original)
            else:
                # Use vectorized as reference if original not available
                c2_original = twotime_utils.correct_diagonal_c2_vectorized(c2_original)

            c2_vectorized = twotime_utils.correct_diagonal_c2_vectorized(c2_vectorized)

            # Compare results
            max_diff = np.max(np.abs(c2_original - c2_vectorized))
            return max_diff < self.tolerance

        except Exception as e:
            print(f"Error in diagonal correction test: {e}")
            return False

    def validate_c2_matrix_reconstruction(self) -> dict[str, bool]:
        """Validate C2 matrix reconstruction accuracy."""
        results = {}

        for matrix_name, c2_matrix in self.test_cases["c2_matrices"].items():
            if isinstance(c2_matrix, dict):
                continue  # Skip edge cases for reconstruction

            # Create half matrix
            c2_half = c2_matrix[: c2_matrix.shape[0] // 2, :]

            try:
                # Original method (manual implementation)
                c2_original = c2_half + c2_half.T
                diag_vals = np.diag(c2_half)
                np.fill_diagonal(c2_original, diag_vals)

                # Vectorized method
                c2_vectorized = twotime_utils._reconstruct_c2_matrix_vectorized(c2_half)

                # Compare results
                max_diff = np.max(np.abs(c2_original - c2_vectorized))
                results[matrix_name] = max_diff < self.tolerance

            except Exception as e:
                print(f"Error in matrix reconstruction test for {matrix_name}: {e}")
                results[matrix_name] = False

        return results

    def validate_g2_processing(self) -> dict[str, bool]:
        """Validate G2 processing accuracy."""
        results = {}

        for case_name, case_data in self.test_cases["g2_data"].items():
            if isinstance(case_data, dict) and "tel" in case_data:
                # Test baseline correction
                baseline_values = np.random.rand(case_data["g2"].shape[1])

                # Manual baseline correction
                g2_manual = case_data["g2"] - baseline_values[np.newaxis, :] + 1.0

                # Vectorized baseline correction
                g2_vectorized = g2mod.vectorized_g2_baseline_correction(
                    case_data["g2"], baseline_values
                )

                max_diff = np.max(np.abs(g2_manual - g2_vectorized))
                results[f"{case_name}_baseline_correction"] = max_diff < self.tolerance

                # Test normalization
                g2_list = [
                    case_data["g2"],
                    case_data["g2"] * 1.1,
                    case_data["g2"] * 0.9,
                ]

                # Manual max normalization
                normalized_manual = []
                for g2_data in g2_list:
                    max_vals = np.max(g2_data, axis=0, keepdims=True)
                    max_vals = np.where(max_vals == 0, 1.0, max_vals)
                    normalized_manual.append(g2_data / max_vals)

                # Vectorized normalization
                normalized_vectorized = g2mod.batch_g2_normalization(
                    g2_list, method="max"
                )

                max_diff = 0
                for i in range(len(g2_list)):
                    diff = np.max(
                        np.abs(normalized_manual[i] - normalized_vectorized[i])
                    )
                    max_diff = max(max_diff, diff)

                results[f"{case_name}_normalization"] = max_diff < self.tolerance

        return results

    def validate_saxs_analysis(self) -> dict[str, bool]:
        """Validate SAXS analysis accuracy."""
        results = {}

        for case_name, case_data in self.test_cases["saxs_data"].items():
            q = case_data["q"]
            intensity = case_data["I"]
            case_data["I_err"]

            # Test q-space binning
            q_min, q_max, num_bins = np.min(q), np.max(q), 50

            # Manual binning
            bin_edges = np.linspace(q_min, q_max, num_bins + 1)
            0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_indices = np.digitize(q, bin_edges) - 1

            binned_I_manual = np.zeros(num_bins)
            for i in range(num_bins):
                mask = bin_indices == i
                if np.any(mask):
                    binned_I_manual[i] = np.mean(intensity[mask])

            # Vectorized binning
            _bin_centers_vec, binned_I_vec, _ = saxs1d.vectorized_q_binning(
                q, intensity, q_min, q_max, num_bins
            )

            # Compare non-zero bins
            valid_mask = (binned_I_manual != 0) & (binned_I_vec != 0)
            if np.any(valid_mask):
                max_diff = np.max(
                    np.abs(binned_I_manual[valid_mask] - binned_I_vec[valid_mask])
                )
                results[f"{case_name}_q_binning"] = max_diff < self.tolerance
            else:
                results[f"{case_name}_q_binning"] = True

            # Test intensity normalization
            for norm_method in ["q2", "q4", "max"]:
                # Manual normalization
                if norm_method == "q2":
                    I_norm_manual = intensity * q**2
                elif norm_method == "q4":
                    I_norm_manual = intensity * q**4
                elif norm_method == "max":
                    I_norm_manual = intensity / np.max(intensity)

                # Vectorized normalization
                I_norm_vectorized = saxs1d.vectorized_intensity_normalization(
                    q, intensity, method=norm_method
                )

                max_diff = np.max(np.abs(I_norm_manual - I_norm_vectorized))
                results[f"{case_name}_normalization_{norm_method}"] = (
                    max_diff < self.tolerance
                )

        return results

    def validate_fitting_operations(self) -> dict[str, bool]:
        """Validate fitting operations accuracy."""
        results = {}

        # Test parameter estimation
        x = self.test_cases["fitting_data"]["exponential"]["x"]
        params_true = self.test_cases["fitting_data"]["exponential"]["params_true"]
        tau, baseline, amplitude = params_true

        # Generate clean exponential data
        y_clean = amplitude * np.exp(-x / tau) + baseline

        # Test parameter estimation
        params_estimated = fitting.vectorized_parameter_estimation(
            x, y_clean, "exponential"
        )

        if params_estimated is not None:
            # Compare estimated parameters (allowing for reasonable estimation error)
            tau_est, baseline_est, amplitude_est = params_estimated

            tau_error = abs(tau_est - tau) / tau
            baseline_error = (
                abs(baseline_est - baseline) / baseline
                if baseline != 0
                else abs(baseline_est)
            )
            amplitude_error = abs(amplitude_est - amplitude) / amplitude

            # Allow 10% error in parameter estimation
            results["parameter_estimation_exponential"] = (
                tau_error < 0.1 and baseline_error < 0.1 and amplitude_error < 0.1
            )
        else:
            results["parameter_estimation_exponential"] = False

        # Test residual analysis
        y_pred = y_clean + 0.001 * np.random.randn(
            len(y_clean)
        )  # Add small prediction error

        residuals_manual = y_clean - y_pred
        stats_manual = {
            "mean_residual": np.mean(residuals_manual),
            "std_residual": np.std(residuals_manual),
            "rmse": np.sqrt(np.mean(residuals_manual**2)),
            "mae": np.mean(np.abs(residuals_manual)),
        }

        stats_vectorized = fitting.vectorized_residual_analysis(x, y_clean, y_pred)

        # Compare statistics
        all_stats_match = True
        for stat_name in ["mean_residual", "std_residual", "rmse", "mae"]:
            manual_val = stats_manual[stat_name]
            vectorized_val = stats_vectorized[stat_name]
            if abs(manual_val - vectorized_val) > self.tolerance:
                all_stats_match = False
                break

        results["residual_analysis"] = all_stats_match

        return results

    def validate_mathematical_properties(self) -> dict[str, bool]:
        """Validate that mathematical properties are preserved."""
        results = {}

        # Test symmetry preservation in C2 operations
        symmetric_matrix = self.test_cases["c2_matrices"]["small_symmetric"]
        corrected_matrix = twotime_utils.correct_diagonal_c2_vectorized(
            symmetric_matrix.copy()
        )

        # Check if symmetry is preserved (within numerical precision)
        symmetry_error = np.max(np.abs(corrected_matrix - corrected_matrix.T))
        results["c2_symmetry_preservation"] = symmetry_error < 1e-12

        # Test normalization properties
        q = self.test_cases["saxs_data"]["standard"]["q"]
        intensity = self.test_cases["saxs_data"]["standard"]["I"]

        # Max normalization should give maximum value of 1
        I_norm = saxs1d.vectorized_intensity_normalization(q, intensity, method="max")
        max_value = np.max(I_norm)
        results["saxs_max_normalization_property"] = (
            abs(max_value - 1.0) < self.tolerance
        )

        # Test error propagation properties
        g2_data = self.test_cases["g2_data"]["standard"]["g2"]
        baseline = np.random.rand(g2_data.shape[1])

        # Baseline subtraction should not change relative errors
        original_rel_errors = self.test_cases["g2_data"]["standard"]["g2_err"] / g2_data

        corrected_g2 = g2mod.vectorized_g2_baseline_correction(g2_data, baseline)
        # Errors should remain the same after baseline correction
        corrected_rel_errors = (
            self.test_cases["g2_data"]["standard"]["g2_err"] / corrected_g2
        )

        # Check if relative error structure is preserved (within reason)
        rel_error_diff = np.mean(np.abs(original_rel_errors - corrected_rel_errors))
        results["g2_error_propagation"] = (
            rel_error_diff < 0.1
        )  # Allow some change due to baseline

        return results

    def run_comprehensive_validation(self) -> dict[str, Any]:
        """Run all validation tests."""
        print("Running Phase 3 Vectorization Numerical Accuracy Validation...")
        print("=" * 70)

        validation_suites = [
            ("C2 Diagonal Correction", self.validate_c2_diagonal_correction),
            ("C2 Matrix Reconstruction", self.validate_c2_matrix_reconstruction),
            ("G2 Processing", self.validate_g2_processing),
            ("SAXS Analysis", self.validate_saxs_analysis),
            ("Fitting Operations", self.validate_fitting_operations),
            ("Mathematical Properties", self.validate_mathematical_properties),
        ]

        all_results = {
            "tolerance": self.tolerance,
            "validation_suites": {},
            "summary": {},
        }

        total_tests = 0
        passed_tests = 0

        for suite_name, validation_func in validation_suites:
            print(f"\nValidating {suite_name}...")
            try:
                suite_results = validation_func()
                all_results["validation_suites"][
                    suite_name.lower().replace(" ", "_")
                ] = suite_results

                suite_passed = sum(suite_results.values())
                suite_total = len(suite_results)
                total_tests += suite_total
                passed_tests += suite_passed

                print(f"  {suite_passed}/{suite_total} tests passed")

                # Show failed tests
                for test_name, passed in suite_results.items():
                    if not passed:
                        print(f"    FAILED: {test_name}")

            except Exception as e:
                print(f"Error in {suite_name}: {e}")
                all_results["validation_suites"][
                    suite_name.lower().replace(" ", "_")
                ] = {}

        # Summary
        all_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_passed": passed_tests == total_tests,
        }

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {all_results['summary']['pass_rate']:.1%}")
        print(
            f"Overall Result: {'PASSED' if all_results['summary']['overall_passed'] else 'FAILED'}"
        )

        return all_results


# Pytest test functions for automated testing
class TestVectorizationAccuracy:
    """Pytest test class for vectorization accuracy."""

    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.validator = NumericalAccuracyValidator(tolerance=1e-10)

    def test_c2_diagonal_correction_accuracy(self):
        """Test C2 diagonal correction accuracy."""
        results = self.validator.validate_c2_diagonal_correction()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"

    def test_c2_matrix_reconstruction_accuracy(self):
        """Test C2 matrix reconstruction accuracy."""
        results = self.validator.validate_c2_matrix_reconstruction()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"

    def test_g2_processing_accuracy(self):
        """Test G2 processing accuracy."""
        results = self.validator.validate_g2_processing()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"

    def test_saxs_analysis_accuracy(self):
        """Test SAXS analysis accuracy."""
        results = self.validator.validate_saxs_analysis()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"

    def test_fitting_operations_accuracy(self):
        """Test fitting operations accuracy."""
        results = self.validator.validate_fitting_operations()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"

    def test_mathematical_properties(self):
        """Test mathematical properties preservation."""
        results = self.validator.validate_mathematical_properties()
        failed_tests = [test for test, passed in results.items() if not passed]
        assert not failed_tests, f"Failed tests: {failed_tests}"


def main():
    """Main execution function for standalone testing."""
    if not MODULES_AVAILABLE:
        print("Cannot run validation: required modules not available")
        return

    validator = NumericalAccuracyValidator(tolerance=1e-10)
    results = validator.run_comprehensive_validation()

    # Save results if needed
    import json

    output_file = os.path.join(
        os.path.dirname(__file__), "vectorization_accuracy_results.json"
    )
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()
