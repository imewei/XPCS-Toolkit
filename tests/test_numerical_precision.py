#!/usr/bin/env python3
"""
Numerical Precision Validation Tests using NumPy Testing Utilities.

This module provides comprehensive numerical precision validation for XPCS
algorithms using numpy.testing utilities to ensure computational accuracy
and numerical stability across different platforms and conditions.
"""

import json
import platform
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as npt

try:
    from scipy import linalg, optimize, stats
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
class PrecisionTestResult:
    """Result of a numerical precision test."""

    test_name: str
    passed: bool
    tolerance_used: float
    max_error: float
    error_type: str  # 'absolute', 'relative', 'ulp'
    platform_info: dict[str, str]
    array_shapes: list[tuple[int, ...]]
    data_types: list[str]
    details: dict[str, Any] = field(default_factory=dict)


class NumericalPrecisionValidator:
    """Comprehensive numerical precision validation framework."""

    def __init__(self, rtol: float = 1e-7, atol: float = 1e-14):
        """Initialize with default tolerances."""
        self.rtol = rtol
        self.atol = atol
        self.platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        }
        self.test_results: list[PrecisionTestResult] = []

    def validate_basic_arithmetic(self) -> list[PrecisionTestResult]:
        """Validate basic arithmetic operations precision."""
        results = []

        # Test 1: Addition associativity
        test_arrays = [
            np.random.randn(100),
            np.random.randn(50, 50),
            np.random.uniform(-1000, 1000, 1000),
        ]

        for i, arr in enumerate(test_arrays):
            # Flatten if multidimensional
            if arr.ndim > 1:
                arr = arr.flatten()

            # Split array into three equal parts
            if len(arr) < 6:  # Skip if array too small
                continue

            split_size = len(arr) // 3
            a = arr[:split_size]
            b = arr[split_size : 2 * split_size]
            c = arr[2 * split_size : 3 * split_size]

            if len(a) == 0 or len(b) == 0 or len(c) == 0:
                continue

            # Test (a + b) + c vs a + (b + c)
            result1 = (a + b) + c
            result2 = a + (b + c)

            try:
                npt.assert_allclose(result1, result2, rtol=self.rtol, atol=self.atol)
                passed = True
                max_error = np.max(np.abs(result1 - result2))
            except AssertionError:
                passed = False
                max_error = np.max(np.abs(result1 - result2))

            result = PrecisionTestResult(
                test_name=f"addition_associativity_array_{i}",
                passed=passed,
                tolerance_used=self.rtol,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[arr.shape],
                data_types=[str(arr.dtype)],
                details={
                    "operation": "addition_associativity",
                    "array_size": len(arr),
                    "array_range": [float(np.min(arr)), float(np.max(arr))],
                },
            )
            results.append(result)

        # Test 2: Multiplication distributivity
        a = np.random.randn(100)
        b = np.random.randn(100)
        c = np.random.randn(100)

        # Test a * (b + c) vs (a * b) + (a * c)
        result1 = a * (b + c)
        result2 = (a * b) + (a * c)

        try:
            npt.assert_allclose(result1, result2, rtol=self.rtol, atol=self.atol)
            passed = True
            max_error = np.max(np.abs(result1 - result2))
        except AssertionError:
            passed = False
            max_error = np.max(np.abs(result1 - result2))

        result = PrecisionTestResult(
            test_name="multiplication_distributivity",
            passed=passed,
            tolerance_used=self.rtol,
            max_error=float(max_error),
            error_type="absolute",
            platform_info=self.platform_info,
            array_shapes=[(100,)],
            data_types=[str(a.dtype)],
            details={
                "operation": "multiplication_distributivity",
                "test_description": "a * (b + c) vs (a * b) + (a * c)",
            },
        )
        results.append(result)

        return results

    def validate_transcendental_functions(self) -> list[PrecisionTestResult]:
        """Validate transcendental function precision."""
        results = []

        # Test 1: exp(log(x)) = x
        x_values = [
            np.random.uniform(0.1, 100, 1000),
            np.random.uniform(1e-6, 1e6, 500),
            np.logspace(-3, 3, 1000),
        ]

        for i, x in enumerate(x_values):
            x_positive = np.abs(x) + 1e-10  # Ensure positive values
            result = np.exp(np.log(x_positive))

            try:
                npt.assert_allclose(result, x_positive, rtol=1e-12, atol=1e-14)
                passed = True
                max_error = np.max(np.abs(result - x_positive))
            except AssertionError:
                passed = False
                max_error = np.max(np.abs(result - x_positive))

            test_result = PrecisionTestResult(
                test_name=f"exp_log_identity_test_{i}",
                passed=passed,
                tolerance_used=1e-12,
                max_error=float(max_error),
                error_type="relative",
                platform_info=self.platform_info,
                array_shapes=[x.shape],
                data_types=[str(x.dtype)],
                details={
                    "operation": "exp(log(x)) = x",
                    "value_range": [
                        float(np.min(x_positive)),
                        float(np.max(x_positive)),
                    ],
                },
            )
            results.append(test_result)

        # Test 2: sin²(x) + cos²(x) = 1
        angles = np.random.uniform(-10 * np.pi, 10 * np.pi, 1000)
        sin_cos_sum = np.sin(angles) ** 2 + np.cos(angles) ** 2
        expected_ones = np.ones_like(angles)

        try:
            npt.assert_allclose(sin_cos_sum, expected_ones, rtol=1e-14, atol=1e-15)
            passed = True
            max_error = np.max(np.abs(sin_cos_sum - expected_ones))
        except AssertionError:
            passed = False
            max_error = np.max(np.abs(sin_cos_sum - expected_ones))

        result = PrecisionTestResult(
            test_name="trigonometric_identity",
            passed=passed,
            tolerance_used=1e-14,
            max_error=float(max_error),
            error_type="absolute",
            platform_info=self.platform_info,
            array_shapes=[angles.shape],
            data_types=[str(angles.dtype)],
            details={
                "operation": "sin²(x) + cos²(x) = 1",
                "angle_range": [float(np.min(angles)), float(np.max(angles))],
            },
        )
        results.append(result)

        return results

    def validate_linear_algebra_precision(self) -> list[PrecisionTestResult]:
        """Validate linear algebra operations precision."""
        results = []

        # Test 1: Matrix multiplication associativity
        sizes = [(10, 10), (25, 25), (50, 50)]

        for size in sizes:
            A = np.random.randn(*size)
            B = np.random.randn(*size)
            C = np.random.randn(*size)

            # Test (A @ B) @ C vs A @ (B @ C)
            result1 = (A @ B) @ C
            result2 = A @ (B @ C)

            try:
                npt.assert_allclose(result1, result2, rtol=1e-12, atol=1e-14)
                passed = True
                max_error = np.max(np.abs(result1 - result2))
            except AssertionError:
                passed = False
                max_error = np.max(np.abs(result1 - result2))

            result = PrecisionTestResult(
                test_name=f"matrix_multiplication_associativity_{size[0]}x{size[1]}",
                passed=passed,
                tolerance_used=1e-12,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[size, size, size],
                data_types=[str(A.dtype)],
                details={
                    "operation": "matrix_multiplication_associativity",
                    "matrix_size": size,
                },
            )
            results.append(result)

        # Test 2: Matrix inverse precision
        for size in [(5, 5), (10, 10), (20, 20)]:
            # Generate well-conditioned matrix
            A = np.random.randn(*size)
            A = A @ A.T + np.eye(size[0]) * 0.1  # Make it positive definite

            try:
                A_inv = np.linalg.inv(A)
                identity_test = A @ A_inv
                expected_identity = np.eye(size[0])

                npt.assert_allclose(
                    identity_test, expected_identity, rtol=1e-10, atol=1e-12
                )
                passed = True
                max_error = np.max(np.abs(identity_test - expected_identity))
            except (AssertionError, np.linalg.LinAlgError):
                passed = False
                max_error = float("inf")

            result = PrecisionTestResult(
                test_name=f"matrix_inverse_precision_{size[0]}x{size[1]}",
                passed=passed,
                tolerance_used=1e-10,
                max_error=float(max_error) if max_error != float("inf") else 1.0,
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[size],
                data_types=[str(A.dtype)],
                details={
                    "operation": "matrix_inverse",
                    "condition_number": float(np.linalg.cond(A))
                    if passed
                    else "singular",
                },
            )
            results.append(result)

        return results

    def validate_statistical_precision(self) -> list[PrecisionTestResult]:
        """Validate statistical computation precision."""
        results = []

        # Test 1: Mean calculation precision (Kahan summation validation)
        test_arrays = [
            np.random.randn(10000),
            np.random.uniform(-1e6, 1e6, 5000),
            np.array([1e20, 1.0, -1e20]),  # Severe cancellation test
        ]

        for i, arr in enumerate(test_arrays):
            # Compare different mean calculation methods
            numpy_mean = np.mean(arr)
            manual_mean = np.sum(arr) / len(arr)

            try:
                npt.assert_allclose([numpy_mean], [manual_mean], rtol=1e-14, atol=1e-15)
                passed = True
                max_error = abs(numpy_mean - manual_mean)
            except AssertionError:
                passed = False
                max_error = abs(numpy_mean - manual_mean)

            result = PrecisionTestResult(
                test_name=f"mean_calculation_precision_{i}",
                passed=passed,
                tolerance_used=1e-14,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[arr.shape],
                data_types=[str(arr.dtype)],
                details={
                    "operation": "mean_calculation",
                    "array_properties": {
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "std": float(np.std(arr)),
                    },
                },
            )
            results.append(result)

        # Test 2: Variance calculation precision
        for i, arr in enumerate(test_arrays[:2]):  # Skip the pathological case
            # Compare numpy variance with manual calculation
            numpy_var = np.var(arr, ddof=0)
            mean_val = np.mean(arr)
            manual_var = np.mean((arr - mean_val) ** 2)

            try:
                npt.assert_allclose([numpy_var], [manual_var], rtol=1e-12, atol=1e-14)
                passed = True
                max_error = abs(numpy_var - manual_var)
            except AssertionError:
                passed = False
                max_error = abs(numpy_var - manual_var)

            result = PrecisionTestResult(
                test_name=f"variance_calculation_precision_{i}",
                passed=passed,
                tolerance_used=1e-12,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[arr.shape],
                data_types=[str(arr.dtype)],
                details={
                    "operation": "variance_calculation",
                    "computed_variance": float(numpy_var),
                },
            )
            results.append(result)

        return results

    def validate_fft_precision(self) -> list[PrecisionTestResult]:
        """Validate FFT precision using round-trip tests."""
        results = []

        # Test 1: FFT round-trip precision
        test_signals = [
            np.random.randn(128),
            np.random.randn(256),
            np.random.randn(512),
            np.sin(2 * np.pi * np.arange(64) / 64),  # Pure sinusoid
        ]

        for i, signal in enumerate(test_signals):
            # Forward and inverse FFT
            fft_signal = np.fft.fft(signal)
            reconstructed = np.fft.ifft(fft_signal)

            try:
                npt.assert_allclose(signal, reconstructed.real, rtol=1e-12, atol=1e-14)
                passed = True
                max_error = np.max(np.abs(signal - reconstructed.real))
            except AssertionError:
                passed = False
                max_error = np.max(np.abs(signal - reconstructed.real))

            result = PrecisionTestResult(
                test_name=f"fft_roundtrip_precision_{i}",
                passed=passed,
                tolerance_used=1e-12,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[signal.shape],
                data_types=[str(signal.dtype)],
                details={
                    "operation": "fft_roundtrip",
                    "signal_length": len(signal),
                    "imaginary_component_max": float(
                        np.max(np.abs(reconstructed.imag))
                    ),
                },
            )
            results.append(result)

        # Test 2: Parseval's theorem validation
        for i, signal in enumerate(test_signals):
            time_energy = np.sum(signal**2)
            freq_energy = np.sum(np.abs(np.fft.fft(signal)) ** 2) / len(signal)

            try:
                npt.assert_allclose(
                    [time_energy], [freq_energy], rtol=1e-12, atol=1e-14
                )
                passed = True
                max_error = abs(time_energy - freq_energy)
            except AssertionError:
                passed = False
                max_error = abs(time_energy - freq_energy)

            result = PrecisionTestResult(
                test_name=f"parseval_theorem_validation_{i}",
                passed=passed,
                tolerance_used=1e-12,
                max_error=float(max_error),
                error_type="absolute",
                platform_info=self.platform_info,
                array_shapes=[signal.shape],
                data_types=[str(signal.dtype)],
                details={
                    "operation": "parseval_theorem",
                    "time_domain_energy": float(time_energy),
                    "frequency_domain_energy": float(freq_energy),
                },
            )
            results.append(result)

        return results

    def validate_correlation_precision(self) -> list[PrecisionTestResult]:
        """Validate correlation calculation precision."""
        results = []

        # Test 1: Autocorrelation symmetry
        signals = [
            np.random.randn(100),
            np.sin(2 * np.pi * np.arange(128) / 16),
            np.random.uniform(-10, 10, 200),
        ]

        for i, signal in enumerate(signals):
            # Calculate autocorrelation using numpy.correlate
            autocorr_full = np.correlate(signal, signal, mode="full")
            mid_point = len(autocorr_full) // 2

            # Test symmetry: R(-τ) = R(τ) for real signals
            left_half = autocorr_full[:mid_point]
            right_half = autocorr_full[mid_point + 1 :][::-1]  # Reverse and align

            min_len = min(len(left_half), len(right_half))
            if min_len > 0:
                try:
                    npt.assert_allclose(
                        left_half[:min_len],
                        right_half[:min_len],
                        rtol=1e-12,
                        atol=1e-14,
                    )
                    passed = True
                    max_error = np.max(
                        np.abs(left_half[:min_len] - right_half[:min_len])
                    )
                except AssertionError:
                    passed = False
                    max_error = np.max(
                        np.abs(left_half[:min_len] - right_half[:min_len])
                    )

                result = PrecisionTestResult(
                    test_name=f"autocorrelation_symmetry_{i}",
                    passed=passed,
                    tolerance_used=1e-12,
                    max_error=float(max_error),
                    error_type="absolute",
                    platform_info=self.platform_info,
                    array_shapes=[signal.shape],
                    data_types=[str(signal.dtype)],
                    details={
                        "operation": "autocorrelation_symmetry",
                        "signal_length": len(signal),
                        "correlation_length": len(autocorr_full),
                    },
                )
                results.append(result)

        return results

    def run_comprehensive_precision_tests(self) -> dict[str, list[PrecisionTestResult]]:
        """Run all precision validation tests."""
        test_suite = {
            "basic_arithmetic": self.validate_basic_arithmetic(),
            "transcendental_functions": self.validate_transcendental_functions(),
            "linear_algebra": self.validate_linear_algebra_precision(),
            "statistical_operations": self.validate_statistical_precision(),
            "fft_operations": self.validate_fft_precision(),
            "correlation_functions": self.validate_correlation_precision(),
        }

        # Store all results
        self.test_results = []
        for category_results in test_suite.values():
            self.test_results.extend(category_results)

        return test_suite

    def generate_precision_report(self) -> str:
        """Generate comprehensive precision validation report."""
        if not self.test_results:
            self.run_comprehensive_precision_tests()

        report = ["Numerical Precision Validation Report", "=" * 60, ""]

        # Platform information
        report.extend(
            [
                "PLATFORM INFORMATION:",
                f"  System: {self.platform_info['system']}",
                f"  Machine: {self.platform_info['machine']}",
                f"  Python: {self.platform_info['python_version']}",
                f"  NumPy: {self.platform_info['numpy_version']}",
                "",
            ]
        )

        # Summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.passed)
        failed_tests = total_tests - passed_tests

        report.extend(
            [
                "PRECISION TEST SUMMARY:",
                f"  Total Tests: {total_tests}",
                f"  Passed: {passed_tests}",
                f"  Failed: {failed_tests}",
                f"  Success Rate: {passed_tests / total_tests * 100:.1f}%",
                f"  Default Tolerance (rtol): {self.rtol:.2e}",
                f"  Default Tolerance (atol): {self.atol:.2e}",
                "",
            ]
        )

        # Group results by category
        categories = {}
        for result in self.test_results:
            category = (
                result.test_name.split("_")[0] if "_" in result.test_name else "general"
            )
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        # Detailed results by category
        for category, results in categories.items():
            report.append(f"{category.upper().replace('_', ' ')} PRECISION TESTS:")
            report.append("-" * 50)

            for result in results:
                status = "✅ PASS" if result.passed else "❌ FAIL"
                report.append(f"  {status}: {result.test_name}")
                report.append(f"    Max Error: {result.max_error:.2e}")
                report.append(f"    Tolerance: {result.tolerance_used:.2e}")
                report.append(f"    Error Type: {result.error_type}")

                if not result.passed:
                    report.append(f"    >>> PRECISION ISSUE: {result.details}")

                report.append("")

            category_passed = sum(1 for r in results if r.passed)
            report.append(
                f"  Category Success: {category_passed}/{len(results)} "
                f"({category_passed / len(results) * 100:.1f}%)"
            )
            report.append("")

        # Overall assessment
        if passed_tests == total_tests:
            report.append("🎉 ALL PRECISION TESTS PASSED!")
            report.append("   Numerical computations are highly accurate.")
        elif passed_tests / total_tests >= 0.95:
            report.append("✅ EXCELLENT: >95% precision tests passed.")
            report.append("   Numerical accuracy is very high.")
        elif passed_tests / total_tests >= 0.85:
            report.append("⚠️  GOOD: 85-95% precision tests passed.")
            report.append("   Minor precision issues detected.")
        else:
            report.append("🚨 ATTENTION: <85% precision tests passed.")
            report.append("   Significant precision issues require investigation.")

        return "\n".join(report)

    def export_precision_results(self, filepath: str) -> None:
        """Export precision test results to JSON."""
        if not self.test_results:
            self.run_comprehensive_precision_tests()

        def convert_for_json(obj):
            """Convert numpy types to JSON-serializable types."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_for_json(v) for v in obj]
            return obj

        export_data = {
            "platform_info": self.platform_info,
            "test_configuration": {"rtol": self.rtol, "atol": self.atol},
            "results": [
                {
                    "test_name": r.test_name,
                    "passed": r.passed,
                    "tolerance_used": r.tolerance_used,
                    "max_error": r.max_error,
                    "error_type": r.error_type,
                    "array_shapes": r.array_shapes,
                    "data_types": r.data_types,
                    "details": convert_for_json(r.details),
                }
                for r in self.test_results
            ],
            "summary": {
                "total_tests": len(self.test_results),
                "passed_tests": sum(1 for r in self.test_results if r.passed),
                "success_rate": sum(1 for r in self.test_results if r.passed)
                / len(self.test_results)
                * 100,
            },
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)


class TestNumericalPrecision(unittest.TestCase):
    """Unit tests for numerical precision validation framework."""

    def setUp(self):
        self.validator = NumericalPrecisionValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator.rtol, float)
        self.assertIsInstance(self.validator.atol, float)
        self.assertIsInstance(self.validator.platform_info, dict)

    def test_basic_arithmetic_validation(self):
        """Test basic arithmetic precision validation."""
        results = self.validator.validate_basic_arithmetic()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIsInstance(result, PrecisionTestResult)
            self.assertIsInstance(result.passed, bool)

    def test_transcendental_function_validation(self):
        """Test transcendental function precision validation."""
        results = self.validator.validate_transcendental_functions()
        self.assertIsInstance(results, list)

        for result in results:
            self.assertIsInstance(result, PrecisionTestResult)

    def test_linear_algebra_validation(self):
        """Test linear algebra precision validation."""
        results = self.validator.validate_linear_algebra_precision()
        self.assertIsInstance(results, list)

        for result in results:
            self.assertIsInstance(result, PrecisionTestResult)

    def test_comprehensive_test_execution(self):
        """Test comprehensive precision test execution."""
        test_suite = self.validator.run_comprehensive_precision_tests()

        self.assertIsInstance(test_suite, dict)
        self.assertGreater(len(test_suite), 0)

        total_tests = sum(len(results) for results in test_suite.values())
        self.assertGreater(total_tests, 0)

    def test_report_generation(self):
        """Test precision report generation."""
        report = self.validator.generate_precision_report()

        self.assertIsInstance(report, str)
        self.assertIn("Numerical Precision Validation Report", report)
        self.assertIn("PLATFORM INFORMATION:", report)

    def test_json_export(self):
        """Test JSON export functionality."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name

        # Export after closing the file to avoid Windows permission issues
        self.validator.export_precision_results(tmp_file_path)

        # Verify file was created and contains valid JSON
        try:
            with open(tmp_file_path) as f:
                data = json.load(f)

            self.assertIn("platform_info", data)
            self.assertIn("results", data)
            self.assertIn("summary", data)
        finally:
            # Clean up with error handling for Windows
            try:
                Path(tmp_file_path).unlink()
            except PermissionError:
                # On Windows, wait a bit and retry
                import time

                time.sleep(0.1)
                try:
                    Path(tmp_file_path).unlink()
                except PermissionError:
                    pass  # If still can't delete, let OS clean up later


if __name__ == "__main__":
    """Command-line interface for numerical precision validation."""
    import argparse

    parser = argparse.ArgumentParser(description="XPCS Numerical Precision Validation")
    parser.add_argument(
        "--rtol", type=float, default=1e-7, help="Relative tolerance for comparisons"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-14, help="Absolute tolerance for comparisons"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[
            "arithmetic",
            "transcendental",
            "linalg",
            "statistics",
            "fft",
            "correlation",
        ],
        help="Run specific validation category",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed precision report"
    )
    parser.add_argument(
        "--export", type=str, metavar="FILE", help="Export results to JSON file"
    )
    parser.add_argument("--unittest", action="store_true", help="Run unit tests")

    args = parser.parse_args()

    if args.unittest:
        unittest.main(argv=[sys.argv[0]], exit=False)
    else:
        validator = NumericalPrecisionValidator(rtol=args.rtol, atol=args.atol)

        if args.category:
            # Run specific category
            category_map = {
                "arithmetic": validator.validate_basic_arithmetic,
                "transcendental": validator.validate_transcendental_functions,
                "linalg": validator.validate_linear_algebra_precision,
                "statistics": validator.validate_statistical_precision,
                "fft": validator.validate_fft_precision,
                "correlation": validator.validate_correlation_precision,
            }

            results = category_map[args.category]()

            print(f"{args.category.upper()} PRECISION VALIDATION:")
            print("-" * 50)
            for result in results:
                status = "PASS" if result.passed else "FAIL"
                print(f"{status}: {result.test_name}")
                print(f"  Max Error: {result.max_error:.2e}")

        elif args.report or len(sys.argv) == 1:
            # Generate full report
            report = validator.generate_precision_report()
            print(report)

            if args.export:
                validator.export_precision_results(args.export)
                print(f"\nResults exported to: {args.export}")

        else:
            # Run comprehensive validation with summary
            test_suite = validator.run_comprehensive_precision_tests()

            total_tests = sum(len(results) for results in test_suite.values())
            total_passed = sum(
                sum(1 for r in results if r.passed) for results in test_suite.values()
            )

            print("Numerical Precision Validation Summary:")
            print(
                f"  Platform: {validator.platform_info['system']} {validator.platform_info['machine']}"
            )
            print(f"  Total Tests: {total_tests}")
            print(f"  Passed: {total_passed}")
            print(f"  Success Rate: {total_passed / total_tests * 100:.1f}%")

            for category, results in test_suite.items():
                passed = sum(1 for r in results if r.passed)
                print(f"  {category}: {passed}/{len(results)}")

            if args.export:
                validator.export_precision_results(args.export)
                print(f"\nResults exported to: {args.export}")
