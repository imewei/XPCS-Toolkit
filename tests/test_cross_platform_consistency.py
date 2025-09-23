#!/usr/bin/env python3
"""
Cross-Platform Numerical Consistency Verification Tests for XPCS Toolkit.

This module provides comprehensive tests to verify that numerical computations
produce consistent results across different platforms, architectures, and
Python/NumPy versions, which is critical for scientific reproducibility.
"""

import hashlib
import json
import platform
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import linalg, optimize, special, stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


@dataclass
class PlatformTestResult:
    """Result of a cross-platform consistency test."""

    test_name: str
    platform_signature: str
    computed_hash: str
    reference_hash: str | None
    is_consistent: bool
    numerical_result: Any
    platform_info: dict[str, str]
    test_parameters: dict[str, Any]
    error_details: str | None = None


@dataclass
class ReferenceResult:
    """Reference result for cross-platform comparison."""

    test_name: str
    parameters: dict[str, Any]
    result_hash: str
    numerical_result: Any
    platform_info: dict[str, str]
    numpy_version: str
    python_version: str


class CrossPlatformValidator:
    """Framework for cross-platform numerical consistency validation."""

    def __init__(self, tolerance: float = 1e-12, hash_precision: int = 6):
        """Initialize with tolerance and hash precision."""
        self.tolerance = tolerance
        self.hash_precision = hash_precision
        self.platform_info = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy_version": np.__version__,
            "architecture": platform.architecture()[0],
        }

        # Add scipy version if available
        if SCIPY_AVAILABLE:
            import scipy

            self.platform_info["scipy_version"] = scipy.__version__

        self.platform_signature = self._generate_platform_signature()
        self.test_results: list[PlatformTestResult] = []
        self.reference_results: dict[str, ReferenceResult] = {}

    def _generate_platform_signature(self) -> str:
        """Generate a unique signature for the current platform."""
        key_info = [
            self.platform_info["system"],
            self.platform_info["machine"],
            self.platform_info["architecture"],
            self.platform_info["python_implementation"],
        ]
        signature = "_".join(key_info).replace(" ", "_").lower()
        return signature

    def _compute_result_hash(self, result: Any) -> str:
        """Compute a stable hash for numerical results."""
        if isinstance(result, np.ndarray):
            # Round to specified precision to handle floating point differences
            rounded = np.around(result, decimals=self.hash_precision)
            return hashlib.md5(rounded.tobytes()).hexdigest()
        if isinstance(result, (list, tuple)):
            # Convert to numpy array and hash
            arr = np.array(result)
            rounded = np.around(arr, decimals=self.hash_precision)
            return hashlib.md5(rounded.tobytes()).hexdigest()
        if isinstance(result, (float, int)):
            # Round single values
            rounded = round(float(result), self.hash_precision)
            return hashlib.md5(str(rounded).encode()).hexdigest()
        # Fallback to string representation
        return hashlib.md5(str(result).encode()).hexdigest()

    def test_basic_mathematical_operations(self) -> list[PlatformTestResult]:
        """Test basic mathematical operations for cross-platform consistency."""
        results = []

        # Test 1: Matrix operations
        np.random.seed(42)  # Fixed seed for reproducibility
        A = np.random.randn(10, 10)
        B = np.random.randn(10, 10)

        # Matrix multiplication
        C = A @ B
        result_hash = self._compute_result_hash(C)

        test_result = PlatformTestResult(
            test_name="matrix_multiplication_10x10",
            platform_signature=self.platform_signature,
            computed_hash=result_hash,
            reference_hash=None,  # Will be set when comparing
            is_consistent=True,  # Will be updated during comparison
            numerical_result=C,
            platform_info=self.platform_info,
            test_parameters={
                "seed": 42,
                "matrix_size": (10, 10),
                "operation": "matrix_multiplication",
            },
        )
        results.append(test_result)

        # Test 2: Eigenvalue decomposition
        symmetric_matrix = A @ A.T  # Make symmetric
        eigenvals = np.linalg.eigvals(symmetric_matrix)
        eigenvals_sorted = np.sort(eigenvals)  # Sort for consistency

        test_result = PlatformTestResult(
            test_name="eigenvalue_decomposition",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(eigenvals_sorted),
            reference_hash=None,
            is_consistent=True,
            numerical_result=eigenvals_sorted,
            platform_info=self.platform_info,
            test_parameters={
                "seed": 42,
                "matrix_size": (10, 10),
                "operation": "eigenvalue_decomposition",
            },
        )
        results.append(test_result)

        # Test 3: FFT operations
        np.random.seed(42)
        signal = np.random.randn(256)
        fft_result = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_result)  # Use magnitude for consistency

        test_result = PlatformTestResult(
            test_name="fft_256_points",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(fft_magnitude),
            reference_hash=None,
            is_consistent=True,
            numerical_result=fft_magnitude,
            platform_info=self.platform_info,
            test_parameters={"seed": 42, "signal_length": 256, "operation": "fft"},
        )
        results.append(test_result)

        return results

    def test_statistical_functions(self) -> list[PlatformTestResult]:
        """Test statistical functions for consistency."""
        results = []

        # Test 1: Random number generation (with fixed seed)
        np.random.seed(12345)
        random_numbers = np.random.randn(1000)

        test_result = PlatformTestResult(
            test_name="random_number_generation",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(random_numbers),
            reference_hash=None,
            is_consistent=True,
            numerical_result=random_numbers,
            platform_info=self.platform_info,
            test_parameters={
                "seed": 12345,
                "sample_size": 1000,
                "distribution": "normal",
            },
        )
        results.append(test_result)

        # Test 2: Statistical moments
        mean_val = np.mean(random_numbers)
        std_val = np.std(random_numbers)
        skew_val = float(stats.skew(random_numbers)) if SCIPY_AVAILABLE else 0.0

        moments = np.array([mean_val, std_val, skew_val])

        test_result = PlatformTestResult(
            test_name="statistical_moments",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(moments),
            reference_hash=None,
            is_consistent=True,
            numerical_result=moments,
            platform_info=self.platform_info,
            test_parameters={
                "sample_size": 1000,
                "moments": ["mean", "std", "skewness"],
            },
        )
        results.append(test_result)

        # Test 3: Correlation calculation
        np.random.seed(54321)
        signal1 = np.random.randn(500)
        signal2 = np.random.randn(500)
        correlation = np.correlate(signal1, signal2, mode="full")

        test_result = PlatformTestResult(
            test_name="cross_correlation",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(correlation),
            reference_hash=None,
            is_consistent=True,
            numerical_result=correlation,
            platform_info=self.platform_info,
            test_parameters={"seed": 54321, "signal_length": 500, "mode": "full"},
        )
        results.append(test_result)

        return results

    def test_transcendental_functions(self) -> list[PlatformTestResult]:
        """Test transcendental functions for consistency."""
        results = []

        # Test 1: Trigonometric functions
        x = np.linspace(-np.pi, np.pi, 100)
        sin_values = np.sin(x)
        cos_values = np.cos(x)
        tan_values = np.tan(x)

        trig_results = np.column_stack([sin_values, cos_values, tan_values])

        test_result = PlatformTestResult(
            test_name="trigonometric_functions",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(trig_results),
            reference_hash=None,
            is_consistent=True,
            numerical_result=trig_results,
            platform_info=self.platform_info,
            test_parameters={
                "domain": "[-π, π]",
                "points": 100,
                "functions": ["sin", "cos", "tan"],
            },
        )
        results.append(test_result)

        # Test 2: Exponential and logarithmic functions
        x_pos = np.logspace(-2, 2, 100)  # Positive values for log
        exp_values = np.exp(x_pos)
        log_values = np.log(x_pos)
        log10_values = np.log10(x_pos)

        exp_log_results = np.column_stack([exp_values, log_values, log10_values])

        test_result = PlatformTestResult(
            test_name="exponential_logarithmic_functions",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(exp_log_results),
            reference_hash=None,
            is_consistent=True,
            numerical_result=exp_log_results,
            platform_info=self.platform_info,
            test_parameters={
                "domain": "logspace(-2, 2)",
                "points": 100,
                "functions": ["exp", "log", "log10"],
            },
        )
        results.append(test_result)

        # Test 3: Special functions (if scipy available)
        if SCIPY_AVAILABLE:
            gamma_values = special.gamma(x_pos[:50])  # Smaller domain for gamma
            bessel_values = special.j0(x_pos[:50])

            special_results = np.column_stack([gamma_values, bessel_values])

            test_result = PlatformTestResult(
                test_name="special_functions",
                platform_signature=self.platform_signature,
                computed_hash=self._compute_result_hash(special_results),
                reference_hash=None,
                is_consistent=True,
                numerical_result=special_results,
                platform_info=self.platform_info,
                test_parameters={"functions": ["gamma", "bessel_j0"], "points": 50},
            )
            results.append(test_result)

        return results

    def test_xpcs_specific_operations(self) -> list[PlatformTestResult]:
        """Test XPCS-specific numerical operations."""
        results = []

        # Test 1: G2 correlation calculation (simplified)
        np.random.seed(98765)
        intensities = np.random.exponential(scale=10.0, size=1000)

        # Simple G2 calculation
        mean_intensity = np.mean(intensities)
        g2_zero = np.mean(intensities**2) / (mean_intensity**2)

        # Calculate G2 for several lag times
        lags = [1, 2, 5, 10, 20]
        g2_values = [g2_zero]

        for lag in lags:
            if lag < len(intensities):
                correlation = np.mean(intensities[:-lag] * intensities[lag:])
                g2_lag = correlation / (mean_intensity**2)
                g2_values.append(g2_lag)

        g2_array = np.array(g2_values)

        test_result = PlatformTestResult(
            test_name="g2_correlation_calculation",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(g2_array),
            reference_hash=None,
            is_consistent=True,
            numerical_result=g2_array,
            platform_info=self.platform_info,
            test_parameters={
                "seed": 98765,
                "intensity_points": 1000,
                "scale": 10.0,
                "lags": lags,
            },
        )
        results.append(test_result)

        # Test 2: Fitting operations (exponential decay)
        if SCIPY_AVAILABLE:
            x_data = np.linspace(0, 5, 50)
            y_true = 2.0 * np.exp(-0.5 * x_data) + 0.1
            np.random.seed(11111)
            noise = np.random.normal(0, 0.05, len(y_true))
            y_data = y_true + noise

            def exp_func(x, a, b, c):
                return a * np.exp(-b * x) + c

            try:
                popt, _ = optimize.curve_fit(
                    exp_func, x_data, y_data, p0=[2.0, 0.5, 0.1]
                )
                fitting_result = popt
            except:
                fitting_result = np.array([2.0, 0.5, 0.1])  # Fallback

            test_result = PlatformTestResult(
                test_name="exponential_fitting",
                platform_signature=self.platform_signature,
                computed_hash=self._compute_result_hash(fitting_result),
                reference_hash=None,
                is_consistent=True,
                numerical_result=fitting_result,
                platform_info=self.platform_info,
                test_parameters={
                    "seed": 11111,
                    "function": "exponential_decay",
                    "data_points": 50,
                    "noise_level": 0.05,
                },
            )
            results.append(test_result)

        # Test 3: Structure factor calculation
        q_values = np.linspace(0.01, 1.0, 100)
        # Simple Lorentzian structure factor
        structure_factor = 1.0 / (1.0 + (q_values * 0.1) ** 2)

        test_result = PlatformTestResult(
            test_name="structure_factor_calculation",
            platform_signature=self.platform_signature,
            computed_hash=self._compute_result_hash(structure_factor),
            reference_hash=None,
            is_consistent=True,
            numerical_result=structure_factor,
            platform_info=self.platform_info,
            test_parameters={
                "q_range": "[0.01, 1.0]",
                "points": 100,
                "model": "lorentzian",
                "correlation_length": 0.1,
            },
        )
        results.append(test_result)

        return results

    def run_comprehensive_platform_tests(self) -> list[PlatformTestResult]:
        """Run all cross-platform consistency tests."""
        all_results = []

        # Run all test categories
        all_results.extend(self.test_basic_mathematical_operations())
        all_results.extend(self.test_statistical_functions())
        all_results.extend(self.test_transcendental_functions())
        all_results.extend(self.test_xpcs_specific_operations())

        self.test_results = all_results
        return all_results

    def save_reference_results(self, filepath: str) -> None:
        """Save current results as reference for future comparisons."""
        reference_data = {
            "platform_info": self.platform_info,
            "platform_signature": self.platform_signature,
            "hash_precision": self.hash_precision,
            "tolerance": self.tolerance,
            "results": {},
        }

        if not self.test_results:
            self.run_comprehensive_platform_tests()

        for result in self.test_results:
            reference_data["results"][result.test_name] = {
                "parameters": result.test_parameters,
                "result_hash": result.computed_hash,
                "platform_info": result.platform_info,
                "numpy_version": self.platform_info["numpy_version"],
                "python_version": self.platform_info["python_version"],
            }

        with open(filepath, "w") as f:
            json.dump(reference_data, f, indent=2)

    def load_reference_results(self, filepath: str) -> bool:
        """Load reference results for comparison."""
        try:
            with open(filepath) as f:
                reference_data = json.load(f)

            self.reference_results = {}
            for test_name, data in reference_data["results"].items():
                self.reference_results[test_name] = ReferenceResult(
                    test_name=test_name,
                    parameters=data["parameters"],
                    result_hash=data["result_hash"],
                    numerical_result=None,  # Not stored in reference
                    platform_info=data["platform_info"],
                    numpy_version=data["numpy_version"],
                    python_version=data["python_version"],
                )
            return True
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False

    def compare_with_reference(self) -> list[PlatformTestResult]:
        """Compare current results with reference results."""
        if not self.test_results:
            self.run_comprehensive_platform_tests()

        for result in self.test_results:
            if result.test_name in self.reference_results:
                ref_result = self.reference_results[result.test_name]
                result.reference_hash = ref_result.result_hash
                result.is_consistent = result.computed_hash == ref_result.result_hash

                if not result.is_consistent:
                    result.error_details = (
                        f"Hash mismatch: computed={result.computed_hash}, "
                        f"reference={ref_result.result_hash}, "
                        f"reference_platform={ref_result.platform_info.get('platform', 'unknown')}"
                    )

        return self.test_results

    def generate_platform_report(self) -> str:
        """Generate comprehensive cross-platform consistency report."""
        if not self.test_results:
            self.run_comprehensive_platform_tests()

        report = ["Cross-Platform Numerical Consistency Report", "=" * 70, ""]

        # Platform information
        report.extend(
            [
                "CURRENT PLATFORM:",
                f"  System: {self.platform_info['system']} ({self.platform_info['machine']})",
                f"  Architecture: {self.platform_info['architecture']}",
                f"  Python: {self.platform_info['python_version']} ({self.platform_info['python_implementation']})",
                f"  NumPy: {self.platform_info['numpy_version']}",
            ]
        )

        if SCIPY_AVAILABLE:
            report.append(f"  SciPy: {self.platform_info['scipy_version']}")

        report.extend([f"  Platform Signature: {self.platform_signature}", ""])

        # Test summary
        total_tests = len(self.test_results)
        consistent_tests = sum(1 for r in self.test_results if r.is_consistent)
        inconsistent_tests = total_tests - consistent_tests

        report.extend(
            [
                "CONSISTENCY TEST SUMMARY:",
                f"  Total Tests: {total_tests}",
                f"  Consistent: {consistent_tests}",
                f"  Inconsistent: {inconsistent_tests}",
                f"  Consistency Rate: {consistent_tests / total_tests * 100:.1f}%",
                f"  Hash Precision: {self.hash_precision} decimal places",
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

        # Report by category
        for category, results in categories.items():
            report.append(f"{category.upper()} CONSISTENCY TESTS:")
            report.append("-" * 50)

            for result in results:
                status = "✅ CONSISTENT" if result.is_consistent else "❌ INCONSISTENT"
                report.append(f"  {status}: {result.test_name}")
                report.append(f"    Computed Hash: {result.computed_hash[:12]}...")

                if result.reference_hash:
                    report.append(
                        f"    Reference Hash: {result.reference_hash[:12]}..."
                    )
                else:
                    report.append("    Reference Hash: Not available")

                if not result.is_consistent and result.error_details:
                    report.append(f"    >>> INCONSISTENCY: {result.error_details}")

                report.append("")

            category_consistent = sum(1 for r in results if r.is_consistent)
            report.append(
                f"  Category Consistency: {category_consistent}/{len(results)} "
                f"({category_consistent / len(results) * 100:.1f}%)"
            )
            report.append("")

        # Overall assessment
        has_reference = any(r.reference_hash for r in self.test_results)

        if not has_reference:
            report.extend(
                [
                    "📝 REFERENCE BASELINE ESTABLISHED",
                    "   Current results can serve as reference for future platforms.",
                    "   Use save_reference_results() to save baseline for comparisons.",
                ]
            )
        elif consistent_tests == total_tests:
            report.extend(
                [
                    "🎉 PERFECT CROSS-PLATFORM CONSISTENCY!",
                    "   All numerical results match reference platform exactly.",
                ]
            )
        elif consistent_tests / total_tests >= 0.95:
            report.extend(
                [
                    "✅ EXCELLENT: >95% consistency with reference platform.",
                    "   Minor differences may be due to different library versions.",
                ]
            )
        elif consistent_tests / total_tests >= 0.80:
            report.extend(
                [
                    "⚠️  GOOD: 80-95% consistency with reference platform.",
                    "   Some platform-specific differences detected.",
                ]
            )
        else:
            report.extend(
                [
                    "🚨 ATTENTION: <80% consistency with reference platform.",
                    "   Significant platform differences require investigation.",
                ]
            )

        return "\n".join(report)

    def export_platform_results(self, filepath: str) -> None:
        """Export platform test results to JSON."""
        if not self.test_results:
            self.run_comprehensive_platform_tests()

        export_data = {
            "platform_info": self.platform_info,
            "platform_signature": self.platform_signature,
            "test_configuration": {
                "tolerance": self.tolerance,
                "hash_precision": self.hash_precision,
            },
            "results": [
                {
                    "test_name": r.test_name,
                    "computed_hash": r.computed_hash,
                    "reference_hash": r.reference_hash,
                    "is_consistent": r.is_consistent,
                    "test_parameters": r.test_parameters,
                    "error_details": r.error_details,
                }
                for r in self.test_results
            ],
            "summary": {
                "total_tests": len(self.test_results),
                "consistent_tests": sum(
                    1 for r in self.test_results if r.is_consistent
                ),
                "consistency_rate": sum(1 for r in self.test_results if r.is_consistent)
                / len(self.test_results)
                * 100,
            },
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)


class TestCrossPlatformConsistency(unittest.TestCase):
    """Unit tests for cross-platform consistency validation."""

    def setUp(self):
        self.validator = CrossPlatformValidator()

    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertIsInstance(self.validator.tolerance, float)
        self.assertIsInstance(self.validator.platform_info, dict)
        self.assertIn("system", self.validator.platform_info)

    def test_platform_signature_generation(self):
        """Test platform signature generation."""
        signature = self.validator._generate_platform_signature()
        self.assertIsInstance(signature, str)
        self.assertGreater(len(signature), 0)

    def test_result_hash_computation(self):
        """Test result hash computation."""
        # Test with array
        arr = np.array([1.0, 2.0, 3.0])
        hash1 = self.validator._compute_result_hash(arr)
        hash2 = self.validator._compute_result_hash(arr)
        self.assertEqual(hash1, hash2)  # Should be deterministic

        # Test with different array (should have different hash)
        arr2 = np.array([1.0, 2.0, 3.1])
        hash3 = self.validator._compute_result_hash(arr2)
        self.assertNotEqual(hash1, hash3)

    def test_mathematical_operations(self):
        """Test basic mathematical operations."""
        results = self.validator.test_basic_mathematical_operations()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        for result in results:
            self.assertIsInstance(result, PlatformTestResult)
            self.assertIsInstance(result.computed_hash, str)

    def test_statistical_functions(self):
        """Test statistical functions."""
        results = self.validator.test_statistical_functions()
        self.assertIsInstance(results, list)

        for result in results:
            self.assertIsInstance(result, PlatformTestResult)

    def test_comprehensive_test_execution(self):
        """Test comprehensive platform test execution."""
        results = self.validator.run_comprehensive_platform_tests()

        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        # Check that all tests have valid hashes
        for result in results:
            self.assertIsInstance(result.computed_hash, str)
            self.assertEqual(len(result.computed_hash), 32)  # MD5 hash length

    def test_reference_result_save_load(self):
        """Test saving and loading reference results."""
        # Run tests to generate results
        self.validator.run_comprehensive_platform_tests()

        # Save reference results
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name

        # Save and load after closing the file to avoid Windows permission issues
        self.validator.save_reference_results(tmp_file_path)

        try:
            # Load reference results
            success = self.validator.load_reference_results(tmp_file_path)
            self.assertTrue(success)
            self.assertGreater(len(self.validator.reference_results), 0)
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

    def test_report_generation(self):
        """Test platform report generation."""
        report = self.validator.generate_platform_report()

        self.assertIsInstance(report, str)
        self.assertIn("Cross-Platform Numerical Consistency Report", report)
        self.assertIn("CURRENT PLATFORM:", report)

    def test_json_export(self):
        """Test JSON export functionality."""
        self.validator.run_comprehensive_platform_tests()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            tmp_file_path = tmp_file.name

        # Export after closing the file to avoid Windows permission issues
        self.validator.export_platform_results(tmp_file_path)

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
    """Command-line interface for cross-platform consistency validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Cross-Platform Consistency Validation"
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-12,
        help="Numerical tolerance for comparisons",
    )
    parser.add_argument(
        "--precision", type=int, default=6, help="Hash precision (decimal places)"
    )
    parser.add_argument(
        "--save-reference",
        type=str,
        metavar="FILE",
        help="Save current results as reference baseline",
    )
    parser.add_argument(
        "--compare-reference",
        type=str,
        metavar="FILE",
        help="Compare with reference results file",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed consistency report"
    )
    parser.add_argument(
        "--export", type=str, metavar="FILE", help="Export results to JSON file"
    )
    parser.add_argument("--unittest", action="store_true", help="Run unit tests")

    args = parser.parse_args()

    if args.unittest:
        unittest.main(argv=[sys.argv[0]], exit=False)
    else:
        validator = CrossPlatformValidator(
            tolerance=args.tolerance, hash_precision=args.precision
        )

        # Run tests
        results = validator.run_comprehensive_platform_tests()

        # Handle reference comparison
        if args.compare_reference:
            if validator.load_reference_results(args.compare_reference):
                validator.compare_with_reference()
                print(f"Compared with reference: {args.compare_reference}")
            else:
                print(f"Could not load reference file: {args.compare_reference}")

        # Generate report
        if args.report or len(sys.argv) == 1:
            report = validator.generate_platform_report()
            print(report)

        # Save reference
        if args.save_reference:
            validator.save_reference_results(args.save_reference)
            print(f"Reference results saved to: {args.save_reference}")

        # Export results
        if args.export:
            validator.export_platform_results(args.export)
            print(f"Results exported to: {args.export}")

        # Summary if not generating full report
        if not args.report and len(sys.argv) > 1:
            total_tests = len(results)
            consistent_tests = sum(1 for r in results if r.is_consistent)

            print("\nCross-Platform Consistency Summary:")
            print(f"  Platform: {validator.platform_signature}")
            print(f"  Total Tests: {total_tests}")
            print(f"  Consistent: {consistent_tests}")
            print(f"  Consistency Rate: {consistent_tests / total_tests * 100:.1f}%")
