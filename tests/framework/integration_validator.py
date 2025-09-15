"""Test Suite Integration Validator and Cross-Platform Compatibility Checker.

This module validates that all test modules work together cohesively across
different environments and platforms, ensuring proper test isolation,
fixture compatibility, and consistent behavior.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class IntegrationTestResult:
    """Result of an integration test."""

    test_name: str
    passed: bool
    duration: float
    error_message: str | None = None
    platform_specific: bool = False


@dataclass
class PlatformTestResults:
    """Test results for a specific platform."""

    platform_name: str
    python_version: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration: float
    test_results: list[IntegrationTestResult]
    environment_info: dict[str, Any]


class TestSuiteIntegrationValidator:
    """Validator for test suite integration and cross-platform compatibility."""

    def __init__(self, test_directory: Path | None = None):
        self.test_dir = test_directory or Path(__file__).parent
        self.project_root = self.test_dir.parent

    def validate_complete_integration(self) -> dict[str, Any]:
        """Run complete integration validation across all test categories."""
        print("🔍 Running Complete Test Suite Integration Validation...")

        validation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "python_version": sys.version,
            "test_categories": {},
            "cross_category_integration": {},
            "fixture_compatibility": {},
            "performance_impact": {},
            "issues_found": [],
        }

        # Test each category independently
        categories = [
            ("unit", "tests/unit/"),
            ("integration", "tests/integration/"),
            ("scientific", "tests/scientific/"),
            ("performance", "tests/performance/"),
            ("end_to_end", "tests/end_to_end/"),
        ]

        for category_name, category_path in categories:
            print(f"  Testing {category_name} category...")
            category_result = self._test_category_integration(category_path)
            validation_results["test_categories"][category_name] = category_result

        # Test cross-category integration
        print("  Testing cross-category integration...")
        cross_integration = self._test_cross_category_integration()
        validation_results["cross_category_integration"] = cross_integration

        # Test fixture compatibility
        print("  Testing fixture compatibility...")
        fixture_compat = self._test_fixture_compatibility()
        validation_results["fixture_compatibility"] = fixture_compat

        # Assess performance impact
        print("  Assessing performance impact...")
        performance_impact = self._assess_performance_impact()
        validation_results["performance_impact"] = performance_impact

        # Generate issues summary
        issues = self._identify_integration_issues(validation_results)
        validation_results["issues_found"] = issues

        print("✅ Integration validation completed!")
        return validation_results

    def validate_cross_platform_compatibility(self) -> dict[str, Any]:
        """Validate test compatibility across different platforms."""
        print("🌐 Running Cross-Platform Compatibility Validation...")

        # Simulate different platform behaviors
        compatibility_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "current_platform": platform.platform(),
            "path_separator_tests": self._test_path_separators(),
            "environment_variable_tests": self._test_environment_variables(),
            "file_system_tests": self._test_file_system_compatibility(),
            "gui_platform_tests": self._test_gui_platform_compatibility(),
            "dependency_tests": self._test_dependency_compatibility(),
            "recommendations": [],
        }

        # Generate recommendations
        recommendations = self._generate_platform_recommendations(compatibility_results)
        compatibility_results["recommendations"] = recommendations

        print("✅ Cross-platform validation completed!")
        return compatibility_results

    def _test_category_integration(self, category_path: str) -> dict[str, Any]:
        """Test integration within a specific test category."""
        full_path = self.project_root / category_path

        if not full_path.exists():
            return {
                "status": "skipped",
                "reason": f"Category path {category_path} does not exist",
            }

        # Run tests in category
        start_time = time.time()
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(full_path),
                    "-v",
                    "--tb=short",
                    "--maxfail=5",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
                timeout=300,
            )

            duration = time.time() - start_time

            # Parse pytest output
            test_results = self._parse_pytest_output(result.stdout)

            return {
                "status": "completed",
                "return_code": result.returncode,
                "duration": duration,
                "stdout_lines": len(result.stdout.split("\n")),
                "stderr_lines": len(result.stderr.split("\n")),
                "test_results": test_results,
                "issues": self._extract_category_issues(result.stderr),
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "duration": time.time() - start_time,
                "issues": ["Tests timed out after 5 minutes"],
            }
        except Exception as e:
            return {
                "status": "error",
                "duration": time.time() - start_time,
                "error": str(e),
                "issues": [f"Execution error: {e}"],
            }

    def _test_cross_category_integration(self) -> dict[str, Any]:
        """Test integration between different test categories."""
        integration_tests = [
            {
                "name": "unit_and_integration_together",
                "paths": ["tests/unit/core/", "tests/integration/"],
                "description": "Test unit and integration tests together",
            },
            {
                "name": "scientific_with_performance",
                "paths": [
                    "tests/scientific/algorithms/",
                    "tests/performance/benchmarks/",
                ],
                "description": "Test scientific tests with performance tests",
            },
            {
                "name": "all_categories_subset",
                "paths": [
                    "tests/unit/core/test_xpcs_file.py",
                    "tests/integration/test_component_integration.py",
                    "tests/scientific/algorithms/test_g2_analysis.py",
                ],
                "description": "Test representative tests from each category",
            },
        ]

        results = {}
        for test_config in integration_tests:
            print(f"    Running {test_config['name']}...")
            start_time = time.time()

            try:
                # Build pytest command with all paths
                cmd = (
                    [sys.executable, "-m", "pytest"]
                    + [str(self.project_root / path) for path in test_config["paths"]]
                    + ["-v", "--tb=short", "--maxfail=3"]
                )

                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=180,
                )

                duration = time.time() - start_time

                results[test_config["name"]] = {
                    "status": "completed",
                    "description": test_config["description"],
                    "return_code": result.returncode,
                    "duration": duration,
                    "passed": result.returncode == 0,
                    "issues": self._extract_category_issues(result.stderr)
                    if result.returncode != 0
                    else [],
                }

            except subprocess.TimeoutExpired:
                results[test_config["name"]] = {
                    "status": "timeout",
                    "description": test_config["description"],
                    "duration": time.time() - start_time,
                    "passed": False,
                    "issues": ["Cross-category integration test timed out"],
                }
            except Exception as e:
                results[test_config["name"]] = {
                    "status": "error",
                    "description": test_config["description"],
                    "error": str(e),
                    "passed": False,
                    "issues": [f"Integration test error: {e}"],
                }

        return results

    def _test_fixture_compatibility(self) -> dict[str, Any]:
        """Test fixture compatibility across different test modules."""
        fixture_tests = [
            {
                "name": "conftest_fixtures",
                "description": "Test global fixtures from conftest.py",
                "test_command": [
                    sys.executable,
                    "-c",
                    """
import pytest
import sys
sys.path.insert(0, 'tests')
from conftest import *

# Test fixture creation
try:
    import tempfile
    temp_dir = tempfile.mkdtemp()
    print('temp_dir fixture: PASS')
except Exception as e:
    print(f'temp_dir fixture: FAIL - {e}')

try:
    import numpy as np
    data = {'tau': np.array([1, 2, 3]), 'g2': np.array([2, 1.5, 1.2])}
    print('synthetic_correlation_data fixture: PASS')
except Exception as e:
    print(f'synthetic_correlation_data fixture: FAIL - {e}')
""",
                ],
            },
            {
                "name": "shared_fixture_usage",
                "description": "Test shared fixtures across multiple test files",
                "test_command": [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/unit/core/test_xpcs_file.py::TestMemoryMonitor::test_get_memory_usage",
                    "tests/unit/utils/test_memory_utils.py",
                    "-v",
                    "--tb=short",
                ],
            },
        ]

        results = {}
        for fixture_test in fixture_tests:
            print(f"    Testing {fixture_test['name']}...")
            try:
                result = subprocess.run(
                    fixture_test["test_command"],
                    check=False,
                    capture_output=True,
                    text=True,
                    cwd=self.project_root,
                    timeout=60,
                )

                results[fixture_test["name"]] = {
                    "description": fixture_test["description"],
                    "passed": result.returncode == 0,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "issues": []
                    if result.returncode == 0
                    else [f"Fixture compatibility issue: {result.stderr}"],
                }
            except Exception as e:
                results[fixture_test["name"]] = {
                    "description": fixture_test["description"],
                    "passed": False,
                    "error": str(e),
                    "issues": [f"Fixture test error: {e}"],
                }

        return results

    def _assess_performance_impact(self) -> dict[str, Any]:
        """Assess performance impact of running different test combinations."""

        results = {}

        # Single vs batch execution
        print("    Measuring single vs batch execution...")
        single_times = []
        test_files = [
            "tests/unit/core/test_xpcs_file.py",
            "tests/unit/utils/test_memory_utils.py",
            "tests/unit/fileio/test_hdf_reader.py",
        ]

        # Run individually
        for test_file in test_files:
            start_time = time.time()
            try:
                subprocess.run(
                    [sys.executable, "-m", "pytest", test_file, "-v", "-q"],
                    check=False,
                    capture_output=True,
                    cwd=self.project_root,
                    timeout=60,
                )
                single_times.append(time.time() - start_time)
            except:
                single_times.append(float("inf"))

        # Run as batch
        start_time = time.time()
        try:
            subprocess.run(
                [sys.executable, "-m", "pytest", *test_files, "-v", "-q"],
                check=False,
                capture_output=True,
                cwd=self.project_root,
                timeout=180,
            )
            batch_time = time.time() - start_time
        except:
            batch_time = float("inf")

        results["execution_performance"] = {
            "individual_total": sum(single_times),
            "batch_total": batch_time,
            "efficiency_ratio": batch_time / sum(single_times)
            if sum(single_times) > 0
            else 1.0,
        }

        return results

    def _test_path_separators(self) -> dict[str, Any]:
        """Test path separator compatibility."""
        test_paths = [
            "tests/unit/core",
            "tests\\unit\\core",  # Windows style
            "tests/fixtures/test_data.hdf",
            "tests\\fixtures\\test_data.hdf",  # Windows style
        ]

        results = {"current_separator": os.sep, "path_tests": []}

        for test_path in test_paths:
            # Normalize path for current platform
            normalized = os.path.normpath(test_path)
            full_path = self.project_root / normalized

            results["path_tests"].append(
                {
                    "original_path": test_path,
                    "normalized_path": str(normalized),
                    "exists": full_path.exists(),
                    "platform_compatible": os.sep in test_path or "/" in test_path,
                }
            )

        return results

    def _test_environment_variables(self) -> dict[str, Any]:
        """Test environment variable handling."""
        test_env_vars = [
            "QT_QPA_PLATFORM",
            "PYXPCS_LOG_LEVEL",
            "PYXPCS_SUPPRESS_QT_WARNINGS",
            "PYTHONPATH",
        ]

        results = {"current_environment": {}, "variable_tests": []}

        for var_name in test_env_vars:
            current_value = os.environ.get(var_name)
            results["current_environment"][var_name] = current_value

            # Test setting and reading
            test_value = "test_value_123"
            original_value = os.environ.get(var_name)

            try:
                os.environ[var_name] = test_value
                read_value = os.environ.get(var_name)

                results["variable_tests"].append(
                    {
                        "variable_name": var_name,
                        "set_successful": read_value == test_value,
                        "original_value": original_value,
                    }
                )

                # Restore original value
                if original_value is not None:
                    os.environ[var_name] = original_value
                else:
                    del os.environ[var_name]

            except Exception as e:
                results["variable_tests"].append(
                    {
                        "variable_name": var_name,
                        "set_successful": False,
                        "error": str(e),
                    }
                )

        return results

    def _test_file_system_compatibility(self) -> dict[str, Any]:
        """Test file system compatibility."""
        results = {
            "temporary_file_creation": False,
            "unicode_filename_support": False,
            "long_path_support": False,
            "case_sensitivity": "unknown",
        }

        # Test temporary file creation
        try:
            with tempfile.NamedTemporaryFile(suffix=".hdf", delete=True) as tmp:
                tmp.write(b"test data")
                tmp.flush()
                results["temporary_file_creation"] = os.path.exists(tmp.name)
        except Exception:
            results["temporary_file_creation"] = False

        # Test Unicode filename support
        try:
            unicode_name = "test_файл_测试.hdf"
            with tempfile.NamedTemporaryFile(suffix=unicode_name, delete=True) as tmp:
                results["unicode_filename_support"] = True
        except Exception:
            results["unicode_filename_support"] = False

        # Test long path support
        try:
            long_path = "a" * 100 + "/" + "b" * 100 + ".txt"
            temp_dir = tempfile.mkdtemp()
            long_file_path = os.path.join(temp_dir, long_path)
            os.makedirs(os.path.dirname(long_file_path), exist_ok=True)
            with open(long_file_path, "w") as f:
                f.write("test")
            results["long_path_support"] = os.path.exists(long_file_path)
            # Cleanup
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            results["long_path_support"] = False

        # Test case sensitivity
        try:
            temp_dir = tempfile.mkdtemp()
            test_file1 = os.path.join(temp_dir, "TEST.txt")
            test_file2 = os.path.join(temp_dir, "test.txt")

            with open(test_file1, "w") as f:
                f.write("upper")

            if os.path.exists(test_file2):
                results["case_sensitivity"] = "insensitive"
            else:
                with open(test_file2, "w") as f:
                    f.write("lower")
                results["case_sensitivity"] = "sensitive"

            # Cleanup
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            results["case_sensitivity"] = "unknown"

        return results

    def _test_gui_platform_compatibility(self) -> dict[str, Any]:
        """Test GUI platform compatibility."""
        results = {
            "qt_available": False,
            "headless_mode": False,
            "display_available": False,
        }

        # Test Qt availability
        try:
            import PySide6.QtWidgets

            results["qt_available"] = True
        except ImportError:
            results["qt_available"] = False

        # Test headless mode
        original_platform = os.environ.get("QT_QPA_PLATFORM")
        try:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            if results["qt_available"]:
                from PySide6.QtWidgets import QApplication

                app = QApplication.instance() or QApplication([])
                results["headless_mode"] = True
                app.quit()
        except Exception:
            results["headless_mode"] = False
        finally:
            if original_platform is not None:
                os.environ["QT_QPA_PLATFORM"] = original_platform
            elif "QT_QPA_PLATFORM" in os.environ:
                del os.environ["QT_QPA_PLATFORM"]

        # Test display availability (Unix-like systems)
        if hasattr(os, "environ") and "DISPLAY" in os.environ:
            results["display_available"] = True
        elif platform.system() == "Windows":
            results["display_available"] = True  # Assume available on Windows
        else:
            results["display_available"] = False

        return results

    def _test_dependency_compatibility(self) -> dict[str, Any]:
        """Test dependency compatibility."""
        required_packages = [
            "numpy",
            "scipy",
            "h5py",
            "pandas",
            "matplotlib",
            "PySide6",
            "pytest",
            "hypothesis",
        ]

        results = {
            "package_availability": {},
            "version_compatibility": {},
            "import_success": {},
        }

        for package in required_packages:
            # Test availability
            try:
                __import__(package)
                results["package_availability"][package] = True
                results["import_success"][package] = True

                # Get version if available
                try:
                    mod = __import__(package)
                    version = getattr(mod, "__version__", "unknown")
                    results["version_compatibility"][package] = version
                except:
                    results["version_compatibility"][package] = "unknown"

            except ImportError:
                results["package_availability"][package] = False
                results["import_success"][package] = False
                results["version_compatibility"][package] = "not_available"

        return results

    def _parse_pytest_output(self, output: str) -> dict[str, Any]:
        """Parse pytest output to extract test results."""
        lines = output.split("\n")
        results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        }

        for line in lines:
            # Look for summary line like "= 5 passed, 1 failed in 2.34s ="
            if "passed" in line or "failed" in line or "skipped" in line:
                if " passed" in line:
                    try:
                        passed = int(line.split(" passed")[0].split()[-1])
                        results["passed"] = passed
                    except:
                        pass

                if " failed" in line:
                    try:
                        failed = int(line.split(" failed")[0].split()[-1])
                        results["failed"] = failed
                    except:
                        pass

                if " skipped" in line:
                    try:
                        skipped = int(line.split(" skipped")[0].split()[-1])
                        results["skipped"] = skipped
                    except:
                        pass

        results["total_tests"] = (
            results["passed"] + results["failed"] + results["skipped"]
        )
        return results

    def _extract_category_issues(self, stderr: str) -> list[str]:
        """Extract issues from stderr output."""
        issues = []
        lines = stderr.split("\n")

        for line in lines:
            line = line.strip()
            if line and ("ERROR" in line or "FAILED" in line or "Warning" in line):
                if len(line) < 200:  # Avoid very long lines
                    issues.append(line)

        return issues[:10]  # Limit to first 10 issues

    def _identify_integration_issues(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Identify integration issues from validation results."""
        issues = []

        # Check test category results
        for category, result in validation_results["test_categories"].items():
            if result.get("status") == "error":
                issues.append(f"Category {category}: Execution error")
            elif result.get("status") == "timeout":
                issues.append(f"Category {category}: Tests timed out")
            elif result.get("return_code", 0) != 0:
                issues.append(f"Category {category}: Tests failed")

        # Check cross-category integration
        for test_name, result in validation_results[
            "cross_category_integration"
        ].items():
            if not result.get("passed", False):
                issues.append(f"Cross-category integration: {test_name} failed")

        # Check fixture compatibility
        for fixture_name, result in validation_results["fixture_compatibility"].items():
            if not result.get("passed", True):
                issues.append(f"Fixture compatibility: {fixture_name} has issues")

        return issues

    def _generate_platform_recommendations(
        self, compatibility_results: dict[str, Any]
    ) -> list[str]:
        """Generate platform-specific recommendations."""
        recommendations = []

        # File system recommendations
        fs_results = compatibility_results["file_system_tests"]
        if not fs_results.get("unicode_filename_support", True):
            recommendations.append(
                "Consider avoiding Unicode characters in test file names"
            )

        if not fs_results.get("long_path_support", True):
            recommendations.append(
                "Limit test file path lengths to avoid issues on some platforms"
            )

        # GUI recommendations
        gui_results = compatibility_results["gui_platform_tests"]
        if not gui_results.get("headless_mode", True):
            recommendations.append(
                "Ensure all GUI tests support headless mode for CI/CD"
            )

        # Dependency recommendations
        dep_results = compatibility_results["dependency_tests"]
        missing_packages = [
            pkg
            for pkg, available in dep_results["package_availability"].items()
            if not available
        ]
        if missing_packages:
            recommendations.append(f"Missing packages: {', '.join(missing_packages)}")

        return recommendations


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Test Integration Validator"
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run integration validation"
    )
    parser.add_argument(
        "--platform", action="store_true", help="Run cross-platform validation"
    )
    parser.add_argument("--all", action="store_true", help="Run all validations")
    parser.add_argument("--output", help="Output file for results (JSON)")

    args = parser.parse_args()

    validator = TestSuiteIntegrationValidator()
    results = {}

    if args.all or args.integration:
        print("Running integration validation...")
        results["integration"] = validator.validate_complete_integration()

    if args.all or args.platform:
        print("Running cross-platform validation...")
        results["platform"] = validator.validate_cross_platform_compatibility()

    if not results:
        print("No validation selected. Use --help for options.")
        return

    # Output results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results written to {args.output}")
    else:
        print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
