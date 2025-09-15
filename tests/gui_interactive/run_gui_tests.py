#!/usr/bin/env python3
"""
Comprehensive GUI test runner for XPCS Toolkit.

This script provides various test execution modes for GUI testing including
quick validation, full test suite, performance testing, and CI-friendly
headless execution.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_environment():
    """Set up environment variables for GUI testing."""
    # Set Qt platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Suppress Qt warnings
    os.environ.setdefault("PYXPCS_SUPPRESS_QT_WARNINGS", "1")

    # Set display for headless systems (if needed)
    if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"


def run_pytest(test_args, verbose=True):
    """Run pytest with specified arguments."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")

    cmd.extend(test_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, cwd=project_root)
    return result.returncode


def run_quick_tests():
    """Run quick GUI validation tests."""
    print("=== Running Quick GUI Tests ===")

    test_args = [
        "tests/gui_interactive/test_main_window.py",
        "-m",
        "gui and not slow",
        "--tb=short",
        "--maxfail=5",
    ]

    return run_pytest(test_args)


def run_full_test_suite():
    """Run complete GUI test suite."""
    print("=== Running Full GUI Test Suite ===")

    test_args = [
        "tests/gui_interactive/",
        "-m",
        "gui",
        "--tb=line",
        "--continue-on-collection-errors",
    ]

    return run_pytest(test_args)


def run_performance_tests():
    """Run performance and responsiveness tests."""
    print("=== Running Performance Tests ===")

    test_args = [
        "tests/gui_interactive/test_performance.py",
        "-m",
        "performance",
        "--tb=short",
        "-x",  # Stop on first failure for performance tests
    ]

    return run_pytest(test_args)


def run_error_handling_tests():
    """Run error handling and edge case tests."""
    print("=== Running Error Handling Tests ===")

    test_args = [
        "tests/gui_interactive/test_error_handling.py",
        "-m",
        "gui",
        "--tb=short",
    ]

    return run_pytest(test_args)


def run_user_scenario_tests():
    """Run user interaction scenario tests."""
    print("=== Running User Scenario Tests ===")

    test_args = [
        "tests/gui_interactive/test_user_scenarios.py",
        "-m",
        "gui and not slow",
        "--tb=short",
    ]

    return run_pytest(test_args)


def run_ci_tests():
    """Run CI-friendly GUI tests (headless, fast)."""
    print("=== Running CI GUI Tests ===")

    # Ensure headless mode
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    test_args = [
        "tests/gui_interactive/",
        "-m",
        "gui and not slow and not performance",
        "--tb=no",
        "--quiet",
        "--maxfail=10",
        "--durations=10",
    ]

    return run_pytest(test_args, verbose=False)


def run_coverage_tests():
    """Run GUI tests with coverage reporting."""
    print("=== Running GUI Tests with Coverage ===")

    test_args = [
        "tests/gui_interactive/",
        "-m",
        "gui and not slow",
        "--cov=xpcs_toolkit",
        "--cov-report=html:htmlcov_gui",
        "--cov-report=term-missing",
        "--tb=short",
    ]

    return run_pytest(test_args)


def run_specific_test_file(test_file):
    """Run a specific test file."""
    print(f"=== Running Specific Test File: {test_file} ===")

    test_path = f"tests/gui_interactive/{test_file}"
    if not Path(project_root / test_path).exists():
        print(f"Error: Test file {test_path} not found")
        return 1

    test_args = [test_path, "-v", "--tb=short"]

    return run_pytest(test_args)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="XPCS Toolkit GUI Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  quick       - Run basic GUI validation tests (fast)
  full        - Run complete GUI test suite
  performance - Run performance and responsiveness tests
  errors      - Run error handling and edge case tests
  scenarios   - Run user interaction scenario tests
  ci          - Run CI-friendly tests (headless, no slow tests)
  coverage    - Run tests with coverage reporting

Examples:
  python run_gui_tests.py quick
  python run_gui_tests.py full --headless
  python run_gui_tests.py performance
  python run_gui_tests.py test_main_window.py
        """,
    )

    parser.add_argument(
        "mode", nargs="?", default="quick", help="Test mode to run (default: quick)"
    )

    parser.add_argument(
        "--headless", action="store_true", help="Force headless mode (offscreen)"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    parser.add_argument("--markers", help="Additional pytest markers to apply")

    args = parser.parse_args()

    # Set up environment
    setup_environment()

    if args.headless:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    # Route to appropriate test function
    if args.mode == "quick":
        return run_quick_tests()
    if args.mode == "full":
        return run_full_test_suite()
    if args.mode == "performance":
        return run_performance_tests()
    if args.mode == "errors":
        return run_error_handling_tests()
    if args.mode == "scenarios":
        return run_user_scenario_tests()
    if args.mode == "ci":
        return run_ci_tests()
    if args.mode == "coverage":
        return run_coverage_tests()
    if args.mode.endswith(".py"):
        return run_specific_test_file(args.mode)
    print(f"Unknown test mode: {args.mode}")
    parser.print_help()
    return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
