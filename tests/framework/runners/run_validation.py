#!/usr/bin/env python3
"""
XPCS Toolkit Test Validation Runner

This script provides comprehensive validation capabilities for the XPCS Toolkit,
including test execution, quality checks, coverage analysis, and reporting.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def run_command(cmd, description, cwd=None):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or get_project_root(),
            capture_output=False,
            text=True,
            check=True
        )
        elapsed = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {description} failed after {elapsed:.2f}s (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found: {cmd[0]}")
        return False


def run_tests(scope="full"):
    """Run pytest with appropriate scope."""
    project_root = get_project_root()

    if scope == "ci":
        # Fast CI tests - exclude slow and flaky tests
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "-m", "not (stress or system_dependent or flaky)",
            "--durations=10"
        ]
        description = "Running CI Test Suite (fast, stable tests only)"
    else:
        # Full test suite
        cmd = [
            sys.executable, "-m", "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--durations=10"
        ]
        description = "Running Full Test Suite"

    return run_command(cmd, description, cwd=project_root)


def run_linting():
    """Run code quality checks."""
    project_root = get_project_root()
    success = True

    # Run ruff linting
    ruff_cmd = [sys.executable, "-m", "ruff", "check", "xpcs_toolkit/", "tests/"]
    if not run_command(ruff_cmd, "Running Ruff Linting", cwd=project_root):
        success = False

    # Run type checking if mypy is available
    try:
        subprocess.run([sys.executable, "-c", "import mypy"], check=True, capture_output=True)
        mypy_cmd = [sys.executable, "-m", "mypy", "xpcs_toolkit/", "--ignore-missing-imports"]
        if not run_command(mypy_cmd, "Running Type Checking", cwd=project_root):
            print("⚠️  Type checking failed but continuing...")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ℹ️  MyPy not available, skipping type checking")

    return success


def run_coverage():
    """Run test coverage analysis."""
    project_root = get_project_root()

    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "--cov=xpcs_toolkit",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "tests/",
        "-v"
    ]

    return run_command(cmd, "Running Test Coverage Analysis", cwd=project_root)


def generate_report():
    """Generate validation report."""
    print(f"\n{'='*60}")
    print("📊 Validation Report")
    print(f"{'='*60}")

    project_root = get_project_root()

    # Count test files
    test_files = list(Path(project_root / "tests").rglob("test_*.py"))
    print(f"📁 Test files found: {len(test_files)}")

    # Check coverage report if it exists
    htmlcov_path = project_root / "htmlcov" / "index.html"
    if htmlcov_path.exists():
        print(f"📈 Coverage report: {htmlcov_path}")

    # Check for common issues
    issues = []

    # Check for __pycache__ directories
    pycache_dirs = list(project_root.rglob("__pycache__"))
    if pycache_dirs:
        issues.append(f"Found {len(pycache_dirs)} __pycache__ directories")

    if issues:
        print(f"\n⚠️  Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n✅ No issues found")


def main():
    """Main validation runner."""
    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Test Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --full      # Run full validation suite
  python run_validation.py --ci        # Run CI-safe tests only
  python run_validation.py --lint-only # Run linting only
  python run_validation.py --coverage  # Run with coverage analysis
        """
    )

    parser.add_argument(
        "--full", action="store_true",
        help="Run full test suite with all validations"
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="Run CI-safe tests only (fast, exclude flaky tests)"
    )
    parser.add_argument(
        "--lint-only", action="store_true",
        help="Run code quality checks only"
    )
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run tests with coverage analysis"
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate validation report"
    )

    args = parser.parse_args()

    # Default to full validation if no specific mode selected
    if not any([args.full, args.ci, args.lint_only, args.coverage, args.report]):
        args.full = True

    success = True
    start_time = time.time()

    print("🚀 XPCS Toolkit Validation Runner")
    print(f"Project root: {get_project_root()}")

    if args.lint_only:
        success = run_linting()
    elif args.coverage:
        success = run_coverage()
    elif args.report:
        generate_report()
    else:
        # Run tests first
        test_scope = "ci" if args.ci else "full"
        if not run_tests(test_scope):
            success = False

        # Run linting for full validation
        if args.full:
            if not run_linting():
                success = False

    # Always generate report at the end
    if not args.lint_only:
        generate_report()

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    if success:
        print(f"✅ Validation completed successfully in {elapsed:.2f}s")
        sys.exit(0)
    else:
        print(f"❌ Validation failed after {elapsed:.2f}s")
        sys.exit(1)


if __name__ == "__main__":
    main()