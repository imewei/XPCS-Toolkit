"""Qt Error Regression Testing Framework.

This module provides a comprehensive framework for tracking and preventing
regression of Qt-related errors in the XPCS Toolkit application.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

try:
    from .qt_test_runner import XpcsQtTestRunner
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from qt_test_runner import XpcsQtTestRunner

try:
    from ..utils.qt_threading_utils import (
        BackgroundThreadTester,
        QtThreadSafetyValidator,
        ThreadingViolationDetector,
    )
except ImportError:
    # Fallback for standalone execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class ErrorBaseline:
    """Represents a baseline of Qt errors for regression testing."""

    timestamp: str
    total_errors: int
    timer_errors: int
    connection_errors: int
    other_errors: int
    error_patterns: dict[str, int]
    test_environment: dict[str, str]
    git_commit: str | None = None
    notes: str = ""


@dataclass
class RegressionTestResult:
    """Result of a regression test run."""

    timestamp: str
    baseline_name: str
    current_errors: ErrorBaseline
    baseline_errors: ErrorBaseline
    regression_detected: bool
    new_errors: list[str]
    fixed_errors: list[str]
    error_delta: dict[str, int]
    improvement_score: float
    recommendations: list[str]


class QtErrorRegressionTester:
    """Framework for Qt error regression testing."""

    def __init__(self, baseline_dir="tests/baselines/qt_errors"):
        """
        Initialize regression tester.

        Args:
            baseline_dir: Directory to store error baselines
        """
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.current_baseline = None
        self.test_runner = XpcsQtTestRunner()

    def create_baseline(
        self, name: str, notes: str = "", force: bool = False
    ) -> ErrorBaseline:
        """
        Create a new error baseline by running comprehensive tests.

        Args:
            name: Name for the baseline
            notes: Optional notes about the baseline
            force: Whether to overwrite existing baseline

        Returns:
            ErrorBaseline object
        """
        baseline_file = self.baseline_dir / f"{name}.json"

        if baseline_file.exists() and not force:
            raise ValueError(
                f"Baseline '{name}' already exists. Use force=True to overwrite."
            )

        print(f"Creating Qt error baseline '{name}'...")

        # Run comprehensive test suite
        results = self.test_runner.run_comprehensive_xpcs_test_suite()

        # Get git commit if available
        git_commit = self._get_git_commit()

        # Create baseline
        baseline = ErrorBaseline(
            timestamp=datetime.now().isoformat(),
            total_errors=results.get("total_qt_errors", 0),
            timer_errors=self._count_error_type(results, "timer"),
            connection_errors=self._count_error_type(results, "connection"),
            other_errors=self._count_error_type(results, "other"),
            error_patterns=self._extract_error_patterns(results),
            test_environment=self._get_test_environment(),
            git_commit=git_commit,
            notes=notes,
        )

        # Save baseline
        self._save_baseline(name, baseline)
        self.current_baseline = baseline

        print(f"Baseline '{name}' created with {baseline.total_errors} total errors")
        return baseline

    def run_regression_test(self, baseline_name: str) -> RegressionTestResult:
        """
        Run regression test against specified baseline.

        Args:
            baseline_name: Name of baseline to test against

        Returns:
            RegressionTestResult object
        """
        # Load baseline
        baseline = self._load_baseline(baseline_name)
        if baseline is None:
            raise ValueError(f"Baseline '{baseline_name}' not found")

        print(f"Running regression test against baseline '{baseline_name}'...")

        # Run current tests
        current_results = self.test_runner.run_comprehensive_xpcs_test_suite()

        # Create current error baseline
        current_baseline = ErrorBaseline(
            timestamp=datetime.now().isoformat(),
            total_errors=current_results.get("total_qt_errors", 0),
            timer_errors=self._count_error_type(current_results, "timer"),
            connection_errors=self._count_error_type(current_results, "connection"),
            other_errors=self._count_error_type(current_results, "other"),
            error_patterns=self._extract_error_patterns(current_results),
            test_environment=self._get_test_environment(),
            git_commit=self._get_git_commit(),
        )

        # Analyze regression
        regression_result = self._analyze_regression(
            baseline, current_baseline, baseline_name
        )

        # Save regression test result
        self._save_regression_result(regression_result)

        return regression_result

    def _analyze_regression(
        self, baseline: ErrorBaseline, current: ErrorBaseline, baseline_name: str
    ) -> RegressionTestResult:
        """Analyze regression between baseline and current results."""

        # Calculate error delta
        error_delta = {
            "total_errors": current.total_errors - baseline.total_errors,
            "timer_errors": current.timer_errors - baseline.timer_errors,
            "connection_errors": current.connection_errors - baseline.connection_errors,
            "other_errors": current.other_errors - baseline.other_errors,
        }

        # Detect regression
        regression_detected = error_delta["total_errors"] > 0

        # Find new and fixed errors
        baseline_patterns = set(baseline.error_patterns.keys())
        current_patterns = set(current.error_patterns.keys())

        new_errors = list(current_patterns - baseline_patterns)
        fixed_errors = list(baseline_patterns - current_patterns)

        # Calculate improvement score (0-100, where 100 is perfect)
        if baseline.total_errors == 0:
            improvement_score = 100.0 if current.total_errors == 0 else 0.0
        else:
            improvement_score = max(
                0, 100 * (1 - current.total_errors / baseline.total_errors)
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            error_delta, new_errors, fixed_errors
        )

        return RegressionTestResult(
            timestamp=datetime.now().isoformat(),
            baseline_name=baseline_name,
            current_errors=current,
            baseline_errors=baseline,
            regression_detected=regression_detected,
            new_errors=new_errors,
            fixed_errors=fixed_errors,
            error_delta=error_delta,
            improvement_score=improvement_score,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        error_delta: dict[str, int],
        new_errors: list[str],
        fixed_errors: list[str],
    ) -> list[str]:
        """Generate recommendations based on regression analysis."""
        recommendations = []

        if error_delta["total_errors"] > 0:
            recommendations.append("🔴 REGRESSION DETECTED: Total Qt errors increased")

        if error_delta["timer_errors"] > 0:
            recommendations.append(
                "⚠️ Timer threading violations increased - review background thread usage"
            )

        if error_delta["connection_errors"] > 0:
            recommendations.append(
                "⚠️ Signal/slot connection errors increased - review Qt5+ syntax usage"
            )

        if new_errors:
            recommendations.append(
                f"🆕 New error patterns detected: {', '.join(new_errors[:3])}..."
            )

        if fixed_errors:
            recommendations.append(
                f"✅ Fixed error patterns: {', '.join(fixed_errors[:3])}..."
            )

        if error_delta["total_errors"] < 0:
            recommendations.append(
                f"🎉 IMPROVEMENT: Reduced Qt errors by {abs(error_delta['total_errors'])}"
            )

        if not recommendations:
            recommendations.append("✅ No regression detected - Qt error count stable")

        return recommendations

    def _count_error_type(self, results: dict, error_type: str) -> int:
        """Count specific type of errors in test results."""
        count = 0

        if "error_summary" not in results:
            return 0

        error_patterns = results["error_summary"].get("error_patterns", {})

        for pattern, messages in error_patterns.items():
            if error_type.lower() in pattern.lower():
                count += len(messages)

        return count

    def _extract_error_patterns(self, results: dict) -> dict[str, int]:
        """Extract error patterns and their counts from test results."""
        patterns = {}

        if "error_summary" not in results:
            return patterns

        error_counts = results["error_summary"].get("error_counts", {})
        return error_counts

    def _get_test_environment(self) -> dict[str, str]:
        """Get test environment information."""
        try:
            from PySide6 import __version__ as pyside_version
        except ImportError:
            pyside_version = "unknown"

        return {
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "pyside6_version": pyside_version,
            "platform": os.name,
            "qt_qpa_platform": os.environ.get("QT_QPA_PLATFORM", "unknown"),
        }

    def _get_git_commit(self) -> str | None:
        """Get current git commit hash."""
        try:
            import subprocess

            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return None

    def _save_baseline(self, name: str, baseline: ErrorBaseline):
        """Save baseline to file."""
        baseline_file = self.baseline_dir / f"{name}.json"
        with open(baseline_file, "w") as f:
            json.dump(asdict(baseline), f, indent=2)

    def _load_baseline(self, name: str) -> ErrorBaseline | None:
        """Load baseline from file."""
        baseline_file = self.baseline_dir / f"{name}.json"
        if not baseline_file.exists():
            return None

        with open(baseline_file) as f:
            data = json.load(f)
            return ErrorBaseline(**data)

    def _save_regression_result(self, result: RegressionTestResult):
        """Save regression test result."""
        results_dir = self.baseline_dir / "regression_results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"regression_{timestamp}.json"

        # Convert dataclass to dict
        result_dict = asdict(result)

        with open(result_file, "w") as f:
            json.dump(result_dict, f, indent=2)

    def list_baselines(self) -> list[str]:
        """List available baselines."""
        baselines = []
        for file in self.baseline_dir.glob("*.json"):
            if file.stem != "regression_results":
                baselines.append(file.stem)
        return sorted(baselines)

    def get_baseline_info(self, name: str) -> ErrorBaseline | None:
        """Get information about a specific baseline."""
        return self._load_baseline(name)

    def compare_baselines(self, baseline1: str, baseline2: str) -> dict[str, Any]:
        """Compare two baselines."""
        b1 = self._load_baseline(baseline1)
        b2 = self._load_baseline(baseline2)

        if b1 is None or b2 is None:
            raise ValueError("One or both baselines not found")

        comparison = {
            "baseline1": baseline1,
            "baseline2": baseline2,
            "total_error_diff": b2.total_errors - b1.total_errors,
            "timer_error_diff": b2.timer_errors - b1.timer_errors,
            "connection_error_diff": b2.connection_errors - b1.connection_errors,
            "time_difference": b2.timestamp,
            "improvement": b2.total_errors < b1.total_errors,
        }

        return comparison

    def generate_trend_report(self, days: int = 30) -> dict[str, Any]:
        """Generate trend report from recent regression results."""
        results_dir = self.baseline_dir / "regression_results"
        if not results_dir.exists():
            return {"error": "No regression results available"}

        # Load recent results
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        recent_results = []

        for result_file in results_dir.glob("regression_*.json"):
            try:
                with open(result_file) as f:
                    data = json.load(f)

                result_time = datetime.fromisoformat(data["timestamp"]).timestamp()
                if result_time > cutoff_time:
                    recent_results.append(data)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

        if not recent_results:
            return {"error": f"No regression results in last {days} days"}

        # Sort by timestamp
        recent_results.sort(key=lambda x: x["timestamp"])

        # Calculate trends
        total_errors = [r["current_errors"]["total_errors"] for r in recent_results]
        improvement_scores = [r["improvement_score"] for r in recent_results]

        trend_report = {
            "period_days": days,
            "total_test_runs": len(recent_results),
            "error_trend": {
                "min_errors": min(total_errors),
                "max_errors": max(total_errors),
                "avg_errors": sum(total_errors) / len(total_errors),
                "latest_errors": total_errors[-1] if total_errors else 0,
            },
            "improvement_trend": {
                "min_score": min(improvement_scores),
                "max_score": max(improvement_scores),
                "avg_score": sum(improvement_scores) / len(improvement_scores),
                "latest_score": improvement_scores[-1] if improvement_scores else 0,
            },
            "regressions_detected": sum(
                1 for r in recent_results if r["regression_detected"]
            ),
            "recent_results": recent_results[-5:],  # Last 5 results
        }

        return trend_report


class ContinuousRegressionMonitor:
    """Continuous monitoring for Qt error regression."""

    def __init__(self, baseline_name: str = "main", monitor_interval: int = 3600):
        """
        Initialize continuous monitor.

        Args:
            baseline_name: Name of baseline to monitor against
            monitor_interval: Monitoring interval in seconds
        """
        self.baseline_name = baseline_name
        self.monitor_interval = monitor_interval
        self.tester = QtErrorRegressionTester()
        self.monitoring = False

    def start_monitoring(self):
        """Start continuous monitoring."""
        import threading

        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.monitoring = False

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                result = self.tester.run_regression_test(self.baseline_name)

                if result.regression_detected:
                    self._alert_regression(result)

                time.sleep(self.monitor_interval)

            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _alert_regression(self, result: RegressionTestResult):
        """Alert about detected regression."""
        print("🚨 Qt ERROR REGRESSION DETECTED!")
        print(f"Baseline: {result.baseline_name}")
        print(f"Error increase: {result.error_delta['total_errors']}")
        print(f"New errors: {', '.join(result.new_errors[:3])}")


# Pytest integration
class QtErrorRegressionPlugin:
    """Pytest plugin for Qt error regression testing."""

    def __init__(self):
        self.tester = QtErrorRegressionTester()

    def pytest_configure(self, config):
        """Configure pytest for regression testing."""
        # Add marker for regression tests
        config.addinivalue_line(
            "markers", "qt_regression: mark test as Qt error regression test"
        )

    @pytest.fixture
    def qt_regression_tester(self):
        """Fixture providing regression tester."""
        return self.tester

    def pytest_runtest_teardown(self, item, nextitem):
        """Check for regressions after each test."""
        if item.get_closest_marker("qt_regression"):
            # Run regression check
            pass


# CLI interface
def main():
    """Command-line interface for Qt error regression testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Qt Error Regression Testing")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create baseline command
    create_parser = subparsers.add_parser(
        "create-baseline", help="Create new error baseline"
    )
    create_parser.add_argument("name", help="Baseline name")
    create_parser.add_argument("--notes", default="", help="Notes about the baseline")
    create_parser.add_argument(
        "--force", action="store_true", help="Overwrite existing baseline"
    )

    # Run regression test command
    test_parser = subparsers.add_parser("test", help="Run regression test")
    test_parser.add_argument("baseline", help="Baseline name to test against")

    # List baselines command
    subparsers.add_parser("list", help="List available baselines")

    # Compare baselines command
    compare_parser = subparsers.add_parser("compare", help="Compare two baselines")
    compare_parser.add_argument("baseline1", help="First baseline")
    compare_parser.add_argument("baseline2", help="Second baseline")

    # Trend report command
    trend_parser = subparsers.add_parser("trend", help="Generate trend report")
    trend_parser.add_argument(
        "--days", type=int, default=30, help="Number of days to analyze"
    )

    args = parser.parse_args()

    tester = QtErrorRegressionTester()

    if args.command == "create-baseline":
        baseline = tester.create_baseline(args.name, args.notes, args.force)
        print(f"✅ Created baseline '{args.name}' with {baseline.total_errors} errors")

    elif args.command == "test":
        result = tester.run_regression_test(args.baseline)
        print("📊 Regression Test Results:")
        print(f"Baseline: {result.baseline_name}")
        print(
            f"Regression detected: {'🔴 YES' if result.regression_detected else '✅ NO'}"
        )
        print(f"Error delta: {result.error_delta['total_errors']}")
        print(f"Improvement score: {result.improvement_score:.1f}%")

        if result.recommendations:
            print("\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  {rec}")

    elif args.command == "list":
        baselines = tester.list_baselines()
        print("📋 Available baselines:")
        for baseline in baselines:
            info = tester.get_baseline_info(baseline)
            print(f"  {baseline}: {info.total_errors} errors ({info.timestamp})")

    elif args.command == "compare":
        comparison = tester.compare_baselines(args.baseline1, args.baseline2)
        print(f"📊 Comparison: {args.baseline1} vs {args.baseline2}")
        print(f"Error difference: {comparison['total_error_diff']}")
        print(f"Improvement: {'✅ YES' if comparison['improvement'] else '❌ NO'}")

    elif args.command == "trend":
        report = tester.generate_trend_report(args.days)
        if "error" in report:
            print(f"❌ {report['error']}")
        else:
            print(f"📈 Trend Report (last {args.days} days):")
            print(f"Test runs: {report['total_test_runs']}")
            print(
                f"Error range: {report['error_trend']['min_errors']}-{report['error_trend']['max_errors']}"
            )
            print(f"Regressions detected: {report['regressions_detected']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
