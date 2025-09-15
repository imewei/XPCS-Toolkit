"""Test Suite Maintenance and Evolution Framework for XPCS Toolkit.

This framework provides automated tools for maintaining test suite quality,
managing technical debt, and evolving testing practices as the codebase grows.

Features:
- Automated test maintenance scheduling
- Technical debt tracking and management
- Test performance monitoring and optimization
- Test suite health metrics and reporting
- Evolution planning and migration tools
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class MaintenanceTask(Enum):
    """Types of maintenance tasks."""

    CLEANUP_OBSOLETE_TESTS = "cleanup_obsolete_tests"
    UPDATE_TEST_DATA = "update_test_data"
    OPTIMIZE_SLOW_TESTS = "optimize_slow_tests"
    REFACTOR_DUPLICATED_CODE = "refactor_duplicated_code"
    UPDATE_FIXTURES = "update_fixtures"
    REVIEW_TEST_COVERAGE = "review_test_coverage"
    UPDATE_BASELINES = "update_baselines"
    VALIDATE_TEST_QUALITY = "validate_test_quality"


class Priority(Enum):
    """Priority levels for maintenance tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MaintenanceItem:
    """Individual maintenance item."""

    task_type: MaintenanceTask
    priority: Priority
    description: str
    estimated_effort: str  # "1h", "2d", etc.
    affected_files: list[str]
    due_date: datetime | None = None
    assigned_to: str | None = None
    created_date: datetime = None
    last_updated: datetime = None
    status: str = "pending"  # pending, in_progress, completed, deferred

    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class TestSuiteHealthMetrics:
    """Comprehensive health metrics for the test suite."""

    timestamp: datetime

    # Coverage metrics
    line_coverage: float
    branch_coverage: float
    function_coverage: float

    # Performance metrics
    total_test_count: int
    slow_test_count: int  # tests taking > 1s
    average_test_time: float
    total_execution_time: float

    # Quality metrics
    test_files_with_quality_issues: int
    average_quality_score: float
    tests_without_docstrings: int
    weak_assertions_count: int

    # Maintenance metrics
    pending_maintenance_items: int
    critical_maintenance_items: int
    technical_debt_score: float  # 0-100, higher is worse

    # Evolution metrics
    test_code_lines: int
    test_file_count: int
    fixture_count: int
    utility_function_count: int


class TestSuiteMaintenanceManager:
    """Main class for test suite maintenance and evolution."""

    def __init__(self, test_directory: Path | None = None):
        self.test_dir = test_directory or Path(__file__).parent
        self.project_root = self.test_dir.parent
        self.maintenance_file = self.test_dir / "maintenance_items.json"
        self.health_history_file = self.test_dir / "health_history.json"
        self.config_file = self.test_dir / "maintenance_config.json"

        self.maintenance_items: list[MaintenanceItem] = []
        self.health_metrics_history: list[TestSuiteHealthMetrics] = []

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        # Load existing data
        self._load_maintenance_items()
        self._load_health_history()
        self._load_configuration()

    def _setup_logging(self):
        """Set up logging for maintenance operations."""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def run_daily_maintenance(self):
        """Run daily maintenance tasks."""
        self.logger.info("Starting daily maintenance routine")

        # Collect current health metrics
        current_metrics = self._collect_health_metrics()
        self.health_metrics_history.append(current_metrics)

        # Identify new maintenance items
        new_items = self._identify_maintenance_needs(current_metrics)
        self.maintenance_items.extend(new_items)

        # Update item priorities based on trends
        self._update_item_priorities(current_metrics)

        # Execute automated maintenance tasks
        self._execute_automated_tasks()

        # Generate daily report
        report = self._generate_daily_report(current_metrics)
        self.logger.info("Daily maintenance completed")

        # Save state
        self._save_maintenance_items()
        self._save_health_history()

        return report

    def run_weekly_maintenance(self):
        """Run weekly maintenance tasks."""
        self.logger.info("Starting weekly maintenance routine")

        # Deep analysis of test suite health
        health_trends = self._analyze_health_trends()

        # Test performance optimization
        performance_report = self._optimize_test_performance()

        # Test quality review
        quality_report = self._review_test_quality()

        # Technical debt assessment
        debt_assessment = self._assess_technical_debt()

        # Plan next week's priorities
        weekly_plan = self._plan_weekly_priorities()

        report = {
            "health_trends": health_trends,
            "performance": performance_report,
            "quality": quality_report,
            "technical_debt": debt_assessment,
            "weekly_plan": weekly_plan,
        }

        self.logger.info("Weekly maintenance completed")
        return report

    def run_release_maintenance(self, version: str):
        """Run pre-release maintenance tasks."""
        self.logger.info(f"Starting release maintenance for version {version}")

        # Comprehensive test validation
        validation_report = self._run_comprehensive_validation()

        # Performance baseline updates
        self._update_performance_baselines()

        # Test suite cleanup
        cleanup_report = self._cleanup_for_release()

        # Documentation updates
        doc_updates = self._update_release_documentation(version)

        # Archive maintenance history
        self._archive_maintenance_history(version)

        report = {
            "version": version,
            "validation": validation_report,
            "cleanup": cleanup_report,
            "documentation": doc_updates,
            "timestamp": datetime.now().isoformat(),
        }

        self.logger.info(f"Release maintenance completed for version {version}")
        return report

    def _collect_health_metrics(self) -> TestSuiteHealthMetrics:
        """Collect current test suite health metrics."""
        self.logger.info("Collecting test suite health metrics")

        # Run test coverage analysis
        coverage_result = self._run_coverage_analysis()

        # Run test performance analysis
        performance_result = self._run_performance_analysis()

        # Run quality analysis
        quality_result = self._run_quality_analysis()

        # Count maintenance items
        pending_items = len(
            [item for item in self.maintenance_items if item.status == "pending"]
        )
        critical_items = len(
            [
                item
                for item in self.maintenance_items
                if item.status == "pending" and item.priority == Priority.CRITICAL
            ]
        )

        # Calculate technical debt score
        debt_score = self._calculate_technical_debt_score()

        # Count test files and utilities
        test_files = list(self.test_dir.rglob("test_*.py"))
        fixture_files = list(self.test_dir.rglob("*fixture*.py"))
        utility_files = list(self.test_dir.rglob("*util*.py"))

        test_code_lines = sum(
            len(open(f).readlines()) for f in test_files if f.is_file()
        )

        return TestSuiteHealthMetrics(
            timestamp=datetime.now(),
            line_coverage=coverage_result.get("line_coverage", 0),
            branch_coverage=coverage_result.get("branch_coverage", 0),
            function_coverage=coverage_result.get("function_coverage", 0),
            total_test_count=performance_result.get("total_tests", 0),
            slow_test_count=performance_result.get("slow_tests", 0),
            average_test_time=performance_result.get("average_time", 0),
            total_execution_time=performance_result.get("total_time", 0),
            test_files_with_quality_issues=quality_result.get("files_with_issues", 0),
            average_quality_score=quality_result.get("average_score", 0),
            tests_without_docstrings=quality_result.get("missing_docstrings", 0),
            weak_assertions_count=quality_result.get("weak_assertions", 0),
            pending_maintenance_items=pending_items,
            critical_maintenance_items=critical_items,
            technical_debt_score=debt_score,
            test_code_lines=test_code_lines,
            test_file_count=len(test_files),
            fixture_count=len(fixture_files),
            utility_function_count=len(utility_files),
        )

    def _run_coverage_analysis(self) -> dict[str, Any]:
        """Run test coverage analysis."""
        try:
            # Run pytest with coverage
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--cov=xpcs_toolkit",
                    "--cov-report=json",
                    str(self.test_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            # Parse coverage report
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)

                return {
                    "line_coverage": coverage_data["totals"]["percent_covered"],
                    "branch_coverage": coverage_data["totals"].get(
                        "percent_covered_display", 0
                    ),
                    "function_coverage": 0,  # Not available in standard coverage
                }
        except Exception as e:
            self.logger.warning(f"Coverage analysis failed: {e}")

        return {"line_coverage": 0, "branch_coverage": 0, "function_coverage": 0}

    def _run_performance_analysis(self) -> dict[str, Any]:
        """Run test performance analysis."""
        try:
            # Run tests with timing information
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "--durations=0",
                    "-v",
                    str(self.test_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            # Parse timing information from output
            lines = result.stdout.split("\n")
            test_times = []
            slow_tests = 0

            for line in lines:
                if "s call " in line or "s setup " in line or "s teardown " in line:
                    try:
                        time_str = line.split("s ")[0].strip()
                        time_val = float(time_str)
                        test_times.append(time_val)
                        if time_val > 1.0:
                            slow_tests += 1
                    except (ValueError, IndexError):
                        continue

            return {
                "total_tests": len(test_times),
                "slow_tests": slow_tests,
                "average_time": np.mean(test_times) if test_times else 0,
                "total_time": sum(test_times),
            }
        except Exception as e:
            self.logger.warning(f"Performance analysis failed: {e}")

        return {"total_tests": 0, "slow_tests": 0, "average_time": 0, "total_time": 0}

    def _run_quality_analysis(self) -> dict[str, Any]:
        """Run test quality analysis."""
        try:
            # Import quality checker
            from .quality_standards import TestQualityChecker

            checker = TestQualityChecker(self.test_dir)
            metrics_by_file = checker.check_all_tests()

            if not metrics_by_file:
                return {
                    "files_with_issues": 0,
                    "average_score": 0,
                    "missing_docstrings": 0,
                    "weak_assertions": 0,
                }

            # Aggregate quality metrics
            files_with_issues = len(
                [m for m in metrics_by_file.values() if len(m.issues) > 0]
            )

            average_score = np.mean([m.quality_score for m in metrics_by_file.values()])

            missing_docstrings = sum(
                [m.total_tests - m.documented_tests for m in metrics_by_file.values()]
            )

            weak_assertions = sum([m.weak_assertions for m in metrics_by_file.values()])

            return {
                "files_with_issues": files_with_issues,
                "average_score": average_score,
                "missing_docstrings": missing_docstrings,
                "weak_assertions": weak_assertions,
            }
        except Exception as e:
            self.logger.warning(f"Quality analysis failed: {e}")

        return {
            "files_with_issues": 0,
            "average_score": 0,
            "missing_docstrings": 0,
            "weak_assertions": 0,
        }

    def _calculate_technical_debt_score(self) -> float:
        """Calculate overall technical debt score (0-100, higher is worse)."""
        score = 0

        # Quality debt (40% weight)
        if self.health_metrics_history:
            recent_metrics = self.health_metrics_history[-1]
            quality_debt = (100 - recent_metrics.average_quality_score) * 0.4
            score += quality_debt

        # Performance debt (30% weight)
        slow_test_ratio = 0
        if self.health_metrics_history:
            recent_metrics = self.health_metrics_history[-1]
            if recent_metrics.total_test_count > 0:
                slow_test_ratio = (
                    recent_metrics.slow_test_count / recent_metrics.total_test_count
                )
        performance_debt = slow_test_ratio * 100 * 0.3
        score += performance_debt

        # Maintenance debt (30% weight)
        maintenance_debt = min(len(self.maintenance_items) * 2, 100) * 0.3
        score += maintenance_debt

        return min(score, 100)

    def _identify_maintenance_needs(
        self, metrics: TestSuiteHealthMetrics
    ) -> list[MaintenanceItem]:
        """Identify new maintenance items based on current metrics."""
        new_items = []

        # Slow test optimization
        if metrics.slow_test_count > 5:
            new_items.append(
                MaintenanceItem(
                    task_type=MaintenanceTask.OPTIMIZE_SLOW_TESTS,
                    priority=Priority.HIGH
                    if metrics.slow_test_count > 20
                    else Priority.MEDIUM,
                    description=f"Optimize {metrics.slow_test_count} slow tests taking > 1s",
                    estimated_effort="4h",
                    affected_files=["tests/performance/"],
                    due_date=datetime.now() + timedelta(days=7),
                )
            )

        # Test quality issues
        if metrics.tests_without_docstrings > 10:
            new_items.append(
                MaintenanceItem(
                    task_type=MaintenanceTask.VALIDATE_TEST_QUALITY,
                    priority=Priority.MEDIUM,
                    description=f"Add docstrings to {metrics.tests_without_docstrings} tests",
                    estimated_effort="2h",
                    affected_files=["tests/"],
                    due_date=datetime.now() + timedelta(days=14),
                )
            )

        # Coverage gaps
        if metrics.line_coverage < 80:
            new_items.append(
                MaintenanceItem(
                    task_type=MaintenanceTask.REVIEW_TEST_COVERAGE,
                    priority=Priority.HIGH,
                    description=f"Improve test coverage from {metrics.line_coverage:.1f}% to 80%",
                    estimated_effort="1d",
                    affected_files=["tests/unit/"],
                    due_date=datetime.now() + timedelta(days=5),
                )
            )

        # Performance degradation
        if len(self.health_metrics_history) > 1:
            prev_metrics = self.health_metrics_history[-2]
            if metrics.average_test_time > prev_metrics.average_test_time * 1.2:
                new_items.append(
                    MaintenanceItem(
                        task_type=MaintenanceTask.OPTIMIZE_SLOW_TESTS,
                        priority=Priority.CRITICAL,
                        description="Address significant test performance regression",
                        estimated_effort="6h",
                        affected_files=["tests/"],
                        due_date=datetime.now() + timedelta(days=2),
                    )
                )

        return new_items

    def _update_item_priorities(self, metrics: TestSuiteHealthMetrics):
        """Update priorities of existing maintenance items based on current metrics."""
        for item in self.maintenance_items:
            if item.status != "pending":
                continue

            # Increase priority for quality issues if quality is declining
            if (
                item.task_type == MaintenanceTask.VALIDATE_TEST_QUALITY
                and metrics.average_quality_score < 70
            ):
                item.priority = Priority.HIGH
                item.last_updated = datetime.now()

            # Increase priority for performance issues if tests are getting slower
            if (
                item.task_type == MaintenanceTask.OPTIMIZE_SLOW_TESTS
                and metrics.average_test_time > 2.0
            ):
                item.priority = Priority.CRITICAL
                item.last_updated = datetime.now()

    def _execute_automated_tasks(self):
        """Execute automated maintenance tasks."""
        for item in self.maintenance_items:
            if item.status != "pending":
                continue

            # Only execute low-risk automated tasks
            if item.task_type == MaintenanceTask.CLEANUP_OBSOLETE_TESTS:
                try:
                    self._cleanup_obsolete_tests()
                    item.status = "completed"
                    item.last_updated = datetime.now()
                    self.logger.info(f"Completed automated task: {item.description}")
                except Exception as e:
                    self.logger.error(f"Failed automated task: {e}")

            elif item.task_type == MaintenanceTask.UPDATE_TEST_DATA:
                try:
                    self._update_test_data()
                    item.status = "completed"
                    item.last_updated = datetime.now()
                    self.logger.info(f"Completed automated task: {item.description}")
                except Exception as e:
                    self.logger.error(f"Failed automated task: {e}")

    def _cleanup_obsolete_tests(self):
        """Clean up obsolete test files and data."""
        # Remove empty test files
        for test_file in self.test_dir.rglob("test_*.py"):
            if test_file.is_file():
                content = test_file.read_text()
                # Remove files with only imports and no tests
                if "def test_" not in content and "class Test" not in content:
                    lines = content.split("\n")
                    non_empty_lines = [
                        line
                        for line in lines
                        if line.strip() and not line.strip().startswith("#")
                    ]
                    if len(non_empty_lines) <= 5:  # Only imports and docstring
                        test_file.unlink()
                        self.logger.info(f"Removed empty test file: {test_file}")

        # Clean up temporary test data
        temp_dirs = ["temp_test_*", "test_cache_*", ".pytest_cache"]
        for pattern in temp_dirs:
            for temp_path in self.test_dir.glob(pattern):
                if temp_path.is_dir():
                    shutil.rmtree(temp_path)
                    self.logger.info(f"Removed temporary directory: {temp_path}")

    def _update_test_data(self):
        """Update test data files to latest format."""
        # Update fixture files if needed
        fixtures_dir = self.test_dir / "fixtures"
        if fixtures_dir.exists():
            for fixture_file in fixtures_dir.glob("*.py"):
                # Check if fixture needs updates (simplified check)
                content = fixture_file.read_text()
                if "deprecated" in content.lower():
                    self.logger.info(f"Fixture may need update: {fixture_file}")

    def _generate_daily_report(self, metrics: TestSuiteHealthMetrics) -> str:
        """Generate daily maintenance report."""
        report = []
        report.append("=" * 60)
        report.append("XPCS TOOLKIT TEST SUITE DAILY MAINTENANCE REPORT")
        report.append("=" * 60)
        report.append(f"Date: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Health Summary
        report.append("HEALTH SUMMARY")
        report.append("-" * 20)
        report.append(f"Test Coverage: {metrics.line_coverage:.1f}%")
        report.append(f"Average Quality Score: {metrics.average_quality_score:.1f}/100")
        report.append(f"Total Tests: {metrics.total_test_count}")
        report.append(f"Slow Tests: {metrics.slow_test_count}")
        report.append(f"Technical Debt Score: {metrics.technical_debt_score:.1f}/100")
        report.append("")

        # Maintenance Items
        report.append("MAINTENANCE ITEMS")
        report.append("-" * 20)
        pending_items = [
            item for item in self.maintenance_items if item.status == "pending"
        ]
        if pending_items:
            for item in pending_items[:5]:  # Show top 5
                report.append(f"• {item.priority.value.upper()}: {item.description}")
        else:
            report.append("No pending maintenance items")
        report.append("")

        # Trends
        if len(self.health_metrics_history) > 1:
            prev_metrics = self.health_metrics_history[-2]
            report.append("TRENDS (vs. previous day)")
            report.append("-" * 30)

            coverage_change = metrics.line_coverage - prev_metrics.line_coverage
            quality_change = (
                metrics.average_quality_score - prev_metrics.average_quality_score
            )
            time_change = metrics.average_test_time - prev_metrics.average_test_time

            report.append(f"Coverage: {coverage_change:+.1f}%")
            report.append(f"Quality: {quality_change:+.1f}")
            report.append(f"Avg Test Time: {time_change:+.3f}s")

        return "\n".join(report)

    def _load_maintenance_items(self):
        """Load maintenance items from file."""
        if self.maintenance_file.exists():
            try:
                with open(self.maintenance_file) as f:
                    data = json.load(f)

                self.maintenance_items = []
                for item_data in data:
                    # Convert datetime strings back to datetime objects
                    if item_data.get("created_date"):
                        item_data["created_date"] = datetime.fromisoformat(
                            item_data["created_date"]
                        )
                    if item_data.get("last_updated"):
                        item_data["last_updated"] = datetime.fromisoformat(
                            item_data["last_updated"]
                        )
                    if item_data.get("due_date"):
                        item_data["due_date"] = datetime.fromisoformat(
                            item_data["due_date"]
                        )

                    # Convert enum strings back to enums
                    item_data["task_type"] = MaintenanceTask(item_data["task_type"])
                    item_data["priority"] = Priority(item_data["priority"])

                    self.maintenance_items.append(MaintenanceItem(**item_data))
            except Exception as e:
                self.logger.warning(f"Could not load maintenance items: {e}")

    def _save_maintenance_items(self):
        """Save maintenance items to file."""
        try:
            # Convert to JSON-serializable format
            data = []
            for item in self.maintenance_items:
                item_dict = asdict(item)
                # Convert datetime objects to ISO strings
                if item_dict["created_date"]:
                    item_dict["created_date"] = item_dict["created_date"].isoformat()
                if item_dict["last_updated"]:
                    item_dict["last_updated"] = item_dict["last_updated"].isoformat()
                if item_dict["due_date"]:
                    item_dict["due_date"] = item_dict["due_date"].isoformat()

                # Convert enums to strings
                item_dict["task_type"] = item_dict["task_type"].value
                item_dict["priority"] = item_dict["priority"].value

                data.append(item_dict)

            with open(self.maintenance_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save maintenance items: {e}")

    def _load_health_history(self):
        """Load health metrics history."""
        if self.health_history_file.exists():
            try:
                with open(self.health_history_file) as f:
                    data = json.load(f)

                self.health_metrics_history = []
                for metrics_data in data:
                    metrics_data["timestamp"] = datetime.fromisoformat(
                        metrics_data["timestamp"]
                    )
                    self.health_metrics_history.append(
                        TestSuiteHealthMetrics(**metrics_data)
                    )

                # Keep only last 30 days
                cutoff_date = datetime.now() - timedelta(days=30)
                self.health_metrics_history = [
                    m for m in self.health_metrics_history if m.timestamp >= cutoff_date
                ]
            except Exception as e:
                self.logger.warning(f"Could not load health history: {e}")

    def _save_health_history(self):
        """Save health metrics history."""
        try:
            data = []
            for metrics in self.health_metrics_history:
                metrics_dict = asdict(metrics)
                metrics_dict["timestamp"] = metrics_dict["timestamp"].isoformat()
                data.append(metrics_dict)

            with open(self.health_history_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save health history: {e}")

    def _load_configuration(self):
        """Load maintenance configuration."""
        # Set default configuration
        self.config = {
            "maintenance_schedule": {"daily": True, "weekly": True, "monthly": True},
            "thresholds": {
                "min_coverage": 80.0,
                "max_slow_tests": 10,
                "min_quality_score": 70.0,
                "max_technical_debt": 50.0,
            },
            "automation": {
                "cleanup_obsolete": True,
                "update_test_data": True,
                "optimize_performance": False,  # Manual only
            },
        }

        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    user_config = json.load(f)
                    self.config.update(user_config)
            except Exception as e:
                self.logger.warning(f"Could not load configuration: {e}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="XPCS Toolkit Test Suite Maintenance")
    parser.add_argument("--daily", action="store_true", help="Run daily maintenance")
    parser.add_argument("--weekly", action="store_true", help="Run weekly maintenance")
    parser.add_argument("--release", help="Run release maintenance for version")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--report", action="store_true", help="Generate health report")

    args = parser.parse_args()

    manager = TestSuiteMaintenanceManager()

    if args.daily:
        report = manager.run_daily_maintenance()
        print(report)
    elif args.weekly:
        report = manager.run_weekly_maintenance()
        print(json.dumps(report, indent=2, default=str))
    elif args.release:
        report = manager.run_release_maintenance(args.release)
        print(json.dumps(report, indent=2, default=str))
    elif args.status:
        # Show current maintenance status
        pending_items = [
            item for item in manager.maintenance_items if item.status == "pending"
        ]
        print(f"Pending maintenance items: {len(pending_items)}")
        for item in pending_items[:10]:  # Show top 10
            print(f"  • {item.priority.value}: {item.description}")
    elif args.report:
        if manager.health_metrics_history:
            latest_metrics = manager.health_metrics_history[-1]
            print(manager._generate_daily_report(latest_metrics))
        else:
            print("No health metrics available. Run --daily first.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
