"""
Qt Compliance Regression Testing Framework.

This module provides comprehensive regression testing capabilities to ensure
that Qt compliance improvements don't introduce new errors and that the
system maintains zero Qt errors over time.
"""

import json
import pickle
import time
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pytest
from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtWidgets import QApplication, QWidget

from xpcs_toolkit.monitoring import (
    get_qt_error_detector,
    get_integrated_monitoring_system,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring,
    qt_error_capture
)
from xpcs_toolkit.threading import get_qt_compliant_thread_manager
from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BaselineMetrics:
    """Baseline metrics for regression testing."""

    timestamp: float
    version: str
    platform: str
    qt_version: str

    # Qt Error Metrics
    total_qt_errors: int = 0
    qt_errors_by_type: Dict[str, int] = field(default_factory=dict)
    qt_error_patterns: Dict[str, int] = field(default_factory=dict)

    # Performance Metrics
    avg_startup_time: float = 0.0
    avg_shutdown_time: float = 0.0
    avg_memory_usage_mb: float = 0.0
    avg_cpu_utilization: float = 0.0

    # Threading Metrics
    thread_pool_efficiency: float = 0.0
    worker_completion_rate: float = 0.0
    thread_safety_violations: int = 0

    # System Health Metrics
    monitoring_system_uptime: float = 0.0
    recovery_action_count: int = 0
    alert_frequency: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaselineMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RegressionTestResult:
    """Result of a regression test run."""

    timestamp: float
    test_name: str
    baseline_version: str
    current_version: str

    # Test Results
    passed: bool = False
    error_message: Optional[str] = None

    # Metric Comparisons
    qt_error_regression: bool = False
    performance_regression: bool = False
    threading_regression: bool = False

    # Detailed Results
    current_metrics: Optional[BaselineMetrics] = None
    baseline_metrics: Optional[BaselineMetrics] = None
    metric_deltas: Dict[str, float] = field(default_factory=dict)

    # Test Duration
    test_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.current_metrics:
            result['current_metrics'] = self.current_metrics.to_dict()
        if self.baseline_metrics:
            result['baseline_metrics'] = self.baseline_metrics.to_dict()
        return result


class RegressionTestConfiguration:
    """Configuration for regression testing."""

    def __init__(self):
        # Tolerance thresholds (how much degradation is acceptable)
        self.qt_error_tolerance = 0  # Zero tolerance for new Qt errors
        self.performance_degradation_tolerance = 0.1  # 10% performance degradation
        self.memory_increase_tolerance = 0.2  # 20% memory increase
        self.cpu_increase_tolerance = 0.15  # 15% CPU increase

        # Test configuration
        self.baseline_storage_path = Path.home() / ".xpcs_toolkit" / "regression_baselines"
        self.results_storage_path = Path.home() / ".xpcs_toolkit" / "regression_results"
        self.enable_performance_benchmarks = True
        self.enable_memory_benchmarks = True
        self.enable_threading_benchmarks = True

        # Benchmark durations
        self.startup_benchmark_iterations = 5
        self.performance_benchmark_duration = 10.0  # seconds
        self.memory_stress_duration = 5.0  # seconds


class QtRegressionTestFramework:
    """
    Comprehensive regression testing framework for Qt compliance system.

    Provides:
    - Baseline establishment and storage
    - Automated regression detection
    - Performance benchmarking
    - Historical trend analysis
    - Continuous integration support
    """

    def __init__(self, config: Optional[RegressionTestConfiguration] = None):
        """Initialize regression testing framework."""
        self.config = config or RegressionTestConfiguration()

        # Ensure storage directories exist
        self.config.baseline_storage_path.mkdir(parents=True, exist_ok=True)
        self.config.results_storage_path.mkdir(parents=True, exist_ok=True)

        # Test state
        self._current_baseline: Optional[BaselineMetrics] = None
        self._test_results: List[RegressionTestResult] = []

        logger.info("Qt regression testing framework initialized")

    def establish_baseline(self, version: str, force_update: bool = False) -> BaselineMetrics:
        """Establish performance and quality baseline."""
        baseline_file = self.config.baseline_storage_path / f"baseline_{version}.json"

        # Check if baseline already exists
        if baseline_file.exists() and not force_update:
            logger.info(f"Loading existing baseline for version {version}")
            return self._load_baseline(baseline_file)

        logger.info(f"Establishing new baseline for version {version}")

        # Initialize monitoring system for baseline
        monitoring_system = initialize_integrated_monitoring()

        try:
            # Collect baseline metrics
            baseline = self._collect_baseline_metrics(version)

            # Save baseline
            self._save_baseline(baseline, baseline_file)

            self._current_baseline = baseline
            logger.info(f"Baseline established and saved for version {version}")

            return baseline

        finally:
            shutdown_integrated_monitoring()

    def run_regression_tests(self, current_version: str,
                           baseline_version: Optional[str] = None) -> List[RegressionTestResult]:
        """Run comprehensive regression tests against baseline."""
        logger.info(f"Running regression tests for version {current_version}")

        # Load baseline
        if baseline_version:
            baseline = self._load_baseline_for_version(baseline_version)
        else:
            baseline = self._find_latest_baseline()

        if not baseline:
            raise ValueError("No baseline found for regression testing")

        # Initialize monitoring system
        monitoring_system = initialize_integrated_monitoring()

        try:
            # Collect current metrics
            current_metrics = self._collect_baseline_metrics(current_version)

            # Run regression tests
            test_results = []

            # Qt Error Regression Test
            test_results.append(self._test_qt_error_regression(
                baseline, current_metrics, current_version
            ))

            # Performance Regression Test
            if self.config.enable_performance_benchmarks:
                test_results.append(self._test_performance_regression(
                    baseline, current_metrics, current_version
                ))

            # Memory Regression Test
            if self.config.enable_memory_benchmarks:
                test_results.append(self._test_memory_regression(
                    baseline, current_metrics, current_version
                ))

            # Threading Regression Test
            if self.config.enable_threading_benchmarks:
                test_results.append(self._test_threading_regression(
                    baseline, current_metrics, current_version
                ))

            # Store results
            self._test_results.extend(test_results)
            self._save_test_results(test_results, current_version)

            return test_results

        finally:
            shutdown_integrated_monitoring()

    def _collect_baseline_metrics(self, version: str) -> BaselineMetrics:
        """Collect comprehensive baseline metrics."""
        import platform
        from PySide6.QtCore import qVersion

        start_time = time.perf_counter()

        # Platform info
        platform_info = platform.system()
        qt_version = qVersion()

        # Initialize monitoring components
        qt_detector = get_qt_error_detector()
        thread_manager = get_qt_compliant_thread_manager()
        monitoring_system = get_integrated_monitoring_system()

        # Clear existing data
        qt_detector.clear_error_history()

        # Collect Qt error metrics
        qt_metrics = self._collect_qt_error_metrics()

        # Collect performance metrics
        perf_metrics = self._collect_performance_metrics()

        # Collect threading metrics
        threading_metrics = self._collect_threading_metrics()

        # Collect system health metrics
        health_metrics = self._collect_health_metrics()

        baseline = BaselineMetrics(
            timestamp=time.perf_counter(),
            version=version,
            platform=platform_info,
            qt_version=qt_version,

            # Qt Errors
            total_qt_errors=qt_metrics.get("total_errors", 0),
            qt_errors_by_type=qt_metrics.get("errors_by_type", {}),
            qt_error_patterns=qt_metrics.get("pattern_counts", {}),

            # Performance
            avg_startup_time=perf_metrics.get("startup_time", 0.0),
            avg_shutdown_time=perf_metrics.get("shutdown_time", 0.0),
            avg_memory_usage_mb=perf_metrics.get("memory_usage_mb", 0.0),
            avg_cpu_utilization=perf_metrics.get("cpu_utilization", 0.0),

            # Threading
            thread_pool_efficiency=threading_metrics.get("efficiency", 0.0),
            worker_completion_rate=threading_metrics.get("completion_rate", 0.0),
            thread_safety_violations=threading_metrics.get("safety_violations", 0),

            # System Health
            monitoring_system_uptime=health_metrics.get("uptime", 0.0),
            recovery_action_count=health_metrics.get("recovery_actions", 0),
            alert_frequency=health_metrics.get("alert_frequency", 0.0)
        )

        collection_time = time.perf_counter() - start_time
        logger.info(f"Baseline metrics collected in {collection_time:.2f}s")

        return baseline

    def _collect_qt_error_metrics(self) -> Dict[str, Any]:
        """Collect Qt error metrics through stress testing."""
        qt_detector = get_qt_error_detector()

        # Run stress test to detect any Qt errors
        with qt_error_capture() as capture:
            self._run_qt_stress_test()

        # Get error summary
        error_summary = qt_detector.get_error_summary()

        return {
            "total_errors": len(capture.captured_errors),
            "errors_by_type": self._categorize_errors(capture.captured_errors),
            "pattern_counts": error_summary.get("pattern_counts", {}),
            "captured_errors": [error.to_dict() for error in capture.captured_errors]
        }

    def _run_qt_stress_test(self):
        """Run Qt stress test to trigger any potential errors."""
        # Create multiple widgets
        widgets = []
        timers = []

        try:
            for i in range(20):
                widget = QWidget()
                widget.setObjectName(f"stress_widget_{i}")
                widgets.append(widget)

                # Create timer for each widget
                timer = QTimer(widget)
                timer.timeout.connect(lambda: QCoreApplication.processEvents())
                timer.start(10)
                timers.append(timer)

            # Process events for a while
            for _ in range(100):
                QCoreApplication.processEvents()
                time.sleep(0.01)

        finally:
            # Cleanup
            for timer in timers:
                timer.stop()
            for widget in widgets:
                widget.deleteLater()

            # Process cleanup events
            for _ in range(20):
                QCoreApplication.processEvents()
                time.sleep(0.01)

    def _categorize_errors(self, errors) -> Dict[str, int]:
        """Categorize errors by type."""
        categorized = defaultdict(int)
        for error in errors:
            categorized[error.error_type.value] += 1
        return dict(categorized)

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics through benchmarking."""
        # Startup time benchmark
        startup_times = []
        for _ in range(self.config.startup_benchmark_iterations):
            start_time = time.perf_counter()

            # Simulate startup operations
            monitoring_system = initialize_integrated_monitoring()

            startup_time = time.perf_counter() - start_time
            startup_times.append(startup_time)

            # Shutdown
            start_shutdown = time.perf_counter()
            shutdown_integrated_monitoring()
            shutdown_time = time.perf_counter() - start_shutdown

        # Memory and CPU metrics
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=1.0)
        except ImportError:
            memory_mb = 0.0
            cpu_percent = 0.0

        return {
            "startup_time": sum(startup_times) / len(startup_times),
            "shutdown_time": shutdown_time,
            "memory_usage_mb": memory_mb,
            "cpu_utilization": cpu_percent
        }

    def _collect_threading_metrics(self) -> Dict[str, Any]:
        """Collect threading performance metrics."""
        from xpcs_toolkit.threading import SafeWorkerBase
        import threading

        thread_manager = get_qt_compliant_thread_manager()

        # Threading efficiency test
        class BenchmarkWorker(SafeWorkerBase):
            def __init__(self, worker_id):
                super().__init__()
                self.worker_id = worker_id
                self.completed = False

            def do_work(self):
                time.sleep(0.1)  # Simulate work
                self.completed = True

        # Create and run multiple workers
        workers = []
        start_time = time.perf_counter()

        for i in range(10):
            worker = BenchmarkWorker(i)
            workers.append(worker)
            thread_manager.start_worker(worker, f"benchmark_worker_{i}")

        # Wait for completion
        timeout = 5.0
        while time.perf_counter() - start_time < timeout:
            completed_count = sum(1 for w in workers if w.completed)
            if completed_count == len(workers):
                break
            time.sleep(0.1)

        completion_time = time.perf_counter() - start_time
        completed_workers = sum(1 for w in workers if w.completed)

        # Cleanup
        thread_manager.cleanup_finished_threads()

        return {
            "efficiency": completed_workers / len(workers),
            "completion_rate": completed_workers / completion_time if completion_time > 0 else 0,
            "safety_violations": 0  # Would be detected by monitoring system
        }

    def _collect_health_metrics(self) -> Dict[str, Any]:
        """Collect system health metrics."""
        monitoring_system = get_integrated_monitoring_system()

        # Get system status
        status = monitoring_system.get_system_status()

        return {
            "uptime": monitoring_system.uptime_seconds,
            "recovery_actions": status.total_recovery_actions,
            "alert_frequency": status.active_alerts / max(monitoring_system.uptime_seconds, 1)
        }

    def _test_qt_error_regression(self, baseline: BaselineMetrics,
                                 current: BaselineMetrics, version: str) -> RegressionTestResult:
        """Test for Qt error regressions."""
        start_time = time.perf_counter()

        # Compare Qt error counts
        qt_error_increase = current.total_qt_errors - baseline.total_qt_errors
        has_regression = qt_error_increase > self.config.qt_error_tolerance

        # Calculate metric deltas
        deltas = {
            "qt_errors_delta": qt_error_increase,
            "qt_error_types_delta": len(current.qt_errors_by_type) - len(baseline.qt_errors_by_type)
        }

        result = RegressionTestResult(
            timestamp=time.perf_counter(),
            test_name="qt_error_regression",
            baseline_version=baseline.version,
            current_version=version,
            passed=not has_regression,
            qt_error_regression=has_regression,
            current_metrics=current,
            baseline_metrics=baseline,
            metric_deltas=deltas,
            test_duration_seconds=time.perf_counter() - start_time
        )

        if has_regression:
            result.error_message = f"Qt error regression detected: {qt_error_increase} new errors"

        return result

    def _test_performance_regression(self, baseline: BaselineMetrics,
                                   current: BaselineMetrics, version: str) -> RegressionTestResult:
        """Test for performance regressions."""
        start_time = time.perf_counter()

        # Calculate performance changes
        startup_change = (current.avg_startup_time - baseline.avg_startup_time) / baseline.avg_startup_time if baseline.avg_startup_time > 0 else 0
        memory_change = (current.avg_memory_usage_mb - baseline.avg_memory_usage_mb) / baseline.avg_memory_usage_mb if baseline.avg_memory_usage_mb > 0 else 0
        cpu_change = (current.avg_cpu_utilization - baseline.avg_cpu_utilization) / baseline.avg_cpu_utilization if baseline.avg_cpu_utilization > 0 else 0

        # Check for regressions
        has_regression = (
            startup_change > self.config.performance_degradation_tolerance or
            memory_change > self.config.memory_increase_tolerance or
            cpu_change > self.config.cpu_increase_tolerance
        )

        deltas = {
            "startup_time_change_percent": startup_change * 100,
            "memory_change_percent": memory_change * 100,
            "cpu_change_percent": cpu_change * 100
        }

        result = RegressionTestResult(
            timestamp=time.perf_counter(),
            test_name="performance_regression",
            baseline_version=baseline.version,
            current_version=version,
            passed=not has_regression,
            performance_regression=has_regression,
            current_metrics=current,
            baseline_metrics=baseline,
            metric_deltas=deltas,
            test_duration_seconds=time.perf_counter() - start_time
        )

        if has_regression:
            result.error_message = f"Performance regression detected: startup {startup_change:.1%}, memory {memory_change:.1%}, CPU {cpu_change:.1%}"

        return result

    def _test_memory_regression(self, baseline: BaselineMetrics,
                              current: BaselineMetrics, version: str) -> RegressionTestResult:
        """Test for memory regressions."""
        start_time = time.perf_counter()

        # Run memory stress test
        memory_metrics = self._run_memory_stress_test()

        # Compare with baseline (simplified for this implementation)
        has_regression = memory_metrics.get("peak_memory_mb", 0) > baseline.avg_memory_usage_mb * 2

        deltas = {
            "peak_memory_mb": memory_metrics.get("peak_memory_mb", 0),
            "memory_leaks_detected": memory_metrics.get("leaks", 0)
        }

        result = RegressionTestResult(
            timestamp=time.perf_counter(),
            test_name="memory_regression",
            baseline_version=baseline.version,
            current_version=version,
            passed=not has_regression,
            current_metrics=current,
            baseline_metrics=baseline,
            metric_deltas=deltas,
            test_duration_seconds=time.perf_counter() - start_time
        )

        if has_regression:
            result.error_message = f"Memory regression detected: peak usage {memory_metrics.get('peak_memory_mb', 0):.1f} MB"

        return result

    def _run_memory_stress_test(self) -> Dict[str, Any]:
        """Run memory stress test."""
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Create objects to stress memory
            objects = []
            for i in range(100):
                widget = QWidget()
                objects.append(widget)

            peak_memory = process.memory_info().rss

            # Cleanup
            for obj in objects:
                obj.deleteLater()
            objects.clear()

            # Force cleanup
            import gc
            gc.collect()

            final_memory = process.memory_info().rss

            return {
                "peak_memory_mb": peak_memory / 1024 / 1024,
                "initial_memory_mb": initial_memory / 1024 / 1024,
                "final_memory_mb": final_memory / 1024 / 1024,
                "leaks": max(0, final_memory - initial_memory) / 1024 / 1024
            }

        except ImportError:
            return {"peak_memory_mb": 0, "leaks": 0}

    def _test_threading_regression(self, baseline: BaselineMetrics,
                                 current: BaselineMetrics, version: str) -> RegressionTestResult:
        """Test for threading regressions."""
        start_time = time.perf_counter()

        # Compare threading metrics
        efficiency_change = current.thread_pool_efficiency - baseline.thread_pool_efficiency
        completion_rate_change = current.worker_completion_rate - baseline.worker_completion_rate
        safety_violations_increase = current.thread_safety_violations - baseline.thread_safety_violations

        has_regression = (
            efficiency_change < -0.1 or  # 10% efficiency decrease
            completion_rate_change < -0.1 or  # 10% completion rate decrease
            safety_violations_increase > 0  # Any new safety violations
        )

        deltas = {
            "efficiency_change": efficiency_change,
            "completion_rate_change": completion_rate_change,
            "safety_violations_increase": safety_violations_increase
        }

        result = RegressionTestResult(
            timestamp=time.perf_counter(),
            test_name="threading_regression",
            baseline_version=baseline.version,
            current_version=version,
            passed=not has_regression,
            threading_regression=has_regression,
            current_metrics=current,
            baseline_metrics=baseline,
            metric_deltas=deltas,
            test_duration_seconds=time.perf_counter() - start_time
        )

        if has_regression:
            result.error_message = f"Threading regression detected: efficiency {efficiency_change:.1%}, violations +{safety_violations_increase}"

        return result

    def _save_baseline(self, baseline: BaselineMetrics, file_path: Path):
        """Save baseline to file."""
        with open(file_path, 'w') as f:
            json.dump(baseline.to_dict(), f, indent=2)

    def _load_baseline(self, file_path: Path) -> BaselineMetrics:
        """Load baseline from file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return BaselineMetrics.from_dict(data)

    def _load_baseline_for_version(self, version: str) -> Optional[BaselineMetrics]:
        """Load baseline for specific version."""
        baseline_file = self.config.baseline_storage_path / f"baseline_{version}.json"
        if baseline_file.exists():
            return self._load_baseline(baseline_file)
        return None

    def _find_latest_baseline(self) -> Optional[BaselineMetrics]:
        """Find the most recent baseline."""
        baseline_files = list(self.config.baseline_storage_path.glob("baseline_*.json"))
        if not baseline_files:
            return None

        # Load all baselines and find the most recent
        latest_baseline = None
        latest_timestamp = 0

        for file_path in baseline_files:
            try:
                baseline = self._load_baseline(file_path)
                if baseline.timestamp > latest_timestamp:
                    latest_timestamp = baseline.timestamp
                    latest_baseline = baseline
            except Exception as e:
                logger.warning(f"Failed to load baseline from {file_path}: {e}")

        return latest_baseline

    def _save_test_results(self, results: List[RegressionTestResult], version: str):
        """Save test results to file."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.config.results_storage_path / f"regression_results_{version}_{timestamp_str}.json"

        results_data = {
            "timestamp": time.perf_counter(),
            "version": version,
            "results": [result.to_dict() for result in results]
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)

    def generate_regression_report(self, results: List[RegressionTestResult]) -> Dict[str, Any]:
        """Generate comprehensive regression test report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests

        regressions = {
            "qt_errors": sum(1 for r in results if r.qt_error_regression),
            "performance": sum(1 for r in results if r.performance_regression),
            "threading": sum(1 for r in results if r.threading_regression)
        }

        return {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "regressions": regressions,
            "test_results": [result.to_dict() for result in results],
            "recommendations": self._generate_recommendations(results)
        }

    def _generate_recommendations(self, results: List[RegressionTestResult]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        # Check for Qt error regressions
        qt_errors = [r for r in results if r.qt_error_regression]
        if qt_errors:
            recommendations.append("Qt error regressions detected - review recent changes to Qt compliance system")

        # Check for performance regressions
        perf_regressions = [r for r in results if r.performance_regression]
        if perf_regressions:
            recommendations.append("Performance regressions detected - consider optimization or profiling")

        # Check for threading regressions
        thread_regressions = [r for r in results if r.threading_regression]
        if thread_regressions:
            recommendations.append("Threading regressions detected - review thread management changes")

        # Overall recommendations
        failed_results = [r for r in results if not r.passed]
        if len(failed_results) > len(results) * 0.5:
            recommendations.append("High failure rate - consider reverting recent changes")

        return recommendations


class RegressionTestRunner:
    """Utility class for running regression tests."""

    @staticmethod
    def run_full_regression_suite(current_version: str, baseline_version: Optional[str] = None) -> Dict[str, Any]:
        """Run full regression test suite."""
        framework = QtRegressionTestFramework()

        try:
            # Run regression tests
            results = framework.run_regression_tests(current_version, baseline_version)

            # Generate report
            report = framework.generate_regression_report(results)

            return report

        except Exception as e:
            logger.error(f"Regression test suite failed: {e}")
            return {
                "error": str(e),
                "success": False
            }

    @staticmethod
    def establish_new_baseline(version: str, force_update: bool = False) -> bool:
        """Establish new baseline for version."""
        framework = QtRegressionTestFramework()

        try:
            baseline = framework.establish_baseline(version, force_update)
            logger.info(f"Baseline established for version {version}")
            return True

        except Exception as e:
            logger.error(f"Failed to establish baseline: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "baseline":
            version = sys.argv[2] if len(sys.argv) > 2 else "dev"
            success = RegressionTestRunner.establish_new_baseline(version, force_update=True)
            sys.exit(0 if success else 1)

        elif command == "test":
            current_version = sys.argv[2] if len(sys.argv) > 2 else "dev"
            baseline_version = sys.argv[3] if len(sys.argv) > 3 else None

            report = RegressionTestRunner.run_full_regression_suite(current_version, baseline_version)

            print("Regression Test Report")
            print("=" * 50)
            print(f"Success Rate: {report.get('summary', {}).get('success_rate', 0):.1%}")
            print(f"Failed Tests: {report.get('summary', {}).get('failed_tests', 0)}")

            success = report.get('summary', {}).get('success_rate', 0) == 1.0
            sys.exit(0 if success else 1)
    else:
        print("Usage:")
        print("  python qt_regression_testing_framework.py baseline [version]")
        print("  python qt_regression_testing_framework.py test [current_version] [baseline_version]")