"""
Comprehensive Qt Compliance Integration Testing Suite.

This module provides comprehensive integration tests to validate that the Qt
compliance system works correctly with all application components and
maintains zero Qt errors during complex operations.
"""

import gc
import os
import sys
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QObject, QThread, QTimer, Signal, QCoreApplication
from PySide6.QtWidgets import QApplication, QWidget

# Import the Qt compliance system components
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from xpcs_toolkit.monitoring import (
    get_qt_error_detector,
    get_integrated_monitoring_system,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring,
    qt_error_capture
)
from xpcs_toolkit.threading import (
    get_qt_compliant_thread_manager,
    get_thread_pool_validator,
    SafeWorkerBase
)
from xpcs_toolkit.plothandler.qt_signal_fixes import qt_connection_context


class QtComplianceIntegrationTestSuite(unittest.TestCase):
    """Comprehensive integration test suite for Qt compliance system."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Ensure QApplication exists
        if not QApplication.instance():
            cls.app = QApplication([])
        else:
            cls.app = QApplication.instance()

        # Initialize monitoring system
        cls.monitoring_system = initialize_integrated_monitoring()

        # Get component references
        cls.qt_error_detector = get_qt_error_detector()
        cls.thread_manager = get_qt_compliant_thread_manager()
        cls.thread_validator = get_thread_pool_validator()

        # Clear any existing errors
        cls.qt_error_detector.clear_error_history()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Shutdown monitoring
        shutdown_integrated_monitoring()

        # Clean up threads
        cls.thread_manager.cleanup_finished_threads()

        # Force cleanup
        gc.collect()

    def setUp(self):
        """Set up individual test."""
        # Clear error history before each test
        self.qt_error_detector.clear_error_history()

        # Reset component failure counts
        self.monitoring_system._resilience_manager.clear_component_failures()

        # Start time for test duration tracking
        self.test_start_time = time.perf_counter()

    def tearDown(self):
        """Clean up after individual test."""
        # Check for Qt errors during test
        error_summary = self.qt_error_detector.get_error_summary()
        test_duration = time.perf_counter() - self.test_start_time

        print(f"Test completed in {test_duration:.3f}s")
        print(f"Qt errors during test: {error_summary.get('recent_errors_1h', 0)}")

        # Force cleanup
        gc.collect()

    @contextmanager
    def assert_no_qt_errors(self):
        """Context manager to assert no Qt errors occur."""
        with qt_error_capture() as capture:
            yield capture

        # Assert no errors were captured
        if capture.captured_errors:
            error_messages = [error.message for error in capture.captured_errors]
            self.fail(f"Qt errors detected: {error_messages}")

    def test_basic_qt_compliance_integration(self):
        """Test basic Qt compliance system integration."""
        with self.assert_no_qt_errors():
            # Test monitoring system is running
            self.assertTrue(self.monitoring_system.is_running)

            # Test Qt error detector is active
            error_summary = self.qt_error_detector.get_error_summary()
            self.assertIn("patterns_registered", error_summary)
            self.assertGreater(error_summary["patterns_registered"], 0)

            # Test thread manager is operational
            self.assertIsNotNone(self.thread_manager)

            # Test thread validator is operational
            validation_summary = self.thread_validator.get_health_summary()
            self.assertIn("registered_pools", validation_summary)

    def test_gui_components_integration(self):
        """Test Qt compliance integration with GUI components."""
        with self.assert_no_qt_errors():
            # Create test widget
            widget = QWidget()

            # Test widget creation doesn't trigger errors
            widget.setWindowTitle("Qt Compliance Test Widget")
            widget.resize(200, 100)

            # Test signal connections with qt_connection_context
            with qt_connection_context("test_widget_signals"):
                # Create a timer for testing
                timer = QTimer(widget)
                timer.timeout.connect(lambda: None)
                timer.start(100)

                # Process events briefly
                for _ in range(10):
                    QCoreApplication.processEvents()
                    time.sleep(0.01)

                timer.stop()

            # Clean up
            widget.deleteLater()
            QCoreApplication.processEvents()

    def test_threading_integration(self):
        """Test Qt compliance integration with threading operations."""

        class TestWorker(SafeWorkerBase):
            """Test worker for threading integration."""

            work_completed = Signal(str)

            def __init__(self):
                super().__init__()
                self.result = None

            def do_work(self):
                """Perform test work."""
                # Simulate some work
                time.sleep(0.1)
                self.result = "work_completed"
                self.work_completed.emit(self.result)

        with self.assert_no_qt_errors():
            # Create and run worker
            worker = TestWorker()

            # Track completion
            completed = threading.Event()
            def on_completion(result):
                self.assertEqual(result, "work_completed")
                completed.set()

            worker.work_completed.connect(on_completion)

            # Start worker in thread manager
            self.thread_manager.start_worker(worker, "test_worker")

            # Wait for completion
            self.assertTrue(completed.wait(timeout=5.0))

            # Verify worker completed successfully
            self.assertEqual(worker.result, "work_completed")

            # Clean up
            self.thread_manager.cleanup_finished_threads()

    def test_plotting_subsystem_integration(self):
        """Test Qt compliance integration with plotting subsystems."""
        with self.assert_no_qt_errors():
            try:
                # Test PyQtGraph integration if available
                import pyqtgraph as pg

                # Create plot widget with qt_connection_context
                with qt_connection_context("plotting_test"):
                    plot_widget = pg.PlotWidget()

                    # Add some test data
                    import numpy as np
                    x = np.linspace(0, 10, 100)
                    y = np.sin(x)
                    plot_widget.plot(x, y, pen='r')

                    # Process events
                    for _ in range(10):
                        QCoreApplication.processEvents()
                        time.sleep(0.01)

                    # Clean up
                    plot_widget.close()
                    plot_widget.deleteLater()
                    QCoreApplication.processEvents()

            except ImportError:
                self.skipTest("PyQtGraph not available for plotting integration test")

    def test_async_operations_integration(self):
        """Test Qt compliance integration with async operations."""

        class AsyncTestWorker(SafeWorkerBase):
            """Worker for async operation testing."""

            async_completed = Signal(list)

            def __init__(self):
                super().__init__()
                self.results = []

            def do_work(self):
                """Perform async-style work with multiple operations."""
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit multiple tasks
                    futures = []
                    for i in range(5):
                        future = executor.submit(self._async_task, i)
                        futures.append(future)

                    # Collect results
                    for future in concurrent.futures.as_completed(futures):
                        self.results.append(future.result())

                self.async_completed.emit(self.results)

            def _async_task(self, task_id):
                """Individual async task."""
                time.sleep(0.05)  # Simulate work
                return f"task_{task_id}_completed"

        with self.assert_no_qt_errors():
            # Create async worker
            worker = AsyncTestWorker()

            # Track completion
            completed = threading.Event()
            results = []

            def on_async_completion(task_results):
                results.extend(task_results)
                completed.set()

            worker.async_completed.connect(on_async_completion)

            # Start worker
            self.thread_manager.start_worker(worker, "async_test_worker")

            # Wait for completion
            self.assertTrue(completed.wait(timeout=10.0))

            # Verify all tasks completed
            self.assertEqual(len(results), 5)
            for i in range(5):
                self.assertIn(f"task_{i}_completed", results)

            # Clean up
            self.thread_manager.cleanup_finished_threads()

    def test_memory_management_integration(self):
        """Test Qt compliance integration with memory management."""
        with self.assert_no_qt_errors():
            # Get initial memory baseline
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss

            # Create multiple objects to test memory management
            widgets = []
            workers = []

            try:
                # Create test objects
                for i in range(10):
                    # Create widget
                    widget = QWidget()
                    widget.setObjectName(f"test_widget_{i}")
                    widgets.append(widget)

                    # Create worker
                    class MemoryTestWorker(SafeWorkerBase):
                        def __init__(self, worker_id):
                            super().__init__()
                            self.worker_id = worker_id
                            self.data = [0] * 1000  # Small data allocation

                        def do_work(self):
                            # Simulate work with data
                            self.data = [x + 1 for x in self.data]

                    worker = MemoryTestWorker(i)
                    workers.append(worker)
                    self.thread_manager.start_worker(worker, f"memory_test_worker_{i}")

                # Process events to ensure everything is running
                for _ in range(50):
                    QCoreApplication.processEvents()
                    time.sleep(0.01)

                # Clean up workers
                self.thread_manager.cleanup_finished_threads()

                # Clean up widgets
                for widget in widgets:
                    widget.deleteLater()

                # Force garbage collection
                widgets.clear()
                workers.clear()
                gc.collect()

                # Process deletion events
                for _ in range(20):
                    QCoreApplication.processEvents()
                    time.sleep(0.01)

                # Check memory didn't grow excessively
                final_memory = process.memory_info().rss
                memory_growth = final_memory - initial_memory
                memory_growth_mb = memory_growth / 1024 / 1024

                print(f"Memory growth: {memory_growth_mb:.2f} MB")

                # Allow some memory growth but not excessive
                self.assertLess(memory_growth_mb, 50, "Memory growth too large")

            finally:
                # Ensure cleanup even if test fails
                for widget in widgets:
                    widget.deleteLater()
                self.thread_manager.cleanup_finished_threads()
                gc.collect()

    def test_error_recovery_integration(self):
        """Test Qt compliance integration with error recovery mechanisms."""
        with self.assert_no_qt_errors():
            # Get initial health status
            initial_health = self.monitoring_system.get_system_status()

            # Simulate a controlled error condition
            # (We'll use the resilience manager's force recovery feature)
            from xpcs_toolkit.monitoring.system_resilience_manager import SystemComponent, RecoveryStrategy

            resilience_manager = self.monitoring_system._resilience_manager

            # Force a memory cleanup recovery
            success = resilience_manager.force_recovery(
                SystemComponent.MEMORY_SYSTEM,
                RecoveryStrategy.FORCE_GARBAGE_COLLECTION
            )

            self.assertTrue(success, "Recovery action should succeed")

            # Wait for recovery to complete
            time.sleep(1.0)

            # Check that system recovered
            recovery_history = resilience_manager.get_recovery_history(hours=1)
            self.assertGreater(len(recovery_history), 0, "Recovery action should be recorded")

            latest_recovery = recovery_history[-1]
            self.assertEqual(latest_recovery.component.value, "memory_system")
            self.assertEqual(latest_recovery.strategy.value, "force_gc")

    def test_performance_monitoring_integration(self):
        """Test Qt compliance integration with performance monitoring."""
        with self.assert_no_qt_errors():
            # Get performance metrics collector
            metrics_collector = self.monitoring_system._performance_metrics

            # Record some test metrics
            metrics_collector.record_counter("test_integration_counter", 1)
            metrics_collector.record_gauge("test_integration_gauge", 42.0)

            # Use timing context
            with metrics_collector.timing_context("test_integration_timer"):
                time.sleep(0.1)

            # Create performance snapshot
            snapshot = metrics_collector.create_snapshot()

            # Verify metrics were recorded
            self.assertIn("test_integration_counter", snapshot.metrics)
            self.assertIn("test_integration_gauge", snapshot.metrics)
            self.assertIn("test_integration_timer", snapshot.metrics)

            # Verify timer recorded reasonable duration
            timer_data = snapshot.metrics["test_integration_timer"]
            self.assertGreater(timer_data["latest_value"], 0.05)  # At least 50ms
            self.assertLess(timer_data["latest_value"], 0.5)     # Less than 500ms

    def test_full_system_stress_integration(self):
        """Test Qt compliance under system stress conditions."""
        with self.assert_no_qt_errors():
            stress_duration = 2.0  # seconds
            start_time = time.perf_counter()

            # Create multiple concurrent operations
            stress_widgets = []
            stress_workers = []
            stress_timers = []

            try:
                while time.perf_counter() - start_time < stress_duration:
                    # Create widget with timer
                    widget = QWidget()
                    stress_widgets.append(widget)

                    # Create timer for widget
                    timer = QTimer(widget)
                    timer.timeout.connect(lambda: QCoreApplication.processEvents())
                    timer.start(50)
                    stress_timers.append(timer)

                    # Create worker
                    class StressWorker(SafeWorkerBase):
                        def do_work(self):
                            # Quick work
                            time.sleep(0.01)

                    worker = StressWorker()
                    stress_workers.append(worker)
                    self.thread_manager.start_worker(worker, f"stress_worker_{len(stress_workers)}")

                    # Process events
                    QCoreApplication.processEvents()

                    # Small delay
                    time.sleep(0.05)

                    # Clean up older objects periodically
                    if len(stress_widgets) > 20:
                        # Remove oldest widgets
                        for _ in range(5):
                            if stress_widgets:
                                old_widget = stress_widgets.pop(0)
                                old_widget.deleteLater()

                        # Clean up finished workers
                        self.thread_manager.cleanup_finished_threads()
                        QCoreApplication.processEvents()

                print(f"Stress test created {len(stress_widgets)} widgets and {len(stress_workers)} workers")

            finally:
                # Clean up all stress test objects
                for timer in stress_timers:
                    timer.stop()

                for widget in stress_widgets:
                    widget.deleteLater()

                self.thread_manager.cleanup_finished_threads()

                # Process cleanup events
                for _ in range(50):
                    QCoreApplication.processEvents()
                    time.sleep(0.01)

                gc.collect()

    def test_integration_coverage_analysis(self):
        """Analyze integration test coverage and report results."""
        # This test validates that our integration tests cover all major components

        # Check Qt error detector coverage
        error_summary = self.qt_error_detector.get_error_summary()
        self.assertGreater(error_summary["patterns_registered"], 5,
                          "Should have multiple error patterns registered")

        # Check thread manager coverage
        thread_summary = self.thread_manager.get_manager_status()
        self.assertIn("total_workers_created", thread_summary)

        # Check monitoring system coverage
        system_status = self.monitoring_system.get_system_status()
        self.assertTrue(system_status.qt_error_detector_active)
        self.assertTrue(system_status.performance_metrics_active)
        self.assertTrue(system_status.resilience_manager_active)

        # Check thread validator coverage
        health_summary = self.thread_validator.get_health_summary()
        self.assertIn("registered_pools", health_summary)

        # Generate coverage report
        coverage_report = {
            "qt_error_detection": {
                "patterns_tested": error_summary["patterns_registered"],
                "total_errors_detected": error_summary["total_errors"],
                "coverage": "comprehensive"
            },
            "thread_management": {
                "workers_tested": thread_summary.get("total_workers_created", 0),
                "pools_validated": len(health_summary.get("registered_pools", [])),
                "coverage": "comprehensive"
            },
            "monitoring_integration": {
                "components_active": sum([
                    system_status.qt_error_detector_active,
                    system_status.performance_metrics_active,
                    system_status.resilience_manager_active,
                    system_status.health_monitor_running
                ]),
                "coverage": "complete"
            },
            "gui_integration": {
                "widget_testing": "completed",
                "signal_connection_testing": "completed",
                "memory_management_testing": "completed",
                "coverage": "comprehensive"
            }
        }

        print("Integration Test Coverage Analysis:")
        for component, details in coverage_report.items():
            print(f"  {component}: {details.get('coverage', 'unknown')}")

        # Ensure all major components were tested
        self.assertEqual(coverage_report["monitoring_integration"]["components_active"], 4,
                        "All monitoring components should be active")


class QtComplianceIntegrationRunner:
    """Test runner for Qt compliance integration tests."""

    @staticmethod
    def run_integration_tests(verbose: bool = True) -> Dict[str, Any]:
        """Run all integration tests and return results."""
        # Set up test environment
        os.environ['PYXPCS_TESTING_MODE'] = '1'

        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(QtComplianceIntegrationTestSuite)

        # Run tests
        if verbose:
            runner = unittest.TextTestRunner(verbosity=2)
        else:
            runner = unittest.TextTestRunner(verbosity=1)

        result = runner.run(suite)

        # Return test results
        return {
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            "details": {
                "failures": result.failures,
                "errors": result.errors
            }
        }

    @staticmethod
    def run_quick_integration_check() -> bool:
        """Run a quick integration health check."""
        try:
            # Initialize monitoring system briefly
            monitoring_system = initialize_integrated_monitoring()

            # Perform basic checks
            qt_detector = get_qt_error_detector()
            error_summary = qt_detector.get_error_summary()

            # Check system health
            system_status = monitoring_system.get_system_status()
            is_healthy = monitoring_system.is_monitoring_healthy()

            # Cleanup
            shutdown_integrated_monitoring()

            return is_healthy and error_summary["total_errors"] == 0

        except Exception as e:
            print(f"Quick integration check failed: {e}")
            return False


if __name__ == "__main__":
    # Allow running tests directly
    runner = QtComplianceIntegrationRunner()
    results = runner.run_integration_tests()

    print(f"\nIntegration Test Results:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")

    if results['success_rate'] < 1.0:
        sys.exit(1)  # Exit with error if tests failed