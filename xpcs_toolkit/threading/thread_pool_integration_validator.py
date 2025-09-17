"""
Thread Pool Integration and Resource Management Validator for XPCS Toolkit.

This module provides comprehensive validation of thread pool integration,
resource management, and overall threading system health to ensure optimal
performance and prevent resource leaks or threading violations.
"""

import gc
import os
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import weakref

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QObject, QThread, QThreadPool, QTimer, Signal, QMutex, QMutexLocker, Slot

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class PoolHealthStatus(Enum):
    """Thread pool health status levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class ResourceValidationResult(Enum):
    """Resource validation results."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    CRITICAL = "critical"


@dataclass
class ThreadPoolHealthMetrics:
    """
    Comprehensive thread pool health metrics for performance monitoring and validation.

    This dataclass provides a complete set of metrics for monitoring thread pool health,
    performance, and resource usage. It includes automatic health score calculation
    and status determination based on configurable thresholds.

    Attributes:
        pool_id (str): Unique identifier for the thread pool being monitored
        timestamp (float): Timestamp when metrics were captured (default: current time)

        # Thread Management Metrics
        active_threads (int): Number of currently active threads
        max_threads (int): Maximum number of threads allowed in the pool
        queued_tasks (int): Number of tasks waiting in the queue
        completed_tasks (int): Total number of successfully completed tasks
        failed_tasks (int): Total number of failed tasks

        # Performance Metrics
        avg_task_duration (float): Average task execution time in seconds
        peak_memory_usage_mb (float): Peak memory usage in megabytes
        cpu_utilization (float): CPU utilization percentage (0-100)
        throughput_tasks_per_second (float): Task completion rate

        # Health Indicators
        thread_creation_failures (int): Number of thread creation failures
        task_timeout_count (int): Number of tasks that exceeded timeout
        resource_leak_count (int): Number of detected resource leaks
        qt_violations (int): Number of Qt threading violations detected

        # Resource Usage
        file_handles_open (int): Number of open file handles
        memory_allocations (int): Number of active memory allocations
        network_connections (int): Number of active network connections
        temporary_files (int): Number of temporary files created

    Methods:
        calculate_health_score() -> float:
            Calculates overall health score (0.0 to 1.0) based on all metrics
        get_health_status() -> PoolHealthStatus:
            Determines health status based on calculated score

    Example:
        >>> metrics = ThreadPoolHealthMetrics(
        ...     pool_id="main_pool",
        ...     active_threads=4,
        ...     max_threads=8,
        ...     completed_tasks=150
        ... )
        >>> health_score = metrics.calculate_health_score()
        >>> status = metrics.get_health_status()
        >>> print(f"Pool health: {status.value} (score: {health_score:.2f})")

    Note:
        Health scores below 0.7 indicate potential issues that should be investigated.
        Scores below 0.5 indicate critical issues requiring immediate attention.
    """

    pool_id: str
    timestamp: float = field(default_factory=time.perf_counter)

    # Thread counts
    active_threads: int = 0
    max_threads: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0

    # Performance metrics
    avg_task_duration: float = 0.0
    peak_memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    throughput_tasks_per_second: float = 0.0

    # Health indicators
    thread_creation_failures: int = 0
    task_timeout_count: int = 0
    resource_leak_count: int = 0
    qt_violations: int = 0

    # Resource usage
    file_handles_open: int = 0
    memory_allocations: int = 0
    network_connections: int = 0
    temporary_files: int = 0

    def calculate_health_score(self) -> float:
        """
        Calculate overall health score (0.0 to 1.0).

        Returns:
            Health score where 1.0 is perfect health
        """
        score = 1.0

        # Penalize high utilization
        if self.max_threads > 0:
            utilization = self.active_threads / self.max_threads
            if utilization > 0.9:
                score -= 0.2
            elif utilization > 0.7:
                score -= 0.1

        # Penalize failures
        if self.thread_creation_failures > 0:
            score -= min(0.3, self.thread_creation_failures * 0.1)

        if self.task_timeout_count > 0:
            score -= min(0.2, self.task_timeout_count * 0.05)

        if self.resource_leak_count > 0:
            score -= min(0.4, self.resource_leak_count * 0.1)

        if self.qt_violations > 0:
            score -= min(0.5, self.qt_violations * 0.2)

        return max(0.0, score)

    def get_health_status(self) -> PoolHealthStatus:
        """Get health status based on metrics."""
        score = self.calculate_health_score()

        if score >= 0.9:
            return PoolHealthStatus.EXCELLENT
        elif score >= 0.7:
            return PoolHealthStatus.GOOD
        elif score >= 0.5:
            return PoolHealthStatus.WARNING
        elif score >= 0.2:
            return PoolHealthStatus.CRITICAL
        else:
            return PoolHealthStatus.FAILED


@dataclass
class ResourceLeakInfo:
    """Information about a detected resource leak."""

    leak_id: str
    resource_type: str
    worker_id: str
    allocation_time: float
    detection_time: float = field(default_factory=time.perf_counter)
    leak_duration: float = 0.0
    stack_trace: str = ""
    cleanup_attempted: bool = False
    cleanup_successful: bool = False

    def __post_init__(self):
        """Calculate leak duration after initialization."""
        self.leak_duration = self.detection_time - self.allocation_time


class ThreadPoolIntegrationValidator(QObject):
    """
    Comprehensive validator for thread pool integration and resource management.

    This validator monitors thread pools, validates resource usage, detects
    leaks, and ensures optimal performance and Qt compliance.
    """

    # Validation signals
    validation_completed = Signal(str, dict)  # pool_id, results
    health_status_changed = Signal(str, str, str)  # pool_id, old_status, new_status
    resource_leak_detected = Signal(str, object)  # pool_id, leak_info
    performance_warning = Signal(str, str, dict)  # pool_id, warning_type, details

    def __init__(self):
        super().__init__()
        self._monitored_pools: Dict[str, QThreadPool] = {}
        self._pool_metrics: Dict[str, ThreadPoolHealthMetrics] = {}
        self._resource_tracking: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._leak_detection: Dict[str, ResourceLeakInfo] = {}

        # Validation timers
        self._health_monitor = QTimer()
        self._health_monitor.timeout.connect(self._validate_all_pools)
        self._health_monitor.start(10000)  # Check every 10 seconds

        self._resource_monitor = QTimer()
        self._resource_monitor.timeout.connect(self._check_resource_leaks)
        self._resource_monitor.start(30000)  # Check every 30 seconds

        # Historical data
        self._health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._validation_results: Dict[str, List[Dict]] = defaultdict(list)

        # Validation settings
        self.max_memory_usage_mb = 1024  # 1GB warning threshold
        self.max_task_duration_warning = 300.0  # 5 minutes
        self.resource_leak_timeout = 600.0  # 10 minutes

        logger.info("ThreadPoolIntegrationValidator initialized")

    def register_thread_pool(self, pool_id: str, thread_pool: QThreadPool):
        """
        Register a thread pool for monitoring and validation.

        Args:
            pool_id: Unique identifier for the pool
            thread_pool: QThreadPool instance to monitor
        """
        self._monitored_pools[pool_id] = thread_pool
        self._pool_metrics[pool_id] = ThreadPoolHealthMetrics(pool_id=pool_id)

        logger.info(f"Registered thread pool '{pool_id}' for validation")

    def unregister_thread_pool(self, pool_id: str):
        """Unregister a thread pool from monitoring."""
        if pool_id in self._monitored_pools:
            del self._monitored_pools[pool_id]
            del self._pool_metrics[pool_id]
            if pool_id in self._health_history:
                del self._health_history[pool_id]

            logger.info(f"Unregistered thread pool '{pool_id}' from validation")

    def validate_thread_pool_health(self, pool_id: str) -> Dict[str, Any]:
        """
        Comprehensive health validation for a specific thread pool.

        Args:
            pool_id: ID of the pool to validate

        Returns:
            Dictionary containing validation results
        """
        if pool_id not in self._monitored_pools:
            return {"error": f"Thread pool '{pool_id}' not registered"}

        thread_pool = self._monitored_pools[pool_id]
        metrics = self._pool_metrics[pool_id]

        validation_results = {
            "pool_id": pool_id,
            "timestamp": time.perf_counter(),
            "validations": {},
            "warnings": [],
            "errors": [],
            "overall_status": "unknown"
        }

        try:
            # Update metrics
            self._update_pool_metrics(pool_id, thread_pool, metrics)

            # Validate thread configuration
            config_result = self._validate_thread_configuration(thread_pool)
            validation_results["validations"]["thread_configuration"] = config_result

            # Validate resource usage
            resource_result = self._validate_resource_usage(pool_id, metrics)
            validation_results["validations"]["resource_usage"] = resource_result

            # Validate performance
            performance_result = self._validate_performance_metrics(metrics)
            validation_results["validations"]["performance"] = performance_result

            # Validate Qt compliance
            qt_result = self._validate_qt_compliance(pool_id)
            validation_results["validations"]["qt_compliance"] = qt_result

            # Calculate overall status
            overall_status = self._calculate_overall_status(validation_results["validations"])
            validation_results["overall_status"] = overall_status

            # Check for status changes
            previous_status = getattr(metrics, '_last_status', None)
            current_status = metrics.get_health_status().value
            if previous_status and previous_status != current_status:
                self.health_status_changed.emit(pool_id, previous_status, current_status)

            metrics._last_status = current_status

            # Store results
            self._validation_results[pool_id].append(validation_results)
            self._health_history[pool_id].append(metrics.calculate_health_score())

            logger.debug(f"Validated thread pool '{pool_id}': {overall_status}")

        except Exception as e:
            validation_results["error"] = f"Validation failed: {e}"
            validation_results["overall_status"] = "error"
            logger.error(f"Thread pool validation failed for '{pool_id}': {e}")

        return validation_results

    def _update_pool_metrics(self, pool_id: str, thread_pool: QThreadPool,
                           metrics: ThreadPoolHealthMetrics):
        """Update thread pool metrics from current state."""
        metrics.timestamp = time.perf_counter()
        metrics.active_threads = thread_pool.activeThreadCount()
        metrics.max_threads = thread_pool.maxThreadCount()

        # Get system resource info if available
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            metrics.peak_memory_usage_mb = memory_info.rss / 1024 / 1024
            metrics.cpu_utilization = process.cpu_percent()

            # Count file handles (Unix-like systems)
            if hasattr(process, 'num_fds'):
                metrics.file_handles_open = process.num_fds()

        except ImportError:
            logger.debug("psutil not available for detailed metrics")

    def _validate_thread_configuration(self, thread_pool: QThreadPool) -> Dict[str, Any]:
        """Validate thread pool configuration."""
        result = {
            "status": ResourceValidationResult.PASSED.value,
            "issues": [],
            "recommendations": []
        }

        max_threads = thread_pool.maxThreadCount()
        active_threads = thread_pool.activeThreadCount()

        # Check thread limits
        cpu_count = os.cpu_count() or 4
        if max_threads > cpu_count * 3:
            result["issues"].append(f"High thread limit ({max_threads}) may cause context switching overhead")
            result["recommendations"].append(f"Consider reducing to {cpu_count * 2}")
            result["status"] = ResourceValidationResult.WARNING.value

        if max_threads < 2:
            result["issues"].append("Very low thread limit may bottleneck parallel operations")
            result["recommendations"].append("Consider increasing minimum threads to 2-4")
            result["status"] = ResourceValidationResult.WARNING.value

        # Check utilization
        if max_threads > 0:
            utilization = active_threads / max_threads
            if utilization > 0.95:
                result["issues"].append(f"High thread utilization ({utilization:.1%})")
                result["recommendations"].append("Consider increasing thread pool size")
                if result["status"] == ResourceValidationResult.PASSED.value:
                    result["status"] = ResourceValidationResult.WARNING.value

        return result

    def _validate_resource_usage(self, pool_id: str, metrics: ThreadPoolHealthMetrics) -> Dict[str, Any]:
        """Validate resource usage patterns."""
        result = {
            "status": ResourceValidationResult.PASSED.value,
            "issues": [],
            "recommendations": []
        }

        # Memory usage validation
        if metrics.peak_memory_usage_mb > self.max_memory_usage_mb:
            result["issues"].append(f"High memory usage: {metrics.peak_memory_usage_mb:.1f}MB")
            result["recommendations"].append("Monitor for memory leaks and optimize data handling")
            result["status"] = ResourceValidationResult.WARNING.value

        # Resource leak validation
        if metrics.resource_leak_count > 0:
            result["issues"].append(f"Resource leaks detected: {metrics.resource_leak_count}")
            result["recommendations"].append("Implement proper resource cleanup in workers")
            result["status"] = ResourceValidationResult.FAILED.value

        # File handle validation
        if metrics.file_handles_open > 1000:
            result["issues"].append(f"High file handle count: {metrics.file_handles_open}")
            result["recommendations"].append("Ensure files are properly closed after use")
            if result["status"] == ResourceValidationResult.PASSED.value:
                result["status"] = ResourceValidationResult.WARNING.value

        return result

    def _validate_performance_metrics(self, metrics: ThreadPoolHealthMetrics) -> Dict[str, Any]:
        """Validate performance metrics."""
        result = {
            "status": ResourceValidationResult.PASSED.value,
            "issues": [],
            "recommendations": []
        }

        # Task duration validation
        if metrics.avg_task_duration > self.max_task_duration_warning:
            result["issues"].append(f"Long average task duration: {metrics.avg_task_duration:.1f}s")
            result["recommendations"].append("Consider breaking down long-running tasks")
            result["status"] = ResourceValidationResult.WARNING.value

        # Timeout validation
        if metrics.task_timeout_count > 0:
            result["issues"].append(f"Task timeouts detected: {metrics.task_timeout_count}")
            result["recommendations"].append("Review task complexity and timeout settings")
            if result["status"] == ResourceValidationResult.PASSED.value:
                result["status"] = ResourceValidationResult.WARNING.value

        # Throughput validation (if we have historical data)
        if metrics.throughput_tasks_per_second < 0.1 and metrics.completed_tasks > 10:
            result["issues"].append("Low task throughput detected")
            result["recommendations"].append("Optimize task processing or increase thread count")
            if result["status"] == ResourceValidationResult.PASSED.value:
                result["status"] = ResourceValidationResult.WARNING.value

        return result

    def _validate_qt_compliance(self, pool_id: str) -> Dict[str, Any]:
        """Validate Qt threading compliance."""
        result = {
            "status": ResourceValidationResult.PASSED.value,
            "issues": [],
            "recommendations": []
        }

        metrics = self._pool_metrics[pool_id]

        # Qt violation validation
        if metrics.qt_violations > 0:
            result["issues"].append(f"Qt threading violations detected: {metrics.qt_violations}")
            result["recommendations"].append("Review Qt object creation and signal/slot usage in threads")
            result["status"] = ResourceValidationResult.FAILED.value

        # Check main thread usage
        main_thread = QtCore.QThread.currentThread()
        current_thread = threading.current_thread()

        if hasattr(current_thread, 'name') and 'ThreadPoolThread' in current_thread.name:
            # We're in a thread pool thread, which is expected for workers
            pass
        elif QtCore.QThread.currentThread() != main_thread:
            # We're in a Qt thread but not main thread
            if pool_id == "main_operations":
                result["issues"].append("Main operations pool accessed from non-main Qt thread")
                result["recommendations"].append("Ensure main operations are called from main thread")
                if result["status"] == ResourceValidationResult.PASSED.value:
                    result["status"] = ResourceValidationResult.WARNING.value

        return result

    def _calculate_overall_status(self, validations: Dict[str, Dict[str, Any]]) -> str:
        """Calculate overall validation status."""
        statuses = [v.get("status", "unknown") for v in validations.values()]

        if ResourceValidationResult.FAILED.value in statuses:
            return ResourceValidationResult.FAILED.value
        elif ResourceValidationResult.CRITICAL.value in statuses:
            return ResourceValidationResult.CRITICAL.value
        elif ResourceValidationResult.WARNING.value in statuses:
            return ResourceValidationResult.WARNING.value
        else:
            return ResourceValidationResult.PASSED.value

    @Slot()
    def _validate_all_pools(self):
        """Validate all registered thread pools."""
        for pool_id in self._monitored_pools:
            try:
                results = self.validate_thread_pool_health(pool_id)
                self.validation_completed.emit(pool_id, results)
            except Exception as e:
                logger.error(f"Failed to validate pool '{pool_id}': {e}")

    @Slot()
    def _check_resource_leaks(self):
        """Check for resource leaks across all pools."""
        current_time = time.perf_counter()

        # Check for long-running resource allocations
        for pool_id, resources in self._resource_tracking.items():
            for resource_id, resource_info in resources.items():
                allocation_time = resource_info.get('allocation_time', current_time)
                age = current_time - allocation_time

                if age > self.resource_leak_timeout:
                    # Potential leak detected
                    leak_info = ResourceLeakInfo(
                        leak_id=f"leak_{pool_id}_{resource_id}",
                        resource_type=resource_info.get('type', 'unknown'),
                        worker_id=resource_info.get('worker_id', 'unknown'),
                        allocation_time=allocation_time,
                        stack_trace=resource_info.get('stack_trace', '')
                    )

                    self._leak_detection[leak_info.leak_id] = leak_info
                    self.resource_leak_detected.emit(pool_id, leak_info)

                    # Update metrics
                    if pool_id in self._pool_metrics:
                        self._pool_metrics[pool_id].resource_leak_count += 1

                    logger.warning(
                        f"Resource leak detected in pool '{pool_id}': "
                        f"Resource '{resource_id}' allocated {age:.1f}s ago"
                    )

    def track_resource_allocation(self, pool_id: str, resource_id: str,
                                resource_type: str, worker_id: str = None):
        """Track resource allocation for leak detection."""
        self._resource_tracking[pool_id][resource_id] = {
            'type': resource_type,
            'worker_id': worker_id,
            'allocation_time': time.perf_counter(),
            'stack_trace': ''.join(traceback.format_stack())
        }

    def track_resource_deallocation(self, pool_id: str, resource_id: str):
        """Track resource deallocation."""
        if pool_id in self._resource_tracking:
            self._resource_tracking[pool_id].pop(resource_id, None)

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for all monitored pools."""
        summary = {
            'timestamp': time.perf_counter(),
            'total_pools': len(self._monitored_pools),
            'pool_health': {},
            'overall_status': 'unknown',
            'total_issues': 0,
            'critical_issues': 0
        }

        statuses = []
        total_issues = 0
        critical_issues = 0

        for pool_id, metrics in self._pool_metrics.items():
            health_status = metrics.get_health_status()
            health_score = metrics.calculate_health_score()

            pool_summary = {
                'status': health_status.value,
                'score': health_score,
                'active_threads': metrics.active_threads,
                'max_threads': metrics.max_threads,
                'resource_leaks': metrics.resource_leak_count,
                'qt_violations': metrics.qt_violations
            }

            summary['pool_health'][pool_id] = pool_summary
            statuses.append(health_status)

            if health_status in [PoolHealthStatus.WARNING, PoolHealthStatus.CRITICAL, PoolHealthStatus.FAILED]:
                total_issues += 1
                if health_status in [PoolHealthStatus.CRITICAL, PoolHealthStatus.FAILED]:
                    critical_issues += 1

        # Determine overall status
        if PoolHealthStatus.FAILED in statuses:
            summary['overall_status'] = 'failed'
        elif PoolHealthStatus.CRITICAL in statuses:
            summary['overall_status'] = 'critical'
        elif PoolHealthStatus.WARNING in statuses:
            summary['overall_status'] = 'warning'
        elif PoolHealthStatus.GOOD in statuses or PoolHealthStatus.EXCELLENT in statuses:
            summary['overall_status'] = 'good'

        summary['total_issues'] = total_issues
        summary['critical_issues'] = critical_issues

        return summary

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        summary = self.get_health_summary()

        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("Thread Pool Integration Validation Report")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Pools Monitored: {summary['total_pools']}")
        report_lines.append(f"Overall Status: {summary['overall_status'].upper()}")
        report_lines.append(f"Total Issues: {summary['total_issues']}")
        report_lines.append(f"Critical Issues: {summary['critical_issues']}")
        report_lines.append("")

        # Pool-specific details
        for pool_id, pool_health in summary['pool_health'].items():
            report_lines.append(f"Pool: {pool_id}")
            report_lines.append("-" * 40)
            report_lines.append(f"  Status: {pool_health['status'].upper()}")
            report_lines.append(f"  Health Score: {pool_health['score']:.2f}")
            report_lines.append(f"  Active Threads: {pool_health['active_threads']}/{pool_health['max_threads']}")

            if pool_health['resource_leaks'] > 0:
                report_lines.append(f"  ⚠️  Resource Leaks: {pool_health['resource_leaks']}")

            if pool_health['qt_violations'] > 0:
                report_lines.append(f"  ❌ Qt Violations: {pool_health['qt_violations']}")

            report_lines.append("")

        # Recent validation results
        report_lines.append("Recent Validation Results:")
        report_lines.append("-" * 40)
        for pool_id in self._monitored_pools:
            if pool_id in self._validation_results and self._validation_results[pool_id]:
                latest_result = self._validation_results[pool_id][-1]
                report_lines.append(f"  {pool_id}: {latest_result['overall_status']}")

                for validation_type, validation_result in latest_result.get('validations', {}).items():
                    status = validation_result.get('status', 'unknown')
                    if status != ResourceValidationResult.PASSED.value:
                        issues = validation_result.get('issues', [])
                        for issue in issues:
                            report_lines.append(f"    • {issue}")

        report_lines.append("=" * 70)

        return '\n'.join(report_lines)

    def shutdown(self):
        """Shutdown the validator and clean up resources."""
        logger.info("Shutting down ThreadPoolIntegrationValidator")

        # Stop monitoring
        self._health_monitor.stop()
        self._resource_monitor.stop()

        # Clear tracking data
        self._monitored_pools.clear()
        self._pool_metrics.clear()
        self._resource_tracking.clear()
        self._leak_detection.clear()

        logger.info("ThreadPoolIntegrationValidator shutdown complete")


# Global validator instance
_global_validator: Optional[ThreadPoolIntegrationValidator] = None


def get_thread_pool_validator() -> ThreadPoolIntegrationValidator:
    """Get the global thread pool validator instance."""
    global _global_validator

    if _global_validator is None:
        _global_validator = ThreadPoolIntegrationValidator()

    return _global_validator


def initialize_thread_pool_validation():
    """Initialize global thread pool validation."""
    validator = get_thread_pool_validator()
    logger.info("Global thread pool validation initialized")
    return validator


@contextmanager
def thread_pool_validation_context():
    """Context manager for thread pool validation."""
    validator = get_thread_pool_validator()
    try:
        yield validator
    except Exception as e:
        logger.error(f"Error in thread pool validation context: {e}")
        raise
    finally:
        # Generate final report
        try:
            report = validator.generate_validation_report()
            logger.info("Thread pool validation report:\n" + report)
        except Exception as e:
            logger.warning(f"Failed to generate validation report: {e}")


def shutdown_thread_pool_validation():
    """Shutdown global thread pool validation."""
    global _global_validator

    if _global_validator is not None:
        _global_validator.shutdown()
        _global_validator = None
        logger.info("Global thread pool validation shutdown")