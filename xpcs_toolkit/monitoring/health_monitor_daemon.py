"""
Automated Health Monitoring Daemon for Qt Compliance System.

This module provides continuous monitoring of Qt threading compliance,
resource usage, and system health with automated alerting and response
capabilities.
"""

import asyncio
import json
import logging
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from PySide6.QtCore import QObject, QThread, QTimer, Signal, QMutex, QMutexLocker

from ..threading.thread_pool_integration_validator import (
    ThreadPoolIntegrationValidator,
    ThreadPoolHealthMetrics,
    PoolHealthStatus,
    get_thread_pool_validator
)
from ..threading.qt_compliant_thread_manager import get_qt_compliant_thread_manager
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitoringInterval(Enum):
    """Monitoring check intervals."""

    REALTIME = 1.0  # 1 second
    FAST = 5.0      # 5 seconds
    NORMAL = 30.0   # 30 seconds
    SLOW = 300.0    # 5 minutes


@dataclass
class HealthAlert:
    """Health monitoring alert."""

    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return asdict(self)

    def format_message(self) -> str:
        """Format alert message for logging."""
        timestamp_str = datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        status = "RESOLVED" if self.resolved else "ACTIVE"
        return f"[{timestamp_str}] {self.severity.value.upper()} - {self.component}: {self.message} ({status})"


@dataclass
class MonitoringConfiguration:
    """Configuration for health monitoring system."""

    # Monitoring intervals
    thread_pool_check_interval: float = MonitoringInterval.NORMAL.value
    qt_compliance_check_interval: float = MonitoringInterval.FAST.value
    resource_check_interval: float = MonitoringInterval.NORMAL.value
    performance_check_interval: float = MonitoringInterval.SLOW.value

    # Alert thresholds
    max_qt_violations_per_minute: int = 5
    max_memory_usage_mb: float = 2048.0
    max_cpu_utilization_percent: float = 85.0
    max_thread_utilization_percent: float = 90.0
    max_resource_leaks: int = 10

    # History retention
    max_alert_history: int = 1000
    max_metrics_history: int = 500
    history_retention_days: int = 7

    # Alert configuration
    enable_file_logging: bool = True
    enable_console_alerts: bool = True
    alert_log_file: Optional[str] = None

    # Recovery configuration
    enable_auto_recovery: bool = True
    recovery_timeout_seconds: float = 30.0


class HealthMonitorDaemon(QObject):
    """
    Automated health monitoring daemon for Qt compliance system.

    Provides continuous monitoring with:
    - Thread pool health validation
    - Qt compliance checking
    - Resource leak detection
    - Performance monitoring
    - Automated alerting
    - Basic auto-recovery
    """

    # Signals
    alert_triggered = Signal(object)  # HealthAlert
    health_status_changed = Signal(str, str, str)  # component, old_status, new_status
    monitoring_started = Signal()
    monitoring_stopped = Signal()

    def __init__(self, config: Optional[MonitoringConfiguration] = None, parent: QObject = None):
        """Initialize health monitoring daemon."""
        super().__init__(parent)

        self.config = config or MonitoringConfiguration()
        self._is_running = False
        self._monitoring_thread: Optional[QThread] = None

        # Monitoring components
        self._thread_pool_validator = get_thread_pool_validator()
        self._thread_manager = get_qt_compliant_thread_manager()

        # Alert management
        self._active_alerts: Dict[str, HealthAlert] = {}
        self._alert_history: deque = deque(maxlen=self.config.max_alert_history)

        # Metrics history
        self._metrics_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_metrics_history)
        )

        # Monitoring state
        self._last_check_times: Dict[str, float] = {}
        self._component_health_status: Dict[str, str] = {}
        self._mutex = QMutex()

        # Timers for different monitoring intervals
        self._timers: Dict[str, QTimer] = {}

        # Setup logging
        self._setup_logging()

        logger.info("Health monitoring daemon initialized")

    def _setup_logging(self):
        """Setup alert logging."""
        if self.config.enable_file_logging:
            if not self.config.alert_log_file:
                # Default log file location
                log_dir = Path.home() / ".xpcs_toolkit" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                self.config.alert_log_file = str(log_dir / "health_alerts.log")

            # Setup file handler for alerts
            file_handler = logging.FileHandler(self.config.alert_log_file)
            file_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    def start_monitoring(self):
        """Start automated health monitoring."""
        if self._is_running:
            logger.warning("Health monitoring already running")
            return

        logger.info("Starting health monitoring daemon")
        self._is_running = True

        # Setup monitoring timers
        self._setup_monitoring_timers()

        # Start timers
        for timer in self._timers.values():
            timer.start()

        self.monitoring_started.emit()
        logger.info("Health monitoring daemon started successfully")

    def stop_monitoring(self):
        """Stop automated health monitoring."""
        if not self._is_running:
            return

        logger.info("Stopping health monitoring daemon")
        self._is_running = False

        # Stop all timers
        for timer in self._timers.values():
            timer.stop()

        # Clear timers
        self._timers.clear()

        self.monitoring_stopped.emit()
        logger.info("Health monitoring daemon stopped")

    def _setup_monitoring_timers(self):
        """Setup monitoring timers for different check intervals."""
        timer_configs = [
            ("thread_pool", self.config.thread_pool_check_interval, self._check_thread_pool_health),
            ("qt_compliance", self.config.qt_compliance_check_interval, self._check_qt_compliance),
            ("resources", self.config.resource_check_interval, self._check_resource_usage),
            ("performance", self.config.performance_check_interval, self._check_performance_metrics),
        ]

        for name, interval, check_func in timer_configs:
            timer = QTimer(self)
            timer.timeout.connect(check_func)
            timer.setInterval(int(interval * 1000))  # Convert to milliseconds
            self._timers[name] = timer

    def _check_thread_pool_health(self):
        """Check thread pool health status."""
        try:
            with QMutexLocker(self._mutex):
                current_time = time.perf_counter()
                self._last_check_times["thread_pool"] = current_time

                # Get health summary
                health_summary = self._thread_pool_validator.get_health_summary()

                # Store metrics
                self._metrics_history["thread_pool_health"].append({
                    "timestamp": current_time,
                    "summary": health_summary
                })

                # Check for issues
                if health_summary.get("critical_issues", 0) > 0:
                    self._trigger_alert(
                        AlertSeverity.CRITICAL,
                        "thread_pool",
                        f"Critical thread pool issues detected: {health_summary['critical_issues']}",
                        health_summary
                    )
                elif health_summary.get("total_issues", 0) > 5:
                    self._trigger_alert(
                        AlertSeverity.WARNING,
                        "thread_pool",
                        f"Multiple thread pool issues detected: {health_summary['total_issues']}",
                        health_summary
                    )

                # Check thread utilization
                for pool_id in health_summary.get("registered_pools", []):
                    pool_metrics = self._thread_pool_validator._pool_metrics.get(pool_id)
                    if pool_metrics:
                        utilization = self._calculate_thread_utilization(pool_metrics)
                        if utilization > self.config.max_thread_utilization_percent:
                            self._trigger_alert(
                                AlertSeverity.WARNING,
                                f"thread_pool_{pool_id}",
                                f"High thread utilization: {utilization:.1f}%",
                                {"utilization": utilization, "pool_id": pool_id}
                            )

        except Exception as e:
            logger.error(f"Thread pool health check failed: {e}")
            logger.debug(traceback.format_exc())

    def _check_qt_compliance(self):
        """Check Qt compliance violations."""
        try:
            with QMutexLocker(self._mutex):
                current_time = time.perf_counter()
                self._last_check_times["qt_compliance"] = current_time

                # Check for Qt violations in recent history
                violation_count = 0
                cutoff_time = current_time - 60.0  # Last minute

                for pool_id, pool_metrics in self._thread_pool_validator._pool_metrics.items():
                    if hasattr(pool_metrics, 'qt_violations') and pool_metrics.timestamp > cutoff_time:
                        violation_count += pool_metrics.qt_violations

                # Store metrics
                self._metrics_history["qt_compliance"].append({
                    "timestamp": current_time,
                    "violations_per_minute": violation_count
                })

                # Check violation threshold
                if violation_count > self.config.max_qt_violations_per_minute:
                    self._trigger_alert(
                        AlertSeverity.ERROR,
                        "qt_compliance",
                        f"Excessive Qt violations: {violation_count} in last minute",
                        {"violations": violation_count, "threshold": self.config.max_qt_violations_per_minute}
                    )

        except Exception as e:
            logger.error(f"Qt compliance check failed: {e}")
            logger.debug(traceback.format_exc())

    def _check_resource_usage(self):
        """Check system resource usage."""
        try:
            with QMutexLocker(self._mutex):
                current_time = time.perf_counter()
                self._last_check_times["resources"] = current_time

                # Get resource metrics
                resource_metrics = self._collect_resource_metrics()

                # Store metrics
                self._metrics_history["resources"].append({
                    "timestamp": current_time,
                    "metrics": resource_metrics
                })

                # Check memory usage
                memory_usage = resource_metrics.get("memory_mb", 0)
                if memory_usage > self.config.max_memory_usage_mb:
                    self._trigger_alert(
                        AlertSeverity.WARNING,
                        "resources",
                        f"High memory usage: {memory_usage:.1f} MB",
                        {"memory_mb": memory_usage, "threshold": self.config.max_memory_usage_mb}
                    )

                # Check resource leaks
                leak_count = resource_metrics.get("resource_leaks", 0)
                if leak_count > self.config.max_resource_leaks:
                    self._trigger_alert(
                        AlertSeverity.ERROR,
                        "resources",
                        f"Resource leaks detected: {leak_count}",
                        {"leak_count": leak_count, "threshold": self.config.max_resource_leaks}
                    )

        except Exception as e:
            logger.error(f"Resource usage check failed: {e}")
            logger.debug(traceback.format_exc())

    def _check_performance_metrics(self):
        """Check performance metrics."""
        try:
            with QMutexLocker(self._mutex):
                current_time = time.perf_counter()
                self._last_check_times["performance"] = current_time

                # Get performance metrics
                perf_metrics = self._collect_performance_metrics()

                # Store metrics
                self._metrics_history["performance"].append({
                    "timestamp": current_time,
                    "metrics": perf_metrics
                })

                # Check CPU utilization
                cpu_usage = perf_metrics.get("cpu_percent", 0)
                if cpu_usage > self.config.max_cpu_utilization_percent:
                    self._trigger_alert(
                        AlertSeverity.WARNING,
                        "performance",
                        f"High CPU utilization: {cpu_usage:.1f}%",
                        {"cpu_percent": cpu_usage, "threshold": self.config.max_cpu_utilization_percent}
                    )

        except Exception as e:
            logger.error(f"Performance metrics check failed: {e}")
            logger.debug(traceback.format_exc())

    def _collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect current resource usage metrics."""
        metrics = {}

        try:
            import psutil
            process = psutil.Process()

            # Memory metrics
            memory_info = process.memory_info()
            metrics["memory_mb"] = memory_info.rss / 1024 / 1024
            metrics["virtual_memory_mb"] = memory_info.vms / 1024 / 1024

            # File handle metrics
            if hasattr(process, 'num_fds'):
                metrics["file_handles"] = process.num_fds()

            # Thread metrics
            metrics["thread_count"] = process.num_threads()

            # Calculate resource leaks from validator
            leak_count = 0
            for pool_metrics in self._thread_pool_validator._pool_metrics.values():
                leak_count += getattr(pool_metrics, 'resource_leak_count', 0)
            metrics["resource_leaks"] = leak_count

        except ImportError:
            logger.debug("psutil not available for resource metrics")
        except Exception as e:
            logger.warning(f"Failed to collect resource metrics: {e}")

        return metrics

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect current performance metrics."""
        metrics = {}

        try:
            import psutil
            process = psutil.Process()

            # CPU metrics
            metrics["cpu_percent"] = process.cpu_percent()

            # Calculate thread pool performance
            total_throughput = 0.0
            for pool_metrics in self._thread_pool_validator._pool_metrics.values():
                total_throughput += getattr(pool_metrics, 'throughput_tasks_per_second', 0.0)
            metrics["throughput_tasks_per_second"] = total_throughput

        except ImportError:
            logger.debug("psutil not available for performance metrics")
        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")

        return metrics

    def _calculate_thread_utilization(self, pool_metrics: ThreadPoolHealthMetrics) -> float:
        """Calculate thread pool utilization percentage."""
        if pool_metrics.max_threads == 0:
            return 0.0
        return (pool_metrics.active_threads / pool_metrics.max_threads) * 100.0

    def _trigger_alert(self, severity: AlertSeverity, component: str,
                      message: str, metrics: Dict[str, Any]):
        """Trigger a health alert."""
        alert_key = f"{component}_{hash(message) % 10000}"

        # Check if this alert is already active
        if alert_key in self._active_alerts:
            return

        # Create alert
        alert = HealthAlert(
            timestamp=time.perf_counter(),
            severity=severity,
            component=component,
            message=message,
            metrics=metrics
        )

        # Store alert
        self._active_alerts[alert_key] = alert
        self._alert_history.append(alert)

        # Log alert
        if self.config.enable_console_alerts:
            log_level = {
                AlertSeverity.INFO: logger.info,
                AlertSeverity.WARNING: logger.warning,
                AlertSeverity.ERROR: logger.error,
                AlertSeverity.CRITICAL: logger.critical
            }[severity]
            log_level(alert.format_message())

        # Emit signal
        self.alert_triggered.emit(alert)

        # Attempt auto-recovery if enabled
        if self.config.enable_auto_recovery:
            self._attempt_auto_recovery(alert)

    def _attempt_auto_recovery(self, alert: HealthAlert):
        """Attempt automatic recovery for the alert."""
        try:
            recovered = False

            if alert.component.startswith("thread_pool"):
                # Thread pool recovery
                recovered = self._recover_thread_pool_issues(alert)
            elif alert.component == "qt_compliance":
                # Qt compliance recovery
                recovered = self._recover_qt_compliance_issues(alert)
            elif alert.component == "resources":
                # Resource recovery
                recovered = self._recover_resource_issues(alert)

            if recovered:
                self._resolve_alert(alert)
                logger.info(f"Auto-recovery successful for: {alert.message}")

        except Exception as e:
            logger.error(f"Auto-recovery failed for {alert.component}: {e}")

    def _recover_thread_pool_issues(self, alert: HealthAlert) -> bool:
        """Attempt to recover thread pool issues."""
        try:
            # Basic thread pool recovery - cleanup finished threads
            self._thread_manager.cleanup_finished_threads()

            # Force garbage collection
            import gc
            gc.collect()

            return True
        except Exception as e:
            logger.error(f"Thread pool recovery failed: {e}")
            return False

    def _recover_qt_compliance_issues(self, alert: HealthAlert) -> bool:
        """Attempt to recover Qt compliance issues."""
        try:
            # Log Qt compliance issue for investigation
            logger.warning(f"Qt compliance issue detected: {alert.message}")
            # For now, just log - more sophisticated recovery could be added
            return False
        except Exception as e:
            logger.error(f"Qt compliance recovery failed: {e}")
            return False

    def _recover_resource_issues(self, alert: HealthAlert) -> bool:
        """Attempt to recover resource issues."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Request memory cleanup from thread pools
            for pool_id in self._thread_pool_validator._thread_pools:
                # Could add specific cleanup methods here
                pass

            return True
        except Exception as e:
            logger.error(f"Resource recovery failed: {e}")
            return False

    def _resolve_alert(self, alert: HealthAlert):
        """Mark an alert as resolved."""
        alert.resolved = True
        alert.resolution_time = time.perf_counter()

        # Remove from active alerts
        for key, active_alert in list(self._active_alerts.items()):
            if active_alert is alert:
                del self._active_alerts[key]
                break

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "is_running": self._is_running,
            "active_alerts": len(self._active_alerts),
            "total_alerts": len(self._alert_history),
            "last_check_times": dict(self._last_check_times),
            "component_status": dict(self._component_health_status)
        }

    def get_alert_history(self, severity: Optional[AlertSeverity] = None,
                         hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get alert history with optional filtering."""
        cutoff_time = None
        if hours:
            cutoff_time = time.perf_counter() - (hours * 3600)

        filtered_alerts = []
        for alert in self._alert_history:
            # Filter by time
            if cutoff_time and alert.timestamp < cutoff_time:
                continue

            # Filter by severity
            if severity and alert.severity != severity:
                continue

            filtered_alerts.append(alert.to_dict())

        return filtered_alerts

    def get_metrics_history(self, component: str, hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get metrics history for a component."""
        if component not in self._metrics_history:
            return []

        cutoff_time = None
        if hours:
            cutoff_time = time.perf_counter() - (hours * 3600)

        filtered_metrics = []
        for metric in self._metrics_history[component]:
            if cutoff_time and metric["timestamp"] < cutoff_time:
                continue
            filtered_metrics.append(metric)

        return filtered_metrics

    def cleanup_old_data(self):
        """Clean up old monitoring data."""
        cutoff_time = time.perf_counter() - (self.config.history_retention_days * 24 * 3600)

        # Clean up alert history
        self._alert_history = deque(
            [alert for alert in self._alert_history if alert.timestamp > cutoff_time],
            maxlen=self.config.max_alert_history
        )

        # Clean up metrics history
        for component in self._metrics_history:
            self._metrics_history[component] = deque(
                [metric for metric in self._metrics_history[component]
                 if metric["timestamp"] > cutoff_time],
                maxlen=self.config.max_metrics_history
            )


# Global instance
_health_monitor_daemon: Optional[HealthMonitorDaemon] = None


def get_health_monitor_daemon(config: Optional[MonitoringConfiguration] = None) -> HealthMonitorDaemon:
    """Get the global health monitoring daemon instance."""
    global _health_monitor_daemon

    if _health_monitor_daemon is None:
        _health_monitor_daemon = HealthMonitorDaemon(config)

    return _health_monitor_daemon


def initialize_health_monitoring(config: Optional[MonitoringConfiguration] = None) -> HealthMonitorDaemon:
    """Initialize and start health monitoring."""
    daemon = get_health_monitor_daemon(config)
    daemon.start_monitoring()
    return daemon


def shutdown_health_monitoring():
    """Shutdown health monitoring."""
    global _health_monitor_daemon

    if _health_monitor_daemon:
        _health_monitor_daemon.stop_monitoring()
        _health_monitor_daemon = None


@contextmanager
def health_monitoring_context(config: Optional[MonitoringConfiguration] = None):
    """Context manager for health monitoring."""
    daemon = initialize_health_monitoring(config)
    try:
        yield daemon
    finally:
        shutdown_health_monitoring()