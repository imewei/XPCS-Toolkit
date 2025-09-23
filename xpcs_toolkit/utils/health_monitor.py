"""
Background Health Monitoring System for XPCS Toolkit.

This module provides non-intrusive background monitoring using existing thread pools
to track system health, resource usage, and reliability metrics without impacting
performance of core operations.
"""

import gc
import os
import threading
import time
import weakref
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import psutil

from .logging_config import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Overall system health status levels."""

    EXCELLENT = "excellent"  # All systems optimal
    GOOD = "good"  # Minor issues, no impact
    WARNING = "warning"  # Issues detected, monitoring
    CRITICAL = "critical"  # Immediate attention required
    EMERGENCY = "emergency"  # System stability at risk


class ResourceType(Enum):
    """Types of system resources to monitor."""

    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    THREADS = "threads"
    HDF5_CONNECTIONS = "hdf5_connections"
    GUI_RESPONSIVENESS = "gui_responsiveness"


@dataclass
class HealthMetric:
    """Individual health metric with thresholds and history."""

    name: str
    current_value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: float = field(default_factory=time.time)

    def update(self, value: float) -> None:
        """Update metric value and history."""
        self.current_value = value
        self.history.append((time.time(), value))
        self.last_updated = time.time()

    def get_status(self) -> HealthStatus:
        """Get current status based on thresholds."""
        if self.current_value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        if self.current_value >= self.threshold_warning:
            return HealthStatus.WARNING
        return HealthStatus.GOOD

    def get_trend(self, window_minutes: float = 5.0) -> str:
        """Get trend over specified time window."""
        if len(self.history) < 2:
            return "insufficient_data"

        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [val for ts, val in self.history if ts >= cutoff_time]

        if len(recent_values) < 2:
            return "insufficient_data"

        # Simple linear trend
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        if abs(trend) < 0.01:  # Threshold for "stable"
            return "stable"
        if trend > 0:
            return "increasing"
        return "decreasing"


class HealthMonitor:
    """
    Non-intrusive health monitoring system using background threads.

    Monitors system resources, application state, and reliability metrics
    without impacting performance of core XPCS operations.
    """

    def __init__(self, monitoring_interval: float = 30.0, history_size: int = 100):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self._metrics: dict[str, HealthMetric] = {}
        self._monitoring_active = False
        self._monitor_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._callbacks: dict[HealthStatus, list[Callable]] = defaultdict(list)
        self._last_alert_times: dict[str, float] = {}
        self._alert_cooldown = 300.0  # 5 minutes between same alerts

        # Track application objects for health monitoring
        self._tracked_objects: set[weakref.ref] = set()

        # Initialize core metrics
        self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize core health metrics with appropriate thresholds."""
        self._metrics = {
            "memory_usage_percent": HealthMetric(
                name="Memory Usage",
                current_value=0.0,
                threshold_warning=75.0,
                threshold_critical=90.0,
                unit="%",
            ),
            "memory_available_gb": HealthMetric(
                name="Available Memory",
                current_value=0.0,
                threshold_warning=2.0,  # < 2GB available
                threshold_critical=0.5,  # < 500MB available
                unit="GB",
            ),
            "cpu_usage_percent": HealthMetric(
                name="CPU Usage",
                current_value=0.0,
                threshold_warning=80.0,
                threshold_critical=95.0,
                unit="%",
            ),
            "thread_count": HealthMetric(
                name="Thread Count",
                current_value=0.0,
                threshold_warning=100,
                threshold_critical=200,
                unit="",
            ),
            "hdf5_connections": HealthMetric(
                name="HDF5 Connections",
                current_value=0.0,
                threshold_warning=50,
                threshold_critical=100,
                unit="",
            ),
            "disk_usage_percent": HealthMetric(
                name="Disk Usage",
                current_value=0.0,
                threshold_warning=85.0,
                threshold_critical=95.0,
                unit="%",
            ),
            "gc_collections_per_hour": HealthMetric(
                name="GC Collections/Hour",
                current_value=0.0,
                threshold_warning=100,
                threshold_critical=300,
                unit="",
            ),
        }

    def start_monitoring(self) -> None:
        """Start background health monitoring."""
        import os

        # Skip starting background threads in test mode to prevent threading issues
        if os.environ.get("XPCS_TEST_MODE") == "1":
            return

        with self._lock:
            if self._monitoring_active:
                logger.debug("Health monitoring already active")
                return

            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop, name="XPCS-HealthMonitor", daemon=True
            )
            self._monitor_thread.start()
            logger.info(
                f"Health monitoring started (interval: {self.monitoring_interval}s)"
            )

    def stop_monitoring(self) -> None:
        """Stop background health monitoring."""
        with self._lock:
            if not self._monitoring_active:
                return

            self._monitoring_active = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

            logger.info("Health monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.debug("Health monitoring loop started")

        # Track GC statistics
        gc_stats_start = gc.get_stats()

        while self._monitoring_active:
            try:
                start_time = time.time()

                # Update all metrics
                self._update_system_metrics()
                self._update_application_metrics()
                self._update_gc_metrics(gc_stats_start)

                # Check for alerts
                self._check_health_alerts()

                # Clean up dead weak references
                self._cleanup_tracked_objects()

                # Sleep for remaining interval time
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                # Don't let monitoring errors crash the monitor
                logger.debug(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval)

        logger.debug("Health monitoring loop ended")

    def _update_system_metrics(self) -> None:
        """Update system-level metrics using psutil."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            self._metrics["memory_usage_percent"].update(memory.percent)
            self._metrics["memory_available_gb"].update(memory.available / (1024**3))

            # CPU metrics (average over short period to avoid blocking)
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._metrics["cpu_usage_percent"].update(cpu_percent)

            # Thread count
            process = psutil.Process()
            thread_count = process.num_threads()
            self._metrics["thread_count"].update(thread_count)

            # Disk usage for current working directory
            disk_usage = psutil.disk_usage(os.getcwd())
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            self._metrics["disk_usage_percent"].update(disk_percent)

        except Exception as e:
            logger.debug(f"Error updating system metrics: {e}")

    def _update_application_metrics(self) -> None:
        """Update application-specific metrics."""
        try:
            # HDF5 connection count (if available)
            hdf5_count = self._get_hdf5_connection_count()
            self._metrics["hdf5_connections"].update(hdf5_count)

        except Exception as e:
            logger.debug(f"Error updating application metrics: {e}")

    def _update_gc_metrics(self, initial_stats: list[dict]) -> None:
        """Update garbage collection metrics."""
        try:
            current_stats = gc.get_stats()
            if len(current_stats) == len(initial_stats):
                # Calculate GC collections per hour
                total_collections = sum(
                    current_stats[i]["collections"] - initial_stats[i]["collections"]
                    for i in range(len(current_stats))
                )

                # Convert to collections per hour
                elapsed_hours = (
                    time.time() - getattr(self, "_monitor_start_time", time.time())
                ) / 3600
                if elapsed_hours > 0:
                    collections_per_hour = total_collections / elapsed_hours
                    self._metrics["gc_collections_per_hour"].update(
                        collections_per_hour
                    )

        except Exception as e:
            logger.debug(f"Error updating GC metrics: {e}")

    def _get_hdf5_connection_count(self) -> float:
        """Get current HDF5 connection count."""
        try:
            # Try to get connection count from HDF5 reader
            from ..fileIO.hdf_reader import _connection_pool

            if hasattr(_connection_pool, "get_pool_size"):
                return float(_connection_pool.get_pool_size())
            if hasattr(_connection_pool, "_pool_size"):
                return float(_connection_pool._pool_size)
        except (ImportError, AttributeError):
            pass

        return 0.0

    def _check_health_alerts(self) -> None:
        """Check metrics and trigger alerts if thresholds exceeded."""
        current_time = time.time()
        overall_status = HealthStatus.EXCELLENT

        for metric_name, metric in self._metrics.items():
            status = metric.get_status()

            # Update overall status to worst individual status
            if status.value == "critical":
                overall_status = HealthStatus.CRITICAL
            elif status.value == "warning" and overall_status.value in [
                "excellent",
                "good",
            ]:
                overall_status = HealthStatus.WARNING
            elif status.value == "good" and overall_status.value == "excellent":
                overall_status = HealthStatus.GOOD

            # Check if we should send alert
            alert_key = f"{metric_name}_{status.value}"
            last_alert = self._last_alert_times.get(alert_key, 0)

            if (
                status in [HealthStatus.WARNING, HealthStatus.CRITICAL]
                and current_time - last_alert > self._alert_cooldown
            ):
                self._trigger_alert(metric_name, metric, status)
                self._last_alert_times[alert_key] = current_time

        # Trigger overall status callbacks
        self._trigger_status_callbacks(overall_status)

    def _trigger_alert(
        self, metric_name: str, metric: HealthMetric, status: HealthStatus
    ) -> None:
        """Trigger alert for specific metric."""
        trend = metric.get_trend()
        message = (
            f"Health Alert [{status.value.upper()}]: {metric.name} = "
            f"{metric.current_value:.1f}{metric.unit} "
            f"(threshold: {metric.threshold_warning if status == HealthStatus.WARNING else metric.threshold_critical:.1f}{metric.unit}, "
            f"trend: {trend})"
        )

        if status == HealthStatus.CRITICAL:
            logger.error(message)
        else:
            logger.warning(message)

        # Trigger specific actions based on metric and status
        self._handle_metric_alert(metric_name, metric, status)

    def _handle_metric_alert(
        self, metric_name: str, metric: HealthMetric, status: HealthStatus
    ) -> None:
        """Handle specific metric alerts with automatic recovery actions."""
        if metric_name == "memory_usage_percent" and status == HealthStatus.CRITICAL:
            # Trigger emergency memory cleanup
            logger.warning(
                "Triggering emergency memory cleanup due to high memory usage"
            )
            self._emergency_memory_cleanup()

        elif metric_name == "hdf5_connections" and status == HealthStatus.WARNING:
            # Clean up HDF5 connections
            logger.warning("Cleaning up HDF5 connections due to high connection count")
            self._cleanup_hdf5_connections()

        elif (
            metric_name == "gc_collections_per_hour" and status == HealthStatus.CRITICAL
        ):
            # Log GC pressure warning
            logger.warning(
                "High garbage collection pressure detected - potential memory leaks"
            )

    def _emergency_memory_cleanup(self) -> None:
        """Perform emergency memory cleanup."""
        try:
            # Force garbage collection
            collected = gc.collect()
            logger.debug(f"Emergency GC freed {collected} objects")

            # Try to trigger cleanup in memory manager if available
            try:
                from .memory_manager import get_memory_manager

                memory_manager = get_memory_manager()
                memory_manager._emergency_cleanup()
                logger.debug("Triggered memory manager emergency cleanup")
            except (ImportError, AttributeError):
                pass

        except Exception as e:
            logger.debug(f"Error during emergency memory cleanup: {e}")

    def _cleanup_hdf5_connections(self) -> None:
        """Clean up HDF5 connections."""
        try:
            from ..fileIO.hdf_reader import _connection_pool

            if hasattr(_connection_pool, "cleanup_idle_connections"):
                _connection_pool.cleanup_idle_connections()
                logger.debug("Cleaned up idle HDF5 connections")
        except (ImportError, AttributeError):
            pass

    def _trigger_status_callbacks(self, status: HealthStatus) -> None:
        """Trigger registered callbacks for overall status."""
        callbacks = self._callbacks.get(status, [])
        for callback in callbacks:
            try:
                callback(status, self.get_health_summary())
            except Exception as e:
                logger.debug(f"Error in health status callback: {e}")

    def _cleanup_tracked_objects(self) -> None:
        """Clean up dead weak references."""
        self._tracked_objects = {
            ref for ref in self._tracked_objects if ref() is not None
        }

    def register_health_callback(
        self, status: HealthStatus, callback: Callable
    ) -> None:
        """Register callback for specific health status."""
        self._callbacks[status].append(callback)

    def track_object(self, obj: Any) -> None:
        """Track object for health monitoring."""
        self._tracked_objects.add(weakref.ref(obj))

    def get_health_summary(self) -> dict[str, Any]:
        """Get comprehensive health summary."""
        with self._lock:
            summary = {
                "overall_status": self._get_overall_status().value,
                "monitoring_active": self._monitoring_active,
                "metrics": {},
                "alerts": [],
                "recommendations": [],
            }

            # Add metric summaries
            for name, metric in self._metrics.items():
                summary["metrics"][name] = {
                    "value": metric.current_value,
                    "unit": metric.unit,
                    "status": metric.get_status().value,
                    "trend": metric.get_trend(),
                    "threshold_warning": metric.threshold_warning,
                    "threshold_critical": metric.threshold_critical,
                }

                # Add to alerts if problematic
                status = metric.get_status()
                if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                    summary["alerts"].append(
                        {
                            "metric": name,
                            "status": status.value,
                            "value": metric.current_value,
                            "message": f"{metric.name} is {status.value}: {metric.current_value:.1f}{metric.unit}",
                        }
                    )

            # Add recommendations
            summary["recommendations"] = self._get_health_recommendations()

            return summary

    def _get_overall_status(self) -> HealthStatus:
        """Calculate overall health status."""
        worst_status = HealthStatus.EXCELLENT

        for metric in self._metrics.values():
            status = metric.get_status()
            if status == HealthStatus.CRITICAL:
                return HealthStatus.CRITICAL
            if status == HealthStatus.WARNING and worst_status != HealthStatus.CRITICAL:
                worst_status = HealthStatus.WARNING
            elif status == HealthStatus.GOOD and worst_status == HealthStatus.EXCELLENT:
                worst_status = HealthStatus.GOOD

        return worst_status

    def _get_health_recommendations(self) -> list[str]:
        """Get health improvement recommendations."""
        recommendations = []

        memory_metric = self._metrics.get("memory_usage_percent")
        if memory_metric and memory_metric.current_value > 70:
            recommendations.append(
                "Consider closing unused data files or reducing dataset sizes"
            )

        cpu_metric = self._metrics.get("cpu_usage_percent")
        if cpu_metric and cpu_metric.current_value > 80:
            recommendations.append(
                "High CPU usage detected - consider reducing parallel operations"
            )

        hdf5_metric = self._metrics.get("hdf5_connections")
        if hdf5_metric and hdf5_metric.current_value > 30:
            recommendations.append(
                "Many HDF5 connections open - consider closing unused files"
            )

        gc_metric = self._metrics.get("gc_collections_per_hour")
        if gc_metric and gc_metric.current_value > 200:
            recommendations.append(
                "High garbage collection rate - potential memory leaks detected"
            )

        return recommendations

    def get_performance_impact(self) -> dict[str, float]:
        """Get monitoring performance impact statistics."""
        return {
            "monitoring_cpu_percent": 0.1,  # Estimated < 0.1% CPU
            "monitoring_memory_mb": 5.0,  # Estimated < 5MB memory
            "monitoring_interval_seconds": self.monitoring_interval,
            "metrics_tracked": len(self._metrics),
            "objects_tracked": len(self._tracked_objects),
        }


# Global health monitor instance
_health_monitor: HealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        with _monitor_lock:
            if _health_monitor is None:
                _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitoring(interval: float = 30.0) -> None:
    """Start background health monitoring."""
    monitor = get_health_monitor()
    monitor.monitoring_interval = interval
    monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop background health monitoring."""
    global _health_monitor
    if _health_monitor:
        _health_monitor.stop_monitoring()


def get_health_status() -> dict[str, Any]:
    """Get current health status summary."""
    monitor = get_health_monitor()
    return monitor.get_health_summary()


def register_health_callback(status: HealthStatus, callback: Callable) -> None:
    """Register callback for specific health status changes."""
    monitor = get_health_monitor()
    monitor.register_health_callback(status, callback)


def track_object_health(obj: Any) -> None:
    """Track object for health monitoring."""
    monitor = get_health_monitor()
    monitor.track_object(obj)


# Context manager for automatic health monitoring
class health_monitoring_context:
    """Context manager for automatic health monitoring during operations."""

    def __init__(self, operation_name: str, alert_on_warning: bool = True):
        self.operation_name = operation_name
        self.alert_on_warning = alert_on_warning
        self.start_metrics = {}

    def __enter__(self):
        # Capture initial metrics
        monitor = get_health_monitor()
        self.start_metrics = {
            name: metric.current_value for name, metric in monitor._metrics.items()
        }
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Check for significant changes
        monitor = get_health_monitor()
        significant_changes = []

        for name, start_value in self.start_metrics.items():
            current_metric = monitor._metrics.get(name)
            if current_metric:
                change_percent = (
                    abs(current_metric.current_value - start_value)
                    / max(start_value, 1.0)
                    * 100
                )
                if change_percent > 10:  # > 10% change
                    significant_changes.append(
                        f"{name}: {start_value:.1f} -> {current_metric.current_value:.1f}"
                    )

        if significant_changes:
            logger.debug(
                f"Operation '{self.operation_name}' caused significant resource changes: {', '.join(significant_changes)}"
            )

        return False  # Don't suppress exceptions
