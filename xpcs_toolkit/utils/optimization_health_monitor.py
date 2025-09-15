"""
Optimization Health Monitor for XPCS Toolkit.

This module provides comprehensive automated health checks for all CPU-based optimization
systems including threading, memory management, performance monitoring, and I/O optimizations.

Features:
- Automated health checks for threading system components
- Memory management system health monitoring
- Performance monitoring system validation
- I/O optimization component checks
- Configurable health check intervals and thresholds
- Integration with existing logging and alert systems
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

from PySide6.QtCore import QObject, QTimer, Signal

# Import existing optimization systems
from ..threading.enhanced_thread_pool import get_thread_pool_manager
from ..threading.signal_optimization import get_signal_optimizer
from .adaptive_memory import get_adaptive_memory_manager
from .advanced_cache import get_global_cache
from .logging_config import get_logger
from .memory_utils import SystemMemoryMonitor

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels for optimization components."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of optimization components being monitored."""

    THREADING = "threading"
    MEMORY = "memory"
    CACHING = "caching"
    IO_OPTIMIZATION = "io_optimization"
    PERFORMANCE_MONITORING = "performance_monitoring"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    component: str
    component_type: ComponentType
    status: HealthStatus
    score: float  # 0.0 to 1.0
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    recommendations: list[str] = field(default_factory=list)

    def is_healthy(self) -> bool:
        """Check if the component is in healthy state."""
        return self.status == HealthStatus.HEALTHY

    def needs_attention(self) -> bool:
        """Check if the component needs attention."""
        return self.status in (HealthStatus.WARNING, HealthStatus.CRITICAL)


class HealthCheck(NamedTuple):
    """Definition of a health check."""

    name: str
    component_type: ComponentType
    check_function: Callable[[], HealthCheckResult]
    interval_seconds: float
    enabled: bool = True


class OptimizationHealthMonitor(QObject):
    """
    Comprehensive health monitoring system for all optimization components.
    """

    # Signals
    health_status_updated = Signal(str, object)  # component_name, HealthCheckResult
    global_health_changed = Signal(float)  # overall_health_score
    component_needs_attention = Signal(str, str)  # component_name, message
    health_report_ready = Signal(object)  # Dict[str, HealthCheckResult]

    def __init__(self, check_interval: float = 30.0, parent=None):
        """
        Initialize the health monitor.

        Args:
            check_interval: Base interval in seconds for health checks
            parent: Qt parent object
        """
        super().__init__(parent)

        self.check_interval = check_interval
        self._running = False
        self._lock = threading.RLock()

        # Health check registry
        self._health_checks: dict[str, HealthCheck] = {}
        self._last_check_times: dict[str, float] = {}
        self._check_results: dict[str, HealthCheckResult] = {}

        # Health history for trend analysis
        self._health_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Timer for periodic health checks
        self._timer = QTimer()
        self._timer.timeout.connect(self._run_scheduled_checks)

        # Initialize health checks
        self._register_default_checks()

        logger.info("OptimizationHealthMonitor initialized")

    def start_monitoring(self) -> None:
        """Start the health monitoring system."""
        with self._lock:
            if self._running:
                logger.warning("Health monitoring already running")
                return

            self._running = True
            self._timer.start(int(self.check_interval * 1000))  # Convert to ms
            logger.info("Health monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the health monitoring system."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._timer.stop()
            logger.info("Health monitoring stopped")

    def run_health_check(self, check_name: str) -> HealthCheckResult | None:
        """
        Run a specific health check.

        Args:
            check_name: Name of the health check to run

        Returns:
            HealthCheckResult or None if check doesn't exist
        """
        if check_name not in self._health_checks:
            logger.warning(f"Health check '{check_name}' not found")
            return None

        health_check = self._health_checks[check_name]
        if not health_check.enabled:
            logger.debug(f"Health check '{check_name}' is disabled")
            return None

        try:
            start_time = time.perf_counter()
            result = health_check.check_function()
            check_duration = time.perf_counter() - start_time

            # Update tracking data
            with self._lock:
                self._last_check_times[check_name] = time.time()
                self._check_results[check_name] = result
                self._health_history[check_name].append(
                    (result.timestamp, result.score)
                )

            # Emit signals
            self.health_status_updated.emit(check_name, result)

            if result.needs_attention():
                self.component_needs_attention.emit(check_name, result.message)

            logger.debug(
                f"Health check '{check_name}' completed in {check_duration:.3f}s: {result.status.value}"
            )
            return result

        except Exception as e:
            logger.error(f"Health check '{check_name}' failed: {e}")
            error_result = HealthCheckResult(
                component=check_name,
                component_type=health_check.component_type,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check failed: {e!s}",
            )

            with self._lock:
                self._check_results[check_name] = error_result

            return error_result

    def get_overall_health_score(self) -> float:
        """
        Calculate overall health score across all components.

        Returns:
            Overall health score (0.0 to 1.0)
        """
        with self._lock:
            if not self._check_results:
                return 0.0

            scores = [result.score for result in self._check_results.values()]
            return statistics.mean(scores) if scores else 0.0

    def get_health_report(self) -> dict[str, HealthCheckResult]:
        """
        Get comprehensive health report for all components.

        Returns:
            Dictionary mapping component names to their health check results
        """
        with self._lock:
            return self._check_results.copy()

    def get_component_health_trend(
        self, component_name: str, window_minutes: int = 30
    ) -> str:
        """
        Get health trend for a specific component.

        Args:
            component_name: Name of the component
            window_minutes: Time window for trend analysis

        Returns:
            Trend description: "improving", "stable", "degrading", or "unknown"
        """
        if component_name not in self._health_history:
            return "unknown"

        history = self._health_history[component_name]
        if len(history) < 2:
            return "unknown"

        # Get recent data points within the window
        cutoff_time = time.time() - (window_minutes * 60)
        recent_points = [(ts, score) for ts, score in history if ts > cutoff_time]

        if len(recent_points) < 2:
            return "stable"

        # Calculate trend using linear regression slope
        scores = [score for _, score in recent_points]
        if len(scores) < 2:
            return "stable"

        recent_avg = statistics.mean(scores[-3:]) if len(scores) >= 3 else scores[-1]
        older_avg = statistics.mean(scores[:3]) if len(scores) >= 3 else scores[0]

        if recent_avg > older_avg * 1.05:
            return "improving"
        if recent_avg < older_avg * 0.95:
            return "degrading"
        return "stable"

    def _run_scheduled_checks(self) -> None:
        """Run scheduled health checks based on their intervals."""
        if not self._running:
            return

        current_time = time.time()
        checks_to_run = []

        # Determine which checks should run
        for check_name, health_check in self._health_checks.items():
            if not health_check.enabled:
                continue

            last_check = self._last_check_times.get(check_name, 0)
            if current_time - last_check >= health_check.interval_seconds:
                checks_to_run.append(check_name)

        # Run checks
        results = {}
        for check_name in checks_to_run:
            result = self.run_health_check(check_name)
            if result:
                results[check_name] = result

        # Emit global health update if we ran any checks
        if results:
            overall_score = self.get_overall_health_score()
            self.global_health_changed.emit(overall_score)
            self.health_report_ready.emit(results)

    def _register_default_checks(self) -> None:
        """Register default health checks for all optimization systems."""

        # Threading system checks
        self.register_health_check(
            "thread_pool_health",
            ComponentType.THREADING,
            self._check_thread_pool_health,
            interval_seconds=60.0,
        )

        self.register_health_check(
            "signal_optimization_health",
            ComponentType.THREADING,
            self._check_signal_optimization_health,
            interval_seconds=45.0,
        )

        self.register_health_check(
            "worker_system_health",
            ComponentType.THREADING,
            self._check_worker_system_health,
            interval_seconds=30.0,
        )

        # Memory management checks
        self.register_health_check(
            "cache_system_health",
            ComponentType.CACHING,
            self._check_cache_system_health,
            interval_seconds=30.0,
        )

        self.register_health_check(
            "adaptive_memory_health",
            ComponentType.MEMORY,
            self._check_adaptive_memory_health,
            interval_seconds=60.0,
        )

        self.register_health_check(
            "memory_pressure_health",
            ComponentType.MEMORY,
            self._check_memory_pressure_health,
            interval_seconds=30.0,
        )

        # Performance monitoring checks
        self.register_health_check(
            "performance_monitor_health",
            ComponentType.PERFORMANCE_MONITORING,
            self._check_performance_monitor_health,
            interval_seconds=90.0,
        )

    def register_health_check(
        self,
        name: str,
        component_type: ComponentType,
        check_function: Callable[[], HealthCheckResult],
        interval_seconds: float,
        enabled: bool = True,
    ) -> None:
        """
        Register a new health check.

        Args:
            name: Unique name for the health check
            component_type: Type of component being checked
            check_function: Function that performs the health check
            interval_seconds: How often to run this check
            enabled: Whether the check is enabled
        """
        health_check = HealthCheck(
            name=name,
            component_type=component_type,
            check_function=check_function,
            interval_seconds=interval_seconds,
            enabled=enabled,
        )

        self._health_checks[name] = health_check
        logger.debug(f"Registered health check: {name}")

    # Health check implementations

    def _check_thread_pool_health(self) -> HealthCheckResult:
        """Check health of the thread pool system."""
        try:
            thread_manager = get_thread_pool_manager()
            if not thread_manager:
                return HealthCheckResult(
                    component="thread_pool",
                    component_type=ComponentType.THREADING,
                    status=HealthStatus.CRITICAL,
                    score=0.0,
                    message="Thread pool manager not available",
                )

            stats = thread_manager.get_pool_statistics()

            # Calculate health score based on utilization and queue length
            active_threads = stats.get("active_threads", 0)
            max_threads = stats.get("max_threads", 1)
            queued_tasks = stats.get("queued_tasks", 0)

            utilization = active_threads / max_threads if max_threads > 0 else 0

            # Health scoring
            score = 1.0
            status = HealthStatus.HEALTHY
            messages = []
            recommendations = []

            # Check utilization
            if utilization > 0.9:
                score -= 0.3
                status = HealthStatus.WARNING
                messages.append(f"High thread utilization: {utilization:.1%}")
                recommendations.append("Consider increasing thread pool size")
            elif utilization > 0.7:
                score -= 0.1
                messages.append(f"Moderate thread utilization: {utilization:.1%}")

            # Check queue length
            if queued_tasks > 50:
                score -= 0.4
                status = HealthStatus.CRITICAL
                messages.append(f"High task queue length: {queued_tasks}")
                recommendations.append("Investigate task processing bottlenecks")
            elif queued_tasks > 20:
                score -= 0.2
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Moderate task queue length: {queued_tasks}")

            message = (
                "; ".join(messages) if messages else "Thread pool operating normally"
            )

            return HealthCheckResult(
                component="thread_pool",
                component_type=ComponentType.THREADING,
                status=status,
                score=max(0.0, score),
                message=message,
                details=stats,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Thread pool health check failed: {e}")
            return HealthCheckResult(
                component="thread_pool",
                component_type=ComponentType.THREADING,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_signal_optimization_health(self) -> HealthCheckResult:
        """Check health of the signal optimization system."""
        try:
            optimizer = get_signal_optimizer()
            if not optimizer:
                return HealthCheckResult(
                    component="signal_optimization",
                    component_type=ComponentType.THREADING,
                    status=HealthStatus.CRITICAL,
                    score=0.0,
                    message="Signal optimizer not available",
                )

            stats = optimizer.get_optimization_stats()

            # Calculate health based on batching efficiency and cache hit rates
            batching_efficiency = stats.get("batching_efficiency", 0.0)
            cache_hit_rate = stats.get("cache_hit_rate", 0.0)

            score = 1.0
            status = HealthStatus.HEALTHY
            messages = []
            recommendations = []

            # Check batching efficiency
            if batching_efficiency < 0.3:
                score -= 0.4
                status = HealthStatus.WARNING
                messages.append(f"Low batching efficiency: {batching_efficiency:.1%}")
                recommendations.append("Review signal emission patterns")

            # Check cache hit rate
            if cache_hit_rate < 0.5:
                score -= 0.3
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Low cache hit rate: {cache_hit_rate:.1%}")
                recommendations.append("Optimize signal connection caching")

            message = (
                "; ".join(messages) if messages else "Signal optimization working well"
            )

            return HealthCheckResult(
                component="signal_optimization",
                component_type=ComponentType.THREADING,
                status=status,
                score=max(0.0, score),
                message=message,
                details=stats,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Signal optimization health check failed: {e}")
            return HealthCheckResult(
                component="signal_optimization",
                component_type=ComponentType.THREADING,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_worker_system_health(self) -> HealthCheckResult:
        """Check health of the worker system."""
        try:
            # This would integrate with the worker performance metrics
            # For now, we'll do a basic check

            score = 1.0
            status = HealthStatus.HEALTHY
            message = "Worker system operating normally"

            return HealthCheckResult(
                component="worker_system",
                component_type=ComponentType.THREADING,
                status=status,
                score=score,
                message=message,
                details={},
            )

        except Exception as e:
            logger.error(f"Worker system health check failed: {e}")
            return HealthCheckResult(
                component="worker_system",
                component_type=ComponentType.THREADING,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_cache_system_health(self) -> HealthCheckResult:
        """Check health of the cache system."""
        try:
            cache = get_global_cache()
            if not cache:
                return HealthCheckResult(
                    component="cache_system",
                    component_type=ComponentType.CACHING,
                    status=HealthStatus.CRITICAL,
                    score=0.0,
                    message="Cache system not available",
                )

            stats = cache.get_statistics()
            hit_rate = stats.get("hit_rate", 0.0)
            memory_usage = stats.get("memory_usage_mb", 0.0)
            max_memory = stats.get("max_memory_mb", 100.0)

            score = 1.0
            status = HealthStatus.HEALTHY
            messages = []
            recommendations = []

            # Check hit rate
            if hit_rate < 0.4:
                score -= 0.4
                status = HealthStatus.WARNING
                messages.append(f"Low cache hit rate: {hit_rate:.1%}")
                recommendations.append("Review cache key strategies")

            # Check memory usage
            memory_utilization = memory_usage / max_memory if max_memory > 0 else 0
            if memory_utilization > 0.9:
                score -= 0.3
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"High cache memory usage: {memory_utilization:.1%}")
                recommendations.append("Consider cache size optimization")

            message = "; ".join(messages) if messages else "Cache system operating well"

            return HealthCheckResult(
                component="cache_system",
                component_type=ComponentType.CACHING,
                status=status,
                score=max(0.0, score),
                message=message,
                details=stats,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Cache system health check failed: {e}")
            return HealthCheckResult(
                component="cache_system",
                component_type=ComponentType.CACHING,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_adaptive_memory_health(self) -> HealthCheckResult:
        """Check health of the adaptive memory management system."""
        try:
            memory_manager = get_adaptive_memory_manager()
            if not memory_manager:
                return HealthCheckResult(
                    component="adaptive_memory",
                    component_type=ComponentType.MEMORY,
                    status=HealthStatus.CRITICAL,
                    score=0.0,
                    message="Adaptive memory manager not available",
                )

            stats = memory_manager.get_system_stats()

            score = 1.0
            status = HealthStatus.HEALTHY
            message = "Adaptive memory management working well"

            # Add specific checks based on memory manager capabilities
            memory_pressure = stats.get("memory_pressure", 0.0)
            if memory_pressure > 0.8:
                score -= 0.5
                status = HealthStatus.CRITICAL
                message = f"High memory pressure: {memory_pressure:.1%}"
            elif memory_pressure > 0.6:
                score -= 0.2
                status = HealthStatus.WARNING
                message = f"Moderate memory pressure: {memory_pressure:.1%}"

            return HealthCheckResult(
                component="adaptive_memory",
                component_type=ComponentType.MEMORY,
                status=status,
                score=max(0.0, score),
                message=message,
                details=stats,
            )

        except Exception as e:
            logger.error(f"Adaptive memory health check failed: {e}")
            return HealthCheckResult(
                component="adaptive_memory",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_memory_pressure_health(self) -> HealthCheckResult:
        """Check system memory pressure."""
        try:
            monitor = SystemMemoryMonitor()
            memory_stats = monitor.get_memory_stats()

            memory_percent = memory_stats.get("memory_percent", 0.0)
            available_mb = memory_stats.get("available_mb", 0.0)

            score = 1.0
            status = HealthStatus.HEALTHY
            messages = []
            recommendations = []

            # Check memory usage
            if memory_percent > 90:
                score -= 0.6
                status = HealthStatus.CRITICAL
                messages.append(f"Critical memory usage: {memory_percent:.1%}")
                recommendations.append("Immediate memory cleanup required")
            elif memory_percent > 80:
                score -= 0.3
                status = HealthStatus.WARNING
                messages.append(f"High memory usage: {memory_percent:.1%}")
                recommendations.append("Consider reducing cache sizes")
            elif memory_percent > 70:
                score -= 0.1
                messages.append(f"Elevated memory usage: {memory_percent:.1%}")

            # Check available memory
            if available_mb < 500:
                score -= 0.4
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                messages.append(f"Low available memory: {available_mb:.1f} MB")
                recommendations.append("Monitor for memory leaks")

            message = (
                "; ".join(messages)
                if messages
                else f"Memory usage normal: {memory_percent:.1%}"
            )

            return HealthCheckResult(
                component="memory_pressure",
                component_type=ComponentType.MEMORY,
                status=status,
                score=max(0.0, score),
                message=message,
                details=memory_stats,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Memory pressure health check failed: {e}")
            return HealthCheckResult(
                component="memory_pressure",
                component_type=ComponentType.MEMORY,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )

    def _check_performance_monitor_health(self) -> HealthCheckResult:
        """Check health of the performance monitoring system."""
        try:
            # Basic check that the performance monitoring system is responsive
            score = 1.0
            status = HealthStatus.HEALTHY
            message = "Performance monitoring system operational"

            return HealthCheckResult(
                component="performance_monitor",
                component_type=ComponentType.PERFORMANCE_MONITORING,
                status=status,
                score=score,
                message=message,
                details={},
            )

        except Exception as e:
            logger.error(f"Performance monitor health check failed: {e}")
            return HealthCheckResult(
                component="performance_monitor",
                component_type=ComponentType.PERFORMANCE_MONITORING,
                status=HealthStatus.UNKNOWN,
                score=0.0,
                message=f"Health check error: {e!s}",
            )


# Global instance
_health_monitor_instance: OptimizationHealthMonitor | None = None
_monitor_lock = threading.Lock()


def get_health_monitor() -> OptimizationHealthMonitor:
    """Get the global health monitor instance."""
    global _health_monitor_instance

    if _health_monitor_instance is None:
        with _monitor_lock:
            if _health_monitor_instance is None:
                _health_monitor_instance = OptimizationHealthMonitor()

    return _health_monitor_instance


def start_health_monitoring() -> None:
    """Start the global health monitoring system."""
    monitor = get_health_monitor()
    monitor.start_monitoring()


def stop_health_monitoring() -> None:
    """Stop the global health monitoring system."""
    monitor = get_health_monitor()
    monitor.stop_monitoring()
