"""
Threading performance monitoring and analytics system for XPCS Toolkit.

This module provides comprehensive performance monitoring for:
- Signal emission patterns and optimization effectiveness
- Thread pool utilization and health
- Worker execution metrics and bottlenecks
- System resource usage and limits
- Real-time analytics and reporting
"""

from __future__ import annotations

import time
import json
import logging
import statistics
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

# Import our optimization systems
from .signal_optimization import get_signal_optimizer
from .enhanced_thread_pool import get_thread_pool_manager
from .optimized_workers import WorkerPerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class SystemSnapshot:
    """Snapshot of system performance at a point in time."""

    timestamp: float = field(default_factory=time.perf_counter)

    # Signal optimization metrics
    signal_batching_stats: Dict[str, Any] = field(default_factory=dict)
    connection_pool_stats: Dict[str, Any] = field(default_factory=dict)
    attribute_cache_stats: Dict[str, Any] = field(default_factory=dict)

    # Thread pool metrics
    thread_pool_stats: Dict[str, Any] = field(default_factory=dict)

    # System resource metrics
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0

    # Application metrics
    active_workers: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0


@dataclass
class PerformanceTrends:
    """Analysis of performance trends over time."""

    # Signal optimization trends
    avg_signal_batching_efficiency: float = 0.0
    signal_emission_rate_trend: str = "stable"  # "improving", "degrading", "stable"
    cache_hit_rate_trend: str = "stable"

    # Threading trends
    thread_utilization_trend: str = "stable"
    task_throughput_trend: str = "stable"
    avg_task_execution_time: float = 0.0

    # System trends
    resource_usage_trend: str = "stable"
    memory_pressure_events: int = 0

    # Overall assessment
    overall_performance_score: float = 100.0  # 0-100 scale
    bottlenecks_identified: List[str] = field(default_factory=list)
    optimization_recommendations: List[str] = field(default_factory=list)


class PerformanceMonitor(QObject):
    """
    Comprehensive performance monitoring system for threading optimizations.
    """

    # Signals for real-time monitoring
    snapshot_updated = Signal(object)  # SystemSnapshot
    trends_analyzed = Signal(object)  # PerformanceTrends
    performance_alert = Signal(str, str, float)  # alert_type, message, severity
    bottleneck_detected = Signal(str, str)  # bottleneck_type, description

    def __init__(self, monitoring_interval: float = 5.0, parent=None):
        super().__init__(parent)
        self.monitoring_interval = monitoring_interval

        # Historical data storage
        self._snapshots: deque[SystemSnapshot] = deque(
            maxlen=1000
        )  # Keep last 1000 snapshots
        self._worker_metrics: Dict[str, WorkerPerformanceMetrics] = {}

        # Alert thresholds
        self._alert_thresholds = {
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 90.0,
            "thread_utilization": 95.0,
            "cache_hit_rate_min": 70.0,
            "signal_batching_efficiency_min": 60.0,
            "avg_task_execution_time_max": 10.0,
        }

        # Trend analysis parameters
        self._trend_window_size = 20  # Number of snapshots to analyze for trends
        self._trend_threshold = 5.0  # Percentage change threshold for trend detection

        # Performance scoring weights
        self._performance_weights = {
            "signal_optimization": 0.25,
            "thread_efficiency": 0.30,
            "resource_usage": 0.25,
            "task_performance": 0.20,
        }

        # Monitoring timer
        self._monitor_timer = QTimer()
        self._monitor_timer.timeout.connect(self._collect_snapshot)
        self._start_time = time.perf_counter()

        # Analysis timer (less frequent)
        self._analysis_timer = QTimer()
        self._analysis_timer.timeout.connect(self._analyze_trends)

        logger.info(
            f"PerformanceMonitor initialized with {monitoring_interval}s interval"
        )

    def start_monitoring(self):
        """Start performance monitoring."""
        self._monitor_timer.start(int(self.monitoring_interval * 1000))
        self._analysis_timer.start(
            int(self.monitoring_interval * 4 * 1000)
        )  # Analyze every 4 snapshots
        logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitor_timer.stop()
        self._analysis_timer.stop()
        logger.info("Performance monitoring stopped")

    def _collect_snapshot(self):
        """Collect a performance snapshot."""
        try:
            snapshot = SystemSnapshot()

            # Collect signal optimization metrics
            signal_optimizer = get_signal_optimizer()
            signal_stats = signal_optimizer.get_comprehensive_statistics()

            snapshot.signal_batching_stats = signal_stats.get("signal_batching", {})
            snapshot.connection_pool_stats = signal_stats.get("connection_pool", {})
            snapshot.attribute_cache_stats = signal_stats.get("attribute_cache", {})

            # Collect thread pool metrics
            thread_manager = get_thread_pool_manager()
            thread_stats = thread_manager.get_global_statistics()

            snapshot.thread_pool_stats = thread_stats
            snapshot.active_workers = thread_stats.get("threads", {}).get(
                "currently_active", 0
            )
            snapshot.queued_tasks = thread_stats.get("tasks", {}).get("total_queued", 0)

            # Collect system resource metrics
            try:
                import psutil

                snapshot.cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                snapshot.memory_usage_mb = memory.used / (1024 * 1024)
                snapshot.memory_usage_percent = memory.percent
            except ImportError:
                logger.debug("psutil not available for system metrics")

            # Store snapshot
            self._snapshots.append(snapshot)

            # Check for alerts
            self._check_performance_alerts(snapshot)

            # Emit snapshot update
            self.snapshot_updated.emit(snapshot)

        except Exception as e:
            logger.error(f"Error collecting performance snapshot: {e}")

    def _check_performance_alerts(self, snapshot: SystemSnapshot):
        """Check for performance alerts based on thresholds."""

        # Memory usage alert
        if (
            snapshot.memory_usage_percent
            > self._alert_thresholds["memory_usage_percent"]
        ):
            self.performance_alert.emit(
                "memory",
                f"High memory usage: {snapshot.memory_usage_percent:.1f}%",
                snapshot.memory_usage_percent / 100.0,
            )

        # CPU usage alert
        if snapshot.cpu_usage > self._alert_thresholds["cpu_usage_percent"]:
            self.performance_alert.emit(
                "cpu",
                f"High CPU usage: {snapshot.cpu_usage:.1f}%",
                snapshot.cpu_usage / 100.0,
            )

        # Thread utilization alert
        thread_stats = snapshot.thread_pool_stats.get("threads", {})
        thread_stats.get("total_capacity", 1)
        utilization = thread_stats.get("utilization_percent", 0)

        if utilization > self._alert_thresholds["thread_utilization"]:
            self.performance_alert.emit(
                "threading",
                f"High thread utilization: {utilization:.1f}%",
                utilization / 100.0,
            )

        # Cache performance alerts
        cache_stats = snapshot.attribute_cache_stats
        if (
            cache_stats.get("hit_ratio", 0) * 100
            < self._alert_thresholds["cache_hit_rate_min"]
        ):
            self.performance_alert.emit(
                "cache",
                f"Low cache hit rate: {cache_stats.get('hit_ratio', 0) * 100:.1f}%",
                0.7,  # Medium severity
            )

        # Signal batching efficiency alert
        signal_stats = snapshot.signal_batching_stats
        reduction_ratio = signal_stats.get("reduction_ratio", 0) * 100
        if reduction_ratio < self._alert_thresholds["signal_batching_efficiency_min"]:
            self.performance_alert.emit(
                "signals",
                f"Low signal batching efficiency: {reduction_ratio:.1f}%",
                0.5,  # Lower severity
            )

    def _analyze_trends(self):
        """Analyze performance trends over recent snapshots."""
        if len(self._snapshots) < self._trend_window_size:
            return

        try:
            recent_snapshots = list(self._snapshots)[-self._trend_window_size :]
            trends = self._calculate_trends(recent_snapshots)

            # Detect bottlenecks
            bottlenecks = self._detect_bottlenecks(recent_snapshots)
            for bottleneck_type, description in bottlenecks:
                self.bottleneck_detected.emit(bottleneck_type, description)

            trends.bottlenecks_identified = [desc for _, desc in bottlenecks]
            trends.optimization_recommendations = self._generate_recommendations(
                trends, recent_snapshots
            )

            # Calculate overall performance score
            trends.overall_performance_score = self._calculate_performance_score(
                trends, recent_snapshots
            )

            # Emit trends analysis
            self.trends_analyzed.emit(trends)

        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")

    def _calculate_trends(self, snapshots: List[SystemSnapshot]) -> PerformanceTrends:
        """Calculate performance trends from snapshots."""
        trends = PerformanceTrends()

        if len(snapshots) < 5:
            return trends

        # Signal batching efficiency trend
        batching_efficiencies = []
        cache_hit_rates = []
        thread_utilizations = []

        for snapshot in snapshots:
            # Signal batching efficiency
            signal_stats = snapshot.signal_batching_stats
            if signal_stats.get("reduction_ratio"):
                batching_efficiencies.append(signal_stats["reduction_ratio"] * 100)

            # Cache hit rate
            cache_stats = snapshot.attribute_cache_stats
            if cache_stats.get("hit_ratio"):
                cache_hit_rates.append(cache_stats["hit_ratio"] * 100)

            # Thread utilization
            thread_stats = snapshot.thread_pool_stats.get("threads", {})
            if thread_stats.get("utilization_percent"):
                thread_utilizations.append(thread_stats["utilization_percent"])

        # Calculate trend directions
        if batching_efficiencies:
            trends.avg_signal_batching_efficiency = statistics.mean(
                batching_efficiencies
            )
            trends.signal_emission_rate_trend = self._calculate_trend_direction(
                batching_efficiencies
            )

        if cache_hit_rates:
            trends.cache_hit_rate_trend = self._calculate_trend_direction(
                cache_hit_rates
            )

        if thread_utilizations:
            trends.thread_utilization_trend = self._calculate_trend_direction(
                thread_utilizations
            )

        # Calculate average task execution time from worker metrics
        execution_times = [
            metrics.execution_time
            for metrics in self._worker_metrics.values()
            if metrics.execution_time > 0
        ]
        if execution_times:
            trends.avg_task_execution_time = statistics.mean(execution_times)
            trends.task_throughput_trend = (
                "improving" if trends.avg_task_execution_time < 2.0 else "stable"
            )

        return trends

    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 3:
            return "stable"

        # Use linear regression slope to determine trend
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)

        # Determine trend based on slope magnitude
        if abs(slope) < self._trend_threshold / len(values):
            return "stable"
        elif slope > 0:
            return "improving"
        else:
            return "degrading"

    def _detect_bottlenecks(
        self, snapshots: List[SystemSnapshot]
    ) -> List[Tuple[str, str]]:
        """Detect performance bottlenecks from recent snapshots."""
        bottlenecks = []

        if not snapshots:
            return bottlenecks

        latest = snapshots[-1]

        # High memory usage bottleneck
        if latest.memory_usage_percent > 80:
            bottlenecks.append(
                (
                    "memory",
                    f"High memory usage ({latest.memory_usage_percent:.1f}%) may be limiting performance",
                )
            )

        # Thread pool bottleneck
        thread_stats = latest.thread_pool_stats.get("threads", {})
        utilization = thread_stats.get("utilization_percent", 0)
        if utilization > 90:
            bottlenecks.append(
                (
                    "threading",
                    f"Thread pool near capacity ({utilization:.1f}%) - consider increasing thread limits",
                )
            )

        # Queue backlog bottleneck
        if latest.queued_tasks > latest.active_workers * 3:
            bottlenecks.append(
                (
                    "queuing",
                    f"Large task queue ({latest.queued_tasks} tasks) indicates processing bottleneck",
                )
            )

        # Signal batching inefficiency
        signal_stats = latest.signal_batching_stats
        if signal_stats.get("reduction_ratio", 0) < 0.5:
            bottlenecks.append(
                (
                    "signals",
                    "Low signal batching efficiency - many signals not being batched optimally",
                )
            )

        # Cache inefficiency
        cache_stats = latest.attribute_cache_stats
        if cache_stats.get("hit_ratio", 0) < 0.6:
            bottlenecks.append(
                (
                    "cache",
                    f"Low cache hit rate ({cache_stats.get('hit_ratio', 0) * 100:.1f}%) - frequent cache misses",
                )
            )

        return bottlenecks

    def _generate_recommendations(
        self, trends: PerformanceTrends, snapshots: List[SystemSnapshot]
    ) -> List[str]:
        """Generate optimization recommendations based on trends and bottlenecks."""
        recommendations = []

        latest = snapshots[-1] if snapshots else None
        if not latest:
            return recommendations

        # Signal optimization recommendations
        if trends.avg_signal_batching_efficiency < 60:
            recommendations.append(
                "Consider adjusting signal batching parameters to improve batching efficiency"
            )

        # Thread pool recommendations
        thread_stats = latest.thread_pool_stats.get("threads", {})
        utilization = thread_stats.get("utilization_percent", 0)

        if utilization > 85:
            recommendations.append(
                "Consider increasing thread pool size or optimizing task execution time"
            )
        elif utilization < 30:
            recommendations.append(
                "Thread pool may be oversized - consider reducing thread count to save resources"
            )

        # Cache recommendations
        cache_stats = latest.attribute_cache_stats
        hit_rate = cache_stats.get("hit_ratio", 0) * 100

        if hit_rate < 70:
            recommendations.append(
                "Low cache hit rate detected - review cached attributes and TTL settings"
            )

        # Resource usage recommendations
        if latest.memory_usage_percent > 75:
            recommendations.append(
                "High memory usage - consider implementing more aggressive garbage collection"
            )

        # Task performance recommendations
        if trends.avg_task_execution_time > 5.0:
            recommendations.append(
                "Long average task execution time - review task complexity and optimization opportunities"
            )

        return recommendations

    def _calculate_performance_score(
        self, trends: PerformanceTrends, snapshots: List[SystemSnapshot]
    ) -> float:
        """Calculate overall performance score (0-100)."""
        if not snapshots:
            return 100.0

        latest = snapshots[-1]
        scores = {}

        # Signal optimization score (0-100)
        signal_stats = latest.signal_batching_stats
        batching_efficiency = signal_stats.get("reduction_ratio", 0) * 100
        cache_stats = latest.attribute_cache_stats
        cache_hit_rate = cache_stats.get("hit_ratio", 0) * 100

        signal_score = batching_efficiency * 0.6 + cache_hit_rate * 0.4
        scores["signal_optimization"] = min(100, max(0, signal_score))

        # Thread efficiency score
        thread_stats = latest.thread_pool_stats.get("threads", {})
        utilization = thread_stats.get("utilization_percent", 0)

        # Optimal utilization is around 70-80%
        if utilization <= 80:
            thread_score = (utilization / 80) * 100
        else:
            thread_score = max(0, 100 - (utilization - 80) * 2)

        scores["thread_efficiency"] = thread_score

        # Resource usage score
        memory_score = max(0, 100 - latest.memory_usage_percent)
        cpu_score = max(0, 100 - latest.cpu_usage)
        resource_score = (memory_score + cpu_score) / 2
        scores["resource_usage"] = resource_score

        # Task performance score
        if trends.avg_task_execution_time > 0:
            # Score based on execution time (faster = better)
            exec_time_score = max(0, 100 - (trends.avg_task_execution_time * 10))
        else:
            exec_time_score = 100
        scores["task_performance"] = exec_time_score

        # Calculate weighted overall score
        overall_score = sum(
            scores.get(category, 100) * weight
            for category, weight in self._performance_weights.items()
        )

        return min(100, max(0, overall_score))

    def add_worker_metrics(self, worker_id: str, metrics: WorkerPerformanceMetrics):
        """Add worker performance metrics for analysis."""
        self._worker_metrics[worker_id] = metrics

    def get_current_snapshot(self) -> Optional[SystemSnapshot]:
        """Get the most recent performance snapshot."""
        return self._snapshots[-1] if self._snapshots else None

    def get_historical_snapshots(self, count: int = 100) -> List[SystemSnapshot]:
        """Get recent historical snapshots."""
        return list(self._snapshots)[-count:]

    def export_performance_report(self, output_path: str | Path) -> bool:
        """Export performance analysis to JSON report."""
        try:
            report_data = {
                "report_metadata": {
                    "generated_at": time.time(),
                    "monitoring_duration": time.perf_counter() - self._start_time,
                    "total_snapshots": len(self._snapshots),
                    "monitoring_interval": self.monitoring_interval,
                },
                "current_snapshot": asdict(self._snapshots[-1])
                if self._snapshots
                else None,
                "recent_snapshots": [asdict(s) for s in list(self._snapshots)[-20:]],
                "alert_thresholds": self._alert_thresholds,
                "worker_metrics_summary": {
                    worker_id: {
                        "execution_time": metrics.execution_time,
                        "signal_batching_efficiency": metrics.signal_batching_efficiency,
                        "cache_hit_rate": metrics.cache_hit_rate,
                    }
                    for worker_id, metrics in self._worker_metrics.items()
                },
            }

            with open(output_path, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info(f"Performance report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export performance report: {e}")
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of optimization effectiveness."""
        if not self._snapshots:
            return {}

        latest = self._snapshots[-1]

        return {
            "signal_optimization": {
                "batching_enabled": latest.signal_batching_stats.get(
                    "batching_enabled", False
                ),
                "total_batched_signals": latest.signal_batching_stats.get(
                    "total_batched_signals", 0
                ),
                "total_emissions": latest.signal_batching_stats.get(
                    "total_emissions", 0
                ),
                "reduction_ratio": latest.signal_batching_stats.get(
                    "reduction_ratio", 0
                ),
                "active_batches": latest.signal_batching_stats.get("active_batches", 0),
            },
            "connection_pooling": {
                "total_connections": latest.connection_pool_stats.get(
                    "total_connections", 0
                ),
                "active_connections": latest.connection_pool_stats.get(
                    "active_connections", 0
                ),
                "reused_connections": latest.connection_pool_stats.get(
                    "reused_connections", 0
                ),
                "reuse_ratio": latest.connection_pool_stats.get("reuse_ratio", 0),
            },
            "attribute_caching": {
                "cache_size": latest.attribute_cache_stats.get("cache_size", 0),
                "hit_count": latest.attribute_cache_stats.get("hit_count", 0),
                "miss_count": latest.attribute_cache_stats.get("miss_count", 0),
                "hit_ratio": latest.attribute_cache_stats.get("hit_ratio", 0),
            },
            "thread_pools": {
                "total_pools": latest.thread_pool_stats.get("total_pools", 0),
                "total_threads": latest.thread_pool_stats.get("threads", {}).get(
                    "total_capacity", 0
                ),
                "active_threads": latest.thread_pool_stats.get("threads", {}).get(
                    "currently_active", 0
                ),
                "utilization_percent": latest.thread_pool_stats.get("threads", {}).get(
                    "utilization_percent", 0
                ),
            },
        }


# Global performance monitor instance
_global_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def initialize_performance_monitoring(
    monitoring_interval: float = 5.0,
) -> PerformanceMonitor:
    """Initialize performance monitoring system."""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor(monitoring_interval)
        _global_performance_monitor.start_monitoring()
    return _global_performance_monitor


def shutdown_performance_monitoring():
    """Shutdown performance monitoring system."""
    global _global_performance_monitor
    if _global_performance_monitor:
        _global_performance_monitor.stop_monitoring()
        _global_performance_monitor = None
