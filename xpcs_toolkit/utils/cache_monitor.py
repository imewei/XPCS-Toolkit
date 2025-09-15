"""
Comprehensive cache monitoring and statistics system for XPCS Toolkit.

This module provides real-time monitoring, performance analytics, and optimization
recommendations for the multi-level caching system.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .adaptive_memory import get_adaptive_memory_manager
from .advanced_cache import get_global_cache
from .computation_cache import get_computation_cache
from .logging_config import get_logger
from .memory_utils import SystemMemoryMonitor
from .metadata_cache import get_metadata_cache

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance alert with contextual information."""

    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "message": self.message,
            "component": self.component,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "recommendations": self.recommendations,
        }


@dataclass
class PerformanceMetric:
    """Individual performance metric with history."""

    name: str
    value: float
    unit: str = ""
    category: str = "general"
    timestamp: float = field(default_factory=time.time)
    history: deque = field(default_factory=lambda: deque(maxlen=100))

    def update(self, new_value: float):
        """Update metric value and add to history."""
        self.history.append((time.time(), self.value))
        self.value = new_value
        self.timestamp = time.time()

    def get_trend(self, window_minutes: float = 5.0) -> str:
        """Get trend direction for the metric."""
        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            value for timestamp, value in self.history if timestamp > cutoff_time
        ]

        if len(recent_values) < 2:
            return "stable"

        # Simple trend analysis
        recent_avg = sum(recent_values[-3:]) / min(3, len(recent_values))
        older_avg = sum(recent_values[:3]) / min(3, len(recent_values))

        if recent_avg > older_avg * 1.1:
            return "increasing"
        if recent_avg < older_avg * 0.9:
            return "decreasing"
        return "stable"


class CacheMonitor:
    """
    Real-time cache monitoring and performance analytics system.

    Features:
    - Real-time metrics collection
    - Performance alerts and notifications
    - Optimization recommendations
    - Trend analysis and forecasting
    - Export capabilities for analysis
    """

    def __init__(
        self,
        monitoring_interval_seconds: float = 30.0,
        history_retention_hours: float = 24.0,
        enable_alerts: bool = True,
    ):
        self.monitoring_interval = monitoring_interval_seconds
        self.history_retention_hours = history_retention_hours
        self.enable_alerts = enable_alerts

        # Cache component references
        self._advanced_cache = get_global_cache()
        self._computation_cache = get_computation_cache()
        self._metadata_cache = get_metadata_cache()
        self._memory_manager = get_adaptive_memory_manager()

        # Metrics storage
        self._metrics: dict[str, PerformanceMetric] = {}
        self._alerts: deque[PerformanceAlert] = deque(maxlen=1000)
        self._alert_history: deque[PerformanceAlert] = deque(maxlen=5000)

        # Alert thresholds
        self._alert_thresholds = self._initialize_alert_thresholds()

        # Monitoring state
        self._monitoring_thread: threading.Thread | None = None
        self._stop_monitoring = threading.Event()
        self._lock = threading.RLock()

        # Performance baselines
        self._baselines: dict[str, float] = {}
        self._baseline_period_hours = 2.0

        # Alert callbacks
        self._alert_callbacks: list[Callable[[PerformanceAlert], None]] = []

        # Start monitoring
        self._start_monitoring()

        logger.info(
            f"CacheMonitor initialized with {monitoring_interval_seconds}s interval"
        )

    def _initialize_alert_thresholds(self) -> dict[str, dict[str, float]]:
        """Initialize default alert thresholds for various metrics."""
        return {
            "memory_pressure": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "cache_hit_rate": {
                "warning": 0.4,  # Below 40% hit rate
                "error": 0.2,  # Below 20% hit rate
                "critical": 0.1,  # Below 10% hit rate
            },
            "cache_memory_utilization": {
                "warning": 85.0,
                "error": 95.0,
                "critical": 98.0,
            },
            "eviction_rate": {
                "warning": 10.0,  # More than 10 evictions per minute
                "error": 30.0,  # More than 30 evictions per minute
                "critical": 60.0,  # More than 60 evictions per minute
            },
            "computation_cache_size": {
                "warning": 80.0,  # 80% of max entries
                "error": 90.0,  # 90% of max entries
                "critical": 95.0,  # 95% of max entries
            },
        }

    def _start_monitoring(self):
        """Start background monitoring thread."""
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_worker, daemon=True, name="CacheMonitor"
        )
        self._monitoring_thread.start()

    def _monitoring_worker(self):
        """Background worker for continuous monitoring."""
        while not self._stop_monitoring.wait(self.monitoring_interval):
            try:
                self._collect_metrics()
                self._analyze_performance()
                self._cleanup_old_data()
            except Exception as e:
                logger.error(f"Error in cache monitoring worker: {e}")

    def _collect_metrics(self):
        """Collect current metrics from all cache components."""
        time.time()

        try:
            # Advanced cache metrics
            advanced_stats = self._advanced_cache.get_stats()
            self._update_metric(
                "cache_l1_entries", advanced_stats["current_counts"]["l1_entries"]
            )
            self._update_metric(
                "cache_l2_entries", advanced_stats["current_counts"]["l2_entries"]
            )
            self._update_metric(
                "cache_l3_entries", advanced_stats["current_counts"]["l3_entries"]
            )

            # Hit rates
            hit_rates = advanced_stats.get("hit_rates", {})
            self._update_metric("cache_l1_hit_rate", hit_rates.get("l1_hit_rate", 0.0))
            self._update_metric("cache_l2_hit_rate", hit_rates.get("l2_hit_rate", 0.0))
            self._update_metric("cache_l3_hit_rate", hit_rates.get("l3_hit_rate", 0.0))
            self._update_metric(
                "cache_overall_hit_rate", hit_rates.get("overall_hit_rate", 0.0)
            )

            # Memory usage
            memory_usage = advanced_stats.get("memory_usage_mb", {})
            self._update_metric(
                "cache_l1_memory_mb", memory_usage.get("current_l1", 0.0)
            )
            self._update_metric(
                "cache_l2_memory_mb", memory_usage.get("current_l2", 0.0)
            )
            self._update_metric(
                "cache_l3_disk_mb", memory_usage.get("current_l3_disk", 0.0)
            )

            # Operation counts
            operation_counts = advanced_stats.get("operation_counts", {})
            self._update_metric(
                "cache_total_gets", operation_counts.get("total_gets", 0)
            )
            self._update_metric(
                "cache_total_puts", operation_counts.get("total_puts", 0)
            )
            self._update_metric(
                "cache_total_evictions", operation_counts.get("total_evictions", 0)
            )

            # Performance metrics
            performance_ms = advanced_stats.get("performance_ms", {})
            self._update_metric(
                "cache_avg_get_time_ms", performance_ms.get("avg_get_time", 0.0)
            )
            self._update_metric(
                "cache_avg_put_time_ms", performance_ms.get("avg_put_time", 0.0)
            )

            # Compression metrics
            compression_stats = advanced_stats.get("compression", {})
            self._update_metric(
                "compression_ratio", compression_stats.get("compression_ratio", 1.0)
            )
            self._update_metric(
                "compression_bytes_saved_pct",
                compression_stats.get("bytes_saved_percentage", 0.0),
            )

        except Exception as e:
            logger.debug(f"Error collecting advanced cache metrics: {e}")

        try:
            # Computation cache metrics
            comp_stats = self._computation_cache.get_computation_stats()
            comp_counts = comp_stats.get("computation_counts", {})

            self._update_metric(
                "comp_cache_g2_entries", comp_counts.get("g2_fitting", 0)
            )
            self._update_metric(
                "comp_cache_saxs_entries", comp_counts.get("saxs_analysis", 0)
            )
            self._update_metric(
                "comp_cache_twotime_entries", comp_counts.get("twotime_correlation", 0)
            )
            self._update_metric(
                "comp_cache_total_entries", comp_stats.get("total_computations", 0)
            )

            # Performance estimates
            perf_estimates = comp_stats.get("performance_estimates", {})
            self._update_metric(
                "comp_cache_time_saved_hours",
                perf_estimates.get("estimated_time_saved_hours", 0.0),
            )

        except Exception as e:
            logger.debug(f"Error collecting computation cache metrics: {e}")

        try:
            # Metadata cache metrics
            meta_stats = self._metadata_cache.get_cache_statistics()
            meta_cache = meta_stats.get("metadata_cache", {})

            self._update_metric(
                "meta_cache_files", meta_cache.get("cached_metadata_files", 0)
            )
            self._update_metric(
                "meta_cache_qmaps", meta_cache.get("cached_qmap_files", 0)
            )
            self._update_metric(
                "meta_cache_prefetch_queue", meta_cache.get("prefetch_queue_size", 0)
            )
            self._update_metric(
                "meta_cache_recent_accesses",
                meta_cache.get("recent_accesses_per_hour", 0),
            )

        except Exception as e:
            logger.debug(f"Error collecting metadata cache metrics: {e}")

        try:
            # Memory manager metrics
            memory_stats = self._memory_manager.get_performance_stats()
            pattern_recognition = memory_stats.get("pattern_recognition", {})
            prefetch_performance = memory_stats.get("prefetch_performance", {})

            self._update_metric(
                "memory_pattern_confidence",
                pattern_recognition.get("pattern_confidence", 0.0),
            )
            self._update_metric(
                "memory_access_records", pattern_recognition.get("access_records", 0)
            )
            self._update_metric(
                "memory_prefetch_hit_rate", prefetch_performance.get("hit_rate", 0.0)
            )
            self._update_metric(
                "memory_saved_mb", prefetch_performance.get("memory_saved_mb", 0.0)
            )

        except Exception as e:
            logger.debug(f"Error collecting memory manager metrics: {e}")

        try:
            # System metrics
            used_mb, available_mb, percent_used = SystemMemoryMonitor.get_memory_info()
            self._update_metric("system_memory_used_mb", used_mb)
            self._update_metric("system_memory_available_mb", available_mb)
            self._update_metric("system_memory_pressure", percent_used)

        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")

    def _update_metric(
        self, name: str, value: float, unit: str = "", category: str = "cache"
    ):
        """Update or create a performance metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = PerformanceMetric(
                    name=name, value=value, unit=unit, category=category
                )
            else:
                self._metrics[name].update(value)

    def _analyze_performance(self):
        """Analyze current metrics and generate alerts if needed."""
        if not self.enable_alerts:
            return

        time.time()

        # Memory pressure analysis
        memory_pressure = self._metrics.get("system_memory_pressure")
        if memory_pressure:
            self._check_threshold_alert(
                "memory_pressure",
                memory_pressure.value,
                "System memory pressure",
                unit="%",
            )

        # Cache hit rate analysis
        overall_hit_rate = self._metrics.get("cache_overall_hit_rate")
        if overall_hit_rate:
            # For hit rate, lower values are worse, so we invert the logic
            self._check_threshold_alert(
                "cache_hit_rate",
                overall_hit_rate.value,
                "Cache hit rate",
                invert_thresholds=True,
                unit="%",
            )

        # Cache memory utilization
        l1_memory = self._metrics.get(
            "cache_l1_memory_mb", PerformanceMetric("", 0.0)
        ).value
        l2_memory = self._metrics.get(
            "cache_l2_memory_mb", PerformanceMetric("", 0.0)
        ).value

        # Calculate cache utilization (simplified)
        if l1_memory > 0 or l2_memory > 0:
            # This would need actual cache limits for accurate calculation
            estimated_utilization = min(
                (l1_memory + l2_memory) / 1500.0 * 100, 100.0
            )  # Assuming 1.5GB total
            self._update_metric("cache_memory_utilization", estimated_utilization, "%")

            self._check_threshold_alert(
                "cache_memory_utilization",
                estimated_utilization,
                "Cache memory utilization",
                unit="%",
            )

        # Eviction rate analysis
        evictions_metric = self._metrics.get("cache_total_evictions")
        if evictions_metric and len(evictions_metric.history) >= 2:
            # Calculate eviction rate per minute
            recent_time, recent_evictions = evictions_metric.history[-1]
            older_time, older_evictions = evictions_metric.history[-2]

            time_diff_minutes = (recent_time - older_time) / 60.0
            if time_diff_minutes > 0:
                eviction_rate = (recent_evictions - older_evictions) / time_diff_minutes
                self._update_metric("cache_eviction_rate_per_minute", eviction_rate)

                self._check_threshold_alert(
                    "eviction_rate",
                    eviction_rate,
                    "Cache eviction rate",
                    unit=" evictions/min",
                )

        # Computation cache size analysis
        comp_total = self._metrics.get("comp_cache_total_entries")
        if comp_total:
            # Assuming max 300 entries (100 per type * 3 types)
            utilization_pct = min(comp_total.value / 300.0 * 100, 100.0)
            self._update_metric("comp_cache_utilization_pct", utilization_pct, "%")

            self._check_threshold_alert(
                "computation_cache_size",
                utilization_pct,
                "Computation cache utilization",
                unit="%",
            )

    def _check_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        description: str,
        invert_thresholds: bool = False,
        unit: str = "",
    ):
        """Check if a metric value crosses alert thresholds."""
        if metric_name not in self._alert_thresholds:
            return

        thresholds = self._alert_thresholds[metric_name]
        level = None
        threshold_value = None

        if invert_thresholds:
            # For metrics where lower is worse (like hit rates)
            if current_value < thresholds.get("critical", 0):
                level = AlertLevel.CRITICAL
                threshold_value = thresholds["critical"]
            elif current_value < thresholds.get("error", 0):
                level = AlertLevel.ERROR
                threshold_value = thresholds["error"]
            elif current_value < thresholds.get("warning", 0):
                level = AlertLevel.WARNING
                threshold_value = thresholds["warning"]
        # For metrics where higher is worse (like memory pressure)
        elif current_value > thresholds.get("critical", 100):
            level = AlertLevel.CRITICAL
            threshold_value = thresholds["critical"]
        elif current_value > thresholds.get("error", 100):
            level = AlertLevel.ERROR
            threshold_value = thresholds["error"]
        elif current_value > thresholds.get("warning", 100):
            level = AlertLevel.WARNING
            threshold_value = thresholds["warning"]

        if level is not None:
            # Generate alert
            recommendations = self._get_recommendations_for_metric(
                metric_name, current_value, level
            )

            alert = PerformanceAlert(
                level=level,
                message=f"{description} is {current_value:.2f}{unit} (threshold: {threshold_value:.2f}{unit})",
                component="cache_system",
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold_value,
                recommendations=recommendations,
            )

            self._add_alert(alert)

    def _get_recommendations_for_metric(
        self, metric_name: str, value: float, level: AlertLevel
    ) -> list[str]:
        """Get optimization recommendations for a specific metric alert."""
        recommendations = []

        if metric_name == "memory_pressure":
            if level == AlertLevel.CRITICAL:
                recommendations.extend(
                    [
                        "Immediately clear L1 cache",
                        "Force promotion of data to disk cache",
                        "Disable prefetching temporarily",
                        "Consider restarting application if memory continues to grow",
                    ]
                )
            elif level == AlertLevel.ERROR:
                recommendations.extend(
                    [
                        "Clear computation caches",
                        "Promote L1 data to L2/L3",
                        "Reduce cache size limits",
                    ]
                )
            else:  # WARNING
                recommendations.extend(
                    [
                        "Monitor memory usage closely",
                        "Consider enabling conservative memory strategy",
                        "Clean up old cached computations",
                    ]
                )

        elif metric_name == "cache_hit_rate":
            if level in [AlertLevel.CRITICAL, AlertLevel.ERROR]:
                recommendations.extend(
                    [
                        "Increase cache memory limits if possible",
                        "Review access patterns for optimization",
                        "Enable more aggressive prefetching",
                        "Check for cache key conflicts or invalidation issues",
                    ]
                )
            else:  # WARNING
                recommendations.extend(
                    [
                        "Analyze cache usage patterns",
                        "Consider adjusting cache eviction policies",
                        "Enable cache warming for frequently accessed files",
                    ]
                )

        elif metric_name == "eviction_rate":
            recommendations.extend(
                [
                    "Increase cache memory limits",
                    "Review eviction policy settings",
                    "Identify and optimize frequently evicted data",
                    "Consider using compression for larger datasets",
                ]
            )

        elif metric_name == "cache_memory_utilization":
            recommendations.extend(
                [
                    "Clean up old cache entries",
                    "Enable more aggressive compression",
                    "Promote data to disk cache",
                    "Review cache size configuration",
                ]
            )

        elif metric_name == "computation_cache_size":
            recommendations.extend(
                [
                    "Clean up old computation results",
                    "Reduce maximum entries per computation type",
                    "Implement more aggressive TTL policies for computations",
                ]
            )

        return recommendations

    def _add_alert(self, alert: PerformanceAlert):
        """Add alert to active alerts and history."""
        with self._lock:
            # Avoid duplicate alerts for the same metric within a short time
            recent_alerts = [
                a
                for a in self._alerts
                if a.metric_name == alert.metric_name
                and time.time() - a.timestamp < 300
            ]  # 5 minutes

            if not recent_alerts:
                self._alerts.append(alert)
                self._alert_history.append(alert)

                # Trigger alert callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")

                logger.warning(
                    f"Performance Alert [{alert.level.value.upper()}]: {alert.message}"
                )

    def _cleanup_old_data(self):
        """Clean up old metrics history and alerts."""
        cutoff_time = time.time() - (self.history_retention_hours * 3600)

        with self._lock:
            # Clean up metric history
            for metric in self._metrics.values():
                # Remove old history points
                while metric.history and metric.history[0][0] < cutoff_time:
                    metric.history.popleft()

            # Clean up old active alerts (keep recent ones)
            alert_cutoff = time.time() - (2 * 3600)  # Keep alerts for 2 hours
            while self._alerts and self._alerts[0].timestamp < alert_cutoff:
                self._alerts.popleft()

    def register_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Register callback function for performance alerts."""
        self._alert_callbacks.append(callback)

    def get_current_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            return {
                name: {
                    "value": metric.value,
                    "unit": metric.unit,
                    "category": metric.category,
                    "timestamp": metric.timestamp,
                    "trend": metric.get_trend(),
                }
                for name, metric in self._metrics.items()
            }

    def get_active_alerts(self) -> list[dict[str, Any]]:
        """Get list of active alerts."""
        with self._lock:
            return [alert.to_dict() for alert in self._alerts]

    def get_alert_history(self, hours: float = 24.0) -> list[dict[str, Any]]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)

        with self._lock:
            return [
                alert.to_dict()
                for alert in self._alert_history
                if alert.timestamp > cutoff_time
            ]

    def get_metric_history(
        self, metric_name: str, hours: float = 2.0
    ) -> list[tuple[float, float]]:
        """Get history for a specific metric."""
        if metric_name not in self._metrics:
            return []

        cutoff_time = time.time() - (hours * 3600)
        metric = self._metrics[metric_name]

        with self._lock:
            return [
                (timestamp, value)
                for timestamp, value in metric.history
                if timestamp > cutoff_time
            ]

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.get_current_metrics()
        active_alerts = self.get_active_alerts()

        # Calculate key performance indicators
        cache_efficiency = (
            metrics.get("cache_overall_hit_rate", {}).get("value", 0.0) * 100
        )
        memory_efficiency = 100.0 - metrics.get("system_memory_pressure", {}).get(
            "value", 0.0
        )

        time_saved_hours = metrics.get("comp_cache_time_saved_hours", {}).get(
            "value", 0.0
        )
        memory_saved_mb = metrics.get("memory_saved_mb", {}).get("value", 0.0)

        alert_counts = defaultdict(int)
        for alert in active_alerts:
            alert_counts[alert["level"]] += 1

        return {
            "overall_health": self._calculate_overall_health(metrics, active_alerts),
            "key_metrics": {
                "cache_efficiency_percent": cache_efficiency,
                "memory_efficiency_percent": memory_efficiency,
                "time_saved_hours": time_saved_hours,
                "memory_saved_mb": memory_saved_mb,
                "total_cache_entries": (
                    metrics.get("cache_l1_entries", {}).get("value", 0)
                    + metrics.get("cache_l2_entries", {}).get("value", 0)
                    + metrics.get("cache_l3_entries", {}).get("value", 0)
                ),
            },
            "alert_summary": {
                "total_active_alerts": len(active_alerts),
                "by_level": dict(alert_counts),
            },
            "recommendations": self._get_general_recommendations(
                metrics, active_alerts
            ),
        }

    def _calculate_overall_health(
        self, metrics: dict[str, Any], active_alerts: list[dict[str, Any]]
    ) -> str:
        """Calculate overall system health score."""
        # Simple health calculation based on alerts and key metrics
        critical_alerts = sum(
            1 for alert in active_alerts if alert["level"] == "critical"
        )
        error_alerts = sum(1 for alert in active_alerts if alert["level"] == "error")
        warning_alerts = sum(
            1 for alert in active_alerts if alert["level"] == "warning"
        )

        if critical_alerts > 0:
            return "critical"
        if error_alerts > 2:
            return "poor"
        if error_alerts > 0 or warning_alerts > 5:
            return "fair"
        if warning_alerts > 0:
            return "good"
        return "excellent"

    def _get_general_recommendations(
        self, metrics: dict[str, Any], active_alerts: list[dict[str, Any]]
    ) -> list[str]:
        """Get general optimization recommendations."""
        recommendations = []

        # Check cache performance
        hit_rate = metrics.get("cache_overall_hit_rate", {}).get("value", 0.0)
        if hit_rate < 0.6:
            recommendations.append(
                "Consider increasing cache size limits to improve hit rate"
            )

        # Check memory pressure
        memory_pressure = metrics.get("system_memory_pressure", {}).get("value", 0.0)
        if memory_pressure > 80:
            recommendations.append(
                "High memory usage detected - enable conservative memory strategy"
            )

        # Check computation cache usage
        comp_entries = metrics.get("comp_cache_total_entries", {}).get("value", 0)
        if comp_entries > 250:
            recommendations.append(
                "Computation cache is getting full - consider cleaning up old results"
            )

        # Check for frequent evictions
        eviction_rate = metrics.get("cache_eviction_rate_per_minute", {}).get(
            "value", 0.0
        )
        if eviction_rate > 5:
            recommendations.append(
                "High eviction rate detected - consider increasing cache memory limits"
            )

        # General recommendations based on alerts
        if len(active_alerts) == 0:
            recommendations.append("Cache system is performing optimally")
        elif len([a for a in active_alerts if a["level"] in ["error", "critical"]]) > 0:
            recommendations.append(
                "Address critical and error alerts immediately for optimal performance"
            )

        return recommendations

    def export_metrics(self, format: str = "json") -> str:
        """Export current metrics and statistics."""
        data = {
            "timestamp": time.time(),
            "metrics": self.get_current_metrics(),
            "active_alerts": self.get_active_alerts(),
            "performance_summary": self.get_performance_summary(),
        }

        if format.lower() == "json":
            return json.dumps(data, indent=2)
        raise ValueError(f"Unsupported export format: {format}")

    def shutdown(self):
        """Shutdown cache monitor gracefully."""
        logger.info("Shutting down CacheMonitor")

        # Stop monitoring thread
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        # Log final statistics
        summary = self.get_performance_summary()
        logger.info(
            f"Final cache performance: {summary['overall_health']} "
            f"(hit rate: {summary['key_metrics']['cache_efficiency_percent']:.1f}%)"
        )


# Global cache monitor instance
_global_cache_monitor: CacheMonitor | None = None


def get_cache_monitor(
    monitoring_interval_seconds: float = 30.0,
    enable_alerts: bool = True,
    reset: bool = False,
) -> CacheMonitor:
    """
    Get or create global cache monitor instance.

    Parameters
    ----------
    monitoring_interval_seconds : float
        Monitoring interval in seconds
    enable_alerts : bool
        Whether to enable performance alerts
    reset : bool
        Whether to reset existing monitor

    Returns
    -------
    CacheMonitor
        Global cache monitor instance
    """
    global _global_cache_monitor

    if _global_cache_monitor is None or reset:
        if _global_cache_monitor is not None:
            _global_cache_monitor.shutdown()

        _global_cache_monitor = CacheMonitor(
            monitoring_interval_seconds=monitoring_interval_seconds,
            enable_alerts=enable_alerts,
        )

    return _global_cache_monitor


def setup_cache_monitoring_gui_integration():
    """Setup cache monitoring integration for GUI applications."""
    monitor = get_cache_monitor()

    # Setup alert callback for GUI notifications
    def gui_alert_callback(alert: PerformanceAlert):
        """Handle alerts for GUI display."""
        # This would integrate with actual GUI notification system
        if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            logger.warning(f"GUI Alert: {alert.message}")
            # Could trigger GUI popup, status bar update, etc.

    monitor.register_alert_callback(gui_alert_callback)

    logger.info("Cache monitoring GUI integration setup complete")
    return monitor
