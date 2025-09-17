"""
Performance Metrics Collection Framework for XPCS Toolkit.

This module provides comprehensive performance metrics collection,
aggregation, and analysis capabilities for monitoring application
performance and identifying optimization opportunities.
"""

import asyncio
import json
import statistics
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QObject, QTimer, Signal

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of performance metrics."""

    COUNTER = "counter"         # Monotonically increasing value
    GAUGE = "gauge"            # Point-in-time value
    HISTOGRAM = "histogram"    # Distribution of values
    TIMER = "timer"           # Timing measurements
    THROUGHPUT = "throughput" # Operations per time unit


class AggregationType(Enum):
    """Types of metric aggregation."""

    SUM = "sum"
    AVERAGE = "average"
    MIN = "min"
    MAX = "max"
    PERCENTILE_50 = "p50"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    COUNT = "count"
    RATE = "rate"


@dataclass
class MetricValue:
    """Individual metric value with metadata."""

    timestamp: float
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "value": self.value,
            "tags": self.tags,
            "metadata": self.metadata
        }


@dataclass
class MetricSeries:
    """Time series of metric values."""

    name: str
    metric_type: MetricType
    unit: str = ""
    description: str = ""
    values: deque = field(default_factory=lambda: deque(maxlen=1000))
    tags: Dict[str, str] = field(default_factory=dict)

    def add_value(self, value: Union[int, float], timestamp: Optional[float] = None,
                  tags: Optional[Dict[str, str]] = None, metadata: Optional[Dict[str, Any]] = None):
        """Add a value to the metric series."""
        if timestamp is None:
            timestamp = time.perf_counter()

        metric_value = MetricValue(
            timestamp=timestamp,
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )

        self.values.append(metric_value)

    def get_latest(self) -> Optional[MetricValue]:
        """Get the latest metric value."""
        return self.values[-1] if self.values else None

    def get_values_in_range(self, start_time: float, end_time: float) -> List[MetricValue]:
        """Get values within a time range."""
        return [v for v in self.values if start_time <= v.timestamp <= end_time]

    def aggregate(self, aggregation: AggregationType,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None) -> Optional[float]:
        """Aggregate metric values."""
        if not self.values:
            return None

        # Get values in range
        if start_time is not None or end_time is not None:
            start_time = start_time or 0
            end_time = end_time or float('inf')
            values = [v.value for v in self.get_values_in_range(start_time, end_time)]
        else:
            values = [v.value for v in self.values]

        if not values:
            return None

        try:
            if aggregation == AggregationType.SUM:
                return sum(values)
            elif aggregation == AggregationType.AVERAGE:
                return statistics.mean(values)
            elif aggregation == AggregationType.MIN:
                return min(values)
            elif aggregation == AggregationType.MAX:
                return max(values)
            elif aggregation == AggregationType.COUNT:
                return len(values)
            elif aggregation == AggregationType.PERCENTILE_50:
                return statistics.median(values)
            elif aggregation == AggregationType.PERCENTILE_90:
                return statistics.quantiles(values, n=10)[8] if len(values) >= 10 else max(values)
            elif aggregation == AggregationType.PERCENTILE_95:
                return statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
            elif aggregation == AggregationType.PERCENTILE_99:
                return statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
            elif aggregation == AggregationType.RATE:
                if len(values) < 2:
                    return 0.0
                time_span = self.values[-1].timestamp - self.values[0].timestamp
                return len(values) / time_span if time_span > 0 else 0.0
            else:
                logger.warning(f"Unknown aggregation type: {aggregation}")
                return None

        except Exception as e:
            logger.error(f"Aggregation failed for {self.name} with {aggregation}: {e}")
            return None


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""

    timestamp: float
    metrics: Dict[str, Any]
    system_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "system_info": self.system_info
        }


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, collector: 'PerformanceMetricsCollector',
                 metric_name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.collector.record_timer(self.metric_name, duration, self.tags)


class PerformanceMetricsCollector(QObject):
    """
    Comprehensive performance metrics collection system.

    Provides:
    - Multiple metric types (counters, gauges, timers, histograms)
    - Automatic aggregation and analysis
    - Time-based snapshots
    - Export capabilities
    - Real-time monitoring integration
    """

    # Signals
    metric_recorded = Signal(str, object)  # metric_name, MetricValue
    snapshot_created = Signal(object)      # PerformanceSnapshot

    def __init__(self, retention_seconds: int = 3600, parent: QObject = None):
        """Initialize performance metrics collector."""
        super().__init__(parent)

        self.retention_seconds = retention_seconds
        self._metrics: Dict[str, MetricSeries] = {}
        self._snapshots: deque = deque(maxlen=100)
        self._mutex = threading.RLock()

        # Automatic cleanup timer - safe initialization for test environments
        try:
            self._cleanup_timer = QTimer(self)
            self._cleanup_timer.timeout.connect(self._cleanup_old_metrics)
            self._cleanup_timer.start(60000)  # Cleanup every minute
        except Exception as e:
            logger.warning(f"Could not initialize cleanup timer: {e}")
            self._cleanup_timer = None

        logger.info("Performance metrics collector initialized")

    def register_metric(self, name: str, metric_type: MetricType,
                       unit: str = "", description: str = "",
                       tags: Optional[Dict[str, str]] = None) -> MetricSeries:
        """Register a new metric series."""
        with self._mutex:
            if name in self._metrics:
                logger.warning(f"Metric '{name}' already registered")
                return self._metrics[name]

            metric_series = MetricSeries(
                name=name,
                metric_type=metric_type,
                unit=unit,
                description=description,
                tags=tags or {}
            )

            self._metrics[name] = metric_series
            logger.debug(f"Registered metric: {name} ({metric_type.value})")
            return metric_series

    def record_counter(self, name: str, value: Union[int, float] = 1,
                      tags: Optional[Dict[str, str]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Record a counter metric value."""
        self._record_metric(name, MetricType.COUNTER, value, tags, metadata)

    def record_gauge(self, name: str, value: Union[int, float],
                    tags: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """Record a gauge metric value."""
        self._record_metric(name, MetricType.GAUGE, value, tags, metadata)

    def record_timer(self, name: str, duration: float,
                    tags: Optional[Dict[str, str]] = None,
                    metadata: Optional[Dict[str, Any]] = None):
        """Record a timer metric value."""
        self._record_metric(name, MetricType.TIMER, duration, tags, metadata)

    def record_histogram(self, name: str, value: Union[int, float],
                        tags: Optional[Dict[str, str]] = None,
                        metadata: Optional[Dict[str, Any]] = None):
        """Record a histogram metric value."""
        self._record_metric(name, MetricType.HISTOGRAM, value, tags, metadata)

    def record_throughput(self, name: str, operations: int, duration: float,
                         tags: Optional[Dict[str, str]] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """Record a throughput metric value."""
        if duration > 0:
            throughput = operations / duration
            self._record_metric(name, MetricType.THROUGHPUT, throughput, tags, metadata)

    def _record_metric(self, name: str, metric_type: MetricType, value: Union[int, float],
                      tags: Optional[Dict[str, str]] = None,
                      metadata: Optional[Dict[str, Any]] = None):
        """Internal method to record a metric value."""
        with self._mutex:
            # Get or create metric series
            if name not in self._metrics:
                self.register_metric(name, metric_type)

            metric_series = self._metrics[name]

            # Verify metric type matches
            if metric_series.metric_type != metric_type:
                logger.warning(f"Metric type mismatch for '{name}': "
                             f"expected {metric_series.metric_type.value}, got {metric_type.value}")
                return

            # Add value
            timestamp = time.perf_counter()
            metric_series.add_value(value, timestamp, tags, metadata)

            # Create metric value for signal
            metric_value = MetricValue(
                timestamp=timestamp,
                value=value,
                tags=tags or {},
                metadata=metadata or {}
            )

            # Emit signal
            self.metric_recorded.emit(name, metric_value)

    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        with self._mutex:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricSeries]:
        """Get all metric series."""
        with self._mutex:
            return dict(self._metrics)

    def aggregate_metric(self, name: str, aggregation: AggregationType,
                        start_time: Optional[float] = None,
                        end_time: Optional[float] = None) -> Optional[float]:
        """Aggregate a metric over a time range."""
        metric_series = self.get_metric(name)
        if not metric_series:
            return None

        return metric_series.aggregate(aggregation, start_time, end_time)

    def create_snapshot(self) -> PerformanceSnapshot:
        """Create a performance snapshot of all current metrics."""
        timestamp = time.perf_counter()
        metrics = {}
        system_info = {}

        with self._mutex:
            # Collect metric aggregations
            for name, series in self._metrics.items():
                latest = series.get_latest()
                if latest:
                    metrics[name] = {
                        "latest_value": latest.value,
                        "latest_timestamp": latest.timestamp,
                        "count": len(series.values),
                        "average": series.aggregate(AggregationType.AVERAGE),
                        "min": series.aggregate(AggregationType.MIN),
                        "max": series.aggregate(AggregationType.MAX),
                        "p95": series.aggregate(AggregationType.PERCENTILE_95),
                        "metric_type": series.metric_type.value,
                        "unit": series.unit
                    }

            # Collect system information
            try:
                import psutil
                process = psutil.Process()
                system_info = {
                    "memory_rss_mb": process.memory_info().rss / 1024 / 1024,
                    "memory_vms_mb": process.memory_info().vms / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "thread_count": process.num_threads(),
                    "file_descriptors": getattr(process, 'num_fds', lambda: 0)()
                }
            except ImportError:
                logger.debug("psutil not available for system info")

        # Create snapshot
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            metrics=metrics,
            system_info=system_info
        )

        # Store snapshot
        self._snapshots.append(snapshot)

        # Emit signal
        self.snapshot_created.emit(snapshot)

        return snapshot

    def get_snapshots(self, hours: Optional[int] = None) -> List[PerformanceSnapshot]:
        """Get performance snapshots, optionally filtered by time."""
        cutoff_time = None
        if hours:
            cutoff_time = time.perf_counter() - (hours * 3600)

        return [
            snapshot for snapshot in self._snapshots
            if cutoff_time is None or snapshot.timestamp > cutoff_time
        ]

    def export_metrics(self, format: str = "json",
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> str:
        """Export metrics in specified format."""
        if format.lower() != "json":
            raise ValueError(f"Unsupported export format: {format}")

        export_data = {
            "export_timestamp": time.perf_counter(),
            "retention_seconds": self.retention_seconds,
            "metrics": {}
        }

        with self._mutex:
            for name, series in self._metrics.items():
                # Get values in range
                if start_time is not None or end_time is not None:
                    start_time = start_time or 0
                    end_time = end_time or float('inf')
                    values = series.get_values_in_range(start_time, end_time)
                else:
                    values = list(series.values)

                export_data["metrics"][name] = {
                    "name": series.name,
                    "type": series.metric_type.value,
                    "unit": series.unit,
                    "description": series.description,
                    "tags": series.tags,
                    "values": [v.to_dict() for v in values],
                    "aggregations": {
                        "count": len(values),
                        "sum": series.aggregate(AggregationType.SUM, start_time, end_time),
                        "average": series.aggregate(AggregationType.AVERAGE, start_time, end_time),
                        "min": series.aggregate(AggregationType.MIN, start_time, end_time),
                        "max": series.aggregate(AggregationType.MAX, start_time, end_time),
                        "p50": series.aggregate(AggregationType.PERCENTILE_50, start_time, end_time),
                        "p95": series.aggregate(AggregationType.PERCENTILE_95, start_time, end_time),
                    }
                }

        return json.dumps(export_data, indent=2)

    def save_metrics(self, file_path: Union[str, Path],
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None):
        """Save metrics to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = self.export_metrics("json", start_time, end_time)
        file_path.write_text(export_data)

        logger.info(f"Metrics saved to: {file_path}")

    def timing_context(self, metric_name: str,
                      tags: Optional[Dict[str, str]] = None) -> TimingContext:
        """Create a timing context manager."""
        return TimingContext(self, metric_name, tags)

    def clear_metrics(self, metric_names: Optional[List[str]] = None):
        """Clear metric data."""
        with self._mutex:
            if metric_names is None:
                # Clear all metrics
                for series in self._metrics.values():
                    series.values.clear()
                logger.info("Cleared all metric data")
            else:
                # Clear specific metrics
                for name in metric_names:
                    if name in self._metrics:
                        self._metrics[name].values.clear()
                        logger.debug(f"Cleared metric data for: {name}")

    def _cleanup_old_metrics(self):
        """Clean up old metric values based on retention policy."""
        cutoff_time = time.perf_counter() - self.retention_seconds

        with self._mutex:
            total_removed = 0
            for series in self._metrics.values():
                # Count values before cleanup
                before_count = len(series.values)

                # Remove old values
                while series.values and series.values[0].timestamp < cutoff_time:
                    series.values.popleft()

                # Count removed values
                removed = before_count - len(series.values)
                total_removed += removed

            if total_removed > 0:
                logger.debug(f"Cleaned up {total_removed} old metric values")

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a performance summary."""
        with self._mutex:
            summary = {
                "metric_count": len(self._metrics),
                "total_data_points": sum(len(series.values) for series in self._metrics.values()),
                "snapshot_count": len(self._snapshots),
                "retention_seconds": self.retention_seconds,
                "metrics_by_type": defaultdict(int)
            }

            # Count metrics by type
            for series in self._metrics.values():
                summary["metrics_by_type"][series.metric_type.value] += 1

            # Convert defaultdict to regular dict
            summary["metrics_by_type"] = dict(summary["metrics_by_type"])

            return summary


# Global instance
_performance_metrics_collector: Optional[PerformanceMetricsCollector] = None


def get_performance_metrics_collector(retention_seconds: int = 3600) -> PerformanceMetricsCollector:
    """Get the global performance metrics collector instance."""
    global _performance_metrics_collector

    if _performance_metrics_collector is None:
        _performance_metrics_collector = PerformanceMetricsCollector(retention_seconds)

    return _performance_metrics_collector


def initialize_performance_metrics(retention_seconds: int = 3600) -> PerformanceMetricsCollector:
    """Initialize performance metrics collection."""
    return get_performance_metrics_collector(retention_seconds)


@contextmanager
def performance_timer(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    collector = get_performance_metrics_collector()
    with collector.timing_context(metric_name, tags) as timer:
        yield timer


def record_performance_counter(name: str, value: Union[int, float] = 1,
                             tags: Optional[Dict[str, str]] = None):
    """Convenience function to record a counter metric."""
    collector = get_performance_metrics_collector()
    collector.record_counter(name, value, tags)


def record_performance_gauge(name: str, value: Union[int, float],
                           tags: Optional[Dict[str, str]] = None):
    """Convenience function to record a gauge metric."""
    collector = get_performance_metrics_collector()
    collector.record_gauge(name, value, tags)


def create_performance_snapshot() -> PerformanceSnapshot:
    """Convenience function to create a performance snapshot."""
    collector = get_performance_metrics_collector()
    return collector.create_snapshot()