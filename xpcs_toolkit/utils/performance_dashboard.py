"""
Real-time Performance Dashboard for XPCS Toolkit CPU Optimizations.

This module provides a comprehensive real-time monitoring dashboard for tracking
the effectiveness of CPU optimizations including threading system performance,
cache hit rates, memory usage, and overall system health metrics.

Features:
- Real-time metrics visualization
- Threading system performance monitoring
- Memory management effectiveness tracking
- Cache system performance analysis
- Historical trend analysis with charts
- Performance alerts and threshold monitoring
- Export capabilities for performance reports
"""

from __future__ import annotations

import json
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Import existing optimization systems
from ..threading.performance_monitor import PerformanceMonitor
from .cache_monitor import CacheMonitor
from .logging_config import get_logger
from .memory_utils import SystemMemoryMonitor
from .optimization_health_monitor import HealthCheckResult, get_health_monitor

logger = get_logger(__name__)


@dataclass
class DashboardMetric:
    """Individual dashboard metric with display properties."""

    name: str
    value: float
    unit: str = ""
    format_string: str = "{:.2f}"
    color: str = "black"
    trend: str = "stable"  # "up", "down", "stable"
    history: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: float = field(default_factory=time.time)

    def update_value(self, new_value: float) -> None:
        """Update the metric value and history."""
        self.history.append((time.time(), self.value))
        self.value = new_value
        self.last_updated = time.time()

        # Update trend
        if len(self.history) >= 2:
            recent_values = [v for _, v in list(self.history)[-5:]]
            older_values = (
                [v for _, v in list(self.history)[-10:-5]]
                if len(self.history) >= 10
                else recent_values
            )

            if len(recent_values) >= 2 and len(older_values) >= 2:
                recent_avg = statistics.mean(recent_values)
                older_avg = statistics.mean(older_values)

                if recent_avg > older_avg * 1.05:
                    self.trend = "up"
                    self.color = "green"
                elif recent_avg < older_avg * 0.95:
                    self.trend = "down"
                    self.color = "red"
                else:
                    self.trend = "stable"
                    self.color = "black"

    def get_formatted_value(self) -> str:
        """Get formatted string representation of the value."""
        return f"{self.format_string.format(self.value)} {self.unit}".strip()


class MetricWidget(QWidget):
    """Widget for displaying a single metric with trend indication."""

    def __init__(self, metric: DashboardMetric, parent=None):
        super().__init__(parent)
        self.metric = metric
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the metric widget UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(5, 5, 5, 5)

        # Metric name label
        self.name_label = QLabel(self.metric.name)
        font = QFont()
        font.setBold(True)
        font.setPointSize(9)
        self.name_label.setFont(font)
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)

        # Value label
        self.value_label = QLabel(self.metric.get_formatted_value())
        value_font = QFont()
        value_font.setPointSize(14)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label)

        # Trend indicator
        self.trend_label = QLabel("●")
        trend_font = QFont()
        trend_font.setPointSize(12)
        self.trend_label.setFont(trend_font)
        self.trend_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.trend_label)

        self.update_display()

    def update_display(self) -> None:
        """Update the widget display with current metric values."""
        self.value_label.setText(self.metric.get_formatted_value())

        # Update trend color
        color_map = {"up": "green", "down": "red", "stable": "gray"}
        color = color_map.get(self.metric.trend, "black")
        self.trend_label.setStyleSheet(f"color: {color};")

        # Update trend tooltip
        trend_text = {"up": "Trending up", "down": "Trending down", "stable": "Stable"}
        self.trend_label.setToolTip(trend_text.get(self.metric.trend, "Unknown trend"))


class HealthStatusWidget(QWidget):
    """Widget for displaying system health status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._health_results: dict[str, HealthCheckResult] = {}
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the health status widget UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("System Health Status")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        # Health table
        self.health_table = QTableWidget()
        self.health_table.setColumnCount(4)
        self.health_table.setHorizontalHeaderLabels(
            ["Component", "Status", "Score", "Message"]
        )
        self.health_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.health_table)

    def update_health_status(
        self, health_results: dict[str, HealthCheckResult]
    ) -> None:
        """Update the health status display."""
        self._health_results = health_results

        # Clear and update table
        self.health_table.setRowCount(len(health_results))

        for row, (component, result) in enumerate(health_results.items()):
            # Component name
            self.health_table.setItem(row, 0, QTableWidgetItem(component))

            # Status with color coding
            status_item = QTableWidgetItem(result.status.value.title())
            if result.status.value == "healthy":
                status_item.setBackground(QColor(144, 238, 144))  # Light green
            elif result.status.value == "warning":
                status_item.setBackground(QColor(255, 255, 0))  # Yellow
            elif result.status.value == "critical":
                status_item.setBackground(QColor(255, 182, 193))  # Light red
            else:
                status_item.setBackground(QColor(211, 211, 211))  # Light gray

            self.health_table.setItem(row, 1, status_item)

            # Health score
            score_text = f"{result.score:.2f}"
            self.health_table.setItem(row, 2, QTableWidgetItem(score_text))

            # Message
            self.health_table.setItem(row, 3, QTableWidgetItem(result.message))


class AlertsWidget(QWidget):
    """Widget for displaying recent alerts and notifications."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._alerts: deque = deque(maxlen=100)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the alerts widget UI."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Recent Alerts")
        font = QFont()
        font.setBold(True)
        font.setPointSize(12)
        title.setFont(font)
        layout.addWidget(title)

        # Alerts text area
        self.alerts_text = QTextEdit()
        self.alerts_text.setReadOnly(True)
        self.alerts_text.setMaximumHeight(200)
        layout.addWidget(self.alerts_text)

        # Clear button
        self.clear_button = QPushButton("Clear Alerts")
        self.clear_button.clicked.connect(self.clear_alerts)
        layout.addWidget(self.clear_button)

    def add_alert(self, level: str, message: str, component: str = "") -> None:
        """Add a new alert to the display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        alert_text = (
            f"[{timestamp}] {level.upper()}: {component} - {message}"
            if component
            else f"[{timestamp}] {level.upper()}: {message}"
        )

        self._alerts.append((time.time(), level, alert_text))
        self._update_alerts_display()

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self._alerts.clear()
        self.alerts_text.clear()

    def _update_alerts_display(self) -> None:
        """Update the alerts display."""
        self.alerts_text.clear()

        # Show most recent alerts first
        for _, level, alert_text in reversed(
            list(self._alerts)[-20:]
        ):  # Show last 20 alerts
            if level.lower() in ["critical", "error"]:
                color = "red"
            elif level.lower() == "warning":
                color = "orange"
            else:
                color = "black"

            self.alerts_text.append(
                f'<span style="color: {color};">{alert_text}</span>'
            )


class PerformanceDashboard(QWidget):
    """
    Main performance dashboard widget for monitoring CPU optimization effectiveness.
    """

    # Signals
    dashboard_updated = Signal()
    metric_threshold_exceeded = Signal(
        str, float, float
    )  # metric_name, value, threshold

    def __init__(self, parent=None):
        super().__init__(parent)

        # Dashboard state
        self._metrics: dict[str, DashboardMetric] = {}
        self._update_interval = 2.0  # seconds
        self._running = False

        # Monitoring systems
        self._health_monitor = get_health_monitor()
        self._performance_monitor: PerformanceMonitor | None = None
        self._cache_monitor: CacheMonitor | None = None
        self._memory_monitor = SystemMemoryMonitor()

        # Timer for updates
        self._timer = QTimer()
        self._timer.timeout.connect(self._update_dashboard)

        # UI components
        self._metric_widgets: dict[str, MetricWidget] = {}
        self._health_widget: HealthStatusWidget | None = None
        self._alerts_widget: AlertsWidget | None = None

        self._setup_ui()
        self._initialize_metrics()
        self._connect_signals()

        logger.info("PerformanceDashboard initialized")

    def _setup_ui(self) -> None:
        """Set up the dashboard UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("XPCS Toolkit - CPU Optimization Performance Dashboard")
        title.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        font.setPointSize(16)
        title.setFont(font)
        layout.addWidget(title)

        # Control buttons
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.start_monitoring)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        controls_layout.addWidget(self.stop_button)

        self.export_button = QPushButton("Export Report")
        self.export_button.clicked.connect(self.export_report)
        controls_layout.addWidget(self.export_button)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Main dashboard area with tabs
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Metrics overview tab
        self._create_metrics_tab()

        # System health tab
        self._create_health_tab()

        # Alerts tab
        self._create_alerts_tab()

        # Historical trends tab
        self._create_trends_tab()

    def _create_metrics_tab(self) -> None:
        """Create the metrics overview tab."""
        metrics_widget = QWidget()
        layout = QVBoxLayout(metrics_widget)

        # Create metric groups
        groups = {
            "Threading Performance": [
                "thread_utilization",
                "task_throughput",
                "signal_batching_efficiency",
            ],
            "Memory Management": ["memory_usage", "cache_hit_rate", "memory_pressure"],
            "System Resources": ["cpu_usage", "available_memory", "disk_io"],
        }

        for group_name, metric_names in groups.items():
            group_box = QGroupBox(group_name)
            group_layout = QGridLayout(group_box)

            for i, metric_name in enumerate(metric_names):
                if metric_name in self._metrics:
                    metric_widget = MetricWidget(self._metrics[metric_name])
                    self._metric_widgets[metric_name] = metric_widget
                    row, col = divmod(i, 3)
                    group_layout.addWidget(metric_widget, row, col)

            layout.addWidget(group_box)

        self.tab_widget.addTab(metrics_widget, "Metrics Overview")

    def _create_health_tab(self) -> None:
        """Create the system health tab."""
        self._health_widget = HealthStatusWidget()
        self.tab_widget.addTab(self._health_widget, "System Health")

    def _create_alerts_tab(self) -> None:
        """Create the alerts tab."""
        self._alerts_widget = AlertsWidget()
        self.tab_widget.addTab(self._alerts_widget, "Alerts & Notifications")

    def _create_trends_tab(self) -> None:
        """Create the historical trends tab."""
        trends_widget = QWidget()
        layout = QVBoxLayout(trends_widget)

        # Placeholder for trend charts
        trends_label = QLabel(
            "Historical trend charts would be implemented here using PyQtGraph or matplotlib"
        )
        trends_label.setAlignment(Qt.AlignCenter)
        trends_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(trends_label)

        self.tab_widget.addTab(trends_widget, "Historical Trends")

    def _initialize_metrics(self) -> None:
        """Initialize dashboard metrics."""

        # Threading metrics
        self._metrics["thread_utilization"] = DashboardMetric(
            "Thread Pool Utilization", 0.0, "%", "{:.1f}"
        )
        self._metrics["task_throughput"] = DashboardMetric(
            "Task Throughput", 0.0, "tasks/sec", "{:.1f}"
        )
        self._metrics["signal_batching_efficiency"] = DashboardMetric(
            "Signal Batching Efficiency", 0.0, "%", "{:.1f}"
        )

        # Memory metrics
        self._metrics["memory_usage"] = DashboardMetric(
            "Memory Usage", 0.0, "%", "{:.1f}"
        )
        self._metrics["cache_hit_rate"] = DashboardMetric(
            "Cache Hit Rate", 0.0, "%", "{:.1f}"
        )
        self._metrics["memory_pressure"] = DashboardMetric(
            "Memory Pressure", 0.0, "", "{:.2f}"
        )

        # System metrics
        self._metrics["cpu_usage"] = DashboardMetric("CPU Usage", 0.0, "%", "{:.1f}")
        self._metrics["available_memory"] = DashboardMetric(
            "Available Memory", 0.0, "GB", "{:.1f}"
        )
        self._metrics["disk_io"] = DashboardMetric(
            "Disk I/O Rate", 0.0, "MB/s", "{:.1f}"
        )

    def _connect_signals(self) -> None:
        """Connect to monitoring system signals."""
        # Connect to health monitor
        self._health_monitor.health_report_ready.connect(self._on_health_report_ready)
        self._health_monitor.component_needs_attention.connect(self._on_component_alert)

        # Connect threshold exceeded signal
        self.metric_threshold_exceeded.connect(self._on_threshold_exceeded)

    def start_monitoring(self) -> None:
        """Start the dashboard monitoring."""
        if self._running:
            return

        self._running = True
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        # Start monitoring systems
        self._health_monitor.start_monitoring()

        # Start dashboard updates
        self._timer.start(int(self._update_interval * 1000))

        if self._alerts_widget:
            self._alerts_widget.add_alert("info", "Dashboard monitoring started")

        logger.info("Dashboard monitoring started")

    def stop_monitoring(self) -> None:
        """Stop the dashboard monitoring."""
        if not self._running:
            return

        self._running = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

        # Stop timer
        self._timer.stop()

        # Stop monitoring systems
        self._health_monitor.stop_monitoring()

        if self._alerts_widget:
            self._alerts_widget.add_alert("info", "Dashboard monitoring stopped")

        logger.info("Dashboard monitoring stopped")

    def export_report(self) -> None:
        """Export performance report to file."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = Path.home() / f"xpcs_performance_report_{timestamp}.json"

            report_data = {
                "timestamp": timestamp,
                "metrics": {},
                "health_status": {},
                "system_info": self._memory_monitor.get_memory_stats(),
            }

            # Add current metric values
            for name, metric in self._metrics.items():
                report_data["metrics"][name] = {
                    "value": metric.value,
                    "unit": metric.unit,
                    "trend": metric.trend,
                    "last_updated": metric.last_updated,
                }

            # Add health status
            health_report = self._health_monitor.get_health_report()
            for component, result in health_report.items():
                report_data["health_status"][component] = {
                    "status": result.status.value,
                    "score": result.score,
                    "message": result.message,
                }

            # Write report
            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)

            if self._alerts_widget:
                self._alerts_widget.add_alert(
                    "info", f"Performance report exported to {report_file}"
                )

            logger.info(f"Performance report exported to {report_file}")

        except Exception as e:
            error_msg = f"Failed to export performance report: {e!s}"
            if self._alerts_widget:
                self._alerts_widget.add_alert("error", error_msg)
            logger.error(error_msg)

    def _update_dashboard(self) -> None:
        """Update all dashboard metrics and displays."""
        if not self._running:
            return

        try:
            # Update metrics from various sources
            self._update_memory_metrics()
            self._update_threading_metrics()
            self._update_cache_metrics()

            # Update metric widget displays
            for _name, widget in self._metric_widgets.items():
                widget.update_display()

            # Check thresholds
            self._check_metric_thresholds()

            # Emit update signal
            self.dashboard_updated.emit()

        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")

    def _update_memory_metrics(self) -> None:
        """Update memory-related metrics."""
        try:
            memory_stats = self._memory_monitor.get_memory_stats()

            if "memory_usage" in self._metrics:
                self._metrics["memory_usage"].update_value(
                    memory_stats.get("memory_percent", 0.0)
                )

            if "available_memory" in self._metrics:
                available_gb = memory_stats.get("available_mb", 0.0) / 1024.0
                self._metrics["available_memory"].update_value(available_gb)

            if "memory_pressure" in self._metrics:
                # Calculate memory pressure as a composite metric
                memory_percent = memory_stats.get("memory_percent", 0.0)
                pressure = min(1.0, memory_percent / 80.0)  # Normalize to 0-1
                self._metrics["memory_pressure"].update_value(pressure)

        except Exception as e:
            logger.error(f"Failed to update memory metrics: {e}")

    def _update_threading_metrics(self) -> None:
        """Update threading-related metrics."""
        try:
            # These would be updated from actual threading system stats
            # For now, using placeholder values

            if "thread_utilization" in self._metrics:
                # This would come from thread pool manager
                self._metrics["thread_utilization"].update_value(45.0)  # Placeholder

            if "task_throughput" in self._metrics:
                # This would come from performance monitor
                self._metrics["task_throughput"].update_value(12.5)  # Placeholder

            if "signal_batching_efficiency" in self._metrics:
                # This would come from signal optimizer
                self._metrics["signal_batching_efficiency"].update_value(
                    78.0
                )  # Placeholder

        except Exception as e:
            logger.error(f"Failed to update threading metrics: {e}")

    def _update_cache_metrics(self) -> None:
        """Update cache-related metrics."""
        try:
            # This would be updated from actual cache monitor
            if "cache_hit_rate" in self._metrics:
                # Placeholder - would come from cache statistics
                self._metrics["cache_hit_rate"].update_value(85.0)  # Placeholder

        except Exception as e:
            logger.error(f"Failed to update cache metrics: {e}")

    def _check_metric_thresholds(self) -> None:
        """Check if any metrics exceed their thresholds."""
        thresholds = {
            "memory_usage": 85.0,
            "memory_pressure": 0.8,
            "cpu_usage": 90.0,
            "thread_utilization": 95.0,
        }

        for metric_name, threshold in thresholds.items():
            if metric_name in self._metrics:
                metric = self._metrics[metric_name]
                if metric.value > threshold:
                    self.metric_threshold_exceeded.emit(
                        metric_name, metric.value, threshold
                    )

    def _on_health_report_ready(
        self, health_results: dict[str, HealthCheckResult]
    ) -> None:
        """Handle health report updates."""
        if self._health_widget:
            self._health_widget.update_health_status(health_results)

    def _on_component_alert(self, component_name: str, message: str) -> None:
        """Handle component alerts."""
        if self._alerts_widget:
            self._alerts_widget.add_alert("warning", message, component_name)

    def _on_threshold_exceeded(
        self, metric_name: str, value: float, threshold: float
    ) -> None:
        """Handle metric threshold exceeded."""
        if self._alerts_widget:
            message = f"Threshold exceeded: {value:.1f} > {threshold:.1f}"
            self._alerts_widget.add_alert("warning", message, metric_name)


# Convenience function for creating dashboard
def create_performance_dashboard(parent=None) -> PerformanceDashboard:
    """Create and return a performance dashboard widget."""
    return PerformanceDashboard(parent)
