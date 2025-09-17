"""
XPCS Toolkit Monitoring Package.

This package provides comprehensive monitoring and health checking capabilities
for the Qt compliance system and overall application health.
"""

from .health_monitor_daemon import (
    HealthMonitorDaemon,
    MonitoringConfiguration,
    HealthAlert,
    AlertSeverity,
    MonitoringInterval,
    get_health_monitor_daemon,
    initialize_health_monitoring,
    shutdown_health_monitoring,
    health_monitoring_context
)

from .performance_metrics_collector import (
    PerformanceMetricsCollector,
    MetricType,
    AggregationType,
    MetricValue,
    MetricSeries,
    PerformanceSnapshot,
    TimingContext,
    get_performance_metrics_collector,
    initialize_performance_metrics,
    performance_timer,
    record_performance_counter,
    record_performance_gauge,
    create_performance_snapshot
)

from .qt_error_detector import (
    QtErrorDetector,
    QtErrorType,
    QtErrorSeverity,
    QtErrorPattern,
    QtErrorOccurrence,
    QtErrorCapture,
    get_qt_error_detector,
    initialize_qt_error_detection,
    qt_error_capture
)

from .system_resilience_manager import (
    SystemResilienceManager,
    RecoveryStrategy,
    SystemComponent,
    ResilienceLevel,
    ResiliencePolicy,
    RecoveryAction,
    SystemHealth,
    get_system_resilience_manager,
    initialize_system_resilience,
    resilience_context
)

from .integrated_monitoring_system import (
    IntegratedMonitoringSystem,
    MonitoringSystemConfiguration,
    SystemStatus,
    get_integrated_monitoring_system,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring,
    integrated_monitoring_context,
    quick_health_check
)

__all__ = [
    # Health monitoring
    "HealthMonitorDaemon",
    "MonitoringConfiguration",
    "HealthAlert",
    "AlertSeverity",
    "MonitoringInterval",
    "get_health_monitor_daemon",
    "initialize_health_monitoring",
    "shutdown_health_monitoring",
    "health_monitoring_context",

    # Performance metrics
    "PerformanceMetricsCollector",
    "MetricType",
    "AggregationType",
    "MetricValue",
    "MetricSeries",
    "PerformanceSnapshot",
    "TimingContext",
    "get_performance_metrics_collector",
    "initialize_performance_metrics",
    "performance_timer",
    "record_performance_counter",
    "record_performance_gauge",
    "create_performance_snapshot",

    # Qt error detection
    "QtErrorDetector",
    "QtErrorType",
    "QtErrorSeverity",
    "QtErrorPattern",
    "QtErrorOccurrence",
    "QtErrorCapture",
    "get_qt_error_detector",
    "initialize_qt_error_detection",
    "qt_error_capture",

    # System resilience
    "SystemResilienceManager",
    "RecoveryStrategy",
    "SystemComponent",
    "ResilienceLevel",
    "ResiliencePolicy",
    "RecoveryAction",
    "SystemHealth",
    "get_system_resilience_manager",
    "initialize_system_resilience",
    "resilience_context",

    # Integrated monitoring system
    "IntegratedMonitoringSystem",
    "MonitoringSystemConfiguration",
    "SystemStatus",
    "get_integrated_monitoring_system",
    "initialize_integrated_monitoring",
    "shutdown_integrated_monitoring",
    "integrated_monitoring_context",
    "quick_health_check"
]