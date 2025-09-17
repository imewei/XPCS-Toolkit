"""
Integrated Monitoring System for XPCS Toolkit.

This module provides a unified interface to all monitoring components,
creating a comprehensive monitoring and maintenance system that ensures
zero Qt errors and optimal application performance.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, QTimer, Signal

from .health_monitor_daemon import (
    HealthMonitorDaemon,
    MonitoringConfiguration,
    get_health_monitor_daemon,
    initialize_health_monitoring
)
from .performance_metrics_collector import (
    PerformanceMetricsCollector,
    get_performance_metrics_collector,
    initialize_performance_metrics
)
from .qt_error_detector import (
    QtErrorDetector,
    get_qt_error_detector,
    initialize_qt_error_detection
)
from .system_resilience_manager import (
    SystemResilienceManager,
    get_system_resilience_manager,
    initialize_system_resilience
)
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MonitoringSystemConfiguration:
    """Configuration for the integrated monitoring system."""

    # Health monitoring configuration
    health_monitoring_config: Optional[MonitoringConfiguration] = None

    # Performance metrics configuration
    metrics_retention_seconds: int = 3600

    # Qt error detection configuration
    enable_qt_handler: bool = True

    # Resilience management configuration
    enable_auto_recovery: bool = True

    # Integration configuration
    enable_automatic_reports: bool = True
    report_interval_minutes: int = 60
    report_output_directory: Optional[str] = None

    # Startup configuration
    start_all_systems: bool = True


@dataclass
class SystemStatus:
    """Overall system monitoring status."""

    timestamp: float = field(default_factory=time.perf_counter)
    health_monitor_running: bool = False
    qt_error_detector_active: bool = False
    performance_metrics_active: bool = False
    resilience_manager_active: bool = False
    total_qt_errors: int = 0
    total_recovery_actions: int = 0
    current_resilience_level: str = "unknown"
    active_alerts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "health_monitor_running": self.health_monitor_running,
            "qt_error_detector_active": self.qt_error_detector_active,
            "performance_metrics_active": self.performance_metrics_active,
            "resilience_manager_active": self.resilience_manager_active,
            "total_qt_errors": self.total_qt_errors,
            "total_recovery_actions": self.total_recovery_actions,
            "current_resilience_level": self.current_resilience_level,
            "active_alerts": self.active_alerts
        }


class IntegratedMonitoringSystem(QObject):
    """
    Integrated monitoring system that coordinates all monitoring components.

    This system provides:
    - Unified initialization and configuration
    - Coordinated monitoring across all subsystems
    - Automatic reporting and alerting
    - Centralized status and health information
    - Simple API for application integration
    """

    # Signals
    system_started = Signal()
    system_stopped = Signal()
    status_updated = Signal(object)  # SystemStatus
    report_generated = Signal(str)   # report_path

    def __init__(self, config: Optional[MonitoringSystemConfiguration] = None, parent: QObject = None):
        """Initialize integrated monitoring system."""
        super().__init__(parent)

        self.config = config or MonitoringSystemConfiguration()

        # Component references
        self._health_monitor: Optional[HealthMonitorDaemon] = None
        self._performance_metrics: Optional[PerformanceMetricsCollector] = None
        self._qt_error_detector: Optional[QtErrorDetector] = None
        self._resilience_manager: Optional[SystemResilienceManager] = None

        # System state
        self._is_running = False
        self._start_time: Optional[float] = None

        # Reporting
        self._report_timer: Optional[QTimer] = None
        self._last_report_time: Optional[float] = None

        logger.info("Integrated monitoring system initialized")

    def start_monitoring(self) -> bool:
        """Start all monitoring systems."""
        if self._is_running:
            logger.warning("Monitoring system already running")
            return True

        try:
            logger.info("Starting integrated monitoring system")
            self._start_time = time.perf_counter()

            # Initialize all components
            self._initialize_components()

            # Start monitoring systems
            if self.config.start_all_systems:
                self._start_all_systems()

            # Setup automatic reporting
            if self.config.enable_automatic_reports:
                self._setup_automatic_reporting()

            self._is_running = True
            self.system_started.emit()

            logger.info("Integrated monitoring system started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start monitoring system: {e}")
            self._cleanup_partial_start()
            return False

    def stop_monitoring(self) -> bool:
        """Stop all monitoring systems."""
        if not self._is_running:
            return True

        try:
            logger.info("Stopping integrated monitoring system")

            # Stop automatic reporting
            if self._report_timer:
                self._report_timer.stop()
                self._report_timer = None

            # Stop all monitoring systems
            self._stop_all_systems()

            self._is_running = False
            self.system_stopped.emit()

            logger.info("Integrated monitoring system stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop monitoring system: {e}")
            return False

    def _initialize_components(self):
        """Initialize all monitoring components."""
        # Initialize performance metrics
        self._performance_metrics = initialize_performance_metrics(
            self.config.metrics_retention_seconds
        )

        # Initialize Qt error detection
        self._qt_error_detector = initialize_qt_error_detection(
            self.config.enable_qt_handler
        )

        # Initialize health monitoring
        self._health_monitor = initialize_health_monitoring(
            self.config.health_monitoring_config
        )

        # Initialize resilience management
        self._resilience_manager = initialize_system_resilience()

        logger.debug("All monitoring components initialized")

    def _start_all_systems(self):
        """Start all monitoring systems."""
        # Start health monitoring
        if self._health_monitor:
            self._health_monitor.start_monitoring()

        # Qt error detector is automatically active when initialized

        # Resilience manager is automatically active when initialized

        logger.debug("All monitoring systems started")

    def _stop_all_systems(self):
        """Stop all monitoring systems."""
        # Stop health monitoring
        if self._health_monitor:
            self._health_monitor.stop_monitoring()

        logger.debug("All monitoring systems stopped")

    def _cleanup_partial_start(self):
        """Cleanup after partial startup failure."""
        try:
            if self._health_monitor:
                self._health_monitor.stop_monitoring()

            if self._report_timer:
                self._report_timer.stop()
                self._report_timer = None

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _setup_automatic_reporting(self):
        """Setup automatic report generation."""
        if self.config.report_interval_minutes <= 0:
            return

        self._report_timer = QTimer(self)
        self._report_timer.timeout.connect(self._generate_automatic_report)
        interval_ms = self.config.report_interval_minutes * 60 * 1000
        self._report_timer.start(interval_ms)

        logger.info(f"Automatic reporting enabled: every {self.config.report_interval_minutes} minutes")

    def _generate_automatic_report(self):
        """Generate automatic monitoring report."""
        try:
            current_time = time.perf_counter()

            # Generate comprehensive report
            report_path = self.generate_comprehensive_report()

            self._last_report_time = current_time
            self.report_generated.emit(report_path)

            logger.info(f"Automatic report generated: {report_path}")

        except Exception as e:
            logger.error(f"Automatic report generation failed: {e}")

    def get_system_status(self) -> SystemStatus:
        """Get current system status."""
        status = SystemStatus()

        try:
            # Health monitor status
            if self._health_monitor:
                health_status = self._health_monitor.get_monitoring_status()
                status.health_monitor_running = health_status.get("is_running", False)
                status.active_alerts = health_status.get("active_alerts", 0)

            # Qt error detector status
            if self._qt_error_detector:
                error_summary = self._qt_error_detector.get_error_summary()
                status.qt_error_detector_active = True
                status.total_qt_errors = error_summary.get("total_errors", 0)

            # Performance metrics status
            if self._performance_metrics:
                metrics_summary = self._performance_metrics.get_performance_summary()
                status.performance_metrics_active = metrics_summary.get("metric_count", 0) > 0

            # Resilience manager status
            if self._resilience_manager:
                system_health = self._resilience_manager.get_system_health()
                status.resilience_manager_active = True
                status.current_resilience_level = system_health.resilience_level.value
                status.total_recovery_actions = system_health.recovery_actions_count

        except Exception as e:
            logger.error(f"Failed to get system status: {e}")

        return status

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive monitoring report."""
        try:
            current_time = time.perf_counter()
            timestamp_str = time.strftime("%Y%m%d_%H%M%S")

            # Determine output directory
            output_dir = Path(self.config.report_output_directory) if self.config.report_output_directory else Path.home() / ".xpcs_toolkit" / "reports"
            output_dir.mkdir(parents=True, exist_ok=True)

            report_path = output_dir / f"monitoring_report_{timestamp_str}.json"

            # Collect data from all systems
            report_data = {
                "report_metadata": {
                    "timestamp": current_time,
                    "timestamp_str": timestamp_str,
                    "uptime_seconds": current_time - (self._start_time or current_time),
                    "monitoring_system_version": "2.0.0"
                },
                "system_status": self.get_system_status().to_dict()
            }

            # Health monitoring data
            if self._health_monitor:
                report_data["health_monitoring"] = {
                    "status": self._health_monitor.get_monitoring_status(),
                    "alert_history": self._health_monitor.get_alert_history(hours=24),
                    "metrics_history": {
                        component: self._health_monitor.get_metrics_history(component, hours=24)
                        for component in ["thread_pool_health", "qt_compliance", "resources", "performance"]
                    }
                }

            # Qt error detection data
            if self._qt_error_detector:
                report_data["qt_error_detection"] = self._qt_error_detector.export_error_report()

            # Performance metrics data
            if self._performance_metrics:
                report_data["performance_metrics"] = {
                    "summary": self._performance_metrics.get_performance_summary(),
                    "snapshots": [
                        snapshot.to_dict()
                        for snapshot in self._performance_metrics.get_snapshots(hours=24)
                    ]
                }

            # Resilience management data
            if self._resilience_manager:
                report_data["resilience_management"] = self._resilience_manager.export_resilience_report()

            # Write report
            import json
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            logger.info(f"Comprehensive report generated: {report_path}")
            return str(report_path)

        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return ""

    def get_qt_error_summary(self) -> Dict[str, Any]:
        """Get Qt error summary."""
        if not self._qt_error_detector:
            return {"error": "Qt error detector not initialized"}

        return self._qt_error_detector.get_error_summary()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self._performance_metrics:
            return {"error": "Performance metrics not initialized"}

        return self._performance_metrics.get_performance_summary()

    def get_resilience_summary(self) -> Dict[str, Any]:
        """Get resilience summary."""
        if not self._resilience_manager:
            return {"error": "Resilience manager not initialized"}

        return self._resilience_manager.get_system_health().to_dict()

    def force_qt_error_check(self) -> Dict[str, Any]:
        """Force Qt error detection check."""
        if not self._qt_error_detector:
            return {"error": "Qt error detector not initialized"}

        # Create error capture to test detection
        with self._qt_error_detector.create_error_capture() as capture:
            # The capture will automatically detect any Qt errors during this block
            pass

        return {
            "captured_errors": len(capture.captured_errors),
            "errors": [error.to_dict() for error in capture.captured_errors]
        }

    def create_performance_snapshot(self) -> Dict[str, Any]:
        """Create performance snapshot."""
        if not self._performance_metrics:
            return {"error": "Performance metrics not initialized"}

        snapshot = self._performance_metrics.create_snapshot()
        return snapshot.to_dict()

    def is_monitoring_healthy(self) -> bool:
        """Check if monitoring system is healthy."""
        try:
            status = self.get_system_status()

            # Check if all critical systems are running
            critical_systems = [
                status.qt_error_detector_active,
                status.performance_metrics_active,
                status.resilience_manager_active
            ]

            # Check if Qt errors are under control
            qt_errors_ok = status.total_qt_errors < 10  # Configurable threshold

            # Check resilience level
            resilience_ok = status.current_resilience_level in ["normal", "degraded"]

            return all(critical_systems) and qt_errors_ok and resilience_ok

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    @property
    def is_running(self) -> bool:
        """Check if monitoring system is running."""
        return self._is_running

    @property
    def uptime_seconds(self) -> float:
        """Get system uptime in seconds."""
        if not self._start_time:
            return 0.0
        return time.perf_counter() - self._start_time


# Global instance
_integrated_monitoring_system: Optional[IntegratedMonitoringSystem] = None


def get_integrated_monitoring_system(config: Optional[MonitoringSystemConfiguration] = None) -> IntegratedMonitoringSystem:
    """Get the global integrated monitoring system instance."""
    global _integrated_monitoring_system

    if _integrated_monitoring_system is None:
        _integrated_monitoring_system = IntegratedMonitoringSystem(config)

    return _integrated_monitoring_system


def initialize_integrated_monitoring(config: Optional[MonitoringSystemConfiguration] = None) -> IntegratedMonitoringSystem:
    """Initialize and start integrated monitoring system."""
    system = get_integrated_monitoring_system(config)
    system.start_monitoring()
    return system


def shutdown_integrated_monitoring():
    """Shutdown integrated monitoring system."""
    global _integrated_monitoring_system

    if _integrated_monitoring_system:
        _integrated_monitoring_system.stop_monitoring()
        _integrated_monitoring_system = None


@contextmanager
def integrated_monitoring_context(config: Optional[MonitoringSystemConfiguration] = None):
    """Context manager for integrated monitoring system."""
    system = initialize_integrated_monitoring(config)
    try:
        yield system
    finally:
        shutdown_integrated_monitoring()


def quick_health_check() -> Dict[str, Any]:
    """Perform quick health check of monitoring system."""
    try:
        system = get_integrated_monitoring_system()

        if not system.is_running:
            return {
                "status": "not_running",
                "message": "Monitoring system is not running"
            }

        status = system.get_system_status()
        is_healthy = system.is_monitoring_healthy()

        return {
            "status": "healthy" if is_healthy else "degraded",
            "system_status": status.to_dict(),
            "uptime_seconds": system.uptime_seconds,
            "qt_errors": status.total_qt_errors,
            "recovery_actions": status.total_recovery_actions,
            "resilience_level": status.current_resilience_level
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Health check failed: {e}"
        }