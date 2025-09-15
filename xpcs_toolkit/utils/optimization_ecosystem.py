"""
Comprehensive CPU Optimization Ecosystem for XPCS Toolkit

This module integrates all three optimization maintenance subagents:
- Subagent 1: Optimization Monitoring & Maintenance
- Subagent 2: User Workflow Profiling
- Subagent 3: Performance Testing & Regression Prevention

Provides unified interface for the complete optimization maintenance ecosystem.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

from .alert_system import Alert, AlertSeverity, AlertSystem, get_alert_system
from .cpu_bottleneck_analyzer import CPUBottleneckAnalyzer, get_bottleneck_analyzer
from .logging_config import get_logger
from .maintenance_scheduler import (
    MaintenanceResult,
    MaintenanceScheduler,
    get_maintenance_scheduler,
)

# Import all subagent components
from .optimization_health_monitor import OptimizationHealthMonitor, get_health_monitor
from .performance_dashboard import PerformanceDashboard, create_performance_dashboard
from .usage_pattern_miner import UsagePatternMiner, get_pattern_miner
from .workflow_optimization_report import (
    ReportFormat,
    WorkflowOptimizationReportGenerator,
    get_report_generator,
)
from .workflow_profiler import WorkflowProfiler, get_workflow_profiler

logger = get_logger(__name__)


class EcosystemState(Enum):
    """Current state of the optimization ecosystem."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EcosystemStatus:
    """Overall ecosystem status information."""

    state: EcosystemState = EcosystemState.STOPPED
    uptime_seconds: float = 0.0
    components_active: dict[str, bool] = field(default_factory=dict)
    last_health_check: float | None = None
    performance_score: float = 0.0
    active_alerts: int = 0
    critical_issues: int = 0
    total_workflows_profiled: int = 0
    total_bottlenecks_found: int = 0
    total_maintenance_tasks: int = 0
    last_error: str | None = None


class OptimizationEcosystem(QObject):
    """
    Unified optimization ecosystem integrating all three subagents.

    Coordinates monitoring, profiling, and testing to provide comprehensive
    CPU optimization maintenance for the XPCS Toolkit.
    """

    # Signals
    state_changed = Signal(str)  # state
    performance_alert = Signal(str, str, int)  # type, message, severity
    bottleneck_detected = Signal(str, str, float)  # type, description, severity
    maintenance_completed = Signal(str, bool)  # task_name, success
    ecosystem_error = Signal(str)  # error_message

    def __init__(self):
        super().__init__()

        self._state = EcosystemState.STOPPED
        self._start_time: float | None = None
        self._lock = threading.RLock()

        # Component references
        self._health_monitor: OptimizationHealthMonitor | None = None
        self._dashboard: PerformanceDashboard | None = None
        self._maintenance_scheduler: MaintenanceScheduler | None = None
        self._alert_system: AlertSystem | None = None
        self._workflow_profiler: WorkflowProfiler | None = None
        self._bottleneck_analyzer: CPUBottleneckAnalyzer | None = None
        self._pattern_miner: UsagePatternMiner | None = None
        self._report_generator: WorkflowOptimizationReportGenerator | None = None

        # Status tracking
        self._status = EcosystemStatus()
        self._update_timer: QTimer | None = None

        logger.info("Optimization ecosystem initialized")

    @property
    def state(self) -> EcosystemState:
        """Current ecosystem state."""
        return self._state

    @property
    def status(self) -> EcosystemStatus:
        """Current ecosystem status."""
        with self._lock:
            return self._status

    def start_ecosystem(
        self,
        enable_dashboard: bool = True,
        enable_profiling: bool = True,
        enable_alerts: bool = True,
        profile_all_workflows: bool = False,
    ) -> bool:
        """
        Start the complete optimization ecosystem.

        Parameters
        ----------
        enable_dashboard : bool
            Whether to start the performance dashboard GUI
        enable_profiling : bool
            Whether to enable workflow profiling
        enable_alerts : bool
            Whether to enable alerting system
        profile_all_workflows : bool
            Whether to profile all workflows automatically

        Returns
        -------
        bool
            True if ecosystem started successfully
        """
        try:
            with self._lock:
                if self._state != EcosystemState.STOPPED:
                    logger.warning(f"Ecosystem already running (state: {self._state})")
                    return False

                self._set_state(EcosystemState.STARTING)

                # Initialize all components
                success = self._initialize_components(
                    enable_dashboard,
                    enable_profiling,
                    enable_alerts,
                    profile_all_workflows,
                )

                if success:
                    self._start_time = time.time()
                    self._set_state(EcosystemState.RUNNING)
                    self._start_status_updates()
                    logger.info("Optimization ecosystem started successfully")
                    return True
                self._set_state(EcosystemState.ERROR)
                logger.error("Failed to start optimization ecosystem")
                return False

        except Exception as e:
            logger.error(f"Error starting ecosystem: {e}")
            self._status.last_error = str(e)
            self._set_state(EcosystemState.ERROR)
            self.ecosystem_error.emit(str(e))
            return False

    def stop_ecosystem(self) -> bool:
        """
        Stop the optimization ecosystem and cleanup resources.

        Returns
        -------
        bool
            True if ecosystem stopped successfully
        """
        try:
            with self._lock:
                if self._state == EcosystemState.STOPPED:
                    return True

                self._set_state(EcosystemState.STOPPING)

                # Stop status updates
                if self._update_timer:
                    self._update_timer.stop()
                    self._update_timer = None

                # Stop all components
                self._stop_components()

                # Reset state
                self._start_time = None
                self._set_state(EcosystemState.STOPPED)

                logger.info("Optimization ecosystem stopped")
                return True

        except Exception as e:
            logger.error(f"Error stopping ecosystem: {e}")
            self._status.last_error = str(e)
            self.ecosystem_error.emit(str(e))
            return False

    def get_comprehensive_report(
        self,
        output_file: str | None = None,
        format: ReportFormat = ReportFormat.JSON,
    ) -> dict[str, Any]:
        """
        Generate comprehensive optimization ecosystem report.

        Parameters
        ----------
        output_file : str, optional
            File path to save report
        format : ReportFormat
            Output format for the report

        Returns
        -------
        dict
            Comprehensive ecosystem report
        """
        try:
            report = {
                "ecosystem_status": self._get_status_dict(),
                "timestamp": time.time(),
                "uptime_hours": self._get_uptime_hours(),
            }

            # Health monitoring report
            if self._health_monitor:
                health_report = self._health_monitor.get_health_report()
                report["health_status"] = {
                    "overall_score": health_report.overall_score,
                    "component_scores": health_report.component_scores,
                    "critical_issues": len(
                        [r for r in health_report.results if r.severity > 0.8]
                    ),
                }

            # Performance monitoring report
            if self._dashboard:
                performance_metrics = self._dashboard.get_current_metrics()
                report["performance_metrics"] = performance_metrics

            # Workflow profiling report
            if self._workflow_profiler:
                profiling_stats = self._workflow_profiler.get_profiling_statistics()
                report["profiling_statistics"] = profiling_stats

            # Bottleneck analysis report
            if self._bottleneck_analyzer:
                bottlenecks = self._bottleneck_analyzer.analyze_recent_profiles()
                report["bottleneck_analysis"] = {
                    "critical_bottlenecks": len(
                        [b for b in bottlenecks if b.severity == "critical"]
                    ),
                    "total_bottlenecks": len(bottlenecks),
                    "top_bottlenecks": [
                        {
                            "type": b.bottleneck_type.value,
                            "severity": b.severity,
                            "impact": b.performance_impact,
                        }
                        for b in sorted(
                            bottlenecks,
                            key=lambda x: x.performance_impact,
                            reverse=True,
                        )[:5]
                    ],
                }

            # Usage pattern analysis
            if self._pattern_miner:
                patterns = self._pattern_miner.mine_recent_patterns()
                report["usage_patterns"] = {
                    "total_patterns": len(patterns),
                    "cache_opportunities": len(
                        [p for p in patterns if "cache" in p.pattern_type.lower()]
                    ),
                    "preload_opportunities": len(
                        [p for p in patterns if "preload" in p.pattern_type.lower()]
                    ),
                }

            # Maintenance activities
            if self._maintenance_scheduler:
                maintenance_stats = (
                    self._maintenance_scheduler.get_maintenance_statistics()
                )
                report["maintenance_activities"] = maintenance_stats

            # Alert summary
            if self._alert_system:
                alert_summary = self._alert_system.get_alert_summary()
                report["alert_summary"] = alert_summary

            # Save report if requested
            if output_file and self._report_generator:
                self._report_generator.generate_ecosystem_report(
                    report, output_file, format
                )

            return report

        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def run_maintenance_cycle(self, force: bool = False) -> list[MaintenanceResult]:
        """
        Run a complete maintenance cycle across all systems.

        Parameters
        ----------
        force : bool
            Whether to force maintenance even if not scheduled

        Returns
        -------
        list
            List of maintenance results
        """
        try:
            results = []

            if self._maintenance_scheduler:
                # Run system maintenance
                maintenance_results = (
                    self._maintenance_scheduler.run_all_maintenance_tasks(force)
                )
                results.extend(maintenance_results)

                # Log results
                successful = len([r for r in maintenance_results if r.success])
                total = len(maintenance_results)
                logger.info(
                    f"Maintenance cycle completed: {successful}/{total} tasks successful"
                )

                # Update status
                self._status.total_maintenance_tasks += total

            return results

        except Exception as e:
            logger.error(f"Error running maintenance cycle: {e}")
            return []

    def analyze_current_performance(self) -> dict[str, Any]:
        """
        Analyze current system performance across all components.

        Returns
        -------
        dict
            Current performance analysis
        """
        try:
            analysis = {
                "timestamp": time.time(),
                "ecosystem_health": 0.0,
                "performance_trends": {},
                "optimization_effectiveness": {},
                "recommendations": [],
            }

            # Health analysis
            if self._health_monitor:
                health_report = self._health_monitor.get_health_report()
                analysis["ecosystem_health"] = health_report.overall_score

                # Add critical issues
                critical_issues = [r for r in health_report.results if r.severity > 0.8]
                if critical_issues:
                    analysis["critical_health_issues"] = len(critical_issues)
                    analysis["recommendations"].extend(
                        [
                            f"Address critical health issue: {issue.component_name}"
                            for issue in critical_issues
                        ]
                    )

            # Performance trends
            if self._dashboard:
                metrics = self._dashboard.get_current_metrics()
                analysis["performance_trends"] = {
                    "threading_efficiency": metrics.get("threading", {}).get(
                        "efficiency", 0
                    ),
                    "memory_utilization": metrics.get("memory", {}).get(
                        "utilization", 0
                    ),
                    "cache_hit_rate": metrics.get("cache", {}).get("hit_rate", 0),
                    "io_performance": metrics.get("io", {}).get("throughput", 0),
                }

            # Recent bottlenecks
            if self._bottleneck_analyzer:
                recent_bottlenecks = self._bottleneck_analyzer.analyze_recent_profiles()
                if recent_bottlenecks:
                    critical_bottlenecks = [
                        b for b in recent_bottlenecks if b.severity == "critical"
                    ]
                    analysis["critical_bottlenecks"] = len(critical_bottlenecks)

                    if critical_bottlenecks:
                        analysis["recommendations"].extend(
                            [
                                f"Optimize {b.bottleneck_type.value}: {b.description[:100]}"
                                for b in critical_bottlenecks[:3]
                            ]
                        )

            # Optimization opportunities
            if self._pattern_miner:
                patterns = self._pattern_miner.mine_recent_patterns()
                high_impact_patterns = [p for p in patterns if p.confidence > 0.8]
                if high_impact_patterns:
                    analysis["optimization_opportunities"] = len(high_impact_patterns)
                    analysis["recommendations"].extend(
                        [
                            f"Implement {p.pattern_type}: estimated {p.estimated_improvement:.1%} improvement"
                            for p in high_impact_patterns[:3]
                        ]
                    )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing current performance: {e}")
            return {"error": str(e), "timestamp": time.time()}

    def _initialize_components(
        self,
        enable_dashboard: bool,
        enable_profiling: bool,
        enable_alerts: bool,
        profile_all_workflows: bool,
    ) -> bool:
        """Initialize all ecosystem components."""
        try:
            # Initialize Subagent 1 components (Monitoring & Maintenance)
            self._health_monitor = get_health_monitor()
            self._health_monitor.start_monitoring()

            self._maintenance_scheduler = get_maintenance_scheduler()
            self._maintenance_scheduler.start_scheduler()

            if enable_alerts:
                self._alert_system = get_alert_system()
                self._alert_system.start_monitoring()

            if enable_dashboard:
                self._dashboard = create_performance_dashboard()
                self._dashboard.start_monitoring()

            # Initialize Subagent 2 components (Workflow Profiling)
            if enable_profiling:
                self._workflow_profiler = get_workflow_profiler()
                if profile_all_workflows:
                    self._workflow_profiler.enable_automatic_profiling()

                self._bottleneck_analyzer = get_bottleneck_analyzer()
                self._pattern_miner = get_pattern_miner()
                self._report_generator = get_report_generator()

            # Connect signals
            self._connect_component_signals()

            # Update component status
            self._update_component_status()

            return True

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            return False

    def _stop_components(self):
        """Stop all ecosystem components."""
        try:
            # Stop Subagent 1 components
            if self._health_monitor:
                self._health_monitor.stop_monitoring()

            if self._maintenance_scheduler:
                self._maintenance_scheduler.stop_scheduler()

            if self._alert_system:
                self._alert_system.stop_monitoring()

            if self._dashboard:
                self._dashboard.stop_monitoring()

            # Stop Subagent 2 components
            if self._workflow_profiler:
                self._workflow_profiler.stop_profiling()

            logger.info("All ecosystem components stopped")

        except Exception as e:
            logger.error(f"Error stopping components: {e}")

    def _connect_component_signals(self):
        """Connect signals from all components."""
        try:
            # Connect health monitor signals
            if self._health_monitor:
                self._health_monitor.health_alert.connect(self._on_health_alert)

            # Connect alert system signals
            if self._alert_system:
                self._alert_system.alert_triggered.connect(self._on_alert_triggered)

            # Connect maintenance signals
            if self._maintenance_scheduler:
                self._maintenance_scheduler.task_completed.connect(
                    self._on_maintenance_completed
                )

            # Connect profiling signals
            if self._bottleneck_analyzer:
                self._bottleneck_analyzer.bottleneck_detected.connect(
                    self._on_bottleneck_detected
                )

        except Exception as e:
            logger.error(f"Error connecting component signals: {e}")

    def _on_health_alert(self, component_name: str, message: str, severity: float):
        """Handle health alert from monitoring system."""
        logger.warning(
            f"Health alert [{component_name}]: {message} (severity: {severity:.2f})"
        )
        self.performance_alert.emit(component_name, message, int(severity * 10))

    def _on_alert_triggered(self, alert: Alert):
        """Handle triggered alert from alert system."""
        logger.warning(f"System alert: {alert.alert_type} - {alert.message}")
        self._status.active_alerts += 1

        if alert.severity == AlertSeverity.CRITICAL:
            self._status.critical_issues += 1

    def _on_maintenance_completed(self, task_name: str, success: bool):
        """Handle completed maintenance task."""
        logger.info(
            f"Maintenance task '{task_name}' {'succeeded' if success else 'failed'}"
        )
        self.maintenance_completed.emit(task_name, success)

    def _on_bottleneck_detected(
        self, bottleneck_type: str, description: str, severity: float
    ):
        """Handle detected bottleneck from analysis."""
        logger.warning(f"Bottleneck detected: {bottleneck_type} - {description}")
        self._status.total_bottlenecks_found += 1
        self.bottleneck_detected.emit(bottleneck_type, description, severity)

    def _set_state(self, new_state: EcosystemState):
        """Update ecosystem state and emit signal."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            self._status.state = new_state
            logger.info(
                f"Ecosystem state changed: {old_state.value} -> {new_state.value}"
            )
            self.state_changed.emit(new_state.value)

    def _start_status_updates(self):
        """Start regular status updates."""
        self._update_timer = QTimer()
        self._update_timer.timeout.connect(self._update_status)
        self._update_timer.start(5000)  # Update every 5 seconds

    def _update_status(self):
        """Update ecosystem status."""
        try:
            with self._lock:
                # Update uptime
                if self._start_time:
                    self._status.uptime_seconds = time.time() - self._start_time

                # Update component status
                self._update_component_status()

                # Update performance score
                self._update_performance_score()

                self._status.last_health_check = time.time()

        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def _update_component_status(self):
        """Update the status of all components."""
        self._status.components_active = {
            "health_monitor": self._health_monitor is not None
            and self._health_monitor.is_monitoring,
            "maintenance_scheduler": self._maintenance_scheduler is not None
            and self._maintenance_scheduler.is_running,
            "alert_system": self._alert_system is not None
            and self._alert_system.is_monitoring,
            "dashboard": self._dashboard is not None and self._dashboard.is_monitoring,
            "workflow_profiler": self._workflow_profiler is not None
            and self._workflow_profiler.is_profiling,
            "bottleneck_analyzer": self._bottleneck_analyzer is not None,
            "pattern_miner": self._pattern_miner is not None,
            "report_generator": self._report_generator is not None,
        }

    def _update_performance_score(self):
        """Calculate overall performance score."""
        try:
            scores = []

            # Health score
            if self._health_monitor:
                health_report = self._health_monitor.get_health_report()
                scores.append(health_report.overall_score)

            # Performance metrics score
            if self._dashboard:
                metrics = self._dashboard.get_current_metrics()
                threading_score = (
                    metrics.get("threading", {}).get("efficiency", 0) / 100.0
                )
                memory_score = metrics.get("memory", {}).get("efficiency", 0) / 100.0
                io_score = metrics.get("io", {}).get("efficiency", 0) / 100.0
                scores.extend([threading_score, memory_score, io_score])

            # Calculate weighted average
            if scores:
                self._status.performance_score = sum(scores) / len(scores)
            else:
                self._status.performance_score = 0.0

        except Exception as e:
            logger.error(f"Error updating performance score: {e}")
            self._status.performance_score = 0.0

    def _get_uptime_hours(self) -> float:
        """Get ecosystem uptime in hours."""
        if self._start_time:
            return (time.time() - self._start_time) / 3600.0
        return 0.0

    def _get_status_dict(self) -> dict[str, Any]:
        """Get status as dictionary."""
        return {
            "state": self._status.state.value,
            "uptime_hours": self._get_uptime_hours(),
            "performance_score": self._status.performance_score,
            "active_alerts": self._status.active_alerts,
            "critical_issues": self._status.critical_issues,
            "components_active": self._status.components_active,
            "total_workflows_profiled": self._status.total_workflows_profiled,
            "total_bottlenecks_found": self._status.total_bottlenecks_found,
            "total_maintenance_tasks": self._status.total_maintenance_tasks,
        }


# Global ecosystem instance
_ecosystem: OptimizationEcosystem | None = None
_ecosystem_lock = threading.RLock()


def get_optimization_ecosystem() -> OptimizationEcosystem:
    """
    Get the global optimization ecosystem instance.

    Returns
    -------
    OptimizationEcosystem
        Global ecosystem instance
    """
    global _ecosystem

    with _ecosystem_lock:
        if _ecosystem is None:
            _ecosystem = OptimizationEcosystem()

    return _ecosystem


def start_optimization_ecosystem(
    enable_dashboard: bool = True,
    enable_profiling: bool = True,
    enable_alerts: bool = True,
    profile_all_workflows: bool = False,
) -> bool:
    """
    Start the complete optimization ecosystem.

    Parameters
    ----------
    enable_dashboard : bool
        Whether to start the performance dashboard GUI
    enable_profiling : bool
        Whether to enable workflow profiling
    enable_alerts : bool
        Whether to enable alerting system
    profile_all_workflows : bool
        Whether to profile all workflows automatically

    Returns
    -------
    bool
        True if ecosystem started successfully
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.start_ecosystem(
        enable_dashboard, enable_profiling, enable_alerts, profile_all_workflows
    )


def stop_optimization_ecosystem() -> bool:
    """
    Stop the optimization ecosystem.

    Returns
    -------
    bool
        True if ecosystem stopped successfully
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.stop_ecosystem()


def get_ecosystem_status() -> EcosystemStatus:
    """
    Get current ecosystem status.

    Returns
    -------
    EcosystemStatus
        Current ecosystem status
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.status


def generate_ecosystem_report(
    output_file: str | None = None, format: ReportFormat = ReportFormat.JSON
) -> dict[str, Any]:
    """
    Generate comprehensive ecosystem report.

    Parameters
    ----------
    output_file : str, optional
        File path to save report
    format : ReportFormat
        Output format for the report

    Returns
    -------
    dict
        Comprehensive ecosystem report
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.get_comprehensive_report(output_file, format)


def run_ecosystem_maintenance(force: bool = False) -> list[MaintenanceResult]:
    """
    Run a complete maintenance cycle across all systems.

    Parameters
    ----------
    force : bool
        Whether to force maintenance even if not scheduled

    Returns
    -------
    list
        List of maintenance results
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.run_maintenance_cycle(force)


def analyze_ecosystem_performance() -> dict[str, Any]:
    """
    Analyze current ecosystem performance.

    Returns
    -------
    dict
        Current performance analysis
    """
    ecosystem = get_optimization_ecosystem()
    return ecosystem.analyze_current_performance()
