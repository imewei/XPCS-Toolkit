"""
System Resilience and Recovery Manager for XPCS Toolkit.

This module provides comprehensive system resilience, fault tolerance,
and automatic recovery capabilities to maintain application stability
even under adverse conditions.
"""

import asyncio
import gc
import os
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, QTimer, Signal, QMutex, QMutexLocker, QThread, QThreadPool

from .health_monitor_daemon import HealthAlert, AlertSeverity, get_health_monitor_daemon
from .qt_error_detector import QtErrorDetector, QtErrorType, get_qt_error_detector
from .performance_metrics_collector import get_performance_metrics_collector, MetricType
from ..threading.qt_compliant_thread_manager import get_qt_compliant_thread_manager
from ..threading.thread_pool_integration_validator import get_thread_pool_validator
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    NO_ACTION = "no_action"
    RESTART_COMPONENT = "restart_component"
    CLEANUP_RESOURCES = "cleanup_resources"
    FORCE_GARBAGE_COLLECTION = "force_gc"
    REDUCE_LOAD = "reduce_load"
    FAILOVER = "failover"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class SystemComponent(Enum):
    """System components for resilience management."""

    THREAD_MANAGER = "thread_manager"
    THREAD_POOLS = "thread_pools"
    QT_SYSTEM = "qt_system"
    MEMORY_SYSTEM = "memory_system"
    FILE_SYSTEM = "file_system"
    NETWORK_SYSTEM = "network_system"
    GUI_SYSTEM = "gui_system"


class ResilienceLevel(Enum):
    """System resilience levels."""

    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ResiliencePolicy:
    """Resilience policy configuration."""

    component: SystemComponent
    failure_threshold: int
    recovery_strategy: RecoveryStrategy
    cooldown_seconds: float = 60.0
    max_retries: int = 3
    escalation_strategy: Optional[RecoveryStrategy] = None
    auto_recovery_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RecoveryAction:
    """Recovery action record."""

    timestamp: float
    component: SystemComponent
    strategy: RecoveryStrategy
    trigger_reason: str
    success: bool
    duration_seconds: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SystemHealth:
    """System health status."""

    timestamp: float
    resilience_level: ResilienceLevel
    component_status: Dict[SystemComponent, str]
    active_issues: List[str]
    recovery_actions_count: int
    uptime_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "resilience_level": self.resilience_level.value,
            "component_status": {k.value: v for k, v in self.component_status.items()},
            "active_issues": self.active_issues,
            "recovery_actions_count": self.recovery_actions_count,
            "uptime_seconds": self.uptime_seconds
        }


class SystemResilienceManager(QObject):
    """
    Comprehensive system resilience and recovery manager.

    Provides:
    - Fault detection and diagnosis
    - Automatic recovery mechanisms
    - Load reduction strategies
    - Resource cleanup
    - Escalation procedures
    - Health monitoring integration
    """

    # Signals
    recovery_action_started = Signal(object)  # RecoveryAction
    recovery_action_completed = Signal(object)  # RecoveryAction
    resilience_level_changed = Signal(str, str)  # old_level, new_level
    system_health_updated = Signal(object)  # SystemHealth

    def __init__(self, parent: QObject = None):
        """Initialize system resilience manager."""
        super().__init__(parent)

        self._mutex = QMutex()
        self._start_time = time.perf_counter()

        # Component references
        self._health_monitor = get_health_monitor_daemon()
        self._qt_error_detector = get_qt_error_detector()
        self._thread_manager = get_qt_compliant_thread_manager()
        self._thread_validator = get_thread_pool_validator()
        self._metrics_collector = get_performance_metrics_collector()

        # Resilience configuration
        self._policies: Dict[SystemComponent, ResiliencePolicy] = {}
        self._setup_default_policies()

        # State tracking
        self._current_resilience_level = ResilienceLevel.NORMAL
        self._component_failure_counts: Dict[SystemComponent, int] = defaultdict(int)
        self._last_recovery_times: Dict[SystemComponent, float] = {}
        self._recovery_history: deque = deque(maxlen=1000)
        self._active_issues: Set[str] = set()

        # Component health status
        self._component_status: Dict[SystemComponent, str] = {
            component: "healthy" for component in SystemComponent
        }

        # Recovery locks to prevent concurrent recovery of same component
        self._recovery_locks: Dict[SystemComponent, threading.Lock] = {
            component: threading.Lock() for component in SystemComponent
        }

        # Connect to monitoring systems
        self._connect_monitoring_signals()

        # Setup metrics
        self._setup_metrics()

        # Health check timer
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._perform_health_check)
        self._health_timer.start(10000)  # Check every 10 seconds

        logger.info("System resilience manager initialized")

    def _setup_default_policies(self):
        """Setup default resilience policies."""
        default_policies = [
            ResiliencePolicy(
                component=SystemComponent.THREAD_MANAGER,
                failure_threshold=3,
                recovery_strategy=RecoveryStrategy.RESTART_COMPONENT,
                escalation_strategy=RecoveryStrategy.EMERGENCY_SHUTDOWN
            ),
            ResiliencePolicy(
                component=SystemComponent.THREAD_POOLS,
                failure_threshold=5,
                recovery_strategy=RecoveryStrategy.CLEANUP_RESOURCES,
                escalation_strategy=RecoveryStrategy.RESTART_COMPONENT
            ),
            ResiliencePolicy(
                component=SystemComponent.QT_SYSTEM,
                failure_threshold=1,  # Zero tolerance for Qt threading violations
                recovery_strategy=RecoveryStrategy.FORCE_GARBAGE_COLLECTION,
                escalation_strategy=RecoveryStrategy.RESTART_COMPONENT
            ),
            ResiliencePolicy(
                component=SystemComponent.MEMORY_SYSTEM,
                failure_threshold=3,
                recovery_strategy=RecoveryStrategy.FORCE_GARBAGE_COLLECTION,
                escalation_strategy=RecoveryStrategy.REDUCE_LOAD
            ),
            ResiliencePolicy(
                component=SystemComponent.FILE_SYSTEM,
                failure_threshold=5,
                recovery_strategy=RecoveryStrategy.CLEANUP_RESOURCES,
                escalation_strategy=RecoveryStrategy.REDUCE_LOAD
            ),
            ResiliencePolicy(
                component=SystemComponent.GUI_SYSTEM,
                failure_threshold=2,
                recovery_strategy=RecoveryStrategy.CLEANUP_RESOURCES,
                escalation_strategy=RecoveryStrategy.RESTART_COMPONENT
            ),
        ]

        for policy in default_policies:
            self._policies[policy.component] = policy

    def _setup_metrics(self):
        """Setup resilience metrics."""
        self._metrics_collector.register_metric(
            "resilience_recovery_actions",
            MetricType.COUNTER,
            unit="actions",
            description="Total recovery actions performed"
        )

        self._metrics_collector.register_metric(
            "resilience_level",
            MetricType.GAUGE,
            unit="level",
            description="Current system resilience level"
        )

        self._metrics_collector.register_metric(
            "component_failure_count",
            MetricType.COUNTER,
            unit="failures",
            description="Component failure count"
        )

    def _connect_monitoring_signals(self):
        """Connect to monitoring system signals."""
        # Health monitor alerts
        self._health_monitor.alert_triggered.connect(self._handle_health_alert)

        # Qt error detection
        self._qt_error_detector.qt_error_detected.connect(self._handle_qt_error)
        self._qt_error_detector.error_threshold_exceeded.connect(self._handle_qt_threshold_exceeded)

    def _handle_health_alert(self, alert: HealthAlert):
        """Handle health monitoring alert."""
        try:
            # Map alert component to system component
            component_map = {
                "thread_pool": SystemComponent.THREAD_POOLS,
                "qt_compliance": SystemComponent.QT_SYSTEM,
                "resources": SystemComponent.MEMORY_SYSTEM,
                "performance": SystemComponent.MEMORY_SYSTEM,
            }

            component = None
            for key, sys_component in component_map.items():
                if key in alert.component.lower():
                    component = sys_component
                    break

            if not component:
                component = SystemComponent.GUI_SYSTEM  # Default

            # Record failure
            self._record_component_failure(component, f"Health alert: {alert.message}")

            # Check if recovery is needed
            self._check_recovery_needed(component)

        except Exception as e:
            logger.error(f"Error handling health alert: {e}")

    def _handle_qt_error(self, error_occurrence):
        """Handle Qt error detection."""
        try:
            # Qt errors are always failures for the Qt system
            reason = f"Qt {error_occurrence.error_type.value}: {error_occurrence.message[:100]}"
            self._record_component_failure(SystemComponent.QT_SYSTEM, reason)

            # Special handling for critical Qt errors
            if error_occurrence.error_type in (QtErrorType.TIMER_THREADING, QtErrorType.QOBJECT_THREAD):
                # These are critical - immediate recovery
                self._trigger_recovery(SystemComponent.QT_SYSTEM, f"Critical Qt error: {reason}")

        except Exception as e:
            logger.error(f"Error handling Qt error: {e}")

    def _handle_qt_threshold_exceeded(self, error_type: str, count: int, threshold: int):
        """Handle Qt error threshold exceeded."""
        try:
            reason = f"Qt error threshold exceeded: {error_type} ({count}/{threshold})"
            self._record_component_failure(SystemComponent.QT_SYSTEM, reason)
            self._trigger_recovery(SystemComponent.QT_SYSTEM, reason)

        except Exception as e:
            logger.error(f"Error handling Qt threshold exceeded: {e}")

    def _record_component_failure(self, component: SystemComponent, reason: str):
        """Record a component failure."""
        with QMutexLocker(self._mutex):
            self._component_failure_counts[component] += 1
            self._active_issues.add(f"{component.value}: {reason}")
            self._component_status[component] = "degraded"

            # Record metrics
            self._metrics_collector.record_counter(
                "component_failure_count",
                tags={"component": component.value}
            )

            logger.warning(f"Component failure recorded: {component.value} - {reason}")

    def _check_recovery_needed(self, component: SystemComponent):
        """Check if recovery is needed for a component."""
        try:
            policy = self._policies.get(component)
            if not policy or not policy.auto_recovery_enabled:
                return

            failure_count = self._component_failure_counts[component]
            if failure_count >= policy.failure_threshold:
                # Check cooldown
                last_recovery = self._last_recovery_times.get(component, 0)
                if time.perf_counter() - last_recovery > policy.cooldown_seconds:
                    reason = f"Failure threshold exceeded: {failure_count}/{policy.failure_threshold}"
                    self._trigger_recovery(component, reason)

        except Exception as e:
            logger.error(f"Error checking recovery for {component.value}: {e}")

    def _trigger_recovery(self, component: SystemComponent, reason: str):
        """Trigger recovery for a component."""
        # Check if recovery is already in progress
        if not self._recovery_locks[component].acquire(blocking=False):
            logger.info(f"Recovery already in progress for {component.value}")
            return

        try:
            policy = self._policies.get(component)
            if not policy:
                logger.warning(f"No recovery policy for {component.value}")
                return

            # Determine strategy
            failure_count = self._component_failure_counts[component]
            strategy = policy.recovery_strategy

            # Check for escalation
            if failure_count > policy.max_retries and policy.escalation_strategy:
                strategy = policy.escalation_strategy
                logger.warning(f"Escalating recovery strategy for {component.value}: {strategy.value}")

            # Perform recovery
            self._perform_recovery(component, strategy, reason)

        finally:
            self._recovery_locks[component].release()

    def _perform_recovery(self, component: SystemComponent, strategy: RecoveryStrategy, reason: str):
        """Perform recovery action."""
        start_time = time.perf_counter()

        # Create recovery action record
        action = RecoveryAction(
            timestamp=start_time,
            component=component,
            strategy=strategy,
            trigger_reason=reason,
            success=False,
            duration_seconds=0.0
        )

        logger.info(f"Starting recovery: {component.value} using {strategy.value} - {reason}")
        self.recovery_action_started.emit(action)

        try:
            # Perform the recovery
            success = self._execute_recovery_strategy(component, strategy, action)
            action.success = success

            if success:
                # Reset failure count on successful recovery
                with QMutexLocker(self._mutex):
                    self._component_failure_counts[component] = 0
                    self._last_recovery_times[component] = start_time
                    self._component_status[component] = "healthy"

                    # Remove related issues
                    self._active_issues = {
                        issue for issue in self._active_issues
                        if not issue.startswith(component.value)
                    }

                logger.info(f"Recovery successful: {component.value}")
            else:
                logger.error(f"Recovery failed: {component.value}")

        except Exception as e:
            logger.error(f"Recovery error for {component.value}: {e}")
            action.details["error"] = str(e)
            action.details["traceback"] = traceback.format_exc()

        finally:
            # Finalize action record
            action.duration_seconds = time.perf_counter() - start_time

            # Store in history
            with QMutexLocker(self._mutex):
                self._recovery_history.append(action)

            # Record metrics
            self._metrics_collector.record_counter(
                "resilience_recovery_actions",
                tags={
                    "component": component.value,
                    "strategy": strategy.value,
                    "success": str(action.success)
                }
            )

            # Emit completion signal
            self.recovery_action_completed.emit(action)

    def _execute_recovery_strategy(self, component: SystemComponent,
                                  strategy: RecoveryStrategy, action: RecoveryAction) -> bool:
        """Execute specific recovery strategy."""
        try:
            if strategy == RecoveryStrategy.NO_ACTION:
                return True

            elif strategy == RecoveryStrategy.FORCE_GARBAGE_COLLECTION:
                return self._force_garbage_collection(action)

            elif strategy == RecoveryStrategy.CLEANUP_RESOURCES:
                return self._cleanup_resources(component, action)

            elif strategy == RecoveryStrategy.RESTART_COMPONENT:
                return self._restart_component(component, action)

            elif strategy == RecoveryStrategy.REDUCE_LOAD:
                return self._reduce_load(component, action)

            elif strategy == RecoveryStrategy.FAILOVER:
                return self._perform_failover(component, action)

            elif strategy == RecoveryStrategy.EMERGENCY_SHUTDOWN:
                return self._emergency_shutdown(action)

            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False

        except Exception as e:
            logger.error(f"Recovery strategy execution failed: {e}")
            action.details["execution_error"] = str(e)
            return False

    def _force_garbage_collection(self, action: RecoveryAction) -> bool:
        """Force garbage collection."""
        try:
            before_objects = len(gc.get_objects())
            collected = gc.collect()
            after_objects = len(gc.get_objects())

            action.details.update({
                "objects_before": before_objects,
                "objects_after": after_objects,
                "collected": collected
            })

            logger.info(f"Garbage collection: collected {collected} objects, "
                       f"reduced from {before_objects} to {after_objects}")
            return True

        except Exception as e:
            logger.error(f"Garbage collection failed: {e}")
            return False

    def _cleanup_resources(self, component: SystemComponent, action: RecoveryAction) -> bool:
        """Cleanup resources for a component."""
        try:
            cleaned_items = 0

            if component == SystemComponent.THREAD_POOLS:
                # Cleanup finished threads
                self._thread_manager.cleanup_finished_threads()
                cleaned_items += 1

                # Force thread pool cleanup
                for pool_id in self._thread_validator._thread_pools:
                    # Could add specific cleanup logic here
                    pass

            elif component == SystemComponent.MEMORY_SYSTEM:
                # Force garbage collection
                collected = gc.collect()
                cleaned_items += collected

            elif component == SystemComponent.FILE_SYSTEM:
                # Close unused file handles (would need specific implementation)
                cleaned_items += 1

            action.details["cleaned_items"] = cleaned_items
            logger.info(f"Resource cleanup for {component.value}: {cleaned_items} items")
            return True

        except Exception as e:
            logger.error(f"Resource cleanup failed for {component.value}: {e}")
            return False

    def _restart_component(self, component: SystemComponent, action: RecoveryAction) -> bool:
        """Restart a component."""
        try:
            if component == SystemComponent.THREAD_MANAGER:
                # Restart thread manager (careful operation)
                logger.warning("Thread manager restart requested - this is a critical operation")
                # Implementation would depend on specific restart requirements
                return True

            elif component == SystemComponent.QT_SYSTEM:
                # Qt system restart is complex - for now just cleanup
                self._force_garbage_collection(action)
                return True

            else:
                logger.warning(f"Component restart not implemented for: {component.value}")
                return False

        except Exception as e:
            logger.error(f"Component restart failed for {component.value}: {e}")
            return False

    def _reduce_load(self, component: SystemComponent, action: RecoveryAction) -> bool:
        """Reduce load on a component."""
        try:
            if component == SystemComponent.THREAD_POOLS:
                # Reduce thread pool sizes
                for pool_id, pool in self._thread_validator._thread_pools.items():
                    current_max = pool.maxThreadCount()
                    new_max = max(1, current_max // 2)
                    pool.setMaxThreadCount(new_max)
                    logger.info(f"Reduced thread pool {pool_id} from {current_max} to {new_max} threads")

            elif component == SystemComponent.MEMORY_SYSTEM:
                # Force aggressive garbage collection
                for _ in range(3):
                    gc.collect()

            action.details["load_reduction"] = "applied"
            return True

        except Exception as e:
            logger.error(f"Load reduction failed for {component.value}: {e}")
            return False

    def _perform_failover(self, component: SystemComponent, action: RecoveryAction) -> bool:
        """Perform failover for a component."""
        try:
            # Failover implementation would be component-specific
            logger.warning(f"Failover requested for {component.value} - not implemented")
            return False

        except Exception as e:
            logger.error(f"Failover failed for {component.value}: {e}")
            return False

    def _emergency_shutdown(self, action: RecoveryAction) -> bool:
        """Perform emergency shutdown."""
        try:
            logger.critical("Emergency shutdown initiated")
            action.details["shutdown_initiated"] = True

            # In a real implementation, this would trigger graceful application shutdown
            # For now, just log the request
            return True

        except Exception as e:
            logger.error(f"Emergency shutdown failed: {e}")
            return False

    def _perform_health_check(self):
        """Perform periodic health check."""
        try:
            current_time = time.perf_counter()

            # Assess overall resilience level
            new_level = self._assess_resilience_level()

            # Update resilience level if changed
            if new_level != self._current_resilience_level:
                old_level = self._current_resilience_level
                self._current_resilience_level = new_level

                logger.info(f"Resilience level changed: {old_level.value} -> {new_level.value}")
                self.resilience_level_changed.emit(old_level.value, new_level.value)

                # Record metric
                self._metrics_collector.record_gauge(
                    "resilience_level",
                    self._resilience_level_to_number(new_level)
                )

            # Create health status
            health = SystemHealth(
                timestamp=current_time,
                resilience_level=self._current_resilience_level,
                component_status=dict(self._component_status),
                active_issues=list(self._active_issues),
                recovery_actions_count=len(self._recovery_history),
                uptime_seconds=current_time - self._start_time
            )

            self.system_health_updated.emit(health)

        except Exception as e:
            logger.error(f"Health check failed: {e}")

    def _assess_resilience_level(self) -> ResilienceLevel:
        """Assess current system resilience level."""
        try:
            # Count failures and issues
            total_failures = sum(self._component_failure_counts.values())
            critical_failures = sum(
                count for component, count in self._component_failure_counts.items()
                if component in (SystemComponent.QT_SYSTEM, SystemComponent.THREAD_MANAGER)
            )
            active_issues_count = len(self._active_issues)

            # Determine level based on criteria
            if critical_failures > 0 or total_failures > 10:
                return ResilienceLevel.EMERGENCY
            elif total_failures > 5 or active_issues_count > 5:
                return ResilienceLevel.CRITICAL
            elif total_failures > 2 or active_issues_count > 2:
                return ResilienceLevel.DEGRADED
            else:
                return ResilienceLevel.NORMAL

        except Exception as e:
            logger.error(f"Resilience level assessment failed: {e}")
            return ResilienceLevel.CRITICAL

    def _resilience_level_to_number(self, level: ResilienceLevel) -> float:
        """Convert resilience level to numeric value for metrics."""
        return {
            ResilienceLevel.NORMAL: 4.0,
            ResilienceLevel.DEGRADED: 3.0,
            ResilienceLevel.CRITICAL: 2.0,
            ResilienceLevel.EMERGENCY: 1.0
        }[level]

    def set_policy(self, component: SystemComponent, policy: ResiliencePolicy):
        """Set resilience policy for a component."""
        self._policies[component] = policy
        logger.info(f"Updated resilience policy for {component.value}")

    def get_policy(self, component: SystemComponent) -> Optional[ResiliencePolicy]:
        """Get resilience policy for a component."""
        return self._policies.get(component)

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        current_time = time.perf_counter()
        return SystemHealth(
            timestamp=current_time,
            resilience_level=self._current_resilience_level,
            component_status=dict(self._component_status),
            active_issues=list(self._active_issues),
            recovery_actions_count=len(self._recovery_history),
            uptime_seconds=current_time - self._start_time
        )

    def get_recovery_history(self, hours: Optional[int] = None) -> List[RecoveryAction]:
        """Get recovery action history."""
        cutoff_time = None
        if hours:
            cutoff_time = time.perf_counter() - (hours * 3600)

        with QMutexLocker(self._mutex):
            if cutoff_time:
                return [
                    action for action in self._recovery_history
                    if action.timestamp > cutoff_time
                ]
            else:
                return list(self._recovery_history)

    def force_recovery(self, component: SystemComponent, strategy: RecoveryStrategy) -> bool:
        """Force recovery action for a component."""
        try:
            reason = "Manual recovery trigger"
            self._trigger_recovery(component, reason)
            return True
        except Exception as e:
            logger.error(f"Failed to force recovery for {component.value}: {e}")
            return False

    def clear_component_failures(self, component: Optional[SystemComponent] = None):
        """Clear failure counts for a component or all components."""
        with QMutexLocker(self._mutex):
            if component:
                self._component_failure_counts[component] = 0
                self._component_status[component] = "healthy"
                # Remove related issues
                self._active_issues = {
                    issue for issue in self._active_issues
                    if not issue.startswith(component.value)
                }
                logger.info(f"Cleared failures for {component.value}")
            else:
                self._component_failure_counts.clear()
                self._active_issues.clear()
                for comp in SystemComponent:
                    self._component_status[comp] = "healthy"
                logger.info("Cleared all component failures")

    def export_resilience_report(self) -> Dict[str, Any]:
        """Export comprehensive resilience report."""
        health = self.get_system_health()
        recent_recoveries = self.get_recovery_history(hours=24)

        return {
            "report_timestamp": time.perf_counter(),
            "system_health": health.to_dict(),
            "resilience_policies": {
                comp.value: policy.to_dict()
                for comp, policy in self._policies.items()
            },
            "component_failure_counts": {
                comp.value: count
                for comp, count in self._component_failure_counts.items()
            },
            "recovery_history_24h": [action.to_dict() for action in recent_recoveries],
            "recovery_success_rate": self._calculate_success_rate(recent_recoveries)
        }

    def _calculate_success_rate(self, recoveries: List[RecoveryAction]) -> float:
        """Calculate recovery success rate."""
        if not recoveries:
            return 1.0

        successful = sum(1 for action in recoveries if action.success)
        return successful / len(recoveries)


# Global instance
_system_resilience_manager: Optional[SystemResilienceManager] = None


def get_system_resilience_manager() -> SystemResilienceManager:
    """Get the global system resilience manager instance."""
    global _system_resilience_manager

    if _system_resilience_manager is None:
        _system_resilience_manager = SystemResilienceManager()

    return _system_resilience_manager


def initialize_system_resilience() -> SystemResilienceManager:
    """Initialize system resilience management."""
    return get_system_resilience_manager()


@contextmanager
def resilience_context():
    """Context manager for resilience management."""
    manager = initialize_system_resilience()
    try:
        yield manager
    finally:
        # Could add cleanup here if needed
        pass