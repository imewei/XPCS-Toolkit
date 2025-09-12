"""
Automated Alert System for XPCS Toolkit CPU Optimizations.

This module provides comprehensive performance degradation detection, automated alerting,
and corrective action systems for all CPU optimization components. It integrates with
existing monitoring systems to provide proactive issue detection and resolution.

Features:
- Real-time performance degradation detection
- Automated corrective actions based on alert types
- Escalation system for persistent issues
- Integration with existing logging and monitoring systems
- Configurable alert thresholds and responses
- Historical alert tracking and trend analysis
- Multiple alert channels (logging, signals, file outputs)
"""

from __future__ import annotations

import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from PySide6.QtCore import QObject, QTimer, Signal

# Import existing optimization and monitoring systems
from .optimization_health_monitor import (
    get_health_monitor,
    HealthCheckResult,
    HealthStatus,
)
from .maintenance_scheduler import get_maintenance_scheduler
from .memory_utils import SystemMemoryMonitor
from .logging_config import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertCategory(Enum):
    """Categories of alerts."""

    PERFORMANCE = "performance"
    MEMORY = "memory"
    THREADING = "threading"
    CACHE = "cache"
    SYSTEM = "system"
    HEALTH = "health"


class AlertStatus(Enum):
    """Status of alerts."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


@dataclass
class AlertRule:
    """Definition of an alert rule."""

    rule_id: str
    name: str
    category: AlertCategory
    severity: AlertSeverity

    # Detection criteria
    metric_name: str
    threshold_value: float
    comparison_operator: str  # "gt", "lt", "eq", "gte", "lte"
    duration_seconds: float = 0.0  # How long condition must persist

    # Action configuration
    corrective_actions: List[str] = field(default_factory=list)
    escalation_time_seconds: float = 300.0  # 5 minutes
    max_escalations: int = 3

    # Conditions
    enabled: bool = True
    conditions: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    last_triggered: float = 0.0
    trigger_count: int = 0
    false_positive_count: int = 0


@dataclass
class Alert:
    """Individual alert instance."""

    alert_id: str
    rule_id: str
    category: AlertCategory
    severity: AlertSeverity

    # Alert details
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float

    # Status tracking
    status: AlertStatus = AlertStatus.ACTIVE
    created_time: float = field(default_factory=time.time)
    updated_time: float = field(default_factory=time.time)
    resolved_time: Optional[float] = None

    # Context
    component: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)
    escalation_level: int = 0

    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.status == AlertStatus.ACTIVE

    def age_seconds(self) -> float:
        """Get age of alert in seconds."""
        return time.time() - self.created_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "status": self.status.value,
            "created_time": self.created_time,
            "updated_time": self.updated_time,
            "resolved_time": self.resolved_time,
            "component": self.component,
            "details": self.details,
            "actions_taken": self.actions_taken,
            "escalation_level": self.escalation_level,
            "age_seconds": self.age_seconds(),
        }


class CorrectiveAction:
    """Definition of corrective actions that can be taken."""

    def __init__(
        self, action_id: str, name: str, action_function: Callable[[Alert], bool]
    ):
        self.action_id = action_id
        self.name = name
        self.action_function = action_function
        self.execution_count = 0
        self.success_count = 0
        self.average_execution_time = 0.0


class AlertSystem(QObject):
    """
    Comprehensive alert system for CPU optimization monitoring and corrective actions.
    """

    # Signals
    alert_triggered = Signal(object)  # Alert
    alert_resolved = Signal(str)  # alert_id
    alert_escalated = Signal(object)  # Alert
    corrective_action_taken = Signal(str, str, bool)  # alert_id, action_name, success
    critical_system_issue = Signal(str, str)  # component, message

    def __init__(self, check_interval: float = 10.0, parent=None):
        """
        Initialize the alert system.

        Args:
            check_interval: How often to check for alert conditions (seconds)
            parent: Qt parent object
        """
        super().__init__(parent)

        self.check_interval = check_interval
        self._running = False
        self._lock = threading.RLock()

        # Alert management
        self._alert_rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque[Alert] = deque(maxlen=10000)

        # Corrective actions registry
        self._corrective_actions: Dict[str, CorrectiveAction] = {}

        # System integrations
        self._health_monitor = get_health_monitor()
        self._maintenance_scheduler = get_maintenance_scheduler()
        self._memory_monitor = SystemMemoryMonitor()

        # Metrics tracking for alert evaluation
        self._metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_metric_update: Dict[str, float] = {}

        # Timer for alert checking
        self._timer = QTimer()
        self._timer.timeout.connect(self._check_alert_conditions)

        # Alert statistics
        self._alert_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Initialize default alert rules and actions
        self._register_default_alert_rules()
        self._register_default_corrective_actions()

        # Connect to monitoring systems
        self._connect_monitoring_signals()

        logger.info("AlertSystem initialized")

    def start_alert_system(self) -> None:
        """Start the alert system."""
        with self._lock:
            if self._running:
                logger.warning("Alert system already running")
                return

            self._running = True
            self._timer.start(int(self.check_interval * 1000))  # Convert to ms
            logger.info("Alert system started")

    def stop_alert_system(self) -> None:
        """Stop the alert system."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._timer.stop()
            logger.info("Alert system stopped")

    def register_alert_rule(self, rule: AlertRule) -> None:
        """
        Register a new alert rule.

        Args:
            rule: The alert rule to register
        """
        with self._lock:
            self._alert_rules[rule.rule_id] = rule
            logger.debug(f"Registered alert rule: {rule.rule_id}")

    def register_corrective_action(self, action: CorrectiveAction) -> None:
        """
        Register a corrective action.

        Args:
            action: The corrective action to register
        """
        with self._lock:
            self._corrective_actions[action.action_id] = action
            logger.debug(f"Registered corrective action: {action.action_id}")

    def trigger_alert(
        self,
        rule_id: str,
        current_value: float,
        component: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Manually trigger an alert.

        Args:
            rule_id: ID of the alert rule
            current_value: Current metric value
            component: Component that triggered the alert
            details: Additional alert details

        Returns:
            Alert ID if alert was created, None otherwise
        """
        if rule_id not in self._alert_rules:
            logger.error(f"Alert rule '{rule_id}' not found")
            return None

        rule = self._alert_rules[rule_id]

        # Create alert
        alert_id = f"{rule_id}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule_id,
            category=rule.category,
            severity=rule.severity,
            title=f"{rule.name} Alert",
            message=f"{rule.metric_name} value {current_value} exceeds threshold {rule.threshold_value}",
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            component=component,
            details=details or {},
        )

        return self._process_new_alert(alert, rule)

    def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: ID of the alert to resolve
            resolution_message: Optional resolution message

        Returns:
            True if alert was resolved successfully
        """
        with self._lock:
            if alert_id not in self._active_alerts:
                logger.warning(f"Alert '{alert_id}' not found in active alerts")
                return False

            alert = self._active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_time = time.time()
            alert.updated_time = time.time()

            if resolution_message:
                alert.actions_taken.append(f"Resolved: {resolution_message}")

            # Move to history
            self._alert_history.append(alert)
            del self._active_alerts[alert_id]

            # Update statistics
            self._alert_stats[alert.category.value]["resolved"] += 1

            # Emit signal
            self.alert_resolved.emit(alert_id)

            logger.info(f"Alert '{alert_id}' resolved: {resolution_message}")
            return True

    def get_active_alerts(self) -> List[Alert]:
        """Get list of all active alerts."""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        with self._lock:
            stats = {
                "active_alerts": len(self._active_alerts),
                "total_alerts_today": 0,
                "by_category": dict(self._alert_stats),
                "by_severity": defaultdict(int),
                "top_components": defaultdict(int),
                "escalation_stats": {
                    "total_escalations": 0,
                    "auto_resolved": 0,
                    "manual_intervention": 0,
                },
            }

            # Calculate today's alerts
            today_start = time.time() - (24 * 3600)  # 24 hours ago
            today_alerts = [
                alert
                for alert in self._alert_history
                if alert.created_time > today_start
            ]
            stats["total_alerts_today"] = len(today_alerts)

            # Calculate severity and component statistics
            all_recent_alerts = list(self._active_alerts.values()) + list(today_alerts)

            for alert in all_recent_alerts:
                stats["by_severity"][alert.severity.value] += 1
                if alert.component:
                    stats["top_components"][alert.component] += 1
                if alert.escalation_level > 0:
                    stats["escalation_stats"]["total_escalations"] += 1

            return stats

    def update_metric_value(self, metric_name: str, value: float) -> None:
        """
        Update a metric value for alert evaluation.

        Args:
            metric_name: Name of the metric
            value: Current metric value
        """
        with self._lock:
            self._metric_values[metric_name].append((time.time(), value))
            self._last_metric_update[metric_name] = time.time()

    def _check_alert_conditions(self) -> None:
        """Check all alert rules for triggering conditions."""
        if not self._running:
            return

        current_time = time.time()

        # Check each alert rule
        for rule_id, rule in self._alert_rules.items():
            if not rule.enabled:
                continue

            try:
                # Get current metric value
                metric_history = self._metric_values.get(rule.metric_name)
                if not metric_history:
                    continue

                # Get the most recent value
                last_timestamp, current_value = metric_history[-1]

                # Skip if metric is too old
                if current_time - last_timestamp > 60.0:  # 1 minute threshold
                    continue

                # Check if condition is met
                condition_met = self._evaluate_condition(
                    current_value, rule.threshold_value, rule.comparison_operator
                )

                if condition_met:
                    # Check duration requirement
                    if rule.duration_seconds > 0:
                        # Check if condition has been met for required duration
                        duration_start = current_time - rule.duration_seconds
                        duration_values = [
                            value
                            for timestamp, value in metric_history
                            if timestamp >= duration_start
                        ]

                        if len(duration_values) < 2:  # Not enough data points
                            continue

                        # Check if all values in duration meet condition
                        duration_met = all(
                            self._evaluate_condition(
                                value, rule.threshold_value, rule.comparison_operator
                            )
                            for value in duration_values
                        )

                        if not duration_met:
                            continue

                    # Check if we already have an active alert for this rule
                    existing_alert = None
                    for alert in self._active_alerts.values():
                        if alert.rule_id == rule_id:
                            existing_alert = alert
                            break

                    if existing_alert:
                        # Update existing alert
                        existing_alert.current_value = current_value
                        existing_alert.updated_time = current_time

                        # Check for escalation
                        if (
                            current_time - existing_alert.created_time
                            >= rule.escalation_time_seconds
                            and existing_alert.escalation_level < rule.max_escalations
                        ):
                            self._escalate_alert(existing_alert, rule)
                    else:
                        # Create new alert
                        alert_id = f"{rule_id}_{int(current_time)}"
                        alert = Alert(
                            alert_id=alert_id,
                            rule_id=rule_id,
                            category=rule.category,
                            severity=rule.severity,
                            title=f"{rule.name}",
                            message=f"{rule.metric_name} value {current_value} {rule.comparison_operator} threshold {rule.threshold_value}",
                            metric_name=rule.metric_name,
                            current_value=current_value,
                            threshold_value=rule.threshold_value,
                        )

                        self._process_new_alert(alert, rule)

                else:
                    # Condition not met - check if we can resolve existing alerts
                    alerts_to_resolve = [
                        alert
                        for alert in self._active_alerts.values()
                        if alert.rule_id == rule_id
                    ]

                    for alert in alerts_to_resolve:
                        self.resolve_alert(
                            alert.alert_id, "Metric returned to normal range"
                        )

            except Exception as e:
                logger.error(f"Error checking alert rule '{rule_id}': {e}")

    def _evaluate_condition(
        self, current_value: float, threshold: float, operator: str
    ) -> bool:
        """Evaluate if a condition is met."""
        if operator == "gt":
            return current_value > threshold
        elif operator == "gte":
            return current_value >= threshold
        elif operator == "lt":
            return current_value < threshold
        elif operator == "lte":
            return current_value <= threshold
        elif operator == "eq":
            return abs(current_value - threshold) < 1e-6
        else:
            logger.error(f"Unknown comparison operator: {operator}")
            return False

    def _process_new_alert(self, alert: Alert, rule: AlertRule) -> str:
        """Process a newly created alert."""
        with self._lock:
            # Add to active alerts
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)

            # Update statistics
            self._alert_stats[alert.category.value]["triggered"] += 1
            rule.trigger_count += 1
            rule.last_triggered = time.time()

            # Execute corrective actions
            if rule.corrective_actions:
                for action_id in rule.corrective_actions:
                    if action_id in self._corrective_actions:
                        self._execute_corrective_action(
                            alert, self._corrective_actions[action_id]
                        )

            # Emit signals
            self.alert_triggered.emit(alert)

            if alert.severity == AlertSeverity.CRITICAL:
                self.critical_system_issue.emit(
                    alert.component or "system", alert.message
                )

            logger.warning(f"Alert triggered: {alert.title} - {alert.message}")

            return alert.alert_id

    def _escalate_alert(self, alert: Alert, rule: AlertRule) -> None:
        """Escalate an alert to the next level."""
        alert.escalation_level += 1
        alert.updated_time = time.time()

        # Add escalation action
        escalation_message = f"Alert escalated to level {alert.escalation_level}"
        alert.actions_taken.append(escalation_message)

        # Execute additional corrective actions for escalated alerts
        if alert.escalation_level == 1:
            # First escalation - try more aggressive actions
            aggressive_actions = ["emergency_cache_cleanup", "critical_memory_cleanup"]
            for action_id in aggressive_actions:
                if action_id in self._corrective_actions:
                    self._execute_corrective_action(
                        alert, self._corrective_actions[action_id]
                    )

        # Emit escalation signal
        self.alert_escalated.emit(alert)

        logger.warning(
            f"Alert escalated: {alert.alert_id} (level {alert.escalation_level})"
        )

    def _execute_corrective_action(
        self, alert: Alert, action: CorrectiveAction
    ) -> bool:
        """Execute a corrective action for an alert."""
        try:
            start_time = time.perf_counter()

            logger.info(
                f"Executing corrective action '{action.name}' for alert '{alert.alert_id}'"
            )

            success = action.action_function(alert)
            execution_time = time.perf_counter() - start_time

            # Update action statistics
            action.execution_count += 1
            if success:
                action.success_count += 1

            # Update average execution time
            action.average_execution_time = (
                action.average_execution_time * (action.execution_count - 1)
                + execution_time
            ) / action.execution_count

            # Record action in alert
            status_text = "completed successfully" if success else "failed"
            alert.actions_taken.append(
                f"Action '{action.name}' {status_text} in {execution_time:.2f}s"
            )
            alert.updated_time = time.time()

            # Emit signal
            self.corrective_action_taken.emit(alert.alert_id, action.name, success)

            logger.info(
                f"Corrective action '{action.name}' {status_text} for alert '{alert.alert_id}'"
            )
            return success

        except Exception as e:
            error_msg = f"Corrective action '{action.name}' failed: {str(e)}"
            logger.error(error_msg)
            alert.actions_taken.append(error_msg)

            self.corrective_action_taken.emit(alert.alert_id, action.name, False)
            return False

    def _connect_monitoring_signals(self) -> None:
        """Connect to existing monitoring system signals."""
        # Connect to health monitor
        self._health_monitor.component_needs_attention.connect(
            self._on_component_needs_attention
        )
        self._health_monitor.health_status_updated.connect(
            self._on_health_status_updated
        )

        # Connect to maintenance scheduler
        self._maintenance_scheduler.critical_maintenance_needed.connect(
            self._on_critical_maintenance_needed
        )

    def _on_component_needs_attention(self, component_name: str, message: str) -> None:
        """Handle component attention alerts from health monitor."""
        # Create a health-based alert
        alert_id = f"health_{component_name}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id="health_check",
            category=AlertCategory.HEALTH,
            severity=AlertSeverity.WARNING,
            title=f"Component Health Issue: {component_name}",
            message=message,
            metric_name="health_score",
            current_value=0.0,  # Would be filled from actual health score
            threshold_value=0.7,  # Health threshold
            component=component_name,
        )

        with self._lock:
            self._active_alerts[alert_id] = alert

        self.alert_triggered.emit(alert)

    def _on_health_status_updated(
        self, component_name: str, health_result: HealthCheckResult
    ) -> None:
        """Handle health status updates."""
        # Update metric values for alert evaluation
        self.update_metric_value(f"{component_name}_health_score", health_result.score)

        # Check for critical health issues
        if health_result.status == HealthStatus.CRITICAL:
            self.trigger_alert(
                "critical_health_alert",
                health_result.score,
                component_name,
                {
                    "health_status": health_result.status.value,
                    "message": health_result.message,
                },
            )

    def _on_critical_maintenance_needed(self, component: str, reason: str) -> None:
        """Handle critical maintenance alerts."""
        alert_id = f"maintenance_{component}_{int(time.time())}"
        alert = Alert(
            alert_id=alert_id,
            rule_id="critical_maintenance",
            category=AlertCategory.SYSTEM,
            severity=AlertSeverity.CRITICAL,
            title=f"Critical Maintenance Required: {component}",
            message=reason,
            metric_name="maintenance_urgency",
            current_value=1.0,
            threshold_value=0.8,
            component=component,
        )

        with self._lock:
            self._active_alerts[alert_id] = alert

        self.critical_system_issue.emit(component, reason)
        self.alert_triggered.emit(alert)

    def _register_default_alert_rules(self) -> None:
        """Register default alert rules for optimization monitoring."""

        # Memory usage alert
        self.register_alert_rule(
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                category=AlertCategory.MEMORY,
                severity=AlertSeverity.WARNING,
                metric_name="memory_percent",
                threshold_value=80.0,
                comparison_operator="gt",
                duration_seconds=30.0,
                corrective_actions=["memory_cleanup", "cache_reduction"],
                escalation_time_seconds=300.0,
            )
        )

        # Critical memory usage alert
        self.register_alert_rule(
            AlertRule(
                rule_id="critical_memory_usage",
                name="Critical Memory Usage",
                category=AlertCategory.MEMORY,
                severity=AlertSeverity.CRITICAL,
                metric_name="memory_percent",
                threshold_value=95.0,
                comparison_operator="gt",
                duration_seconds=10.0,
                corrective_actions=[
                    "emergency_memory_cleanup",
                    "critical_cache_cleanup",
                ],
                escalation_time_seconds=120.0,
            )
        )

        # Thread pool saturation alert
        self.register_alert_rule(
            AlertRule(
                rule_id="thread_pool_saturation",
                name="Thread Pool Saturation",
                category=AlertCategory.THREADING,
                severity=AlertSeverity.WARNING,
                metric_name="thread_utilization",
                threshold_value=90.0,
                comparison_operator="gt",
                duration_seconds=60.0,
                corrective_actions=["thread_pool_expansion", "task_queue_optimization"],
                escalation_time_seconds=600.0,
            )
        )

        # Cache hit rate degradation alert
        self.register_alert_rule(
            AlertRule(
                rule_id="low_cache_hit_rate",
                name="Low Cache Hit Rate",
                category=AlertCategory.CACHE,
                severity=AlertSeverity.WARNING,
                metric_name="cache_hit_rate",
                threshold_value=50.0,
                comparison_operator="lt",
                duration_seconds=120.0,
                corrective_actions=["cache_optimization", "cache_warming"],
                escalation_time_seconds=900.0,
            )
        )

        # System performance degradation alert
        self.register_alert_rule(
            AlertRule(
                rule_id="performance_degradation",
                name="Performance Degradation",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric_name="overall_performance_score",
                threshold_value=70.0,
                comparison_operator="lt",
                duration_seconds=180.0,
                corrective_actions=["performance_optimization", "system_tuning"],
                escalation_time_seconds=1200.0,
            )
        )

    def _register_default_corrective_actions(self) -> None:
        """Register default corrective actions."""

        # Memory cleanup action
        self.register_corrective_action(
            CorrectiveAction(
                action_id="memory_cleanup",
                name="Memory Cleanup",
                action_function=self._memory_cleanup_action,
            )
        )

        # Emergency memory cleanup action
        self.register_corrective_action(
            CorrectiveAction(
                action_id="emergency_memory_cleanup",
                name="Emergency Memory Cleanup",
                action_function=self._emergency_memory_cleanup_action,
            )
        )

        # Cache optimization action
        self.register_corrective_action(
            CorrectiveAction(
                action_id="cache_optimization",
                name="Cache Optimization",
                action_function=self._cache_optimization_action,
            )
        )

        # Thread pool expansion action
        self.register_corrective_action(
            CorrectiveAction(
                action_id="thread_pool_expansion",
                name="Thread Pool Expansion",
                action_function=self._thread_pool_expansion_action,
            )
        )

        # Performance optimization action
        self.register_corrective_action(
            CorrectiveAction(
                action_id="performance_optimization",
                name="Performance Optimization",
                action_function=self._performance_optimization_action,
            )
        )

    # Corrective action implementations

    def _memory_cleanup_action(self, alert: Alert) -> bool:
        """Perform memory cleanup."""
        try:
            # Trigger maintenance scheduler memory optimization
            success = self._maintenance_scheduler.run_task_now("memory_optimization")

            if success:
                logger.info("Memory cleanup corrective action completed successfully")
                return True
            else:
                logger.warning("Memory cleanup corrective action failed")
                return False

        except Exception as e:
            logger.error(f"Memory cleanup action failed: {e}")
            return False

    def _emergency_memory_cleanup_action(self, alert: Alert) -> bool:
        """Perform emergency memory cleanup."""
        try:
            # Trigger critical memory cleanup
            success = self._maintenance_scheduler.run_task_now(
                "critical_memory_cleanup"
            )

            if success:
                logger.info(
                    "Emergency memory cleanup corrective action completed successfully"
                )
                return True
            else:
                logger.warning("Emergency memory cleanup corrective action failed")
                return False

        except Exception as e:
            logger.error(f"Emergency memory cleanup action failed: {e}")
            return False

    def _cache_optimization_action(self, alert: Alert) -> bool:
        """Perform cache optimization."""
        try:
            # Trigger cache cleanup
            success = self._maintenance_scheduler.run_task_now("cache_cleanup")

            if success:
                logger.info(
                    "Cache optimization corrective action completed successfully"
                )
                return True
            else:
                logger.warning("Cache optimization corrective action failed")
                return False

        except Exception as e:
            logger.error(f"Cache optimization action failed: {e}")
            return False

    def _thread_pool_expansion_action(self, alert: Alert) -> bool:
        """Perform thread pool expansion."""
        try:
            # Trigger thread pool tuning
            success = self._maintenance_scheduler.run_task_now("thread_pool_tuning")

            if success:
                logger.info(
                    "Thread pool expansion corrective action completed successfully"
                )
                return True
            else:
                logger.warning("Thread pool expansion corrective action failed")
                return False

        except Exception as e:
            logger.error(f"Thread pool expansion action failed: {e}")
            return False

    def _performance_optimization_action(self, alert: Alert) -> bool:
        """Perform general performance optimization."""
        try:
            # Run multiple optimization tasks
            actions_success = []

            # Cache cleanup
            actions_success.append(
                self._maintenance_scheduler.run_task_now("cache_cleanup")
            )

            # Memory optimization
            actions_success.append(
                self._maintenance_scheduler.run_task_now("memory_optimization")
            )

            # Signal optimization
            actions_success.append(
                self._maintenance_scheduler.run_task_now("signal_optimization")
            )

            # Consider successful if at least 2 out of 3 actions succeed
            success_count = sum(1 for success in actions_success if success)
            overall_success = success_count >= 2

            if overall_success:
                logger.info(
                    "Performance optimization corrective action completed successfully"
                )
                return True
            else:
                logger.warning(
                    "Performance optimization corrective action partially failed"
                )
                return False

        except Exception as e:
            logger.error(f"Performance optimization action failed: {e}")
            return False


# Global instance
_alert_system_instance: Optional[AlertSystem] = None
_alert_lock = threading.Lock()


def get_alert_system() -> AlertSystem:
    """Get the global alert system instance."""
    global _alert_system_instance

    if _alert_system_instance is None:
        with _alert_lock:
            if _alert_system_instance is None:
                _alert_system_instance = AlertSystem()

    return _alert_system_instance


def start_alert_system() -> None:
    """Start the global alert system."""
    alert_system = get_alert_system()
    alert_system.start_alert_system()


def stop_alert_system() -> None:
    """Stop the global alert system."""
    alert_system = get_alert_system()
    alert_system.stop_alert_system()
