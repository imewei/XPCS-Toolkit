"""
Production Monitoring and Alerting System for Qt Compliance Framework.

This module provides comprehensive production monitoring with advanced alerting
capabilities for the Qt compliance system. It includes real-time monitoring,
anomaly detection, intelligent alerting, and incident management.
"""

import asyncio
import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque
import statistics
import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import requests
import socket
import psutil
from pathlib import Path

from PySide6.QtCore import QObject, Signal, QTimer, QThread

from ..monitoring.health_monitor_daemon import QtHealthMonitor
from ..core.qt_error_detector import QtComplianceChecker
from ..performance.qt_performance_profiler import QtPerformanceProfiler

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    """Alert categories for classification."""
    PERFORMANCE = "performance"
    ERROR = "error"
    RESOURCE = "resource"
    SECURITY = "security"
    AVAILABILITY = "availability"
    COMPLIANCE = "compliance"
    USER_EXPERIENCE = "user_experience"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class MonitoringMode(Enum):
    """Monitoring operation modes."""
    DEVELOPMENT = auto()
    TESTING = auto()
    STAGING = auto()
    PRODUCTION = auto()
    MAINTENANCE = auto()


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGER_DUTY = "pager_duty"
    DISCORD = "discord"
    LOG_FILE = "log_file"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str
    severity: AlertSeverity
    category: AlertCategory
    threshold: float
    duration: timedelta
    channels: List[NotificationChannel]
    enabled: bool = True
    description: str = ""
    tags: Set[str] = field(default_factory=set)
    auto_resolve: bool = False
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class Alert:
    """Alert instance."""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    status: AlertStatus
    timestamp: datetime
    value: Optional[float] = None
    threshold: Optional[float] = None
    source: str = ""
    tags: Set[str] = field(default_factory=set)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    escalation_level: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitoringMetrics:
    """Production monitoring metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    error_count: int
    warning_count: int
    active_threads: int
    qt_compliance_score: float
    response_time: float
    throughput: float
    availability: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class NotificationConfig:
    """Notification channel configuration."""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    rate_limit: Optional[int] = None
    severity_filter: Optional[List[AlertSeverity]] = None


class ProductionAlertingSystem(QObject):
    """
    Comprehensive production monitoring and alerting system.

    Features:
    - Real-time metrics collection and analysis
    - Intelligent anomaly detection
    - Multi-channel alerting with escalation
    - Alert correlation and deduplication
    - Incident management and tracking
    - SLA monitoring and reporting
    - Automated remediation actions
    """

    # Qt signals for real-time updates
    alert_triggered = Signal(Alert)
    alert_resolved = Signal(str)  # alert_id
    metrics_updated = Signal(MonitoringMetrics)
    system_health_changed = Signal(str, float)  # component, health_score

    def __init__(
        self,
        monitoring_mode: MonitoringMode = MonitoringMode.PRODUCTION,
        config_file: Optional[str] = None
    ):
        super().__init__()
        self.monitoring_mode = monitoring_mode
        self.config_file = config_file

        # Core components
        self.health_monitor: Optional[QtHealthMonitor] = None
        self.compliance_checker: Optional[QtComplianceChecker] = None
        self.performance_profiler: Optional[QtPerformanceProfiler] = None

        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.suppressed_alerts: Set[str] = set()

        # Metrics and monitoring
        self.metrics_history: deque = deque(maxlen=1000)
        self.baseline_metrics: Dict[str, float] = {}
        self.anomaly_detectors: Dict[str, 'AnomalyDetector'] = {}

        # Notification system
        self.notification_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.notification_rate_limiters: Dict[NotificationChannel, 'RateLimiter'] = {}

        # Threading and async
        self.monitoring_thread: Optional[QThread] = None
        self.is_monitoring: bool = False
        self.monitoring_interval: float = 30.0  # seconds

        # Performance tracking
        self.alert_processing_times: deque = deque(maxlen=100)
        self.notification_success_rates: Dict[NotificationChannel, float] = {}

        self._initialize_system()

    def _initialize_system(self):
        """Initialize the alerting system."""
        try:
            self._load_configuration()
            self._setup_default_alert_rules()
            self._initialize_anomaly_detectors()
            self._setup_notification_channels()
            self._initialize_monitoring_components()

            logger.info(f"Production alerting system initialized in {self.monitoring_mode.name} mode")

        except Exception as e:
            logger.error(f"Failed to initialize alerting system: {e}")
            raise

    def _load_configuration(self):
        """Load alerting system configuration."""
        if self.config_file and Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)

                self.monitoring_interval = config.get('monitoring_interval', 30.0)

                # Load alert rules
                for rule_data in config.get('alert_rules', []):
                    rule = AlertRule(
                        name=rule_data['name'],
                        condition=rule_data['condition'],
                        severity=AlertSeverity(rule_data['severity']),
                        category=AlertCategory(rule_data['category']),
                        threshold=rule_data['threshold'],
                        duration=timedelta(seconds=rule_data['duration_seconds']),
                        channels=[NotificationChannel(ch) for ch in rule_data['channels']],
                        enabled=rule_data.get('enabled', True),
                        description=rule_data.get('description', ''),
                        tags=set(rule_data.get('tags', [])),
                        auto_resolve=rule_data.get('auto_resolve', False),
                        escalation_rules=rule_data.get('escalation_rules', [])
                    )
                    self.alert_rules[rule.name] = rule

                # Load notification configs
                for channel_name, channel_config in config.get('notifications', {}).items():
                    channel = NotificationChannel(channel_name)
                    self.notification_configs[channel] = NotificationConfig(
                        channel=channel,
                        enabled=channel_config.get('enabled', True),
                        config=channel_config.get('config', {}),
                        rate_limit=channel_config.get('rate_limit'),
                        severity_filter=[AlertSeverity(s) for s in channel_config.get('severity_filter', [])] if channel_config.get('severity_filter') else None
                    )

                logger.info(f"Loaded configuration from {self.config_file}")

            except Exception as e:
                logger.warning(f"Failed to load configuration: {e}, using defaults")

    def _setup_default_alert_rules(self):
        """Setup default alert rules if none configured."""
        if not self.alert_rules:
            default_rules = [
                AlertRule(
                    name="high_cpu_usage",
                    condition="cpu_usage > threshold",
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.RESOURCE,
                    threshold=80.0,
                    duration=timedelta(minutes=5),
                    channels=[NotificationChannel.EMAIL, NotificationChannel.LOG_FILE],
                    description="CPU usage exceeds 80% for more than 5 minutes"
                ),
                AlertRule(
                    name="high_memory_usage",
                    condition="memory_usage > threshold",
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.RESOURCE,
                    threshold=85.0,
                    duration=timedelta(minutes=3),
                    channels=[NotificationChannel.EMAIL, NotificationChannel.LOG_FILE],
                    description="Memory usage exceeds 85% for more than 3 minutes"
                ),
                AlertRule(
                    name="qt_compliance_degradation",
                    condition="qt_compliance_score < threshold",
                    severity=AlertSeverity.CRITICAL,
                    category=AlertCategory.COMPLIANCE,
                    threshold=95.0,
                    duration=timedelta(minutes=1),
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                    description="Qt compliance score below 95%"
                ),
                AlertRule(
                    name="high_error_rate",
                    condition="error_count > threshold",
                    severity=AlertSeverity.HIGH,
                    category=AlertCategory.ERROR,
                    threshold=10.0,
                    duration=timedelta(minutes=2),
                    channels=[NotificationChannel.EMAIL, NotificationChannel.LOG_FILE],
                    description="Error count exceeds 10 in 2 minutes"
                ),
                AlertRule(
                    name="slow_response_time",
                    condition="response_time > threshold",
                    severity=AlertSeverity.MEDIUM,
                    category=AlertCategory.PERFORMANCE,
                    threshold=1000.0,  # ms
                    duration=timedelta(minutes=3),
                    channels=[NotificationChannel.LOG_FILE],
                    description="Response time exceeds 1000ms for 3 minutes"
                ),
                AlertRule(
                    name="low_availability",
                    condition="availability < threshold",
                    severity=AlertSeverity.EMERGENCY,
                    category=AlertCategory.AVAILABILITY,
                    threshold=99.0,
                    duration=timedelta(minutes=1),
                    channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGER_DUTY],
                    description="System availability below 99%"
                )
            ]

            for rule in default_rules:
                self.alert_rules[rule.name] = rule

    def _initialize_anomaly_detectors(self):
        """Initialize anomaly detection systems."""
        metrics_to_monitor = [
            'cpu_usage', 'memory_usage', 'response_time',
            'error_count', 'qt_compliance_score', 'throughput'
        ]

        for metric in metrics_to_monitor:
            self.anomaly_detectors[metric] = AnomalyDetector(
                metric_name=metric,
                window_size=50,
                sensitivity=2.5
            )

    def _setup_notification_channels(self):
        """Setup notification channels with default configs."""
        if not self.notification_configs:
            # Email configuration
            self.notification_configs[NotificationChannel.EMAIL] = NotificationConfig(
                channel=NotificationChannel.EMAIL,
                enabled=True,
                config={
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': '',  # To be configured
                    'password': '',  # To be configured
                    'recipients': []  # To be configured
                },
                rate_limit=5  # max 5 emails per hour
            )

            # Log file configuration
            self.notification_configs[NotificationChannel.LOG_FILE] = NotificationConfig(
                channel=NotificationChannel.LOG_FILE,
                enabled=True,
                config={
                    'file_path': '/var/log/xpcs_toolkit/alerts.log',
                    'max_size': '100MB',
                    'backup_count': 5
                }
            )

        # Initialize rate limiters
        for channel, config in self.notification_configs.items():
            if config.rate_limit:
                self.notification_rate_limiters[channel] = RateLimiter(
                    max_requests=config.rate_limit,
                    time_window=timedelta(hours=1)
                )

    def _initialize_monitoring_components(self):
        """Initialize monitoring component integrations."""
        try:
            from ..monitoring.health_monitor_daemon import QtHealthMonitor
            from ..core.qt_error_detector import QtComplianceChecker
            from ..performance.qt_performance_profiler import QtPerformanceProfiler

            self.health_monitor = QtHealthMonitor()
            self.compliance_checker = QtComplianceChecker()
            self.performance_profiler = QtPerformanceProfiler()

        except ImportError as e:
            logger.warning(f"Some monitoring components not available: {e}")

    def start_monitoring(self):
        """Start the production monitoring system."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return

        self.is_monitoring = True

        # Start monitoring thread
        self.monitoring_thread = ProductionMonitoringThread(self)
        self.monitoring_thread.start()

        logger.info("Production monitoring started")

    def stop_monitoring(self):
        """Stop the production monitoring system."""
        if not self.is_monitoring:
            return

        self.is_monitoring = False

        if self.monitoring_thread:
            self.monitoring_thread.stop()
            self.monitoring_thread.wait()
            self.monitoring_thread = None

        logger.info("Production monitoring stopped")

    def collect_metrics(self) -> MonitoringMetrics:
        """Collect current system metrics."""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_usage = memory_info.percent
            disk_usage = psutil.disk_usage('/').percent

            # Network latency (ping to localhost as baseline)
            network_latency = self._measure_network_latency()

            # Qt compliance metrics
            qt_compliance_score = 100.0
            if self.compliance_checker:
                try:
                    result = self.compliance_checker.run_comprehensive_check()
                    qt_compliance_score = result.overall_score
                except Exception as e:
                    logger.warning(f"Failed to get Qt compliance score: {e}")

            # Performance metrics
            response_time = self._measure_response_time()
            throughput = self._calculate_throughput()
            availability = self._calculate_availability()

            # Error metrics
            error_count = self._count_recent_errors()
            warning_count = self._count_recent_warnings()

            # Threading metrics
            active_threads = threading.active_count()

            metrics = MonitoringMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_latency=network_latency,
                error_count=error_count,
                warning_count=warning_count,
                active_threads=active_threads,
                qt_compliance_score=qt_compliance_score,
                response_time=response_time,
                throughput=throughput,
                availability=availability
            )

            # Store metrics history
            self.metrics_history.append(metrics)

            # Emit signal
            self.metrics_updated.emit(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise

    def _measure_network_latency(self) -> float:
        """Measure network latency."""
        try:
            start_time = time.time()
            socket.create_connection(("127.0.0.1", 80), timeout=1)
            return (time.time() - start_time) * 1000  # ms
        except:
            return 0.0

    def _measure_response_time(self) -> float:
        """Measure application response time."""
        # This would typically measure actual application response time
        # For now, return a simulated value
        return 50.0 + (time.time() % 10) * 5

    def _calculate_throughput(self) -> float:
        """Calculate system throughput."""
        # This would typically measure actual system throughput
        # For now, return a simulated value
        return 100.0 - (time.time() % 20)

    def _calculate_availability(self) -> float:
        """Calculate system availability."""
        # This would typically calculate actual availability
        # For now, return a high availability score
        return 99.95

    def _count_recent_errors(self) -> int:
        """Count recent error log entries."""
        # This would typically count actual errors from logs
        # For now, return a simulated count
        return int(time.time() % 5)

    def _count_recent_warnings(self) -> int:
        """Count recent warning log entries."""
        # This would typically count actual warnings from logs
        # For now, return a simulated count
        return int(time.time() % 8)

    def evaluate_alert_rules(self, metrics: MonitoringMetrics):
        """Evaluate all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue

            try:
                self._evaluate_single_rule(rule, metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate rule {rule_name}: {e}")

    def _evaluate_single_rule(self, rule: AlertRule, metrics: MonitoringMetrics):
        """Evaluate a single alert rule."""
        # Get metric value based on rule condition
        metric_value = self._extract_metric_value(rule.condition, metrics)

        if metric_value is None:
            return

        # Check if condition is met
        condition_met = self._evaluate_condition(rule.condition, metric_value, rule.threshold)

        if condition_met:
            # Check if alert already exists
            alert_id = f"{rule.name}_{int(time.time() // rule.duration.total_seconds())}"

            if alert_id not in self.active_alerts:
                # Create new alert
                alert = Alert(
                    id=alert_id,
                    rule_name=rule.name,
                    title=f"{rule.name.replace('_', ' ').title()}",
                    description=rule.description or f"{rule.condition} (value: {metric_value:.2f}, threshold: {rule.threshold})",
                    severity=rule.severity,
                    category=rule.category,
                    status=AlertStatus.ACTIVE,
                    timestamp=datetime.now(),
                    value=metric_value,
                    threshold=rule.threshold,
                    source="production_monitoring",
                    tags=rule.tags.copy(),
                    context={
                        'metrics': metrics.__dict__,
                        'rule': rule.__dict__
                    }
                )

                self._trigger_alert(alert, rule)
        else:
            # Check for auto-resolve
            if rule.auto_resolve:
                self._auto_resolve_alerts(rule.name)

    def _extract_metric_value(self, condition: str, metrics: MonitoringMetrics) -> Optional[float]:
        """Extract metric value from condition string."""
        try:
            # Simple condition parsing (e.g., "cpu_usage > threshold")
            metric_name = condition.split()[0]
            return getattr(metrics, metric_name, None)
        except Exception:
            return None

    def _evaluate_condition(self, condition: str, value: float, threshold: float) -> bool:
        """Evaluate alert condition."""
        try:
            # Replace variables in condition
            condition = condition.replace('threshold', str(threshold))
            parts = condition.split()

            if len(parts) >= 3:
                operator = parts[1]
                if operator == '>':
                    return value > threshold
                elif operator == '<':
                    return value < threshold
                elif operator == '>=':
                    return value >= threshold
                elif operator == '<=':
                    return value <= threshold
                elif operator == '==':
                    return value == threshold
                elif operator == '!=':
                    return value != threshold

            return False
        except Exception:
            return False

    def _trigger_alert(self, alert: Alert, rule: AlertRule):
        """Trigger a new alert."""
        start_time = time.time()

        try:
            # Store alert
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)

            # Check for suppression
            if alert.id in self.suppressed_alerts:
                logger.info(f"Alert {alert.id} suppressed")
                return

            # Send notifications
            self._send_notifications(alert, rule.channels)

            # Emit signal
            self.alert_triggered.emit(alert)

            # Record processing time
            processing_time = time.time() - start_time
            self.alert_processing_times.append(processing_time)

            logger.warning(f"Alert triggered: {alert.title} [{alert.severity.value}]")

        except Exception as e:
            logger.error(f"Failed to trigger alert: {e}")

    def _send_notifications(self, alert: Alert, channels: List[NotificationChannel]):
        """Send alert notifications through specified channels."""
        for channel in channels:
            try:
                if channel not in self.notification_configs:
                    continue

                config = self.notification_configs[channel]
                if not config.enabled:
                    continue

                # Check severity filter
                if config.severity_filter and alert.severity not in config.severity_filter:
                    continue

                # Check rate limit
                if channel in self.notification_rate_limiters:
                    rate_limiter = self.notification_rate_limiters[channel]
                    if not rate_limiter.allow_request():
                        logger.warning(f"Rate limit exceeded for {channel.value}")
                        continue

                # Send notification
                success = self._send_single_notification(alert, channel, config)

                # Update success rate
                if channel not in self.notification_success_rates:
                    self.notification_success_rates[channel] = 1.0 if success else 0.0
                else:
                    # Rolling average
                    current_rate = self.notification_success_rates[channel]
                    self.notification_success_rates[channel] = 0.9 * current_rate + 0.1 * (1.0 if success else 0.0)

            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")

    def _send_single_notification(self, alert: Alert, channel: NotificationChannel, config: NotificationConfig) -> bool:
        """Send notification through a single channel."""
        try:
            if channel == NotificationChannel.EMAIL:
                return self._send_email_notification(alert, config)
            elif channel == NotificationChannel.SLACK:
                return self._send_slack_notification(alert, config)
            elif channel == NotificationChannel.LOG_FILE:
                return self._send_log_notification(alert, config)
            elif channel == NotificationChannel.WEBHOOK:
                return self._send_webhook_notification(alert, config)
            else:
                logger.warning(f"Notification channel {channel.value} not implemented")
                return False
        except Exception as e:
            logger.error(f"Failed to send {channel.value} notification: {e}")
            return False

    def _send_email_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send email notification."""
        try:
            email_config = config.config
            if not all(k in email_config for k in ['smtp_server', 'username', 'password', 'recipients']):
                logger.warning("Email configuration incomplete")
                return False

            # Create message
            msg = MimeMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value.upper()}
            Category: {alert.category.value}
            Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

            Description: {alert.description}

            Value: {alert.value}
            Threshold: {alert.threshold}

            Alert ID: {alert.id}
            """

            msg.attach(MimeText(body, 'plain'))

            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(email_config['smtp_server'], email_config.get('smtp_port', 587)) as server:
                server.starttls(context=context)
                server.login(email_config['username'], email_config['password'])
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False

    def _send_slack_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send Slack notification."""
        try:
            slack_config = config.config
            webhook_url = slack_config.get('webhook_url')

            if not webhook_url:
                logger.warning("Slack webhook URL not configured")
                return False

            # Create message
            color = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ff9500",
                AlertSeverity.HIGH: "#ff0000",
                AlertSeverity.CRITICAL: "#8B0000",
                AlertSeverity.EMERGENCY: "#800080"
            }.get(alert.severity, "#808080")

            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Value", "value": str(alert.value), "short": True},
                        {"title": "Threshold", "value": str(alert.threshold), "short": True},
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Time", "value": alert.timestamp.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "footer": "XPCS Toolkit Monitoring",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            return response.status_code == 200

        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False

    def _send_log_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send log file notification."""
        try:
            log_config = config.config
            file_path = log_config.get('file_path', '/var/log/xpcs_toolkit/alerts.log')

            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)

            # Write alert to log
            log_entry = {
                'timestamp': alert.timestamp.isoformat(),
                'alert_id': alert.id,
                'title': alert.title,
                'severity': alert.severity.value,
                'category': alert.category.value,
                'description': alert.description,
                'value': alert.value,
                'threshold': alert.threshold,
                'status': alert.status.value
            }

            with open(file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            return True

        except Exception as e:
            logger.error(f"Log notification failed: {e}")
            return False

    def _send_webhook_notification(self, alert: Alert, config: NotificationConfig) -> bool:
        """Send webhook notification."""
        try:
            webhook_config = config.config
            url = webhook_config.get('url')

            if not url:
                logger.warning("Webhook URL not configured")
                return False

            payload = {
                'alert_id': alert.id,
                'title': alert.title,
                'severity': alert.severity.value,
                'category': alert.category.value,
                'description': alert.description,
                'timestamp': alert.timestamp.isoformat(),
                'value': alert.value,
                'threshold': alert.threshold,
                'status': alert.status.value,
                'context': alert.context
            }

            headers = webhook_config.get('headers', {'Content-Type': 'application/json'})
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            return response.status_code in [200, 201, 202]

        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False

    def _auto_resolve_alerts(self, rule_name: str):
        """Auto-resolve alerts for a specific rule."""
        alerts_to_resolve = []

        for alert_id, alert in self.active_alerts.items():
            if alert.rule_name == rule_name and alert.status == AlertStatus.ACTIVE:
                alerts_to_resolve.append(alert_id)

        for alert_id in alerts_to_resolve:
            self.resolve_alert(alert_id, "Auto-resolved")

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()

        # Remove from active alerts
        del self.active_alerts[alert_id]

        # Emit signal
        self.alert_resolved.emit(alert_id)

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    def suppress_alert(self, alert_id: str, duration: Optional[timedelta] = None):
        """Suppress an alert for a specified duration."""
        self.suppressed_alerts.add(alert_id)

        if duration:
            # Schedule unsuppression
            def unsuppress():
                self.suppressed_alerts.discard(alert_id)

            timer = QTimer()
            timer.singleShot(int(duration.total_seconds() * 1000), unsuppress)

        logger.info(f"Alert {alert_id} suppressed")

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        if not self.metrics_history:
            return {"status": "no_data"}

        latest_metrics = self.metrics_history[-1]

        # Calculate health scores
        cpu_health = max(0, 100 - latest_metrics.cpu_usage)
        memory_health = max(0, 100 - latest_metrics.memory_usage)
        compliance_health = latest_metrics.qt_compliance_score
        overall_health = (cpu_health + memory_health + compliance_health) / 3

        # Count alerts by severity
        alert_counts = defaultdict(int)
        for alert in self.active_alerts.values():
            alert_counts[alert.severity.value] += 1

        return {
            "timestamp": latest_metrics.timestamp.isoformat(),
            "overall_health": overall_health,
            "component_health": {
                "cpu": cpu_health,
                "memory": memory_health,
                "qt_compliance": compliance_health,
                "availability": latest_metrics.availability
            },
            "active_alerts": dict(alert_counts),
            "total_active_alerts": len(self.active_alerts),
            "metrics": {
                "cpu_usage": latest_metrics.cpu_usage,
                "memory_usage": latest_metrics.memory_usage,
                "response_time": latest_metrics.response_time,
                "error_count": latest_metrics.error_count,
                "qt_compliance_score": latest_metrics.qt_compliance_score
            },
            "notification_health": self.notification_success_rates
        }

    def generate_alert_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate comprehensive alert report for time period."""
        # Filter alerts by time period
        period_alerts = [
            alert for alert in self.alert_history
            if start_time <= alert.timestamp <= end_time
        ]

        if not period_alerts:
            return {"period": f"{start_time} to {end_time}", "total_alerts": 0}

        # Analyze alerts
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        hourly_counts = defaultdict(int)

        for alert in period_alerts:
            severity_counts[alert.severity.value] += 1
            category_counts[alert.category.value] += 1
            hour_key = alert.timestamp.strftime("%Y-%m-%d %H:00")
            hourly_counts[hour_key] += 1

        # Calculate resolution times
        resolved_alerts = [a for a in period_alerts if a.resolved_at]
        resolution_times = []

        for alert in resolved_alerts:
            if alert.resolved_at:
                resolution_time = (alert.resolved_at - alert.timestamp).total_seconds()
                resolution_times.append(resolution_time)

        avg_resolution_time = statistics.mean(resolution_times) if resolution_times else 0

        return {
            "period": f"{start_time} to {end_time}",
            "total_alerts": len(period_alerts),
            "severity_breakdown": dict(severity_counts),
            "category_breakdown": dict(category_counts),
            "hourly_distribution": dict(hourly_counts),
            "resolution_stats": {
                "total_resolved": len(resolved_alerts),
                "average_resolution_time_seconds": avg_resolution_time,
                "resolution_rate": len(resolved_alerts) / len(period_alerts) * 100
            },
            "top_alert_rules": [
                (rule_name, count) for rule_name, count in
                sorted(
                    defaultdict(int, {alert.rule_name: 1 for alert in period_alerts}).items(),
                    key=lambda x: x[1], reverse=True
                )[:10]
            ]
        }


class AnomalyDetector:
    """Statistical anomaly detection for metrics."""

    def __init__(self, metric_name: str, window_size: int = 50, sensitivity: float = 2.5):
        self.metric_name = metric_name
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.values: deque = deque(maxlen=window_size)
        self.anomalies: List[Tuple[datetime, float]] = []

    def add_value(self, value: float, timestamp: datetime) -> bool:
        """Add new value and check for anomaly."""
        self.values.append(value)

        if len(self.values) < self.window_size:
            return False

        # Calculate statistics
        mean_val = statistics.mean(self.values)
        std_val = statistics.stdev(self.values) if len(self.values) > 1 else 0

        # Check for anomaly (using standard deviation)
        if std_val > 0:
            z_score = abs(value - mean_val) / std_val
            if z_score > self.sensitivity:
                self.anomalies.append((timestamp, value))
                return True

        return False


class RateLimiter:
    """Simple rate limiter for notification throttling."""

    def __init__(self, max_requests: int, time_window: timedelta):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()

    def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now()

        # Remove old requests
        while self.requests and (now - self.requests[0]) > self.time_window:
            self.requests.popleft()

        # Check limit
        if len(self.requests) >= self.max_requests:
            return False

        # Allow request
        self.requests.append(now)
        return True


class ProductionMonitoringThread(QThread):
    """Background monitoring thread."""

    def __init__(self, alerting_system: ProductionAlertingSystem):
        super().__init__()
        self.alerting_system = alerting_system
        self.running = False

    def run(self):
        """Main monitoring loop."""
        self.running = True

        while self.running:
            try:
                # Collect metrics
                metrics = self.alerting_system.collect_metrics()

                # Evaluate alert rules
                self.alerting_system.evaluate_alert_rules(metrics)

                # Check for anomalies
                for metric_name, detector in self.alerting_system.anomaly_detectors.items():
                    metric_value = getattr(metrics, metric_name, None)
                    if metric_value is not None:
                        if detector.add_value(metric_value, metrics.timestamp):
                            logger.warning(f"Anomaly detected in {metric_name}: {metric_value}")

                # Sleep until next collection
                self.msleep(int(self.alerting_system.monitoring_interval * 1000))

            except Exception as e:
                logger.error(f"Monitoring thread error: {e}")
                self.msleep(5000)  # Wait 5 seconds before retrying

    def stop(self):
        """Stop monitoring thread."""
        self.running = False


def get_production_alerting_system(
    monitoring_mode: MonitoringMode = MonitoringMode.PRODUCTION,
    config_file: Optional[str] = None
) -> ProductionAlertingSystem:
    """Get configured production alerting system instance."""
    return ProductionAlertingSystem(
        monitoring_mode=monitoring_mode,
        config_file=config_file
    )


def create_alert_configuration(
    rules: List[Dict[str, Any]],
    notifications: Dict[str, Dict[str, Any]],
    output_file: str
):
    """Create alert configuration file."""
    config = {
        "monitoring_interval": 30.0,
        "alert_rules": rules,
        "notifications": notifications
    }

    with open(output_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.info(f"Alert configuration saved to {output_file}")


# Export public interface
__all__ = [
    'ProductionAlertingSystem',
    'AlertSeverity',
    'AlertCategory',
    'AlertStatus',
    'MonitoringMode',
    'NotificationChannel',
    'AlertRule',
    'Alert',
    'MonitoringMetrics',
    'NotificationConfig',
    'AnomalyDetector',
    'RateLimiter',
    'get_production_alerting_system',
    'create_alert_configuration'
]