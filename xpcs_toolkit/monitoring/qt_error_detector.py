"""
Qt Error Detection and Alerting System for XPCS Toolkit.

This module provides comprehensive Qt error detection, pattern analysis,
and alerting capabilities to maintain zero Qt errors in the application.
"""

import re
import sys
import time
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Pattern, Set, Tuple

from PySide6.QtCore import QObject, QTimer, Signal, QMutex, QMutexLocker, qInstallMessageHandler, QtMsgType

from .health_monitor_daemon import HealthAlert, AlertSeverity
from .performance_metrics_collector import get_performance_metrics_collector, MetricType
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class QtErrorType(Enum):
    """Types of Qt errors and warnings."""

    TIMER_THREADING = "timer_threading"
    SIGNAL_CONNECTION = "signal_connection"
    STYLE_HINTS = "style_hints"
    QOBJECT_THREAD = "qobject_thread"
    RESOURCE_LEAK = "resource_leak"
    WIDGET_CREATION = "widget_creation"
    UNKNOWN = "unknown"


class QtErrorSeverity(Enum):
    """Qt error severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class QtErrorPattern:
    """Qt error pattern definition."""

    name: str
    pattern: Pattern[str]
    error_type: QtErrorType
    severity: QtErrorSeverity
    description: str
    suggested_fix: str = ""
    auto_recoverable: bool = False

    def matches(self, message: str) -> bool:
        """Check if message matches this pattern."""
        return bool(self.pattern.search(message))


@dataclass
class QtErrorOccurrence:
    """Single Qt error occurrence."""

    timestamp: float
    message: str
    error_type: QtErrorType
    severity: QtErrorSeverity
    pattern_name: str
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QtErrorCapture:
    """Context manager for capturing Qt errors."""

    def __init__(self, detector: 'QtErrorDetector'):
        self.detector = detector
        self.captured_errors: List[QtErrorOccurrence] = []
        self.original_handler = None

    def __enter__(self):
        self.original_handler = self.detector._message_handler
        self.detector._message_handler = self._capture_handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.detector._message_handler = self.original_handler

    def _capture_handler(self, msg_type: QtMsgType, context, message: str):
        """Capture Qt messages."""
        error = self.detector._process_qt_message(msg_type, message, capture_stack=True)
        if error:
            self.captured_errors.append(error)

        # Call original handler if it exists
        if self.original_handler:
            self.original_handler(msg_type, context, message)


class QtErrorDetector(QObject):
    """
    Comprehensive Qt error detection and alerting system.

    Features:
    - Real-time Qt message monitoring
    - Pattern-based error classification
    - Error frequency analysis
    - Automatic alerting
    - Performance impact measurement
    - Error trend analysis
    """

    # Signals
    qt_error_detected = Signal(object)  # QtErrorOccurrence
    error_pattern_triggered = Signal(str, int)  # pattern_name, frequency
    error_threshold_exceeded = Signal(str, int, int)  # error_type, count, threshold

    def __init__(self, enable_qt_handler: bool = True, parent: QObject = None):
        """Initialize Qt error detector."""
        super().__init__(parent)

        self.enable_qt_handler = enable_qt_handler
        self._mutex = QMutex()

        # Error patterns
        self._error_patterns: List[QtErrorPattern] = []
        self._setup_default_patterns()

        # Error tracking
        self._error_history: deque = deque(maxlen=10000)
        self._error_counts: Dict[QtErrorType, int] = defaultdict(int)
        self._pattern_counts: Dict[str, int] = defaultdict(int)
        self._frequency_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Configuration
        self._alert_thresholds: Dict[QtErrorType, int] = {
            QtErrorType.TIMER_THREADING: 1,      # Zero tolerance
            QtErrorType.SIGNAL_CONNECTION: 5,     # Allow some PyQtGraph warnings
            QtErrorType.STYLE_HINTS: 10,         # Common with PyQtGraph
            QtErrorType.QOBJECT_THREAD: 1,       # Zero tolerance
            QtErrorType.RESOURCE_LEAK: 1,        # Zero tolerance
            QtErrorType.WIDGET_CREATION: 3,      # Some tolerance
            QtErrorType.UNKNOWN: 5               # Moderate tolerance
        }

        # Suppression patterns (for known acceptable warnings)
        self._suppression_patterns: Set[str] = {
            "QStyleHints.*colorSchemeChanged",  # PyQtGraph cosmetic warning
            "qt.svg.*renderer.*warning",        # SVG rendering warnings (cosmetic)
        }

        # Performance metrics
        self._metrics_collector = get_performance_metrics_collector()
        self._setup_metrics()

        # Qt message handler
        self._message_handler: Optional[Callable] = None
        self._original_qt_handler = None

        if self.enable_qt_handler:
            self._install_qt_handler()

        # Analysis timer
        self._analysis_timer = QTimer(self)
        self._analysis_timer.timeout.connect(self._analyze_error_patterns)
        self._analysis_timer.start(30000)  # Analyze every 30 seconds

        logger.info("Qt error detector initialized")

    def _setup_default_patterns(self):
        """Setup default Qt error patterns."""
        patterns = [
            # Timer threading violations
            QtErrorPattern(
                name="timer_threading_violation",
                pattern=re.compile(r"QObject::startTimer.*Timers can only be used with threads started with QThread"),
                error_type=QtErrorType.TIMER_THREADING,
                severity=QtErrorSeverity.CRITICAL,
                description="Timer created in non-Qt thread",
                suggested_fix="Use QtCompliantTimerManager.create_timer()",
                auto_recoverable=False
            ),

            # Signal/slot connection issues
            QtErrorPattern(
                name="signal_connection_warning",
                pattern=re.compile(r"QObject::connect.*unique connections require.*pointer to member function"),
                error_type=QtErrorType.SIGNAL_CONNECTION,
                severity=QtErrorSeverity.WARNING,
                description="Signal/slot connection issue",
                suggested_fix="Use modern Qt5+ syntax with safe_connect()",
                auto_recoverable=False
            ),

            # QStyleHints warnings (from PyQtGraph)
            QtErrorPattern(
                name="style_hints_warning",
                pattern=re.compile(r"qt\.core\.qobject\.connect.*QStyleHints.*colorSchemeChanged"),
                error_type=QtErrorType.STYLE_HINTS,
                severity=QtErrorSeverity.WARNING,
                description="QStyleHints connection warning",
                suggested_fix="Use qt_connection_context() for PyQtGraph widgets",
                auto_recoverable=True
            ),

            # QObject thread safety violations
            QtErrorPattern(
                name="qobject_thread_violation",
                pattern=re.compile(r"QObject.*thread.*affinity.*different thread"),
                error_type=QtErrorType.QOBJECT_THREAD,
                severity=QtErrorSeverity.CRITICAL,
                description="QObject accessed from wrong thread",
                suggested_fix="Ensure Qt operations happen in main thread",
                auto_recoverable=False
            ),

            # Resource leak warnings
            QtErrorPattern(
                name="resource_leak_warning",
                pattern=re.compile(r"QObject.*destroyed while.*still has.*children|leak.*detected"),
                error_type=QtErrorType.RESOURCE_LEAK,
                severity=QtErrorSeverity.WARNING,
                description="Potential resource leak",
                suggested_fix="Use SafeWorkerBase for automatic resource management",
                auto_recoverable=True
            ),

            # Widget creation issues
            QtErrorPattern(
                name="widget_creation_warning",
                pattern=re.compile(r"QWidget.*created.*before.*QApplication|QPixmap.*GUI thread"),
                error_type=QtErrorType.WIDGET_CREATION,
                severity=QtErrorSeverity.WARNING,
                description="Widget creation thread issue",
                suggested_fix="Create widgets in main thread only",
                auto_recoverable=False
            ),
        ]

        self._error_patterns.extend(patterns)

    def _setup_metrics(self):
        """Setup performance metrics for error detection."""
        self._metrics_collector.register_metric(
            "qt_errors_total",
            MetricType.COUNTER,
            unit="errors",
            description="Total Qt errors detected"
        )

        self._metrics_collector.register_metric(
            "qt_errors_by_type",
            MetricType.COUNTER,
            unit="errors",
            description="Qt errors by type"
        )

        self._metrics_collector.register_metric(
            "qt_error_detection_latency",
            MetricType.TIMER,
            unit="seconds",
            description="Time to detect and process Qt errors"
        )

    def _install_qt_handler(self):
        """Install Qt message handler."""
        def qt_message_handler(msg_type: QtMsgType, context, message: str):
            """Handle Qt messages."""
            try:
                # Process the Qt message
                error = self._process_qt_message(msg_type, message)

                # Call original handler if it exists
                if self._original_qt_handler:
                    self._original_qt_handler(msg_type, context, message)

            except Exception as e:
                # Don't let error detection itself crash the application
                logger.error(f"Qt message handler error: {e}")

        # Install the handler
        self._message_handler = qt_message_handler
        qInstallMessageHandler(qt_message_handler)
        logger.debug("Qt message handler installed")

    def _process_qt_message(self, msg_type: QtMsgType, message: str,
                           capture_stack: bool = False) -> Optional[QtErrorOccurrence]:
        """Process a Qt message and detect errors."""
        start_time = time.perf_counter()

        try:
            # Skip if message should be suppressed
            if self._is_suppressed(message):
                return None

            # Map Qt message type to severity
            severity_map = {
                QtMsgType.QtDebugMsg: QtErrorSeverity.DEBUG,
                QtMsgType.QtInfoMsg: QtErrorSeverity.INFO,
                QtMsgType.QtWarningMsg: QtErrorSeverity.WARNING,
                QtMsgType.QtCriticalMsg: QtErrorSeverity.CRITICAL,
                QtMsgType.QtFatalMsg: QtErrorSeverity.FATAL,
            }

            severity = severity_map.get(msg_type, QtErrorSeverity.WARNING)

            # Skip debug messages unless they match critical patterns
            if severity == QtErrorSeverity.DEBUG:
                has_critical_pattern = any(
                    pattern.severity in (QtErrorSeverity.CRITICAL, QtErrorSeverity.FATAL)
                    and pattern.matches(message)
                    for pattern in self._error_patterns
                )
                if not has_critical_pattern:
                    return None

            # Match against error patterns
            matched_pattern = None
            error_type = QtErrorType.UNKNOWN

            for pattern in self._error_patterns:
                if pattern.matches(message):
                    matched_pattern = pattern
                    error_type = pattern.error_type
                    severity = pattern.severity  # Use pattern severity
                    break

            # Create error occurrence
            timestamp = time.perf_counter()
            stack_trace = None

            if capture_stack:
                stack_trace = ''.join(traceback.format_stack())

            error = QtErrorOccurrence(
                timestamp=timestamp,
                message=message,
                error_type=error_type,
                severity=severity,
                pattern_name=matched_pattern.name if matched_pattern else "unknown",
                stack_trace=stack_trace,
                context={
                    "qt_msg_type": msg_type.name,
                    "detection_latency": time.perf_counter() - start_time
                }
            )

            # Store error
            with QMutexLocker(self._mutex):
                self._error_history.append(error)
                self._error_counts[error_type] += 1

                if matched_pattern:
                    self._pattern_counts[matched_pattern.name] += 1
                    self._frequency_windows[matched_pattern.name].append(timestamp)

            # Record metrics
            self._metrics_collector.record_counter("qt_errors_total")
            self._metrics_collector.record_counter(
                "qt_errors_by_type",
                tags={"error_type": error_type.value, "severity": severity.value}
            )
            self._metrics_collector.record_timer(
                "qt_error_detection_latency",
                time.perf_counter() - start_time
            )

            # Emit signals
            self.qt_error_detected.emit(error)

            # Check thresholds
            threshold = self._alert_thresholds.get(error_type, 10)
            if self._error_counts[error_type] >= threshold:
                self.error_threshold_exceeded.emit(
                    error_type.value,
                    self._error_counts[error_type],
                    threshold
                )

            # Log error
            log_level = {
                QtErrorSeverity.DEBUG: logger.debug,
                QtErrorSeverity.INFO: logger.info,
                QtErrorSeverity.WARNING: logger.warning,
                QtErrorSeverity.CRITICAL: logger.error,
                QtErrorSeverity.FATAL: logger.critical
            }[severity]

            log_level(f"Qt {error_type.value}: {message}")

            return error

        except Exception as e:
            logger.error(f"Error processing Qt message: {e}")
            return None

    def _is_suppressed(self, message: str) -> bool:
        """Check if message should be suppressed."""
        for pattern in self._suppression_patterns:
            if re.search(pattern, message):
                return True
        return False

    def _analyze_error_patterns(self):
        """Analyze error patterns and frequencies."""
        try:
            current_time = time.perf_counter()
            analysis_window = 300.0  # 5 minutes

            with QMutexLocker(self._mutex):
                # Analyze frequency patterns
                for pattern_name, timestamps in self._frequency_windows.items():
                    # Remove old timestamps
                    cutoff = current_time - analysis_window
                    while timestamps and timestamps[0] < cutoff:
                        timestamps.popleft()

                    # Check frequency
                    frequency = len(timestamps)
                    if frequency > 0:
                        self.error_pattern_triggered.emit(pattern_name, frequency)

                        # Log high frequency patterns
                        if frequency > 10:
                            logger.warning(f"High frequency Qt error pattern: {pattern_name} "
                                         f"({frequency} times in {analysis_window/60:.1f} minutes)")

        except Exception as e:
            logger.error(f"Error pattern analysis failed: {e}")

    def add_error_pattern(self, pattern: QtErrorPattern):
        """Add a custom error pattern."""
        with QMutexLocker(self._mutex):
            self._error_patterns.append(pattern)
            logger.info(f"Added Qt error pattern: {pattern.name}")

    def add_suppression_pattern(self, pattern: str):
        """Add a suppression pattern."""
        with QMutexLocker(self._mutex):
            self._suppression_patterns.add(pattern)
            logger.info(f"Added Qt suppression pattern: {pattern}")

    def set_alert_threshold(self, error_type: QtErrorType, threshold: int):
        """Set alert threshold for an error type."""
        self._alert_thresholds[error_type] = threshold
        logger.info(f"Set alert threshold for {error_type.value}: {threshold}")

    def get_error_summary(self) -> Dict[str, Any]:
        """Get error detection summary."""
        with QMutexLocker(self._mutex):
            current_time = time.perf_counter()

            # Recent errors (last hour)
            recent_cutoff = current_time - 3600
            recent_errors = [
                error for error in self._error_history
                if error.timestamp > recent_cutoff
            ]

            # Error counts by type
            recent_counts = defaultdict(int)
            for error in recent_errors:
                recent_counts[error.error_type] += 1

            return {
                "total_errors": len(self._error_history),
                "recent_errors_1h": len(recent_errors),
                "error_counts_by_type": dict(self._error_counts),
                "recent_counts_by_type": dict(recent_counts),
                "pattern_counts": dict(self._pattern_counts),
                "alert_thresholds": {k.value: v for k, v in self._alert_thresholds.items()},
                "patterns_registered": len(self._error_patterns),
                "suppression_patterns": len(self._suppression_patterns)
            }

    def get_recent_errors(self, hours: Optional[int] = None,
                         error_type: Optional[QtErrorType] = None,
                         severity: Optional[QtErrorSeverity] = None) -> List[QtErrorOccurrence]:
        """Get recent errors with optional filtering."""
        cutoff_time = None
        if hours:
            cutoff_time = time.perf_counter() - (hours * 3600)

        with QMutexLocker(self._mutex):
            filtered_errors = []
            for error in self._error_history:
                # Filter by time
                if cutoff_time and error.timestamp < cutoff_time:
                    continue

                # Filter by type
                if error_type and error.error_type != error_type:
                    continue

                # Filter by severity
                if severity and error.severity != severity:
                    continue

                filtered_errors.append(error)

            return filtered_errors

    def create_error_capture(self) -> QtErrorCapture:
        """Create a Qt error capture context manager."""
        return QtErrorCapture(self)

    def clear_error_history(self):
        """Clear error history."""
        with QMutexLocker(self._mutex):
            self._error_history.clear()
            self._error_counts.clear()
            self._pattern_counts.clear()
            for window in self._frequency_windows.values():
                window.clear()

            logger.info("Cleared Qt error history")

    def export_error_report(self) -> Dict[str, Any]:
        """Export comprehensive error report."""
        summary = self.get_error_summary()
        recent_errors = self.get_recent_errors(hours=24)

        return {
            "report_timestamp": time.perf_counter(),
            "summary": summary,
            "recent_errors_24h": [error.to_dict() for error in recent_errors],
            "error_patterns": [
                {
                    "name": pattern.name,
                    "type": pattern.error_type.value,
                    "severity": pattern.severity.value,
                    "description": pattern.description,
                    "suggested_fix": pattern.suggested_fix,
                    "count": self._pattern_counts.get(pattern.name, 0)
                }
                for pattern in self._error_patterns
            ]
        }


# Global instance
_qt_error_detector: Optional[QtErrorDetector] = None


def get_qt_error_detector(enable_qt_handler: bool = True) -> QtErrorDetector:
    """Get the global Qt error detector instance."""
    global _qt_error_detector

    if _qt_error_detector is None:
        _qt_error_detector = QtErrorDetector(enable_qt_handler)

    return _qt_error_detector


def initialize_qt_error_detection(enable_qt_handler: bool = True) -> QtErrorDetector:
    """Initialize Qt error detection."""
    return get_qt_error_detector(enable_qt_handler)


@contextmanager
def qt_error_capture():
    """Context manager for capturing Qt errors."""
    detector = get_qt_error_detector()
    with detector.create_error_capture() as capture:
        yield capture