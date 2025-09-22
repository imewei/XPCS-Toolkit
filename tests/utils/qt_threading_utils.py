"""Qt Threading Violation Detection Utilities.

This module provides utilities for detecting and preventing Qt threading violations,
specifically focusing on QTimer usage and signal/slot connection issues.
"""

import contextlib
import functools
import threading
import time
import warnings

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import QObject, QThread, QTimer, Signal


class ThreadingViolationDetector:
    """Detect Qt threading violations in real-time."""

    def __init__(self):
        self.violations = []
        self.monitored_objects = set()
        self.thread_registry = {}
        self.original_timer_start = None
        self.original_qobject_init = None

    def start_monitoring(self):
        """Start monitoring for threading violations."""
        self._patch_timer_methods()
        self._patch_qobject_methods()

    def stop_monitoring(self):
        """Stop monitoring and restore original methods."""
        self._restore_timer_methods()
        self._restore_qobject_methods()

    def _patch_timer_methods(self):
        """Patch QTimer methods to detect threading violations."""
        if self.original_timer_start is None:
            self.original_timer_start = QTimer.start

        def monitored_timer_start(timer_self, *args, **kwargs):
            # Check if timer is being started in appropriate thread
            current_thread = QThread.currentThread()
            app_thread = (
                QtWidgets.QApplication.instance().thread()
                if QtWidgets.QApplication.instance()
                else None
            )

            violation_detected = False
            violation_message = ""

            if app_thread is None:
                violation_detected = True
                violation_message = "QTimer started without QApplication"
            elif current_thread != app_thread and not isinstance(
                current_thread, QThread
            ):
                violation_detected = True
                violation_message = (
                    f"QTimer started in non-Qt thread: {type(current_thread)}"
                )

            if violation_detected:
                self.violations.append(
                    {
                        "type": "timer_threading_violation",
                        "message": violation_message,
                        "timer_object": timer_self,
                        "thread": current_thread,
                        "stack_trace": self._get_stack_trace(),
                        "timestamp": time.time(),
                    }
                )

                # Optionally prevent the violation
                warnings.warn(
                    f"Qt Threading Violation: {violation_message}",
                    UserWarning,
                    stacklevel=2,
                )

            return self.original_timer_start(timer_self, *args, **kwargs)

        QTimer.start = monitored_timer_start

    def _patch_qobject_methods(self):
        """Patch QObject methods to detect other violations."""
        if self.original_qobject_init is None:
            self.original_qobject_init = QObject.__init__

        def monitored_qobject_init(qobject_self, parent=None):
            # Track object creation thread
            current_thread = QThread.currentThread()
            self.thread_registry[id(qobject_self)] = current_thread
            self.monitored_objects.add(qobject_self)

            return self.original_qobject_init(qobject_self, parent)

        QObject.__init__ = monitored_qobject_init

    def _restore_timer_methods(self):
        """Restore original QTimer methods."""
        if self.original_timer_start:
            QTimer.start = self.original_timer_start
            self.original_timer_start = None

    def _restore_qobject_methods(self):
        """Restore original QObject methods."""
        if self.original_qobject_init:
            QObject.__init__ = self.original_qobject_init
            self.original_qobject_init = None

    def _get_stack_trace(self):
        """Get current stack trace for violation context."""
        import traceback

        return traceback.format_stack()

    def get_violations(self):
        """Get all detected violations."""
        return self.violations.copy()

    def clear_violations(self):
        """Clear all recorded violations."""
        self.violations.clear()

    def has_timer_violations(self):
        """Check if any timer violations were detected."""
        return any(v["type"] == "timer_threading_violation" for v in self.violations)

    def get_violation_summary(self):
        """Get summary of detected violations."""
        summary = {
            "total_violations": len(self.violations),
            "timer_violations": 0,
            "other_violations": 0,
            "threads_involved": set(),
        }

        for violation in self.violations:
            if violation["type"] == "timer_threading_violation":
                summary["timer_violations"] += 1
            else:
                summary["other_violations"] += 1

            summary["threads_involved"].add(str(violation["thread"]))

        summary["threads_involved"] = list(summary["threads_involved"])
        return summary


class QtThreadSafetyValidator:
    """Validate Qt thread safety patterns."""

    @staticmethod
    def validate_timer_usage(timer_obj, context="unknown"):
        """
        Validate that QTimer is being used safely.

        Args:
            timer_obj: QTimer instance to validate
            context: Context description for error reporting

        Returns:
            Tuple of (is_safe, issues)
        """
        issues = []
        is_safe = True

        if not isinstance(timer_obj, QTimer):
            return False, ["Object is not a QTimer"]

        # Check current thread
        current_thread = QThread.currentThread()
        app = QtWidgets.QApplication.instance()

        if app is None:
            issues.append("No QApplication instance available")
            is_safe = False
        else:
            app_thread = app.thread()

            if current_thread != app_thread and not isinstance(current_thread, QThread):
                issues.append(f"Timer in non-Qt thread: {type(current_thread)}")
                is_safe = False

        # Check if timer has parent
        if timer_obj.parent() is None:
            issues.append("Timer has no parent - potential memory leak")

        # Check timer interval
        if hasattr(timer_obj, "interval") and timer_obj.interval() <= 0:
            issues.append("Timer interval is zero or negative")

        return is_safe, issues

    @staticmethod
    def validate_signal_connection(signal, slot, context="unknown"):
        """
        Validate signal/slot connection for Qt5+ compatibility.

        Args:
            signal: Signal object
            slot: Slot callable or bound method
            context: Context description

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        is_valid = True

        # Check if signal has connect method
        if not hasattr(signal, "connect"):
            issues.append("Signal object missing connect method")
            is_valid = False

        # Check if slot is callable
        if not callable(slot):
            issues.append("Slot is not callable")
            is_valid = False

        # Check for old-style signal/slot syntax (Qt4)
        if isinstance(signal, str) or isinstance(slot, str):
            issues.append("Using deprecated string-based signal/slot syntax")
            is_valid = False

        # Check for bound method issues
        if hasattr(slot, "__self__"):
            slot_object = slot.__self__
            if isinstance(slot_object, QObject):
                # Check if slot object is in same thread as signal object
                if hasattr(signal, "parent") and signal.parent():
                    signal_thread = signal.parent().thread()
                    slot_thread = slot_object.thread()
                    if signal_thread != slot_thread:
                        issues.append("Signal and slot in different threads")

        return is_valid, issues

    @staticmethod
    def validate_gui_thread_access(operation_name="unknown"):
        """
        Validate that GUI operations are happening in the main thread.

        Args:
            operation_name: Name of the operation being validated

        Returns:
            Tuple of (is_main_thread, thread_info)
        """
        current_thread = QThread.currentThread()
        app = QtWidgets.QApplication.instance()

        if app is None:
            return False, "No QApplication available"

        main_thread = app.thread()
        is_main_thread = current_thread == main_thread

        thread_info = {
            "current_thread": str(current_thread),
            "main_thread": str(main_thread),
            "is_main_thread": is_main_thread,
            "operation": operation_name,
        }

        return is_main_thread, thread_info


class ThreadSafeQtDecorator:
    """Decorators for enforcing Qt thread safety."""

    @staticmethod
    def require_main_thread(func):
        """Decorator to ensure function runs in main Qt thread."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            is_main_thread, thread_info = (
                QtThreadSafetyValidator.validate_gui_thread_access(func.__name__)
            )

            if not is_main_thread:
                raise RuntimeError(
                    f"Function {func.__name__} must be called from main Qt thread. "
                    f"Current thread: {thread_info['current_thread']}"
                )

            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def validate_timer_creation(func):
        """Decorator to validate QTimer creation."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            # If function returns a QTimer, validate it
            if isinstance(result, QTimer):
                is_safe, issues = QtThreadSafetyValidator.validate_timer_usage(
                    result, func.__name__
                )
                if not is_safe:
                    warnings.warn(
                        f"Timer creation issues in {func.__name__}: {', '.join(issues)}",
                        UserWarning,
                        stacklevel=2,
                    )

            return result

        return wrapper

    @staticmethod
    def monitor_threading_violations(func):
        """Decorator to monitor function for threading violations."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            detector = ThreadingViolationDetector()
            detector.start_monitoring()

            try:
                result = func(*args, **kwargs)
                violations = detector.get_violations()

                if violations:
                    violation_summary = detector.get_violation_summary()
                    warnings.warn(
                        f"Threading violations detected in {func.__name__}: "
                        f"{violation_summary['total_violations']} violations",
                        UserWarning,
                        stacklevel=2,
                    )

                return result

            finally:
                detector.stop_monitoring()

        return wrapper


class BackgroundThreadTester:
    """Test background thread compliance with Qt requirements."""

    def __init__(self):
        self.test_results = []

    def test_timer_in_background_thread(self):
        """Test timer creation in background thread (should fail)."""
        violation_detected = False
        error_message = None

        def background_operation():
            nonlocal violation_detected, error_message
            try:
                # This should trigger a Qt threading violation
                timer = QTimer()
                timer.start(100)
                time.sleep(0.2)
                timer.stop()
            except Exception as e:
                violation_detected = True
                error_message = str(e)

        thread = threading.Thread(target=background_operation)
        thread.start()
        thread.join()

        result = {
            "test_name": "timer_in_background_thread",
            "violation_detected": violation_detected,
            "error_message": error_message,
            "expected_result": "should_fail",
        }

        self.test_results.append(result)
        return result

    def test_proper_qthread_timer(self):
        """Test timer creation in proper QThread (should succeed)."""

        class TimerWorker(QThread):
            timer_created = Signal()
            timer_worked = Signal()

            def run(self):
                # This should work - timer in QThread
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(self.timer_worked.emit)
                self.timer_created.emit()
                timer.start(10)
                self.exec()

        worker = TimerWorker()
        timer_created = False
        timer_worked = False

        def on_timer_created():
            nonlocal timer_created
            timer_created = True

        def on_timer_worked():
            nonlocal timer_worked
            timer_worked = True
            worker.quit()

        worker.timer_created.connect(on_timer_created)
        worker.timer_worked.connect(on_timer_worked)

        worker.start()
        worker.wait(1000)  # Wait up to 1 second

        result = {
            "test_name": "proper_qthread_timer",
            "timer_created": timer_created,
            "timer_worked": timer_worked,
            "success": timer_created and timer_worked,
            "expected_result": "should_succeed",
        }

        self.test_results.append(result)
        return result

    def test_signal_connection_threading(self):
        """Test signal/slot connections across threads."""

        class SignalObject(QObject):
            test_signal = Signal(str)

        class SlotObject(QObject):
            def __init__(self):
                super().__init__()
                self.received_messages = []

            def test_slot(self, message):
                self.received_messages.append(message)

        signal_obj = SignalObject()
        slot_obj = SlotObject()

        # Test connection
        connection_valid, issues = QtThreadSafetyValidator.validate_signal_connection(
            signal_obj.test_signal, slot_obj.test_slot, "threading_test"
        )

        # Test actual connection
        signal_obj.test_signal.connect(slot_obj.test_slot)
        signal_obj.test_signal.emit("test_message")

        # Process events to ensure delivery
        if QtWidgets.QApplication.instance():
            QtWidgets.QApplication.instance().processEvents()

        result = {
            "test_name": "signal_connection_threading",
            "connection_valid": connection_valid,
            "validation_issues": issues,
            "message_received": len(slot_obj.received_messages) > 0,
            "expected_result": "should_succeed",
        }

        self.test_results.append(result)
        return result

    def run_all_tests(self):
        """Run all background thread compliance tests."""
        tests = [
            self.test_timer_in_background_thread,
            self.test_proper_qthread_timer,
            self.test_signal_connection_threading,
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                results.append(
                    {"test_name": test.__name__, "error": str(e), "success": False}
                )

        return results

    def get_compliance_report(self):
        """Generate compliance report from test results."""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.get("success", False))
        failed_tests = total_tests - passed_tests

        report = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "compliance_score": (passed_tests / total_tests) * 100
            if total_tests > 0
            else 0,
            "test_details": self.test_results,
            "recommendations": [],
        }

        # Generate recommendations based on failures
        for result in self.test_results:
            if not result.get("success", True):
                test_name = result["test_name"]
                if "timer" in test_name and "background" in test_name:
                    report["recommendations"].append(
                        "Use QThread instead of threading.Thread for Qt timer operations"
                    )
                elif "signal" in test_name:
                    report["recommendations"].append(
                        "Ensure signal/slot connections use proper Qt5+ syntax"
                    )

        return report


# Utility functions for common Qt threading patterns
def create_thread_safe_timer(parent=None, interval=1000, single_shot=False):
    """
    Create a thread-safe QTimer with validation.

    Args:
        parent: Parent QObject
        interval: Timer interval in milliseconds
        single_shot: Whether timer should be single-shot

    Returns:
        QTimer instance or None if creation failed
    """
    try:
        # Validate thread context
        is_main_thread, thread_info = (
            QtThreadSafetyValidator.validate_gui_thread_access("timer_creation")
        )

        if not is_main_thread:
            warnings.warn(
                f"Creating timer in non-main thread: {thread_info['current_thread']}",
                UserWarning,
                stacklevel=2,
            )

        timer = QTimer(parent)
        timer.setInterval(interval)
        timer.setSingleShot(single_shot)

        # Validate the created timer
        is_safe, issues = QtThreadSafetyValidator.validate_timer_usage(
            timer, "create_thread_safe_timer"
        )

        if not is_safe:
            warnings.warn(
                f"Timer creation issues: {', '.join(issues)}", UserWarning, stacklevel=2
            )

        return timer

    except Exception as e:
        warnings.warn(
            f"Failed to create thread-safe timer: {e}", UserWarning, stacklevel=2
        )
        return None


def safe_signal_connect(
    signal, slot, connection_type=QtCore.Qt.ConnectionType.AutoConnection
):
    """
    Safely connect signal to slot with validation.

    Args:
        signal: Qt signal object
        slot: Callable slot
        connection_type: Qt connection type

    Returns:
        True if connection successful, False otherwise
    """
    try:
        # Validate connection
        is_valid, issues = QtThreadSafetyValidator.validate_signal_connection(
            signal, slot
        )

        if not is_valid:
            warnings.warn(
                f"Signal connection issues: {', '.join(issues)}",
                UserWarning,
                stacklevel=2,
            )
            return False

        # Perform connection
        signal.connect(slot, connection_type)
        return True

    except Exception as e:
        warnings.warn(f"Failed to connect signal: {e}", UserWarning, stacklevel=2)
        return False


# High-level utility functions
def detect_threading_violations(operation_func, *args, **kwargs):
    """
    Detect threading violations during operation execution.

    Args:
        operation_func: Function to execute while monitoring
        *args: Arguments for operation_func
        **kwargs: Keyword arguments for operation_func

    Returns:
        Tuple of (result, violations_detected, violation_details)
    """
    detector = ThreadingViolationDetector()
    detector.start_monitoring()

    try:
        result = operation_func(*args, **kwargs)
        violations = detector.get_violations()
        violations_detected = len(violations) > 0
        return result, violations_detected, violations
    finally:
        detector.stop_monitoring()


# Context manager for threading violation detection
@contextlib.contextmanager
def detect_threading_violations():
    """Context manager to detect Qt threading violations."""
    detector = ThreadingViolationDetector()
    detector.start_monitoring()

    try:
        yield detector
    finally:
        detector.stop_monitoring()
