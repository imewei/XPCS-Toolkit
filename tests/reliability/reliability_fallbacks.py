"""
Fallback implementations for reliability framework testing.

This module provides mock implementations when the actual reliability modules
are not available, allowing tests to run successfully.
"""

import contextlib
import enum


class XPCSBaseError(Exception):
    pass


class XPCSDataError(XPCSBaseError):
    pass


class XPCSFileError(XPCSBaseError):
    pass


class XPCSValidationError(XPCSBaseError):
    pass


class XPCSMemoryError(XPCSBaseError):
    pass


def convert_exception(exc):
    return exc


def handle_exceptions(exc_type):
    def decorator(func):
        return func

    return decorator


@contextlib.contextmanager
def exception_context():
    yield


class ValidationLevel(enum.Enum):
    NONE = "none"
    BASIC = "basic"
    FULL = "full"


def validate_input(check_types=False, cache_results=False, level=ValidationLevel.BASIC):
    """Efficient fallback decorator for validate_input."""

    def decorator(func):
        # For performance tests, just return the original function unchanged
        return func

    return decorator


def get_validation_cache():
    return {}


def get_fallback_manager():
    return SmartFallbackManager()


@contextlib.contextmanager
def reliability_context(max_retries=3, retry_delay=0.01):
    """Mock reliability context that doesn't actually retry."""
    yield


class SmartFallbackManager:
    def get_fallback_value(self, key, default=None):
        return default


class HealthStatus(enum.Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthMonitor:
    def __init__(self, monitoring_interval=1.0):
        self.status = HealthStatus.HEALTHY
        self.monitoring_interval = monitoring_interval

    def get_status(self):
        return self.status

    def check_system_health(self):
        return True

    def start_monitoring(self):
        pass

    def stop_monitoring(self):
        pass

    def get_health_summary(self):
        return {
            "overall_status": "excellent",
            "cpu_usage": 10.0,
            "memory_usage": 30.0,
            "disk_usage": 40.0,
            "alerts": [],
        }


def get_health_monitor():
    return HealthMonitor()


@contextlib.contextmanager
def health_monitoring_context():
    yield


class StateTransition(enum.Enum):
    VALID = "valid"
    INVALID = "invalid"


class StateValidationLevel(enum.Enum):
    NONE = "none"
    BASIC = "basic"
    FULL = "full"


class LockFreeStateValidator:
    def __init__(self):
        self._tracked_objects = set()
        self._total_transitions = 0
        self._valid_transitions = 0
        self._invalid_transitions = 0

    def validate_state_transition(self, from_state, to_state):
        self._total_transitions += 1
        self._valid_transitions += 1
        return StateTransition.VALID

    def start_background_validation(self, interval=1.0):
        pass

    def stop_background_validation(self):
        pass

    def track_object(self, obj):
        self._tracked_objects.add(id(obj))

    def get_statistics(self):
        return {
            "tracked_objects": len(self._tracked_objects),
            "total_transitions": self._total_transitions,
            "invalid_transitions": self._invalid_transitions,
            "valid_transitions": self._valid_transitions,
        }


# Global validator instance for tracking
_global_validator = None


def get_state_validator():
    global _global_validator
    if _global_validator is None:
        _global_validator = LockFreeStateValidator()
    return _global_validator


def track_object_state(obj, state=None):
    """Track object state in global validator."""
    validator = get_state_validator()
    validator.track_object(obj)
    return obj
