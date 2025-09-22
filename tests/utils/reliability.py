#!/usr/bin/env python3
"""
Test Reliability Framework for XPCS Toolkit

This module provides utilities for detecting and handling test reliability issues,
including flaky tests, timing dependencies, and resource contention.

Created: 2025-09-16
"""

import functools
import queue
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest


class FlakinessDetector:
    """Detect and analyze test flakiness patterns."""

    def __init__(self, min_runs: int = 3, success_threshold: float = 0.8):
        self.min_runs = min_runs
        self.success_threshold = success_threshold
        self.test_results: dict[str, list[bool]] = {}

    def record_result(self, test_name: str, success: bool) -> None:
        """Record a test result."""
        if test_name not in self.test_results:
            self.test_results[test_name] = []
        self.test_results[test_name].append(success)

    def is_flaky(self, test_name: str) -> bool:
        """Determine if a test is flaky based on recorded results."""
        if test_name not in self.test_results:
            return False

        results = self.test_results[test_name]
        if len(results) < self.min_runs:
            return False

        success_rate = sum(results) / len(results)
        return 0 < success_rate < self.success_threshold

    def get_flaky_tests(self) -> list[str]:
        """Get list of all flaky tests."""
        return [test for test in self.test_results if self.is_flaky(test)]

    def get_reliability_report(self) -> dict[str, Any]:
        """Generate comprehensive reliability report."""
        report = {
            "total_tests": len(self.test_results),
            "flaky_tests": [],
            "reliable_tests": [],
            "insufficient_data": [],
        }

        for test_name, results in self.test_results.items():
            if len(results) < self.min_runs:
                report["insufficient_data"].append(test_name)
            elif self.is_flaky(test_name):
                success_rate = sum(results) / len(results)
                report["flaky_tests"].append(
                    {
                        "name": test_name,
                        "success_rate": success_rate,
                        "total_runs": len(results),
                    }
                )
            else:
                report["reliable_tests"].append(test_name)

        return report


class TestStabilizer:
    """Stabilize flaky tests through various techniques."""

    @staticmethod
    def with_retry(
        max_attempts: int = 3, delay: float = 0.1, backoff_factor: float = 2.0
    ):
        """Retry decorator for flaky tests."""

        def decorator(test_func: Callable) -> Callable:
            @functools.wraps(test_func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_attempts):
                    try:
                        return test_func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            time.sleep(current_delay)
                            current_delay *= backoff_factor

                # If all attempts failed, raise the last exception
                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def with_timeout(timeout_seconds: float):
        """Timeout decorator for tests that might hang."""

        def decorator(test_func: Callable) -> Callable:
            @functools.wraps(test_func)
            def wrapper(*args, **kwargs):
                result_queue = queue.Queue()
                exception_queue = queue.Queue()

                def target():
                    try:
                        result = test_func(*args, **kwargs)
                        result_queue.put(result)
                    except Exception as e:
                        exception_queue.put(e)

                thread = threading.Thread(target=target)
                thread.start()
                thread.join(timeout_seconds)

                if thread.is_alive():
                    # Force thread termination is not safe in Python,
                    # so we just raise a timeout exception
                    raise TimeoutError(
                        f"Test {test_func.__name__} timed out after {timeout_seconds}s"
                    )

                if not exception_queue.empty():
                    raise exception_queue.get()

                if not result_queue.empty():
                    return result_queue.get()

                return None

            return wrapper

        return decorator

    @staticmethod
    def with_deterministic_timing():
        """Decorator to make timing-dependent tests more deterministic."""

        def decorator(test_func: Callable) -> Callable:
            @functools.wraps(test_func)
            def wrapper(*args, **kwargs):
                # Mock time-related functions for deterministic behavior
                with patch("time.time") as mock_time, patch("time.sleep") as mock_sleep:
                    # Create deterministic time progression
                    current_time = [1000.0]  # Starting time

                    def increment_time():
                        current_time[0] += 0.1
                        return current_time[0]

                    mock_time.side_effect = increment_time
                    mock_sleep.side_effect = lambda duration: setattr(
                        current_time, 0, current_time[0] + duration
                    )

                    return test_func(*args, **kwargs)

            return wrapper

        return decorator


class ResourceLockManager:
    """Manage resource locks to prevent test interference."""

    def __init__(self):
        self._locks: dict[str, threading.Lock] = {}

    def get_lock(self, resource_name: str) -> threading.Lock:
        """Get or create a lock for a resource."""
        if resource_name not in self._locks:
            self._locks[resource_name] = threading.Lock()
        return self._locks[resource_name]

    @contextmanager
    def acquire_resource(self, resource_name: str, timeout: float = 5.0):
        """Acquire exclusive access to a resource."""
        lock = self.get_lock(resource_name)
        acquired = lock.acquire(timeout=timeout)

        if not acquired:
            raise TimeoutError(
                f"Could not acquire lock for resource '{resource_name}' within {timeout}s"
            )

        try:
            yield
        finally:
            lock.release()


# Global instances
_flakiness_detector = FlakinessDetector()
_resource_lock_manager = ResourceLockManager()


def get_flakiness_detector() -> FlakinessDetector:
    """Get the global flakiness detector."""
    return _flakiness_detector


def get_resource_lock_manager() -> ResourceLockManager:
    """Get the global resource lock manager."""
    return _resource_lock_manager


def reliable_test(
    max_retries: int = 2,
    timeout: float | None = None,
    deterministic_timing: bool = False,
    required_resources: list[str] | None = None,
):
    """Decorator to make tests more reliable."""

    def decorator(test_func: Callable) -> Callable:
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            # Apply timeout if specified
            if timeout:
                test_func_with_timeout = TestStabilizer.with_timeout(timeout)(test_func)
            else:
                test_func_with_timeout = test_func

            # Apply deterministic timing if requested
            if deterministic_timing:
                test_func_with_timeout = TestStabilizer.with_deterministic_timing()(
                    test_func_with_timeout
                )

            # Apply retry logic
            test_func_with_retry = TestStabilizer.with_retry(max_retries + 1)(
                test_func_with_timeout
            )

            # Handle resource locking
            if required_resources:

                def resource_locked_test(*args, **kwargs):
                    for resource in required_resources:
                        with _resource_lock_manager.acquire_resource(resource):
                            pass  # All resources acquired
                    return test_func_with_retry(*args, **kwargs)

                return resource_locked_test(*args, **kwargs)
            return test_func_with_retry(*args, **kwargs)

        return wrapper

    return decorator


class TestEnvironmentValidator:
    """Validate test environment prerequisites."""

    @staticmethod
    def check_memory_available(required_mb: int) -> bool:
        """Check if sufficient memory is available."""
        try:
            import psutil

            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            return available_mb >= required_mb
        except ImportError:
            return True  # Assume sufficient if psutil not available

    @staticmethod
    def check_disk_space(required_mb: int, path: str = "/tmp") -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil

            free_bytes = shutil.disk_usage(path).free
            free_mb = free_bytes / (1024 * 1024)
            return free_mb >= required_mb
        except Exception:
            return True  # Assume sufficient if check fails

    @staticmethod
    def check_network_available() -> bool:
        """Check if network connectivity is available."""
        try:
            import socket

            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect(("8.8.8.8", 53))
            return True
        except Exception:
            return False

    @staticmethod
    def check_display_available() -> bool:
        """Check if display is available for GUI tests."""
        import os

        return (
            os.environ.get("DISPLAY") is not None
            or os.environ.get("WAYLAND_DISPLAY") is not None
        )

    @classmethod
    def validate_environment(cls, requirements: dict[str, Any]) -> dict[str, bool]:
        """Validate multiple environment requirements."""
        results = {}

        if "memory_mb" in requirements:
            results["memory"] = cls.check_memory_available(requirements["memory_mb"])

        if "disk_mb" in requirements:
            path = requirements.get("disk_path", "/tmp")
            results["disk_space"] = cls.check_disk_space(requirements["disk_mb"], path)

        if requirements.get("network"):
            results["network"] = cls.check_network_available()

        if requirements.get("display"):
            results["display"] = cls.check_display_available()

        return results


def validate_test_environment(**requirements):
    """Decorator to validate test environment before running test."""

    def decorator(test_func: Callable) -> Callable:
        @functools.wraps(test_func)
        def wrapper(*args, **kwargs):
            validator = TestEnvironmentValidator()
            validation_results = validator.validate_environment(requirements)

            # Check if any requirements failed
            failed_requirements = [
                req for req, passed in validation_results.items() if not passed
            ]

            if failed_requirements:
                pytest.skip(
                    f"Environment requirements not met: {', '.join(failed_requirements)}"
                )

            return test_func(*args, **kwargs)

        return wrapper

    return decorator


# Convenience decorators
def require_memory(mb: int):
    """Require minimum memory for test."""
    return validate_test_environment(memory_mb=mb)


def require_disk_space(mb: int, path: str = "/tmp"):
    """Require minimum disk space for test."""
    return validate_test_environment(disk_mb=mb, disk_path=path)


def require_network():
    """Require network connectivity for test."""
    return validate_test_environment(network=True)


def require_display():
    """Require display for GUI tests."""
    return validate_test_environment(display=True)
