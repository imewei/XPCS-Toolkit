#!/usr/bin/env python3
"""
Test Isolation Utilities for XPCS Toolkit Test Suite

This module provides utilities for improved test isolation, cleanup,
and reliability across the entire test suite.

Created: 2025-09-16
"""

import gc
import os
import shutil
import threading
import time
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

    # Mock h5py for basic testing
    class MockH5pyFile:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def create_dataset(self, *args, **kwargs):
            pass

        def create_group(self, *args, **kwargs):
            return self

        def __getitem__(self, key):
            return self

        def __setitem__(self, key, value):
            pass

    class MockH5py:
        File = MockH5pyFile

    h5py = MockH5py()


class TestIsolationManager:
    """Manages test isolation and cleanup between test runs."""

    def __init__(self):
        self._temp_dirs: list[Path] = []
        self._temp_files: list[Path] = []
        self._mock_patches: list[Any] = []
        self._weak_refs: set[weakref.ref] = set()
        self._threads: list[threading.Thread] = []
        self._original_env: dict[str, str] = {}

    def register_temp_dir(self, path: Path) -> Path:
        """Register a temporary directory for cleanup."""
        self._temp_dirs.append(path)
        return path

    def register_temp_file(self, path: Path) -> Path:
        """Register a temporary file for cleanup."""
        self._temp_files.append(path)
        return path

    def register_mock_patch(self, mock_patch: Any) -> Any:
        """Register a mock patch for cleanup."""
        self._mock_patches.append(mock_patch)
        return mock_patch

    def register_object(self, obj: Any) -> Any:
        """Register an object for weak reference tracking."""
        self._weak_refs.add(weakref.ref(obj))
        return obj

    def register_thread(self, thread: threading.Thread) -> threading.Thread:
        """Register a thread for proper cleanup."""
        self._threads.append(thread)
        return thread

    def backup_env_var(self, var_name: str) -> None:
        """Backup an environment variable for restoration."""
        self._original_env[var_name] = os.environ.get(var_name)

    def cleanup(self) -> None:
        """Perform comprehensive cleanup of all registered resources."""
        # Clean up mock patches
        for patch_obj in reversed(self._mock_patches):
            try:
                if hasattr(patch_obj, "stop"):
                    patch_obj.stop()
            except Exception:
                pass  # Ignore cleanup errors

        # Clean up threads
        for thread in self._threads:
            try:
                if thread.is_alive():
                    thread.join(timeout=1.0)
            except Exception:
                pass

        # Restore environment variables
        for var_name, original_value in self._original_env.items():
            if original_value is None:
                os.environ.pop(var_name, None)
            else:
                os.environ[var_name] = original_value

        # Clean up temporary files
        for temp_file in self._temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

        # Clean up temporary directories
        for temp_dir in reversed(self._temp_dirs):
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
            except Exception:
                pass

        # Force garbage collection
        for _ in range(3):
            gc.collect()

        # Small delay to allow system cleanup
        time.sleep(0.01)

        # Clear all registrations
        self._temp_dirs.clear()
        self._temp_files.clear()
        self._mock_patches.clear()
        self._weak_refs.clear()
        self._threads.clear()
        self._original_env.clear()


@pytest.fixture(scope="function")
def isolation_manager():
    """Provide test isolation manager with automatic cleanup."""
    manager = TestIsolationManager()
    try:
        yield manager
    finally:
        manager.cleanup()


@contextmanager
def isolated_test_environment():
    """Context manager for completely isolated test environment."""
    manager = TestIsolationManager()
    try:
        yield manager
    finally:
        manager.cleanup()


class TestPerformanceMonitor:
    """Monitor test performance and detect slow/flaky tests."""

    def __init__(self):
        self._test_times: dict[str, list[float]] = {}
        self._slow_threshold = 1.0  # seconds
        self._flaky_variance_threshold = 0.5  # 50% variance indicates flakiness

    def record_test_time(self, test_name: str, duration: float) -> None:
        """Record execution time for a test."""
        if test_name not in self._test_times:
            self._test_times[test_name] = []
        self._test_times[test_name].append(duration)

    def get_slow_tests(self) -> list[str]:
        """Get list of tests that exceed the slow threshold."""
        slow_tests = []
        for test_name, times in self._test_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > self._slow_threshold:
                slow_tests.append(test_name)
        return slow_tests

    def get_flaky_tests(self) -> list[str]:
        """Get list of tests with high variance in execution times."""
        flaky_tests = []
        for test_name, times in self._test_times.items():
            if len(times) < 2:
                continue

            avg_time = sum(times) / len(times)
            variance = sum((t - avg_time) ** 2 for t in times) / len(times)
            coefficient_of_variation = (variance**0.5) / avg_time if avg_time > 0 else 0

            if coefficient_of_variation > self._flaky_variance_threshold:
                flaky_tests.append(test_name)

        return flaky_tests

    def get_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "total_tests_monitored": len(self._test_times),
            "slow_tests": self.get_slow_tests(),
            "flaky_tests": self.get_flaky_tests(),
            "average_test_times": {
                test: sum(times) / len(times)
                for test, times in self._test_times.items()
            },
        }


# Global performance monitor instance
_performance_monitor = TestPerformanceMonitor()


def get_performance_monitor() -> TestPerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


class TestDataFactory:
    """Factory for creating consistent test data across test suites."""

    @staticmethod
    def create_minimal_hdf5_file(temp_dir: Path, file_name: str = "test.h5") -> Path:
        """Create a minimal HDF5 file for testing."""
        import h5py
        import numpy as np

        file_path = temp_dir / file_name

        with h5py.File(file_path, "w") as f:
            # Create minimal XPCS structure
            entry = f.create_group("xpcs")

            # Required attributes
            entry.attrs["detector"] = "test_detector"
            entry.attrs["sample"] = "test_sample"

            # Minimal datasets
            entry.create_dataset("g2", data=np.array([1.0, 0.8, 0.6, 0.4, 0.2]))
            entry.create_dataset("tau", data=np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]))
            entry.create_dataset(
                "g2_err", data=np.array([0.01, 0.01, 0.01, 0.01, 0.01])
            )

        return file_path

    @staticmethod
    def create_synthetic_xpcs_data(
        shape: tuple = (100, 100), num_frames: int = 10
    ) -> dict[str, Any]:
        """Create synthetic XPCS data for testing."""
        import numpy as np

        return {
            "saxs_2d": np.random.poisson(100, (num_frames, *shape)),
            "g2": np.exp(-np.logspace(-6, -2, 50)),
            "tau": np.logspace(-6, -2, 50),
            "g2_err": np.ones(50) * 0.01,
            "metadata": {
                "detector_distance": 2.0,
                "pixel_size": 75e-6,
                "wavelength": 1.24e-10,
                "exposure_time": 0.1,
            },
        }


def require_test_data(test_func):
    """Decorator to ensure test data is available before running test."""

    def wrapper(*args, **kwargs):
        # Could check for required test data files here
        return test_func(*args, **kwargs)

    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 0.1):
    """Decorator to retry flaky tests."""

    def decorator(test_func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return test_func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2**attempt))  # Exponential backoff
                    continue
            raise last_exception

        return wrapper

    return decorator


def skip_if_no_display():
    """Skip test if no display is available."""
    return pytest.mark.skipif(
        os.environ.get("DISPLAY") is None and os.environ.get("WAYLAND_DISPLAY") is None,
        reason="No display available for GUI tests",
    )


def skip_if_insufficient_memory(required_mb: int = 1000):
    """Skip test if insufficient memory is available."""
    import psutil

    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    return pytest.mark.skipif(
        available_mb < required_mb,
        reason=f"Insufficient memory: {available_mb:.0f}MB < {required_mb}MB required",
    )


# Performance monitoring decorator
def monitor_performance(test_func):
    """Decorator to monitor test performance."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = test_func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            test_name = f"{test_func.__module__}.{test_func.__name__}"
            _performance_monitor.record_test_time(test_name, duration)

    return wrapper
