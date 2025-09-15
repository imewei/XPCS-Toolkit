"""Error handling test fixtures and utilities."""

import os
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest


@pytest.fixture
def error_temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for error handling tests."""
    temp_dir = tempfile.mkdtemp(prefix="xpcs_error_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def corrupted_hdf5_file(error_temp_dir) -> str:
    """Create a corrupted HDF5 file for testing error handling."""
    file_path = os.path.join(error_temp_dir, "corrupted_test.h5")

    # Create a valid HDF5 file first
    with h5py.File(file_path, "w") as f:
        f.create_dataset("test_data", data=np.random.rand(10, 10))

    # Corrupt the file by truncating it
    with open(file_path, "r+b") as f:
        f.seek(100)  # Move to position 100
        f.truncate()  # Truncate file at this position

    return file_path


@pytest.fixture
def invalid_hdf5_file(error_temp_dir) -> str:
    """Create an invalid HDF5 file (not actually HDF5 format)."""
    file_path = os.path.join(error_temp_dir, "invalid_test.h5")

    # Write some non-HDF5 content
    with open(file_path, "w") as f:
        f.write("This is not a valid HDF5 file")

    return file_path


@pytest.fixture
def missing_file_path(error_temp_dir) -> str:
    """Return path to a non-existent file."""
    return os.path.join(error_temp_dir, "nonexistent_file.h5")


@pytest.fixture
def permission_denied_file(error_temp_dir) -> str:
    """Create a file with no read permissions."""
    file_path = os.path.join(error_temp_dir, "no_permission.h5")

    # Create a valid file first
    with h5py.File(file_path, "w") as f:
        f.create_dataset("test_data", data=np.random.rand(5, 5))

    # Remove read permissions (if not on Windows)
    if os.name != "nt":
        os.chmod(file_path, 0o000)

    return file_path


@pytest.fixture
def memory_limited_environment():
    """Mock environment with limited memory."""
    with patch("psutil.virtual_memory") as mock_memory:
        mock_memory.return_value = Mock(
            total=1024 * 1024 * 1024,  # 1 GB
            available=128 * 1024 * 1024,  # 128 MB available
            percent=87.5,  # 87.5% memory usage
        )
        yield mock_memory


@pytest.fixture
def disk_space_limited_environment():
    """Mock environment with limited disk space."""
    with patch("shutil.disk_usage") as mock_disk:
        mock_disk.return_value = (
            1024 * 1024 * 1024,  # 1 GB total
            50 * 1024 * 1024,  # 50 MB free
            974 * 1024 * 1024,  # 974 MB used
        )
        yield mock_disk


@pytest.fixture
def file_handle_exhausted_environment():
    """Mock environment where file handle limit is reached."""
    original_open = open
    open_count = 0
    max_files = 5

    def limited_open(*args, **kwargs):
        nonlocal open_count
        open_count += 1
        if open_count > max_files:
            raise OSError("Too many open files")
        return original_open(*args, **kwargs)

    with patch("builtins.open", side_effect=limited_open):
        yield max_files


@contextmanager
def expect_error_logging(logger_name: str, error_level: str = "ERROR"):
    """Context manager to verify error logging occurs."""
    with patch(f"{logger_name}.{error_level.lower()}") as mock_log:
        yield mock_log


@pytest.fixture
def mock_h5py_errors():
    """Fixture providing various h5py error scenarios."""

    def create_error_scenarios():
        return {
            "file_not_found": h5py.h5f.FileNotFoundError,
            "invalid_file": ValueError("Unable to open file"),
            "permission_denied": PermissionError("Permission denied"),
            "io_error": OSError("I/O operation failed"),
            "corrupted_file": h5py.h5e.HDF5ExtError("Corrupted data"),
            "dataset_not_found": KeyError("Dataset not found"),
            "memory_error": MemoryError("Unable to allocate memory"),
        }

    return create_error_scenarios()


@pytest.fixture
def numpy_error_scenarios():
    """Fixture providing various numpy error scenarios."""

    def create_scenarios():
        return {
            "overflow": np.array([1e308, 1e308]),  # Will overflow
            "underflow": np.array([1e-324, 1e-324]),  # Will underflow
            "division_by_zero": np.array([1.0, 0.0]),  # For division
            "invalid_operation": np.array([np.inf, -np.inf, np.nan]),
            "empty_array": np.array([]),
            "single_element": np.array([1.0]),
            "negative_values": np.array([-1, -2, -3]),
            "extreme_values": np.array(
                [np.finfo(np.float64).max, np.finfo(np.float64).min]
            ),
        }

    return create_scenarios()


class ErrorInjector:
    """Utility class for systematic error injection."""

    def __init__(self):
        self.active_patches = []
        self.error_count = 0

    def inject_io_error(self, target_func: str, error_type: Exception = OSError):
        """Inject I/O errors into specified functions."""

        def error_side_effect(*args, **kwargs):
            self.error_count += 1
            raise error_type(f"Injected I/O error #{self.error_count}")

        patcher = patch(target_func, side_effect=error_side_effect)
        self.active_patches.append(patcher)
        return patcher.start()

    def inject_memory_error(self, target_func: str):
        """Inject memory errors."""

        def memory_error_side_effect(*args, **kwargs):
            self.error_count += 1
            raise MemoryError(f"Injected memory error #{self.error_count}")

        patcher = patch(target_func, side_effect=memory_error_side_effect)
        self.active_patches.append(patcher)
        return patcher.start()

    def inject_timeout_error(self, target_func: str, timeout_seconds: float = 0.1):
        """Inject timeout errors by making functions hang."""

        def timeout_side_effect(*args, **kwargs):
            time.sleep(timeout_seconds * 2)  # Sleep longer than timeout
            raise TimeoutError(f"Operation timed out after {timeout_seconds}s")

        patcher = patch(target_func, side_effect=timeout_side_effect)
        self.active_patches.append(patcher)
        return patcher.start()

    def cleanup(self):
        """Clean up all active patches."""
        for patcher in self.active_patches:
            patcher.stop()
        self.active_patches.clear()
        self.error_count = 0


@pytest.fixture
def error_injector():
    """Provide error injection utility."""
    injector = ErrorInjector()
    yield injector
    injector.cleanup()


class ResourceExhaustion:
    """Utility for simulating resource exhaustion scenarios."""

    @staticmethod
    def simulate_memory_pressure(threshold: float = 0.9):
        """Simulate high memory pressure."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.get_cached_memory_monitor"
        ) as mock_monitor:
            mock_instance = Mock()
            mock_instance.get_memory_info.return_value = (
                8000,  # 8GB used
                1000,  # 1GB available
                threshold,
            )
            mock_instance.is_memory_pressure_high.return_value = True
            mock_instance.get_memory_status.return_value = Mock(percent_used=threshold)
            mock_monitor.return_value = mock_instance
            return mock_monitor

    @staticmethod
    def simulate_disk_full():
        """Simulate disk full condition."""
        with (
            patch("os.path.getsize") as mock_size,
            patch("shutil.disk_usage") as mock_usage,
        ):
            mock_size.return_value = 1000000
            mock_usage.return_value = (1000000000, 0, 1000000000)  # No free space
            return mock_usage

    @staticmethod
    def simulate_network_failure():
        """Simulate network connectivity issues."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = OSError("Network is unreachable")
            return mock_urlopen


@pytest.fixture
def resource_exhaustion():
    """Provide resource exhaustion simulation utilities."""
    return ResourceExhaustion()


@pytest.fixture
def edge_case_data():
    """Generate edge case data scenarios."""
    return {
        "empty_arrays": {
            "float64": np.array([], dtype=np.float64),
            "int32": np.array([], dtype=np.int32),
            "complex128": np.array([], dtype=np.complex128),
        },
        "single_element_arrays": {
            "normal": np.array([1.0]),
            "zero": np.array([0.0]),
            "negative": np.array([-1.0]),
            "inf": np.array([np.inf]),
            "nan": np.array([np.nan]),
        },
        "extreme_values": {
            "max_float": np.array([np.finfo(np.float64).max]),
            "min_float": np.array([np.finfo(np.float64).min]),
            "max_int": np.array([np.iinfo(np.int64).max]),
            "min_int": np.array([np.iinfo(np.int64).min]),
        },
        "special_values": {
            "all_zeros": np.zeros(1000),
            "all_ones": np.ones(1000),
            "alternating": np.array([1, -1] * 500),
            "geometric_series": np.array([2**i for i in range(50)]),
        },
        "problematic_shapes": {
            "very_long": np.random.rand(1000000),
            "very_wide": np.random.rand(1000, 1000),
            "single_row": np.random.rand(1, 1000),
            "single_column": np.random.rand(1000, 1),
            "single_element_2d": np.random.rand(1, 1),
        },
    }


@pytest.fixture
def threading_error_scenarios():
    """Provide threading-related error scenarios."""

    class ThreadingErrors:
        @staticmethod
        def deadlock_simulation():
            """Simulate deadlock conditions."""
            lock1 = threading.Lock()
            lock2 = threading.Lock()

            def task1():
                with lock1:
                    time.sleep(0.1)
                    with lock2:
                        pass

            def task2():
                with lock2:
                    time.sleep(0.1)
                    with lock1:
                        pass

            thread1 = threading.Thread(target=task1)
            thread2 = threading.Thread(target=task2)

            return thread1, thread2

        @staticmethod
        def race_condition_data():
            """Provide data structure for race condition testing."""
            return {"counter": 0, "lock": threading.Lock()}

        @staticmethod
        def thread_exception():
            """Create a thread that raises an exception."""

            def failing_task():
                raise RuntimeError("Thread exception for testing")

            return threading.Thread(target=failing_task)

    return ThreadingErrors()
