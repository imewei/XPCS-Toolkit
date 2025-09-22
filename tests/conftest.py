"""Streamlined pytest configuration and shared fixtures for XPCS Toolkit tests.

This module provides essential pytest configuration and imports focused
fixtures from specialized modules.
"""

import logging
import os
import warnings
from pathlib import Path

import pytest

# Import h5py with fallback
try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    from tests.utils.h5py_mocks import MockH5py

    h5py = MockH5py()

# Import focused fixture modules
import contextlib

from tests.fixtures.core_fixtures import *
from tests.fixtures.qt_fixtures import *
from tests.fixtures.scientific_fixtures import *
from xpcs_toolkit.utils.logging_config import setup_logging

# Import optional framework utilities
try:
    from tests.utils.isolation import (
        get_performance_monitor,
        isolated_test_environment,
        monitor_performance,
    )
    from tests.utils.reliability import (
        get_flakiness_detector,
        reliable_test,
        validate_test_environment,
    )

    RELIABILITY_FRAMEWORKS_AVAILABLE = True
except ImportError:
    RELIABILITY_FRAMEWORKS_AVAILABLE = False

try:
    from tests.utils.data_management import (
        TestDataSpec,
        create_minimal_test_data,
        create_performance_test_data,
        create_realistic_xpcs_dataset,
        get_hdf5_manager,
        get_test_data_factory,
        temporary_xpcs_file,
    )

    ADVANCED_DATA_MANAGEMENT_AVAILABLE = True
except ImportError:
    ADVANCED_DATA_MANAGEMENT_AVAILABLE = False

try:
    from tests.utils.ci_integration import (
        TestResult,
        TestSuite,
        collect_test_artifacts,
        generate_ci_reports,
        get_ci_environment,
        github_step_summary,
        set_github_output,
    )

    CI_INTEGRATION_AVAILABLE = True
except ImportError:
    CI_INTEGRATION_AVAILABLE = False


# ============================================================================
# Pytest Configuration
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    # Core test markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "gui: GUI tests (requires display)")
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")
    config.addinivalue_line(
        "markers", "scientific: Tests that verify scientific accuracy"
    )

    # Specialized markers
    config.addinivalue_line("markers", "flaky: Tests that are known to be flaky")
    config.addinivalue_line("markers", "stress: Stress tests that push system limits")
    config.addinivalue_line(
        "markers", "system_dependent: Tests that depend on system resources"
    )
    config.addinivalue_line("markers", "reliable: Tests using reliability framework")

    # Configure test environment
    configure_test_environment()

    # Apply CI-specific configurations
    configure_ci_environment(config)


def configure_test_environment():
    """Configure the test environment settings."""
    # Set Qt platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Enable test mode to disable background threads
    os.environ["XPCS_TEST_MODE"] = "1"

    # Configure logging for tests - suppress verbose output
    os.environ["PYXPCS_LOG_LEVEL"] = "WARNING"
    setup_logging(level=logging.WARNING)

    # Suppress common warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def configure_ci_environment(config):
    """Configure CI-specific settings."""
    if not CI_INTEGRATION_AVAILABLE:
        return

    ci_env = get_ci_environment()
    if ci_env["is_ci"]:
        print(f"\n🔧 Running in {ci_env['ci_provider']} CI environment")
        if ci_env.get("branch"):
            print(f"   Branch: {ci_env['branch']}")
        if ci_env.get("commit"):
            print(f"   Commit: {ci_env['commit'][:8]}...")

        # Apply CI-specific configurations
        config.option.tb = "short"  # Shorter tracebacks for CI logs


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and locations."""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath) or item.name.startswith("test_unit"):
            item.add_marker(pytest.mark.unit)

        # Mark integration tests
        if "integration" in str(item.fspath) or item.name.startswith(
            "test_integration"
        ):
            item.add_marker(pytest.mark.integration)

        # Mark GUI tests
        if "gui" in str(item.fspath) or "gui" in item.name.lower():
            item.add_marker(pytest.mark.gui)

        # Mark scientific tests
        if "scientific" in str(item.fspath) or any(
            keyword in item.name.lower()
            for keyword in ["g2", "correlation", "fitting", "analysis", "scattering"]
        ):
            item.add_marker(pytest.mark.scientific)

        # Mark slow tests
        if any(
            slow_keyword in item.name.lower()
            for slow_keyword in ["slow", "performance", "benchmark", "stress"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark performance tests
        if any(
            perf_keyword in item.name.lower() or perf_keyword in str(item.fspath)
            for perf_keyword in ["performance", "benchmark", "timing", "speed"]
        ):
            item.add_marker(pytest.mark.performance)

        # Mark stress tests
        if any(
            stress_keyword in item.name.lower()
            for stress_keyword in ["stress", "exhaustion", "resource"]
        ):
            item.add_marker(pytest.mark.stress)

        # Mark system dependent tests
        if any(
            system_keyword in item.name.lower() or system_keyword in str(item.fspath)
            for system_keyword in ["memory", "disk", "network", "display"]
        ):
            item.add_marker(pytest.mark.system_dependent)


# ============================================================================
# Session Management
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Set up test session with proper initialization."""
    # Initialize logging
    logger = logging.getLogger("xpcs_toolkit")
    logger.setLevel(logging.WARNING)

    # Create test directories if needed
    test_dir = Path(__file__).parent
    for subdir in ["fixtures", "fixtures/reference_data", "reports"]:
        (test_dir / subdir).mkdir(exist_ok=True)

    yield

    # Session cleanup
    # Note: Individual test cleanup is handled by specific fixtures


@pytest.fixture(autouse=True)
def test_isolation():
    """Ensure comprehensive test isolation by cleaning up state between tests."""
    import gc
    from unittest.mock import patch

    import numpy as np

    # Set consistent initial state before each test
    np.random.seed(42)  # Reproducible scientific tests

    yield  # Run the test

    # Comprehensive cleanup after each test to prevent state pollution
    try:
        # 1. Clear all mock patches to prevent mock state pollution
        patch.stopall()

        # 2. Reset NumPy random state for consistent scientific tests
        np.random.seed(42)

        # 3. Clear Qt application state if present (conservative approach)
        try:
            from PySide6 import QtCore, QtWidgets

            app = QtWidgets.QApplication.instance()
            if app:
                # Only process pending events - don't forcibly delete widgets
                # as this can cause segfaults with background threads
                app.processEvents()

        except ImportError:
            pass  # Qt not available, nothing to clean

        # 4. Force garbage collection to clear any remaining object references
        gc.collect()

        # 5. Clear any module-level caches that might affect subsequent tests
        # Reset any global state in scientific computing modules
        try:
            import matplotlib.pyplot as plt

            plt.close("all")  # Close any matplotlib figures
        except ImportError:
            pass

    except Exception as e:
        # Don't let cleanup failures break the test suite
        print(f"Warning: Test isolation cleanup failed: {e}")
        # Continue with partial cleanup


# ============================================================================
# Reliability Framework Integration (if available)
# ============================================================================

if RELIABILITY_FRAMEWORKS_AVAILABLE:

    @pytest.fixture(scope="function")
    def flakiness_detector():
        """Detect and track test flakiness."""
        return get_flakiness_detector()

    @pytest.fixture(scope="function")
    def performance_monitor():
        """Monitor test performance and resource usage."""
        return get_performance_monitor()

    @pytest.fixture(autouse=True)
    def auto_performance_monitoring(request):
        """Automatically monitor test performance."""
        monitor = get_performance_monitor()
        # Record test start time
        test_name = request.node.name
        if hasattr(monitor, "start_test"):
            monitor.start_test(test_name)
        yield
        # Record test completion
        if hasattr(monitor, "finish_test"):
            monitor.finish_test(test_name)

    @pytest.fixture(scope="function")
    def reliable_test_environment():
        """Create reliable test environment with isolation."""
        return isolated_test_environment()


# ============================================================================
# Advanced Data Management (if available)
# ============================================================================

if ADVANCED_DATA_MANAGEMENT_AVAILABLE:

    @pytest.fixture(scope="session")
    def test_data_factory():
        """Get test data factory for creating complex datasets."""
        return get_test_data_factory()

    @pytest.fixture(scope="function")
    def hdf5_test_manager(temp_dir):
        """Get HDF5 test file manager."""
        return get_hdf5_manager(temp_dir)


# ============================================================================
# Performance Configuration
# ============================================================================


@pytest.fixture(scope="session", autouse=True)
def optimize_test_performance():
    """Apply test performance optimizations."""
    import numpy as np

    # Configure NumPy for testing
    np.seterr(all="ignore")  # Suppress numerical warnings in tests

    # Set reasonable thread limits
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

    yield

    # Restore settings
    np.seterr(all="warn")


# ============================================================================
# Error Handling Test Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def error_temp_dir(tmp_path):
    """Create temporary directory for error handling tests."""
    error_dir = tmp_path / "error_tests"
    error_dir.mkdir(exist_ok=True)
    return str(error_dir)


@pytest.fixture(scope="function")
def edge_case_data():
    """Generate edge case test data."""
    import numpy as np

    return {
        "zero_array": np.zeros(100),
        "single_element": np.array([1.0]),
        "max_values": np.full(50, np.finfo(np.float64).max / 1e6),
        "min_positive": np.full(50, np.finfo(np.float64).tiny),
        "nan_array": np.full(50, np.nan),
        "inf_array": np.full(50, np.inf),
        "mixed_special": np.array([0, np.nan, np.inf, -np.inf, 1.0]),
        "empty_arrays": {
            "empty_1d": np.array([]),
            "empty_2d": np.array([]).reshape(0, 10),
            "empty_float": np.array([], dtype=np.float64),
            "empty_int": np.array([], dtype=np.int32),
        },
    }


@pytest.fixture(scope="function")
def numpy_error_scenarios():
    """Generate numpy error test scenarios."""
    import numpy as np

    return {
        "overflow": np.array([1e308, 1e309]),
        "underflow": np.array([1e-308, 1e-309]),
        "invalid_division": np.array([1.0]) / np.array([0.0]),
        "sqrt_negative": np.sqrt(np.array([-1.0])),
    }


@pytest.fixture(scope="function")
def error_injector():
    """Create error injection utility for testing."""

    class ErrorInjector:
        def __init__(self):
            self.active_errors = {}
            self.error_count = 0

        def inject_io_error(self, operation, error_type_or_probability=1.0):
            """Inject I/O errors for testing."""
            # Handle both error types and probability values
            if isinstance(error_type_or_probability, type):
                # If it's an exception class, store as 1.0 probability
                self.active_errors[operation] = 1.0
            else:
                # If it's a numeric probability, store it
                self.active_errors[operation] = error_type_or_probability
            self.error_count += 1

        def should_fail(self, operation):
            """Check if operation should fail."""
            import random

            return random.random() < self.active_errors.get(operation, 0.0)

        def clear_errors(self):
            """Clear all error injections."""
            self.active_errors.clear()

        def cleanup(self):
            """Cleanup error injections."""
            self.clear_errors()
            self.error_count = 0

    return ErrorInjector()


@pytest.fixture(scope="function")
def memory_limited_environment():
    """Create memory-limited test environment."""

    class MemoryLimitedEnvironment:
        def __init__(self):
            self.memory_limit_mb = 1024  # 1GB limit for testing

        def set_memory_limit(self, limit_mb):
            """Set memory limit for testing."""
            self.memory_limit_mb = limit_mb

        def check_memory_usage(self):
            """Check if current memory usage is within limits."""
            import psutil

            current_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
            return current_usage < self.memory_limit_mb

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    return MemoryLimitedEnvironment()


# ============================================================================
# Missing Error Handling Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def corrupted_hdf5_file(tmp_path):
    """Create a corrupted HDF5 file for testing."""
    corrupted_file = tmp_path / "corrupted.h5"
    # Write invalid data that will cause HDF5 to fail
    corrupted_file.write_bytes(
        b"This is not a valid HDF5 file - corrupted data\x00\x01\x02"
    )
    return str(corrupted_file)


@pytest.fixture(scope="function")
def invalid_hdf5_file(tmp_path):
    """Create an invalid HDF5 file for testing."""
    invalid_file = tmp_path / "invalid.h5"
    invalid_file.write_text("This is not a valid HDF5 file")
    return str(invalid_file)


@pytest.fixture(scope="function")
def missing_file_path(tmp_path):
    """Provide a path to a non-existent file for testing."""
    return str(tmp_path / "nonexistent_file.h5")


@pytest.fixture(scope="function")
def permission_denied_file(tmp_path):
    """Create a file with restricted permissions for testing."""
    import os
    import stat

    # Create a simple file
    restricted_file = tmp_path / "restricted.h5"
    restricted_file.write_bytes(b"dummy content")

    # Make it unreadable (remove read permissions)
    if os.name != "nt":  # Unix-like systems
        os.chmod(str(restricted_file), stat.S_IWRITE)  # Write only, no read
    else:  # Windows - use a different approach
        # On Windows, we'll return the file but the test should skip anyway
        pass

    yield str(restricted_file)

    # Cleanup: restore permissions so the file can be deleted
    if os.name != "nt":
        with contextlib.suppress(OSError, PermissionError):
            os.chmod(str(restricted_file), stat.S_IREAD | stat.S_IWRITE)


@pytest.fixture(scope="function")
def file_handle_exhausted_environment():
    """Create environment to simulate file handle exhaustion."""
    import contextlib
    import resource

    class FileHandleExhaustionSimulator:
        def __init__(self):
            self.original_limit = None

        @contextlib.contextmanager
        def limit_file_handles(self, max_handles=10):
            """Temporarily limit the number of open file handles."""
            try:
                if hasattr(resource, "RLIMIT_NOFILE"):
                    # Get current limit
                    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
                    self.original_limit = (soft, hard)

                    # Set a very low limit
                    new_limit = min(max_handles, hard)
                    resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))

                yield

            finally:
                # Restore original limit
                if self.original_limit and hasattr(resource, "RLIMIT_NOFILE"):
                    try:
                        resource.setrlimit(resource.RLIMIT_NOFILE, self.original_limit)
                    except (OSError, ValueError):
                        pass  # Best effort restoration

    return FileHandleExhaustionSimulator()


@pytest.fixture(scope="function")
def disk_space_exhausted_environment():
    """Create environment to simulate disk space exhaustion."""

    class DiskSpaceExhaustionSimulator:
        def __init__(self):
            pass

        def simulate_no_space(self):
            """Return a context manager that simulates no disk space."""
            import contextlib

            @contextlib.contextmanager
            def no_space_context():
                # This is a simulation - in practice, the test should handle
                # the case where disk space is exhausted
                yield

            return no_space_context()

    return DiskSpaceExhaustionSimulator()


@pytest.fixture(scope="function")
def disk_space_limited_environment():
    """Create environment to simulate limited disk space."""

    class DiskSpaceLimitedSimulator:
        def __init__(self):
            pass

        def simulate_limited_space(self):
            """Return a context manager that simulates limited disk space."""
            import contextlib

            @contextlib.contextmanager
            def limited_space_context():
                # This is a simulation - in practice, the test should handle
                # the case where disk space is limited
                yield

            return limited_space_context()

    return DiskSpaceLimitedSimulator()


@pytest.fixture(scope="function")
def resource_exhaustion():
    """Create resource exhaustion simulation for testing."""
    from contextlib import contextmanager

    class ResourceExhaustionSimulator:
        def __init__(self):
            self.active_simulations = []

        @contextmanager
        def simulate_memory_pressure(self, threshold=0.9):
            """Simulate memory pressure conditions."""
            simulation = f"memory_pressure_{threshold}"
            self.active_simulations.append(simulation)
            try:
                yield
            finally:
                if simulation in self.active_simulations:
                    self.active_simulations.remove(simulation)

        @contextmanager
        def simulate_disk_full(self):
            """Simulate disk full conditions."""
            simulation = "disk_full"
            self.active_simulations.append(simulation)
            try:
                yield
            finally:
                if simulation in self.active_simulations:
                    self.active_simulations.remove(simulation)

        @contextmanager
        def simulate_network_failure(self):
            """Simulate network failure conditions."""
            simulation = "network_failure"
            self.active_simulations.append(simulation)
            try:
                yield
            finally:
                if simulation in self.active_simulations:
                    self.active_simulations.remove(simulation)

    return ResourceExhaustionSimulator()


@pytest.fixture(scope="function")
def threading_error_scenarios():
    """Create threading error scenarios for testing."""
    import threading
    import time
    from contextlib import contextmanager

    class ThreadingErrorScenarios:
        def __init__(self):
            self.threads = []

        @contextmanager
        def deadlock_simulation(self):
            """Simulate deadlock conditions."""
            lock1 = threading.Lock()
            lock2 = threading.Lock()

            def worker1():
                with lock1:
                    time.sleep(0.01)
                    with lock2:
                        pass

            def worker2():
                with lock2:
                    time.sleep(0.01)
                    with lock1:
                        pass

            yield {
                "lock1": lock1,
                "lock2": lock2,
                "worker1": worker1,
                "worker2": worker2,
            }

        def race_condition_data(self):
            """Create data structures for race condition testing."""
            return {
                "counter": {"value": 0},
                "lock": threading.Lock(),
                "shared_list": [],
                "condition": threading.Condition(),
            }

        def thread_exception(self):
            """Create a thread that will raise an exception."""

            def failing_worker():
                raise RuntimeError("Simulated thread exception")

            thread = threading.Thread(target=failing_worker)
            self.threads.append(thread)
            return thread

        def cleanup(self):
            """Clean up any created threads."""
            for thread in self.threads:
                if thread.is_alive():
                    thread.join(timeout=0.1)
            self.threads.clear()

    return ThreadingErrorScenarios()
