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

from xpcs_toolkit.utils.logging_config import setup_logging

# Import focused fixture modules
from tests.fixtures.core_fixtures import *
from tests.fixtures.scientific_fixtures import *
from tests.fixtures.qt_fixtures import *

# Import optional framework utilities
try:
    from tests.utils.isolation import isolated_test_environment, get_performance_monitor, monitor_performance
    from tests.utils.reliability import get_flakiness_detector, reliable_test, validate_test_environment
    RELIABILITY_FRAMEWORKS_AVAILABLE = True
except ImportError:
    RELIABILITY_FRAMEWORKS_AVAILABLE = False

try:
    from tests.utils.data_management import (
        TestDataSpec, get_test_data_factory, get_hdf5_manager,
        temporary_xpcs_file, create_minimal_test_data, create_performance_test_data,
        create_realistic_xpcs_dataset
    )
    ADVANCED_DATA_MANAGEMENT_AVAILABLE = True
except ImportError:
    ADVANCED_DATA_MANAGEMENT_AVAILABLE = False

try:
    from tests.utils.ci_integration import (
        get_ci_environment, generate_ci_reports, collect_test_artifacts,
        TestSuite, TestResult, set_github_output, github_step_summary
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
    config.addinivalue_line("markers", "scientific: Tests that verify scientific accuracy")

    # Specialized markers
    config.addinivalue_line("markers", "flaky: Tests that are known to be flaky")
    config.addinivalue_line("markers", "stress: Stress tests that push system limits")
    config.addinivalue_line("markers", "system_dependent: Tests that depend on system resources")
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
    if ci_env['is_ci']:
        print(f"\n🔧 Running in {ci_env['ci_provider']} CI environment")
        if ci_env.get('branch'):
            print(f"   Branch: {ci_env['branch']}")
        if ci_env.get('commit'):
            print(f"   Commit: {ci_env['commit'][:8]}...")

        # Apply CI-specific configurations
        config.option.tb = 'short'  # Shorter tracebacks for CI logs


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their names and locations."""
    for item in items:
        # Mark unit tests
        if "unit" in str(item.fspath) or item.name.startswith("test_unit"):
            item.add_marker(pytest.mark.unit)

        # Mark integration tests
        if "integration" in str(item.fspath) or item.name.startswith("test_integration"):
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


# ============================================================================
# Reliability Framework Integration (if available)
# ============================================================================

if RELIABILITY_FRAMEWORKS_AVAILABLE:
    from tests.utils.isolation import isolation_manager

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
        if hasattr(monitor, 'start_test'):
            monitor.start_test(test_name)
        yield
        # Record test completion
        if hasattr(monitor, 'finish_test'):
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
    np.seterr(all='ignore')  # Suppress numerical warnings in tests

    # Set reasonable thread limits
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")

    yield

    # Restore settings
    np.seterr(all='warn')

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
        'zero_array': np.zeros(100),
        'single_element': np.array([1.0]),
        'max_values': np.full(50, np.finfo(np.float64).max / 1e6),
        'min_positive': np.full(50, np.finfo(np.float64).tiny),
        'nan_array': np.full(50, np.nan),
        'inf_array': np.full(50, np.inf),
        'mixed_special': np.array([0, np.nan, np.inf, -np.inf, 1.0]),
        'empty_arrays': {
            'empty_1d': np.array([]),
            'empty_2d': np.array([]).reshape(0, 10),
            'empty_float': np.array([], dtype=np.float64),
            'empty_int': np.array([], dtype=np.int32)
        }
    }


@pytest.fixture(scope="function")
def numpy_error_scenarios():
    """Generate numpy error test scenarios."""
    import numpy as np
    return {
        'overflow': np.array([1e308, 1e309]),
        'underflow': np.array([1e-308, 1e-309]),
        'invalid_division': np.array([1.0]) / np.array([0.0]),
        'sqrt_negative': np.sqrt(np.array([-1.0])),
    }


@pytest.fixture(scope="function")  
def error_injector():
    """Create error injection utility for testing."""
    class ErrorInjector:
        def __init__(self):
            self.active_errors = {}
            
        def inject_io_error(self, operation, probability=1.0):
            """Inject I/O errors for testing."""
            self.active_errors[operation] = probability
            
        def should_fail(self, operation):
            """Check if operation should fail."""
            import random
            return random.random() < self.active_errors.get(operation, 0.0)
            
        def clear_errors(self):
            """Clear all error injections."""
            self.active_errors.clear()
            
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
