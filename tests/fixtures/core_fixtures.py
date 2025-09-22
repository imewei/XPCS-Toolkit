"""Core test fixtures for XPCS Toolkit tests.

This module provides basic fixtures for temporary directories, files,
logging configuration, and test utilities.
"""

import logging
import os
import shutil
import tempfile
import warnings
from collections.abc import Generator
from pathlib import Path

import pytest

from xpcs_toolkit.utils.logging_config import get_logger

# ============================================================================
# Temporary Directory and File Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="xpcs_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_file(temp_dir) -> str:
    """Create temporary file path."""
    return os.path.join(temp_dir, "test_file.tmp")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    fixtures_path = Path(__file__).parent
    fixtures_path.mkdir(exist_ok=True)
    return fixtures_path


# ============================================================================
# Logging and Test Configuration Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def test_logger() -> logging.Logger:
    """Configure logger for test sessions with appropriate level."""
    # Suppress INFO level logs during tests to reduce noise
    logger = get_logger(__name__)
    logger.setLevel(logging.WARNING)
    return logger


@pytest.fixture(scope="function")
def capture_logs() -> Generator[list, None, None]:
    """Capture log messages during test execution."""
    log_messages = []

    class TestLogHandler(logging.Handler):
        def emit(self, record):
            log_messages.append(record)

    handler = TestLogHandler()
    logger = logging.getLogger("xpcs_toolkit")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    yield log_messages

    logger.removeHandler(handler)


@pytest.fixture(scope="function")
def performance_timer():
    """Timer for performance testing."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.perf_counter()
            return end - self.start_time

    return Timer()


# ============================================================================
# Warning Suppression Configuration
# ============================================================================


def suppress_common_warnings():
    """Suppress common scientific computing warnings during tests."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="h5py")


# Auto-suppress warnings when this module is imported
suppress_common_warnings()
