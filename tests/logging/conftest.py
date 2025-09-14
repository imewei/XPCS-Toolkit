"""
Common fixtures for XPCS Toolkit logging tests.

This module provides shared fixtures and utilities for all logging-related tests,
reducing duplication and ensuring consistent test environments across different
test categories.
"""

import logging
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

# Add project root to path for testing
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from xpcs_toolkit.utils.logging_config import reset_logging_config  # noqa: E402


@pytest.fixture(scope="session")
def logging_temp_dir():
    """Create a temporary directory for logging tests."""
    with TemporaryDirectory(prefix="logging_tests_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def clean_logging_environment():
    """Clean logging environment before each test."""
    # Store original environment
    original_env = {}
    logging_vars = [
        "PYXPCS_LOG_LEVEL",
        "PYXPCS_LOG_FILE",
        "PYXPCS_LOG_DIR",
        "PYXPCS_LOG_FORMAT",
        "PYXPCS_LOG_MAX_SIZE",
        "PYXPCS_LOG_BACKUP_COUNT",
        "PYXPCS_LOG_DISABLE_FILE",
        "PYXPCS_LOG_DISABLE_CONSOLE",
        "PYXPCS_SUPPRESS_QT_WARNINGS",
    ]

    for var in logging_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]

    # Clear logging state
    reset_logging_config()

    # Clear any existing handlers
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        root_logger.removeHandler(handler)
        handler.close()

    yield

    # Restore environment
    for var in logging_vars:
        if var in os.environ:
            del os.environ[var]

    for var, value in original_env.items():
        os.environ[var] = value

    # Final cleanup
    reset_logging_config()


@pytest.fixture
def clean_logging_state(logging_temp_dir):
    """Ensure clean logging state for each test with temp directory."""
    # Clear any existing handlers
    root_logger = logging.getLogger()
    handlers = root_logger.handlers[:]
    for handler in handlers:
        root_logger.removeHandler(handler)
        handler.close()

    # Reset logging configuration
    reset_logging_config()

    # Set up clean environment with temp directory
    os.environ["PYXPCS_LOG_DIR"] = str(logging_temp_dir)

    yield logging_temp_dir

    # Cleanup
    reset_logging_config()


@pytest.fixture
def isolated_logger():
    """Create an isolated logger for testing."""
    logger_name = f"test.isolated.{os.getpid()}"
    logger = logging.getLogger(logger_name)

    # Ensure logger doesn't propagate to root
    logger.propagate = False

    yield logger

    # Cleanup
    logger.handlers.clear()
    logger.propagate = True


@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {}, clear=False) as mock_env:
        yield mock_env


@pytest.fixture
def memory_handler():
    """Create a memory handler for capturing log messages."""
    from logging.handlers import MemoryHandler

    # Create target handler that stores records
    target_records = []

    class RecordCollector(logging.Handler):
        def emit(self, record):
            target_records.append(record)

    collector = RecordCollector()
    memory_handler = MemoryHandler(capacity=1000, target=collector)

    yield memory_handler, target_records

    # Cleanup
    memory_handler.close()


@pytest.fixture
def temp_log_file(logging_temp_dir):
    """Create a temporary log file for testing."""
    log_file = logging_temp_dir / "test.log"
    yield log_file

    # Cleanup is handled by temp directory fixture


@pytest.fixture(autouse=True)
def reset_logging_after_test():
    """Automatically reset logging after each test."""
    yield
    reset_logging_config()


# Common test utilities
class LogCapture:
    """Utility class for capturing log messages during tests."""

    def __init__(self):
        self.records = []
        self.handler = logging.Handler()
        self.handler.emit = self._emit

    def _emit(self, record):
        self.records.append(record)

    def clear(self):
        self.records.clear()

    def get_messages(self, level=None):
        """Get log messages, optionally filtered by level."""
        if level is None:
            return [record.getMessage() for record in self.records]
        return [
            record.getMessage() for record in self.records if record.levelno == level
        ]


@pytest.fixture
def log_capture():
    """Provide a log capture utility."""
    return LogCapture()
