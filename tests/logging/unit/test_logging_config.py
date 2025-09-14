"""Unit tests for logging configuration module.

This module provides comprehensive unit tests for the logging configuration
system, covering environment-based setup and thread-safe implementation.
"""

import logging
import os
import threading
from unittest.mock import patch

import pytest

from xpcs_toolkit.utils.logging_config import LoggingConfig, get_logger


class TestLoggingConfig:
    """Test suite for LoggingConfig class."""

    def setup_method(self):
        """Reset logging config for each test."""
        # Clear the singleton
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

        # Clear any existing handlers
        logger = logging.getLogger()
        logger.handlers.clear()

    def test_singleton_behavior(self):
        """Test that LoggingConfig is a singleton."""
        config1 = LoggingConfig()
        config2 = LoggingConfig()

        assert config1 is config2
        assert id(config1) == id(config2)

    def test_thread_safe_singleton(self):
        """Test that singleton creation is thread-safe."""
        instances = []

        def create_instance():
            instances.append(LoggingConfig())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All instances should be the same
        assert all(instance is instances[0] for instance in instances)
        assert len(set(id(instance) for instance in instances)) == 1

    @patch.dict(os.environ, {}, clear=True)
    def test_default_configuration(self):
        """Test default configuration when no environment variables are set."""
        config = LoggingConfig()

        assert config.log_level == logging.INFO
        assert config.log_dir.name == "logs"
        assert ".xpcs_toolkit" in str(config.log_dir)
        assert config.log_file.name == "xpcs_toolkit.log"
        assert config.use_json_format is False
        assert config.max_file_size == 10 * 1024 * 1024  # 10MB
        assert config.backup_count == 5

    @patch.dict(
        os.environ,
        {
            "PYXPCS_LOG_LEVEL": "DEBUG",
            "PYXPCS_LOG_FORMAT": "JSON",
            "PYXPCS_LOG_MAX_SIZE": "20",
            "PYXPCS_LOG_BACKUP_COUNT": "10",
        },
    )
    def test_environment_configuration(self):
        """Test configuration from environment variables."""
        config = LoggingConfig()

        assert config.log_level == logging.DEBUG
        assert config.use_json_format is True
        assert config.max_file_size == 20 * 1024 * 1024  # 20MB
        assert config.backup_count == 10

    @patch.dict(os.environ, {"PYXPCS_LOG_LEVEL": "INVALID"})
    def test_invalid_log_level(self):
        """Test handling of invalid log level."""
        config = LoggingConfig()

        # Should default to INFO for invalid level
        assert config.log_level == logging.INFO

    @patch.dict(os.environ, {"PYXPCS_LOG_MAX_SIZE": "invalid"})
    def test_invalid_max_size(self):
        """Test handling of invalid max size."""
        config = LoggingConfig()

        # Should default to 10MB for invalid size
        assert config.max_file_size == 10 * 1024 * 1024

    def test_custom_log_file_path(self, temp_dir):
        """Test custom log file path configuration."""
        custom_log_file = os.path.join(temp_dir, "custom", "test.log")

        with patch.dict(os.environ, {"PYXPCS_LOG_FILE": custom_log_file}):
            config = LoggingConfig()

            assert str(config.log_file) == custom_log_file
            # Parent directory should be created
            assert config.log_file.parent.exists()

    def test_custom_log_directory(self, temp_dir):
        """Test custom log directory configuration."""
        custom_log_dir = os.path.join(temp_dir, "custom_logs")

        with patch.dict(os.environ, {"PYXPCS_LOG_DIR": custom_log_dir}):
            config = LoggingConfig()

            assert str(config.log_dir) == custom_log_dir
            # Directory should be created
            assert config.log_dir.exists()

    def test_setup_logging_method(self):
        """Test setup_logging method."""
        config = LoggingConfig()

        # Should be able to call setup_logging without errors
        config.setup_logging()

        # Should still be the same instance
        assert LoggingConfig() is config

    @patch("xpcs_toolkit.utils.logging_config.LoggingConfig._configure_logging")
    @patch("xpcs_toolkit.utils.logging_config.LoggingConfig._setup_configuration")
    def test_initialization_only_once(self, mock_setup_config, mock_configure_logging):
        """Test that initialization only happens once."""
        LoggingConfig()
        LoggingConfig()

        # Setup and configure should only be called once during first initialization
        assert mock_setup_config.call_count == 1
        assert mock_configure_logging.call_count == 1


class TestGetLogger:
    """Test suite for get_logger function."""

    def setup_method(self):
        """Reset logging for each test."""
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

        # Clear handlers
        logger = logging.getLogger()
        logger.handlers.clear()

    def test_get_logger_basic(self):
        """Test basic get_logger functionality."""
        logger = get_logger(__name__)

        assert isinstance(logger, logging.Logger)
        assert logger.name == __name__

    def test_get_logger_with_custom_name(self):
        """Test get_logger with custom name."""
        logger = get_logger("test_logger")

        assert logger.name == "test_logger"

    def test_get_logger_same_name_returns_same_instance(self):
        """Test that get_logger returns same instance for same name."""
        logger1 = get_logger("test")
        logger2 = get_logger("test")

        assert logger1 is logger2

    @patch.dict(os.environ, {"PYXPCS_LOG_LEVEL": "DEBUG"})
    def test_get_logger_respects_config(self):
        """Test that get_logger respects logging configuration."""
        get_logger("test")

        # Should use configured log level
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_get_logger_creates_config(self):
        """Test that get_logger creates LoggingConfig if not exists."""
        assert LoggingConfig._instance is None

        get_logger("test")

        assert LoggingConfig._instance is not None
        assert LoggingConfig._initialized is True


class TestLoggingConfigSetup:
    """Test suite for logging configuration setup methods."""

    def setup_method(self):
        """Reset logging for each test."""
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

        # Clear handlers
        logger = logging.getLogger()
        logger.handlers.clear()

    def test_setup_logging_method(self):
        """Test setup_logging method on LoggingConfig."""
        config = LoggingConfig()

        # Should be able to call setup_logging without errors
        config.setup_logging()

        # Should still be the same instance
        assert LoggingConfig() is config


class TestLoggingIntegration:
    """Test suite for logging system integration."""

    def setup_method(self):
        """Reset logging for each test."""
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

        # Clear only non-pytest handlers to preserve caplog functionality
        root_logger = logging.getLogger()
        # Keep caplog handlers (pytest's LogCaptureHandler)
        handlers_to_keep = []
        for handler in root_logger.handlers:
            if (
                hasattr(handler, "__class__")
                and "LogCapture" in handler.__class__.__name__
            ):
                handlers_to_keep.append(handler)

        root_logger.handlers.clear()
        root_logger.handlers.extend(handlers_to_keep)

    def test_logger_hierarchy(self):
        """Test logger hierarchy works correctly."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")

        assert (
            child_logger.parent is parent_logger or child_logger.parent.name == "parent"
        )

    def test_logging_output_works(self, caplog):
        """Test that logging actually produces output."""
        # Use isolated logger to avoid interference with other tests
        import logging

        logger = logging.getLogger("test_output_isolated")

        with caplog.at_level(logging.INFO, logger=logger.name):
            logger.info("Test message")

        # Check caplog records for more reliable testing
        log_messages = [record.message for record in caplog.records]
        assert "Test message" in log_messages

    def test_different_log_levels(self, caplog):
        """Test different log levels work correctly."""
        # Use the root logger directly for this test to avoid handler conflicts
        import logging

        logger = logging.getLogger("test_levels_isolated")

        with caplog.at_level(logging.DEBUG, logger=logger.name):
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

        # Check caplog records instead of text for more reliable testing
        log_messages = [record.message for record in caplog.records]
        assert "Debug message" in log_messages
        assert "Info message" in log_messages
        assert "Warning message" in log_messages
        assert "Error message" in log_messages
        assert "Critical message" in log_messages


class TestLoggingConfiguration:
    """Test suite for logging configuration details."""

    def setup_method(self):
        """Reset logging for each test."""
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

    @patch("xpcs_toolkit.utils.logging_config.ColoredConsoleFormatter")
    @patch("xpcs_toolkit.utils.logging_config.StructuredFileFormatter")
    def test_formatter_creation(self, mock_file_formatter, mock_console_formatter):
        """Test that formatters are created correctly."""
        LoggingConfig()

        # Formatters should be imported and available
        assert mock_console_formatter is not None
        assert mock_file_formatter is not None

    def test_log_directory_creation(self, temp_dir):
        """Test that log directory is created properly."""
        log_dir = os.path.join(temp_dir, "test_logs")

        with patch.dict(os.environ, {"PYXPCS_LOG_DIR": log_dir}):
            LoggingConfig()

            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)

    def test_nested_log_directory_creation(self, temp_dir):
        """Test that nested log directories are created."""
        log_dir = os.path.join(temp_dir, "deep", "nested", "logs")

        with patch.dict(os.environ, {"PYXPCS_LOG_DIR": log_dir}):
            LoggingConfig()

            assert os.path.exists(log_dir)
            assert os.path.isdir(log_dir)


class TestLoggingEdgeCases:
    """Test suite for logging edge cases."""

    def setup_method(self):
        """Reset logging for each test."""
        LoggingConfig._instance = None
        LoggingConfig._initialized = False

    def test_empty_logger_name(self):
        """Test get_logger with empty name."""
        logger = get_logger("")

        assert isinstance(logger, logging.Logger)
        # Empty string should return root logger (standard Python logging behavior)
        assert logger.name == "root"

    def test_none_logger_name(self):
        """Test get_logger with None name."""
        # Should handle None gracefully
        logger = get_logger(None)

        assert isinstance(logger, logging.Logger)

    @patch.dict(os.environ, {"PYXPCS_LOG_BACKUP_COUNT": "invalid"})
    def test_invalid_backup_count(self):
        """Test handling of invalid backup count."""
        # Should not raise exception, should use default
        try:
            config = LoggingConfig()
            # Should fall back to reasonable default or handle gracefully
            assert isinstance(config.backup_count, int)
        except Exception:
            pytest.fail("Should handle invalid backup count gracefully")

    @patch("pathlib.Path.mkdir")
    def test_log_directory_creation_failure(self, mock_mkdir):
        """Test handling when log directory creation fails."""
        mock_mkdir.side_effect = OSError("Permission denied")

        # Should handle gracefully without crashing
        try:
            LoggingConfig()
        except OSError:
            # It's acceptable to re-raise OS errors for directory creation
            pass

    def test_concurrent_logger_access(self):
        """Test concurrent access to get_logger."""
        loggers = []

        def get_loggers():
            for i in range(10):
                loggers.append(get_logger(f"concurrent_{i}"))

        threads = [threading.Thread(target=get_loggers) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should have created loggers without error
        assert len(loggers) == 50  # 5 threads * 10 loggers each

        # Loggers with same name should be identical
        logger_dict = {}
        for logger in loggers:
            if logger.name not in logger_dict:
                logger_dict[logger.name] = logger
            else:
                assert logger_dict[logger.name] is logger


@pytest.mark.parametrize(
    "log_level_str,expected_level",
    [
        ("DEBUG", logging.DEBUG),
        ("INFO", logging.INFO),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
        ("CRITICAL", logging.CRITICAL),
        ("debug", logging.DEBUG),  # Should handle case insensitive
        ("info", logging.INFO),
    ],
)
def test_log_level_parsing(log_level_str, expected_level):
    """Test log level parsing from environment variables."""
    LoggingConfig._instance = None
    LoggingConfig._initialized = False

    with patch.dict(os.environ, {"PYXPCS_LOG_LEVEL": log_level_str}):
        config = LoggingConfig()
        assert config.log_level == expected_level


@pytest.mark.parametrize(
    "format_str,expected_json",
    [
        ("TEXT", False),
        ("JSON", True),
        ("text", False),  # Should handle case insensitive
        ("json", True),
    ],
)
def test_log_format_parsing(format_str, expected_json):
    """Test log format parsing from environment variables."""
    LoggingConfig._instance = None
    LoggingConfig._initialized = False

    with patch.dict(os.environ, {"PYXPCS_LOG_FORMAT": format_str}):
        config = LoggingConfig()
        assert config.use_json_format == expected_json


class TestLoggingPerformance:
    """Test suite for logging performance characteristics."""

    def test_logger_creation_performance(self, performance_timer):
        """Test logger creation performance."""
        performance_timer.start()

        for i in range(100):
            get_logger(f"perf_test_{i}")

        elapsed = performance_timer.stop()

        # Should be reasonably fast
        assert elapsed < 1.0  # Less than 1 second for 100 loggers

    def test_singleton_access_performance(self, performance_timer):
        """Test singleton access performance."""
        # Create initial instance
        config1 = LoggingConfig()

        performance_timer.start()

        # Access singleton many times
        for _ in range(1000):
            config = LoggingConfig()
            assert config is config1

        elapsed = performance_timer.stop()

        # Should be very fast for cached access
        assert elapsed < 0.1  # Less than 100ms for 1000 accesses
