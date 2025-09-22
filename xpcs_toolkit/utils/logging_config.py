"""
Centralized logging configuration for XPCS Toolkit.

This module provides a robust, configurable logging system with support for:
- Environment variable-based configuration
- Multiple output handlers (console + file with rotation)
- Structured logging with consistent formatting
- JSON format option for structured logging
- Thread-safe implementation for GUI applications
- Integration with existing LoggerWriter helper

Environment Variables:
    PYXPCS_LOG_LEVEL: DEBUG/INFO/WARNING/ERROR/CRITICAL (default: INFO)
    PYXPCS_LOG_FILE: Custom log file path
    PYXPCS_LOG_DIR: Log directory (default: ~/.xpcs_toolkit/logs)
    PYXPCS_LOG_FORMAT: TEXT/JSON for structured logging (default: TEXT)
    PYXPCS_LOG_MAX_SIZE: Maximum log file size in MB (default: 10)
    PYXPCS_LOG_BACKUP_COUNT: Number of backup log files (default: 5)
    PYXPCS_SUPPRESS_QT_WARNINGS: 1 to suppress Qt logging (default: 0)

Usage:
    from xpcs_toolkit.utils.logging_config import get_logger
    logger = get_logger(__name__)
    logger.info("Application started")
"""

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Any

from .log_formatters import (
    ColoredConsoleFormatter,
    JSONFormatter,
    StructuredFileFormatter,
)


class LoggingConfig:
    """Singleton logging configuration manager."""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_configuration()
            self._configure_logging()
            LoggingConfig._initialized = True

    def setup_logging(self):
        """Public method to setup or reset logging configuration."""
        self._setup_configuration()
        self._configure_logging()

    def _setup_configuration(self):
        """Setup logging configuration from environment variables."""
        # Log level configuration
        level_str = os.environ.get("PYXPCS_LOG_LEVEL", "INFO").upper()
        self.log_level = getattr(logging, level_str, logging.INFO)

        # Log directory configuration
        default_log_dir = os.path.join(os.path.expanduser("~"), ".xpcs_toolkit", "logs")
        self.log_dir = Path(os.environ.get("PYXPCS_LOG_DIR", default_log_dir))

        # Create log directory if it doesn't exist
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Fall back to default directory if the requested directory cannot be created
            self.log_dir = Path(default_log_dir)
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError):
                # Final fallback to temp directory
                import tempfile

                self.log_dir = Path(tempfile.gettempdir()) / "xpcs_toolkit_logs"
                self.log_dir.mkdir(parents=True, exist_ok=True)

        # Log file configuration
        default_log_file = self.log_dir / "xpcs_toolkit.log"
        custom_log_file = os.environ.get("PYXPCS_LOG_FILE")
        if custom_log_file:
            self.log_file = Path(custom_log_file)
            # Ensure parent directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.log_file = default_log_file

        # Log format configuration
        format_type = os.environ.get("PYXPCS_LOG_FORMAT", "TEXT").upper()
        self.use_json_format = format_type == "JSON"

        # Log rotation configuration
        try:
            max_size_mb = float(os.environ.get("PYXPCS_LOG_MAX_SIZE", "10"))
            self.max_file_size = int(max_size_mb * 1024 * 1024)  # MB to bytes
        except ValueError:
            self.max_file_size = 10 * 1024 * 1024  # Default 10MB

        try:
            self.backup_count = int(os.environ.get("PYXPCS_LOG_BACKUP_COUNT", "5"))
        except ValueError:
            self.backup_count = 5  # Default backup count

        # Qt warnings suppression
        self.suppress_qt_warnings = (
            os.environ.get("PYXPCS_SUPPRESS_QT_WARNINGS", "0") == "1"
        )

        # Application info
        self.app_name = "XPCS Toolkit"
        self.app_version = self._get_app_version()

    def _get_app_version(self) -> str:
        """Get application version using importlib.metadata to avoid circular imports."""
        try:
            from importlib.metadata import version

            return version("xpcs-toolkit")
        except Exception:
            return "unknown"

    def _configure_logging(self):
        """Configure the logging system."""
        # Get root logger and clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(self.log_level)

        # Setup console handler
        self._setup_console_handler()

        # Setup file handler with rotation
        self._setup_file_handler()

        # Configure Qt logging if needed
        if self.suppress_qt_warnings:
            self._suppress_qt_logging()

        # Log configuration summary
        self._log_configuration_summary()

    def _setup_console_handler(self):
        """Setup colored console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        # Use colored formatter for console
        console_formatter = ColoredConsoleFormatter()
        console_handler.setFormatter(console_formatter)

        # Add to root logger
        logging.getLogger().addHandler(console_handler)

    def _setup_file_handler(self):
        """Setup rotating file handler."""
        try:
            # Use rotating file handler for automatic log rotation
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(self.log_file),
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding="utf-8",
            )
            file_handler.setLevel(self.log_level)

            # Choose formatter based on configuration
            if self.use_json_format:
                file_formatter = JSONFormatter(
                    app_name=self.app_name, app_version=self.app_version
                )
            else:
                file_formatter = StructuredFileFormatter()

            file_handler.setFormatter(file_formatter)

            # Add to root logger
            logging.getLogger().addHandler(file_handler)

        except (OSError, PermissionError) as e:
            # Fallback: log to console if file logging fails
            fallback_logger = logging.getLogger(__name__)
            fallback_logger.warning(
                f"Failed to setup file logging: {e}. Using console only."
            )

    def _suppress_qt_logging(self):
        """Suppress Qt-related logging messages."""
        # Suppress specific Qt loggers
        qt_loggers = [
            "Qt",
            "qt.qpa.plugin",
            "qt.qpa.xcb",
            "qt.qpa.fonts",
            "qt.svg",
            "qt.network.ssl",
        ]

        for logger_name in qt_loggers:
            qt_logger = logging.getLogger(logger_name)
            qt_logger.setLevel(logging.CRITICAL)

        # Set Qt logging rules environment variable
        os.environ["QT_LOGGING_RULES"] = (
            "*.debug=false;qt.qpa.plugin=false;qt.svg=false"
        )

    def _log_configuration_summary(self):
        """Log the current logging configuration."""
        config_logger = logging.getLogger(__name__)
        config_logger.info(f"{self.app_name} v{self.app_version} logging initialized")
        config_logger.debug(f"Log level: {logging.getLevelName(self.log_level)}")
        config_logger.debug(f"Log file: {self.log_file}")
        config_logger.debug(f"Log directory: {self.log_dir}")
        config_logger.debug(f"Format: {'JSON' if self.use_json_format else 'TEXT'}")
        config_logger.debug(
            f"Max file size: {self.max_file_size / (1024 * 1024):.1f} MB"
        )
        config_logger.debug(f"Backup count: {self.backup_count}")

    def get_logger_info(self) -> dict[str, Any]:
        """Get current logging configuration info."""
        return {
            "log_level": logging.getLevelName(self.log_level),
            "log_file": str(self.log_file),
            "log_dir": str(self.log_dir),
            "format": "JSON" if self.use_json_format else "TEXT",
            "max_file_size_mb": self.max_file_size / (1024 * 1024),
            "backup_count": self.backup_count,
            "suppress_qt_warnings": self.suppress_qt_warnings,
            "app_name": self.app_name,
            "app_version": self.app_version,
        }

    def update_log_level(self, level: str | int):
        """Update the log level for all handlers."""
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)

        # Update all handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)

        self.log_level = level
        config_logger = logging.getLogger(__name__)
        config_logger.info(f"Log level updated to {logging.getLevelName(level)}")


# Global configuration instance
_config = None
_config_lock = threading.Lock()


def initialize_logging() -> LoggingConfig:
    """Initialize the logging configuration (called automatically on first use)."""
    global _config
    if _config is None:
        with _config_lock:
            if _config is None:
                _config = LoggingConfig()
    return _config


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get a configured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance

    Usage:
        logger = get_logger(__name__)
        logger.info("This is a log message")
    """
    # Initialize configuration on first use
    initialize_logging()

    # Return logger with specified name
    if name is None:
        name = "xpcs_toolkit"

    return logging.getLogger(name)


def get_logging_config() -> LoggingConfig:
    """Get the current logging configuration instance."""
    return initialize_logging()


# Note: reset_logging_config and get_log_directory are defined later with enhanced functionality


def set_log_level(level: str | int):
    """
    Set the logging level globally.

    Args:
        level: Log level (string like 'DEBUG', 'INFO' or integer)
    """
    config = initialize_logging()
    config.update_log_level(level)


def get_log_file_path() -> Path:
    """Get the current log file path."""
    config = initialize_logging()
    return config.log_file


def get_log_directory() -> Path:
    """Get the current log directory path."""
    config = initialize_logging()
    return config.log_dir


def reset_logging_config():
    """Reset the logging configuration singleton for testing purposes."""
    global _config
    with _config_lock:
        if _config is not None:
            # Clean up existing handlers
            root_logger = logging.getLogger()
            handlers = root_logger.handlers[:]
            for handler in handlers:
                root_logger.removeHandler(handler)
                handler.close()

        _config = None
        LoggingConfig._instance = None
        LoggingConfig._initialized = False


def log_system_info():
    """Log useful system information for debugging."""
    logger = get_logger(__name__)

    import platform
    import sys

    logger.info("=== System Information ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")

    # Log Qt information if available
    try:
        from PySide6 import __version__ as pyside_version

        logger.info(f"PySide6: {pyside_version}")
    except ImportError:
        pass

    # Log numpy/scipy info if available
    try:
        import numpy as np

        logger.info(f"NumPy: {np.__version__}")
    except ImportError:
        pass

    try:
        import scipy

        logger.info(f"SciPy: {scipy.__version__}")
    except ImportError:
        pass

    logger.info("=== End System Information ===")


def setup_exception_logging():
    """Setup logging for uncaught exceptions."""

    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Don't log keyboard interrupts
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger = get_logger("uncaught_exception")
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = log_exception


# Compatibility function for existing code
def setup_logging(level: str | int | None = None) -> logging.Logger:
    """
    Setup logging with optional level override.

    This function provides compatibility with existing code patterns.

    Args:
        level: Optional log level override

    Returns:
        Root logger instance
    """
    config = initialize_logging()

    if level is not None:
        config.update_log_level(level)

    return logging.getLogger()
