"""
Custom log formatters for XPCS Toolkit.

This module provides specialized formatters for different logging outputs:
- ColoredConsoleFormatter: Colored console output with clean formatting
- StructuredFileFormatter: Detailed file logging with context
- JSONFormatter: Structured JSON logging for log analysis

Features:
- Color-coded log levels for console output
- Thread-safe formatting
- Structured data support
- Performance-optimized formatters
- Consistent timestamp formatting
"""

import json
import logging
import os
import sys
import threading
import traceback
from datetime import datetime


class ColoredConsoleFormatter(logging.Formatter):
    """
    Colored console formatter with clean, readable output.

    Features:
    - Color-coded log levels
    - Clean module name formatting
    - Thread information for debugging
    - Exception formatting
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    def __init__(self, use_colors: bool | None = None):
        super().__init__()
        # Auto-detect color support
        if use_colors is None:
            use_colors = self._supports_color()
        self.use_colors = use_colors

    def _supports_color(self) -> bool:
        """Check if the terminal supports colors."""
        # Check if we're running in a terminal that supports colors
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            return False

        # Check common environment variables for color support
        term = os.environ.get("TERM", "").lower()
        if "color" in term or term in ("xterm", "xterm-256color", "screen"):
            return True

        # Check if running in common IDEs
        if any(ide in os.environ.get("_", "") for ide in ("pycharm", "vscode", "code")):
            return True

        # Default to no colors for safety
        return False

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors or color not in self.COLORS:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['RESET']}"

    def _format_module_name(self, name: str) -> str:
        """Format module name for clean display."""
        if name.startswith("xpcs_toolkit."):
            # Shorten xpcs_toolkit module names
            short_name = name.replace("xpcs_toolkit.", "")
            return short_name[:20]  # Limit length
        return name[:20]

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for console output."""
        # Create a copy to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Format timestamp
        timestamp = datetime.fromtimestamp(record_copy.created).strftime("%H:%M:%S.%f")[
            :-3
        ]

        # Format level with color
        level = record_copy.levelname
        colored_level = self._colorize(f"{level:8}", level)

        # Format module name
        module_name = self._format_module_name(record_copy.name)

        # Build the basic message
        basic_msg = (
            f"{timestamp} {colored_level} {module_name:20} | {record_copy.getMessage()}"
        )

        # Add exception information if present
        if record_copy.exc_info:
            exc_text = self.formatException(record_copy.exc_info)
            basic_msg = f"{basic_msg}\n{exc_text}"

        return basic_msg

    def formatException(self, ei) -> str:
        """Format exception with colors."""
        lines = traceback.format_exception(*ei)
        error_text = "".join(lines).rstrip()
        return self._colorize(error_text, "ERROR")


class StructuredFileFormatter(logging.Formatter):
    """
    Structured file formatter with detailed context information.

    Features:
    - Detailed timestamps
    - Full module paths
    - Thread and process information
    - Exception context
    - Performance timing
    """

    def __init__(self):
        super().__init__()
        self.start_time = datetime.now()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record for file output."""
        # Create a copy to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Format timestamp with microseconds
        timestamp = datetime.fromtimestamp(record_copy.created).isoformat(
            timespec="microseconds"
        )

        # Calculate elapsed time since formatter creation
        elapsed = datetime.fromtimestamp(record_copy.created) - self.start_time
        elapsed_str = f"{elapsed.total_seconds():.3f}s"

        # Get thread information
        thread_name = getattr(record_copy, "threadName", "MainThread")

        # Build structured message
        parts = [
            f"{timestamp}",
            f"[{elapsed_str:>8}]",
            f"[{record_copy.levelname:8}]",
            f"[{record_copy.name}]",
            f"[{thread_name}]",
            f"{record_copy.funcName}:{record_copy.lineno}",
            f"- {record_copy.getMessage()}",
        ]

        basic_msg = " ".join(parts)

        # Add exception information if present
        if record_copy.exc_info:
            exc_text = self.formatException(record_copy.exc_info)
            basic_msg = f"{basic_msg}\n{exc_text}"

        return basic_msg


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging and log analysis.

    Features:
    - Machine-readable JSON output
    - Structured exception handling
    - Application context
    - Performance metrics
    - Thread-safe JSON serialization
    """

    def __init__(self, app_name: str = "XPCS Toolkit", app_version: str = "unknown"):
        super().__init__()
        self.app_name = app_name
        self.app_version = app_version
        self.start_time = datetime.now()
        self._lock = threading.Lock()

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        with self._lock:
            return self._format_json(record)

    def _format_json(self, record: logging.LogRecord) -> str:
        """Internal JSON formatting method."""
        # Base log entry structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": getattr(record, "threadName", "MainThread"),
            "app_name": self.app_name,
            "app_version": self.app_version,
        }

        # Add elapsed time
        elapsed = datetime.fromtimestamp(record.created) - self.start_time
        log_entry["elapsed_seconds"] = round(elapsed.total_seconds(), 3)

        # Add process information
        log_entry["process_id"] = record.process

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info).split("\n"),
            }

        # Add extra fields from the log record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "funcName",
                "lineno",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ):
                # Only include JSON-serializable values
                if self._is_json_serializable(value):
                    extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        # Serialize to JSON
        try:
            return json.dumps(log_entry, default=str, separators=(",", ":"))
        except (TypeError, ValueError) as e:
            # Fallback for serialization errors
            fallback_entry = {
                "timestamp": log_entry["timestamp"],
                "level": "ERROR",
                "logger": "logging.formatter",
                "message": f"JSON serialization error: {e}. Original message: {record.getMessage()}",
                "app_name": self.app_name,
            }
            return json.dumps(fallback_entry, separators=(",", ":"))

    def _is_json_serializable(self, value) -> bool:
        """Check if a value can be JSON serialized."""
        try:
            json.dumps(value, default=str)
            return True
        except (TypeError, ValueError):
            return False


class PerformanceFormatter(StructuredFileFormatter):
    """
    Enhanced formatter that includes performance timing information.

    Useful for performance analysis and debugging slow operations.
    """

    def __init__(self):
        super().__init__()
        self._last_record_time = None

    def format(self, record: logging.LogRecord) -> str:
        """Format record with performance timing."""
        current_time = datetime.fromtimestamp(record.created)

        # Calculate time since last log message
        delta_str = ""
        if self._last_record_time:
            delta = current_time - self._last_record_time
            delta_ms = delta.total_seconds() * 1000
            delta_str = f"[+{delta_ms:6.1f}ms]"

        self._last_record_time = current_time

        # Get the base formatted message
        base_msg = super().format(record)

        # Insert delta timing after timestamp
        if delta_str:
            parts = base_msg.split("] ", 2)
            if len(parts) >= 2:
                return f"{parts[0]}] {delta_str} {parts[1]}"

        return base_msg


def create_formatter(formatter_type: str, **kwargs) -> logging.Formatter:
    """
    Factory function to create formatters.

    Args:
        formatter_type: Type of formatter ('console', 'file', 'json', 'performance')
        **kwargs: Additional arguments for the formatter

    Returns:
        Configured formatter instance
    """
    formatters = {
        "console": ColoredConsoleFormatter,
        "file": StructuredFileFormatter,
        "json": JSONFormatter,
        "performance": PerformanceFormatter,
    }

    if formatter_type not in formatters:
        raise ValueError(f"Unknown formatter type: {formatter_type}")

    formatter_class = formatters[formatter_type]
    return formatter_class(**kwargs)
