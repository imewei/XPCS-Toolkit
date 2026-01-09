"""
Logging utilities for comprehensive application monitoring.

This module provides logging utilities for session context management,
rate-limited logging, method timing, and path sanitization.

Features:
    - LoggingContext: Session-level context with correlation IDs (contextvars)
    - SessionContextFilter: Logging filter that adds session context to records
    - RateLimitedLogger: Token bucket rate-limited logging wrapper
    - log_timing: Decorator for logging method entry/exit with timing
    - sanitize_path: Path sanitization for privacy in logs

Environment Variables:
    PYXPCS_LOG_RATE_LIMIT: Default rate limit in msgs/sec (default: 10.0)
    PYXPCS_LOG_SANITIZE_PATHS: Path sanitization mode: none/home/hash (default: home)
    PYXPCS_LOG_SESSION_ID: Enable session IDs: 1/0 (default: 1)

Usage:
    from xpcsviewer.utils.log_utils import (
        LoggingContext,
        RateLimitedLogger,
        log_timing,
        sanitize_path,
    )

    # Session context
    with LoggingContext(operation="data_analysis") as ctx:
        logger.info("Starting analysis")  # Includes session_id

    # Rate-limited logging
    rate_limited = RateLimitedLogger(logger, rate_per_second=5.0)
    rate_limited.debug("High-frequency event")

    # Timing decorator
    @log_timing()
    def process_data(data):
        ...

    # Path sanitization
    logger.info(f"Loaded: {sanitize_path(file_path)}")
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

if TYPE_CHECKING:
    from typing import ParamSpec

    P = ParamSpec("P")

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for session tracking
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="no-session"
)
_operation: contextvars.ContextVar[str] = contextvars.ContextVar(
    "operation", default=""
)
_current_file: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_file", default=""
)


def _get_env_float(name: str, default: float) -> float:
    """Get a float environment variable with fallback to default."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        result = float(value)
        return result if result > 0 else default
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    """Get a boolean environment variable with fallback to default."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _get_sanitize_mode() -> Literal["none", "home", "hash"]:
    """Get path sanitization mode from environment."""
    value = os.environ.get("PYXPCS_LOG_SANITIZE_PATHS", "home").lower()
    if value in ("none", "home", "hash"):
        return value  # type: ignore[return-value]
    return "home"


class LoggingContext:
    """
    Manage session-level logging context with correlation IDs.

    Uses contextvars for thread-safe session tracking. All log entries
    within a context share the same session_id for correlation.

    Attributes:
        session_id: UUID4 identifier for the session (read-only after creation)

    Example:
        with LoggingContext(operation="batch_analysis") as ctx:
            logger.info("Starting batch")  # Includes session_id
            ctx.update_file("/path/to/data.hdf5")
            process_file()
    """

    _token_session: contextvars.Token[str] | None = None
    _token_operation: contextvars.Token[str] | None = None
    _token_file: contextvars.Token[str] | None = None

    def __init__(
        self,
        session_id: str | None = None,
        operation: str | None = None,
        current_file: str | None = None,
    ) -> None:
        """
        Initialize a logging context.

        Args:
            session_id: Session ID (auto-generated UUID4 if not provided)
            operation: Current high-level operation name
            current_file: Currently loaded file path (will be sanitized)
        """
        self._session_id = session_id or str(uuid.uuid4())[:8]
        self._operation = operation or ""
        self._current_file = sanitize_path(current_file) if current_file else ""
        self._started_at = time.time()

    def __enter__(self) -> LoggingContext:
        """Enter the context manager, setting contextvars."""
        self._token_session = _session_id.set(self._session_id)
        self._token_operation = _operation.set(self._operation)
        self._token_file = _current_file.set(self._current_file)
        return self

    def __exit__(self, *args: object) -> None:
        """Exit the context manager, restoring previous contextvars."""
        if self._token_session is not None:
            _session_id.reset(self._token_session)
        if self._token_operation is not None:
            _operation.reset(self._token_operation)
        if self._token_file is not None:
            _current_file.reset(self._token_file)

    @property
    def session_id(self) -> str:
        """Get the session ID."""
        return self._session_id

    def update_operation(self, operation: str) -> None:
        """Update the current operation context."""
        self._operation = operation[:100]  # Max 100 chars per data model
        _operation.set(self._operation)

    def update_file(self, file_path: str | os.PathLike[str] | None) -> None:
        """Update the current file context (path will be sanitized)."""
        if file_path is None:
            self._current_file = ""
        else:
            self._current_file = sanitize_path(file_path)
        _current_file.set(self._current_file)

    @staticmethod
    def get_current() -> LoggingContext | None:
        """
        Get the current logging context if one is active.

        Returns:
            Current LoggingContext or None if no context is active
        """
        session = _session_id.get()
        if session == "no-session":
            return None
        # Create a read-only view of current context
        ctx = LoggingContext.__new__(LoggingContext)
        ctx._session_id = session
        ctx._operation = _operation.get()
        ctx._current_file = _current_file.get()
        ctx._started_at = 0.0  # Unknown for recovered context
        return ctx


class SessionContextFilter(logging.Filter):
    """
    Logging filter that adds session context to log records.

    Adds the following attributes to each LogRecord:
        - session_id: Current session ID or 'no-session'
        - operation: Current operation name or ''
        - current_file: Current file path (sanitized) or ''

    This filter should be added to handlers via logging_config.py.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add session context attributes to the log record.

        Args:
            record: The log record to enrich

        Returns:
            True (always allows the record through)
        """
        record.session_id = _session_id.get()  # type: ignore[attr-defined]
        record.operation = _operation.get()  # type: ignore[attr-defined]
        record.current_file = _current_file.get()  # type: ignore[attr-defined]
        return True


class RateLimitedLogger:
    """
    Rate-limited logger wrapper using token bucket algorithm.

    Prevents log flooding by limiting messages per second. Each unique
    message template is rate-limited independently.

    Attributes:
        suppressed_count: Total number of suppressed messages

    Example:
        logger = logging.getLogger(__name__)
        rate_limited = RateLimitedLogger(logger, rate_per_second=5.0)

        def on_mouse_move(x, y):
            rate_limited.debug(f"Mouse at ({x}, {y})")  # Max 5/sec
    """

    def __init__(
        self,
        logger: logging.Logger,
        rate_per_second: float | None = None,
        burst_size: float | None = None,
    ) -> None:
        """
        Initialize the rate-limited logger.

        Args:
            logger: The underlying logger to wrap
            rate_per_second: Maximum messages per second (default from env var)
            burst_size: Initial token bucket size (default: rate_per_second)
        """
        self._logger = logger
        default_rate = _get_env_float("PYXPCS_LOG_RATE_LIMIT", 10.0)
        self._rate = rate_per_second if rate_per_second is not None else default_rate
        self._burst_size = burst_size if burst_size is not None else self._rate

        # Token bucket state: {message_key: (tokens, last_update, suppressed_count)}
        self._buckets: dict[str, tuple[float, float, int]] = {}

    def _get_message_key(self, msg: str) -> str:
        """Generate a key for the message (first 50 chars for grouping)."""
        return msg[:50]

    def _try_consume(self, msg_key: str) -> bool:
        """
        Try to consume a token for the given message key.

        Returns True if message should be logged, False if suppressed.
        """
        now = time.monotonic()

        if msg_key not in self._buckets:
            # Initialize bucket with full tokens
            self._buckets[msg_key] = (self._burst_size - 1, now, 0)
            return True

        tokens, last_update, suppressed = self._buckets[msg_key]

        # Refill tokens based on elapsed time
        elapsed = now - last_update
        tokens = min(self._burst_size, tokens + elapsed * self._rate)

        if tokens >= 1.0:
            # Consume a token and log
            self._buckets[msg_key] = (tokens - 1, now, 0)
            return True
        else:
            # Suppress and increment counter
            self._buckets[msg_key] = (tokens, now, suppressed + 1)
            return False

    def _log(self, level: int, msg: str, *args: object, **kwargs: object) -> bool:
        """Internal logging method with rate limiting."""
        msg_key = self._get_message_key(msg)
        if self._try_consume(msg_key):
            self._logger.log(level, msg, *args, **kwargs)
            return True
        return False

    def debug(self, msg: str, *args: object, **kwargs: object) -> bool:
        """Log DEBUG message if not rate-limited. Returns True if logged."""
        return self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> bool:
        """Log INFO message if not rate-limited. Returns True if logged."""
        return self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> bool:
        """Log WARNING message if not rate-limited. Returns True if logged."""
        return self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> bool:
        """Log ERROR message if not rate-limited. Returns True if logged."""
        return self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args: object, **kwargs: object) -> bool:
        """Log CRITICAL message if not rate-limited. Returns True if logged."""
        return self._log(logging.CRITICAL, msg, *args, **kwargs)

    def get_suppressed_count(self, msg_key: str | None = None) -> int:
        """
        Get count of suppressed messages.

        Args:
            msg_key: Specific message key, or None for total

        Returns:
            Number of suppressed messages
        """
        if msg_key is not None:
            if msg_key in self._buckets:
                return self._buckets[msg_key][2]
            return 0
        return sum(bucket[2] for bucket in self._buckets.values())

    def reset(self) -> None:
        """Clear all rate limit state."""
        self._buckets.clear()


def log_timing(
    logger: logging.Logger | None = None,
    level: int = logging.DEBUG,
    threshold_ms: float | None = None,
    threshold_level: int = logging.WARNING,
    include_args: bool = False,
) -> Callable[[F], F]:
    """
    Decorator for logging method entry/exit with timing.

    Logs the method name and execution time. Optionally logs at a higher
    level if execution exceeds a threshold.

    Args:
        logger: Logger to use (default: logger for decorated function's module)
        level: Normal log level (default: DEBUG)
        threshold_ms: Time threshold in ms for elevated logging
        threshold_level: Level to use when threshold exceeded (default: WARNING)
        include_args: Whether to include function arguments in log

    Returns:
        Decorated function

    Example:
        @log_timing()
        def process_data(data):
            ...  # Logs: "process_data completed in 123.45ms"

        @log_timing(threshold_ms=1000)
        def slow_operation():
            ...  # Logs WARNING if > 1000ms
    """

    def decorator(func: F) -> F:
        func_logger = logger or logging.getLogger(func.__module__)
        func_name = func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> object:
            start_time = time.perf_counter()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Determine log level based on threshold
                log_level = level
                if threshold_ms is not None and elapsed_ms > threshold_ms:
                    log_level = threshold_level

                # Build message
                if include_args:
                    args_str = ", ".join(
                        [repr(a)[:50] for a in args[:3]]
                        + [f"{k}={v!r}"[:30] for k, v in list(kwargs.items())[:3]]
                    )
                    msg = f"{func_name}({args_str}) completed in {elapsed_ms:.2f}ms"
                else:
                    msg = f"{func_name} completed in {elapsed_ms:.2f}ms"

                func_logger.log(log_level, msg)
                return result

            except Exception:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                func_logger.error(
                    f"{func_name} failed after {elapsed_ms:.2f}ms",
                    exc_info=True,
                )
                raise

        return wrapper  # type: ignore[return-value]

    return decorator


def sanitize_path(
    path: str | os.PathLike[str] | None,
    mode: Literal["none", "home", "hash"] | None = None,
) -> str:
    """
    Sanitize file paths for logging privacy.

    Args:
        path: File path to sanitize
        mode: Sanitization mode (default from PYXPCS_LOG_SANITIZE_PATHS env var)
            - 'none': No sanitization, full path
            - 'home': Replace home directory with ~
            - 'hash': Replace home with ~ and hash the filename

    Returns:
        Sanitized path string

    Example:
        sanitize_path('/Users/john/data/file.h5')
        # mode='none' -> '/Users/john/data/file.h5'
        # mode='home' -> '~/data/file.h5'
        # mode='hash' -> '~/data/a1b2c3d4.h5'
    """
    if path is None:
        return "<none>"

    path_str = str(path)
    effective_mode = mode if mode is not None else _get_sanitize_mode()

    if effective_mode == "none":
        return path_str

    # Get home directory
    home = os.path.expanduser("~")

    # Replace home directory with ~
    if path_str.startswith(home):
        path_str = "~" + path_str[len(home) :]

    if effective_mode == "hash":
        # Hash the filename portion
        p = Path(path_str)
        if p.name:
            name_hash = hashlib.sha256(p.name.encode()).hexdigest()[:8]
            path_str = str(p.parent / f"{name_hash}{p.suffix}")

    return path_str


def get_session_context() -> dict[str, str]:
    """
    Get the current session context as a dictionary.

    Returns:
        Dictionary with session_id, operation, and current_file
    """
    return {
        "session_id": _session_id.get(),
        "operation": _operation.get(),
        "current_file": _current_file.get(),
    }
