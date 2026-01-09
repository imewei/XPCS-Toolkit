"""
Unit tests for xpcsviewer.utils.log_utils.

Tests cover:
    - LoggingContext: Session context management with contextvars
    - SessionContextFilter: Log record enrichment
    - RateLimitedLogger: Token bucket rate limiting
    - log_timing: Method timing decorator
    - sanitize_path: Path privacy sanitization
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xpcsviewer.utils.log_utils import (
    LoggingContext,
    RateLimitedLogger,
    SessionContextFilter,
    get_session_context,
    log_timing,
    sanitize_path,
)

# =============================================================================
# LoggingContext Tests (T012)
# =============================================================================


class TestLoggingContext:
    """Tests for LoggingContext class."""

    def test_context_generates_session_id(self):
        """LoggingContext generates a UUID4-like session_id when not provided."""
        with LoggingContext() as ctx:
            assert ctx.session_id is not None
            assert len(ctx.session_id) == 8  # First 8 chars of UUID4

    def test_context_uses_provided_session_id(self):
        """LoggingContext uses explicitly provided session_id."""
        with LoggingContext(session_id="custom-id") as ctx:
            assert ctx.session_id == "custom-id"

    def test_context_sets_contextvars(self):
        """LoggingContext sets contextvars within the context."""
        with LoggingContext(session_id="test-123", operation="testing") as ctx:
            context = get_session_context()
            assert context["session_id"] == "test-123"
            assert context["operation"] == "testing"

    def test_context_restores_on_exit(self):
        """LoggingContext restores previous values on exit."""
        # Before context
        before = get_session_context()
        assert before["session_id"] == "no-session"

        with LoggingContext(session_id="inner"):
            inner = get_session_context()
            assert inner["session_id"] == "inner"

        # After context - should be restored
        after = get_session_context()
        assert after["session_id"] == "no-session"

    def test_nested_contexts(self):
        """Nested LoggingContexts work correctly."""
        with LoggingContext(session_id="outer") as outer:
            assert get_session_context()["session_id"] == "outer"

            with LoggingContext(session_id="inner") as inner:
                assert get_session_context()["session_id"] == "inner"

            # Back to outer
            assert get_session_context()["session_id"] == "outer"

    def test_update_operation(self):
        """update_operation updates the operation field."""
        with LoggingContext(operation="initial") as ctx:
            assert get_session_context()["operation"] == "initial"

            ctx.update_operation("updated")
            assert get_session_context()["operation"] == "updated"

    def test_update_file(self):
        """update_file updates and sanitizes the current_file field."""
        with LoggingContext() as ctx:
            ctx.update_file("/path/to/file.hdf5")
            # Path should be sanitized (home mode by default)
            current_file = get_session_context()["current_file"]
            assert current_file is not None

    def test_update_file_none(self):
        """update_file with None clears the current_file field."""
        with LoggingContext() as ctx:
            ctx.update_file("/some/path.hdf5")
            ctx.update_file(None)
            assert get_session_context()["current_file"] == ""

    def test_get_current_returns_none_outside_context(self):
        """get_current returns None when no context is active."""
        assert LoggingContext.get_current() is None

    def test_get_current_returns_context_inside(self):
        """get_current returns context info when context is active."""
        with LoggingContext(session_id="active", operation="test"):
            ctx = LoggingContext.get_current()
            assert ctx is not None
            assert ctx.session_id == "active"

    def test_operation_max_length(self):
        """Operation is truncated to 100 characters."""
        long_op = "x" * 200
        with LoggingContext() as ctx:
            ctx.update_operation(long_op)
            assert len(get_session_context()["operation"]) == 100


# =============================================================================
# SessionContextFilter Tests (T016)
# =============================================================================


class TestSessionContextFilter:
    """Tests for SessionContextFilter class."""

    def test_filter_adds_session_id(self):
        """SessionContextFilter adds session_id to log records."""
        filter_ = SessionContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with LoggingContext(session_id="filter-test"):
            result = filter_.filter(record)

        assert result is True
        assert hasattr(record, "session_id")
        assert record.session_id == "filter-test"  # type: ignore[attr-defined]

    def test_filter_adds_operation(self):
        """SessionContextFilter adds operation to log records."""
        filter_ = SessionContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with LoggingContext(operation="filtering"):
            filter_.filter(record)

        assert record.operation == "filtering"  # type: ignore[attr-defined]

    def test_filter_always_returns_true(self):
        """SessionContextFilter always allows records through."""
        filter_ = SessionContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        assert filter_.filter(record) is True

    def test_filter_defaults_to_no_session(self):
        """SessionContextFilter defaults to 'no-session' outside context."""
        filter_ = SessionContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        filter_.filter(record)
        assert record.session_id == "no-session"  # type: ignore[attr-defined]


# =============================================================================
# RateLimitedLogger Tests (T013)
# =============================================================================


class TestRateLimitedLogger:
    """Tests for RateLimitedLogger class."""

    def test_first_message_always_logged(self):
        """First message is always logged."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=1.0)

        result = rate_limited.info("test message")

        assert result is True
        mock_logger.log.assert_called_once()

    def test_rate_limiting_suppresses_excess(self):
        """Messages exceeding rate limit are suppressed."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=1.0)

        # First message passes
        assert rate_limited.info("test") is True
        # Second immediate message should be suppressed
        assert rate_limited.info("test") is False

        # Only one call should have been made
        assert mock_logger.log.call_count == 1

    def test_rate_limiting_recovers_over_time(self):
        """Rate limit recovers as time passes."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=100.0)

        # Deplete tokens
        rate_limited.info("test")
        rate_limited.info("test")

        # Wait for token recovery
        time.sleep(0.02)  # Should recover ~2 tokens at 100/sec

        # Should be able to log again
        assert rate_limited.info("test") is True

    def test_different_messages_rate_limited_separately(self):
        """Different message templates are rate-limited independently."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=1.0)

        # Both unique messages should pass
        assert rate_limited.info("message A") is True
        assert rate_limited.info("message B") is True

        assert mock_logger.log.call_count == 2

    def test_get_suppressed_count(self):
        """get_suppressed_count returns correct counts."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=1.0)

        rate_limited.info("test")  # Logged
        rate_limited.info("test")  # Suppressed
        rate_limited.info("test")  # Suppressed

        assert rate_limited.get_suppressed_count() == 2

    def test_reset_clears_state(self):
        """reset clears all rate limit state."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=1.0)

        rate_limited.info("test")
        rate_limited.info("test")  # Suppressed

        rate_limited.reset()

        # After reset, should be able to log again
        assert rate_limited.info("test") is True

    def test_all_log_levels(self):
        """All log levels work correctly."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=100.0)

        assert rate_limited.debug("debug") is True
        assert rate_limited.info("info") is True
        assert rate_limited.warning("warning") is True
        assert rate_limited.error("error") is True
        assert rate_limited.critical("critical") is True

        assert mock_logger.log.call_count == 5

    def test_burst_size_default(self):
        """Burst size defaults to rate_per_second."""
        mock_logger = MagicMock(spec=logging.Logger)
        rate_limited = RateLimitedLogger(mock_logger, rate_per_second=5.0)

        # Should be able to log 5 messages immediately (burst)
        results = [rate_limited.info(f"msg{i}") for i in range(5)]
        assert all(results)

    def test_env_var_default_rate(self):
        """Uses PYXPCS_LOG_RATE_LIMIT env var for default rate."""
        with patch.dict(os.environ, {"PYXPCS_LOG_RATE_LIMIT": "20.0"}):
            mock_logger = MagicMock(spec=logging.Logger)
            rate_limited = RateLimitedLogger(mock_logger)
            assert rate_limited._rate == 20.0


# =============================================================================
# log_timing Decorator Tests (T015)
# =============================================================================


class TestLogTiming:
    """Tests for log_timing decorator."""

    def test_logs_execution_time(self):
        """log_timing logs the execution time."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger)
        def fast_function():
            return 42

        result = fast_function()

        assert result == 42
        mock_logger.log.assert_called_once()
        call_args = mock_logger.log.call_args
        assert "completed in" in call_args[0][1]
        assert "ms" in call_args[0][1]

    def test_logs_at_specified_level(self):
        """log_timing logs at the specified level."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger, level=logging.INFO)
        def test_func():
            pass

        test_func()

        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.INFO

    def test_threshold_elevates_level(self):
        """log_timing elevates level when threshold exceeded."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(
            logger=mock_logger,
            level=logging.DEBUG,
            threshold_ms=1,
            threshold_level=logging.WARNING,
        )
        def slow_function():
            time.sleep(0.01)  # 10ms > 1ms threshold

        slow_function()

        call_args = mock_logger.log.call_args
        assert call_args[0][0] == logging.WARNING

    def test_includes_args_when_requested(self):
        """log_timing includes function arguments when include_args=True."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger, include_args=True)
        def func_with_args(x, y=10):
            return x + y

        func_with_args(5, y=20)

        call_args = mock_logger.log.call_args
        message = call_args[0][1]
        assert "5" in message or "x=" in message

    def test_logs_error_on_exception(self):
        """log_timing logs error and re-raises on exception."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger)
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            failing_function()

        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args
        assert "failed after" in call_args[0][0]

    def test_preserves_function_metadata(self):
        """log_timing preserves the decorated function's metadata."""

        @log_timing()
        def documented_function():
            """This is the docstring."""
            pass

        assert documented_function.__name__ == "documented_function"
        assert "docstring" in documented_function.__doc__

    def test_returns_function_result(self):
        """log_timing returns the original function's result."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger)
        def compute():
            return {"key": "value", "count": 42}

        result = compute()
        assert result == {"key": "value", "count": 42}


# =============================================================================
# sanitize_path Tests (T014)
# =============================================================================


class TestSanitizePath:
    """Tests for sanitize_path function."""

    def test_none_returns_marker(self):
        """sanitize_path returns '<none>' for None input."""
        assert sanitize_path(None) == "<none>"

    def test_mode_none_returns_full_path(self):
        """sanitize_path with mode='none' returns full path."""
        path = "/Users/john/data/file.hdf5"
        result = sanitize_path(path, mode="none")
        assert result == path

    def test_mode_home_replaces_home_dir(self):
        """sanitize_path with mode='home' replaces home directory with ~."""
        home = os.path.expanduser("~")
        path = f"{home}/data/file.hdf5"
        result = sanitize_path(path, mode="home")
        assert result == "~/data/file.hdf5"

    def test_mode_home_preserves_non_home_paths(self):
        """sanitize_path with mode='home' preserves paths not under home."""
        path = "/var/log/app.log"
        result = sanitize_path(path, mode="home")
        assert result == path

    def test_mode_hash_hashes_filename(self):
        """sanitize_path with mode='hash' hashes the filename."""
        home = os.path.expanduser("~")
        path = f"{home}/data/secret_file.hdf5"
        result = sanitize_path(path, mode="hash")

        # Should have ~ prefix
        assert result.startswith("~/data/")
        # Should not contain original filename
        assert "secret_file" not in result
        # Should preserve extension
        assert result.endswith(".hdf5")
        # Should have 8-char hash
        filename = Path(result).stem
        assert len(filename) == 8

    def test_accepts_path_object(self):
        """sanitize_path accepts pathlib.Path objects."""
        home = Path.home()
        path = home / "data" / "file.hdf5"
        result = sanitize_path(path, mode="home")
        assert result == "~/data/file.hdf5"

    def test_env_var_default_mode(self):
        """sanitize_path uses PYXPCS_LOG_SANITIZE_PATHS env var."""
        with patch.dict(os.environ, {"PYXPCS_LOG_SANITIZE_PATHS": "none"}):
            home = os.path.expanduser("~")
            path = f"{home}/data/file.hdf5"
            result = sanitize_path(path)
            assert result == path  # No sanitization

    def test_invalid_env_mode_defaults_to_home(self):
        """Invalid env var mode defaults to 'home'."""
        with patch.dict(os.environ, {"PYXPCS_LOG_SANITIZE_PATHS": "invalid"}):
            home = os.path.expanduser("~")
            path = f"{home}/data/file.hdf5"
            result = sanitize_path(path)
            assert result == "~/data/file.hdf5"


# =============================================================================
# Integration Tests
# =============================================================================


class TestLogUtilsIntegration:
    """Integration tests for log_utils module."""

    def test_context_with_filter_integration(self):
        """LoggingContext works with SessionContextFilter."""
        filter_ = SessionContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        with LoggingContext(
            session_id="integration",
            operation="testing",
        ) as ctx:
            ctx.update_file("/path/to/data.hdf5")
            filter_.filter(record)

            assert record.session_id == "integration"  # type: ignore[attr-defined]
            assert record.operation == "testing"  # type: ignore[attr-defined]

    def test_timing_with_context(self):
        """log_timing works within LoggingContext."""
        mock_logger = MagicMock(spec=logging.Logger)

        @log_timing(logger=mock_logger)
        def timed_operation():
            return "result"

        with LoggingContext(session_id="timed"):
            result = timed_operation()

        assert result == "result"
        mock_logger.log.assert_called_once()
