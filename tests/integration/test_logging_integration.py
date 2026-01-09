"""
Integration tests for comprehensive logging system.

Tests verify:
- JSON formatter produces valid, parseable JSON with correct schema
- Session context correlation across log entries
- Rate limiting behavior under load
- Path sanitization in log output
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

from xpcsviewer.utils.log_formatters import JSONFormatter
from xpcsviewer.utils.log_utils import (
    LoggingContext,
    RateLimitedLogger,
    SessionContextFilter,
    sanitize_path,
)
from xpcsviewer.utils.logging_config import get_logger


class TestJSONFormatterSchema:
    """Test JSON formatter produces valid schema for log aggregation (T077)."""

    def test_json_formatter_produces_valid_json(self) -> None:
        """Verify JSON formatter output is parseable JSON."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.json_schema")
        logger.setLevel(logging.DEBUG)

        # Create a handler with JSON formatter
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            logger.info("Test message")
            handler.flush()

            output = stream.getvalue()
            entry = json.loads(output)

            assert isinstance(entry, dict)
        finally:
            logger.removeHandler(handler)

    def test_json_entry_contains_required_fields(self) -> None:
        """Verify JSON entries contain all required fields for aggregation."""
        formatter = JSONFormatter(app_name="XPCS Viewer", app_version="2.0.0")
        logger = logging.getLogger("test.required_fields")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        # Add session context filter
        context_filter = SessionContextFilter()
        handler.addFilter(context_filter)
        logger.addHandler(handler)

        try:
            with LoggingContext(
                operation="test_operation", current_file="/tmp/test.h5"
            ):
                logger.info("Test with context")
            handler.flush()

            output = stream.getvalue()
            entry = json.loads(output)

            # Required fields per spec
            assert "timestamp" in entry
            assert "level" in entry
            assert "logger" in entry
            assert "message" in entry
            assert "module" in entry
            assert "function" in entry
            assert "line" in entry
            assert "thread" in entry
            assert "session_id" in entry
            assert "app_name" in entry
            assert "app_version" in entry
            assert "elapsed_seconds" in entry
            assert "elapsed_ms" in entry
            assert "process_id" in entry

            # Context fields when set
            assert "operation" in entry
            assert entry["operation"] == "test_operation"

        finally:
            logger.removeHandler(handler)
            handler.removeFilter(context_filter)

    def test_json_field_types(self) -> None:
        """Verify JSON fields have correct types for aggregation systems."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.field_types")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            logger.warning("Type check message")
            handler.flush()

            entry = json.loads(stream.getvalue())

            # String fields
            assert isinstance(entry["timestamp"], str)
            assert isinstance(entry["level"], str)
            assert isinstance(entry["logger"], str)
            assert isinstance(entry["message"], str)
            assert isinstance(entry["module"], str)
            assert isinstance(entry["function"], str)
            assert isinstance(entry["thread"], str)
            assert isinstance(entry["session_id"], str)
            assert isinstance(entry["app_name"], str)
            assert isinstance(entry["app_version"], str)

            # Numeric fields
            assert isinstance(entry["line"], int)
            assert isinstance(entry["elapsed_seconds"], (int, float))
            assert isinstance(entry["elapsed_ms"], (int, float))
            assert isinstance(entry["process_id"], int)

        finally:
            logger.removeHandler(handler)

    def test_json_exception_structure(self) -> None:
        """Verify exception information is properly structured."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.exception")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Caught an error")
            handler.flush()

            entry = json.loads(stream.getvalue())

            assert "exception" in entry
            exc_info = entry["exception"]
            assert "type" in exc_info
            assert exc_info["type"] == "ValueError"
            assert "message" in exc_info
            assert "Test error" in exc_info["message"]
            assert "traceback" in exc_info
            assert isinstance(exc_info["traceback"], list)

        finally:
            logger.removeHandler(handler)


class TestSessionCorrelation:
    """Test session context correlation across log entries (T092)."""

    def test_session_id_consistency_within_context(self) -> None:
        """Verify all logs within a context share the same session_id."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.session_consistency")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        context_filter = SessionContextFilter()
        handler.addFilter(context_filter)
        logger.addHandler(handler)

        try:
            with LoggingContext(operation="consistency_test") as ctx:
                expected_session = ctx.session_id
                logger.info("First message")
                logger.info("Second message")
                logger.info("Third message")
            handler.flush()

            lines = stream.getvalue().strip().split("\n")
            assert len(lines) == 3

            session_ids = []
            for line in lines:
                entry = json.loads(line)
                session_ids.append(entry["session_id"])

            # All should be the same
            assert all(sid == expected_session for sid in session_ids)

        finally:
            logger.removeHandler(handler)
            handler.removeFilter(context_filter)

    def test_nested_contexts_preserve_session(self) -> None:
        """Verify nested contexts maintain session continuity."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.nested_context")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        context_filter = SessionContextFilter()
        handler.addFilter(context_filter)
        logger.addHandler(handler)

        try:
            with LoggingContext(operation="outer") as outer_ctx:
                logger.info("In outer context")
                outer_session = outer_ctx.session_id

                # Inner context gets its own session
                with LoggingContext(operation="inner") as inner_ctx:
                    logger.info("In inner context")
                    inner_session = inner_ctx.session_id

                # Back to outer
                logger.info("Back in outer")

            handler.flush()

            lines = stream.getvalue().strip().split("\n")
            entries = [json.loads(line) for line in lines]

            # First and third should have outer session
            assert entries[0]["session_id"] == outer_session
            assert entries[0]["operation"] == "outer"
            assert entries[2]["session_id"] == outer_session
            assert entries[2]["operation"] == "outer"

            # Second should have inner session
            assert entries[1]["session_id"] == inner_session
            assert entries[1]["operation"] == "inner"

        finally:
            logger.removeHandler(handler)
            handler.removeFilter(context_filter)


class TestRateLimitingUnderLoad:
    """Test rate limiting behavior under high-frequency logging (T093)."""

    def test_rate_limiting_suppresses_excess_messages(self) -> None:
        """Verify rate limiter suppresses messages beyond rate limit."""
        logger = logging.getLogger("test.rate_limit")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        try:
            rate_limited = RateLimitedLogger(
                logger, rate_per_second=5.0, burst_size=5.0
            )

            logged_count = 0
            # Try to log 20 identical messages rapidly (same message key)
            for _ in range(20):
                if rate_limited.debug("Repeated high-frequency event"):
                    logged_count += 1

            # Should have logged burst_size messages initially
            # then suppressed the rest
            assert logged_count <= 6  # burst_size + 1 margin
            assert rate_limited.get_suppressed_count() >= 14

        finally:
            logger.removeHandler(handler)

    def test_rate_limiting_recovers_over_time(self) -> None:
        """Verify rate limiter allows messages again after time passes."""
        logger = logging.getLogger("test.rate_recovery")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        try:
            rate_limited = RateLimitedLogger(
                logger, rate_per_second=10.0, burst_size=2.0
            )

            # Use same message to stay in same rate limit bucket
            msg = "Recovery test message"

            # Exhaust the burst
            rate_limited.debug(msg)
            rate_limited.debug(msg)

            # This should be suppressed (burst exhausted)
            suppressed = not rate_limited.debug(msg)
            assert suppressed

            # Wait for token refill (150ms should give ~1.5 tokens at 10/sec)
            time.sleep(0.15)

            # Should be able to log again (same message)
            logged = rate_limited.debug(msg)
            assert logged

        finally:
            logger.removeHandler(handler)


class TestPathSanitization:
    """Test path sanitization in log output (T094)."""

    def test_home_mode_replaces_home_directory(self) -> None:
        """Verify home mode replaces home directory with ~."""
        import os

        home = os.path.expanduser("~")
        test_path = f"{home}/data/experiment.h5"

        sanitized = sanitize_path(test_path, mode="home")

        assert sanitized == "~/data/experiment.h5"
        assert home not in sanitized

    def test_hash_mode_hashes_filename(self) -> None:
        """Verify hash mode hashes the filename while preserving extension."""
        import os

        home = os.path.expanduser("~")
        test_path = f"{home}/data/sensitive_name.h5"

        sanitized = sanitize_path(test_path, mode="hash")

        assert sanitized.startswith("~/data/")
        assert sanitized.endswith(".h5")
        assert "sensitive_name" not in sanitized
        # Hash should be 8 characters
        filename = Path(sanitized).stem
        assert len(filename) == 8

    def test_none_mode_preserves_full_path(self) -> None:
        """Verify none mode preserves the full path."""
        test_path = "/some/full/path/to/file.h5"

        sanitized = sanitize_path(test_path, mode="none")

        assert sanitized == test_path

    def test_sanitize_none_path(self) -> None:
        """Verify None path is handled gracefully."""
        sanitized = sanitize_path(None)

        assert sanitized == "<none>"

    def test_sanitize_path_in_json_output(self) -> None:
        """Verify sanitized paths appear correctly in JSON logs."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.path_json")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)

        context_filter = SessionContextFilter()
        handler.addFilter(context_filter)
        logger.addHandler(handler)

        try:
            import os

            home = os.path.expanduser("~")
            test_path = f"{home}/secret/data.h5"

            with LoggingContext(current_file=test_path):
                logger.info("Loading file")
            handler.flush()

            entry = json.loads(stream.getvalue())

            # current_file should be sanitized
            assert entry.get("current_file", "").startswith("~")
            assert home not in entry.get("current_file", "")

        finally:
            logger.removeHandler(handler)
            handler.removeFilter(context_filter)


class TestMultipleEntriesParsing:
    """Test parsing multiple JSON log entries."""

    def test_multiple_entries_are_parseable(self) -> None:
        """Verify multiple JSON entries can be parsed line-by-line."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.multi_entry")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            for i in range(10):
                logger.info(f"Message {i}")
            handler.flush()

            lines = stream.getvalue().strip().split("\n")
            assert len(lines) == 10

            entries = []
            for line in lines:
                entry = json.loads(line)
                entries.append(entry)

            # Verify each entry has a message
            for i, entry in enumerate(entries):
                assert f"Message {i}" in entry["message"]

        finally:
            logger.removeHandler(handler)

    def test_entries_can_be_queried_by_field(self) -> None:
        """Verify entries can be filtered by any structured field."""
        formatter = JSONFormatter(app_name="Test", app_version="1.0.0")
        logger = logging.getLogger("test.queryable")
        logger.setLevel(logging.DEBUG)

        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        try:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            handler.flush()

            lines = stream.getvalue().strip().split("\n")
            entries = [json.loads(line) for line in lines]

            # Query by level
            warnings = [e for e in entries if e["level"] == "WARNING"]
            assert len(warnings) == 1
            assert "Warning message" in warnings[0]["message"]

            errors = [e for e in entries if e["level"] == "ERROR"]
            assert len(errors) == 1

            # Query by module
            from_module = [e for e in entries if "test.queryable" in e["logger"]]
            assert len(from_module) == 4

        finally:
            logger.removeHandler(handler)
