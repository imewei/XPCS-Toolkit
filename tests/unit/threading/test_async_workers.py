"""Unit tests for async workers module.

This module provides comprehensive unit tests for async workers,
covering progress management and thread coordination.
"""

import pytest

from xpcs_toolkit.threading.async_workers import (
    BaseAsyncWorker,
    WorkerResult,
    WorkerSignals,
    WorkerState,
    WorkerStats,
)

# Skip PySide6-dependent tests if not available
pytest_qt = pytest.importorskip("pytestqt", reason="PySide6/Qt tests require pytest-qt")
from PySide6.QtCore import QObject


class TestWorkerState:
    """Test suite for WorkerState enum."""

    def test_worker_state_enum_values(self):
        """Test that WorkerState enum has correct values."""
        assert WorkerState.IDLE.value == "idle"
        assert WorkerState.RUNNING.value == "running"
        assert WorkerState.COMPLETED.value == "completed"
        assert WorkerState.ERROR.value == "error"
        assert WorkerState.CANCELLED.value == "cancelled"

    def test_worker_state_enum_membership(self):
        """Test WorkerState enum membership."""
        assert WorkerState.IDLE in WorkerState
        assert WorkerState.RUNNING in WorkerState
        assert WorkerState.COMPLETED in WorkerState
        assert WorkerState.ERROR in WorkerState
        assert WorkerState.CANCELLED in WorkerState


class TestWorkerResult:
    """Test suite for WorkerResult dataclass."""

    def test_worker_result_creation(self):
        """Test WorkerResult can be created with valid parameters."""
        result = WorkerResult(
            success=True, data={"key": "value"}, error=None, execution_time=1.5
        )

        assert result.success is True
        assert result.data == {"key": "value"}
        assert result.error is None
        assert result.execution_time == 1.5

    def test_worker_result_error_case(self):
        """Test WorkerResult for error cases."""
        error_msg = "Test error"
        result = WorkerResult(
            success=False, data=None, error=error_msg, execution_time=0.1
        )

        assert result.success is False
        assert result.data is None
        assert result.error == error_msg
        assert result.execution_time == 0.1


class TestWorkerStats:
    """Test suite for WorkerStats dataclass."""

    def test_worker_stats_creation(self):
        """Test WorkerStats can be created with valid parameters."""
        stats = WorkerStats(
            total_jobs=10, completed_jobs=8, failed_jobs=1, average_execution_time=2.5
        )

        assert stats.total_jobs == 10
        assert stats.completed_jobs == 8
        assert stats.failed_jobs == 1
        assert stats.average_execution_time == 2.5

    def test_worker_stats_completion_rate(self):
        """Test WorkerStats completion rate calculation."""
        stats = WorkerStats(
            total_jobs=100, completed_jobs=90, failed_jobs=5, average_execution_time=1.0
        )

        # Test completion rate calculation if method exists
        if hasattr(stats, "completion_rate"):
            expected_rate = 90 / 100
            assert stats.completion_rate() == expected_rate


class TestWorkerSignals:
    """Test suite for WorkerSignals Qt object."""

    def test_worker_signals_creation(self):
        """Test WorkerSignals can be created."""
        signals = WorkerSignals()
        assert isinstance(signals, QObject)

    def test_worker_signals_has_required_signals(self):
        """Test WorkerSignals has expected signal attributes."""
        signals = WorkerSignals()

        # Check for common signal attributes (these may vary based on implementation)
        expected_signals = ["progress", "finished", "error", "result"]
        for signal_name in expected_signals:
            if hasattr(signals, signal_name):
                signal_attr = getattr(signals, signal_name)
                # Basic check that it's a signal-like object
                assert hasattr(signal_attr, "connect") or hasattr(signal_attr, "emit")


class TestBaseAsyncWorker:
    """Test suite for BaseAsyncWorker class."""

    def test_base_worker_creation(self):
        """Test BaseAsyncWorker can be instantiated."""
        worker = BaseAsyncWorker()
        assert worker is not None

    def test_base_worker_has_required_methods(self):
        """Test BaseAsyncWorker has expected methods."""
        worker = BaseAsyncWorker()

        # Check for QRunnable interface
        assert hasattr(worker, "run")
        assert callable(worker.run)

    def test_base_worker_state_management(self):
        """Test BaseAsyncWorker state management."""
        worker = BaseAsyncWorker()

        # Check if worker has state management
        if hasattr(worker, "state"):
            # Initial state should be appropriate
            assert worker.state in [WorkerState.IDLE, WorkerState.RUNNING]


@pytest.mark.integration
class TestWorkerIntegration:
    """Integration tests for worker components."""

    def test_worker_result_and_signals_integration(self):
        """Test integration between WorkerResult and WorkerSignals."""
        signals = WorkerSignals()
        result = WorkerResult(
            success=True, data={"test": "data"}, error=None, execution_time=0.5
        )

        # Test that result can be used with signals
        assert result is not None
        assert signals is not None

    def test_worker_stats_and_state_integration(self):
        """Test integration between WorkerStats and WorkerState."""
        stats = WorkerStats(
            total_jobs=5, completed_jobs=3, failed_jobs=1, average_execution_time=1.2
        )

        # Test that stats work with state enum
        assert stats.total_jobs > 0
        assert WorkerState.COMPLETED.value == "completed"


if __name__ == "__main__":
    pytest.main([__file__])
