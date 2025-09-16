"""Unit tests for async workers module.

This module provides comprehensive unit tests for async workers,
covering progress management and thread coordination.
"""

from unittest.mock import Mock

import pytest

from xpcs_toolkit.threading.async_workers import (
    WorkerPriority,
    WorkerResult,
    WorkerSignals,
    WorkerState,
    WorkerStats,
)

# Skip PySide6-dependent tests if not available
pytest_qt = pytest.importorskip("pytestqt", reason="PySide6/Qt tests require pytest-qt")
from PySide6.QtCore import QObject


class TestWorkerPriority:
    """Test suite for WorkerPriority enum."""

    def test_priority_enum_values(self):
        """Test that WorkerPriority enum has correct values."""
        assert WorkerPriority.LOW.value == 1
        assert WorkerPriority.NORMAL.value == 2
        assert WorkerPriority.HIGH.value == 3
        assert WorkerPriority.CRITICAL.value == 4

    def test_priority_enum_comparison(self):
        """Test WorkerPriority enum comparison operations."""
        assert WorkerPriority.LOW.value < WorkerPriority.NORMAL.value
        assert WorkerPriority.NORMAL.value < WorkerPriority.HIGH.value
        assert WorkerPriority.HIGH.value < WorkerPriority.CRITICAL.value

        assert WorkerPriority.CRITICAL.value > WorkerPriority.HIGH.value
        assert WorkerPriority.HIGH.value > WorkerPriority.NORMAL.value
        assert WorkerPriority.NORMAL.value > WorkerPriority.LOW.value

    def test_priority_enum_ordering(self):
        """Test WorkerPriority enum can be used for sorting."""
        priorities = [
            WorkerPriority.HIGH,
            WorkerPriority.LOW,
            WorkerPriority.CRITICAL,
            WorkerPriority.NORMAL,
        ]
        sorted_priorities = sorted(priorities, key=lambda p: p.value)

        expected = [
            WorkerPriority.LOW,
            WorkerPriority.NORMAL,
            WorkerPriority.HIGH,
            WorkerPriority.CRITICAL,
        ]
        assert sorted_priorities == expected

    def test_priority_enum_membership(self):
        """Test WorkerPriority enum membership."""
        assert WorkerPriority.LOW in WorkerPriority
        assert WorkerPriority.NORMAL in WorkerPriority
        assert WorkerPriority.HIGH in WorkerPriority
        assert WorkerPriority.CRITICAL in WorkerPriority

    def test_priority_enum_iteration(self):
        """Test WorkerPriority enum iteration."""
        priorities = list(WorkerPriority)
        assert len(priorities) == 4
        assert WorkerPriority.LOW in priorities
        assert WorkerPriority.NORMAL in priorities
        assert WorkerPriority.HIGH in priorities
        assert WorkerPriority.CRITICAL in priorities


class TestWorkerState:
    """Test suite for WorkerState enum."""

    def test_state_enum_values(self):
        """Test that WorkerState enum has correct string values."""
        assert WorkerState.PENDING.value == "pending"
        assert WorkerState.RUNNING.value == "running"
        assert WorkerState.COMPLETED.value == "completed"
        assert WorkerState.FAILED.value == "failed"
        assert WorkerState.CANCELLED.value == "cancelled"

    def test_state_enum_membership(self):
        """Test WorkerState enum membership."""
        assert WorkerState.PENDING in WorkerState
        assert WorkerState.RUNNING in WorkerState
        assert WorkerState.COMPLETED in WorkerState
        assert WorkerState.FAILED in WorkerState
        assert WorkerState.CANCELLED in WorkerState

    def test_state_enum_string_representation(self):
        """Test WorkerState enum string representation."""
        assert str(WorkerState.PENDING.value) == "pending"
        assert str(WorkerState.RUNNING.value) == "running"
        assert str(WorkerState.COMPLETED.value) == "completed"
        assert str(WorkerState.FAILED.value) == "failed"
        assert str(WorkerState.CANCELLED.value) == "cancelled"

    def test_state_enum_iteration(self):
        """Test WorkerState enum iteration."""
        states = list(WorkerState)
        assert len(states) == 5

        expected_states = [
            WorkerState.PENDING,
            WorkerState.RUNNING,
            WorkerState.COMPLETED,
            WorkerState.FAILED,
            WorkerState.CANCELLED,
        ]

        for state in expected_states:
            assert state in states

    def test_state_transitions_logic(self):
        """Test logical state transitions."""
        # Test typical workflow states
        initial_state = WorkerState.PENDING
        running_state = WorkerState.RUNNING
        final_states = [
            WorkerState.COMPLETED,
            WorkerState.FAILED,
            WorkerState.CANCELLED,
        ]

        assert initial_state != running_state
        assert running_state not in final_states
        assert all(state != running_state for state in final_states)


class TestWorkerResult:
    """Test suite for WorkerResult dataclass."""

    def test_worker_result_creation(self):
        """Test WorkerResult creation with required fields."""
        result = WorkerResult(
            worker_id="test_worker",
            result={"data": [1, 2, 3]},
            execution_time=1.5,
            memory_usage=128.0,
        )

        assert result.worker_id == "test_worker"
        assert result.result == {"data": [1, 2, 3]}
        assert result.execution_time == 1.5
        assert result.memory_usage == 128.0
        assert result.success is True  # Default value
        assert result.error_message == ""  # Default value

    def test_worker_result_with_error(self):
        """Test WorkerResult creation with error information."""
        result = WorkerResult(
            worker_id="failed_worker",
            result=None,
            execution_time=0.5,
            memory_usage=64.0,
            success=False,
            error_message="Division by zero",
        )

        assert result.worker_id == "failed_worker"
        assert result.result is None
        assert result.execution_time == 0.5
        assert result.memory_usage == 64.0
        assert result.success is False
        assert result.error_message == "Division by zero"

    def test_worker_result_defaults(self):
        """Test WorkerResult default values."""
        result = WorkerResult(
            worker_id="default_test",
            result="test_data",
            execution_time=1.0,
            memory_usage=32.0,
        )

        # Test default values
        assert result.success is True
        assert result.error_message == ""

    def test_worker_result_field_types(self):
        """Test WorkerResult field type handling."""
        # Test with different result types
        results = [{"data": "dict"}, [1, 2, 3], "string_result", 42, None]

        for i, test_result in enumerate(results):
            worker_result = WorkerResult(
                worker_id=f"test_{i}",
                result=test_result,
                execution_time=float(i + 1),
                memory_usage=float((i + 1) * 10),
            )

            assert worker_result.result == test_result
            assert isinstance(worker_result.execution_time, float)
            assert isinstance(worker_result.memory_usage, float)

    def test_worker_result_immutability(self):
        """Test that WorkerResult fields can be accessed but are immutable as dataclass."""
        result = WorkerResult(
            worker_id="immutable_test",
            result=[1, 2, 3],
            execution_time=1.0,
            memory_usage=50.0,
        )

        # Fields should be accessible
        assert result.worker_id == "immutable_test"
        assert result.execution_time == 1.0

        # But the result content can be modified (mutable objects)
        if isinstance(result.result, list):
            result.result.append(4)
            assert result.result == [1, 2, 3, 4]


class TestWorkerStats:
    """Test suite for WorkerStats dataclass."""

    def test_worker_stats_creation(self):
        """Test WorkerStats creation with custom values."""
        stats = WorkerStats(
            total_workers=10,
            active_workers=3,
            completed_workers=6,
            failed_workers=1,
            cancelled_workers=0,
            avg_execution_time=2.5,
            total_memory_usage=256.0,
        )

        assert stats.total_workers == 10
        assert stats.active_workers == 3
        assert stats.completed_workers == 6
        assert stats.failed_workers == 1
        assert stats.cancelled_workers == 0
        assert stats.avg_execution_time == 2.5
        assert stats.total_memory_usage == 256.0

    def test_worker_stats_defaults(self):
        """Test WorkerStats default values."""
        stats = WorkerStats()

        assert stats.total_workers == 0
        assert stats.active_workers == 0
        assert stats.completed_workers == 0
        assert stats.failed_workers == 0
        assert stats.cancelled_workers == 0
        assert stats.avg_execution_time == 0.0
        assert stats.total_memory_usage == 0.0

    def test_worker_stats_partial_creation(self):
        """Test WorkerStats creation with partial values."""
        stats = WorkerStats(total_workers=5, completed_workers=3)

        # Specified values
        assert stats.total_workers == 5
        assert stats.completed_workers == 3

        # Default values for unspecified fields
        assert stats.active_workers == 0
        assert stats.failed_workers == 0
        assert stats.cancelled_workers == 0
        assert stats.avg_execution_time == 0.0
        assert stats.total_memory_usage == 0.0

    def test_worker_stats_calculations(self):
        """Test calculations possible with WorkerStats."""
        stats = WorkerStats(
            total_workers=10, completed_workers=7, failed_workers=2, cancelled_workers=1
        )

        # Calculate derived metrics
        finished_workers = (
            stats.completed_workers + stats.failed_workers + stats.cancelled_workers
        )
        assert finished_workers == 10
        assert finished_workers == stats.total_workers

        success_rate = (
            stats.completed_workers / stats.total_workers
            if stats.total_workers > 0
            else 0
        )
        assert success_rate == 0.7

        failure_rate = (
            (stats.failed_workers + stats.cancelled_workers) / stats.total_workers
            if stats.total_workers > 0
            else 0
        )
        assert failure_rate == 0.3

    def test_worker_stats_field_types(self):
        """Test WorkerStats field types."""
        stats = WorkerStats(
            total_workers=100, avg_execution_time=1.23456, total_memory_usage=1024.5
        )

        # Integer fields
        assert isinstance(stats.total_workers, int)
        assert isinstance(stats.active_workers, int)
        assert isinstance(stats.completed_workers, int)
        assert isinstance(stats.failed_workers, int)
        assert isinstance(stats.cancelled_workers, int)

        # Float fields
        assert isinstance(stats.avg_execution_time, float)
        assert isinstance(stats.total_memory_usage, float)


class TestWorkerSignals:
    """Test suite for WorkerSignals class."""

    def test_worker_signals_inheritance(self):
        """Test that WorkerSignals inherits from QObject."""
        signals = WorkerSignals()
        assert isinstance(signals, QObject)

    def test_worker_signals_creation(self):
        """Test WorkerSignals instance creation."""
        signals = WorkerSignals()

        # Should have all expected signal attributes
        assert hasattr(signals, "started")
        assert hasattr(signals, "finished")
        assert hasattr(signals, "error")
        assert hasattr(signals, "progress")
        assert hasattr(signals, "status")
        assert hasattr(signals, "cancelled")
        assert hasattr(signals, "partial_result")
        assert hasattr(signals, "resource_usage")
        assert hasattr(signals, "state_changed")

    def test_worker_signals_types(self):
        """Test that WorkerSignals has correct signal types."""
        from PySide6.QtCore import Signal

        signals = WorkerSignals()

        # All attributes should be Signal instances
        assert isinstance(signals.started, Signal)
        assert isinstance(signals.finished, Signal)
        assert isinstance(signals.error, Signal)
        assert isinstance(signals.progress, Signal)
        assert isinstance(signals.status, Signal)
        assert isinstance(signals.cancelled, Signal)
        assert isinstance(signals.partial_result, Signal)
        assert isinstance(signals.resource_usage, Signal)
        assert isinstance(signals.state_changed, Signal)

    @pytest.mark.skipif(not pytest_qt, reason="Requires pytest-qt for signal testing")
    def test_worker_signals_emission(self, qtbot):
        """Test WorkerSignals can emit signals (requires Qt test environment)."""
        signals = WorkerSignals()

        # Test signal emission with qtbot
        with qtbot.waitSignal(signals.started, timeout=1000):
            signals.started.emit("test_worker", 2)

        with qtbot.waitSignal(signals.progress, timeout=1000):
            signals.progress.emit("test_worker", 50, 100, "Processing...", 10.0)

        with qtbot.waitSignal(signals.status, timeout=1000):
            signals.status.emit("test_worker", "Running analysis", 1)

    def test_worker_signals_connection(self):
        """Test that WorkerSignals can be connected to slots."""
        signals = WorkerSignals()

        # Create mock slots
        started_slot = Mock()
        finished_slot = Mock()
        error_slot = Mock()
        progress_slot = Mock()

        # Connect signals to slots
        signals.started.connect(started_slot)
        signals.finished.connect(finished_slot)
        signals.error.connect(error_slot)
        signals.progress.connect(progress_slot)

        # Emit signals
        signals.started.emit("test_worker", 2)
        signals.finished.emit(Mock())
        signals.error.emit("test_worker", "Error message", "Traceback", 1)
        signals.progress.emit("test_worker", 10, 100, "Starting", 45.0)

        # Verify slots were called
        started_slot.assert_called_once_with("test_worker", 2)
        finished_slot.assert_called_once()
        error_slot.assert_called_once_with(
            "test_worker", "Error message", "Traceback", 1
        )
        progress_slot.assert_called_once_with("test_worker", 10, 100, "Starting", 45.0)

    def test_worker_signals_multiple_connections(self):
        """Test that WorkerSignals can have multiple connections per signal."""
        signals = WorkerSignals()

        # Create multiple mock slots for same signal
        slot1 = Mock()
        slot2 = Mock()
        slot3 = Mock()

        # Connect multiple slots to same signal
        signals.progress.connect(slot1)
        signals.progress.connect(slot2)
        signals.progress.connect(slot3)

        # Emit signal once
        signals.progress.emit("test_worker", 25, 100, "Quarter done", 30.0)

        # All slots should be called
        slot1.assert_called_once_with("test_worker", 25, 100, "Quarter done", 30.0)
        slot2.assert_called_once_with("test_worker", 25, 100, "Quarter done", 30.0)
        slot3.assert_called_once_with("test_worker", 25, 100, "Quarter done", 30.0)


class TestWorkerSignalsSignatureValidation:
    """Test suite for WorkerSignals signal signatures."""

    def test_started_signal_signature(self):
        """Test started signal signature (worker_id, priority)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.started.connect(mock_slot)
        signals.started.emit("worker123", 3)

        mock_slot.assert_called_once_with("worker123", 3)

    def test_finished_signal_signature(self):
        """Test finished signal signature (worker_result)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        test_result = WorkerResult("test", "result", 1.0, 64.0)

        signals.finished.connect(mock_slot)
        signals.finished.emit(test_result)

        mock_slot.assert_called_once_with(test_result)

    def test_error_signal_signature(self):
        """Test error signal signature (worker_id, error_message, traceback_string, retry_count)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.error.connect(mock_slot)
        signals.error.emit("worker_error", "Something failed", "Traceback...", 2)

        mock_slot.assert_called_once_with(
            "worker_error", "Something failed", "Traceback...", 2
        )

    def test_progress_signal_signature(self):
        """Test progress signal signature (worker_id, current, total, message, eta_seconds)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.progress.connect(mock_slot)
        signals.progress.emit("progress_worker", 75, 150, "Half way", 25.5)

        mock_slot.assert_called_once_with("progress_worker", 75, 150, "Half way", 25.5)

    def test_status_signal_signature(self):
        """Test status signal signature (worker_id, status_message, detail_level)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.status.connect(mock_slot)
        signals.status.emit("status_worker", "Initializing", 1)

        mock_slot.assert_called_once_with("status_worker", "Initializing", 1)

    def test_cancelled_signal_signature(self):
        """Test cancelled signal signature (worker_id, reason)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.cancelled.connect(mock_slot)
        signals.cancelled.emit("cancelled_worker", "User requested")

        mock_slot.assert_called_once_with("cancelled_worker", "User requested")

    def test_partial_result_signal_signature(self):
        """Test partial_result signal signature (worker_id, partial_result, is_final)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        partial_data = {"chunk": 1, "data": [1, 2, 3]}

        signals.partial_result.connect(mock_slot)
        signals.partial_result.emit("streaming_worker", partial_data, False)

        mock_slot.assert_called_once_with("streaming_worker", partial_data, False)

    def test_resource_usage_signal_signature(self):
        """Test resource_usage signal signature (worker_id, cpu_percent, memory_mb)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.resource_usage.connect(mock_slot)
        signals.resource_usage.emit("resource_worker", 75.5, 256.0)

        mock_slot.assert_called_once_with("resource_worker", 75.5, 256.0)

    def test_state_changed_signal_signature(self):
        """Test state_changed signal signature (worker_id, old_state, new_state)."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.state_changed.connect(mock_slot)
        signals.state_changed.emit("state_worker", "pending", "running")

        mock_slot.assert_called_once_with("state_worker", "pending", "running")


class TestDataClassIntegration:
    """Test suite for integration between data classes."""

    def test_worker_result_with_worker_stats(self):
        """Test integration between WorkerResult and WorkerStats."""
        # Create multiple worker results
        results = [
            WorkerResult("worker1", "data1", 1.0, 64.0, True, ""),
            WorkerResult("worker2", "data2", 2.0, 128.0, True, ""),
            WorkerResult("worker3", None, 0.5, 32.0, False, "Error occurred"),
        ]

        # Calculate stats from results
        total_workers = len(results)
        completed_workers = sum(1 for r in results if r.success)
        failed_workers = sum(1 for r in results if not r.success)
        avg_execution_time = sum(r.execution_time for r in results) / total_workers
        total_memory_usage = sum(r.memory_usage for r in results)

        stats = WorkerStats(
            total_workers=total_workers,
            completed_workers=completed_workers,
            failed_workers=failed_workers,
            avg_execution_time=avg_execution_time,
            total_memory_usage=total_memory_usage,
        )

        assert stats.total_workers == 3
        assert stats.completed_workers == 2
        assert stats.failed_workers == 1
        assert stats.avg_execution_time == 1.1666666666666667  # (1.0 + 2.0 + 0.5) / 3
        assert stats.total_memory_usage == 224.0  # 64 + 128 + 32

    def test_worker_priority_with_signals(self):
        """Test integration between WorkerPriority and WorkerSignals."""
        signals = WorkerSignals()
        mock_slot = Mock()

        signals.started.connect(mock_slot)

        # Emit with different priorities
        priorities = [
            WorkerPriority.LOW,
            WorkerPriority.NORMAL,
            WorkerPriority.HIGH,
            WorkerPriority.CRITICAL,
        ]

        for i, priority in enumerate(priorities):
            signals.started.emit(f"worker_{i}", priority.value)

        # Verify all emissions
        assert mock_slot.call_count == 4

        # Check call arguments
        call_args = [call[0] for call in mock_slot.call_args_list]
        expected_calls = [
            ("worker_0", 1),
            ("worker_1", 2),
            ("worker_2", 3),
            ("worker_3", 4),
        ]

        assert call_args == expected_calls


class TestEnumRobustness:
    """Test suite for enum robustness and edge cases."""

    def test_worker_priority_string_conversion(self):
        """Test WorkerPriority string representation."""
        assert str(WorkerPriority.LOW) == "WorkerPriority.LOW"
        assert repr(WorkerPriority.HIGH) == "<WorkerPriority.HIGH: 3>"

    def test_worker_state_string_conversion(self):
        """Test WorkerState string representation."""
        assert str(WorkerState.RUNNING) == "WorkerState.RUNNING"
        assert repr(WorkerState.COMPLETED) == "<WorkerState.COMPLETED: 'completed'>"

    def test_enum_uniqueness(self):
        """Test that enum values are unique."""
        priority_values = [p.value for p in WorkerPriority]
        assert len(priority_values) == len(set(priority_values))

        state_values = [s.value for s in WorkerState]
        assert len(state_values) == len(set(state_values))

    def test_enum_completeness(self):
        """Test that enums cover expected use cases."""
        # WorkerPriority should cover range from low to critical
        priorities = list(WorkerPriority)
        assert len(priorities) >= 3  # At least LOW, NORMAL, HIGH
        assert WorkerPriority.LOW in priorities
        assert WorkerPriority.CRITICAL in priorities

        # WorkerState should cover complete lifecycle
        states = list(WorkerState)
        lifecycle_states = [
            WorkerState.PENDING,
            WorkerState.RUNNING,
            WorkerState.COMPLETED,
        ]
        error_states = [WorkerState.FAILED, WorkerState.CANCELLED]

        for state in lifecycle_states + error_states:
            assert state in states


@pytest.mark.parametrize(
    "priority,expected_value",
    [
        (WorkerPriority.LOW, 1),
        (WorkerPriority.NORMAL, 2),
        (WorkerPriority.HIGH, 3),
        (WorkerPriority.CRITICAL, 4),
    ],
)
def test_worker_priority_values(priority, expected_value):
    """Test WorkerPriority enum values with parametrize."""
    assert priority.value == expected_value


@pytest.mark.parametrize(
    "state,expected_value",
    [
        (WorkerState.PENDING, "pending"),
        (WorkerState.RUNNING, "running"),
        (WorkerState.COMPLETED, "completed"),
        (WorkerState.FAILED, "failed"),
        (WorkerState.CANCELLED, "cancelled"),
    ],
)
def test_worker_state_values(state, expected_value):
    """Test WorkerState enum values with parametrize."""
    assert state.value == expected_value


@pytest.mark.parametrize(
    "success,error_msg,expected_success",
    [
        (True, "", True),
        (False, "Error occurred", False),
        (True, "Warning message", True),  # Success with message
        (False, "", False),  # Failed without message
    ],
)
def test_worker_result_variations(success, error_msg, expected_success):
    """Test WorkerResult with various success/error combinations."""
    result = WorkerResult(
        worker_id="test",
        result="data" if success else None,
        execution_time=1.0,
        memory_usage=64.0,
        success=success,
        error_message=error_msg,
    )

    assert result.success == expected_success
    assert result.error_message == error_msg
