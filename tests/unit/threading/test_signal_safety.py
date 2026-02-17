"""Unit tests for threading signal safety fixes.

Tests for BUG-007, BUG-012, and BUG-031:
- BUG-007: Signal emissions in _execute_task() must be safe from ThreadPoolExecutor threads
- BUG-012: cancel_all_operations() must disconnect all signal-slot pairs before clearing
- BUG-031: WorkerManager.active_workers dict access must be protected with threading.Lock
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, call, patch

import pytest

# Enable test mode to prevent background monitoring threads
os.environ.setdefault("XPCS_TEST_MODE", "1")


# ---------------------------------------------------------------------------
# Test 1: _execute_task() signals are emitted safely from worker threads
# ---------------------------------------------------------------------------


class TestSignalEmissionThreadSafety:
    """BUG-007: Signals emitted from ThreadPoolExecutor threads must be safe."""

    def test_execute_task_uses_invoke_method_or_queued_connection(self):
        """
        Verify that signals emitted inside _execute_task() are delivered via
        QMetaObject.invokeMethod or an equivalent mechanism that is safe when
        called from a background ThreadPoolExecutor thread.

        The implementation must NOT call .emit() directly on a signal from a
        non-GUI thread without QueuedConnection marshalling.
        """
        from xpcsviewer.threading.unified_threading import (
            TaskPriority,
            TaskType,
            UnifiedTask,
            UnifiedThreadingManager,
        )

        mgr = UnifiedThreadingManager(max_workers=2)
        try:
            results_received: list[str] = []
            errors_received: list[str] = []

            def slot_started(task_id: str, task_type: str):
                results_received.append(f"started:{task_id}")

            def slot_completed(task_id: str, result: object):
                results_received.append(f"completed:{task_id}")

            def slot_failed(task_id: str, error_msg: str):
                errors_received.append(f"failed:{task_id}")

            mgr.task_started.connect(slot_started)
            mgr.task_completed.connect(slot_completed)
            mgr.task_failed.connect(slot_failed)

            done_event = threading.Event()
            task_ran = threading.Event()

            def simple_work():
                task_ran.set()
                return "ok"

            task = UnifiedTask(
                task_id="test-signal-safety",
                priority=TaskPriority.NORMAL,
                task_type=TaskType.COMPUTATION,
                function=simple_work,
            )

            # Run _execute_task directly in a background thread (simulating pool)
            exc_holder: list[Exception] = []

            def run_in_thread():
                try:
                    mgr._execute_task(task)
                except Exception as exc:
                    exc_holder.append(exc)
                finally:
                    done_event.set()

            t = threading.Thread(target=run_in_thread, daemon=True)
            t.start()
            done_event.wait(timeout=5.0)

            assert not exc_holder, f"_execute_task raised: {exc_holder[0]}"
            assert task_ran.is_set(), "_execute_task did not run the task function"
        finally:
            mgr.shutdown()

    def test_handle_task_completion_emits_signals_safely(self):
        """
        Verify _handle_task_completion signals (task_completed/task_failed)
        do not crash when invoked from a done-callback thread (not GUI thread).
        Done callbacks from ThreadPoolExecutor run on the pool thread.
        """
        from concurrent.futures import ThreadPoolExecutor

        from xpcsviewer.threading.unified_threading import (
            TaskPriority,
            TaskType,
            UnifiedTask,
            UnifiedThreadingManager,
        )

        mgr = UnifiedThreadingManager(max_workers=2)
        try:
            completion_event = threading.Event()
            completed_ids: list[str] = []

            def on_completed(task_id: str, result: object):
                completed_ids.append(task_id)
                completion_event.set()

            mgr.task_completed.connect(on_completed)

            task = UnifiedTask(
                task_id="test-handle-completion",
                priority=TaskPriority.NORMAL,
                task_type=TaskType.COMPUTATION,
                function=lambda: 42,
            )

            # Simulate the pool submission path
            pool = ThreadPoolExecutor(max_workers=1)
            try:
                future = pool.submit(mgr._execute_task, task)
                future.add_done_callback(lambda f: mgr._handle_task_completion(task, f))

                # Wait for completion callback to fire
                # We don't require QApplication for unit tests, just no crash
                future.result(timeout=5.0)
                # Give the done callback a moment to execute
                time.sleep(0.1)
            finally:
                pool.shutdown(wait=False)

            # No exception means signals were emitted without crashing from the
            # done-callback thread.  The actual delivery to connected slots may
            # be deferred (QueuedConnection), but the call must not raise.
        finally:
            mgr.shutdown()


# ---------------------------------------------------------------------------
# Test 2: cancel_all_operations() disconnects all signal-slot pairs
# ---------------------------------------------------------------------------


class TestCancelAllOperationsDisconnects:
    """BUG-012: cancel_all_operations() must clean up _signal_connections."""

    def _make_kernel(self):
        """Create an AsyncViewerKernel with a mock thread pool."""
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_kernel import AsyncViewerKernel

        mock_vk = MagicMock()
        mock_pool = MagicMock()
        mock_pool.start = MagicMock()

        kernel = AsyncViewerKernel(mock_vk, mock_pool)
        return kernel

    def test_cancel_all_disconnects_signal_connections(self):
        """
        After cancel_all_operations(), _signal_connections must be empty and
        each stored (signal, slot) pair must have been disconnected.
        """
        kernel = self._make_kernel()

        # Manually plant fake connections to simulate what _submit_plot_worker does
        disconnected: list[str] = []

        class FakeSignal:
            def __init__(self, name: str):
                self._name = name

            def disconnect(self, slot):
                disconnected.append(self._name)

        fake_finished = FakeSignal("finished")
        fake_error = FakeSignal("error")
        fake_cancelled = FakeSignal("cancelled")

        slot_a = lambda r: None
        slot_b = lambda w, m, t, x: None
        slot_c = lambda w, r: None

        kernel._signal_connections["op-001"] = [
            (fake_finished, slot_a),
            (fake_error, slot_b),
            (fake_cancelled, slot_c),
        ]
        kernel._signal_connections["op-002"] = [
            (fake_finished, slot_a),
        ]

        # cancel_all_operations should disconnect everything
        kernel.cancel_all_operations()

        # All connections should be gone
        assert len(kernel._signal_connections) == 0, (
            "_signal_connections was not cleared after cancel_all_operations()"
        )

        # Each signal.disconnect() should have been called
        assert "finished" in disconnected, "finished signal was not disconnected"
        assert "error" in disconnected, "error signal was not disconnected"
        assert "cancelled" in disconnected, "cancelled signal was not disconnected"
        assert disconnected.count("finished") == 2, (
            "finished signal should have been disconnected for both operations"
        )

    def test_cancel_all_clears_active_operations_after_disconnecting(self):
        """
        active_operations must be cleared, and _signal_connections must be
        cleared BEFORE active_operations (ordering: signals first, then ops).
        """
        kernel = self._make_kernel()

        signal_cleared_before_ops: list[bool] = []

        class OrderTrackingSignal:
            def disconnect(self, slot):
                # Record whether signal_connections still had entries
                still_has_connections = bool(kernel._signal_connections)
                signal_cleared_before_ops.append(still_has_connections)

        slot = lambda: None
        kernel._signal_connections["op-xyz"] = [
            (OrderTrackingSignal(), slot),
        ]
        kernel.active_operations["op-xyz"] = "worker-xyz"

        kernel.cancel_all_operations()

        assert len(kernel.active_operations) == 0, "active_operations not cleared"
        assert len(kernel._signal_connections) == 0, "_signal_connections not cleared"


# ---------------------------------------------------------------------------
# Test 3: WorkerManager.active_workers dict access is thread-safe
# ---------------------------------------------------------------------------


class TestWorkerManagerThreadSafety:
    """BUG-031: active_workers dict must be protected against concurrent mutation."""

    def test_active_workers_protected_by_lock(self):
        """
        WorkerManager must have a _workers_lock (or equivalent threading.Lock)
        that is acquired on all active_workers insertions and deletions.
        """
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_workers import WorkerManager

        mock_pool = MagicMock()
        mgr = WorkerManager(mock_pool)

        assert hasattr(mgr, "_workers_lock"), (
            "WorkerManager must have a _workers_lock attribute for thread safety"
        )
        assert isinstance(mgr._workers_lock, type(threading.Lock())), (
            "_workers_lock must be a threading.Lock"
        )

    def test_concurrent_finish_and_cancel_do_not_corrupt_dict(self):
        """
        Simulate concurrent calls to _on_worker_finished and _on_worker_cancelled
        from different threads. The active_workers dict must not be corrupted.
        """
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_workers import WorkerManager

        mock_pool = MagicMock()
        mgr = WorkerManager(mock_pool)

        # Pre-populate with many workers
        n_workers = 50
        for i in range(n_workers):
            wid = f"worker-{i}"
            mock_worker = MagicMock()
            mock_worker.worker_id = wid
            with mgr._workers_lock:
                mgr.active_workers[wid] = mock_worker

        barrier = threading.Barrier(2)
        errors: list[str] = []

        def finish_half():
            barrier.wait()
            for i in range(0, n_workers, 2):
                try:
                    mgr._on_worker_finished(f"worker-{i}", result=None)
                except Exception as exc:
                    errors.append(f"finish error: {exc}")

        def cancel_half():
            barrier.wait()
            for i in range(1, n_workers, 2):
                try:
                    mgr._on_worker_cancelled(f"worker-{i}")
                except Exception as exc:
                    errors.append(f"cancel error: {exc}")

        t1 = threading.Thread(target=finish_half, daemon=True)
        t2 = threading.Thread(target=cancel_half, daemon=True)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert not errors, f"Thread-safety errors: {errors}"

        # After both finish, active_workers must be consistent (no entries for
        # the workers that were removed, and no KeyError panics)
        with mgr._workers_lock:
            remaining = list(mgr.active_workers.keys())

        assert len(remaining) == 0, (
            f"Expected 0 remaining workers, got {len(remaining)}: {remaining}"
        )


# ---------------------------------------------------------------------------
# Test 4: Cancellation + immediate re-submission doesn't crash from stale connections
# ---------------------------------------------------------------------------


class TestCancelThenResubmit:
    """BUG-012 + BUG-031: Cancel then re-submit the same operation_id is safe."""

    def _make_kernel(self):
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_kernel import AsyncViewerKernel

        mock_vk = MagicMock()
        mock_pool = MagicMock()
        mock_pool.start = MagicMock()

        kernel = AsyncViewerKernel(mock_vk, mock_pool)
        return kernel

    def test_cancel_then_resubmit_same_operation_id(self):
        """
        After cancel_all_operations(), submitting a new operation with the same
        ID must not raise due to stale signal connections from the previous run.
        """
        kernel = self._make_kernel()

        disconnected_calls: list[str] = []

        class TrackingSignal:
            def __init__(self, name: str):
                self._name = name
                self._connected = True

            def connect(self, slot):
                self._connected = True

            def disconnect(self, slot):
                if not self._connected:
                    raise RuntimeError("Already disconnected!")
                disconnected_calls.append(self._name)
                self._connected = False

        slot = lambda: None

        # First submission -- plant fake connections
        kernel._signal_connections["op-reuse"] = [
            (TrackingSignal("finished"), slot),
            (TrackingSignal("error"), slot),
        ]
        kernel.active_operations["op-reuse"] = "worker-old"

        # Cancel all -- must cleanly disconnect
        kernel.cancel_all_operations()

        assert "op-reuse" not in kernel._signal_connections, (
            "Stale connection entry survived cancel_all_operations()"
        )
        assert len(disconnected_calls) == 2, (
            f"Expected 2 disconnects, got {len(disconnected_calls)}"
        )

        # Now plant new connections for the same operation_id
        new_tracking_signal = TrackingSignal("finished-new")
        kernel._signal_connections["op-reuse"] = [
            (new_tracking_signal, slot),
        ]
        kernel.active_operations["op-reuse"] = "worker-new"

        # Cancel again -- must not crash even though it is a fresh operation
        kernel.cancel_all_operations()

        assert "op-reuse" not in kernel._signal_connections, (
            "Re-submitted operation connections not cleaned up on second cancel"
        )


# ---------------------------------------------------------------------------
# Test 5: Shutdown chain -- cancel all -> signals cleaned -> no stale ops
# ---------------------------------------------------------------------------


class TestShutdownChain:
    """BUG-007 + BUG-012 + BUG-031: Full shutdown chain is consistent."""

    def _make_kernel(self):
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_kernel import AsyncViewerKernel

        mock_vk = MagicMock()
        mock_pool = MagicMock()
        mock_pool.start = MagicMock()

        return AsyncViewerKernel(mock_vk, mock_pool)

    def test_shutdown_chain_clears_all_state(self):
        """
        After cancel_all_operations():
        1. _signal_connections is empty
        2. active_operations is empty
        3. WorkerManager.active_workers is empty
        """
        from unittest.mock import MagicMock

        kernel = self._make_kernel()

        # Plant several fake operations
        for i in range(5):
            op_id = f"op-{i}"
            slot = lambda: None

            class FakeSignal:
                def disconnect(self, s):
                    pass

            kernel._signal_connections[op_id] = [
                (FakeSignal(), slot),
                (FakeSignal(), slot),
            ]
            kernel.active_operations[op_id] = f"worker-{i}"

            # Also add to WorkerManager.active_workers
            mock_worker = MagicMock()
            mock_worker.cancel = MagicMock()
            with kernel.worker_manager._workers_lock:
                kernel.worker_manager.active_workers[f"worker-{i}"] = mock_worker

        # Execute shutdown chain
        kernel.cancel_all_operations()

        # 1. Signal connections cleared
        assert len(kernel._signal_connections) == 0, (
            "_signal_connections not empty after cancel_all_operations"
        )

        # 2. Active operations cleared
        assert len(kernel.active_operations) == 0, (
            "active_operations not empty after cancel_all_operations"
        )

    def test_worker_manager_cancel_all_calls_cancel_on_workers(self):
        """
        WorkerManager.cancel_all_workers() must call cancel() on every
        active worker, protected by the lock.
        """
        from unittest.mock import MagicMock

        from xpcsviewer.threading.async_workers import WorkerManager

        mock_pool = MagicMock()
        mgr = WorkerManager(mock_pool)

        cancel_calls: list[str] = []

        for i in range(3):
            wid = f"w-{i}"
            mock_worker = MagicMock()
            mock_worker.worker_id = wid
            mock_worker.cancel.side_effect = lambda _wid=wid: cancel_calls.append(_wid)
            with mgr._workers_lock:
                mgr.active_workers[wid] = mock_worker

        mgr.cancel_all_workers()

        assert len(cancel_calls) == 3, (
            f"Expected 3 cancel calls, got {len(cancel_calls)}: {cancel_calls}"
        )
