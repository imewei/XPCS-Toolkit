"""Unit tests for BUG-009: threading.Event cross-thread cancellation in AverageToolbox.

Tests verify:
1. is_killed is a threading.Event (provides proper memory fence semantics)
2. Setting is_killed from the main thread is visible to worker thread without delay
"""

import threading
import time

import pytest


class TestIsKilledIsThreadingEvent:
    """Test 1: is_killed as threading.Event provides proper cross-thread cancellation."""

    def test_is_killed_is_threading_event_instance(self):
        """is_killed must be a threading.Event, not a plain bool."""
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])

        assert isinstance(toolbox.is_killed, threading.Event), (
            "is_killed must be threading.Event to guarantee cross-thread memory fence. "
            f"Got {type(toolbox.is_killed).__name__} instead."
        )

    def test_is_killed_starts_unset(self):
        """is_killed must start in the unset (not killed) state."""
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])
        assert not toolbox.is_killed.is_set(), (
            "is_killed must start unset — worker should not be pre-killed."
        )

    def test_kill_method_sets_event(self):
        """Calling kill() must set the threading.Event."""
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])
        toolbox.kill()
        assert toolbox.is_killed.is_set(), (
            "kill() must call is_killed.set() to signal cancellation."
        )

    def test_event_provides_memory_fence_semantics(self):
        """threading.Event guarantees that set() is visible to all threads.

        threading.Event internally uses a Condition (which uses a Lock) to
        synchronise state, providing the memory fence that a plain bool lacks.
        This test verifies the class attribute, not just the value.
        """
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])

        # threading.Event exposes these synchronisation primitives
        assert hasattr(toolbox.is_killed, "is_set"), (
            "is_killed.is_set() must exist for non-blocking poll from worker thread."
        )
        assert hasattr(toolbox.is_killed, "set"), (
            "is_killed.set() must exist for main-thread cancellation signal."
        )
        assert hasattr(toolbox.is_killed, "wait"), (
            "is_killed.wait() must exist (confirms threading.Event API)."
        )


class TestCrossThreadCancellationVisibility:
    """Test 2: Setting is_killed from main thread is visible to worker thread."""

    def test_worker_thread_observes_kill_signal(self):
        """Worker thread must observe kill signal set by the main thread.

        This test starts a real thread that polls is_killed.is_set().
        The main thread calls kill() and we assert the worker thread
        sees the change within a tight deadline (50 ms).
        """
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])

        observed_kill = threading.Event()  # set by worker when it sees is_killed

        def worker():
            # Poll until kill signal is seen or we time out
            deadline = time.monotonic() + 2.0  # 2 s safety timeout
            while time.monotonic() < deadline:
                if toolbox.is_killed.is_set():
                    observed_kill.set()
                    return
                time.sleep(0.001)  # 1 ms poll interval

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        # Tiny sleep to let the worker thread enter its poll loop
        time.sleep(0.005)

        # Main thread signals cancellation
        toolbox.kill()

        # Worker thread must observe the event within 50 ms
        observed = observed_kill.wait(timeout=0.05)

        t.join(timeout=0.5)

        assert observed, (
            "Worker thread did not observe kill signal within 50 ms. "
            "A plain bool flag has no memory fence and may not be visible "
            "across threads. threading.Event guarantees visibility."
        )

    def test_kill_is_idempotent(self):
        """Calling kill() multiple times must not raise and event stays set."""
        from xpcsviewer.module.average_toolbox import AverageToolbox

        toolbox = AverageToolbox(work_dir="/tmp", flist=["a.h5", "b.h5"])
        toolbox.kill()
        toolbox.kill()  # second call must not raise
        assert toolbox.is_killed.is_set()

    def test_separate_instances_have_independent_events(self):
        """Two AverageToolbox instances must have independent is_killed events."""
        from xpcsviewer.module.average_toolbox import AverageToolbox

        tb1 = AverageToolbox(work_dir="/tmp", flist=["a.h5"])
        tb2 = AverageToolbox(work_dir="/tmp", flist=["b.h5"])

        tb1.kill()

        assert tb1.is_killed.is_set(), "tb1 should be killed"
        assert not tb2.is_killed.is_set(), (
            "Killing tb1 must not affect tb2 — each instance needs its own Event."
        )
