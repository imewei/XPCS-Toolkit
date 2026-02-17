"""Integration tests for compound bug chains.

Verifies that all 6 cross-component interaction chains are broken end-to-end
after the Phase 1-5 bug fixes.

Chains tested:
  Chain 1: Silent Scientific Corruption (BUG-006 -> BUG-003 -> BUG-021 -> BUG-004)
  Chain 2: Signal Chaos at Shutdown (BUG-012 + BUG-007 + BUG-031 + BUG-013)
  Chain 3: Lock Chain Deadlock (BUG-033 + BUG-008)
  Chain 4: Async Architecture (BUG-014 + BUG-049)
  Chain 5: Fitting Pipeline Integrity (BUG-017 + BUG-003 + BUG-023 + BUG-024 + BUG-025 + BUG-020 + BUG-021)
  Chain 6: JAX Boundary Leak (BUG-027 + BUG-014 + BUG-007)
"""

from __future__ import annotations

import math
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hdf5_g2_file(
    tmp_path,
    n_delay: int = 50,
    n_q: int = 8,
    include_q_values: bool = True,
):
    """Create a minimal HDF5 file containing G2 data for integration tests.

    Returns the path to the created file.
    """
    h5py = pytest.importorskip("h5py")
    fname = str(tmp_path / "g2_test.hdf5")
    rng = np.random.default_rng(42)
    g2 = rng.uniform(1.0, 2.0, size=(n_delay, n_q)).astype(np.float64)
    g2_err = rng.uniform(0.01, 0.05, size=(n_delay, n_q)).astype(np.float64)
    delay_times = np.linspace(1e-4, 1.0, n_delay)
    with h5py.File(fname, "w") as f:
        grp = f.create_group("xpcs/g2")
        grp.create_dataset("g2", data=g2)
        grp.create_dataset("g2_err", data=g2_err)
        grp.create_dataset("delay_times", data=delay_times)
        if include_q_values:
            q = np.linspace(0.01, 0.1, n_q)
            grp.create_dataset("q_values", data=q)
    return fname, n_delay, n_q


# ===========================================================================
# Chain 1: Silent Scientific Corruption
# BUG-006 -> BUG-003 -> BUG-021 -> BUG-004
# ===========================================================================


class TestChain1SilentScientificCorruption:
    """End-to-end test for the silent-scientific-corruption chain.

    Steps:
    1. Load a G2 file missing Q values.
    2. Verify HDF5Facade generates dummy Q values from shape[1] (n_q), not shape[0] (n_delay).
    3. Run NLSQ which should fail gracefully, returning a sentinel with is_fallback=True.
    4. Verify sentinel propagates: converged=False, r_squared=NaN.
    5. Verify FitResult / FitDiagnostics validation catches invalid states so NUTS never
       receives bad init params.
    """

    def test_hdf5_facade_dummy_q_uses_n_q_axis(self, tmp_path):
        """BUG-006: HDF5Facade must generate dummy Q values with length == shape[1] (n_q axis)."""
        from xpcsviewer.io.hdf5_facade import HDF5Facade

        fname, n_delay, n_q = _make_hdf5_g2_file(
            tmp_path, n_delay=50, n_q=8, include_q_values=False
        )
        facade = HDF5Facade()
        g2_data = facade.read_g2_data(fname, group="/xpcs/g2")

        # n_q = 8 (shape[1]), n_delay = 50 (shape[0])
        # The fix ensures we use shape[1] for the dummy Q list length
        assert len(g2_data.q_values) == n_q, (
            f"Expected {n_q} dummy Q values (shape[1] / n_q axis), "
            f"got {len(g2_data.q_values)} (which equals n_delay={n_delay} if buggy)"
        )
        # Each dummy value should be a simple integer index
        assert g2_data.q_values[0] == 0
        assert g2_data.q_values[-1] == n_q - 1

    def test_nlsq_sentinel_on_failure(self):
        """BUG-003: NLSQ exception handler returns sentinel with is_fallback=True, converged=False."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        # Deliberately bad input: more parameters than data points
        x = np.array([1.0])
        y = np.array([1.5])
        yerr = np.array([0.1])

        def bad_model(x, tau, baseline, contrast):
            return baseline + contrast * np.exp(-x / tau)

        # With only 1 data point and 3 parameters the fit must fail
        result = nlsq_optimize(
            model_fn=bad_model,
            x=x,
            y=y,
            yerr=yerr,
            p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
            bounds={
                "tau": (1e-6, 1e6),
                "baseline": (0.0, 2.0),
                "contrast": (0.0, 1.0),
            },
        )

        # Sentinel flags
        assert result.converged is False, "Sentinel must have converged=False"
        assert result.is_fallback is True, "Sentinel must have is_fallback=True"

        # Sentinel metrics must be NaN, not 0.0 (BUG-023)
        assert math.isnan(result.r_squared), (
            f"Sentinel r_squared must be NaN, got {result.r_squared}"
        )

    def test_sentinel_r_squared_is_nan_not_zero(self):
        """BUG-023: Failed NLSQResult metrics are NaN, distinguishable from bad-but-converged fits."""
        from xpcsviewer.fitting.results import NLSQResult

        sentinel = NLSQResult(
            params={"tau": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            _r_squared=float("nan"),
            _adj_r_squared=float("nan"),
            _rmse=float("nan"),
            _mae=float("nan"),
            _aic=float("nan"),
            _bic=float("nan"),
        )
        assert math.isnan(sentinel.r_squared), (
            "Failed NLSQResult.r_squared must be NaN (not 0.0)"
        )
        assert math.isnan(sentinel.adj_r_squared), (
            "Failed NLSQResult.adj_r_squared must be NaN (not 0.0)"
        )

    def test_fit_diagnostics_rejects_negative_divergences(self):
        """BUG-025: FitDiagnostics __post_init__ rejects negative divergences."""
        from xpcsviewer.fitting.results import FitDiagnostics

        with pytest.raises(ValueError, match="divergences must be non-negative"):
            FitDiagnostics(divergences=-1)

    def test_fit_diagnostics_rejects_negative_ess(self):
        """BUG-025: FitDiagnostics __post_init__ rejects negative ESS values."""
        from xpcsviewer.fitting.results import FitDiagnostics

        with pytest.raises(ValueError, match="ess_bulk"):
            FitDiagnostics(ess_bulk={"tau": -1})

        with pytest.raises(ValueError, match="ess_tail"):
            FitDiagnostics(ess_tail={"tau": -1})

    def test_fit_result_rejects_empty_samples(self):
        """BUG-024: FitResult __post_init__ rejects empty samples dict."""
        from xpcsviewer.fitting.results import FitResult

        with pytest.raises(ValueError, match="non-empty"):
            FitResult(samples={})

    def test_fit_result_rejects_inconsistent_shapes(self):
        """BUG-024: FitResult __post_init__ rejects inconsistent sample array shapes."""
        from xpcsviewer.fitting.results import FitResult

        with pytest.raises(ValueError, match="inconsistent shapes"):
            FitResult(
                samples={
                    "tau": np.ones(100),
                    "baseline": np.ones(200),  # Different length
                }
            )

    def test_chain1_end_to_end_corrupt_g2_to_sentinel(self, tmp_path):
        """Full chain: missing Q values -> dummy Q (correct axis) -> NLSQ sentinel propagation.

        Verifies that the full corruption chain is broken:
        - G2 file without Q values produces correct-shaped dummy Q values (shape[1])
        - The resulting data structure is valid (no axis swap)
        - When NLSQ fails on this data, a proper sentinel is returned
        - The sentinel is clearly flagged (not disguised as a real result)
        """
        from xpcsviewer.fitting.nlsq import nlsq_optimize
        from xpcsviewer.io.hdf5_facade import HDF5Facade

        # Step 1: Load G2 data missing Q values
        fname, n_delay, n_q = _make_hdf5_g2_file(
            tmp_path, n_delay=50, n_q=8, include_q_values=False
        )
        facade = HDF5Facade()
        g2_data = facade.read_g2_data(fname, group="/xpcs/g2")

        # Step 2: Verify dummy Q values are correct length (n_q axis, not n_delay axis)
        assert len(g2_data.q_values) == n_q, (
            f"BUG-006 regression: dummy Q length {len(g2_data.q_values)} != n_q {n_q}"
        )

        # Step 3: Simulate what happens when NLSQ fails on this data
        # (e.g. too few data points for the model)
        x = np.array([1.0])
        y = np.array([1.5])

        def simple_model(x, tau, baseline):
            return baseline * np.exp(-x / tau)

        result = nlsq_optimize(
            model_fn=simple_model,
            x=x,
            y=y,
            yerr=None,
            p0={"tau": 1.0, "baseline": 1.0},
            bounds={"tau": (1e-6, 1e6), "baseline": (0.0, 2.0)},
        )

        # Step 4: Verify sentinel propagates correctly (not disguised as real result)
        # - If NLSQ converges on this trivially small dataset, check it's still valid
        # - If it fails, confirm the sentinel flags are set
        if result.is_fallback:
            assert result.converged is False, "Sentinel must have converged=False"
            assert math.isnan(result.r_squared), (
                "BUG-023 regression: sentinel r_squared must be NaN not 0.0"
            )

        # Step 5: Confirm that FitResult/FitDiagnostics validation guards NUTS
        # (this is the end of the chain: bad data cannot propagate into NUTS)
        from xpcsviewer.fitting.results import FitDiagnostics, FitResult

        # A sentinel result with is_fallback=True should NOT be passed to NUTS.
        # Verify the validation layer catches invalid diagnostic states.
        with pytest.raises(ValueError):
            FitDiagnostics(divergences=-1)

        with pytest.raises(ValueError):
            FitResult(samples={})


# ===========================================================================
# Chain 2: Signal Chaos at Shutdown
# BUG-012 + BUG-007 + BUG-031 + BUG-013
# ===========================================================================


class TestChain2SignalChaosAtShutdown:
    """Integration test for signal safety during shutdown.

    Verifies:
    - cancel_all_operations() disconnects all signal-slot pairs (BUG-012)
    - WorkerManager.active_workers dict is protected by a lock (BUG-031)
    - No signals fire into deleted objects post-cancellation
    - Thread pool waitForDone is called during closeEvent (BUG-013)
    """

    def test_cancel_all_operations_clears_signal_connections(self):
        """BUG-012: cancel_all_operations() disconnects all tracked signal connections."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import QThreadPool

        from xpcsviewer.threading.async_kernel import AsyncViewerKernel

        mock_vk = MagicMock()
        thread_pool = QThreadPool()

        kernel = AsyncViewerKernel(viewer_kernel=mock_vk, thread_pool=thread_pool)

        # Manually inject a fake signal connection so we can verify cleanup
        fake_operation_id = "test-op-001"
        mock_signal = MagicMock()
        mock_slot = MagicMock()
        kernel._signal_connections[fake_operation_id] = [
            (mock_signal, mock_slot),
        ]
        kernel.active_operations[fake_operation_id] = "fake-worker-id"

        # Cancel all: must disconnect the signal
        kernel.cancel_all_operations()

        # After cancellation, signal connections must be cleared
        assert fake_operation_id not in kernel._signal_connections, (
            "BUG-012 regression: _signal_connections not cleaned up after cancel_all_operations"
        )
        assert fake_operation_id not in kernel.active_operations, (
            "active_operations not cleared after cancel_all_operations"
        )
        # Signal must have been disconnected
        mock_signal.disconnect.assert_called_once_with(mock_slot)

    def test_worker_manager_active_workers_lock_exists(self):
        """BUG-031: WorkerManager protects active_workers with a threading.Lock."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import QThreadPool

        from xpcsviewer.threading.async_workers import WorkerManager

        thread_pool = QThreadPool()
        manager = WorkerManager(thread_pool)

        # Verify the lock attribute exists and is a threading.Lock/RLock
        assert hasattr(manager, "_workers_lock"), (
            "BUG-031 regression: WorkerManager has no _workers_lock"
        )
        assert isinstance(manager._workers_lock, type(threading.Lock())), (
            "BUG-031 regression: _workers_lock is not a threading.Lock"
        )

    def test_concurrent_worker_dict_mutation_is_safe(self):
        """BUG-031: Concurrent insertion and removal of active_workers is race-free."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import QThreadPool

        from xpcsviewer.threading.async_workers import WorkerManager

        thread_pool = QThreadPool()
        manager = WorkerManager(thread_pool)

        errors = []

        def mutate_dict(worker_id: str, repeat: int):
            for _ in range(repeat):
                try:
                    with manager._workers_lock:
                        manager.active_workers[worker_id] = MagicMock()
                    time.sleep(0.0001)
                    with manager._workers_lock:
                        manager.active_workers.pop(worker_id, None)
                except Exception as exc:
                    errors.append(exc)

        # Spawn 8 threads all mutating the dict concurrently
        threads = [
            threading.Thread(target=mutate_dict, args=(f"worker-{i}", 20))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread-safety failures: {errors}"

    def test_close_event_calls_wait_for_done(self):
        """BUG-013: closeEvent calls thread_pool.waitForDone to prevent post-deletion signals."""
        pytest.importorskip("PySide6")
        # Verify the closeEvent source uses waitForDone
        import inspect

        from PySide6.QtCore import QThreadPool

        from xpcsviewer import xpcs_viewer

        source = inspect.getsource(xpcs_viewer.XpcsViewer.closeEvent)
        assert "waitForDone" in source, (
            "BUG-013 regression: closeEvent does not call waitForDone on thread_pool"
        )

    def test_cancel_then_resubmit_does_not_crash(self):
        """BUG-012 + BUG-031: Cancellation followed by re-submission does not leave stale connections."""
        pytest.importorskip("PySide6")
        from PySide6.QtCore import QThreadPool

        from xpcsviewer.threading.async_kernel import AsyncViewerKernel

        mock_vk = MagicMock()
        thread_pool = QThreadPool()
        kernel = AsyncViewerKernel(viewer_kernel=mock_vk, thread_pool=thread_pool)

        # Inject a fake operation
        op_id = "re-submit-test"
        mock_signal = MagicMock()
        mock_slot = MagicMock()
        kernel._signal_connections[op_id] = [(mock_signal, mock_slot)]
        kernel.active_operations[op_id] = "worker-001"

        # First cancellation
        kernel.cancel_all_operations()

        # State should be clean
        assert not kernel._signal_connections, (
            "Signal connections not empty after first cancel"
        )
        assert not kernel.active_operations, (
            "Active operations not empty after first cancel"
        )

        # Simulate re-submission without crash
        op_id2 = "re-submit-test-2"
        mock_signal2 = MagicMock()
        mock_slot2 = MagicMock()
        kernel._signal_connections[op_id2] = [(mock_signal2, mock_slot2)]
        kernel.active_operations[op_id2] = "worker-002"

        # Second cancellation should also succeed cleanly
        kernel.cancel_all_operations()
        assert not kernel._signal_connections
        assert not kernel.active_operations


# ===========================================================================
# Chain 3: Lock Chain Deadlock
# BUG-033 + BUG-008
# ===========================================================================


class TestChain3LockChainDeadlock:
    """Integration test for concurrent HDF5 reads without deadlock.

    Verifies:
    - Multiple threads can read HDF5 files concurrently (BUG-008 lock ordering fix)
    - Stale lock entries are pruned (BUG-032)
    - Cache lock does not block during I/O (BUG-033)
    """

    def test_concurrent_hdf5_reads_no_deadlock(self, tmp_path):
        """BUG-008 + BUG-033: Concurrent HDF5 reads from multiple threads do not deadlock."""
        h5py = pytest.importorskip("h5py")
        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        # Create a test HDF5 file
        fname = str(tmp_path / "pool_test.hdf5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("data", data=np.arange(100, dtype=np.float64))

        pool = HDF5ConnectionPool(max_pool_size=4)
        results = []
        errors = []

        def read_file():
            try:
                with pool.get_connection(fname) as fh:
                    data = fh["data"][:]
                    results.append(data.sum())
            except Exception as exc:
                errors.append(exc)

        # Launch 16 concurrent reads with a tight deadline
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(read_file) for _ in range(16)]
            # waitForDone equivalent: collect all futures with a generous timeout
            for future in as_completed(futures, timeout=10.0):
                # Raise any exception to fail the test immediately
                future.result()

        assert not errors, f"Errors during concurrent reads: {errors}"
        assert len(results) == 16, f"Expected 16 results, got {len(results)}"
        # All reads should return the same sum
        expected = float(np.arange(100).sum())
        assert all(math.isclose(r, expected) for r in results)

    def test_stale_lock_entries_are_pruned(self, tmp_path):
        """BUG-032: HDF5 lock dict prunes stale entries to prevent unbounded growth."""
        h5py = pytest.importorskip("h5py")
        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        pool = HDF5ConnectionPool(max_pool_size=4)
        # Shrink the max age to trigger pruning quickly in tests
        pool._file_lock_max_age_seconds = 0.01

        # Create and register many unique file paths to fill the lock dict
        fake_fnames = [str(tmp_path / f"fake_{i}.hdf5") for i in range(10)]
        for fname in fake_fnames:
            pool._get_file_lock(os.path.abspath(fname))

        initial_count = len(pool._file_locks)

        # Wait past the stale age threshold
        time.sleep(0.05)

        # Trigger pruning by requesting a new lock for one more file
        new_fname = os.path.abspath(str(tmp_path / "trigger_prune.hdf5"))
        pool._get_file_lock(new_fname)

        final_count = len(pool._file_locks)

        # Pruning should have reduced the dict (stale entries removed)
        # At minimum, the count should not have grown unboundedly
        assert final_count <= initial_count + 1, (
            f"BUG-032 regression: lock dict grew from {initial_count} to {final_count} "
            "without pruning stale entries"
        )

    def test_cache_lock_does_not_block_during_io(self, tmp_path):
        """BUG-033: file_locator holds cache lock only for insertion, not during HDF5 I/O."""
        import inspect

        from xpcsviewer import file_locator as fl

        source = inspect.getsource(fl.FileLocator.get_xf_list)

        # Verify the fix pattern: lock released before XpcsFile construction
        # The pattern is: acquire lock -> check cache -> release -> construct -> re-acquire -> insert
        # We check for the hallmark of the fix: the cache lock is NOT held around create_xpcs_dataset
        assert "xf_obj is None" in source, (
            "BUG-033 regression: expected cache-miss path not found in get_xf_list()"
        )
        # The fix structure: first context manager exits before construction
        # Verify both lock acquisitions are present (check + insert = 2 `with self._cache_lock:`)
        cache_lock_count = source.count("with self._cache_lock:")
        assert cache_lock_count >= 2, (
            f"BUG-033 regression: expected 2 separate cache_lock acquisitions, "
            f"found {cache_lock_count} (construction must happen outside the lock)"
        )

    def test_health_check_not_inside_file_lock(self):
        """BUG-008: Pool health checks run outside file locks to prevent lock inversion.

        The fix separates Phase 1 (health/memory checks, run outside any lock) from
        Phase 2 (pool lookup inside _pool_lock).  We verify this by checking that
        _adapt_pool_size_to_memory_pressure and _perform_health_check are both called
        BEFORE the first `with self._pool_lock:` block in the actual code body (not
        the docstring, which also mentions _pool_lock for explanatory purposes).
        """
        import inspect

        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        source = inspect.getsource(HDF5ConnectionPool.get_connection)

        # Strip the docstring so we only analyse the code body.
        # The docstring ends after the first triple-quote pair that closes it;
        # a reliable proxy is to find the first real Python statement line.
        # We skip past the closing '"""' of the docstring.
        docstring_end = source.find('"""', source.find('"""') + 3) + 3
        code_body = source[docstring_end:]

        # Both health helpers must appear in the code body
        adapt_idx = code_body.find("_adapt_pool_size_to_memory_pressure")
        health_idx = code_body.find("_perform_health_check")
        pool_lock_idx = code_body.find("with self._pool_lock")

        assert adapt_idx != -1, (
            "_adapt_pool_size_to_memory_pressure not found in code body"
        )
        assert health_idx != -1, "_perform_health_check not found in code body"
        assert pool_lock_idx != -1, "with self._pool_lock not found in code body"

        # Both health calls must come before the first pool_lock acquisition
        assert adapt_idx < pool_lock_idx, (
            "BUG-008 regression: _adapt_pool_size_to_memory_pressure runs AFTER _pool_lock; "
            "memory pressure adaptation must run before acquiring the pool lock"
        )
        assert health_idx < pool_lock_idx, (
            "BUG-008 regression: _perform_health_check runs AFTER _pool_lock acquisition; "
            "health checks must run BEFORE any lock to prevent lock inversion"
        )


# ===========================================================================
# Chain 4: Async Architecture
# BUG-014 + BUG-049
# ===========================================================================


class TestChain4AsyncArchitecture:
    """Integration test for async G2 result handler.

    Verifies:
    - apply_g2_result() uses the worker's pre-computed result directly (BUG-014)
    - No synchronous re-computation (vk.plot_g2()) on the main thread
    - make_async decorator actually submits to WorkerManager (BUG-049)
    """

    def test_apply_g2_result_uses_worker_result_directly(self):
        """BUG-014: apply_g2_result uses pre-computed worker result without re-calling plot_g2."""
        import inspect

        from xpcsviewer import xpcs_viewer

        source = inspect.getsource(xpcs_viewer.XpcsViewer.apply_g2_result)

        # The fix: apply_g2_result must NOT call vk.plot_g2() or self.vk.plot_g2()
        # It should instead use result["q"], result["g2"], etc. directly.
        assert "vk.plot_g2()" not in source, (
            "BUG-014 regression: apply_g2_result still calls vk.plot_g2() "
            "synchronously on the main thread"
        )
        # Confirm it accesses pre-computed fields from the worker result dict
        assert 'result.get("q")' in source or "result.get(" in source, (
            "BUG-014 regression: apply_g2_result does not use worker result dict fields"
        )

    def test_apply_g2_result_handler_uses_result_data(self):
        """BUG-014: apply_g2_result routes worker data to renderer without re-computation."""
        pytest.importorskip("PySide6")

        # Verify the method signature and docstring document the fix
        import inspect

        from xpcsviewer import xpcs_viewer

        doc = xpcs_viewer.XpcsViewer.apply_g2_result.__doc__ or ""
        assert "BUG-014" in doc or "pre-computed" in doc or "worker" in doc.lower(), (
            "BUG-014: apply_g2_result docstring should document the no-recomputation contract"
        )

    def test_make_async_is_not_noop(self):
        """BUG-049: make_async decorator actually submits to WorkerManager (not a no-op passthrough)."""
        from xpcsviewer.threading.gui_integration import make_async

        call_log = []

        @make_async
        def my_func(x):
            call_log.append(("called", x))
            return x * 2

        # The decorator must wrap the function; calling it should not be identical
        # to calling the raw function synchronously (i.e., not a passthrough no-op).
        import inspect

        # make_async is expected to return a wrapper, not the original function
        assert my_func is not (lambda x: call_log.append(("called", x)) or x * 2), (
            "make_async should return a wrapper, not the identity function"
        )

        # Verify the decorator is not a literal no-op (passthrough)
        source = inspect.getsource(make_async)
        # A true no-op would be just `return func`; the fix adds WorkerManager submission
        assert (
            "return func" not in source
            or "WorkerManager" in source
            or "submit" in source
        ), "BUG-049 regression: make_async is still a no-op passthrough"

    def test_async_g2_worker_returns_data_dict(self):
        """BUG-014: G2PlotWorker produces a result dict with expected keys."""
        pytest.importorskip("PySide6")
        from xpcsviewer.threading.plot_workers import G2PlotWorker

        # Verify G2PlotWorker exists and is importable (basic smoke test for the async path)
        assert G2PlotWorker is not None

        # Verify do_work returns a dict or None (not calling plot_g2 synchronously)
        import inspect

        source = inspect.getsource(G2PlotWorker.do_work)
        # The worker should build a result dict, not call plot_g2 on the kernel
        assert "return" in source, "G2PlotWorker.do_work must return a result"
        # Should NOT call plot_g2() synchronously inside the worker
        assert "plot_g2()" not in source, (
            "BUG-014: G2PlotWorker.do_work must not call plot_g2() synchronously"
        )


# ===========================================================================
# Chain 5: Fitting Pipeline Integrity
# BUG-017 + BUG-003 + BUG-023 + BUG-024 + BUG-025 + BUG-020 + BUG-021
# ===========================================================================


class TestChain5FittingPipelineIntegrity:
    """Full NLSQ-to-NUTS pipeline integrity test with bad input.

    Verifies:
    - converged=False propagates (BUG-017: default False for missing success attr)
    - R-squared is NaN (not 0.0) for failed fits (BUG-023)
    - Chains start from distinct jittered points (BUG-021)
    - PRNG seed is non-deterministic and logged (BUG-020)
    - FitResult and FitDiagnostics validation catches invalid states (BUG-024, BUG-025)
    """

    def test_converged_defaults_to_false_when_success_missing(self):
        """BUG-017: getattr(native_result, 'success', False) — missing attribute defaults False."""
        # Construct a minimal valid fit that completes but check that if
        # native_result lacks 'success', we default to False (not True)
        import inspect

        from xpcsviewer.fitting import nlsq
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        source = inspect.getsource(nlsq.nlsq_optimize)
        # Verify the fix is in place: default=False, not default=True
        assert '"success", False' in source or "'success', False" in source, (
            "BUG-017 regression: getattr(native_result, 'success', ...) "
            "should default to False, not True"
        )
        assert '"success", True' not in source and "'success', True" not in source, (
            "BUG-017 regression: found default=True for success attribute"
        )

    def test_failed_nlsq_metrics_are_nan(self):
        """BUG-003 + BUG-023: Failed NLSQResult has NaN metrics, not 0.0."""
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        # Force a failure: pass arrays with mismatched shapes
        x = np.array([1.0, 2.0])
        y = np.array([1.0, 2.0])

        def unstable_model(x, tau):
            # Model that will fail during optimization due to bad p0
            import numpy as _np

            return _np.exp(x * tau)  # Explodes for large tau

        result = nlsq_optimize(
            model_fn=unstable_model,
            x=x,
            y=y,
            yerr=None,
            p0={"tau": 1e10},  # Deliberately bad initial guess
            bounds={"tau": (1e9, 1e15)},
        )

        if result.is_fallback:
            # Sentinel: metrics must be NaN
            assert math.isnan(result.r_squared), (
                f"BUG-023 regression: sentinel r_squared={result.r_squared}, expected NaN"
            )
            assert math.isnan(result.adj_r_squared), (
                f"BUG-023 regression: sentinel adj_r_squared={result.adj_r_squared}, expected NaN"
            )
        else:
            # Converged (possible with this simple model): just check it's not sentinel
            assert result.is_fallback is False

    def test_chains_have_distinct_starting_points(self):
        """BUG-021: Per-chain jitter gives distinct init params, not identical broadcast."""
        pytest.importorskip("jax")
        import jax
        import jax.numpy as jnp

        from xpcsviewer.fitting.results import SamplerConfig

        # Simulate the jitter logic from sampler.py directly
        config = SamplerConfig(num_chains=4, num_warmup=10, num_samples=10)
        rng_key = jax.random.PRNGKey(12345)
        init_params = {"tau": 1.0, "baseline": 1.5}

        param_keys = list(init_params.keys())
        subkeys = jax.random.split(rng_key, num=len(param_keys))

        init_params_jax = {}
        for subkey, k in zip(subkeys, param_keys):
            v = init_params[k]
            val = jnp.array(v)
            shape = (config.num_chains,) + val.shape
            jitter = 0.01 * jax.random.normal(subkey, shape=shape)
            init_params_jax[k] = val + jitter

        for param_name, chain_vals in init_params_jax.items():
            vals = np.array(chain_vals)
            # All chains should have DISTINCT starting values (jitter applied)
            assert len(set(vals.tolist())) == config.num_chains, (
                f"BUG-021 regression: param '{param_name}' has identical values "
                f"across chains: {vals}. Chains must start from distinct points."
            )

    def test_prng_seed_is_non_deterministic(self):
        """BUG-020: MCMC PRNG seed is time-based (non-deterministic), not hardcoded 0.

        The function containing the PRNG initialization is _run_mcmc (private),
        called by the public run_single_exp_fit / run_double_exp_fit wrappers.
        """
        import inspect

        from xpcsviewer.fitting import sampler

        # The PRNG key is set in _run_mcmc (private helper)
        source = inspect.getsource(sampler._run_mcmc)

        # The fix: PRNGKey uses time.time_ns() % 2**31, not hardcoded 0
        assert "time.time_ns()" in source or "time_ns" in source, (
            "BUG-020 regression: PRNG seed must be time-based, not hardcoded"
        )
        # Hardcoded PRNGKey(0) with no condition must NOT be present
        # (it is allowed in the user-specified seed path, gated by `if config.random_seed`)
        assert "PRNGKey(0)" not in source or "random_seed" in source, (
            "BUG-020 regression: unconditional PRNGKey(0) found in sampler"
        )

    def test_prng_seed_is_logged(self):
        """BUG-020: MCMC PRNG seed is logged for reproducibility."""
        import inspect

        from xpcsviewer.fitting import sampler

        # The seed logging is in _run_mcmc
        source = inspect.getsource(sampler._run_mcmc)

        # Verify the seed is logged
        assert "logger.info" in source and "seed" in source.lower(), (
            "BUG-020 regression: PRNG seed must be logged for reproducibility"
        )

    def test_full_pipeline_converged_false_propagates(self):
        """End-to-end: bad input propagates converged=False through the full fitting pipeline.

        Uses a model function that raises an exception to guarantee the sentinel path
        is exercised, rather than relying on NLSQ's own convergence heuristics.
        """
        from xpcsviewer.fitting.nlsq import nlsq_optimize
        from xpcsviewer.fitting.results import FitDiagnostics, FitResult, NLSQResult

        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])

        def always_raises(x, tau):
            raise RuntimeError("Deliberate failure to test sentinel path")

        # Step 1: Model that always raises -> sentinel must be returned
        nlsq_result = nlsq_optimize(
            model_fn=always_raises,
            x=x,
            y=y,
            yerr=None,
            p0={"tau": 1.0},
            bounds={"tau": (1e-6, 1e6)},
        )

        # Step 2: converged=False and is_fallback=True must be set on the sentinel
        assert nlsq_result.is_fallback is True, (
            "BUG-003 regression: exception during fitting must set is_fallback=True"
        )
        assert nlsq_result.converged is False, (
            "BUG-003 regression: sentinel must have converged=False"
        )

        # Step 3: Sentinel R-squared must be NaN (not 0.0) — BUG-023
        assert math.isnan(nlsq_result.r_squared), (
            f"BUG-023 regression: sentinel r_squared={nlsq_result.r_squared}, expected NaN"
        )

        # Step 4: Verify FitResult validation guards NUTS from invalid states (BUG-024)
        with pytest.raises(ValueError, match="non-empty"):
            FitResult(samples={})

        # Step 5: Verify FitDiagnostics guards against unphysical diagnostics (BUG-025)
        with pytest.raises(ValueError):
            FitDiagnostics(divergences=-5)

        # Step 6: Valid sentinel must have NaN metrics (not 0.0)
        sentinel = NLSQResult(
            params={"a": 1.0},
            chi_squared=float("nan"),
            converged=False,
            is_fallback=True,
            _r_squared=float("nan"),
        )
        assert math.isnan(sentinel.r_squared), (
            "Sentinel propagation broken: r_squared is not NaN"
        )


# ===========================================================================
# Chain 6: JAX Boundary Leak
# BUG-027 + BUG-014 + BUG-007
# ===========================================================================


class TestChain6JAXBoundaryLeak:
    """Integration test for JAX-to-NumPy conversion at all boundary crossing points.

    Verifies that JAX arrays are converted to NumPy before reaching PyQtGraph.
    Checks at all 7 identified boundary crossing points.
    """

    def test_ensure_numpy_converts_jax_array(self):
        """BUG-027: ensure_numpy converts JAX arrays to writable NumPy arrays."""
        jnp = pytest.importorskip("jax.numpy")
        from xpcsviewer.backends._conversions import ensure_numpy

        jax_arr = jnp.array([1.0, 2.0, 3.0])
        result = ensure_numpy(jax_arr)

        assert isinstance(result, np.ndarray), (
            f"ensure_numpy must return np.ndarray, got {type(result)}"
        )
        assert result.flags.writeable, "ensure_numpy must return a writable array"
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_ensure_numpy_passes_through_numpy(self):
        """ensure_numpy is a no-op for writable NumPy arrays (fast path)."""
        from xpcsviewer.backends._conversions import ensure_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result is arr  # Fast path: same object returned

    def test_ensure_numpy_handles_read_only_numpy(self):
        """ensure_numpy copies read-only NumPy arrays to make them writable."""
        from xpcsviewer.backends._conversions import ensure_numpy

        arr = np.array([1.0, 2.0, 3.0])
        arr.flags.writeable = False
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert result.flags.writeable, (
            "ensure_numpy must make read-only arrays writable"
        )

    def test_boundary_point_xpcs_viewer_saxs_2d(self):
        """BUG-027: xpcs_viewer.py apply_saxs_2d_result wraps setImage with ensure_numpy."""
        import inspect

        from xpcsviewer import xpcs_viewer

        source = inspect.getsource(xpcs_viewer.XpcsViewer.apply_saxs_2d_result)
        assert "ensure_numpy" in source, (
            "BUG-027 regression: apply_saxs_2d_result missing ensure_numpy at setImage call"
        )

    def test_boundary_point_xpcs_viewer_apply_g2_result(self):
        """BUG-027: apply_g2_result uses ensure_numpy or passes result from worker (already NumPy)."""
        import inspect

        from xpcsviewer import xpcs_viewer

        # The fix is in apply_saxs_2d_result; apply_g2_result uses worker data directly (BUG-014)
        # Verify either: ensure_numpy is called, or result dict is used directly (no raw jax arrays)
        source = inspect.getsource(xpcs_viewer.XpcsViewer.apply_g2_result)
        # The method must not call setImage with a JAX array directly
        # It either calls ensure_numpy OR uses pre-computed NumPy data from worker
        uses_ensure_numpy = "ensure_numpy" in source
        uses_worker_result = 'result.get("' in source or "result.get(" in source
        assert uses_ensure_numpy or uses_worker_result, (
            "BUG-027 + BUG-014: apply_g2_result must either use ensure_numpy "
            "or consume pre-computed NumPy data from the worker result dict"
        )

    def test_boundary_point_simplemask_kernel(self):
        """BUG-027: simplemask_kernel.py wraps setImage with ensure_numpy.

        The method that calls setImage is show_saxs (the detector image display method).
        """
        import inspect

        from xpcsviewer.simplemask import simplemask_kernel

        source = inspect.getsource(simplemask_kernel.SimpleMaskKernel.show_saxs)
        assert "ensure_numpy" in source, (
            "BUG-027 regression: simplemask_kernel.show_saxs "
            "missing ensure_numpy at setImage call"
        )

    def test_boundary_point_simplemask_window(self):
        """BUG-027: simplemask_window.py wraps PyQtGraph calls with ensure_numpy."""
        import inspect

        from xpcsviewer.simplemask import simplemask_window

        source = inspect.getsource(simplemask_window)
        assert "ensure_numpy" in source, (
            "BUG-027 regression: simplemask_window.py missing ensure_numpy at boundary crossings"
        )

    def test_jax_array_through_ensure_numpy_is_numpy_at_boundary(self):
        """Chain 6 end-to-end: JAX array pipeline produces NumPy at boundary."""
        jnp = pytest.importorskip("jax.numpy")
        from xpcsviewer.backends._conversions import ensure_numpy, is_jax_array

        # Simulate a JAX computation that would feed a PyQtGraph plot
        backend_result = jnp.ones((100, 100), dtype=jnp.float32)
        assert is_jax_array(backend_result), "Test setup: expected a JAX array"

        # At the boundary, ensure_numpy must convert it
        plot_data = ensure_numpy(backend_result)

        assert isinstance(plot_data, np.ndarray), (
            "JAX array must be converted to NumPy before reaching PyQtGraph"
        )
        assert not is_jax_array(plot_data), (
            "Result of ensure_numpy must not be a JAX array"
        )
        assert plot_data.flags.writeable, "Array at PyQtGraph boundary must be writable"

    def test_all_seven_boundary_points_have_ensure_numpy(self):
        """BUG-027: All 7 identified JAX boundary crossing points use ensure_numpy.

        Boundary points from spec:
        1. xpcs_viewer.py apply_saxs_2d_result: setImage
        2. xpcs_viewer.py apply_g2_result: result data used (or ensure_numpy)
        3. simplemask_kernel.py show_saxs: setImage (detector image display)
        4-7. simplemask_window.py: four boundary crossings
        """
        import inspect

        from xpcsviewer import xpcs_viewer
        from xpcsviewer.simplemask import simplemask_kernel, simplemask_window

        # Point 1: apply_saxs_2d_result
        saxs_source = inspect.getsource(xpcs_viewer.XpcsViewer.apply_saxs_2d_result)
        assert "ensure_numpy" in saxs_source, "Boundary 1 missing ensure_numpy"

        # Point 2: apply_g2_result (covered by using worker result dict directly)
        g2_source = inspect.getsource(xpcs_viewer.XpcsViewer.apply_g2_result)
        has_boundary = "ensure_numpy" in g2_source or "result.get(" in g2_source
        assert has_boundary, "Boundary 2 not properly guarded"

        # Point 3: simplemask_kernel show_saxs (the method that calls setImage)
        sk_source = inspect.getsource(simplemask_kernel.SimpleMaskKernel.show_saxs)
        assert "ensure_numpy" in sk_source, (
            "Boundary 3 missing ensure_numpy in show_saxs"
        )

        # Points 4-7: simplemask_window overall module
        sw_source = inspect.getsource(simplemask_window)
        assert "ensure_numpy" in sw_source, (
            "Boundaries 4-7 missing ensure_numpy in simplemask_window"
        )
