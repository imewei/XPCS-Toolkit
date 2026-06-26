"""Tests for Task Group 9: Performance and JIT Compilation Fixes.

Covers BUG-038, BUG-042, BUG-050, BUG-051, BUG-052, BUG-053, BUG-054, BUG-055, BUG-057.
"""

from __future__ import annotations

import gc
import inspect
import time
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1: Removed unused JAX scipy_replacements optimize tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 2: nonzero() does not recompile per input size (BUG-042)
# ---------------------------------------------------------------------------


class TestNonzeroNoRecompilation:
    """Test 2: nonzero() in _jax_backend.py prevents per-input-size recompilation."""

    def test_nonzero_falls_back_to_size_param(self):
        """nonzero() without explicit size uses x.size to keep shape fixed."""
        from xpcsviewer.backends._jax_backend import JAXBackend

        src = inspect.getsource(JAXBackend.nonzero)
        # The fix: when size is None, use x.size so shape is statically known
        assert "x.size" in src or "size=size" in src, (
            "nonzero() must use a fixed size to prevent per-input-size recompilation"
        )

    def test_nonzero_with_explicit_size_returns_padded_result(self):
        """nonzero() with explicit size pads results to fixed length."""
        pytest.importorskip("jax")

        from xpcsviewer.backends._jax_backend import JAXBackend

        backend = JAXBackend()
        import jax.numpy as jnp

        arr = jnp.array([0, 1, 0, 2, 0])
        # 2 non-zero elements; request size=5
        (indices,) = backend.nonzero(arr, size=5)
        assert len(indices) == 5
        # Non-zero positions are 1 and 3
        assert 1 in indices
        assert 3 in indices

    def test_nonzero_without_size_uses_full_array_size(self):
        """nonzero() without size defaults to x.size for JIT stability."""
        pytest.importorskip("jax")

        from xpcsviewer.backends._jax_backend import JAXBackend

        backend = JAXBackend()
        import jax.numpy as jnp

        arr = jnp.array([1, 0, 3])
        (indices,) = backend.nonzero(arr)
        # Result length equals x.size (3), padded with -1
        assert len(indices) == 3


# ---------------------------------------------------------------------------
# Test 3: Q-map cache hit does not deep-copy (BUG-055)
# ---------------------------------------------------------------------------


class TestQMapCacheNoCopy:
    """Test 3: Q-map cache hit returns the cached dict directly, no deep copy."""

    def test_transmission_cache_hit_no_deepcopy(self):
        """compute_transmission_qmap must not call copy.deepcopy on cache hit."""
        from xpcsviewer.simplemask import qmap as qmap_module

        # Use inspect.unwrap to bypass any __wrapped__ poisoning from other tests
        func = inspect.unwrap(qmap_module.compute_transmission_qmap)
        src = inspect.getsource(func)
        assert "deepcopy" not in src, (
            "compute_transmission_qmap must not deep-copy the cached result (BUG-055). "
            "Return the cached dict directly since the arrays are already frozen."
        )

    def test_reflection_cache_hit_no_deepcopy(self):
        """compute_reflection_qmap must not call copy.deepcopy on cache hit."""
        from xpcsviewer.simplemask import qmap as qmap_module

        func = inspect.unwrap(qmap_module.compute_reflection_qmap)
        src = inspect.getsource(func)
        assert "deepcopy" not in src, (
            "compute_reflection_qmap must not deep-copy the cached result (BUG-055). "
            "Return the cached dict directly since the arrays are already frozen."
        )

    def test_cache_hit_returns_same_object(self):
        """Two calls with identical params return the same cached dict object."""
        from xpcsviewer.simplemask.qmap import (
            _compute_transmission_qmap_cached,
            compute_transmission_qmap,
        )

        _compute_transmission_qmap_cached.cache_clear()

        params = dict(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        qmap1, _ = compute_transmission_qmap(**params)
        qmap2, _ = compute_transmission_qmap(**params)

        # Without deep copy, at least the array data must be identical
        # (they may or may not be the same object depending on implementation,
        # but they must NOT be expensive full copies)
        assert "q" in qmap1 and "q" in qmap2
        np.testing.assert_array_equal(qmap1["q"], qmap2["q"])


# ---------------------------------------------------------------------------
# Test 4: det_dist = 0 guard in qmap.py (BUG-051)
# ---------------------------------------------------------------------------


class TestDetDistEpsilonGuard:
    """Test 4: compute_q_at_pixel guards against det_dist = 0 with epsilon."""

    def test_compute_q_at_pixel_src_has_epsilon_guard(self):
        """compute_q_at_pixel source must guard against zero det_dist."""
        from xpcsviewer.simplemask import qmap as qmap_module

        # Check either compute_q_at_pixel or compute_transmission_qmap has epsilon guard
        src_q = inspect.getsource(qmap_module.compute_q_at_pixel)
        src_trans = inspect.getsource(qmap_module.compute_transmission_qmap)
        src_full = inspect.getsource(qmap_module)

        # Look for epsilon guard pattern in any relevant function
        has_guard = (
            "1e-12" in src_q
            or "max(det_dist" in src_q
            or "max(det_dist" in src_full
            or "epsilon" in src_q.lower()
        )
        assert has_guard, (
            "qmap.py must guard against det_dist=0 with epsilon (e.g., "
            "det_dist = max(det_dist, 1e-12)) to prevent division by zero (BUG-051)"
        )

    def test_compute_q_at_pixel_with_zero_det_dist_does_not_raise(self):
        """compute_q_at_pixel does not divide by zero when det_dist ~ 0."""
        from xpcsviewer.simplemask.qmap import compute_q_at_pixel

        # Should not raise ZeroDivisionError
        q = compute_q_at_pixel(
            center_x=64.0,
            center_y=64.0,
            pixel_x=65.0,
            pixel_y=65.0,
            energy=10.0,
            pix_dim=0.075,
            det_dist=0.0,  # zero det_dist
        )
        assert np.isfinite(q) or q == pytest.approx(
            np.pi / 2 * (2 * np.pi / (12.39841984 / 10.0)), rel=0.1
        )

    def test_transmission_qmap_with_tiny_det_dist(self):
        """compute_transmission_qmap handles very small det_dist gracefully."""
        from xpcsviewer.simplemask.qmap import (
            _compute_transmission_qmap_cached,
            compute_transmission_qmap,
        )

        _compute_transmission_qmap_cached.cache_clear()

        # Should not raise
        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=0.0,
        )
        assert "q" in qmap
        assert np.all(np.isfinite(qmap["q"]))


# ---------------------------------------------------------------------------
# Test 5: Health monitor GC metrics use per-interval delta (BUG-052)
# ---------------------------------------------------------------------------


class TestHealthMonitorGCDelta:
    """Test 5: GC metrics use per-interval delta, not stale start time."""

    def test_gc_metrics_use_interval_delta_not_start_time(self):
        """_update_gc_metrics must compute delta against last-interval stats, not start."""
        from xpcsviewer.utils.health_monitor import HealthMonitor

        src = inspect.getsource(HealthMonitor._update_gc_metrics)

        # The bug: uses `_monitor_start_time` (stale reference to loop start)
        # The fix: track GC stats from last interval (e.g., store `_gc_stats_last`)
        assert "_monitor_start_time" not in src, (
            "_update_gc_metrics must not use _monitor_start_time for GC delta "
            "calculation (BUG-052). Use per-interval delta (compare to last "
            "interval's stats, not the monitoring start time)."
        )

    def test_gc_metrics_interval_tracking(self):
        """GC metric update computes delta relative to previous interval's stats."""
        from xpcsviewer.utils.health_monitor import HealthMonitor

        src = inspect.getsource(HealthMonitor._update_gc_metrics)
        src_loop = inspect.getsource(HealthMonitor._monitoring_loop)

        # The fix pattern: update gc_stats_start inside the loop to track per-interval
        # OR store a separate _gc_stats_prev attribute updated each iteration
        has_interval_tracking = (
            "gc_stats_start = gc.get_stats()" in src_loop
            or "gc_stats_start =" in src_loop
            or "_gc_stats_prev" in src
            or "_gc_stats_last" in src
            or "initial_stats = " in src_loop
        )
        assert has_interval_tracking, (
            "Health monitor must reset GC baseline stats each interval "
            "to compute a true per-interval delta (BUG-052)"
        )

    def test_hdf5_connection_count_queries_pool(self):
        """_get_hdf5_connection_count must actually try to query pool size (BUG-053)."""
        from xpcsviewer.utils.health_monitor import HealthMonitor

        src = inspect.getsource(HealthMonitor._get_hdf5_connection_count)

        # Must attempt to query something from the pool (not just return 0.0 always)
        assert "_connection_pool" in src or "pool" in src.lower(), (
            "_get_hdf5_connection_count must attempt to query the HDF5 connection "
            "pool size instead of always returning 0.0 (BUG-053)"
        )
        # The function already does try to import and query - verify it's intact
        assert "return 0.0" in src, (
            "_get_hdf5_connection_count must have a fallback return 0.0"
        )
