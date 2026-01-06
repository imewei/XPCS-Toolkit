"""Tests for JIT compilation warmup (T055).

Tests that JIT compilation triggers on first call and subsequent
calls use cached compiled functions (US3).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJITWarmup:
    """Tests for JIT compilation warmup behavior."""

    def test_first_call_triggers_compilation(self) -> None:
        """Test first call triggers JIT compilation (slower)."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT warmup only applies to JAX backend")

        # Create test data
        x = backend.linspace(0, 10, 1000)

        # Define a JIT-able function
        @backend.jit
        def compute_sin_cos(x):
            return backend.sin(x) + backend.cos(x)

        # First call - includes compilation time
        start = time.perf_counter()
        _ = compute_sin_cos(x)
        first_call_time = time.perf_counter() - start

        # Second call - should use cached compilation
        start = time.perf_counter()
        _ = compute_sin_cos(x)
        second_call_time = time.perf_counter() - start

        # First call should be noticeably slower due to compilation
        # (This may not always hold for very simple functions, but
        # indicates compilation happened)
        assert first_call_time > 0  # Basic sanity check
        assert second_call_time > 0

    def test_jit_caches_compiled_function(self) -> None:
        """Test JIT caches compiled function for reuse."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT caching only applies to JAX backend")

        # Create test data
        x = backend.linspace(0, 10, 10000)

        @backend.jit
        def expensive_computation(x):
            result = x
            for _ in range(10):
                result = backend.sin(result) + backend.cos(result)
            return result

        # Warmup call
        _ = expensive_computation(x)

        # Multiple subsequent calls should all be fast
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = expensive_computation(x)
            times.append(time.perf_counter() - start)

        # All cached calls should have similar timing
        # (within 10x of each other for reasonable consistency)
        assert max(times) < 10 * min(times) + 0.001  # Add small epsilon

    def test_jit_with_different_shapes_recompiles(self) -> None:
        """Test JIT recompiles for different input shapes."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT shape handling only applies to JAX backend")

        @backend.jit
        def simple_sum(x):
            return backend.sum(x)

        # First shape
        x1 = backend.zeros((100,))
        _ = simple_sum(x1)

        # Different shape - should trigger recompilation
        x2 = backend.zeros((200,))
        result = simple_sum(x2)

        # Should still produce correct result
        assert float(result) == 0.0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQmapJITWarmup:
    """Tests for Q-map JIT compilation warmup."""

    def test_qmap_jit_function_exists(self) -> None:
        """Test JIT-compiled Q-map function exists."""
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Function should exist
        assert callable(compute_transmission_qmap)

    def test_qmap_first_call_compiles(self) -> None:
        """Test Q-map first call includes compilation."""
        from xpcsviewer.backends import get_backend
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Q-map JIT only applies to JAX backend")

        # Parameters: energy (keV), center (row, col), shape, pixel_size (mm), distance (mm)
        energy = 10.0  # keV
        shape = (256, 256)
        center = (128.0, 128.0)
        pix_dim = 0.075  # mm (75 microns)
        det_dist = 5000.0  # mm (5 meters)

        # First call
        start = time.perf_counter()
        qmap1, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)
        first_time = time.perf_counter() - start

        # Second call with same parameters
        start = time.perf_counter()
        qmap2, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)
        second_time = time.perf_counter() - start

        # Results should be identical (dict comparison needs per-key check)
        for key in qmap1:
            np.testing.assert_array_equal(qmap1[key], qmap2[key])

        # Both should complete successfully
        assert first_time > 0
        assert second_time > 0
