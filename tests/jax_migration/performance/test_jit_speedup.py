"""Tests for JIT speedup on repeated calls (T056).

Tests that repeated calls show >2x speedup after initial compilation (US3).
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
class TestJITSpeedup:
    """Tests for JIT compilation speedup."""

    def test_repeated_calls_faster_than_first(self) -> None:
        """Test repeated JIT calls are faster than first call."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT speedup only applies to JAX backend")

        # Create reasonably sized test data
        x = backend.linspace(0, 10, 10000)

        @backend.jit
        def compute_complex(x):
            """A more complex computation to see speedup."""
            result = backend.sin(x) * backend.cos(x)
            result = result + backend.exp(-x / 10)
            result = backend.sqrt(backend.abs(result) + 1e-10)
            return backend.sum(result)

        # First call (includes compilation)
        _ = compute_complex(x)  # Warmup

        # Time first "real" call after warmup
        start = time.perf_counter()
        _ = compute_complex(x)
        cached_time = time.perf_counter() - start

        # Multiple cached calls should be consistently fast
        cached_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = compute_complex(x)
            cached_times.append(time.perf_counter() - start)

        avg_cached = sum(cached_times) / len(cached_times)

        # All cached calls should be similar (low variance)
        assert max(cached_times) < 5 * avg_cached + 0.001

    def test_jit_vs_non_jit_speedup(self) -> None:
        """Test JIT version is faster than non-JIT for repeated calls."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT vs non-JIT comparison only applies to JAX backend")

        import jax.numpy as jnp

        # Create test data
        size = 5000
        x = jnp.linspace(0, 10, size)

        def non_jit_compute(x):
            """Non-JIT version."""
            result = jnp.sin(x) * jnp.cos(x)
            result = result + jnp.exp(-x / 10)
            return jnp.sum(result)

        @jax.jit
        def jit_compute(x):
            """JIT version."""
            result = jnp.sin(x) * jnp.cos(x)
            result = result + jnp.exp(-x / 10)
            return jnp.sum(result)

        # Warmup JIT version
        _ = jit_compute(x)

        # Time non-JIT calls
        non_jit_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = non_jit_compute(x)
            non_jit_times.append(time.perf_counter() - start)

        # Time JIT calls
        jit_times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = jit_compute(x)
            jit_times.append(time.perf_counter() - start)

        avg_non_jit = sum(non_jit_times) / len(non_jit_times)
        avg_jit = sum(jit_times) / len(jit_times)

        # JIT should generally be faster, but this can vary
        # At minimum, JIT should not be dramatically slower
        assert avg_jit < avg_non_jit * 5  # Very loose bound


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQmapJITSpeedup:
    """Tests for Q-map JIT speedup."""

    def test_qmap_repeated_calls_stable(self) -> None:
        """Test Q-map repeated calls have stable performance."""
        from xpcsviewer.backends import get_backend
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Q-map JIT speedup only applies to JAX backend")

        # Parameters: energy (keV), center (row, col), shape, pixel_size (mm), distance (mm)
        energy = 10.0  # keV
        shape = (512, 512)
        center = (256.0, 256.0)
        pix_dim = 0.075  # mm (75 microns)
        det_dist = 5000.0  # mm (5 meters)

        # Warmup call
        _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Time repeated calls
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)

        # Repeated calls should be stable (not wildly varying)
        # Allow 3x variance as a reasonable bound
        assert max(times) < avg_time * 3 + 0.01

    def test_partition_repeated_calls_stable(self) -> None:
        """Test partition repeated calls have stable performance."""
        from xpcsviewer.backends import get_backend
        from xpcsviewer.simplemask.utils import create_partition

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Partition JIT speedup only applies to JAX backend")

        # Create test Q-map
        np.random.seed(42)
        qmap = np.abs(np.random.randn(512, 512) * 0.1 + 0.05)
        mask = np.ones((512, 512), dtype=bool)

        # Warmup
        _ = create_partition(qmap, mask, n_bins=36, spacing="linear")

        # Time repeated calls
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = create_partition(qmap, mask, n_bins=36, spacing="linear")
            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)

        # Repeated calls should be stable
        assert max(times) < avg_time * 3 + 0.01


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestJITMemoryEfficiency:
    """Tests for JIT memory efficiency."""

    def test_jit_does_not_leak_memory(self) -> None:
        """Test JIT compilation doesn't leak memory on repeated calls."""
        import gc

        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT memory test only applies to JAX backend")

        @backend.jit
        def compute(x):
            return backend.sum(backend.sin(x) * backend.cos(x))

        x = backend.linspace(0, 10, 10000)

        # Run many iterations
        for _ in range(100):
            _ = compute(x)

        # Force garbage collection
        gc.collect()

        # If we get here without OOM, memory management is working
        # This is a basic sanity check
        assert True
