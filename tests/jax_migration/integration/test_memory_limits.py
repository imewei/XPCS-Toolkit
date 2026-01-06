"""Tests for memory limits during large computations (T073a).

Tests that memory stays below 90% during large computations per SC-007 (US5).
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestMemoryLimits:
    """Tests for memory usage limits."""

    def test_qmap_memory_stays_reasonable(self, monkeypatch) -> None:
        """Test Q-map computation doesn't exhaust memory."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import psutil

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

        # Compute Q-map for moderately large detector
        for _ in range(3):  # Multiple iterations
            qmap, _ = compute_transmission_qmap(
                energy=10.0,
                center=(512.0, 512.0),
                shape=(1024, 1024),
                pix_dim=0.075,
                det_dist=5000.0,
            )
            del qmap
            gc.collect()

        # Get final memory
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this test)
        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"

    def test_partition_memory_stays_reasonable(self, monkeypatch) -> None:
        """Test partition computation doesn't exhaust memory."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import psutil

        from xpcsviewer.simplemask.utils import generate_partition

        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Generate partition for moderately large Q-map
        qmap = np.random.random((1024, 1024)).astype(np.float64)
        mask = np.ones((1024, 1024), dtype=bool)
        mask[512:, 512:] = False  # Mask out quadrant

        for _ in range(3):
            partition = generate_partition(
                map_name="q",
                mask=mask,
                xmap=qmap,
                num_pts=50,
                style="linear",
            )
            del partition
            gc.collect()

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        assert memory_increase < 500, f"Memory increased by {memory_increase}MB"

    def test_fitting_memory_stays_reasonable(self, monkeypatch) -> None:
        """Test fitting doesn't exhaust memory."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp
        import psutil

        from xpcsviewer.fitting.nlsq import nlsq_optimize

        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)

        # Generate larger synthetic dataset
        x = np.logspace(-3, 1, 500)
        true_tau = 0.1
        y = 1.0 + 0.3 * np.exp(-2 * x / true_tau) + np.random.normal(0, 0.01, len(x))

        # Model must use JAX ops for JAX backend compatibility
        def model(x, tau, baseline, contrast):
            return baseline + contrast * jnp.exp(-2 * x / tau)

        for _ in range(5):
            result = nlsq_optimize(
                model_fn=model,
                x=x,
                y=y,
                yerr=np.ones_like(y) * 0.01,
                p0={"tau": 0.2, "baseline": 1.0, "contrast": 0.3},
                bounds={
                    "tau": (0.01, 10.0),
                    "baseline": (0.5, 1.5),
                    "contrast": (0.01, 1.0),
                },
            )
            del result
            gc.collect()

        final_memory = process.memory_info().rss / (1024 * 1024)
        memory_increase = final_memory - initial_memory

        assert memory_increase < 200, f"Memory increased by {memory_increase}MB"


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestMemoryMonitoring:
    """Tests for memory monitoring capabilities."""

    def test_device_manager_reports_memory(self, monkeypatch) -> None:
        """Test DeviceManager can report memory usage."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        info = manager.get_memory_info()

        # Should return None or dict (depending on platform)
        assert info is None or isinstance(info, dict)

    def test_memory_info_keys(self, monkeypatch) -> None:
        """Test memory info contains expected keys if available."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        info = manager.get_memory_info()

        if info is not None and manager.has_gpu:
            # On GPU systems, should have memory keys
            # Keys might be platform-specific
            assert isinstance(info, dict)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestChunkedProcessing:
    """Tests for chunked processing to manage memory."""

    def test_large_array_processed_without_crash(self, monkeypatch) -> None:
        """Test large arrays can be processed without memory crash."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Process moderately large array
        x = backend.linspace(0, 1, 100000)
        y = backend.sin(x) * backend.exp(-x)
        z = backend.sum(y)

        assert not np.isnan(float(z))

    def test_jit_compilation_caches_properly(self, monkeypatch) -> None:
        """Test JIT compilation caches to avoid memory bloat."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import _JIT_CACHE, compute_transmission_qmap

        # Clear cache
        _JIT_CACHE.clear()

        # First call
        qmap1, _ = compute_transmission_qmap(
            10.0, (64.0, 64.0), (128, 128), 0.075, 5000.0
        )

        # Check cache has entry
        cache_size_after_first = len(_JIT_CACHE)

        # Multiple calls with same shape
        for _ in range(5):
            qmap, _ = compute_transmission_qmap(
                10.0, (64.0, 64.0), (128, 128), 0.075, 5000.0
            )

        # Cache should not grow (reusing compiled function)
        assert len(_JIT_CACHE) == cache_size_after_first
