"""Tests for GPU system launch (T069).

Tests that application launches correctly on GPU systems (US5).
"""

from __future__ import annotations

import pytest

# Check if GPU is available
try:
    import jax

    GPU_AVAILABLE = len(jax.devices("gpu")) > 0
except (ImportError, RuntimeError):
    GPU_AVAILABLE = False


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestGPULaunch:
    """Tests for GPU system launch."""

    def test_backend_detects_gpu(self, monkeypatch) -> None:
        """Test backend detects GPU when available."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()
        assert backend is not None
        assert backend.name == "jax"
        assert backend.supports_gpu

    def test_gpu_device_listed(self, monkeypatch) -> None:
        """Test GPU device is listed in available devices."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        devices = manager.available_devices

        assert any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)

    def test_qmap_on_gpu(self, monkeypatch) -> None:
        """Test Q-map computation runs on GPU."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, units = compute_transmission_qmap(
            energy=10.0,
            center=(128.0, 128.0),
            shape=(256, 256),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        assert "q" in qmap
        assert qmap["q"].shape == (256, 256)


class TestGPUDetection:
    """Tests for GPU detection logic."""

    def test_gpu_detection_does_not_raise(self, monkeypatch) -> None:
        """Test GPU detection doesn't raise exceptions."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        # Should not raise even if no GPU
        from xpcsviewer.backends._device import DeviceManager

        manager = DeviceManager()
        _ = manager.has_gpu  # Should not raise

    def test_gpu_detection_returns_bool(self, monkeypatch) -> None:
        """Test GPU detection returns boolean."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.backends._device import DeviceManager

        manager = DeviceManager()
        result = manager.has_gpu

        assert isinstance(result, bool)


class TestGPUMemoryInfo:
    """Tests for GPU memory information."""

    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_memory_info_available(self, monkeypatch) -> None:
        """Test memory info is available on GPU systems."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        if manager.has_gpu:
            # Memory info should be retrievable
            info = manager.get_memory_info()
            assert info is not None

    def test_memory_info_graceful_on_cpu(self, monkeypatch) -> None:
        """Test memory info returns gracefully on CPU."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        # Should not raise even on CPU-only systems
        info = manager.get_memory_info()
        # Returns None or empty dict on CPU
        assert info is None or isinstance(info, dict)
