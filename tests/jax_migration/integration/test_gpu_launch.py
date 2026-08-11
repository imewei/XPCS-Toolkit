"""Tests for GPU system launch (T069).

Tests that application launches correctly on GPU systems (US5).
"""

from __future__ import annotations


class TestGPUDetection:
    """Tests for GPU detection logic."""

    def test_gpu_detection_does_not_raise(self, monkeypatch) -> None:
        """Test GPU detection doesn't raise exceptions."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()
        backend = get_backend()
        _ = backend.supports_gpu  # Should not raise

    def test_gpu_detection_returns_bool(self, monkeypatch) -> None:
        """Test GPU detection returns boolean."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()
        backend = get_backend()
        result = backend.supports_gpu

        assert isinstance(result, bool)
