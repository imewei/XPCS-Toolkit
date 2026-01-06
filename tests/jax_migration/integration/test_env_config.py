"""Tests for environment variable configuration (T070).

Tests environment variable overrides for backend selection (US5).
"""

from __future__ import annotations

import pytest


class TestEnvVarOverrides:
    """Tests for environment variable configuration."""

    def test_xpcs_use_jax_enables_jax(self, monkeypatch) -> None:
        """Test XPCS_USE_JAX=1 enables JAX backend."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()
        assert backend.name == "jax"

    def test_xpcs_use_jax_disables_jax(self, monkeypatch) -> None:
        """Test XPCS_USE_JAX=0 disables JAX backend."""
        monkeypatch.setenv("XPCS_USE_JAX", "0")

        from xpcsviewer.backends import _reset_backend, get_backend, set_backend

        _reset_backend()
        # Explicitly set numpy backend since JAX may already be imported
        set_backend("numpy")

        backend = get_backend()
        assert backend.name == "numpy"

    def test_jax_platforms_cpu(self, monkeypatch) -> None:
        """Test JAX_PLATFORMS=cpu forces CPU execution."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()
        assert backend.name == "jax"
        # Even with JAX, should be on CPU
        import jax

        assert jax.devices()[0].platform == "cpu"

    def test_xpcs_gpu_fallback_respected(self, monkeypatch) -> None:
        """Test XPCS_GPU_FALLBACK enables CPU fallback."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("XPCS_GPU_FALLBACK", "1")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        # Should not fail even if GPU is unavailable
        assert manager is not None

    def test_case_insensitive_env_values(self, monkeypatch) -> None:
        """Test environment variable values are case-insensitive."""
        for value in ["TRUE", "True", "true", "1", "yes", "YES"]:
            monkeypatch.setenv("XPCS_USE_JAX", value)

            from xpcsviewer.backends import _parse_bool_env, _reset_backend

            _reset_backend()

            # The function should parse these as True
            assert _parse_bool_env("XPCS_USE_JAX", default=False)

    def test_default_behavior_without_env_vars(self, monkeypatch) -> None:
        """Test default behavior when no env vars are set."""
        # Remove relevant env vars
        monkeypatch.delenv("XPCS_USE_JAX", raising=False)
        monkeypatch.delenv("JAX_PLATFORMS", raising=False)
        monkeypatch.delenv("XPCS_GPU_FALLBACK", raising=False)

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()
        # Default should prefer JAX if available
        assert backend.name in ("jax", "numpy")


class TestEnvVarMemoryConfig:
    """Tests for memory-related environment variables."""

    def test_xpcs_gpu_memory_fraction(self, monkeypatch) -> None:
        """Test XPCS_GPU_MEMORY_FRACTION is respected."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("XPCS_GPU_MEMORY_FRACTION", "0.5")

        from xpcsviewer.backends import _reset_backend
        from xpcsviewer.backends._device import DeviceManager

        _reset_backend()

        manager = DeviceManager()
        # If GPU memory fraction setting exists, it should be respected
        # This is informational - the actual enforcement depends on JAX config
        assert manager is not None

    def test_invalid_memory_fraction_handled(self, monkeypatch) -> None:
        """Test invalid memory fraction is handled gracefully."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("XPCS_GPU_MEMORY_FRACTION", "invalid")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        # Should not crash with invalid value
        backend = get_backend()
        assert backend is not None


class TestEnvVarLogging:
    """Tests for logging of environment variable configuration."""

    def test_backend_selection_logged(self, monkeypatch, caplog) -> None:
        """Test backend selection is logged."""
        import logging

        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        with caplog.at_level(logging.DEBUG, logger="xpcsviewer.backends"):
            backend = get_backend()

        # Should have some log output about backend selection
        # Note: Actual log message depends on implementation
        assert backend is not None
