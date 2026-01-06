"""Tests for CPU fallback when GPU fails (US1).

Tests FR-002: GPU to CPU fallback
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestGPUToCPUFallback:
    """Test automatic fallback from GPU to CPU."""

    def test_fallback_enabled_by_default(self):
        """Test that fallback is enabled by default."""
        from xpcsviewer.backends import DeviceConfig

        config = DeviceConfig()
        assert config.allow_gpu_fallback is True

    def test_fallback_when_gpu_unavailable(self):
        """Test automatic fallback to CPU when GPU is unavailable."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        # Request GPU with fallback enabled
        config = DeviceConfig(
            preferred_device=DeviceType.GPU,
            allow_gpu_fallback=True,
        )
        dm.configure(config)

        # Should be on CPU if GPU not available
        if not dm.gpu_available:
            assert not dm.is_gpu_enabled
            assert dm.current_device.device_type == DeviceType.CPU

        DeviceManager.reset()

    def test_no_fallback_raises_error(self):
        """Test RuntimeError when GPU unavailable and fallback disabled."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        if dm.gpu_available:
            pytest.skip("GPU is available, cannot test fallback error")

        config = DeviceConfig(
            preferred_device=DeviceType.GPU,
            allow_gpu_fallback=False,
        )

        with pytest.raises(RuntimeError, match="GPU requested but not available"):
            dm.configure(config)

        DeviceManager.reset()

    def test_fallback_env_var(self):
        """Test XPCS_GPU_FALLBACK environment variable."""
        from xpcsviewer.backends import DeviceConfig

        with patch.dict(os.environ, {"XPCS_GPU_FALLBACK": "true"}):
            config = DeviceConfig.from_environment()
            assert config.allow_gpu_fallback is True

        with patch.dict(os.environ, {"XPCS_GPU_FALLBACK": "false"}):
            config = DeviceConfig.from_environment()
            assert config.allow_gpu_fallback is False


class TestComputationAfterFallback:
    """Test that computations work correctly after fallback."""

    def test_array_operations_after_fallback(self):
        """Test that array operations work correctly after CPU fallback."""
        from xpcsviewer.backends import (
            DeviceConfig,
            DeviceManager,
            DeviceType,
            get_backend,
            reset_backend,
        )

        DeviceManager.reset()
        reset_backend()

        dm = DeviceManager()
        dm.configure(
            DeviceConfig(
                preferred_device=DeviceType.GPU,
                allow_gpu_fallback=True,
            )
        )

        backend = get_backend()

        # Basic operations should work regardless of device
        x = backend.array([1.0, 2.0, 3.0])
        y = backend.array([4.0, 5.0, 6.0])

        result = backend.to_numpy(backend.hypot(x, y))
        expected = np.hypot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(result, expected)

        DeviceManager.reset()
        reset_backend()

    def test_meshgrid_after_fallback(self):
        """Test that meshgrid works correctly after CPU fallback."""
        from xpcsviewer.backends import (
            DeviceConfig,
            DeviceManager,
            DeviceType,
            get_backend,
            reset_backend,
        )

        DeviceManager.reset()
        reset_backend()

        dm = DeviceManager()
        dm.configure(
            DeviceConfig(
                preferred_device=DeviceType.GPU,
                allow_gpu_fallback=True,
            )
        )

        backend = get_backend()

        x = backend.arange(0, 3)
        y = backend.arange(0, 2)
        xx, yy = backend.meshgrid(x, y)

        np.testing.assert_array_equal(backend.to_numpy(xx), [[0, 1, 2], [0, 1, 2]])
        np.testing.assert_array_equal(backend.to_numpy(yy), [[0, 0, 0], [1, 1, 1]])

        DeviceManager.reset()
        reset_backend()


class TestFallbackLogging:
    """Test logging behavior during fallback."""

    def test_fallback_logs_warning(self, caplog):
        """Test that fallback generates a warning log."""
        import logging

        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        if dm.gpu_available:
            pytest.skip("GPU is available, cannot test fallback logging")

        with caplog.at_level(logging.WARNING):
            dm.configure(
                DeviceConfig(
                    preferred_device=DeviceType.GPU,
                    allow_gpu_fallback=True,
                )
            )

        # Should have logged a warning about fallback
        assert any(
            "GPU requested but not available" in record.message
            or "falling back to CPU" in record.message
            for record in caplog.records
        )

        DeviceManager.reset()


class TestFallbackStateConsistency:
    """Test state consistency after fallback."""

    def test_device_state_consistent_after_fallback(self):
        """Test that device state is consistent after fallback."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        dm.configure(
            DeviceConfig(
                preferred_device=DeviceType.GPU,
                allow_gpu_fallback=True,
            )
        )

        # State should be consistent
        if dm.is_gpu_enabled:
            assert dm.current_device.device_type == DeviceType.GPU
        else:
            assert dm.current_device.device_type == DeviceType.CPU

        DeviceManager.reset()

    def test_multiple_fallback_attempts(self):
        """Test that multiple configuration attempts with fallback work."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        # First attempt
        dm.configure(
            DeviceConfig(
                preferred_device=DeviceType.GPU,
                allow_gpu_fallback=True,
            )
        )
        first_device = dm.current_device.device_type

        # Second attempt - should be consistent
        dm.configure(
            DeviceConfig(
                preferred_device=DeviceType.GPU,
                allow_gpu_fallback=True,
            )
        )
        second_device = dm.current_device.device_type

        assert first_device == second_device

        DeviceManager.reset()
