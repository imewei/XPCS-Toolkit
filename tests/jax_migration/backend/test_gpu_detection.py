"""Tests for GPU detection and automatic selection (US1).

Tests FR-001: Automatic device detection
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


class TestGPUDetection:
    """Test GPU detection functionality."""

    def test_gpu_available_property(self):
        """Test that gpu_available property correctly detects GPU presence."""
        from xpcsviewer.backends import DeviceManager

        DeviceManager.reset()
        dm = DeviceManager()

        # Property should return a boolean
        assert isinstance(dm.gpu_available, bool)

        # If JAX is not available, GPU should not be available
        if not dm.jax_available:
            assert dm.gpu_available is False

        DeviceManager.reset()

    @pytest.mark.jax
    def test_gpu_detection_with_jax(self, require_jax):
        """Test GPU detection when JAX is available."""
        from xpcsviewer.backends import DeviceManager

        DeviceManager.reset()
        dm = DeviceManager()

        assert dm.jax_available is True
        # GPU availability depends on hardware
        assert isinstance(dm.gpu_available, bool)

        DeviceManager.reset()

    def test_automatic_device_selection_cpu_only(self):
        """Test that CPU is selected when GPU is not available."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        config = DeviceConfig(preferred_device=DeviceType.CPU)
        dm.configure(config)

        assert not dm.is_gpu_enabled
        assert dm.current_device is not None
        assert dm.current_device.device_type == DeviceType.CPU

        DeviceManager.reset()

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_automatic_device_selection_gpu(self, require_jax, require_gpu):
        """Test that GPU is selected when available and requested."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        config = DeviceConfig(preferred_device=DeviceType.GPU)
        dm.configure(config)

        assert dm.is_gpu_enabled
        assert dm.current_device is not None
        assert dm.current_device.device_type == DeviceType.GPU

        DeviceManager.reset()


class TestEnvironmentVariableGPUControl:
    """Test GPU control via environment variables."""

    def test_xpcs_use_gpu_false(self):
        """Test XPCS_USE_GPU=false forces CPU."""
        from xpcsviewer.backends import DeviceConfig, DeviceType

        with patch.dict(os.environ, {"XPCS_USE_GPU": "false"}):
            config = DeviceConfig.from_environment()
            assert config.preferred_device == DeviceType.CPU

    def test_xpcs_use_gpu_true(self):
        """Test XPCS_USE_GPU=true requests GPU."""
        from xpcsviewer.backends import DeviceConfig, DeviceType

        with patch.dict(os.environ, {"XPCS_USE_GPU": "true"}):
            config = DeviceConfig.from_environment()
            assert config.preferred_device == DeviceType.GPU

    def test_memory_fraction_from_environment(self):
        """Test XPCS_GPU_MEMORY_FRACTION is parsed correctly."""
        from xpcsviewer.backends import DeviceConfig

        with patch.dict(os.environ, {"XPCS_GPU_MEMORY_FRACTION": "0.75"}):
            config = DeviceConfig.from_environment()
            assert config.memory_fraction == 0.75

    def test_fallback_from_environment(self):
        """Test XPCS_GPU_FALLBACK is parsed correctly."""
        from xpcsviewer.backends import DeviceConfig

        with patch.dict(os.environ, {"XPCS_GPU_FALLBACK": "false"}):
            config = DeviceConfig.from_environment()
            assert config.allow_gpu_fallback is False


class TestDeviceInfo:
    """Test device information reporting."""

    def test_device_info_cpu(self):
        """Test DeviceInfo for CPU device."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()
        dm.configure(DeviceConfig(preferred_device=DeviceType.CPU))

        device_info = dm.current_device
        assert device_info is not None
        assert device_info.device_type == DeviceType.CPU
        assert device_info.name == "CPU"

        DeviceManager.reset()

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_device_info_gpu(self, require_jax, require_gpu):
        """Test DeviceInfo for GPU device."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()
        dm.configure(DeviceConfig(preferred_device=DeviceType.GPU))

        device_info = dm.current_device
        assert device_info is not None
        assert device_info.device_type == DeviceType.GPU
        assert "gpu" in device_info.name.lower() or "cuda" in device_info.name.lower()

        DeviceManager.reset()


class TestBackendGPUCapabilities:
    """Test backend GPU capabilities."""

    def test_numpy_backend_no_gpu(self, numpy_backend):
        """Test NumPy backend reports no GPU support."""
        assert numpy_backend.supports_gpu is False

    @pytest.mark.jax
    def test_jax_backend_gpu_support(self, jax_backend):
        """Test JAX backend can support GPU."""
        # supports_gpu indicates capability, not current state
        # JAX can support GPU if drivers are present
        assert isinstance(jax_backend.supports_gpu, bool)
