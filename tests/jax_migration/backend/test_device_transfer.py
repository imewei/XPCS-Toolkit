"""Tests for device transfer and placement.

Tests FR-001: Automatic device detection
Tests FR-002: GPU to CPU fallback
Tests SC-007: Memory stays below 90%
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


class TestDeviceManager:
    """Test DeviceManager singleton."""

    def test_device_manager_singleton(self):
        """Test that DeviceManager is a singleton."""
        from xpcsviewer.backends import DeviceManager

        dm1 = DeviceManager()
        dm2 = DeviceManager()
        assert dm1 is dm2

    def test_device_manager_reset(self):
        """Test DeviceManager reset for testing."""
        from xpcsviewer.backends import DeviceManager

        dm1 = DeviceManager()
        DeviceManager.reset()
        dm2 = DeviceManager()
        # After reset, should be different instance
        assert dm1 is not dm2
        DeviceManager.reset()

    def test_jax_available_property(self):
        """Test jax_available property detection."""
        from xpcsviewer.backends import DeviceManager

        DeviceManager.reset()
        dm = DeviceManager()

        try:
            import jax  # noqa: F401

            assert dm.jax_available is True
        except ImportError:
            assert dm.jax_available is False
        DeviceManager.reset()


class TestDeviceConfig:
    """Test DeviceConfig dataclass."""

    def test_device_config_defaults(self):
        """Test DeviceConfig default values."""
        from xpcsviewer.backends import DeviceConfig, DeviceType

        config = DeviceConfig()
        assert config.preferred_device == DeviceType.CPU
        assert config.allow_gpu_fallback is True
        assert config.memory_fraction == 0.9

    def test_device_config_from_environment_defaults(self):
        """Test DeviceConfig.from_environment with default env vars."""
        from xpcsviewer.backends import DeviceConfig, DeviceType

        with patch.dict(os.environ, {}, clear=True):
            # Clear relevant env vars
            for key in [
                "XPCS_USE_GPU",
                "XPCS_GPU_FALLBACK",
                "XPCS_GPU_MEMORY_FRACTION",
            ]:
                os.environ.pop(key, None)

            config = DeviceConfig.from_environment()
            assert config.preferred_device == DeviceType.CPU
            assert config.allow_gpu_fallback is True
            assert config.memory_fraction == 0.9

    def test_device_config_from_environment_gpu_enabled(self):
        """Test DeviceConfig.from_environment with GPU enabled."""
        from xpcsviewer.backends import DeviceConfig, DeviceType

        with patch.dict(
            os.environ,
            {
                "XPCS_USE_GPU": "true",
                "XPCS_GPU_FALLBACK": "false",
                "XPCS_GPU_MEMORY_FRACTION": "0.8",
            },
        ):
            config = DeviceConfig.from_environment()
            assert config.preferred_device == DeviceType.GPU
            assert config.allow_gpu_fallback is False
            assert config.memory_fraction == 0.8

    def test_device_config_invalid_memory_fraction(self):
        """Test DeviceConfig validation for memory_fraction."""
        from xpcsviewer.backends import DeviceConfig

        with pytest.raises(ValueError, match="memory_fraction"):
            DeviceConfig(memory_fraction=1.5)

        with pytest.raises(ValueError, match="memory_fraction"):
            DeviceConfig(memory_fraction=0.0)

        with pytest.raises(ValueError, match="memory_fraction"):
            DeviceConfig(memory_fraction=-0.1)


class TestArrayConversions:
    """Test array conversion utilities."""

    def test_ensure_numpy_from_numpy(self):
        """Test ensure_numpy with NumPy array input."""
        from xpcsviewer.backends import ensure_numpy

        arr = np.array([1, 2, 3])
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_ensure_numpy_from_list(self):
        """Test ensure_numpy with list input."""
        from xpcsviewer.backends import ensure_numpy

        data = [1, 2, 3]
        result = ensure_numpy(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    @pytest.mark.jax
    def test_ensure_numpy_from_jax(self, require_jax):
        """Test ensure_numpy with JAX array input."""
        import jax.numpy as jnp

        from xpcsviewer.backends import ensure_numpy

        arr = jnp.array([1, 2, 3])
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_ensure_backend_array_numpy(self):
        """Test ensure_backend_array with NumPy backend."""
        from xpcsviewer.backends import ensure_backend_array, set_backend

        set_backend("numpy")
        data = [1, 2, 3]
        result = ensure_backend_array(data)
        assert isinstance(result, np.ndarray)

    @pytest.mark.jax
    def test_ensure_backend_array_jax(self, require_jax):
        """Test ensure_backend_array with JAX backend."""
        import jax.numpy as jnp

        from xpcsviewer.backends import ensure_backend_array, set_backend

        set_backend("jax")
        data = [1, 2, 3]
        result = ensure_backend_array(data)
        assert isinstance(result, jnp.ndarray)


class TestGPUFallback:
    """Test GPU to CPU fallback behavior (FR-002)."""

    def test_fallback_when_gpu_unavailable_with_fallback_enabled(self):
        """Test fallback to CPU when GPU unavailable and fallback enabled."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()

        # Configure for GPU with fallback
        config = DeviceConfig(
            preferred_device=DeviceType.GPU,
            allow_gpu_fallback=True,
        )

        # This should not raise even if GPU is unavailable
        dm.configure(config)

        # Should be on CPU if GPU not available
        if not dm.gpu_available:
            assert not dm.is_gpu_enabled
        DeviceManager.reset()

    def test_fallback_disabled_raises_when_gpu_unavailable(self):
        """Test that RuntimeError is raised when GPU unavailable and fallback disabled."""
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


class TestDevicePlacement:
    """Test device placement functionality."""

    def test_place_on_device_cpu(self):
        """Test placing array on CPU device."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()
        dm.configure(DeviceConfig(preferred_device=DeviceType.CPU))

        arr = np.array([1, 2, 3])
        result = dm.place_on_device(arr)

        # Should still be valid array
        np.testing.assert_array_equal(np.asarray(result), [1, 2, 3])
        DeviceManager.reset()

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_place_on_device_gpu(self, require_jax, require_gpu):
        """Test placing array on GPU device."""
        from xpcsviewer.backends import DeviceConfig, DeviceManager, DeviceType

        DeviceManager.reset()
        dm = DeviceManager()
        dm.configure(DeviceConfig(preferred_device=DeviceType.GPU))

        arr = np.array([1, 2, 3])
        result = dm.place_on_device(arr)

        # Verify it's on GPU
        assert dm.is_gpu_enabled
        np.testing.assert_array_equal(np.asarray(result), [1, 2, 3])
        DeviceManager.reset()


class TestJITCompilation:
    """Test JIT compilation functionality."""

    def test_numpy_jit_is_noop(self, numpy_backend):
        """Test that NumPy backend JIT is a no-op."""

        def add(a, b):
            return a + b

        jitted = numpy_backend.jit(add)
        assert jitted is add  # Should be same function

    @pytest.mark.jax
    def test_jax_jit_compiles(self, jax_backend, tolerance_float64):
        """Test that JAX backend JIT actually compiles."""
        import jax.numpy as jnp

        def square(x):
            return x * x

        jitted = jax_backend.jit(square)
        assert jitted is not square  # Should be different (compiled)

        # Should produce same results
        x = jax_backend.array([1.0, 2.0, 3.0])
        result = jax_backend.to_numpy(jitted(x))
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(result, expected, **tolerance_float64)


class TestGradientComputation:
    """Test automatic differentiation functionality."""

    def test_numpy_grad_raises(self, numpy_backend):
        """Test that NumPy backend grad raises NotImplementedError."""

        def f(x):
            return x**2

        with pytest.raises(NotImplementedError, match="does not support"):
            numpy_backend.grad(f)

    @pytest.mark.jax
    def test_jax_grad_computes(self, jax_backend, tolerance_float64):
        """Test that JAX backend grad computes gradients."""

        def f(x):
            return jax_backend.sum(x**2)

        grad_f = jax_backend.grad(f)
        x = jax_backend.array([1.0, 2.0, 3.0])
        result = jax_backend.to_numpy(grad_f(x))

        # d/dx (x^2) = 2x
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(result, expected, **tolerance_float64)

    @pytest.mark.jax
    def test_jax_value_and_grad(self, jax_backend, tolerance_float64):
        """Test that JAX backend value_and_grad computes both."""

        def f(x):
            return jax_backend.sum(x**2)

        vg_f = jax_backend.value_and_grad(f)
        x = jax_backend.array([1.0, 2.0, 3.0])
        value, grad = vg_f(x)

        # f(x) = 1 + 4 + 9 = 14
        assert abs(float(value) - 14.0) < tolerance_float64["atol"]

        # d/dx (x^2) = 2x
        expected_grad = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(
            jax_backend.to_numpy(grad), expected_grad, **tolerance_float64
        )
