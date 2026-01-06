"""Tests for backend detection and initialization.

Tests FR-001: Automatic device detection
Tests FR-010: Environment variable configuration
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest


class TestBackendDetection:
    """Test backend detection and initialization."""

    def test_get_backend_returns_backend_protocol(self):
        """Test that get_backend returns a BackendProtocol instance."""
        from xpcsviewer.backends import BackendProtocol, get_backend

        backend = get_backend()
        assert isinstance(backend, BackendProtocol)

    def test_backend_has_required_properties(self):
        """Test that backend has all required properties."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()

        # Check required properties
        assert hasattr(backend, "name")
        assert hasattr(backend, "supports_gpu")
        assert hasattr(backend, "supports_jit")
        assert hasattr(backend, "supports_grad")
        assert hasattr(backend, "pi")

        # Check property types
        assert isinstance(backend.name, str)
        assert isinstance(backend.supports_gpu, bool)
        assert isinstance(backend.supports_jit, bool)
        assert isinstance(backend.supports_grad, bool)
        assert isinstance(backend.pi, float)

    def test_numpy_backend_properties(self, numpy_backend):
        """Test NumPy backend has correct properties."""
        assert numpy_backend.name == "numpy"
        assert numpy_backend.supports_gpu is False
        assert numpy_backend.supports_jit is False
        assert numpy_backend.supports_grad is False
        assert abs(numpy_backend.pi - np.pi) < 1e-15

    @pytest.mark.jax
    def test_jax_backend_properties(self, jax_backend):
        """Test JAX backend has correct properties."""
        assert jax_backend.name == "jax"
        assert jax_backend.supports_jit is True
        assert jax_backend.supports_grad is True
        assert abs(jax_backend.pi - np.pi) < 1e-15


class TestEnvironmentConfiguration:
    """Test environment variable configuration (FR-010)."""

    def test_xpcs_use_jax_false(self):
        """Test XPCS_USE_JAX=false forces NumPy backend."""
        from xpcsviewer.backends import get_backend, reset_backend

        reset_backend()
        with patch.dict(os.environ, {"XPCS_USE_JAX": "false"}):
            reset_backend()
            backend = get_backend()
            assert backend.name == "numpy"
        reset_backend()

    @pytest.mark.jax
    def test_xpcs_use_jax_true(self, require_jax):
        """Test XPCS_USE_JAX=true forces JAX backend."""
        from xpcsviewer.backends import get_backend, reset_backend

        reset_backend()
        with patch.dict(os.environ, {"XPCS_USE_JAX": "true"}):
            reset_backend()
            backend = get_backend()
            assert backend.name == "jax"
        reset_backend()

    def test_xpcs_use_jax_auto_with_jax_available(self, require_jax):
        """Test XPCS_USE_JAX=auto selects JAX when available."""
        from xpcsviewer.backends import get_backend, reset_backend

        reset_backend()
        with patch.dict(os.environ, {"XPCS_USE_JAX": "auto"}):
            reset_backend()
            backend = get_backend()
            # Should use JAX if available
            assert backend.name == "jax"
        reset_backend()


class TestSetBackend:
    """Test explicit backend setting."""

    def test_set_backend_numpy(self):
        """Test setting NumPy backend explicitly."""
        from xpcsviewer.backends import get_backend, set_backend

        set_backend("numpy")
        backend = get_backend()
        assert backend.name == "numpy"

    @pytest.mark.jax
    def test_set_backend_jax(self, require_jax):
        """Test setting JAX backend explicitly."""
        from xpcsviewer.backends import get_backend, set_backend

        set_backend("jax")
        backend = get_backend()
        assert backend.name == "jax"

    def test_set_backend_invalid_raises(self):
        """Test setting invalid backend raises ValueError."""
        from xpcsviewer.backends import set_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("invalid_backend")

    def test_set_backend_case_insensitive(self):
        """Test backend name is case-insensitive."""
        from xpcsviewer.backends import get_backend, set_backend

        set_backend("NUMPY")
        assert get_backend().name == "numpy"

        set_backend("NumPy")
        assert get_backend().name == "numpy"


class TestBackendArrayCreation:
    """Test array creation methods work correctly."""

    def test_zeros(self, backend):
        """Test zeros creation."""
        arr = backend.zeros((3, 4))
        np_arr = backend.to_numpy(arr)
        assert np_arr.shape == (3, 4)
        assert np.all(np_arr == 0)

    def test_ones(self, backend):
        """Test ones creation."""
        arr = backend.ones((2, 3))
        np_arr = backend.to_numpy(arr)
        assert np_arr.shape == (2, 3)
        assert np.all(np_arr == 1)

    def test_arange(self, backend):
        """Test arange creation."""
        arr = backend.arange(0, 10, 2)
        np_arr = backend.to_numpy(arr)
        np.testing.assert_array_equal(np_arr, [0, 2, 4, 6, 8])

    def test_linspace(self, backend):
        """Test linspace creation."""
        arr = backend.linspace(0, 1, 5)
        np_arr = backend.to_numpy(arr)
        np.testing.assert_allclose(np_arr, [0, 0.25, 0.5, 0.75, 1.0])

    def test_meshgrid(self, backend):
        """Test meshgrid creation."""
        x = backend.arange(0, 3)
        y = backend.arange(0, 2)
        xx, yy = backend.meshgrid(x, y)
        np.testing.assert_array_equal(backend.to_numpy(xx), [[0, 1, 2], [0, 1, 2]])
        np.testing.assert_array_equal(backend.to_numpy(yy), [[0, 0, 0], [1, 1, 1]])


class TestBackendMathOperations:
    """Test mathematical operations."""

    def test_sin_cos(self, backend, tolerance_float64):
        """Test sin and cos."""
        x = backend.linspace(0, backend.pi, 5)
        sin_x = backend.to_numpy(backend.sin(x))
        cos_x = backend.to_numpy(backend.cos(x))

        expected_sin = np.sin(np.linspace(0, np.pi, 5))
        expected_cos = np.cos(np.linspace(0, np.pi, 5))

        np.testing.assert_allclose(sin_x, expected_sin, **tolerance_float64)
        np.testing.assert_allclose(cos_x, expected_cos, **tolerance_float64)

    def test_arctan2(self, backend, tolerance_float64):
        """Test arctan2."""
        y = backend.array([1.0, 1.0, -1.0, -1.0])
        x = backend.array([1.0, -1.0, 1.0, -1.0])
        result = backend.to_numpy(backend.arctan2(y, x))

        expected = np.arctan2([1.0, 1.0, -1.0, -1.0], [1.0, -1.0, 1.0, -1.0])
        np.testing.assert_allclose(result, expected, **tolerance_float64)

    def test_hypot(self, backend, tolerance_float64):
        """Test hypot."""
        x = backend.array([3.0, 5.0, 8.0])
        y = backend.array([4.0, 12.0, 15.0])
        result = backend.to_numpy(backend.hypot(x, y))

        expected = np.hypot([3.0, 5.0, 8.0], [4.0, 12.0, 15.0])
        np.testing.assert_allclose(result, expected, **tolerance_float64)

    def test_exp_log(self, backend, tolerance_float64):
        """Test exp and log."""
        x = backend.array([1.0, 2.0, 3.0])
        exp_x = backend.to_numpy(backend.exp(x))
        log_exp_x = backend.to_numpy(backend.log(backend.exp(x)))

        expected_exp = np.exp([1.0, 2.0, 3.0])
        np.testing.assert_allclose(exp_x, expected_exp, **tolerance_float64)
        np.testing.assert_allclose(log_exp_x, [1.0, 2.0, 3.0], **tolerance_float64)


class TestFloat64Configuration:
    """Test float64 configuration (FR-003)."""

    @pytest.mark.jax
    def test_jax_float64_enabled(self, require_jax):
        """Test that JAX is configured for float64."""
        import jax

        # Create float array and verify it's float64
        from xpcsviewer.backends import get_backend, set_backend

        set_backend("jax")
        backend = get_backend()

        arr = backend.array([1.0, 2.0, 3.0])
        np_arr = backend.to_numpy(arr)
        assert np_arr.dtype == np.float64

    def test_numpy_float64_default(self):
        """Test that NumPy uses float64 by default."""
        from xpcsviewer.backends import get_backend, set_backend

        set_backend("numpy")
        backend = get_backend()

        arr = backend.array([1.0, 2.0, 3.0])
        np_arr = backend.to_numpy(arr)
        assert np_arr.dtype == np.float64
