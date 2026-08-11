"""Tests for numerical equivalence between CPU and GPU (US1).

Tests FR-008: Numerical accuracy across devices
Tests SC-004: 1e-6 relative tolerance for Q-values
"""

from __future__ import annotations

import numpy as np
import pytest

# Tolerance for numerical equivalence (SC-004)
RTOL = 1e-6
ATOL = 1e-8


class TestBackendAbstractionEquivalence:
    """Test equivalence through the backend abstraction layer."""

    def test_numpy_jax_backend_equivalence(self, numpy_backend):
        """Test NumPy and JAX backends produce equivalent results."""
        pytest.importorskip("jax")

        from xpcsviewer.backends._jax_backend import JAXBackend

        jax_backend = JAXBackend()

        # Test data
        rng = np.random.default_rng(42)
        x_np = rng.uniform(-10, 10, (50, 50))
        y_np = rng.uniform(-10, 10, (50, 50))

        # NumPy backend computations
        x_numpy = numpy_backend.array(x_np)
        y_numpy = numpy_backend.array(y_np)
        hypot_numpy = numpy_backend.to_numpy(numpy_backend.hypot(x_numpy, y_numpy))
        arctan2_numpy = numpy_backend.to_numpy(numpy_backend.arctan2(y_numpy, x_numpy))
        sin_numpy = numpy_backend.to_numpy(numpy_backend.sin(x_numpy))

        # JAX backend computations
        x_jax = jax_backend.array(x_np)
        y_jax = jax_backend.array(y_np)
        hypot_jax = jax_backend.to_numpy(jax_backend.hypot(x_jax, y_jax))
        arctan2_jax = jax_backend.to_numpy(jax_backend.arctan2(y_jax, x_jax))
        sin_jax = jax_backend.to_numpy(jax_backend.sin(x_jax))

        # Compare results
        np.testing.assert_allclose(hypot_numpy, hypot_jax, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(arctan2_numpy, arctan2_jax, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(sin_numpy, sin_jax, rtol=RTOL, atol=ATOL)

    def test_statistical_operations_equivalence(self, numpy_backend):
        """Test statistical operations produce equivalent results."""
        pytest.importorskip("jax")

        from xpcsviewer.backends._jax_backend import JAXBackend

        jax_backend = JAXBackend()

        # Test data
        rng = np.random.default_rng(42)
        data_np = rng.uniform(0, 100, (100, 100))

        # NumPy backend
        data_numpy = numpy_backend.array(data_np)
        mean_numpy = numpy_backend.to_numpy(numpy_backend.mean(data_numpy))
        std_numpy = numpy_backend.to_numpy(numpy_backend.std(data_numpy))

        # JAX backend
        data_jax = jax_backend.array(data_np)
        mean_jax = jax_backend.to_numpy(jax_backend.mean(data_jax))
        std_jax = jax_backend.to_numpy(jax_backend.std(data_jax))

        np.testing.assert_allclose(mean_numpy, mean_jax, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(std_numpy, std_jax, rtol=RTOL, atol=ATOL)
