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


class TestCPUGPUNumericalEquivalence:
    """Test numerical equivalence between CPU and GPU computations."""

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_basic_array_operations_equivalence(self, require_jax, require_gpu):
        """Test basic array operations produce equivalent results on CPU and GPU."""
        import jax
        import jax.numpy as jnp

        # Create test data
        np_data = np.random.default_rng(42).random((100, 100))

        # CPU computation
        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]

        x_cpu = jax.device_put(jnp.array(np_data), cpu_device)
        x_gpu = jax.device_put(jnp.array(np_data), gpu_device)

        # Test sin
        sin_cpu = np.asarray(jnp.sin(x_cpu))
        sin_gpu = np.asarray(jnp.sin(x_gpu))
        np.testing.assert_allclose(sin_cpu, sin_gpu, rtol=RTOL, atol=ATOL)

        # Test cos
        cos_cpu = np.asarray(jnp.cos(x_cpu))
        cos_gpu = np.asarray(jnp.cos(x_gpu))
        np.testing.assert_allclose(cos_cpu, cos_gpu, rtol=RTOL, atol=ATOL)

        # Test exp
        exp_cpu = np.asarray(jnp.exp(x_cpu))
        exp_gpu = np.asarray(jnp.exp(x_gpu))
        np.testing.assert_allclose(exp_cpu, exp_gpu, rtol=RTOL, atol=ATOL)

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_arctan2_equivalence(self, require_jax, require_gpu):
        """Test arctan2 produces equivalent results on CPU and GPU."""
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        y_np = rng.uniform(-10, 10, (100, 100))
        x_np = rng.uniform(-10, 10, (100, 100))

        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]

        y_cpu = jax.device_put(jnp.array(y_np), cpu_device)
        x_cpu = jax.device_put(jnp.array(x_np), cpu_device)
        y_gpu = jax.device_put(jnp.array(y_np), gpu_device)
        x_gpu = jax.device_put(jnp.array(x_np), gpu_device)

        result_cpu = np.asarray(jnp.arctan2(y_cpu, x_cpu))
        result_gpu = np.asarray(jnp.arctan2(y_gpu, x_gpu))

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=RTOL, atol=ATOL)

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_hypot_equivalence(self, require_jax, require_gpu):
        """Test hypot produces equivalent results on CPU and GPU."""
        import jax
        import jax.numpy as jnp

        rng = np.random.default_rng(42)
        x_np = rng.uniform(-100, 100, (100, 100))
        y_np = rng.uniform(-100, 100, (100, 100))

        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]

        x_cpu = jax.device_put(jnp.array(x_np), cpu_device)
        y_cpu = jax.device_put(jnp.array(y_np), cpu_device)
        x_gpu = jax.device_put(jnp.array(x_np), gpu_device)
        y_gpu = jax.device_put(jnp.array(y_np), gpu_device)

        result_cpu = np.asarray(jnp.hypot(x_cpu, y_cpu))
        result_gpu = np.asarray(jnp.hypot(x_gpu, y_gpu))

        np.testing.assert_allclose(result_cpu, result_gpu, rtol=RTOL, atol=ATOL)

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_meshgrid_equivalence(self, require_jax, require_gpu):
        """Test meshgrid produces equivalent results on CPU and GPU."""
        import jax
        import jax.numpy as jnp

        x_np = np.linspace(-1, 1, 256)
        y_np = np.linspace(-1, 1, 256)

        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]

        x_cpu = jax.device_put(jnp.array(x_np), cpu_device)
        y_cpu = jax.device_put(jnp.array(y_np), cpu_device)
        x_gpu = jax.device_put(jnp.array(x_np), gpu_device)
        y_gpu = jax.device_put(jnp.array(y_np), gpu_device)

        xx_cpu, yy_cpu = jnp.meshgrid(x_cpu, y_cpu)
        xx_gpu, yy_gpu = jnp.meshgrid(x_gpu, y_gpu)

        np.testing.assert_allclose(
            np.asarray(xx_cpu), np.asarray(xx_gpu), rtol=RTOL, atol=ATOL
        )
        np.testing.assert_allclose(
            np.asarray(yy_cpu), np.asarray(yy_gpu), rtol=RTOL, atol=ATOL
        )


class TestQMapLikeComputation:
    """Test Q-map-like computations for CPU/GPU equivalence."""

    @pytest.mark.jax
    @pytest.mark.gpu
    def test_qmap_computation_equivalence(self, require_jax, require_gpu):
        """Test Q-map computation produces equivalent results."""
        import jax
        import jax.numpy as jnp

        # Detector geometry parameters (typical XPCS values)
        det_distance = 5.0  # meters
        pixel_size = 75e-6  # meters
        wavelength = 1.0e-10  # meters (1 Angstrom)
        beam_center_x = 256
        beam_center_y = 256
        shape = (512, 512)

        # Create pixel coordinates
        y_np = np.arange(shape[0])
        x_np = np.arange(shape[1])
        xx_np, yy_np = np.meshgrid(x_np, y_np, indexing="xy")

        # Pixel positions relative to beam center
        dx_np = (xx_np - beam_center_x) * pixel_size
        dy_np = (yy_np - beam_center_y) * pixel_size

        # CPU computation
        cpu_device = jax.devices("cpu")[0]
        gpu_device = jax.devices("gpu")[0]

        dx_cpu = jax.device_put(jnp.array(dx_np), cpu_device)
        dy_cpu = jax.device_put(jnp.array(dy_np), cpu_device)
        dx_gpu = jax.device_put(jnp.array(dx_np), gpu_device)
        dy_gpu = jax.device_put(jnp.array(dy_np), gpu_device)

        # Q-map computation
        def compute_q(dx, dy, det_dist, wl):
            r = jnp.hypot(dx, dy)
            tth = jnp.arctan2(r, det_dist)
            q = 4 * jnp.pi * jnp.sin(tth / 2) / wl
            phi = jnp.rad2deg(jnp.arctan2(dy, dx))
            return q, phi

        q_cpu, phi_cpu = compute_q(dx_cpu, dy_cpu, det_distance, wavelength)
        q_gpu, phi_gpu = compute_q(dx_gpu, dy_gpu, det_distance, wavelength)

        # Verify equivalence within tolerance (SC-004)
        np.testing.assert_allclose(
            np.asarray(q_cpu),
            np.asarray(q_gpu),
            rtol=RTOL,
            atol=ATOL,
            err_msg="Q values differ between CPU and GPU beyond tolerance",
        )
        np.testing.assert_allclose(
            np.asarray(phi_cpu),
            np.asarray(phi_gpu),
            rtol=RTOL,
            atol=ATOL,
            err_msg="Phi values differ between CPU and GPU beyond tolerance",
        )


class TestBackendAbstractionEquivalence:
    """Test equivalence through the backend abstraction layer."""

    def test_numpy_jax_backend_equivalence(self, numpy_backend, tolerance_float64):
        """Test NumPy and JAX backends produce equivalent results."""
        pytest.importorskip("jax")

        from xpcsviewer.backends import set_backend
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

    def test_statistical_operations_equivalence(self, numpy_backend, tolerance_float64):
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
