"""Tests for Q-map gradient computation (T062).

Tests that gradients can be computed for Q-map parameters using JAX auto-diff (US4).
"""

from __future__ import annotations

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
class TestQmapGradients:
    """Tests for Q-map gradient computation."""

    def test_qmap_gradient_wrt_beam_center(self) -> None:
        """Test gradient of Q-map with respect to beam center."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient computation only applies to JAX backend")

        # Define a simple Q-map-like function for testing gradients
        def q_at_point(center_x, center_y, px, py, pix_dim, det_dist, k0):
            """Compute Q at a single point given beam center."""
            dx = px - center_x
            dy = py - center_y
            r = jnp.sqrt(dx**2 + dy**2) * pix_dim
            alpha = jnp.arctan(r / det_dist)
            return jnp.sin(alpha) * k0

        # Test parameters
        center_x = 128.0
        center_y = 128.0
        px, py = 200.0, 200.0  # Detector pixel
        pix_dim = 0.075  # mm
        det_dist = 5000.0  # mm
        k0 = 0.5  # Approximate wavevector

        # Compute gradient with respect to beam center
        grad_fn = jax.grad(q_at_point, argnums=(0, 1))
        grad_x, grad_y = grad_fn(center_x, center_y, px, py, pix_dim, det_dist, k0)

        # Gradients should be non-zero (Q changes with beam center)
        assert not jnp.isnan(grad_x)
        assert not jnp.isnan(grad_y)
        assert abs(float(grad_x)) > 1e-10 or abs(float(grad_y)) > 1e-10

    def test_qmap_gradient_wrt_detector_distance(self) -> None:
        """Test gradient of Q-map with respect to detector distance."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient computation only applies to JAX backend")

        def q_at_point(det_dist, center_x, center_y, px, py, pix_dim, k0):
            """Compute Q at a single point given detector distance."""
            dx = px - center_x
            dy = py - center_y
            r = jnp.sqrt(dx**2 + dy**2) * pix_dim
            alpha = jnp.arctan(r / det_dist)
            return jnp.sin(alpha) * k0

        # Test parameters
        det_dist = 5000.0
        center_x, center_y = 128.0, 128.0
        px, py = 200.0, 200.0
        pix_dim = 0.075
        k0 = 0.5

        # Compute gradient with respect to detector distance
        grad_fn = jax.grad(q_at_point, argnums=0)
        grad_dist = grad_fn(det_dist, center_x, center_y, px, py, pix_dim, k0)

        # Gradient should be negative (larger distance = smaller Q)
        assert not jnp.isnan(grad_dist)
        assert float(grad_dist) < 0

    def test_qmap_value_and_grad(self) -> None:
        """Test value_and_grad for Q-map computation."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient computation only applies to JAX backend")

        def total_q_squared(center_x, center_y, pixels, pix_dim, det_dist, k0):
            """Sum of Q^2 over all pixels (differentiable loss function)."""
            q_sum = 0.0
            for px, py in pixels:
                dx = px - center_x
                dy = py - center_y
                r = jnp.sqrt(dx**2 + dy**2) * pix_dim
                alpha = jnp.arctan(r / det_dist)
                q = jnp.sin(alpha) * k0
                q_sum = q_sum + q**2
            return q_sum

        # Test parameters
        center_x, center_y = 128.0, 128.0
        pixels = [(130.0, 130.0), (150.0, 150.0), (200.0, 200.0)]
        pix_dim = 0.075
        det_dist = 5000.0
        k0 = 0.5

        # Use value_and_grad
        val_grad_fn = jax.value_and_grad(total_q_squared, argnums=(0, 1))
        value, (grad_x, grad_y) = val_grad_fn(
            center_x, center_y, pixels, pix_dim, det_dist, k0
        )

        # Both value and gradients should be valid
        assert not jnp.isnan(value)
        assert float(value) > 0
        assert not jnp.isnan(grad_x)
        assert not jnp.isnan(grad_y)

    def test_backend_grad_wrapper(self) -> None:
        """Test backend.grad wrapper for gradient computation."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient computation only applies to JAX backend")

        def simple_loss(x):
            return backend.sum(x**2)

        x = backend.linspace(-1, 1, 10)

        # Use backend.grad
        grad_fn = backend.grad(simple_loss)
        gradients = grad_fn(x)

        # Gradient of x^2 is 2x
        expected = 2 * x
        np.testing.assert_allclose(
            np.asarray(gradients), np.asarray(expected), rtol=1e-5
        )

    def test_backend_value_and_grad_wrapper(self) -> None:
        """Test backend.value_and_grad wrapper."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient computation only applies to JAX backend")

        def simple_loss(x):
            return backend.sum(x**2)

        x = backend.linspace(-1, 1, 10)

        # Use backend.value_and_grad
        val_grad_fn = backend.value_and_grad(simple_loss)
        value, gradients = val_grad_fn(x)

        # Value should be sum of squares
        expected_value = float(backend.sum(x**2))
        assert abs(float(value) - expected_value) < 1e-10

        # Gradient of x^2 is 2x
        expected_grad = 2 * x
        np.testing.assert_allclose(
            np.asarray(gradients), np.asarray(expected_grad), rtol=1e-5
        )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestGradientNumericalAccuracy:
    """Tests for gradient numerical accuracy."""

    def test_gradient_finite_difference_comparison(self) -> None:
        """Test JAX gradients match finite difference approximation."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient comparison only applies to JAX backend")

        def loss_fn(x):
            return jnp.sum(jnp.sin(x) * jnp.cos(x))

        x = jnp.linspace(0, 2 * jnp.pi, 20)
        eps = 1e-5

        # JAX gradient
        jax_grad = jax.grad(loss_fn)(x)

        # Finite difference gradient
        fd_grad = []
        for i in range(len(x)):
            x_plus = x.at[i].add(eps)
            x_minus = x.at[i].add(-eps)
            fd = (loss_fn(x_plus) - loss_fn(x_minus)) / (2 * eps)
            fd_grad.append(float(fd))

        np.testing.assert_allclose(
            np.asarray(jax_grad), np.array(fd_grad), rtol=1e-4, atol=1e-6
        )

    def test_second_order_gradients(self) -> None:
        """Test second-order gradients (Hessian)."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Hessian computation only applies to JAX backend")

        def simple_loss(x):
            return jnp.sum(x**3)

        x = jnp.array([1.0, 2.0, 3.0])

        # First derivative of x^3 is 3x^2
        # Second derivative is 6x
        hessian_fn = jax.hessian(simple_loss)
        hessian = hessian_fn(x)

        # Hessian should be diagonal with values 6*x_i
        expected_diag = 6 * x
        np.testing.assert_allclose(
            np.diag(np.asarray(hessian)), np.asarray(expected_diag), rtol=1e-5
        )
