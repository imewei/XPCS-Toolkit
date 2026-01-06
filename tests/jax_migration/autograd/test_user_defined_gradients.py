"""Tests for user-defined gradient functions (T067a).

Tests auto-diff with user-defined objective functions per FR-011 (US4).
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
class TestUserDefinedGradients:
    """Tests for user-defined gradient computation."""

    def test_custom_loss_function_gradient(self) -> None:
        """Test gradient of user-defined loss function."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("User-defined gradients only apply to JAX backend")

        # User-defined custom loss
        def custom_loss(params, data, target):
            """Custom weighted MSE loss."""
            weights = jnp.array([1.0, 2.0, 3.0])
            predictions = params[0] * data + params[1]
            residuals = predictions - target
            return jnp.sum(weights * residuals**2)

        # Test data
        data = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([2.0, 4.0, 6.0])  # y = 2x
        params = jnp.array([1.5, 0.5])  # Initial guess

        # Compute gradient
        grad_fn = jax.grad(custom_loss, argnums=0)
        gradients = grad_fn(params, data, target)

        # Gradients should be non-zero
        assert not jnp.isnan(gradients[0])
        assert not jnp.isnan(gradients[1])
        assert jnp.any(gradients != 0)

    def test_custom_g2_model_gradient(self) -> None:
        """Test gradient for custom g2 correlation model."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Custom model gradients only apply to JAX backend")

        def g2_model(params, tau):
            """g2 = baseline + contrast * exp(-2*(tau/tau_c)^beta)"""
            baseline, contrast, tau_c, beta = params
            return baseline + contrast * jnp.exp(-2 * (tau / tau_c) ** beta)

        def g2_loss(params, tau, data):
            """MSE loss for g2 fitting."""
            predictions = g2_model(params, tau)
            return jnp.mean((predictions - data) ** 2)

        # Test data
        tau = jnp.logspace(-3, 1, 50)
        true_params = jnp.array([1.0, 0.3, 0.1, 1.5])
        data = g2_model(true_params, tau)

        # Initial guess (offset from truth)
        init_params = jnp.array([1.1, 0.25, 0.15, 1.3])

        # Compute gradient
        grad_fn = jax.grad(g2_loss, argnums=0)
        gradients = grad_fn(init_params, tau, data)

        # All 4 gradients should be computed
        assert gradients.shape == (4,)
        assert not jnp.any(jnp.isnan(gradients))

    def test_value_and_grad_custom_function(self) -> None:
        """Test value_and_grad for custom function."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Value and grad only applies to JAX backend")

        def custom_objective(x):
            """User-defined objective with multiple terms."""
            regularization = 0.1 * jnp.sum(x**2)
            data_term = jnp.sum(jnp.sin(x))
            return data_term + regularization

        x = jnp.array([0.5, 1.0, 1.5, 2.0])

        # Use value_and_grad
        val_grad_fn = jax.value_and_grad(custom_objective)
        value, gradients = val_grad_fn(x)

        # Check value
        expected_value = float(custom_objective(x))
        assert abs(float(value) - expected_value) < 1e-10

        # Check gradients: d/dx[sin(x) + 0.1*x^2] = cos(x) + 0.2*x
        expected_grad = jnp.cos(x) + 0.2 * x
        np.testing.assert_allclose(
            np.asarray(gradients), np.asarray(expected_grad), rtol=1e-5
        )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestGradientAPIUsability:
    """Tests for gradient API usability and ergonomics."""

    def test_backend_grad_with_argnums(self) -> None:
        """Test backend.grad with multiple arguments."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Gradient API only applies to JAX backend")

        def multi_arg_loss(x, y, z):
            return jnp.sum(x**2) + 2 * jnp.sum(y**2) + 3 * jnp.sum(z**2)

        x = jnp.ones(3)
        y = jnp.ones(3) * 2
        z = jnp.ones(3) * 3

        # Gradient with respect to first arg
        grad_x = backend.grad(multi_arg_loss, argnums=0)(x, y, z)
        np.testing.assert_allclose(np.asarray(grad_x), 2 * np.asarray(x), rtol=1e-5)

        # Gradient with respect to second arg
        grad_y = backend.grad(multi_arg_loss, argnums=1)(x, y, z)
        np.testing.assert_allclose(np.asarray(grad_y), 4 * np.asarray(y), rtol=1e-5)

        # Gradient with respect to all args
        grad_all = backend.grad(multi_arg_loss, argnums=(0, 1, 2))(x, y, z)
        assert len(grad_all) == 3

    def test_jit_combined_with_grad(self) -> None:
        """Test JIT compilation combined with gradient computation."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("JIT+grad only applies to JAX backend")

        def loss_fn(params):
            return jnp.sum(params**2)

        # JIT-compiled gradient function
        jit_grad = jax.jit(jax.grad(loss_fn))

        params = jnp.array([1.0, 2.0, 3.0])

        # Warmup
        _ = jit_grad(params)

        # Compute
        gradients = jit_grad(params)

        expected = 2 * params
        np.testing.assert_allclose(
            np.asarray(gradients), np.asarray(expected), rtol=1e-5
        )

    def test_gradient_through_control_flow(self) -> None:
        """Test gradients work through JAX control flow primitives."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Control flow gradients only apply to JAX backend")

        def loss_with_cond(x):
            """Loss with conditional logic."""
            return jax.lax.cond(x > 0, lambda v: v**2, lambda v: -(v**2), x)

        # Positive x
        grad_pos = jax.grad(loss_with_cond)(jnp.array(2.0))
        assert float(grad_pos) == 4.0  # d/dx[x^2] = 2x at x=2

        # Negative x
        grad_neg = jax.grad(loss_with_cond)(jnp.array(-2.0))
        assert float(grad_neg) == 4.0  # d/dx[-x^2] = -2x at x=-2


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestGradientEdgeCases:
    """Tests for gradient edge cases."""

    def test_gradient_nan_handling(self) -> None:
        """Test gradient handling of NaN values."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("NaN handling only applies to JAX backend")

        def safe_log_loss(x):
            """Log loss with safe handling."""
            # Use where to avoid log(0)
            safe_x = jnp.where(x > 0, x, 1e-10)
            return -jnp.sum(jnp.log(safe_x))

        x = jnp.array([0.5, 0.1, 0.01])
        grad_fn = jax.grad(safe_log_loss)
        gradients = grad_fn(x)

        # Gradients should not contain NaN
        assert not jnp.any(jnp.isnan(gradients))

    def test_gradient_large_array(self) -> None:
        """Test gradient computation on large arrays."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Large array gradients only apply to JAX backend")

        def large_array_loss(x):
            return jnp.mean(x**2)

        # Large array (1M elements)
        x = jnp.ones(1000000) * 0.5

        grad_fn = jax.jit(jax.grad(large_array_loss))

        # Warmup
        _ = grad_fn(x)

        # Compute
        gradients = grad_fn(x)

        # Gradient of mean(x^2) is 2x/n
        n = len(x)
        expected = 2 * x / n
        np.testing.assert_allclose(
            np.asarray(gradients), np.asarray(expected), rtol=1e-5
        )

    def test_gradient_matrix_operations(self) -> None:
        """Test gradients through matrix operations match finite differences."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Matrix gradients only apply to JAX backend")

        def matrix_loss(A):
            """Loss involving matrix multiplication."""
            return jnp.sum(A @ A.T)

        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        grad_fn = jax.grad(matrix_loss)
        gradient = grad_fn(A)

        # Verify gradient via finite differences
        eps = 1e-5
        fd_gradient = np.zeros_like(A)
        for i in range(2):
            for j in range(2):
                A_plus = A.at[i, j].add(eps)
                A_minus = A.at[i, j].add(-eps)
                fd_gradient[i, j] = (matrix_loss(A_plus) - matrix_loss(A_minus)) / (
                    2 * eps
                )

        np.testing.assert_allclose(np.asarray(gradient), fd_gradient, rtol=1e-4)
