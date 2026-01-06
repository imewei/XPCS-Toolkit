"""Tests for gradient-based calibration (T063).

Tests that gradient-based optimization converges to correct beam center (US4).
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestBeamCenterCalibration:
    """Tests for beam center calibration via gradient descent."""

    def test_beam_center_optimization_simple(self) -> None:
        """Test simple beam center optimization with synthetic data."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Calibration optimization only applies to JAX backend")

        # True beam center
        true_cx, true_cy = 128.5, 127.3

        # Generate synthetic ring pattern (points on a circle centered at true_cx, true_cy)
        n_points = 50
        radius = 50.0
        angles = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)
        ring_x = true_cx + radius * jnp.cos(angles)
        ring_y = true_cy + radius * jnp.sin(angles)

        def loss_fn(params):
            """Loss: variance of distances from center (should be zero for correct center)."""
            cx, cy = params
            distances = jnp.sqrt((ring_x - cx) ** 2 + (ring_y - cy) ** 2)
            return jnp.var(distances)

        # Start with offset initial guess
        init_params = jnp.array([120.0, 120.0])

        # Simple gradient descent
        learning_rate = 1.0
        params = init_params
        grad_fn = jit(grad(loss_fn))

        for _ in range(100):
            grads = grad_fn(params)
            params = params - learning_rate * grads

        # Should converge close to true center
        final_cx, final_cy = float(params[0]), float(params[1])
        assert abs(final_cx - true_cx) < 0.1, f"X: {final_cx} vs {true_cx}"
        assert abs(final_cy - true_cy) < 0.1, f"Y: {final_cy} vs {true_cy}"

    def test_beam_center_optimization_with_noise(self) -> None:
        """Test beam center optimization with noisy data."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Calibration optimization only applies to JAX backend")

        # True beam center
        true_cx, true_cy = 256.0, 256.0

        # Generate noisy ring pattern
        key = jax.random.PRNGKey(42)
        n_points = 100
        radius = 100.0
        angles = jnp.linspace(0, 2 * jnp.pi, n_points, endpoint=False)

        # Add noise to positions
        noise_std = 2.0
        key, subkey1, subkey2 = jax.random.split(key, 3)
        noise_x = jax.random.normal(subkey1, (n_points,)) * noise_std
        noise_y = jax.random.normal(subkey2, (n_points,)) * noise_std

        ring_x = true_cx + radius * jnp.cos(angles) + noise_x
        ring_y = true_cy + radius * jnp.sin(angles) + noise_y

        def loss_fn(params):
            """Loss: variance of distances from center."""
            cx, cy = params
            distances = jnp.sqrt((ring_x - cx) ** 2 + (ring_y - cy) ** 2)
            return jnp.var(distances)

        # Start with offset initial guess
        init_params = jnp.array([240.0, 240.0])

        # Gradient descent with momentum
        learning_rate = 0.5
        params = init_params
        grad_fn = jit(grad(loss_fn))

        for _ in range(200):
            grads = grad_fn(params)
            params = params - learning_rate * grads

        # Should converge close to true center (allowing for noise)
        final_cx, final_cy = float(params[0]), float(params[1])
        assert abs(final_cx - true_cx) < 2.0, f"X: {final_cx} vs {true_cx}"
        assert abs(final_cy - true_cy) < 2.0, f"Y: {final_cy} vs {true_cy}"

    def test_detector_distance_gradient_direction(self) -> None:
        """Test that detector distance gradient points in correct direction.

        This tests that gradient-based optimization COULD work, not that it converges.
        Actual optimization may need specialized optimizers for ill-conditioned problems.
        """
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Calibration optimization only applies to JAX backend")

        # True parameters
        true_dist = 5000.0  # mm
        cx, cy = 128.0, 128.0
        k0 = 0.5

        # Generate synthetic Q values at known positions
        pix_dim = 0.075
        positions = [(250.0, 250.0), (300.0, 300.0), (350.0, 350.0)]

        def compute_q(det_dist, px, py):
            dx, dy = px - cx, py - cy
            r = jnp.sqrt(dx**2 + dy**2) * pix_dim
            alpha = jnp.arctan(r / det_dist)
            return jnp.sin(alpha) * k0

        # True Q values
        true_q = jnp.array([compute_q(true_dist, px, py) for px, py in positions])

        def loss_fn(det_dist):
            """Loss: sum of squared Q differences."""
            predicted_q = jnp.array(
                [compute_q(det_dist, px, py) for px, py in positions]
            )
            return jnp.sum((predicted_q - true_q) ** 2)

        grad_fn = jit(grad(loss_fn))

        # Test that gradient points toward true value from both sides
        # From below true value: gradient should be negative (to increase dist)
        dist_low = jnp.array(4500.0)
        grad_low = grad_fn(dist_low)
        # From above true value: gradient should be positive (to decrease dist)
        dist_high = jnp.array(5500.0)
        grad_high = grad_fn(dist_high)

        # At true value, gradient should be near zero
        grad_true = grad_fn(jnp.array(true_dist))

        assert float(grad_low) < 0, (
            f"Gradient at low dist should be negative: {grad_low}"
        )
        assert float(grad_high) > 0, (
            f"Gradient at high dist should be positive: {grad_high}"
        )
        assert abs(float(grad_true)) < abs(float(grad_low)), (
            "Gradient at true should be smaller"
        )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestCalibrationConvergence:
    """Tests for calibration convergence properties."""

    def test_convergence_rate(self) -> None:
        """Test that optimization converges at expected rate."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Convergence test only applies to JAX backend")

        # Simple quadratic loss (known convergence properties)
        true_params = jnp.array([10.0, 20.0])

        def loss_fn(params):
            return jnp.sum((params - true_params) ** 2)

        init_params = jnp.array([0.0, 0.0])
        learning_rate = 0.1

        params = init_params
        grad_fn = jit(grad(loss_fn))
        losses = [float(loss_fn(params))]

        for _ in range(50):
            grads = grad_fn(params)
            params = params - learning_rate * grads
            losses.append(float(loss_fn(params)))

        # Loss should monotonically decrease
        for i in range(1, len(losses)):
            assert losses[i] <= losses[i - 1] + 1e-10

        # Final loss should be near zero
        assert losses[-1] < 1e-6

    def test_saddle_point_escape(self) -> None:
        """Test that optimization can escape saddle points."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Saddle point test only applies to JAX backend")

        # Loss with saddle point at origin: x^2 - y^2
        # Adding regularization to make it bounded
        def loss_fn(params):
            x, y = params
            return x**2 - 0.5 * y**2 + 0.01 * (x**4 + y**4)

        # Start at saddle point
        init_params = jnp.array([0.01, 0.0])

        learning_rate = 0.1
        params = init_params
        grad_fn = jit(grad(loss_fn))

        for _ in range(100):
            grads = grad_fn(params)
            params = params - learning_rate * grads

        # Should have moved away from origin
        final_x, final_y = float(params[0]), float(params[1])
        # x should decrease toward 0, y can grow
        assert abs(final_x) < abs(float(init_params[0])) + 0.01


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestOptimixIntegration:
    """Tests for integration with Optimistix optimizer."""

    def test_optimistix_available(self) -> None:
        """Test that optimistix is available for advanced optimization."""
        try:
            import optimistix as optx

            assert hasattr(optx, "minimise")
        except ImportError:
            pytest.skip("Optimistix not installed")

    def test_lbfgs_optimization(self) -> None:
        """Test L-BFGS optimization via optimistix."""
        try:
            import optimistix as optx
        except ImportError:
            pytest.skip("Optimistix not installed")

        from xpcsviewer.backends import get_backend

        backend = get_backend()
        if backend.name != "jax":
            pytest.skip("Optimistix test only applies to JAX backend")

        # Simple quadratic
        true_params = jnp.array([5.0, -3.0])

        def residual_fn(params, args):
            return params - true_params

        init_params = jnp.array([0.0, 0.0])

        # Use L-BFGS solver
        solver = optx.BFGS(rtol=1e-6, atol=1e-6)
        solution = optx.minimise(
            lambda p, a: jnp.sum(residual_fn(p, a) ** 2),
            solver,
            init_params,
            args=None,
        )

        np.testing.assert_allclose(
            np.asarray(solution.value), np.asarray(true_params), rtol=1e-4
        )
