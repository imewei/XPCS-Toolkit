"""Gradient-based calibration for detector geometry.

This module provides functions for automatic calibration of detector
geometry parameters (beam center, detector distance) using gradient-based
optimization with JAX.

Implements T066 for US4 (Gradient-Based Calibration).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

import numpy as np

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def refine_beam_center(
    ring_points: NDArray[np.floating],
    initial_center: tuple[float, float],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    learning_rate: float = 1.0,
) -> tuple[float, float, dict]:
    """Refine beam center using gradient-based optimization.

    Given a set of points that should form a ring (e.g., from a diffraction
    pattern), find the center that minimizes the variance of distances
    from the center to all points.

    Args:
        ring_points: Nx2 array of (x, y) coordinates forming a ring pattern
        initial_center: Initial guess for (center_x, center_y)
        max_iterations: Maximum optimization iterations
        tolerance: Convergence tolerance for loss change
        learning_rate: Initial learning rate for gradient descent

    Returns:
        Tuple of (refined_center_x, refined_center_y, diagnostics_dict)
        where diagnostics_dict contains optimization history.

    Raises:
        RuntimeError: If JAX backend is not available.
    """
    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError(
            "Gradient-based calibration requires JAX backend. "
            "Set XPCS_USE_JAX=1 to enable."
        )

    import jax
    import jax.numpy as jnp

    # Convert inputs to JAX arrays
    points = jnp.array(ring_points)
    params = jnp.array([initial_center[0], initial_center[1]])

    def loss_fn(params):
        """Variance of distances from center to ring points."""
        cx, cy = params
        # Add epsilon before sqrt to prevent an undefined gradient when a
        # calibration point coincides exactly with the beam center
        # (i.e., squared distance == 0 => d/dx sqrt(0) is undefined). (BUG-034)
        distances = jnp.sqrt(
            (points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2 + 1e-12
        )
        return jnp.var(distances)

    # Single JIT over value_and_grad: one compilation, one kernel dispatch per step.
    # Previously two separate jax.jit calls (grad_fn + loss_fn_jit) caused a redundant
    # XLA compilation and an extra device round-trip on every convergence-check iteration.
    val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    # Optimization loop — prime the cache with the first forward+backward pass.
    loss, grads = val_and_grad_fn(params)
    losses = [float(loss)]
    lr = learning_rate

    for i in range(max_iterations):
        params = params - lr * grads

        # Recompute value+grad together (one kernel dispatch)
        loss, grads = val_and_grad_fn(params)

        # Check convergence every 10 steps to reduce host-device sync overhead
        if i % 10 == 0 or i == max_iterations - 1:
            loss_val = float(loss)
            losses.append(loss_val)

            if np.isnan(loss_val):
                logger.warning(
                    "NaN loss detected in beam center refinement, stopping early"
                )
                break

            # Check convergence
            if len(losses) >= 2 and abs(losses[-1] - losses[-2]) < tolerance:
                logger.debug(
                    f"Beam center refinement converged after {i + 1} iterations"
                )
                break

        # Adaptive learning rate decay
        if i > 0 and i % 20 == 0:
            lr *= 0.5

    final_cx, final_cy = float(params[0]), float(params[1])

    diagnostics = {
        "iterations": len(losses) - 1,
        "losses": np.array(losses),
        "converged": abs(losses[-1] - losses[-2]) < tolerance
        if len(losses) >= 2
        else False,
        "final_loss": losses[-1],
        "initial_center": initial_center,
    }

    logger.info(
        f"Beam center refined: ({initial_center[0]:.2f}, {initial_center[1]:.2f}) -> "
        f"({final_cx:.2f}, {final_cy:.2f})"
    )

    return final_cx, final_cy, diagnostics


def compute_center_from_ring(
    image: NDArray[np.floating],
    mask: NDArray[np.bool_] | None = None,
    intensity_threshold: float | None = None,
    initial_center: tuple[float, float] | None = None,
) -> tuple[float, float, dict]:
    """Compute beam center from diffraction ring pattern.

    Detects bright pixels forming a ring and refines center using
    gradient-based optimization.

    Args:
        image: 2D detector image with diffraction ring
        mask: Optional boolean mask (True = valid pixels)
        intensity_threshold: Threshold for detecting ring pixels.
            If None, uses mean + 2*std.
        initial_center: Initial guess. If None, uses image center.

    Returns:
        Tuple of (center_x, center_y, diagnostics_dict)

    Raises:
        ValueError: If no ring pixels detected or insufficient points.
    """
    # Apply mask
    if mask is not None:
        valid_image = np.where(mask, image, np.nan)
    else:
        valid_image = image.copy()

    # Determine threshold
    if intensity_threshold is None:
        valid_values = valid_image[~np.isnan(valid_image)]
        intensity_threshold = np.mean(valid_values) + 2 * np.std(valid_values)

    # Find ring pixels
    ring_mask = valid_image > intensity_threshold
    ring_coords = np.array(np.where(ring_mask)).T  # Nx2 array of (row, col)

    if len(ring_coords) < 10:
        raise ValueError(
            f"Insufficient ring pixels detected ({len(ring_coords)}). "
            "Adjust intensity_threshold."
        )

    # Convert to x, y (column, row)
    ring_points = ring_coords[:, ::-1].astype(np.float64)

    # Initial center guess
    if initial_center is None:
        initial_center = (image.shape[1] / 2, image.shape[0] / 2)

    return refine_beam_center(ring_points, initial_center)


def create_calibration_objective(
    target_q_values: NDArray[np.floating],
    pixel_positions: list[tuple[float, float]],
    pix_dim: float,
    k0: float,
) -> Callable:
    """Create a differentiable objective function for geometry calibration.

    The objective measures how well predicted Q-values match target Q-values
    for a set of pixel positions, given geometry parameters.

    Args:
        target_q_values: Array of target Q-values at each position
        pixel_positions: List of (x, y) pixel coordinates
        pix_dim: Pixel size in mm
        k0: Wavevector magnitude (2π/λ)

    Returns:
        Callable objective function(params) -> loss where
        params = [center_x, center_y, det_dist].

    Example:
        >>> objective = create_calibration_objective(target_q, positions, pix_dim, k0)
        >>> loss = objective(jnp.array([128.0, 128.0, 5000.0]))
    """
    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError("Calibration objective requires JAX backend.")

    import jax.numpy as jnp

    target_q = jnp.array(target_q_values)
    # Pre-convert pixel positions to a JAX array so the objective is fully
    # vectorised and traceable without a Python for-loop (BUG-057).
    positions_arr = jnp.array([(float(x), float(y)) for x, y in pixel_positions])

    def objective(params):
        """Sum of squared Q-value differences (vectorised, JIT-traceable)."""
        cx, cy, det_dist = params[0], params[1], params[2]
        dx = positions_arr[:, 0] - cx
        dy = positions_arr[:, 1] - cy
        r = jnp.sqrt(dx**2 + dy**2) * pix_dim
        alpha = jnp.arctan(r / det_dist)
        predicted_q = jnp.sin(alpha) * k0
        return jnp.sum((predicted_q - target_q) ** 2)

    return objective


def minimize_with_grad(
    objective: Callable,
    initial_params: NDArray[np.floating],
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    learning_rate: float = 0.01,
) -> tuple[NDArray, dict]:
    """Minimize objective using gradient descent.

    Simple gradient descent optimizer for user-defined objective functions.
    For more sophisticated optimization, use optimistix or scipy.optimize.

    Args:
        objective: Differentiable objective function(params) -> scalar
        initial_params: Initial parameter values
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        learning_rate: Learning rate (step size)

    Returns:
        Tuple of (optimal_params, diagnostics_dict)
    """
    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError("Gradient minimization requires JAX backend.")

    import jax
    import jax.numpy as jnp

    params = jnp.array(initial_params)
    # Single JIT over value_and_grad: one compilation, one kernel dispatch per step.
    val_and_grad_fn = jax.jit(jax.value_and_grad(objective))

    # Prime cache with first forward+backward pass.
    loss_val, grads = val_and_grad_fn(params)
    losses = [float(loss_val)]
    lr = learning_rate

    for i in range(max_iterations):
        params = params - lr * grads
        loss_val, grads = val_and_grad_fn(params)

        # Check convergence every 10 steps to reduce host-device sync overhead
        if i % 10 == 0 or i == max_iterations - 1:
            loss = float(loss_val)
            losses.append(loss)

            if np.isnan(loss):
                logger.warning("NaN loss detected in calibration, stopping early")
                break

            # Check convergence
            if len(losses) >= 2 and abs(losses[-1] - losses[-2]) < tolerance:
                break

        # Adaptive learning rate
        if i > 0 and i % 50 == 0:
            lr *= 0.8

    diagnostics = {
        "iterations": len(losses) - 1,
        "losses": np.array(losses),
        "converged": abs(losses[-1] - losses[-2]) < tolerance
        if len(losses) >= 2
        else False,
        "final_loss": losses[-1],
    }

    return ensure_numpy(params), diagnostics
