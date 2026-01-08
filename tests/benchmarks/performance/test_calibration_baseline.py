"""Baseline benchmarks for calibration optimization.

Establishes performance baselines before JIT optimization.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture
def calibration_data() -> dict:
    """Generate data for calibration benchmarks."""
    rng = np.random.default_rng(42)
    n_points = 1000

    # Simulate detector geometry
    pix_dim = 75e-6  # 75 micron pixels
    det_dist = 5.0  # 5 meter sample-detector distance
    wavelength = 1.0e-10  # 1 Angstrom
    k0 = 2 * np.pi / wavelength

    # Pixel positions
    positions = rng.uniform(0, 512, size=(n_points, 2))

    # Target Q values (simulated calibration targets)
    target_q = rng.uniform(0.01, 0.1, size=n_points)

    return {
        "positions": positions,
        "target_q": target_q,
        "pix_dim": pix_dim,
        "det_dist": det_dist,
        "k0": k0,
        "center": np.array([256.0, 256.0, det_dist]),  # cx, cy, det_dist
        "n_points": n_points,
    }


def compute_q_numpy(
    params: np.ndarray,
    positions: np.ndarray,
    pix_dim: float,
    k0: float,
) -> np.ndarray:
    """Compute Q values using NumPy (baseline)."""
    cx, cy, det_dist = params[0], params[1], params[2]
    dx = positions[:, 0] - cx
    dy = positions[:, 1] - cy
    r = np.sqrt(dx**2 + dy**2) * pix_dim
    alpha = np.arctan(r / det_dist)
    return np.sin(alpha) * k0


def objective_numpy(
    params: np.ndarray,
    positions: np.ndarray,
    target_q: np.ndarray,
    pix_dim: float,
    k0: float,
) -> float:
    """Calibration objective function using NumPy (baseline)."""
    predicted_q = compute_q_numpy(params, positions, pix_dim, k0)
    return float(np.sum((predicted_q - target_q) ** 2))


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def compute_q_jax(
    params: jnp.ndarray,
    positions: jnp.ndarray,
    pix_dim: float,
    k0: float,
) -> jnp.ndarray:
    """Compute Q values using JAX (for JIT comparison)."""
    cx, cy, det_dist = params[0], params[1], params[2]
    dx = positions[:, 0] - cx
    dy = positions[:, 1] - cy
    r = jnp.sqrt(dx**2 + dy**2) * pix_dim
    alpha = jnp.arctan(r / det_dist)
    return jnp.sin(alpha) * k0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
def objective_jax(
    params: jnp.ndarray,
    positions: jnp.ndarray,
    target_q: jnp.ndarray,
    pix_dim: float,
    k0: float,
) -> jnp.ndarray:
    """Calibration objective function using JAX (for JIT comparison)."""
    predicted_q = compute_q_jax(params, positions, pix_dim, k0)
    return jnp.sum((predicted_q - target_q) ** 2)


class TestCalibrationBaseline:
    """Baseline benchmarks for calibration optimization."""

    def test_numpy_objective_correctness(self, calibration_data: dict) -> None:
        """Verify NumPy objective produces valid results."""
        data = calibration_data
        result = objective_numpy(
            data["center"],
            data["positions"],
            data["target_q"],
            data["pix_dim"],
            data["k0"],
        )

        assert np.isfinite(result)
        assert result >= 0  # Sum of squares is non-negative

    def test_numpy_objective_baseline_timing(
        self, calibration_data: dict, benchmark
    ) -> None:
        """Record baseline timing for NumPy objective."""
        data = calibration_data

        def run_objective():
            return objective_numpy(
                data["center"],
                data["positions"],
                data["target_q"],
                data["pix_dim"],
                data["k0"],
            )

        result = benchmark(run_objective)
        assert result is not None

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_objective_correctness(self, calibration_data: dict) -> None:
        """Verify JAX objective produces valid results."""
        data = calibration_data
        params = jnp.array(data["center"])
        positions = jnp.array(data["positions"])
        target_q = jnp.array(data["target_q"])

        result = objective_jax(
            params,
            positions,
            target_q,
            data["pix_dim"],
            data["k0"],
        )

        assert jnp.isfinite(result)
        assert float(result) >= 0

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jax_objective_baseline_timing(
        self, calibration_data: dict, benchmark
    ) -> None:
        """Record baseline timing for JAX objective (non-JIT)."""
        data = calibration_data
        params = jnp.array(data["center"])
        positions = jnp.array(data["positions"])
        target_q = jnp.array(data["target_q"])

        def run_jax_objective():
            return objective_jax(
                params,
                positions,
                target_q,
                data["pix_dim"],
                data["k0"],
            )

        result = benchmark(run_jax_objective)
        assert result is not None
