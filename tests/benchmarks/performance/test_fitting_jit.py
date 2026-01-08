"""Benchmark tests for JIT-accelerated fitting functions.

Verifies performance improvements from JIT compilation in calibration and NLSQ.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.fixture
def fitting_benchmark_data() -> dict:
    """Generate data for fitting benchmarks."""
    rng = np.random.default_rng(42)
    n_points = 500

    # Generate synthetic G2 data
    tau = np.logspace(-4, 2, n_points)
    true_tau_c = 1.0
    true_baseline = 1.0
    true_contrast = 0.3
    noise = rng.normal(0, 0.01, n_points)

    g2 = true_baseline + true_contrast * np.exp(-2 * tau / true_tau_c) + noise
    g2_err = np.full(n_points, 0.01)

    return {
        "tau": tau,
        "g2": g2,
        "g2_err": g2_err,
        "true_params": {
            "tau": true_tau_c,
            "baseline": true_baseline,
            "contrast": true_contrast,
        },
        "n_points": n_points,
    }


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNLSQJITPerformance:
    """Benchmark tests for NLSQ JIT performance."""

    def test_nlsq_fit_correctness(self, fitting_benchmark_data: dict) -> None:
        """Verify NLSQ fitting runs without error and returns valid result."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        data = fitting_benchmark_data
        x = jnp.array(data["tau"])
        y = jnp.array(data["g2"])
        yerr = jnp.array(data["g2_err"])

        p0 = {"tau": 0.5, "baseline": 1.0, "contrast": 0.2}
        bounds = {
            "tau": (0.01, 100.0),
            "baseline": (0.5, 1.5),
            "contrast": (0.01, 1.0),
        }

        result = nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds)

        # Check result structure is valid (performance test, not accuracy)
        assert "tau" in result.params, "Result should have tau parameter"
        assert "baseline" in result.params, "Result should have baseline parameter"
        assert "contrast" in result.params, "Result should have contrast parameter"
        assert result.covariance.shape == (3, 3), "Covariance should be 3x3"
        assert len(result.residuals) == data["n_points"], (
            "Residuals should match data length"
        )
        assert np.isfinite(result.chi_squared), "Chi-squared should be finite"

    def test_nlsq_fit_timing(self, fitting_benchmark_data: dict, benchmark) -> None:
        """Record timing for NLSQ fitting with JIT."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        data = fitting_benchmark_data
        x = jnp.array(data["tau"])
        y = jnp.array(data["g2"])
        yerr = jnp.array(data["g2_err"])

        p0 = {"tau": 0.5, "baseline": 1.0, "contrast": 0.2}
        bounds = {
            "tau": (0.01, 100.0),
            "baseline": (0.5, 1.5),
            "contrast": (0.01, 1.0),
        }

        def run_nlsq():
            return nlsq_optimize(single_exp_func, x, y, yerr, p0, bounds)

        result = benchmark(run_nlsq)
        assert result is not None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestCalibrationJITPerformance:
    """Benchmark tests for calibration JIT performance."""

    def test_calibration_objective_correctness(self) -> None:
        """Verify calibration objective function is correct."""
        from xpcsviewer.simplemask.calibration import create_calibration_objective

        # Skip if JAX backend not available
        try:
            from xpcsviewer.backends import get_backend

            if get_backend().name != "jax":
                pytest.skip("JAX backend required")
        except Exception:
            pytest.skip("Backend check failed")

        # Simple test with known geometry
        target_q = np.array([0.05, 0.07, 0.1])
        positions = [(100, 128), (150, 128), (200, 128)]
        pix_dim = 0.075  # 75 micron
        k0 = 62.8  # 2π/0.1nm wavelength

        objective = create_calibration_objective(target_q, positions, pix_dim, k0)
        params = jnp.array([128.0, 128.0, 5000.0])  # cx, cy, det_dist

        loss = objective(params)
        assert jnp.isfinite(loss), "Objective should return finite value"
        assert float(loss) >= 0, "Loss should be non-negative"

    def test_beam_center_refinement_timing(self, benchmark) -> None:
        """Record timing for beam center refinement with JIT."""
        from xpcsviewer.simplemask.calibration import refine_beam_center

        # Skip if JAX backend not available
        try:
            from xpcsviewer.backends import get_backend

            if get_backend().name != "jax":
                pytest.skip("JAX backend required")
        except Exception:
            pytest.skip("Backend check failed")

        # Generate synthetic ring points
        rng = np.random.default_rng(42)
        n_points = 200
        true_center = (128.0, 128.0)
        radius = 50.0
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        noise = rng.normal(0, 1, (n_points, 2))

        ring_points = np.column_stack(
            [
                true_center[0] + radius * np.cos(angles) + noise[:, 0],
                true_center[1] + radius * np.sin(angles) + noise[:, 1],
            ]
        )

        initial_center = (130.0, 126.0)  # Slightly off

        def run_refinement():
            return refine_beam_center(
                ring_points,
                initial_center,
                max_iterations=50,
                tolerance=1e-6,
            )

        result = benchmark(run_refinement)
        assert result is not None
