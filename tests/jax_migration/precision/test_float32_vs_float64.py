"""Tests for float32 vs float64 precision (T082).

Tests that float64 precision is maintained in scientific computations.
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
class TestFloat64Default:
    """Tests for float64 as default precision."""

    def test_jax_x64_enabled(self, monkeypatch) -> None:
        """Test JAX x64 mode is enabled by default."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        assert jax.config.jax_enable_x64, "JAX x64 mode should be enabled"

    def test_linspace_produces_float64(self, monkeypatch) -> None:
        """Test backend.linspace produces float64 arrays."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()
        x = backend.linspace(0, 1, 100)

        # JAX arrays should be float64
        assert x.dtype == jnp.float64

    def test_array_creation_float64(self, monkeypatch) -> None:
        """Test array creation preserves float64."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Create from list
        arr = backend.array([1.0, 2.0, 3.0])
        assert arr.dtype == jnp.float64

        # Create from NumPy float64
        np_arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        jax_arr = backend.from_numpy(np_arr)
        assert jax_arr.dtype == jnp.float64


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestFloat32VsFloat64Precision:
    """Tests comparing float32 vs float64 precision."""

    def test_trigonometric_precision(self, monkeypatch) -> None:
        """Test trigonometric functions maintain precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Small angle where precision matters
        small_angle = backend.array(1e-8)

        # sin(x) ≈ x for small x
        sin_val = backend.sin(small_angle)
        expected = 1e-8  # sin(x) ≈ x for small x

        # Float64 should maintain precision
        assert abs(float(sin_val) - expected) < 1e-14

    def test_cumulative_sum_precision(self, monkeypatch) -> None:
        """Test cumulative sums don't lose precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Sum many small values
        n = 10000
        small_val = 1e-10
        arr = backend.full((n,), small_val)

        total = backend.sum(arr)
        expected = n * small_val

        # Float64 should maintain precision
        relative_error = abs(float(total) - expected) / expected
        assert relative_error < 1e-10

    def test_exponential_precision(self, monkeypatch) -> None:
        """Test exponential functions maintain precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Large negative exponent
        x = backend.array(-50.0)
        result = backend.exp(x)

        # exp(-50) ≈ 1.93e-22
        expected = np.exp(-50.0)

        relative_error = abs(float(result) - expected) / expected
        assert relative_error < 1e-10


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQMapPrecision:
    """Tests for Q-map computation precision."""

    def test_qmap_precision_float64(self, monkeypatch) -> None:
        """Test Q-map maintains float64 precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Q-map should be float64
        assert qmap["q"].dtype == np.float64

    def test_qmap_small_angle_precision(self, monkeypatch) -> None:
        """Test Q-map precision for small angles near beam center."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Large detector to test small angles near center
        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(512.0, 512.0),
            shape=(1024, 1024),
            pix_dim=0.075,
            det_dist=10000.0,  # Far detector for small angles
        )

        # Check precision near beam center
        # At center, Q should be exactly 0
        center_q = qmap["q"][512, 512]
        assert abs(center_q) < 1e-10

        # Check adjacent pixel has small but non-zero Q
        adjacent_q = qmap["q"][513, 512]
        assert adjacent_q > 0
        # Q at 1 pixel from center should be small (< 1e-4 for this geometry)
        assert adjacent_q < 1e-4


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestFittingPrecision:
    """Tests for fitting precision."""

    def test_nlsq_residual_precision(self, monkeypatch) -> None:
        """Test NLSQ residual computation precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp

        from xpcsviewer.fitting.nlsq import nlsq_optimize

        # Synthetic data with known parameters
        x = np.logspace(-3, 1, 100)
        tau_true = 0.1
        y = 1.0 + 0.3 * np.exp(-2 * x / tau_true)

        def model(x, tau, baseline, contrast):
            return baseline + contrast * jnp.exp(-2 * x / tau)

        result = nlsq_optimize(
            model_fn=model,
            x=x,
            y=y,
            yerr=np.ones_like(y) * 0.001,
            p0={"tau": 0.2, "baseline": 1.0, "contrast": 0.3},
            bounds={
                "tau": (0.01, 10.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )

        # Check parameters recovered with high precision
        assert abs(result.params["tau"] - tau_true) < 1e-6
        assert abs(result.params["baseline"] - 1.0) < 1e-6
        assert abs(result.params["contrast"] - 0.3) < 1e-6
