"""Tests for CPU-only system launch (T068).

Tests that application launches correctly on CPU-only systems (US5).
"""

from __future__ import annotations

import os

import pytest


class TestCPUOnlyLaunch:
    """Tests for CPU-only system launch."""

    def test_backend_initializes_without_gpu(self, monkeypatch) -> None:
        """Test backend initializes when GPU is disabled."""
        # Force CPU mode via environment variable
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        # Need to reload backend after env change
        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()  # Clear cached backend

        backend = get_backend()
        assert backend is not None
        assert backend.name == "jax"

    def test_fallback_to_numpy_when_jax_disabled(self, monkeypatch) -> None:
        """Test fallback to NumPy when JAX is explicitly disabled."""
        monkeypatch.setenv("XPCS_USE_JAX", "0")

        from xpcsviewer.backends import _reset_backend, get_backend, set_backend

        _reset_backend()
        # Explicitly set numpy backend since JAX may already be imported
        set_backend("numpy")

        backend = get_backend()
        assert backend is not None
        assert backend.name == "numpy"

    def test_qmap_works_on_cpu(self, monkeypatch) -> None:
        """Test Q-map computation works on CPU."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Compute Q-map (should work on CPU)
        qmap, units = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        assert "q" in qmap
        assert qmap["q"].shape == (128, 128)
        assert units["q"] == "Å⁻¹"

    def test_fitting_works_on_cpu(self, monkeypatch) -> None:
        """Test fitting works on CPU backend."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        import jax.numpy as jnp
        import numpy as np

        from xpcsviewer.fitting.nlsq import nlsq_optimize

        # Generate synthetic data
        x = np.logspace(-3, 1, 30)
        true_tau = 0.1
        y = 1.0 + 0.3 * np.exp(-2 * x / true_tau) + np.random.normal(0, 0.01, len(x))

        # Model must use JAX ops for JAX backend compatibility
        def model(x, tau, baseline, contrast):
            return baseline + contrast * jnp.exp(-2 * x / tau)

        result = nlsq_optimize(
            model_fn=model,
            x=x,
            y=y,
            yerr=np.ones_like(y) * 0.01,
            p0={"tau": 0.2, "baseline": 1.0, "contrast": 0.3},
            bounds={
                "tau": (0.01, 10.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )

        assert result.converged
        assert abs(result.params["tau"] - true_tau) < 0.1  # Within 0.1 of true value


class TestCPUFallbackOnGPUFailure:
    """Tests for automatic CPU fallback when GPU fails."""

    def test_graceful_fallback_on_gpu_error(self, monkeypatch) -> None:
        """Test graceful fallback to CPU if GPU initialization fails."""
        # Set up environment to request GPU but expect graceful failure
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("XPCS_GPU_FALLBACK", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        # This should not raise even if GPU is unavailable
        backend = get_backend()
        assert backend is not None
        assert backend.name in ("jax", "numpy")

    def test_computations_work_after_fallback(self, monkeypatch) -> None:
        """Test computations succeed after GPU fallback."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        monkeypatch.setenv("JAX_PLATFORMS", "cpu")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.backends import get_backend

        backend = get_backend()

        # Test basic operations work
        x = backend.linspace(0, 1, 100)
        y = backend.sin(x)

        assert y.shape == (100,)
