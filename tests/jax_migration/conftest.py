"""Pytest fixtures for JAX migration tests.

Provides:
    - Backend fixtures (jax_backend, numpy_backend, backend)
    - Device fixtures (cpu_device, gpu_device)
    - Tolerance fixtures for numerical comparisons
    - Sample data generators
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from xpcsviewer.backends import BackendProtocol


# Numerical tolerances for equivalence tests
FLOAT64_RTOL = 1e-10
FLOAT64_ATOL = 1e-12
FLOAT32_RTOL = 1e-5
FLOAT32_ATOL = 1e-6

# Q-map specific tolerances (per SC-004)
QMAP_RTOL = 1e-6
QMAP_ATOL = 1e-8


@pytest.fixture
def numpy_backend() -> BackendProtocol:
    """Provide NumPy backend for testing."""
    from xpcsviewer.backends import set_backend
    from xpcsviewer.backends._numpy_backend import NumPyBackend

    set_backend("numpy")
    return NumPyBackend()


@pytest.fixture
def jax_backend() -> BackendProtocol:
    """Provide JAX backend for testing.

    Skips test if JAX is not available.
    """
    pytest.importorskip("jax")

    from xpcsviewer.backends import set_backend
    from xpcsviewer.backends._jax_backend import JAXBackend

    set_backend("jax")
    return JAXBackend()


@pytest.fixture(params=["numpy", "jax"])
def backend(request) -> BackendProtocol:
    """Parametrized fixture providing both backends.

    Use this to run the same test against both backends.
    """
    if request.param == "jax":
        pytest.importorskip("jax")

    from xpcsviewer.backends import get_backend, set_backend

    set_backend(request.param)
    return get_backend()


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import jax

        devices = jax.devices("gpu")
        return len(devices) > 0
    except (ImportError, RuntimeError):
        return False


@pytest.fixture
def require_gpu(gpu_available: bool):
    """Skip test if GPU is not available."""
    if not gpu_available:
        pytest.skip("GPU not available")


@pytest.fixture
def require_jax():
    """Skip test if JAX is not available."""
    pytest.importorskip("jax")


@pytest.fixture
def sample_qmap_data() -> dict:
    """Generate sample Q-map computation input data."""
    # Typical detector geometry parameters
    return {
        "detector_distance": 5.0,  # meters
        "pixel_size": 75e-6,  # meters (75 microns)
        "wavelength": 1.0e-10,  # meters (1 Angstrom)
        "beam_center_x": 512,
        "beam_center_y": 512,
        "detector_shape": (1024, 1024),
    }


@pytest.fixture
def sample_g2_data() -> dict:
    """Generate sample G2 correlation data for fitting tests."""
    rng = np.random.default_rng(42)

    # Generate synthetic G2 data with known parameters
    tau_true = 1.0  # seconds
    baseline_true = 1.0
    contrast_true = 0.3

    # Logarithmically spaced delay times (typical for XPCS)
    x = np.logspace(-3, 2, 50)  # 1ms to 100s

    # Single exponential model
    y_true = baseline_true + contrast_true * np.exp(-2 * x / tau_true)

    # Add realistic noise
    noise_level = 0.01
    y = y_true + rng.normal(0, noise_level, size=y_true.shape)
    yerr = np.full_like(y, noise_level)

    return {
        "x": x,
        "y": y,
        "yerr": yerr,
        "true_params": {
            "tau": tau_true,
            "baseline": baseline_true,
            "contrast": contrast_true,
        },
    }


@pytest.fixture
def sample_stretched_exp_data() -> dict:
    """Generate sample stretched exponential data."""
    rng = np.random.default_rng(43)

    tau_true = 1.0
    baseline_true = 1.0
    contrast_true = 0.3
    beta_true = 0.7  # stretching exponent

    x = np.logspace(-3, 2, 50)
    y_true = baseline_true + contrast_true * np.exp(
        -np.power(2 * x / tau_true, beta_true)
    )

    noise_level = 0.01
    y = y_true + rng.normal(0, noise_level, size=y_true.shape)
    yerr = np.full_like(y, noise_level)

    return {
        "x": x,
        "y": y,
        "yerr": yerr,
        "true_params": {
            "tau": tau_true,
            "baseline": baseline_true,
            "contrast": contrast_true,
            "beta": beta_true,
        },
    }


@pytest.fixture
def tolerance_float64() -> dict:
    """Tolerance settings for float64 comparisons."""
    return {"rtol": FLOAT64_RTOL, "atol": FLOAT64_ATOL}


@pytest.fixture
def tolerance_float32() -> dict:
    """Tolerance settings for float32 comparisons."""
    return {"rtol": FLOAT32_RTOL, "atol": FLOAT32_ATOL}


@pytest.fixture
def tolerance_qmap() -> dict:
    """Tolerance settings for Q-map comparisons (SC-004)."""
    return {"rtol": QMAP_RTOL, "atol": QMAP_ATOL}


# Mark all tests in this directory with jax marker
def pytest_collection_modifyitems(items):
    """Add jax marker to all tests in jax_migration directory."""
    for item in items:
        if "jax_migration" in str(item.fspath):
            item.add_marker(pytest.mark.jax)
