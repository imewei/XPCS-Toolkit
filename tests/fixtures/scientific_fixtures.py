"""Scientific data fixtures for XPCS Toolkit tests.

This module provides fixtures for synthetic XPCS datasets, correlation functions,
scattering patterns, detector geometries, and Q-space mappings.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    from tests.utils.h5py_mocks import MockH5py

    h5py = MockH5py()


# ============================================================================
# Random Seed and Reproducibility
# ============================================================================


@pytest.fixture(scope="function")
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


# ============================================================================
# Synthetic Scientific Data Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def synthetic_correlation_data(random_seed) -> dict[str, np.ndarray]:
    """Generate synthetic G2 correlation function data."""
    # Time points (logarithmic spacing)
    tau = np.logspace(-6, 2, 50)  # 1μs to 100s

    # Synthetic G2 with exponential decay + noise
    beta = 0.8  # contrast
    tau_c = 1e-3  # correlation time (1ms)
    g2 = 1 + beta * np.exp(-tau / tau_c)

    # Add realistic noise
    noise_level = 0.02
    g2_noise = g2 + np.random.normal(0, noise_level * g2, size=g2.shape)
    g2_err = noise_level * g2

    return {
        "tau": tau,
        "g2": g2_noise,
        "g2_err": g2_err,
        "g2_theory": g2,
        "beta": beta,
        "tau_c": tau_c,
    }


@pytest.fixture(scope="function")
def synthetic_scattering_data(random_seed) -> dict[str, np.ndarray]:
    """Generate synthetic SAXS scattering data."""
    # Q-space points
    q = np.linspace(0.001, 0.1, 100)  # Å⁻¹

    # Synthetic scattering with power law + noise
    intensity = 1e6 * q ** (-2.5) + 100  # Background
    intensity_noise = intensity + np.random.poisson(np.sqrt(intensity))
    intensity_err = np.sqrt(intensity)

    return {
        "q": q,
        "intensity": intensity_noise,
        "intensity_err": intensity_err,
        "intensity_theory": intensity,
    }


@pytest.fixture(scope="function")
def detector_geometry() -> dict[str, Any]:
    """Standard detector geometry parameters."""
    return {
        "detector_distance": 5.0,  # meters
        "pixel_size": 75e-6,  # meters
        "detector_shape": (2048, 2048),
        "beam_center": (1024.0, 1024.0),  # pixels
        "wavelength": 1.24e-10,  # meters (10 keV)
        "q_range": (1e-3, 1e-1),  # Å⁻¹
    }


@pytest.fixture(scope="function")
def qmap_data(detector_geometry) -> dict[str, np.ndarray]:
    """Generate Q-space mapping data."""
    shape = detector_geometry["detector_shape"]

    # Create coordinate arrays
    y, x = np.ogrid[: shape[0], : shape[1]]

    # Center coordinates
    cx, cy = detector_geometry["beam_center"]

    # Distance from beam center
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Convert to Q (simplified)
    pixel_size = detector_geometry["pixel_size"]
    distance = detector_geometry["detector_distance"]
    wavelength = detector_geometry["wavelength"]

    # Scattering angle
    theta = np.arctan(r * pixel_size / distance) / 2

    # Q magnitude
    q = 4 * np.pi * np.sin(theta) / wavelength

    # Azimuthal angle
    phi = np.arctan2(y - cy, x - cx)

    return {
        "qx": q * np.cos(phi),
        "qy": q * np.sin(phi),
        "q_magnitude": q,
        "phi": phi,
        "pixel_positions": {"x": x, "y": y},
    }


# ============================================================================
# HDF5 Test File Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def minimal_xpcs_hdf5(temp_dir, synthetic_correlation_data) -> str:
    """Create minimal XPCS HDF5 file for testing."""
    filepath = Path(temp_dir) / "minimal_test.h5"

    if not H5PY_AVAILABLE:
        return str(filepath)

    with h5py.File(filepath, "w") as f:
        # Basic structure
        exchange = f.create_group("exchange")

        # Add minimal required datasets
        tau_data = synthetic_correlation_data["tau"]
        g2_data = synthetic_correlation_data["g2"]

        exchange.create_dataset("tau", data=tau_data)
        exchange.create_dataset("g2", data=g2_data)

        # Add basic metadata
        f.attrs["instrument"] = "APS 8-ID-I"
        f.attrs["sample"] = "test_sample"
        f.attrs["temperature"] = 295.0

    return str(filepath)


@pytest.fixture(scope="function")
def comprehensive_xpcs_hdf5(
    temp_dir, synthetic_correlation_data, synthetic_scattering_data, qmap_data
) -> str:
    """Create comprehensive XPCS HDF5 file for testing."""
    filepath = Path(temp_dir) / "comprehensive_test.h5"

    if not H5PY_AVAILABLE:
        return str(filepath)

    with h5py.File(filepath, "w") as f:
        # Exchange group with correlation data
        exchange = f.create_group("exchange")
        exchange.create_dataset("tau", data=synthetic_correlation_data["tau"])
        exchange.create_dataset("g2", data=synthetic_correlation_data["g2"])
        exchange.create_dataset("g2_err", data=synthetic_correlation_data["g2_err"])

        # SAXS data
        saxs_1d = f.create_group("saxs_1d")
        saxs_1d.create_dataset("q", data=synthetic_scattering_data["q"])
        saxs_1d.create_dataset("intensity", data=synthetic_scattering_data["intensity"])
        saxs_1d.create_dataset(
            "intensity_err", data=synthetic_scattering_data["intensity_err"]
        )

        # Q-mapping data
        qmap = f.create_group("qmap")
        qmap.create_dataset("qx", data=qmap_data["qx"])
        qmap.create_dataset("qy", data=qmap_data["qy"])
        qmap.create_dataset("q_magnitude", data=qmap_data["q_magnitude"])

        # Comprehensive metadata
        f.attrs.update(
            {
                "instrument": "APS 8-ID-I",
                "sample": "comprehensive_test_sample",
                "temperature": 295.0,
                "exposure_time": 0.1,
                "num_frames": 10000,
                "detector": "Lambda",
            }
        )

    return str(filepath)


# ============================================================================
# Scientific Validation Utilities
# ============================================================================


@pytest.fixture(scope="function")
def assert_arrays_close():
    """Fixture providing array comparison utility."""

    def _assert_close(
        actual, expected, rtol=1e-7, atol=1e-14, err_msg="", verbose=True
    ):
        """Assert that two arrays are element-wise equal within a tolerance."""
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=err_msg, verbose=verbose
        )

    return _assert_close


@pytest.fixture(scope="function")
def correlation_function_validator():
    """Fixture providing G2 correlation function validation."""

    def _validate_g2(tau, g2, expected_params=None):
        """Validate G2 correlation function properties."""
        # Basic sanity checks
        assert len(tau) == len(g2), "tau and g2 must have same length"
        assert np.all(tau > 0), "tau values must be positive"
        assert np.all(g2 >= 1.0), "G2 values must be >= 1.0 (correlation baseline)"

        # Check for reasonable decay
        if len(g2) > 10:
            initial_g2 = np.mean(g2[:5])  # Average of first few points
            final_g2 = np.mean(g2[-5:])  # Average of last few points
            assert initial_g2 > final_g2, "G2 should decay from initial to final tau"

        # If expected parameters provided, validate fit
        if expected_params is not None:
            if isinstance(expected_params, dict):
                beta_expected = expected_params.get("beta", None)
                if beta_expected:
                    measured_contrast = np.max(g2) - 1.0
                    assert abs(measured_contrast - beta_expected) < 0.1, (
                        f"Measured contrast {measured_contrast:.3f} differs from expected {beta_expected:.3f}"
                    )

        return True

    return _validate_g2


@pytest.fixture(scope="function")
def create_test_dataset():
    """Fixture for creating custom test datasets."""

    def _create_dataset(size=100, data_type="correlation", **kwargs):
        """Create various types of test datasets."""
        if data_type == "correlation":
            tau = np.logspace(-6, 2, size)
            beta = kwargs.get("beta", 0.8)
            tau_c = kwargs.get("tau_c", 1e-3)
            noise = kwargs.get("noise", 0.02)

            g2_theory = 1 + beta * np.exp(-tau / tau_c)
            g2_noise = g2_theory + np.random.normal(0, noise * g2_theory)

            return {"tau": tau, "g2": g2_noise, "g2_theory": g2_theory}

        if data_type == "scattering":
            q = np.linspace(0.001, 0.1, size)
            power = kwargs.get("power", -2.5)
            background = kwargs.get("background", 100)

            intensity = 1e6 * q**power + background
            intensity_noise = intensity + np.random.poisson(np.sqrt(intensity))

            return {"q": q, "intensity": intensity_noise, "intensity_theory": intensity}

        raise ValueError(f"Unknown data_type: {data_type}")

    return _create_dataset
