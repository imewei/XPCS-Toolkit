"""Pytest configuration and shared fixtures for XPCS Toolkit tests.

This module provides comprehensive fixtures for testing scientific computing
functionality, including synthetic XPCS datasets, HDF5 test files, and
logging configuration.
"""

import logging
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Generator

import h5py
import numpy as np
import pytest

from xpcs_toolkit.utils.logging_config import get_logger, setup_logging

# Suppress common scientific computing warnings during tests
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


# ============================================================================
# Test Configuration and Setup
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "gui: GUI tests (requires display)")
    config.addinivalue_line("markers", "slow: Tests that take more than 1 second")
    config.addinivalue_line(
        "markers", "scientific: Tests that verify scientific accuracy"
    )

    # Set Qt platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Configure logging for tests - suppress verbose output
    os.environ["PYXPCS_LOG_LEVEL"] = "WARNING"
    setup_logging(level=logging.WARNING)


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their characteristics."""
    for item in items:
        # Mark slow tests
        if "slow" in item.keywords or any(
            marker in item.name.lower()
            for marker in ["performance", "benchmark", "stress"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark GUI tests
        if any(
            gui_keyword in item.name.lower()
            for gui_keyword in ["gui", "widget", "qt", "pyside"]
        ):
            item.add_marker(pytest.mark.gui)


# ============================================================================
# Temporary Directory and File Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def temp_dir() -> Generator[str, None, None]:
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="xpcs_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_file(temp_dir) -> str:
    """Create temporary file path."""
    return os.path.join(temp_dir, "test_file.tmp")


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to test fixtures directory."""
    fixtures_path = Path(__file__).parent / "fixtures"
    fixtures_path.mkdir(exist_ok=True)
    return fixtures_path


# ============================================================================
# Scientific Data Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def random_seed() -> int:
    """Fixed random seed for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    return seed


@pytest.fixture(scope="function")
def synthetic_correlation_data(random_seed) -> Dict[str, np.ndarray]:
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
def synthetic_scattering_data(random_seed) -> Dict[str, np.ndarray]:
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
def detector_geometry() -> Dict[str, Any]:
    """Standard detector geometry parameters."""
    return {
        "pixel_size": 75e-6,  # 75 μm pixels
        "det_dist": 5.0,  # 5 m sample-detector distance
        "beam_center_x": 512.0,  # pixels
        "beam_center_y": 512.0,  # pixels
        "X_energy": 8.0,  # keV
        "detector_size": (1024, 1024),  # pixels
        "wavelength": 1.55e-10,  # meters (8 keV)
    }


@pytest.fixture(scope="function")
def qmap_data(detector_geometry) -> Dict[str, np.ndarray]:
    """Generate Q-space mapping data."""
    geom = detector_geometry
    ny, nx = geom["detector_size"]

    # Create coordinate arrays
    x = np.arange(nx) - geom["beam_center_x"]
    y = np.arange(ny) - geom["beam_center_y"]
    X, Y = np.meshgrid(x, y)

    # Calculate Q-space mapping
    pixel_size = geom["pixel_size"]
    det_dist = geom["det_dist"]
    wavelength = geom["wavelength"]

    # Scattering angles
    theta = 0.5 * np.arctan(np.sqrt(X**2 + Y**2) * pixel_size / det_dist)

    # Q magnitude
    q_magnitude = 4 * np.pi * np.sin(theta) / wavelength

    # Azimuthal angle
    phi = np.arctan2(Y, X) * 180 / np.pi

    # Create binned maps (simplified)
    q_bins = np.linspace(0, q_magnitude.max(), 50)
    phi_bins = np.linspace(-180, 180, 36)

    # Dynamic and static Q maps (simplified)
    dqmap = np.digitize(q_magnitude.flatten(), q_bins).reshape(q_magnitude.shape)
    sqmap = np.digitize(q_magnitude.flatten(), q_bins).reshape(q_magnitude.shape)

    return {
        "dqmap": dqmap.astype(np.int32),
        "sqmap": sqmap.astype(np.int32),
        "q_magnitude": q_magnitude,
        "phi": phi,
        "dqlist": q_bins[:-1],
        "sqlist": q_bins[:-1],
        "dplist": phi_bins[:-1],
        "splist": phi_bins[:-1],
        "bcx": geom["beam_center_x"],
        "bcy": geom["beam_center_y"],
        "pixel_size": geom["pixel_size"],
        "det_dist": geom["det_dist"],
        "X_energy": geom["X_energy"],
        "dynamic_num_pts": len(q_bins) - 1,
        "static_num_pts": len(q_bins) - 1,
        "mask": np.ones(geom["detector_size"], dtype=np.int32),
    }


# ============================================================================
# HDF5 Test File Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def minimal_xpcs_hdf5(temp_dir, qmap_data, synthetic_correlation_data) -> str:
    """Create minimal XPCS HDF5 file for testing."""
    hdf_file = os.path.join(temp_dir, "minimal_xpcs.hdf")

    with h5py.File(hdf_file, "w") as f:
        # Basic NeXus structure
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"

        # XPCS-specific structure
        xpcs = f.create_group("xpcs")

        # Q-map data
        qmap_group = xpcs.create_group("qmap")
        for key, value in qmap_data.items():
            qmap_group.create_dataset(key, data=value)

        # Add index mappings
        qmap_group.create_dataset(
            "static_index_mapping", data=np.arange(qmap_data["static_num_pts"])
        )
        qmap_group.create_dataset(
            "dynamic_index_mapping", data=np.arange(qmap_data["dynamic_num_pts"])
        )
        qmap_group.create_dataset("map_names", data=[b"q", b"phi"])
        qmap_group.create_dataset("map_units", data=[b"1/A", b"degree"])

        # Multi-tau correlation data
        multitau = xpcs.create_group("multitau")
        multitau.create_dataset(
            "g2", data=synthetic_correlation_data["g2"].reshape(1, -1)
        )
        multitau.create_dataset(
            "g2_err", data=synthetic_correlation_data["g2_err"].reshape(1, -1)
        )
        multitau.create_dataset("tau", data=synthetic_correlation_data["tau"])
        multitau.create_dataset("stride_frame", data=1)
        multitau.create_dataset("avg_frame", data=1)
        multitau.create_dataset("t0", data=0.001)
        multitau.create_dataset("t1", data=0.001)
        multitau.create_dataset("start_time", data=1000000000)

        # Add minimal SAXS data
        multitau.create_dataset("saxs_1d", data=np.random.rand(1, 50))
        multitau.create_dataset(
            "Iqp", data=np.random.rand(qmap_data["static_num_pts"], 50)
        )
        multitau.create_dataset("Int_t", data=np.random.rand(2, 100))

    return hdf_file


@pytest.fixture(scope="function")
def comprehensive_xpcs_hdf5(
    temp_dir, qmap_data, synthetic_correlation_data, synthetic_scattering_data
) -> str:
    """Create comprehensive XPCS HDF5 file with all analysis data."""
    hdf_file = os.path.join(temp_dir, "comprehensive_xpcs.hdf")

    with h5py.File(hdf_file, "w") as f:
        # Start with minimal structure
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"

        # Add measurement metadata
        entry.create_dataset("start_time", data="2024-01-01T00:00:00")
        entry.create_dataset("end_time", data="2024-01-01T01:00:00")

        # Instrument details
        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"

        source = instrument.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source.create_dataset("name", data="Advanced Photon Source")
        source.create_dataset("probe", data="x-ray")

        detector = instrument.create_group("detector")
        detector.attrs["NX_class"] = "NXdetector"
        detector.create_dataset("detector_number", data=1)

        # XPCS analysis groups
        xpcs = f.create_group("xpcs")

        # Enhanced Q-map with all metadata
        qmap_group = xpcs.create_group("qmap")
        for key, value in qmap_data.items():
            qmap_group.create_dataset(key, data=value)

        # Multi-tau with fitting results
        multitau = xpcs.create_group("multitau")
        corr_data = synthetic_correlation_data

        # Multiple Q-points
        n_q = 5
        g2_matrix = np.tile(corr_data["g2"], (n_q, 1))
        g2_err_matrix = np.tile(corr_data["g2_err"], (n_q, 1))

        multitau.create_dataset("g2", data=g2_matrix)
        multitau.create_dataset("g2_err", data=g2_err_matrix)
        multitau.create_dataset("tau", data=corr_data["tau"])

        # Fitting results
        fit_group = multitau.create_group("fit_results")
        fit_group.create_dataset("beta", data=np.full(n_q, corr_data["beta"]))
        fit_group.create_dataset("tau_c", data=np.full(n_q, corr_data["tau_c"]))
        fit_group.create_dataset("baseline", data=np.ones(n_q))
        fit_group.create_dataset("chi_squared", data=np.random.rand(n_q))

        # SAXS data
        saxs_data = synthetic_scattering_data
        multitau.create_dataset("saxs_1d", data=saxs_data["intensity"].reshape(1, -1))
        multitau.create_dataset("q_saxs", data=saxs_data["q"])

        # Two-time correlation (small for testing)
        twotime = xpcs.create_group("twotime")
        twotime_data = np.random.rand(50, 50) + 1.0  # 50x50 for speed
        twotime.create_dataset("g2", data=twotime_data)
        twotime.create_dataset("elapsed_time", data=np.linspace(0, 100, 50))

        # Stability analysis
        stability = xpcs.create_group("stability")
        stability.create_dataset("intensity_time", data=np.random.rand(1000))
        stability.create_dataset("time_points", data=np.linspace(0, 1000, 1000))
        stability.create_dataset("mean_intensity", data=1000.0)
        stability.create_dataset("std_intensity", data=50.0)

    return hdf_file


# ============================================================================
# Logging and Testing Utilities
# ============================================================================


@pytest.fixture(scope="function")
def test_logger():
    """Create logger for individual tests."""
    return get_logger("test")


@pytest.fixture(scope="function")
def capture_logs(caplog):
    """Capture logs with appropriate level."""
    with caplog.at_level(logging.DEBUG):
        yield caplog


@pytest.fixture(scope="function")
def performance_timer():
    """Timer fixture for performance tests."""
    import time

    class Timer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed

        @property
        def elapsed(self):
            if self.start_time is None:
                return 0
            end = self.end_time or time.perf_counter()
            return end - self.start_time

    return Timer()


# ============================================================================
# Scientific Computing Utilities
# ============================================================================


@pytest.fixture(scope="function")
def assert_arrays_close():
    """Custom assertion for array comparisons with scientific tolerance."""

    def _assert_close(actual, expected, rtol=1e-7, atol=1e-14, msg=""):
        """Assert arrays are close within scientific tolerance."""
        try:
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as e:
            if msg:
                raise AssertionError(f"{msg}: {e}")
            raise

    return _assert_close


@pytest.fixture(scope="function")
def correlation_function_validator():
    """Validator for correlation function properties."""

    def _validate(tau, g2, g2_err=None):
        """Validate correlation function properties."""
        # Basic shape checks
        assert len(tau) == len(g2), "tau and g2 must have same length"
        if g2_err is not None:
            assert len(g2_err) == len(g2), "g2_err must have same length as g2"

        # Physical constraints
        assert np.all(tau > 0), "All tau values must be positive"
        assert np.all(g2 >= 1.0), "G2 values must be >= 1 (correlation inequality)"

        # Monotonicity (generally expected for simple systems)
        if len(tau) > 1:
            assert np.all(np.diff(tau) > 0), (
                "tau values must be monotonically increasing"
            )

        # Error bars should be positive
        if g2_err is not None:
            assert np.all(g2_err >= 0), "Error bars must be non-negative"

    return _validate


# ============================================================================
# Parametrized Test Data
# ============================================================================


@pytest.fixture(
    params=[
        {"q_min": 0.001, "q_max": 0.1, "n_points": 50},
        {"q_min": 0.005, "q_max": 0.05, "n_points": 25},
        {"q_min": 0.01, "q_max": 0.2, "n_points": 100},
    ]
)
def q_range_params(request):
    """Parametrized Q-range configurations for testing."""
    return request.param


@pytest.fixture(
    params=[
        {"tau_min": 1e-6, "tau_max": 1e2, "n_tau": 50},
        {"tau_min": 1e-5, "tau_max": 1e1, "n_tau": 30},
        {"tau_min": 1e-4, "tau_max": 1e3, "n_tau": 100},
    ]
)
def tau_range_params(request):
    """Parametrized time range configurations for testing."""
    return request.param


# ============================================================================
# Test Data Factories
# ============================================================================


@pytest.fixture(scope="session")
def create_test_dataset():
    """Factory for creating test datasets with specific parameters."""

    def _create(dataset_type="minimal", **kwargs):
        """Create test dataset of specified type."""
        if dataset_type == "minimal":
            return _create_minimal_dataset(**kwargs)
        elif dataset_type == "realistic":
            return _create_realistic_dataset(**kwargs)
        elif dataset_type == "edge_case":
            return _create_edge_case_dataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    return _create


def _create_minimal_dataset(**kwargs):
    """Create minimal test dataset."""
    n_points = kwargs.get("n_points", 10)
    return {
        "tau": np.logspace(-3, 0, n_points),
        "g2": 1 + 0.5 * np.exp(-np.logspace(-3, 0, n_points) / 0.1),
        "metadata": {"type": "minimal", "n_points": n_points},
    }


def _create_realistic_dataset(**kwargs):
    """Create realistic test dataset with noise."""
    n_points = kwargs.get("n_points", 50)
    noise_level = kwargs.get("noise_level", 0.02)

    tau = np.logspace(-6, 2, n_points)
    g2_clean = 1 + 0.8 * np.exp(-tau / 1e-3)
    g2_noise = g2_clean + np.random.normal(0, noise_level * g2_clean)

    return {
        "tau": tau,
        "g2": g2_noise,
        "g2_clean": g2_clean,
        "g2_err": noise_level * g2_clean,
        "metadata": {"type": "realistic", "noise_level": noise_level},
    }


def _create_edge_case_dataset(**kwargs):
    """Create edge case dataset for robustness testing."""
    return {
        "tau": np.array([1e-10, 1e10]),  # Extreme time scales
        "g2": np.array([1.0, 1.0]),  # Flat correlation
        "metadata": {"type": "edge_case"},
    }
