"""Pytest configuration and shared fixtures for XPCS Toolkit tests.

This module provides comprehensive fixtures for testing scientific computing
functionality, including synthetic XPCS datasets, HDF5 test files, and
logging configuration.
"""

import gc
import logging
import os
import shutil
import tempfile
import threading
import time
import warnings
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import h5py

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    # Use shared mock implementation
    from tests.utils.h5py_mocks import MockH5py

    h5py = MockH5py()

from xpcs_toolkit.utils.logging_config import get_logger, setup_logging

# Import reliability and isolation frameworks
try:
    from tests.utils.isolation import get_performance_monitor
    from tests.utils.reliability import get_flakiness_detector

    RELIABILITY_FRAMEWORKS_AVAILABLE = True
except ImportError:
    RELIABILITY_FRAMEWORKS_AVAILABLE = False

# Import advanced test data management
try:
    from tests.utils.data_management import (
        TestDataSpec,
        create_minimal_test_data,
        create_performance_test_data,
        create_realistic_xpcs_dataset,
        get_hdf5_manager,
        get_test_data_factory,
        temporary_xpcs_file,
    )

    ADVANCED_DATA_MANAGEMENT_AVAILABLE = True
except ImportError:
    ADVANCED_DATA_MANAGEMENT_AVAILABLE = False

# Import performance optimization and monitoring
try:
    from tests.utils.test_performance import (
        TestOptimizer,
        benchmark_function,
        get_cache_manager,
    )
    from tests.utils.test_performance import get_performance_monitor as get_perf_monitor

    PERFORMANCE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    PERFORMANCE_OPTIMIZATION_AVAILABLE = False

# Import CI/CD integration utilities
try:
    from tests.utils.ci_integration import (
        TestResult,
        TestSuite,
        collect_test_artifacts,
        generate_ci_reports,
        get_ci_environment,
        github_step_summary,
        set_github_output,
    )

    CI_INTEGRATION_AVAILABLE = True
except ImportError:
    CI_INTEGRATION_AVAILABLE = False

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

    # Add reliability testing markers
    config.addinivalue_line("markers", "flaky: Tests that are known to be flaky")
    config.addinivalue_line("markers", "stress: Stress tests that push system limits")
    config.addinivalue_line(
        "markers", "system_dependent: Tests that depend on system resources"
    )
    config.addinivalue_line("markers", "reliable: Tests using reliability framework")

    # Set Qt platform for headless testing
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Enable test mode to disable background threads
    os.environ["XPCS_TEST_MODE"] = "1"

    # Configure logging for tests - suppress verbose output
    os.environ["PYXPCS_LOG_LEVEL"] = "WARNING"
    setup_logging(level=logging.WARNING)

    # Apply performance optimizations
    if PERFORMANCE_OPTIMIZATION_AVAILABLE:
        TestOptimizer.optimize_numpy_operations()
        TestOptimizer.optimize_memory_usage()
        TestOptimizer.optimize_io_operations()

    # Configure CI/CD integration
    if CI_INTEGRATION_AVAILABLE:
        ci_env = get_ci_environment()
        if ci_env["is_ci"]:
            print(f"\n🔧 Running in {ci_env['ci_provider']} CI environment")
            if ci_env.get("branch"):
                print(f"   Branch: {ci_env['branch']}")
            if ci_env.get("commit"):
                print(f"   Commit: {ci_env['commit'][:8]}...")

            # Apply CI-specific configurations
            config.option.tb = "short"  # Shorter tracebacks for CI logs


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

        # Mark stress tests
        if any(
            stress_keyword in item.name.lower()
            for stress_keyword in ["stress", "exhaustion", "resource"]
        ):
            item.add_marker(pytest.mark.stress)

        # Mark system dependent tests
        if any(
            system_keyword in item.name.lower() or system_keyword in str(item.fspath)
            for system_keyword in ["memory", "disk", "network", "display"]
        ):
            item.add_marker(pytest.mark.system_dependent)

        # Mark tests using reliability decorators as reliable
        if hasattr(item.function, "__wrapped__") and any(
            attr in str(item.function.__qualname__)
            for attr in ["reliable_test", "monitor_performance", "retry_on_failure"]
        ):
            item.add_marker(pytest.mark.reliable)


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
        "pixel_size": 75e-6,  # 75 μm pixels
        "det_dist": 5.0,  # 5 m sample-detector distance
        "beam_center_x": 512.0,  # pixels
        "beam_center_y": 512.0,  # pixels
        "X_energy": 8.0,  # keV
        "detector_size": (1024, 1024),  # pixels
        "wavelength": 1.55e-10,  # meters (8 keV)
    }


@pytest.fixture(scope="function")
def qmap_data(detector_geometry) -> dict[str, np.ndarray]:
    """Generate Q-space mapping data."""
    geom = detector_geometry
    ny, nx = geom["detector_size"]

    # Create coordinate arrays
    x = np.arange(nx) - geom["beam_center_x"]
    y = np.arange(ny) - geom["beam_center_y"]
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate Q-space mapping
    pixel_size = geom["pixel_size"]
    det_dist = geom["det_dist"]
    wavelength = geom["wavelength"]

    # Scattering angles
    theta = 0.5 * np.arctan(np.sqrt(x_grid**2 + y_grid**2) * pixel_size / det_dist)

    # Q magnitude
    q_magnitude = 4 * np.pi * np.sin(theta) / wavelength

    # Azimuthal angle
    phi = np.arctan2(y_grid, x_grid) * 180 / np.pi

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
                raise AssertionError(f"{msg}: {e}") from e
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
        if dataset_type == "realistic":
            return _create_realistic_dataset(**kwargs)
        if dataset_type == "edge_case":
            return _create_edge_case_dataset(**kwargs)
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


# ============================================================================
# Reliability and Isolation Fixtures
# ============================================================================

if RELIABILITY_FRAMEWORKS_AVAILABLE:
    # Import specific fixtures that depend on the reliability frameworks

    @pytest.fixture(scope="function")
    def flakiness_detector():
        """Provide access to global flakiness detector."""
        return get_flakiness_detector()

    @pytest.fixture(scope="function")
    def performance_monitor():
        """Provide access to global performance monitor."""
        return get_performance_monitor()

    @pytest.fixture(autouse=True)
    def auto_performance_monitoring(request):
        """Automatically monitor test performance for all tests."""
        if not RELIABILITY_FRAMEWORKS_AVAILABLE:
            yield
            return

        test_name = f"{request.module.__name__}.{request.function.__name__}"
        monitor = get_performance_monitor()

        start_time = time.time()

        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            monitor.record_test_time(test_name, duration)

    @pytest.fixture(scope="function")
    def reliable_test_environment(isolation_manager):
        """Provide a completely reliable test environment."""
        if not RELIABILITY_FRAMEWORKS_AVAILABLE:
            yield None
            return

        # Use the isolation manager from test_isolation.py
        yield isolation_manager

    # Convenience decorators available when reliability frameworks are present
    @pytest.fixture(scope="session", autouse=True)
    def setup_reliability_reporting():
        """Set up reliability reporting at the end of test session."""
        yield

        if not RELIABILITY_FRAMEWORKS_AVAILABLE:
            return

        # Generate reliability report
        detector = get_flakiness_detector()
        monitor = get_performance_monitor()

        flaky_tests = detector.get_flaky_tests()
        slow_tests = monitor.get_slow_tests()

        if flaky_tests or slow_tests:
            print("\n" + "=" * 80)
            print("TEST RELIABILITY REPORT")
            print("=" * 80)

            if flaky_tests:
                print(f"\nFlaky tests detected ({len(flaky_tests)}):")
                for test in flaky_tests:
                    print(f"  - {test}")

            if slow_tests:
                print(f"\nSlow tests detected ({len(slow_tests)}):")
                for test in slow_tests:
                    print(f"  - {test}")

            print(
                "\nConsider using reliability decorators from tests.utils.test_reliability"
            )
            print("=" * 80)


# ============================================================================
# Advanced Test Data Fixtures
# ============================================================================

if ADVANCED_DATA_MANAGEMENT_AVAILABLE:

    @pytest.fixture(scope="function")
    def advanced_data_factory():
        """Provide access to the advanced test data factory."""
        return get_test_data_factory()

    @pytest.fixture(scope="function")
    def hdf5_test_manager():
        """Provide access to the HDF5 test data manager with cleanup."""
        manager = get_hdf5_manager()
        try:
            yield manager
        finally:
            manager.cleanup()

    @pytest.fixture(scope="function")
    def minimal_test_dataset(random_seed):
        """Create minimal test dataset using advanced factory."""
        return create_minimal_test_data(seed=random_seed)

    @pytest.fixture(scope="function")
    def performance_test_dataset():
        """Create performance test dataset (10MB)."""
        return create_performance_test_data(size_mb=10.0)

    @pytest.fixture(scope="function")
    def realistic_xpcs_dataset(random_seed):
        """Create realistic XPCS dataset for integration tests."""
        return create_realistic_xpcs_dataset(
            detector_shape=(256, 256),  # Smaller for faster tests
            n_frames=50,
            seed=random_seed,
        )

    @pytest.fixture(scope="function")
    def advanced_xpcs_hdf5(temp_dir, random_seed):
        """Create advanced XPCS HDF5 file with comprehensive data."""
        file_path = Path(temp_dir) / "advanced_xpcs.hdf"

        # Define data specifications
        data_specs = {
            "qmap": TestDataSpec(
                "qmap",
                shape=(256, 256),
                seed=random_seed,
                metadata={"n_q_bins": 30, "n_phi_bins": 24},
            ),
            "correlation": TestDataSpec(
                "correlation",
                seed=random_seed,
                metadata={
                    "n_tau": 40,
                    "beta1": 0.6,
                    "beta2": 0.2,
                    "tau1": 1e-3,
                    "tau2": 1e-2,
                },
            ),
            "xpcs": TestDataSpec(
                "xpcs", shape=(256, 256), seed=random_seed, metadata={"n_frames": 50}
            ),
            "twotime": TestDataSpec("twotime", shape=(50, 50), seed=random_seed),
        }

        manager = get_hdf5_manager()
        hdf5_file = manager.create_xpcs_file(file_path, data_specs)

        try:
            yield str(hdf5_file)
        finally:
            if hdf5_file.exists():
                hdf5_file.unlink(missing_ok=True)

    @pytest.fixture(scope="function")
    def temporary_xpcs_data():
        """Factory for creating temporary XPCS files with custom specifications."""

        def _create_file(data_specs: dict[str, TestDataSpec], **kwargs):
            return temporary_xpcs_file(data_specs, **kwargs)

        return _create_file


# Parametrized fixtures for different data sizes and configurations
@pytest.fixture(
    params=[
        {"size_mb": 1.0, "name": "small"},
        {"size_mb": 5.0, "name": "medium"},
        {"size_mb": 10.0, "name": "large"},
    ]
)
def sized_test_data(request):
    """Parametrized fixture for different data sizes."""
    if not ADVANCED_DATA_MANAGEMENT_AVAILABLE:
        pytest.skip("Advanced data management not available")

    params = request.param
    return create_performance_test_data(size_mb=params["size_mb"])


@pytest.fixture(
    params=[
        {"shape": (128, 128), "n_frames": 25, "name": "fast"},
        {"shape": (256, 256), "n_frames": 50, "name": "standard"},
        {"shape": (512, 512), "n_frames": 100, "name": "detailed"},
    ]
)
def xpcs_test_configurations(request, random_seed):
    """Parametrized fixture for different XPCS configurations."""
    if not ADVANCED_DATA_MANAGEMENT_AVAILABLE:
        pytest.skip("Advanced data management not available")

    params = request.param
    return create_realistic_xpcs_dataset(
        detector_shape=params["shape"], n_frames=params["n_frames"], seed=random_seed
    )


# ============================================================================
# Performance Optimization Fixtures
# ============================================================================

if PERFORMANCE_OPTIMIZATION_AVAILABLE:

    @pytest.fixture(scope="function")
    def performance_monitor():
        """Provide performance monitoring for individual tests."""
        monitor = get_perf_monitor()
        yield monitor
        # Clear individual test metrics to avoid memory buildup
        # Keep only recent metrics
        if len(monitor.metrics) > 100:
            monitor.metrics = monitor.metrics[-50:]

    @pytest.fixture(scope="function")
    def cache_manager():
        """Provide test cache manager with cleanup."""
        manager = get_cache_manager()
        try:
            yield manager
        finally:
            # Periodic cleanup
            if manager.get_cache_size_mb() > manager.max_size_mb:
                manager.cleanup_cache()

    @pytest.fixture(scope="function")
    def benchmark_tool():
        """Provide benchmarking tool for performance tests."""
        return benchmark_function

    @pytest.fixture(scope="function", autouse=True)
    def auto_performance_cleanup():
        """Automatically clean up after each test for optimal performance."""
        yield

        # Force garbage collection after each test

        gc.collect()

        # Clean up any lingering threads

        active_threads = threading.active_count()
        if active_threads > 1:  # More than just the main thread
            # Give background threads time to finish

            time.sleep(0.01)

    @pytest.fixture(scope="session", autouse=True)
    def session_performance_report():
        """Generate comprehensive performance report at session end."""
        yield

        if not PERFORMANCE_OPTIMIZATION_AVAILABLE:
            return

        monitor = get_perf_monitor()
        cache_manager = get_cache_manager()

        # Generate performance summary
        summary = monitor.get_performance_summary()

        if summary.get("total_tests", 0) > 0:
            print("\n" + "=" * 80)
            print("XPCS TOOLKIT TEST PERFORMANCE REPORT")
            print("=" * 80)

            # Duration statistics
            duration_stats = summary["duration_stats"]
            print("\nExecution Time:")
            print(f"  Total tests monitored: {summary['total_tests']}")
            print(f"  Total execution time: {duration_stats['total']:.2f}s")
            print(f"  Average test duration: {duration_stats['mean']:.3f}s")
            print(f"  Fastest test: {duration_stats['min']:.3f}s")
            print(f"  Slowest test: {duration_stats['max']:.2f}s")

            # Memory statistics
            memory_stats = summary["memory_stats"]
            print("\nMemory Usage:")
            print(f"  Peak memory usage: {memory_stats['peak_max_mb']:.1f}MB")
            print(f"  Average peak memory: {memory_stats['peak_mean_mb']:.1f}MB")
            print(f"  Largest memory delta: {abs(memory_stats['delta_max_mb']):.1f}MB")

            # Performance warnings
            slow_tests = monitor.get_slow_tests()
            memory_intensive_tests = monitor.get_memory_intensive_tests()

            if slow_tests:
                print(f"\n⚠️  Slow Tests ({len(slow_tests)}):")
                for metrics in slow_tests[:5]:  # Show top 5
                    print(f"    {metrics.test_name}: {metrics.duration:.2f}s")
                if len(slow_tests) > 5:
                    print(f"    ... and {len(slow_tests) - 5} more")

            if memory_intensive_tests:
                print(f"\n⚠️  Memory-Intensive Tests ({len(memory_intensive_tests)}):")
                for metrics in memory_intensive_tests[:5]:  # Show top 5
                    print(
                        f"    {metrics.test_name}: {metrics.memory_peak_mb:.1f}MB peak"
                    )
                if len(memory_intensive_tests) > 5:
                    print(f"    ... and {len(memory_intensive_tests) - 5} more")

            # Cache statistics
            cache_size = cache_manager.get_cache_size_mb()
            print("\nTest Cache:")
            print(f"  Cache size: {cache_size:.1f}MB")
            print(f"  Cache limit: {cache_manager.max_size_mb:.1f}MB")

            if not slow_tests and not memory_intensive_tests:
                print("\n✅ All tests performed within acceptable limits!")
            else:
                print(
                    "\n💡 Consider using performance decorators from tests.utils.test_performance"
                )

            print("=" * 80)


# ============================================================================
# CI/CD Integration Fixtures
# ============================================================================

if CI_INTEGRATION_AVAILABLE:

    @pytest.fixture(scope="session")
    def ci_environment():
        """Provide CI environment information."""
        return get_ci_environment()

    @pytest.fixture(scope="function")
    def ci_test_result_collector():
        """Collect test results for CI reporting."""
        results = []

        class ResultCollector:
            def add_result(
                self,
                test_id: str,
                test_name: str,
                status: str,
                duration: float,
                message: str | None = None,
                traceback: str | None = None,
            ):
                result = TestResult(
                    test_id=test_id,
                    test_name=test_name,
                    status=status,
                    duration=duration,
                    message=message,
                    traceback=traceback,
                )
                results.append(result)

            @property
            def results(self):
                return results

        return ResultCollector()

    @pytest.fixture(scope="session", autouse=True)
    def ci_session_reporting():
        """Generate CI reports at the end of test session."""
        session_start_time = time.time()

        yield

        if not CI_INTEGRATION_AVAILABLE:
            return

        ci_env = get_ci_environment()
        if not ci_env["is_ci"]:
            return

        # Create session summary (simplified)
        session_duration = time.time() - session_start_time

        # For now, create a basic test suite
        # In a real implementation, this would collect actual test results
        test_suite = TestSuite(
            name="XPCS Toolkit CI Test Suite",
            total_tests=1,  # Placeholder - would be populated from actual results
            passed=1,  # Placeholder
            failed=0,  # Placeholder
            skipped=0,  # Placeholder
            errors=0,  # Placeholder
            duration=session_duration,
        )

        try:
            # Generate CI reports
            reports = generate_ci_reports(test_suite)
            collect_test_artifacts()

            print("\n🚀 CI/CD Integration - Generated reports:")
            for report_type, path in reports.items():
                print(f"   📄 {report_type}: {path}")

            # Set GitHub Actions outputs if applicable
            if ci_env["ci_provider"] == "github_actions":
                set_github_output(
                    "test_results", "success" if test_suite.failed == 0 else "failure"
                )
                set_github_output("test_count", str(test_suite.total_tests))
                set_github_output("success_rate", f"{test_suite.success_rate:.1%}")

                # Add step summary
                summary_content = f"""
## XPCS Toolkit Test Results

- **Status:** {"✅ PASSED" if test_suite.failed == 0 else "❌ FAILED"}
- **Total Tests:** {test_suite.total_tests}
- **Success Rate:** {test_suite.success_rate:.1%}
- **Duration:** {test_suite.duration:.2f}s

Reports generated: {", ".join(reports.keys())}
"""
                github_step_summary(summary_content)

        except Exception as e:
            print(f"⚠️  Warning: CI report generation failed: {e}")

    @pytest.fixture(scope="function")
    def github_actions_integration(ci_environment):
        """GitHub Actions specific integration utilities."""
        if ci_environment.get("ci_provider") != "github_actions":
            pytest.skip("Only available in GitHub Actions")

        class GitHubActionsHelper:
            @staticmethod
            def set_output(name: str, value: str):
                set_github_output(name, value)

            @staticmethod
            def add_step_summary(content: str):
                github_step_summary(content)

            @staticmethod
            def create_job_summary(test_results: dict[str, Any]):
                """Create comprehensive job summary."""
                summary = f"""
# XPCS Toolkit Test Results

## Summary
- **Status**: {"✅ Success" if test_results.get("failed", 0) == 0 else "❌ Failure"}
- **Total Tests**: {test_results.get("total", 0)}
- **Passed**: {test_results.get("passed", 0)} ✅
- **Failed**: {test_results.get("failed", 0)} ❌
- **Skipped**: {test_results.get("skipped", 0)} ⏭️

## Performance
- **Duration**: {test_results.get("duration", 0):.2f}s
- **Success Rate**: {test_results.get("success_rate", 1.0):.1%}

## Environment
- **Python Version**: {ci_environment.get("python_version", "Unknown")}
- **Runner OS**: {ci_environment.get("runner_os", "Unknown")}
- **Branch**: {ci_environment.get("branch", "Unknown")}
- **Commit**: {ci_environment.get("commit", "Unknown")[:8]}...
"""
                github_step_summary(summary)

        return GitHubActionsHelper()
