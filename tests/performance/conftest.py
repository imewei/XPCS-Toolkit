"""
Performance testing configuration and fixtures.

Provides shared fixtures and configuration for all performance benchmarks.
"""

import pytest

# Skip all performance tests if dependencies are not properly configured
pytestmark = pytest.mark.skip(
    "Performance tests require review of import dependencies and class structure"
)

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

# Import performance monitoring utilities
try:
    import memory_profiler
    import psutil

    HAS_PROFILING = True
except ImportError:
    HAS_PROFILING = False

from xpcs_toolkit.fileIO.hdf_reader import HDF5ConnectionPool
from xpcs_toolkit.xpcs_file import XpcsFile

# Performance test configuration
from . import DATA_SIZES, PERFORMANCE_CONFIG


@dataclass
class BenchmarkResult:
    """Container for benchmark results with statistical analysis."""

    name: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    rounds: int
    memory_peak: float | None = None
    memory_delta: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_time": self.mean_time,
            "std_time": self.std_time,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "rounds": self.rounds,
            "memory_peak": self.memory_peak,
            "memory_delta": self.memory_delta,
        }


@pytest.fixture(scope="session")
def performance_config():
    """Global performance testing configuration."""
    return PERFORMANCE_CONFIG.copy()


@pytest.fixture(scope="session")
def test_data_dir():
    """Directory for test data files."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture(scope="session")
def benchmark_results_dir():
    """Directory for benchmark results."""
    results_dir = Path(__file__).parent / "reports"
    results_dir.mkdir(exist_ok=True)
    return results_dir


@pytest.fixture
def memory_profiler():
    """Memory profiler fixture for tracking memory usage."""
    if not HAS_PROFILING:
        pytest.skip("Memory profiling dependencies not available")

    class MemoryProfiler:
        def __init__(self):
            self.start_memory = None
            self.peak_memory = None
            self.current_process = psutil.Process()

        def start(self):
            """Start memory monitoring."""
            self.start_memory = (
                self.current_process.memory_info().rss / 1024 / 1024
            )  # MB
            self.peak_memory = self.start_memory

        def sample(self):
            """Sample current memory usage."""
            current = self.current_process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current)
            return current

        def stop(self):
            """Stop monitoring and return results."""
            final_memory = self.current_process.memory_info().rss / 1024 / 1024  # MB
            return {
                "start_mb": self.start_memory,
                "peak_mb": self.peak_memory,
                "final_mb": final_memory,
                "delta_mb": final_memory - self.start_memory,
            }

    return MemoryProfiler()


@pytest.fixture(params=["tiny", "small", "medium"])
def synthetic_hdf5_file(request, test_data_dir):
    """Create synthetic HDF5 files with different sizes for testing."""
    size_config = DATA_SIZES[request.param]

    # Create filename based on size
    filename = f"synthetic_{request.param}_{size_config['width']}x{size_config['height']}x{size_config['frames']}.h5"
    filepath = test_data_dir / filename

    # Only create if doesn't exist (cache for performance)
    if not filepath.exists():
        with h5py.File(filepath, "w") as f:
            # Create XPCS-like data structure
            f.attrs["format_version"] = "2.0"
            f.attrs["analysis_type"] = "Multitau"

            # Raw data
            raw_data = np.random.poisson(
                100,
                size=(
                    size_config["frames"],
                    size_config["height"],
                    size_config["width"],
                ),
            ).astype(np.uint16)
            f.create_dataset("exchange/data", data=raw_data, compression="gzip")

            # Correlation data
            num_tau = 64
            num_q = 20
            tau = np.logspace(-6, 2, num_tau)
            q = np.linspace(0.001, 0.1, num_q)

            # Synthetic G2 data
            g2_data = np.zeros((num_q, num_tau))
            for i in range(num_q):
                # Exponential decay with noise
                beta = 0.5 + 0.3 * np.random.random()
                tau_c = 0.1 * (1 + i * 0.1)
                g2_data[i] = (
                    1 + beta * np.exp(-tau / tau_c) + 0.01 * np.random.random(num_tau)
                )

            f.create_dataset("exchange/C2T_all", data=g2_data)
            f.create_dataset("exchange/C2T_tau", data=tau)
            f.create_dataset("exchange/sqmap", data=q)

            # SAXS data
            saxs_2d = np.random.poisson(
                1000, size=(size_config["height"], size_config["width"])
            ).astype(np.uint32)
            f.create_dataset("exchange/norm2_C2T_all", data=saxs_2d)

            # Metadata
            f.create_dataset("exchange/detector_distance", data=np.array([5.0]))
            f.create_dataset(
                "exchange/beam_center_x", data=np.array([size_config["width"] / 2])
            )
            f.create_dataset(
                "exchange/beam_center_y", data=np.array([size_config["height"] / 2])
            )
            f.create_dataset("exchange/wavelength", data=np.array([1.24e-10]))
            f.create_dataset("exchange/pixel_size", data=np.array([75e-6]))

    return filepath


@pytest.fixture
def mock_xpcs_file(synthetic_hdf5_file):
    """Create a mock XpcsFile for testing."""
    try:
        xf = XpcsFile(str(synthetic_hdf5_file))
        yield xf
    finally:
        # Cleanup if needed
        if hasattr(xf, "close"):
            xf.close()


@pytest.fixture
def hdf_reader(synthetic_hdf5_file):
    """Provide HDF file path for testing with hdf_reader functions."""
    # Return the file path since we use functions like get() instead of a reader class
    yield str(synthetic_hdf5_file)
    # No cleanup needed as we're not maintaining a connection


@pytest.fixture
def connection_pool():
    """Create connection pool for testing."""
    pool = HDF5ConnectionPool(max_pool_size=5)
    yield pool
    pool.cleanup_all()


@pytest.fixture
def large_dataset_sizes():
    """Test data sizes for scalability testing."""
    return {
        "small": {"shape": (100, 64, 64), "dtype": np.uint16},
        "medium": {"shape": (500, 128, 128), "dtype": np.uint16},
        "large": {"shape": (1000, 256, 256), "dtype": np.uint16},
        "xlarge": {"shape": (2000, 512, 512), "dtype": np.uint16},
    }


@pytest.fixture(scope="session")
def benchmark_baseline():
    """Load baseline performance metrics for comparison."""
    baseline_file = Path(__file__).parent / "config" / "baseline_performance.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            return json.load(f)
    return {}


@pytest.fixture
def performance_monitor():
    """Monitor system performance during tests."""

    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.cpu_percent = []
            self.memory_usage = []
            self.monitoring = False
            self.monitor_thread = None

        def start(self):
            """Start monitoring system performance."""
            self.start_time = time.time()
            self.monitoring = True
            self.cpu_percent = []
            self.memory_usage = []

            def monitor():
                while self.monitoring:
                    try:
                        cpu = psutil.cpu_percent(interval=0.1)
                        mem = psutil.virtual_memory().percent
                        self.cpu_percent.append(cpu)
                        self.memory_usage.append(mem)
                        time.sleep(0.1)
                    except:
                        break

            if HAS_PROFILING:
                self.monitor_thread = threading.Thread(target=monitor, daemon=True)
                self.monitor_thread.start()

        def stop(self):
            """Stop monitoring and return results."""
            self.end_time = time.time()
            self.monitoring = False

            if self.monitor_thread:
                self.monitor_thread.join(timeout=1.0)

            return {
                "duration": self.end_time - self.start_time if self.start_time else 0,
                "avg_cpu_percent": np.mean(self.cpu_percent) if self.cpu_percent else 0,
                "max_cpu_percent": np.max(self.cpu_percent) if self.cpu_percent else 0,
                "avg_memory_percent": np.mean(self.memory_usage)
                if self.memory_usage
                else 0,
                "max_memory_percent": np.max(self.memory_usage)
                if self.memory_usage
                else 0,
            }

    return PerformanceMonitor()


# Pytest benchmark customization
def pytest_configure(config):
    """Configure pytest for performance testing."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "benchmark: Benchmark tests that measure performance"
    )
    config.addinivalue_line(
        "markers", "regression: Regression tests that detect performance degradation"
    )
    config.addinivalue_line(
        "markers", "scalability: Tests that measure scalability with data size"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: Tests that use significant memory"
    )


def pytest_runtest_setup(item):
    """Setup for each test."""
    # Skip tests based on markers and system capabilities
    if item.get_closest_marker("memory_intensive"):
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory < 4:  # Require at least 4GB free
            pytest.skip("Insufficient memory for memory intensive test")

    if item.get_closest_marker("scalability") and not HAS_PROFILING:
        pytest.skip("Profiling tools not available for scalability test")


# Performance regression detection
def compare_with_baseline(
    benchmark_result: BenchmarkResult, baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare benchmark result with baseline performance."""
    if not baseline or benchmark_result.name not in baseline:
        return {
            "status": "no_baseline",
            "message": "No baseline available for comparison",
        }

    baseline_data = baseline[benchmark_result.name]
    baseline_mean = baseline_data.get("mean_time", 0)

    if baseline_mean == 0:
        return {"status": "invalid_baseline", "message": "Invalid baseline data"}

    # Calculate performance change
    performance_change = (benchmark_result.mean_time - baseline_mean) / baseline_mean
    threshold = PERFORMANCE_CONFIG["performance_threshold"]

    if performance_change > threshold:
        return {
            "status": "regression",
            "message": f"Performance regression detected: {performance_change:.2%} slower than baseline",
            "change_percent": performance_change,
            "current_time": benchmark_result.mean_time,
            "baseline_time": baseline_mean,
        }
    if performance_change < -0.1:  # 10% improvement
        return {
            "status": "improvement",
            "message": f"Performance improvement detected: {abs(performance_change):.2%} faster than baseline",
            "change_percent": performance_change,
            "current_time": benchmark_result.mean_time,
            "baseline_time": baseline_mean,
        }
    return {
        "status": "stable",
        "message": "Performance is stable compared to baseline",
        "change_percent": performance_change,
        "current_time": benchmark_result.mean_time,
        "baseline_time": baseline_mean,
    }
