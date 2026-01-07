"""Benchmark tests for Q-map computation (T061).

Benchmarks for measuring JIT speedup on Q-map and partition operations (US3).
"""

from __future__ import annotations

import numpy as np
import pytest

# Check if JAX is available
try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestQmapBenchmarks:
    """Benchmarks for Q-map computation."""

    @pytest.mark.benchmark(group="qmap")
    def test_qmap_small_detector(self, benchmark) -> None:
        """Benchmark Q-map on small detector (256x256)."""
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Parameters: energy (keV), center (row, col), shape, pixel_size (mm), distance (mm)
        energy = 10.0  # keV
        shape = (256, 256)
        center = (128.0, 128.0)
        pix_dim = 0.075  # mm (75 microns)
        det_dist = 5000.0  # mm (5 meters)

        # Warmup
        _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Benchmark
        result = benchmark(
            compute_transmission_qmap, energy, center, shape, pix_dim, det_dist
        )

        assert result[0] is not None

    @pytest.mark.benchmark(group="qmap")
    def test_qmap_medium_detector(self, benchmark) -> None:
        """Benchmark Q-map on medium detector (512x512)."""
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        energy = 10.0  # keV
        shape = (512, 512)
        center = (256.0, 256.0)
        pix_dim = 0.075  # mm
        det_dist = 5000.0  # mm

        # Warmup
        _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Benchmark
        result = benchmark(
            compute_transmission_qmap, energy, center, shape, pix_dim, det_dist
        )

        assert result[0] is not None

    @pytest.mark.benchmark(group="qmap")
    def test_qmap_large_detector(self, benchmark) -> None:
        """Benchmark Q-map on large detector (1024x1024)."""
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        energy = 10.0  # keV
        shape = (1024, 1024)
        center = (512.0, 512.0)
        pix_dim = 0.075  # mm
        det_dist = 5000.0  # mm

        # Warmup
        _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Benchmark
        result = benchmark(
            compute_transmission_qmap, energy, center, shape, pix_dim, det_dist
        )

        assert result[0] is not None

    @pytest.mark.benchmark(group="qmap")
    def test_qmap_xlarge_detector_2048(self, benchmark) -> None:
        """Benchmark Q-map on extra-large detector (2048x2048) for SC-001.

        SC-001: Q-map computation completes at least 5x faster on GPU
        compared to CPU-only execution for large detectors (2048x2048 pixels).

        Note: This test measures baseline performance. GPU vs CPU comparison
        requires running on systems with GPU hardware.
        """
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        energy = 10.0  # keV
        shape = (2048, 2048)
        center = (1024.0, 1024.0)
        pix_dim = 0.075  # mm (75 microns)
        det_dist = 5000.0  # mm (5 meters)

        # Warmup
        _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Benchmark - returns (qmap_dict, metadata)
        result = benchmark(
            compute_transmission_qmap, energy, center, shape, pix_dim, det_dist
        )

        # Result is a tuple: (qmap_dict, metadata)
        qmap_dict, metadata = result
        assert qmap_dict is not None
        assert "q" in qmap_dict
        # Verify output shape matches expected 2048x2048
        assert qmap_dict["q"].shape == (2048, 2048)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestPartitionBenchmarks:
    """Benchmarks for partition computation."""

    @pytest.fixture
    def qmap_512(self):
        """Create 512x512 Q-map for benchmarking."""
        np.random.seed(42)
        return np.abs(np.random.randn(512, 512) * 0.1 + 0.05)

    @pytest.fixture
    def mask_512(self):
        """Create 512x512 mask for benchmarking."""
        return np.ones((512, 512), dtype=bool)

    @pytest.mark.benchmark(group="partition")
    def test_partition_linear_36bins(self, benchmark, qmap_512, mask_512) -> None:
        """Benchmark partition with 36 linear bins."""
        from xpcsviewer.simplemask.utils import create_partition

        # Warmup
        _ = create_partition(qmap_512, mask_512, n_bins=36, spacing="linear")

        # Benchmark
        result = benchmark(
            create_partition, qmap_512, mask_512, n_bins=36, spacing="linear"
        )

        assert result is not None

    @pytest.mark.benchmark(group="partition")
    def test_partition_log_36bins(self, benchmark, qmap_512, mask_512) -> None:
        """Benchmark partition with 36 log bins."""
        from xpcsviewer.simplemask.utils import create_partition

        # Warmup
        _ = create_partition(qmap_512, mask_512, n_bins=36, spacing="log")

        # Benchmark
        result = benchmark(
            create_partition, qmap_512, mask_512, n_bins=36, spacing="log"
        )

        assert result is not None


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestBackendOperationBenchmarks:
    """Benchmarks for backend operations."""

    @pytest.fixture
    def large_array(self):
        """Create large array for benchmarking."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()
        return backend.linspace(0, 10, 100000)

    @pytest.mark.benchmark(group="backend_ops")
    def test_sin_cos_benchmark(self, benchmark, large_array) -> None:
        """Benchmark sin/cos operations."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()

        @backend.jit
        def compute(x):
            return backend.sin(x) + backend.cos(x)

        # Warmup
        _ = compute(large_array)

        # Benchmark
        result = benchmark(compute, large_array)
        assert result is not None

    @pytest.mark.benchmark(group="backend_ops")
    def test_exp_log_benchmark(self, benchmark, large_array) -> None:
        """Benchmark exp/log operations."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()

        @backend.jit
        def compute(x):
            return backend.exp(-x) + backend.log(x + 1)

        # Warmup
        _ = compute(large_array)

        # Benchmark
        result = benchmark(compute, large_array)
        assert result is not None

    @pytest.mark.benchmark(group="backend_ops")
    def test_matmul_benchmark(self, benchmark) -> None:
        """Benchmark matrix multiplication."""
        from xpcsviewer.backends import get_backend

        backend = get_backend()

        # Use smaller matrices to avoid timeout on CPU-only systems
        a = backend.ones((200, 200))
        b = backend.ones((200, 200))

        @backend.jit
        def compute(a, b):
            return a @ b

        # Warmup
        _ = compute(a, b)

        # Benchmark
        result = benchmark(compute, a, b)
        assert result is not None
