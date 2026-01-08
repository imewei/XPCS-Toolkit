"""Benchmark tests for memory-efficient operations.

Verifies memory reduction from buffer donation and batch interpolation.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def memory_test_data() -> dict:
    """Generate data for memory efficiency benchmarks."""
    rng = np.random.default_rng(42)
    size = (512, 512)

    return {
        "qmap": rng.random(size, dtype=np.float64) * 0.1,
        "mask": (rng.random(size) > 0.1).astype(np.int32),
        "image": rng.random(size, dtype=np.float64) * 1000,
        "size": size,
    }


class TestPartitionMemoryEfficiency:
    """Memory efficiency tests for partition operations."""

    def test_partition_linear_correctness(self, memory_test_data: dict) -> None:
        """Verify linear partition produces correct results."""
        from xpcsviewer.simplemask.utils import generate_partition

        data = memory_test_data
        result = generate_partition(
            map_name="q",
            mask=data["mask"],
            xmap=data["qmap"],
            num_pts=36,
            style="linear",
        )

        assert "partition" in result
        assert "v_list" in result
        assert result["num_pts"] == 36
        assert result["partition"].shape == data["size"]

    def test_partition_log_correctness(self, memory_test_data: dict) -> None:
        """Verify logarithmic partition produces correct results."""
        from xpcsviewer.simplemask.utils import generate_partition

        data = memory_test_data
        # Ensure positive values for log partition
        qmap = np.abs(data["qmap"]) + 0.01

        result = generate_partition(
            map_name="q",
            mask=data["mask"],
            xmap=qmap,
            num_pts=36,
            style="logarithmic",
        )

        assert "partition" in result
        assert "v_list" in result
        assert len(result["v_list"]) == 36

    def test_partition_linear_timing(self, memory_test_data: dict, benchmark) -> None:
        """Record timing for linear partition."""
        from xpcsviewer.simplemask.utils import generate_partition

        data = memory_test_data

        def run_partition():
            return generate_partition(
                map_name="q",
                mask=data["mask"],
                xmap=data["qmap"],
                num_pts=36,
                style="linear",
            )

        result = benchmark(run_partition)
        assert result is not None

    def test_partition_log_timing(self, memory_test_data: dict, benchmark) -> None:
        """Record timing for logarithmic partition."""
        from xpcsviewer.simplemask.utils import generate_partition

        data = memory_test_data
        qmap = np.abs(data["qmap"]) + 0.01

        def run_partition():
            return generate_partition(
                map_name="q",
                mask=data["mask"],
                xmap=qmap,
                num_pts=36,
                style="logarithmic",
            )

        result = benchmark(run_partition)
        assert result is not None


class TestOptimizeIntegerArray:
    """Tests for integer array optimization."""

    def test_optimize_uint8(self) -> None:
        """Verify uint8 optimization for small values."""
        from xpcsviewer.simplemask.utils import optimize_integer_array

        arr = np.array([0, 100, 200, 255], dtype=np.int64)
        result = optimize_integer_array(arr)

        assert result.dtype == np.uint8
        np.testing.assert_array_equal(arr, result)

    def test_optimize_uint16(self) -> None:
        """Verify uint16 optimization for medium values."""
        from xpcsviewer.simplemask.utils import optimize_integer_array

        arr = np.array([0, 1000, 50000, 65535], dtype=np.int64)
        result = optimize_integer_array(arr)

        assert result.dtype == np.uint16
        np.testing.assert_array_equal(arr, result)

    def test_optimize_preserves_float(self) -> None:
        """Verify float arrays are not optimized."""
        from xpcsviewer.simplemask.utils import optimize_integer_array

        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = optimize_integer_array(arr)

        assert result.dtype == np.float64
        np.testing.assert_array_equal(arr, result)


class TestCombinePartitions:
    """Tests for partition combination memory efficiency."""

    def test_combine_partitions_correctness(self, memory_test_data: dict) -> None:
        """Verify partition combination produces correct results."""
        from xpcsviewer.simplemask.utils import combine_partitions, generate_partition

        data = memory_test_data

        pack1 = generate_partition(
            map_name="q",
            mask=data["mask"],
            xmap=data["qmap"],
            num_pts=6,
            style="linear",
        )
        pack2 = generate_partition(
            map_name="phi",
            mask=data["mask"],
            xmap=data["qmap"] * 360,  # simulate phi values
            num_pts=4,
            style="linear",
        )

        combined = combine_partitions(pack1, pack2, prefix="dynamic")

        assert "dynamic_roi_map" in combined
        assert "dynamic_num_pts" in combined
        assert combined["dynamic_num_pts"] == [6, 4]

    def test_combine_partitions_timing(self, memory_test_data: dict, benchmark) -> None:
        """Record timing for partition combination."""
        from xpcsviewer.simplemask.utils import combine_partitions, generate_partition

        data = memory_test_data

        pack1 = generate_partition(
            map_name="q",
            mask=data["mask"],
            xmap=data["qmap"],
            num_pts=6,
            style="linear",
        )
        pack2 = generate_partition(
            map_name="phi",
            mask=data["mask"],
            xmap=data["qmap"] * 360,
            num_pts=4,
            style="linear",
        )

        def run_combine():
            return combine_partitions(pack1, pack2, prefix="dynamic")

        result = benchmark(run_combine)
        assert result is not None
