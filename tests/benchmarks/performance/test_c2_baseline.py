"""Baseline benchmarks for C2 two-time correlation statistics.

Establishes performance baselines before vectorization optimization.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def c2_statistics_data() -> dict:
    """Generate data for C2 statistics benchmarks."""
    rng = np.random.default_rng(42)
    size = 200  # 200x200 C2 matrix

    # Simulate C2 correlation matrix (symmetric)
    c2_base = rng.random((size, size), dtype=np.float64)
    c2_matrix = (c2_base + c2_base.T) / 2  # Make symmetric
    c2_matrix += np.eye(size) * 0.1  # Boost diagonal

    return {
        "c2_matrix": c2_matrix,
        "size": size,
    }


def off_diagonal_stats_loop(c2_array: np.ndarray) -> dict:
    """Calculate off-diagonal statistics using loop (baseline)."""
    n = c2_array.shape[0]
    off_diag_sum = 0.0
    off_diag_count = 0

    for i in range(n):
        for j in range(n):
            if i != j:
                off_diag_sum += c2_array[i, j]
                off_diag_count += 1

    off_diag_mean = off_diag_sum / off_diag_count if off_diag_count > 0 else 0.0

    return {
        "off_diagonal_mean": off_diag_mean,
        "off_diagonal_sum": off_diag_sum,
        "off_diagonal_count": off_diag_count,
    }


def off_diagonal_stats_vectorized(c2_array: np.ndarray) -> dict:
    """Calculate off-diagonal statistics using vectorized ops (target optimization)."""
    n = c2_array.shape[0]

    # Total sum and diagonal sum
    total_sum = np.sum(c2_array)
    diag_sum = np.trace(c2_array)

    # Off-diagonal is total minus diagonal
    off_diag_sum = total_sum - diag_sum
    off_diag_count = n * n - n

    off_diag_mean = off_diag_sum / off_diag_count if off_diag_count > 0 else 0.0

    return {
        "off_diagonal_mean": off_diag_mean,
        "off_diagonal_sum": off_diag_sum,
        "off_diagonal_count": off_diag_count,
    }


def diagonal_correction_loop(c2_array: np.ndarray) -> np.ndarray:
    """Apply diagonal correction using loop (baseline for Numba)."""
    result = c2_array.copy()
    n = c2_array.shape[0]

    for i in range(n):
        if i == 0:
            result[i, i] = c2_array[i, i + 1]
        elif i == n - 1:
            result[i, i] = c2_array[i - 1, i]
        else:
            result[i, i] = (c2_array[i - 1, i] + c2_array[i, i + 1]) / 2.0

    return result


class TestC2StatisticsBaseline:
    """Baseline benchmarks for C2 statistics operations."""

    def test_loop_stats_correctness(self, c2_statistics_data: dict) -> None:
        """Verify loop-based stats produces correct results."""
        data = c2_statistics_data
        result = off_diagonal_stats_loop(data["c2_matrix"])

        assert np.isfinite(result["off_diagonal_mean"])
        assert result["off_diagonal_count"] == data["size"] ** 2 - data["size"]

    def test_vectorized_stats_correctness(self, c2_statistics_data: dict) -> None:
        """Verify vectorized stats produces correct results."""
        data = c2_statistics_data
        result = off_diagonal_stats_vectorized(data["c2_matrix"])

        assert np.isfinite(result["off_diagonal_mean"])
        assert result["off_diagonal_count"] == data["size"] ** 2 - data["size"]

    def test_stats_equivalence(self, c2_statistics_data: dict) -> None:
        """Verify loop and vectorized produce same results."""
        data = c2_statistics_data

        loop_result = off_diagonal_stats_loop(data["c2_matrix"])
        vec_result = off_diagonal_stats_vectorized(data["c2_matrix"])

        np.testing.assert_allclose(
            loop_result["off_diagonal_mean"],
            vec_result["off_diagonal_mean"],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            loop_result["off_diagonal_sum"],
            vec_result["off_diagonal_sum"],
            rtol=1e-10,
        )

    def test_loop_stats_baseline_timing(
        self, c2_statistics_data: dict, benchmark
    ) -> None:
        """Record baseline timing for loop-based stats."""
        data = c2_statistics_data

        def run_loop_stats():
            return off_diagonal_stats_loop(data["c2_matrix"])

        result = benchmark(run_loop_stats)
        assert result is not None

    def test_vectorized_stats_timing(self, c2_statistics_data: dict, benchmark) -> None:
        """Record timing for vectorized stats (comparison)."""
        data = c2_statistics_data

        def run_vectorized_stats():
            return off_diagonal_stats_vectorized(data["c2_matrix"])

        result = benchmark(run_vectorized_stats)
        assert result is not None

    def test_diagonal_correction_baseline_timing(
        self, c2_statistics_data: dict, benchmark
    ) -> None:
        """Record baseline timing for diagonal correction (Numba target)."""
        data = c2_statistics_data

        def run_diagonal_correction():
            return diagonal_correction_loop(data["c2_matrix"])

        result = benchmark(run_diagonal_correction)
        assert result is not None
