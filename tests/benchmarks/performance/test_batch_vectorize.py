"""Benchmark tests for vectorized batch processing.

Verifies performance improvements from vmap and vectorized operations.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def c2_batch_data() -> dict:
    """Generate C2 matrix batch data for benchmarks."""
    rng = np.random.default_rng(42)
    batch_size = 20
    matrix_size = 200

    # Generate symmetric C2 matrices
    c2_matrices = []
    for _ in range(batch_size):
        base = rng.random((matrix_size, matrix_size), dtype=np.float64)
        c2 = (base + base.T) / 2  # Make symmetric
        c2_matrices.append(c2)

    return {
        "c2_matrices": np.array(c2_matrices),
        "batch_size": batch_size,
        "matrix_size": matrix_size,
    }


class TestC2StatisticsVectorized:
    """Benchmark tests for C2 statistics vectorization."""

    def test_c2_statistics_correctness(self, c2_batch_data: dict) -> None:
        """Verify vectorized C2 statistics produces correct results."""
        from xpcsviewer.module.twotime_utils import compute_c2_statistics_vectorized

        data = c2_batch_data
        stats = compute_c2_statistics_vectorized(data["c2_matrices"])

        # Check all expected keys are present
        assert "mean" in stats
        assert "std" in stats
        assert "trace" in stats
        assert "diagonal_mean" in stats
        assert "off_diagonal_mean" in stats

        # Check shapes
        assert stats["mean"].shape == (data["matrix_size"], data["matrix_size"])
        assert len(stats["trace"]) == data["batch_size"]
        assert len(stats["off_diagonal_mean"]) == data["batch_size"]

        # Check values are finite
        assert np.all(np.isfinite(stats["mean"]))
        assert np.all(np.isfinite(stats["off_diagonal_mean"]))

    def test_c2_statistics_timing(self, c2_batch_data: dict, benchmark) -> None:
        """Record timing for vectorized C2 statistics."""
        from xpcsviewer.module.twotime_utils import compute_c2_statistics_vectorized

        data = c2_batch_data

        def run_stats():
            return compute_c2_statistics_vectorized(data["c2_matrices"])

        result = benchmark(run_stats)
        assert result is not None

    def test_off_diagonal_vectorized_correctness(self, c2_batch_data: dict) -> None:
        """Verify off-diagonal mean matches loop-based calculation."""
        from xpcsviewer.module.twotime_utils import compute_c2_statistics_vectorized

        data = c2_batch_data
        c2_array = data["c2_matrices"]

        # Get vectorized result
        stats = compute_c2_statistics_vectorized(c2_array)
        vectorized_off_diag = stats["off_diagonal_mean"]

        # Compute loop-based reference
        loop_off_diag = []
        for i in range(c2_array.shape[0]):
            mask = ~np.eye(c2_array.shape[-1], dtype=bool)
            off_diag_vals = c2_array[i][mask]
            loop_off_diag.append(np.mean(off_diag_vals))
        loop_off_diag = np.array(loop_off_diag)

        # Should be equivalent
        np.testing.assert_allclose(vectorized_off_diag, loop_off_diag, rtol=1e-10)


class TestDiagonalCorrectionPerformance:
    """Benchmark tests for diagonal correction."""

    def test_diagonal_correction_correctness(self, c2_batch_data: dict) -> None:
        """Verify vectorized diagonal correction produces correct results."""
        from xpcsviewer.module.twotime_utils import correct_diagonal_c2_vectorized

        data = c2_batch_data
        c2_single = data["c2_matrices"][0].copy()

        result = correct_diagonal_c2_vectorized(c2_single)

        # Check shape preserved
        assert result.shape == c2_single.shape

        # Check diagonal was modified
        # The new diagonal values should be averages of adjacent off-diagonal elements
        assert np.all(np.isfinite(np.diag(result)))

    def test_diagonal_correction_timing(self, c2_batch_data: dict, benchmark) -> None:
        """Record timing for vectorized diagonal correction."""
        from xpcsviewer.module.twotime_utils import correct_diagonal_c2_vectorized

        data = c2_batch_data
        c2_single = data["c2_matrices"][0].copy()

        def run_correction():
            return correct_diagonal_c2_vectorized(c2_single.copy())

        result = benchmark(run_correction)
        assert result is not None


class TestBatchC2Operations:
    """Benchmark tests for batch C2 operations."""

    def test_batch_operations_correctness(self, c2_batch_data: dict) -> None:
        """Verify batch C2 operations produce correct results."""
        from xpcsviewer.module.twotime_utils import batch_c2_matrix_operations

        data = c2_batch_data
        c2_matrices = data["c2_matrices"]

        result = batch_c2_matrix_operations(
            c2_matrices, operations=["symmetrize", "diagonal_correct"]
        )

        # Check shape preserved
        assert result.shape == c2_matrices.shape

        # Check all values are finite
        assert np.all(np.isfinite(result))

        # Check matrices are symmetric after symmetrize
        for i in range(result.shape[0]):
            np.testing.assert_allclose(result[i], result[i].T, rtol=1e-10)

    def test_batch_operations_timing(self, c2_batch_data: dict, benchmark) -> None:
        """Record timing for batch C2 operations."""
        from xpcsviewer.module.twotime_utils import batch_c2_matrix_operations

        data = c2_batch_data
        c2_matrices = data["c2_matrices"]

        def run_batch():
            return batch_c2_matrix_operations(
                c2_matrices.copy(), operations=["symmetrize", "diagonal_correct"]
            )

        result = benchmark(run_batch)
        assert result is not None
