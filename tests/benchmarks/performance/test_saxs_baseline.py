"""Baseline benchmarks for SAXS 1D binning.

Establishes performance baselines before vectorization optimization.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def saxs_binning_data() -> dict:
    """Generate data for SAXS binning benchmarks."""
    rng = np.random.default_rng(42)
    size = (512, 512)
    num_bins = 100

    # Simulate detector data
    intensities = rng.random(size, dtype=np.float64) * 1000
    bin_indices = rng.integers(-1, num_bins + 1, size=size)  # -1 for invalid pixels

    return {
        "intensities": intensities,
        "bin_indices": bin_indices,
        "num_bins": num_bins,
        "size": size,
    }


def loop_based_binning(
    intensities: np.ndarray, bin_indices: np.ndarray, num_bins: int
) -> np.ndarray:
    """Original loop-based binning implementation (baseline)."""
    sums = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)

    flat_intensities = intensities.flatten()
    flat_indices = bin_indices.flatten()

    for i, idx in enumerate(flat_indices):
        if 0 <= idx < num_bins:
            sums[idx] += flat_intensities[i]
            counts[idx] += 1

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(counts > 0, sums / counts, 0.0)

    return result


def vectorized_binning(
    intensities: np.ndarray, bin_indices: np.ndarray, num_bins: int
) -> np.ndarray:
    """Vectorized binning using np.bincount (target optimization)."""
    flat_intensities = intensities.flatten()
    flat_indices = bin_indices.flatten()

    # Create valid mask
    valid_mask = (flat_indices >= 0) & (flat_indices < num_bins)
    valid_indices = flat_indices[valid_mask]
    valid_intensities = flat_intensities[valid_mask]

    # Use bincount for vectorized accumulation
    sums = np.bincount(valid_indices, weights=valid_intensities, minlength=num_bins)
    counts = np.bincount(valid_indices, minlength=num_bins)

    # Compute means
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(counts > 0, sums / counts, 0.0)

    return result


class TestSaxsBinningBaseline:
    """Baseline benchmarks for SAXS binning operations."""

    def test_loop_binning_correctness(self, saxs_binning_data: dict) -> None:
        """Verify loop-based binning produces correct results."""
        data = saxs_binning_data
        result = loop_based_binning(
            data["intensities"], data["bin_indices"], data["num_bins"]
        )

        assert len(result) == data["num_bins"]
        assert np.all(np.isfinite(result))

    def test_vectorized_binning_correctness(self, saxs_binning_data: dict) -> None:
        """Verify vectorized binning produces correct results."""
        data = saxs_binning_data
        result = vectorized_binning(
            data["intensities"], data["bin_indices"], data["num_bins"]
        )

        assert len(result) == data["num_bins"]
        assert np.all(np.isfinite(result))

    def test_binning_equivalence(self, saxs_binning_data: dict) -> None:
        """Verify loop and vectorized produce same results."""
        data = saxs_binning_data

        loop_result = loop_based_binning(
            data["intensities"], data["bin_indices"], data["num_bins"]
        )
        vec_result = vectorized_binning(
            data["intensities"], data["bin_indices"], data["num_bins"]
        )

        np.testing.assert_allclose(loop_result, vec_result, rtol=1e-10)

    def test_loop_binning_baseline_timing(
        self, saxs_binning_data: dict, benchmark
    ) -> None:
        """Record baseline timing for loop-based binning."""
        data = saxs_binning_data

        def run_loop_binning():
            return loop_based_binning(
                data["intensities"], data["bin_indices"], data["num_bins"]
            )

        result = benchmark(run_loop_binning)
        assert result is not None

    def test_vectorized_binning_timing(
        self, saxs_binning_data: dict, benchmark
    ) -> None:
        """Record timing for vectorized binning (comparison)."""
        data = saxs_binning_data

        def run_vectorized_binning():
            return vectorized_binning(
                data["intensities"], data["bin_indices"], data["num_bins"]
            )

        result = benchmark(run_vectorized_binning)
        assert result is not None
