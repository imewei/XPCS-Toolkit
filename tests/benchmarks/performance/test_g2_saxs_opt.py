"""Benchmarks for G2 and SAXS vectorized operation optimizations.

Measures the impact of:
- Replacing nested Python loops with np.add.at scatter-accumulate in
  vectorized_q_binning() (saxs1d.py).
- Replacing per-curve list comprehension with axis= trapezoid call in
  vectorized_intensity_normalization() (saxs1d.py).
- Vectorized batch fast path in batch_g2_normalization() (g2mod.py).
- ensure_numpy() guards in g2mod.py preventing silent JAX host-device transfers.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers: reference (old) implementations
# ---------------------------------------------------------------------------


def _q_binning_loop(q_values, intensities, q_min, q_max, num_bins):
    """Original nested-loop implementation (baseline)."""
    bin_edges = np.linspace(q_min, q_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_indices = np.digitize(q_values, bin_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < num_bins)
    valid_indices = bin_indices[valid]
    bin_counts = np.bincount(valid_indices, minlength=num_bins).astype(np.float64)[
        :num_bins
    ]
    valid_intensities = (
        intensities[..., valid] if intensities.ndim > 1 else intensities[valid]
    )
    if intensities.ndim == 1:
        binned_intensity = np.zeros(num_bins)
        for b in range(num_bins):
            if bin_counts[b] > 0:
                binned_intensity[b] = np.mean(valid_intensities[valid_indices == b])
    else:
        num_phi = intensities.shape[0]
        binned_intensity = np.zeros((num_phi, num_bins))
        for b in range(num_bins):
            if bin_counts[b] > 0:
                mask_b = valid_indices == b
                for phi_idx in range(num_phi):
                    binned_intensity[phi_idx, b] = np.mean(
                        valid_intensities[phi_idx, mask_b]
                    )
    return bin_centers, binned_intensity, bin_counts


def _area_norm_loop(intensities, q_values):
    """Original list-comprehension area normalization (baseline)."""
    areas = np.array(
        [np.trapezoid(intensities[i], q_values) for i in range(intensities.shape[0])]
    )
    return intensities / areas[:, np.newaxis]


def _batch_norm_loop(g2_data_list):
    """Original per-item loop batch normalization (baseline)."""
    normalized_data = []
    for g2_data in g2_data_list:
        max_vals = np.max(g2_data, axis=0, keepdims=True)
        max_vals = np.where(max_vals == 0, 1.0, max_vals)
        normalized_data.append(g2_data / max_vals)
    return normalized_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def binning_1d_data():
    rng = np.random.default_rng(0)
    n_pixels = 50_000
    q = rng.uniform(0.01, 1.0, n_pixels)
    I = rng.exponential(1.0, n_pixels)
    return q, I, 0.01, 1.0, 200


@pytest.fixture
def binning_2d_data():
    """2D data: 32 phi slices x 50k pixels — the worst-case nested loop."""
    rng = np.random.default_rng(1)
    n_pixels = 50_000
    n_phi = 32
    q = rng.uniform(0.01, 1.0, n_pixels)
    I = rng.exponential(1.0, (n_phi, n_pixels))
    return q, I, 0.01, 1.0, 200


@pytest.fixture
def area_norm_data():
    rng = np.random.default_rng(2)
    n_phi, n_q = 64, 512
    q = np.linspace(0.01, 1.0, n_q)
    I = rng.exponential(1.0, (n_phi, n_q))
    return I, q


@pytest.fixture
def batch_norm_data():
    """Batch of 50 G2 datasets, each (100 time points x 36 q values)."""
    rng = np.random.default_rng(3)
    return [rng.random((100, 36)) + 0.5 for _ in range(50)]


# ---------------------------------------------------------------------------
# Benchmarks: q-binning 1D
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="q_binning_1d")
def test_q_binning_1d_loop(benchmark, binning_1d_data):
    """Baseline: nested Python loop (1-D)."""
    q, I, q_min, q_max, num_bins = binning_1d_data
    benchmark(_q_binning_loop, q, I, q_min, q_max, num_bins)


@pytest.mark.benchmark(group="q_binning_1d")
def test_q_binning_1d_vectorized(benchmark, binning_1d_data):
    """Optimized: np.add.at scatter-accumulate (1-D)."""
    from xpcsviewer.module.saxs1d import vectorized_q_binning

    q, I, q_min, q_max, num_bins = binning_1d_data
    benchmark(vectorized_q_binning, q, I, q_min, q_max, num_bins)


# ---------------------------------------------------------------------------
# Benchmarks: q-binning 2D (critical path)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="q_binning_2d")
def test_q_binning_2d_loop(benchmark, binning_2d_data):
    """Baseline: doubly nested Python loop (2-D phi x bins)."""
    q, I, q_min, q_max, num_bins = binning_2d_data
    benchmark(_q_binning_loop, q, I, q_min, q_max, num_bins)


@pytest.mark.benchmark(group="q_binning_2d")
def test_q_binning_2d_vectorized(benchmark, binning_2d_data):
    """Optimized: broadcast np.add.at across phi slices simultaneously."""
    from xpcsviewer.module.saxs1d import vectorized_q_binning

    q, I, q_min, q_max, num_bins = binning_2d_data
    benchmark(vectorized_q_binning, q, I, q_min, q_max, num_bins)


# ---------------------------------------------------------------------------
# Benchmarks: area normalization
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="area_norm")
def test_area_norm_loop(benchmark, area_norm_data):
    """Baseline: list comprehension over phi slices."""
    I, q = area_norm_data
    benchmark(_area_norm_loop, I, q)


@pytest.mark.benchmark(group="area_norm")
def test_area_norm_vectorized(benchmark, area_norm_data):
    """Optimized: np.trapezoid with axis= keyword."""
    from xpcsviewer.module.saxs1d import vectorized_intensity_normalization

    I, q = area_norm_data
    benchmark(vectorized_intensity_normalization, q, I, "area")


# ---------------------------------------------------------------------------
# Benchmarks: batch G2 normalization
# ---------------------------------------------------------------------------


@pytest.mark.benchmark(group="batch_g2_norm")
def test_batch_g2_norm_loop(benchmark, batch_norm_data):
    """Baseline: per-item loop with individual np.max dispatches."""
    benchmark(_batch_norm_loop, batch_norm_data)


@pytest.mark.benchmark(group="batch_g2_norm")
def test_batch_g2_norm_vectorized(benchmark, batch_norm_data):
    """Optimized: single np.stack + batched max/divide kernel."""
    from xpcsviewer.module.g2mod import batch_g2_normalization

    benchmark(batch_g2_normalization, batch_norm_data, "max")


# ---------------------------------------------------------------------------
# Correctness checks (not benchmarks) — ensure optimized == reference
# ---------------------------------------------------------------------------


def test_q_binning_1d_correctness(binning_1d_data):
    from xpcsviewer.module.saxs1d import vectorized_q_binning

    q, I, q_min, q_max, num_bins = binning_1d_data
    bc_ref, bi_ref, cnt_ref = _q_binning_loop(q, I, q_min, q_max, num_bins)
    bc_opt, bi_opt, cnt_opt = vectorized_q_binning(q, I, q_min, q_max, num_bins)
    np.testing.assert_array_equal(cnt_ref, cnt_opt)
    np.testing.assert_allclose(bi_ref, bi_opt, rtol=1e-12)


def test_q_binning_2d_correctness(binning_2d_data):
    from xpcsviewer.module.saxs1d import vectorized_q_binning

    q, I, q_min, q_max, num_bins = binning_2d_data
    bc_ref, bi_ref, cnt_ref = _q_binning_loop(q, I, q_min, q_max, num_bins)
    bc_opt, bi_opt, cnt_opt = vectorized_q_binning(q, I, q_min, q_max, num_bins)
    np.testing.assert_array_equal(cnt_ref, cnt_opt)
    np.testing.assert_allclose(bi_ref, bi_opt, rtol=1e-12)


def test_area_norm_correctness(area_norm_data):
    from xpcsviewer.module.saxs1d import vectorized_intensity_normalization

    I, q = area_norm_data
    ref = _area_norm_loop(I, q)
    opt = vectorized_intensity_normalization(q, I, "area")
    np.testing.assert_allclose(ref, opt, rtol=1e-12)


def test_batch_g2_norm_correctness(batch_norm_data):
    from xpcsviewer.module.g2mod import batch_g2_normalization

    ref = _batch_norm_loop(batch_norm_data)
    opt = batch_g2_normalization(batch_norm_data, "max")
    for r, o in zip(ref, opt):
        np.testing.assert_allclose(r, o, rtol=1e-12)
