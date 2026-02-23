"""Comprehensive baseline benchmarks for all 6 computational hot paths.

Covers:
1. Q-map computation (JIT-cached)
2. Two-time correlation C2 cleaning/visualization
3. G2 vectorized operations (baseline correction, normalization, ensemble stats, interpolation)
4. SAXS 1D processing (q-binning, background subtraction, batch analysis)
5. NLSQ fitting
6. HDF5 I/O caching (FFT computation)

All benchmarks use synthetic data to avoid HDF5 file dependencies.
"""

from __future__ import annotations

import gc
import os
import time
import tracemalloc

import numpy as np
import pytest

# Force CPU-only JAX
os.environ.setdefault("JAX_PLATFORMS", "cpu")

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def qmap_geometry() -> dict:
    """Detector geometry for Q-map computation."""
    return {
        "energy": 10.0,
        "center": (256.0, 256.0),
        "shape": (512, 512),
        "pix_dim": 0.075,
        "det_dist": 5000.0,
    }


@pytest.fixture
def qmap_geometry_large() -> dict:
    """Large detector geometry for stress testing."""
    return {
        "energy": 10.0,
        "center": (1024.0, 1024.0),
        "shape": (2048, 2048),
        "pix_dim": 0.075,
        "det_dist": 5000.0,
    }


@pytest.fixture
def c2_matrix() -> dict:
    """Synthetic C2 correlation matrix with some NaN/inf."""
    rng = np.random.default_rng(42)
    size = 500
    base = rng.random((size, size), dtype=np.float64)
    c2 = (base + base.T) / 2
    c2 += np.eye(size) * 0.1
    # Inject ~1% NaN and ~0.5% inf
    nan_mask = rng.random((size, size)) < 0.01
    inf_mask = rng.random((size, size)) < 0.005
    c2[nan_mask] = np.nan
    c2[inf_mask] = np.inf
    return {"c2": c2, "c2_clean": np.nan_to_num(c2), "size": size}


@pytest.fixture
def g2_data() -> dict:
    """Synthetic G2 correlation data."""
    rng = np.random.default_rng(42)
    n_tau = 200
    n_q = 50
    n_batch = 5

    tau = np.logspace(-6, 2, n_tau)
    datasets = []
    for _ in range(n_batch):
        g2 = np.zeros((n_tau, n_q))
        for j in range(n_q):
            tau_c = 10 ** rng.uniform(-3, 1)
            beta = rng.uniform(0.1, 0.5)
            g2[:, j] = 1.0 + beta * np.exp(-tau / tau_c) + rng.normal(0, 0.01, n_tau)
        datasets.append(g2)

    baselines = 1.0 + rng.uniform(-0.02, 0.02, n_q)

    return {
        "tau": tau,
        "g2_single": datasets[0],
        "g2_list": datasets,
        "baselines": baselines,
        "n_tau": n_tau,
        "n_q": n_q,
        "n_batch": n_batch,
    }


@pytest.fixture
def saxs_data() -> dict:
    """Synthetic SAXS 1D data."""
    rng = np.random.default_rng(42)
    n_points = 1000

    q = np.linspace(0.01, 0.5, n_points)
    intensity = 1000 * np.exp(-q * 20) + rng.normal(0, 5, n_points)
    intensity_err = np.abs(rng.normal(5, 1, n_points))
    bg_intensity = 50 * np.exp(-q * 10) + rng.normal(0, 2, n_points)
    bg_err = np.abs(rng.normal(2, 0.5, n_points))

    return {
        "q": q,
        "intensity": intensity,
        "intensity_err": intensity_err,
        "bg_q": q,
        "bg_intensity": bg_intensity,
        "bg_err": bg_err,
        "n_points": n_points,
    }


@pytest.fixture
def fitting_data() -> dict:
    """Synthetic G2 curve for fitting."""
    rng = np.random.default_rng(42)
    n_points = 500

    tau = np.logspace(-4, 2, n_points)
    true_params = {"tau": 1.0, "baseline": 1.0, "contrast": 0.3}
    noise = rng.normal(0, 0.01, n_points)
    g2 = (
        true_params["baseline"]
        + true_params["contrast"] * np.exp(-2 * tau / true_params["tau"])
        + noise
    )
    g2_err = np.full(n_points, 0.01)

    return {
        "tau": tau,
        "g2": g2,
        "g2_err": g2_err,
        "true_params": true_params,
        "n_points": n_points,
    }


@pytest.fixture
def fft_data() -> dict:
    """Synthetic intensity vs time data for FFT."""
    rng = np.random.default_rng(42)
    n_frames = 10000
    t = np.arange(n_frames, dtype=np.float64)
    # Simulate intensity with periodic oscillation + noise
    intensity = 1000 + 50 * np.sin(2 * np.pi * t / 100) + rng.normal(0, 10, n_frames)
    return {"intensity": intensity, "n_frames": n_frames}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def measure_memory(func, *args, **kwargs):
    """Measure peak memory allocation of a function call."""
    gc.collect()
    tracemalloc.start()
    result = func(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / (1024 * 1024)  # Convert to MB


def measure_cold_warm(func, *args, warmup_runs=3, timed_runs=10, **kwargs):
    """Measure cold (first call) and warm (subsequent calls) wall times."""
    gc.collect()

    # Cold run
    t0 = time.perf_counter_ns()
    func(*args, **kwargs)
    cold_ns = time.perf_counter_ns() - t0

    # Warmup
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    # Warm runs
    warm_times = []
    for _ in range(timed_runs):
        t0 = time.perf_counter_ns()
        func(*args, **kwargs)
        warm_times.append(time.perf_counter_ns() - t0)

    return {
        "cold_ms": cold_ns / 1e6,
        "warm_mean_ms": np.mean(warm_times) / 1e6,
        "warm_std_ms": np.std(warm_times) / 1e6,
        "warm_min_ms": np.min(warm_times) / 1e6,
        "warm_max_ms": np.max(warm_times) / 1e6,
    }


# ===========================================================================
# 1. Q-MAP COMPUTATION
# ===========================================================================


class TestQmapBaseline:
    """Baseline benchmarks for Q-map computation (hot path #1)."""

    def test_qmap_transmission_correctness(self, qmap_geometry: dict) -> None:
        """Verify transmission Q-map produces valid output."""
        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        g = qmap_geometry
        qmap, qmap_unit = compute_transmission_qmap(
            g["energy"], g["center"], g["shape"], g["pix_dim"], g["det_dist"]
        )
        assert "q" in qmap
        assert qmap["q"].shape == g["shape"]
        assert np.all(np.isfinite(qmap["q"]))
        assert "Å" in qmap_unit["q"] or "A" in qmap_unit["q"]

    def test_qmap_transmission_timing(self, qmap_geometry: dict, benchmark) -> None:
        """Benchmark transmission Q-map (512x512)."""
        g = qmap_geometry

        def run():
            # Force cache miss by varying center slightly each time
            from xpcsviewer.simplemask.qmap import _compute_transmission_qmap_backend

            return _compute_transmission_qmap_backend(
                g["energy"], g["center"], g["shape"], g["pix_dim"], g["det_dist"]
            )

        result = benchmark(run)
        assert result is not None

    def test_qmap_transmission_cold_warm(self, qmap_geometry: dict) -> None:
        """Measure cold vs warm timing for Q-map (includes JIT compilation)."""
        from xpcsviewer.simplemask.qmap import _compute_transmission_qmap_backend

        g = qmap_geometry
        timings = measure_cold_warm(
            _compute_transmission_qmap_backend,
            g["energy"],
            g["center"],
            g["shape"],
            g["pix_dim"],
            g["det_dist"],
        )
        # Cold includes JIT compilation; warm should be significantly faster
        print(f"\n  Q-map 512x512 cold: {timings['cold_ms']:.1f}ms")
        print(
            f"  Q-map 512x512 warm: {timings['warm_mean_ms']:.2f}ms +/- {timings['warm_std_ms']:.2f}ms"
        )

    def test_qmap_transmission_memory(self, qmap_geometry: dict) -> None:
        """Measure peak memory for Q-map computation."""
        from xpcsviewer.simplemask.qmap import _compute_transmission_qmap_backend

        g = qmap_geometry
        _, peak_mb = measure_memory(
            _compute_transmission_qmap_backend,
            g["energy"],
            g["center"],
            g["shape"],
            g["pix_dim"],
            g["det_dist"],
        )
        print(f"\n  Q-map 512x512 peak memory: {peak_mb:.2f} MB")
        # 512x512 * 7 arrays * 8 bytes = ~14 MB theoretical minimum
        assert peak_mb < 200, f"Memory usage too high: {peak_mb:.1f} MB"

    @pytest.mark.slow
    def test_qmap_large_detector(self, qmap_geometry_large: dict) -> None:
        """Stress test: Q-map on 2048x2048 detector."""
        from xpcsviewer.simplemask.qmap import _compute_transmission_qmap_backend

        g = qmap_geometry_large
        timings = measure_cold_warm(
            _compute_transmission_qmap_backend,
            g["energy"],
            g["center"],
            g["shape"],
            g["pix_dim"],
            g["det_dist"],
            warmup_runs=1,
            timed_runs=3,
        )
        print(f"\n  Q-map 2048x2048 cold: {timings['cold_ms']:.1f}ms")
        print(f"  Q-map 2048x2048 warm: {timings['warm_mean_ms']:.1f}ms")


# ===========================================================================
# 2. TWO-TIME CORRELATION
# ===========================================================================


class TestTwotimeBaseline:
    """Baseline benchmarks for two-time correlation cleaning (hot path #2)."""

    def test_c2_clean_nan_to_num_correctness(self, c2_matrix: dict) -> None:
        """Verify nan_to_num cleaning removes all NaN/inf."""
        from xpcsviewer.module.twotime import clean_c2_for_visualization

        result = clean_c2_for_visualization(c2_matrix["c2"], method="nan_to_num")
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_c2_clean_nan_to_num_timing(self, c2_matrix: dict, benchmark) -> None:
        """Benchmark C2 cleaning with nan_to_num (500x500)."""
        from xpcsviewer.module.twotime import clean_c2_for_visualization

        c2 = c2_matrix["c2"]

        def run():
            return clean_c2_for_visualization(c2, method="nan_to_num")

        result = benchmark(run)
        assert result is not None

    def test_c2_clean_interpolate_timing(self, c2_matrix: dict, benchmark) -> None:
        """Benchmark C2 cleaning with interpolation (500x500)."""
        from xpcsviewer.module.twotime import clean_c2_for_visualization

        c2 = c2_matrix["c2"]

        def run():
            return clean_c2_for_visualization(c2, method="interpolate")

        result = benchmark(run)
        assert result is not None

    def test_c2_clean_memory(self, c2_matrix: dict) -> None:
        """Measure peak memory for C2 cleaning."""
        from xpcsviewer.module.twotime import clean_c2_for_visualization

        c2 = c2_matrix["c2"]
        _, peak_mb = measure_memory(clean_c2_for_visualization, c2, method="nan_to_num")
        print(f"\n  C2 clean 500x500 peak memory: {peak_mb:.2f} MB")
        # 500x500 * 8 bytes * ~3 copies = ~6 MB theoretical
        assert peak_mb < 50


# ===========================================================================
# 3. G2 VECTORIZED OPERATIONS
# ===========================================================================


class TestG2VectorizedBaseline:
    """Baseline benchmarks for G2 vectorized ops (hot path #3)."""

    def test_baseline_correction_timing(self, g2_data: dict, benchmark) -> None:
        """Benchmark vectorized baseline correction."""
        from xpcsviewer.module.g2mod import vectorized_g2_baseline_correction

        g2 = g2_data["g2_single"]
        baselines = g2_data["baselines"]

        def run():
            return vectorized_g2_baseline_correction(g2, baselines)

        result = benchmark(run)
        assert result is not None

    def test_batch_normalization_timing(self, g2_data: dict, benchmark) -> None:
        """Benchmark batch G2 normalization."""
        from xpcsviewer.module.g2mod import batch_g2_normalization

        g2_list = g2_data["g2_list"]

        def run():
            return batch_g2_normalization(g2_list, method="max")

        result = benchmark(run)
        assert result is not None

    def test_ensemble_statistics_timing(self, g2_data: dict, benchmark) -> None:
        """Benchmark ensemble statistics computation (includes batched corrcoef)."""
        from xpcsviewer.module.g2mod import compute_g2_ensemble_statistics

        g2_list = g2_data["g2_list"]

        def run():
            return compute_g2_ensemble_statistics(g2_list)

        result = benchmark(run)
        assert result is not None

    def test_ensemble_statistics_memory(self, g2_data: dict) -> None:
        """Measure peak memory for ensemble statistics."""
        from xpcsviewer.module.g2mod import compute_g2_ensemble_statistics

        g2_list = g2_data["g2_list"]
        _, peak_mb = measure_memory(compute_g2_ensemble_statistics, g2_list)
        print(f"\n  G2 ensemble stats peak memory: {peak_mb:.2f} MB")

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_g2_interpolation_jax_timing(self, g2_data: dict, benchmark) -> None:
        """Benchmark JAX vmap+interpax interpolation."""
        from xpcsviewer.module.g2mod import vectorized_g2_interpolation

        tau = g2_data["tau"]
        g2 = g2_data["g2_single"]
        target_tau = np.logspace(-5, 1.5, 150)

        def run():
            return vectorized_g2_interpolation(tau, g2, target_tau)

        result = benchmark(run)
        assert result is not None

    def test_g2_interpolation_cold_warm(self, g2_data: dict) -> None:
        """Measure cold vs warm for G2 interpolation (JIT compile impact)."""
        from xpcsviewer.module.g2mod import vectorized_g2_interpolation

        tau = g2_data["tau"]
        g2 = g2_data["g2_single"]
        target_tau = np.logspace(-5, 1.5, 150)

        timings = measure_cold_warm(
            vectorized_g2_interpolation,
            tau,
            g2,
            target_tau,
            warmup_runs=2,
            timed_runs=5,
        )
        print(f"\n  G2 interp cold: {timings['cold_ms']:.1f}ms")
        print(
            f"  G2 interp warm: {timings['warm_mean_ms']:.2f}ms +/- {timings['warm_std_ms']:.2f}ms"
        )


# ===========================================================================
# 4. SAXS PROCESSING
# ===========================================================================


class TestSaxsBaseline:
    """Baseline benchmarks for SAXS 1D processing (hot path #4)."""

    def test_q_binning_correctness(self, saxs_data: dict) -> None:
        """Verify vectorized q-binning produces valid output."""
        from xpcsviewer.module.saxs1d import vectorized_q_binning

        d = saxs_data
        q_bins, I_bins, counts = vectorized_q_binning(
            d["q"], d["intensity"], 0.01, 0.5, 100
        )
        assert len(q_bins) == 100
        assert np.all(np.isfinite(I_bins))
        assert np.all(counts >= 0)

    def test_q_binning_timing(self, saxs_data: dict, benchmark) -> None:
        """Benchmark vectorized q-binning (1000 points, 100 bins)."""
        from xpcsviewer.module.saxs1d import vectorized_q_binning

        d = saxs_data

        def run():
            return vectorized_q_binning(d["q"], d["intensity"], 0.01, 0.5, 100)

        result = benchmark(run)
        assert result is not None

    def test_background_subtraction_timing(self, saxs_data: dict, benchmark) -> None:
        """Benchmark vectorized background subtraction."""
        from xpcsviewer.module.saxs1d import vectorized_background_subtraction

        d = saxs_data
        fg = (d["q"], d["intensity"], d["intensity_err"])
        bg = (d["bg_q"], d["bg_intensity"], d["bg_err"])

        def run():
            return vectorized_background_subtraction(fg, bg, weight=1.0)

        result = benchmark(run)
        assert result is not None

    def test_batch_analysis_timing(self, saxs_data: dict, benchmark) -> None:
        """Benchmark batch SAXS analysis with normalize+trim operations."""
        from xpcsviewer.module.saxs1d import batch_saxs_analysis

        d = saxs_data
        data_list = [(d["q"], d["intensity"])] * 10
        ops = [
            {"type": "normalize", "method": "max"},
            {"type": "trim", "q_range": (0.02, 0.4)},
        ]

        def run():
            return batch_saxs_analysis(data_list, ops)

        result = benchmark(run)
        assert result is not None

    def test_q_binning_large_detector(self, benchmark) -> None:
        """Stress test: q-binning on large flattened detector (512x512)."""
        from xpcsviewer.module.saxs1d import vectorized_q_binning

        rng = np.random.default_rng(42)
        n = 512 * 512
        q = rng.random(n) * 0.5
        intensity = rng.random(n) * 1000

        def run():
            return vectorized_q_binning(q, intensity, 0.01, 0.5, 200)

        result = benchmark(run)
        assert result is not None


# ===========================================================================
# 5. NLSQ FITTING
# ===========================================================================


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
class TestNLSQBaseline:
    """Baseline benchmarks for NLSQ fitting (hot path #5)."""

    def test_nlsq_single_exp_correctness(self, fitting_data: dict) -> None:
        """Verify NLSQ fitting converges to reasonable parameters."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        d = fitting_data
        result = nlsq_optimize(
            single_exp_func,
            d["tau"],
            d["g2"],
            d["g2_err"],
            p0={"tau": 0.5, "baseline": 1.0, "contrast": 0.2},
            bounds={
                "tau": (0.01, 100.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )
        assert result.converged or not result.is_fallback
        assert np.isfinite(result.chi_squared)

    def test_nlsq_single_exp_timing(self, fitting_data: dict, benchmark) -> None:
        """Benchmark NLSQ single-exp fitting (500 points)."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        d = fitting_data

        def run():
            return nlsq_optimize(
                single_exp_func,
                d["tau"],
                d["g2"],
                d["g2_err"],
                p0={"tau": 0.5, "baseline": 1.0, "contrast": 0.2},
                bounds={
                    "tau": (0.01, 100.0),
                    "baseline": (0.5, 1.5),
                    "contrast": (0.01, 1.0),
                },
            )

        result = benchmark(run)
        assert result is not None

    def test_nlsq_cold_warm(self, fitting_data: dict) -> None:
        """Measure cold vs warm for NLSQ (JIT impact)."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        d = fitting_data

        def run():
            return nlsq_optimize(
                single_exp_func,
                d["tau"],
                d["g2"],
                d["g2_err"],
                p0={"tau": 0.5, "baseline": 1.0, "contrast": 0.2},
                bounds={
                    "tau": (0.01, 100.0),
                    "baseline": (0.5, 1.5),
                    "contrast": (0.01, 1.0),
                },
            )

        timings = measure_cold_warm(run, warmup_runs=2, timed_runs=5)
        print(f"\n  NLSQ single_exp cold: {timings['cold_ms']:.1f}ms")
        print(
            f"  NLSQ single_exp warm: {timings['warm_mean_ms']:.1f}ms +/- {timings['warm_std_ms']:.1f}ms"
        )

    def test_nlsq_memory(self, fitting_data: dict) -> None:
        """Measure peak memory for NLSQ fitting."""
        from xpcsviewer.fitting.models import single_exp_func
        from xpcsviewer.fitting.nlsq import nlsq_optimize

        d = fitting_data
        _, peak_mb = measure_memory(
            nlsq_optimize,
            single_exp_func,
            d["tau"],
            d["g2"],
            d["g2_err"],
            p0={"tau": 0.5, "baseline": 1.0, "contrast": 0.2},
            bounds={
                "tau": (0.01, 100.0),
                "baseline": (0.5, 1.5),
                "contrast": (0.01, 1.0),
            },
        )
        print(f"\n  NLSQ single_exp peak memory: {peak_mb:.2f} MB")


# ===========================================================================
# 6. HDF5 I/O CACHING (FFT computation)
# ===========================================================================


class TestFFTCacheBaseline:
    """Baseline benchmarks for FFT-based computation (hot path #6)."""

    def test_fft_computation_timing(self, fft_data: dict, benchmark) -> None:
        """Benchmark FFT of intensity vs time (10k frames)."""
        intensity = fft_data["intensity"]

        def run():
            fft_values = np.fft.fft(intensity)
            fft_magnitudes = np.abs(fft_values)
            n = len(intensity)
            freqs = np.fft.fftfreq(n, 1.0)
            pos_mask = freqs > 0
            return freqs[pos_mask], fft_magnitudes[pos_mask]

        result = benchmark(run)
        assert result is not None

    def test_fft_memory(self, fft_data: dict) -> None:
        """Measure peak memory for FFT computation."""
        intensity = fft_data["intensity"]

        def run():
            fft_values = np.fft.fft(intensity)
            fft_magnitudes = np.abs(fft_values)
            n = len(intensity)
            freqs = np.fft.fftfreq(n, 1.0)
            pos_mask = freqs > 0
            return freqs[pos_mask], fft_magnitudes[pos_mask]

        _, peak_mb = measure_memory(run)
        print(f"\n  FFT 10k frames peak memory: {peak_mb:.4f} MB")

    def test_fft_large_timing(self, benchmark) -> None:
        """Stress test: FFT on 100k frames."""
        rng = np.random.default_rng(42)
        intensity = rng.random(100_000) * 1000

        def run():
            fft_values = np.fft.fft(intensity)
            return np.abs(fft_values)

        result = benchmark(run)
        assert result is not None

    def test_saxs_log_computation_timing(self, benchmark) -> None:
        """Benchmark SAXS log computation (common I/O post-processing)."""
        rng = np.random.default_rng(42)
        saxs_2d = rng.random((512, 512), dtype=np.float64) * 1000

        def run():
            with np.errstate(divide="ignore", invalid="ignore"):
                result = np.where(saxs_2d > 0, np.log10(saxs_2d), 0)
            return result

        result = benchmark(run)
        assert result is not None

    def test_saxs_log_memory(self) -> None:
        """Measure peak memory for SAXS log computation."""
        rng = np.random.default_rng(42)
        saxs_2d = rng.random((512, 512), dtype=np.float64) * 1000

        def run():
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(saxs_2d > 0, np.log10(saxs_2d), 0)

        _, peak_mb = measure_memory(run)
        print(f"\n  SAXS log 512x512 peak memory: {peak_mb:.2f} MB")
