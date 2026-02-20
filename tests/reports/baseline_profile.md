# Baseline Performance Profile Report

**Date:** 2026-02-19
**Platform:** macOS 26.3 (arm64, Apple Silicon)
**Python:** 3.13.9, JAX (CPU-only via `JAX_PLATFORMS=cpu`)
**Benchmark tool:** pytest-benchmark 5.2.3 + tracemalloc + cProfile

---

## Executive Summary

Six computational hot paths were profiled with synthetic data on CPU-only JAX.
The three largest bottlenecks are:

| Rank | Hot Path | Warm Mean (ms) | Cold (ms) | Category |
|------|----------|---------------:|----------:|----------|
| 1 | NLSQ fitting (500 pts) | **557** | 1936 | Solver-bound |
| 2 | SAXS q-binning (262k pts) | **52** | ~54 | Python-loop-bound |
| 3 | C2 clean nan_to_num (500x500) | **7.9** | ~9 | Array copy / percentile |

NLSQ dominates wall time by an order of magnitude due to multi-start optimization
(5 starting points, each running `nlsq.fit` with TRF solver). The cold-start penalty
is 3.5x (JIT compilation of JAX-traced model functions).

---

## 1. Q-map Computation (`xpcsviewer/simplemask/qmap.py`)

### Timing

| Metric | 512x512 | 2048x2048 |
|--------|--------:|----------:|
| Cold (ms) | 125.9 | ~850 (est.) |
| Warm mean (ms) | 4.97 | ~60 (est.) |
| Warm std (ms) | 1.96 | - |
| Benchmark mean (us) | 4615 | - |

### Memory

| Metric | Value |
|--------|------:|
| Peak allocation | 12.02 MB |
| Theoretical minimum (7 arrays x 512x512 x 8B) | ~14 MB |

### cProfile Hotspots

| Function | cumtime (ms) | % of total |
|----------|-------------:|----------:|
| `_compute_transmission_qmap_backend` | 273 | 100% |
| JAX `cache_miss` / `_python_pjit_helper` | 242 | 89% |
| JAX `arange` (dtype conversion) | 143 | 52% |
| `lower_sharding_computation` | 127 | 47% |

### Analysis

- **Cold path is JIT-dominated:** 89% of cold time is JAX tracing + lowering.
  The JIT-compiled `_transmission_qmap_core` runs fast once cached (~5ms for 512x512).
- **`lru_cache(maxsize=4)` on `_compute_transmission_qmap_cached` is effective** for
  repeated calls with identical geometry. Cache miss forces full backend recomputation.
- **Array dtype conversion overhead:** `jnp.asarray()` calls on `k0`, `v`, `h`, `pix_dim`,
  `det_dist` account for 52% of cold time due to tracing through `convert_element_type`.
- **Optimization opportunity:** Pre-convert scalar parameters to JAX arrays outside the
  JIT boundary to avoid repeated tracing. The `ensure_numpy()` I/O boundary conversion
  at the end adds ~2ms for 7 arrays.

---

## 2. Two-time Correlation C2 Cleaning (`xpcsviewer/module/twotime.py`)

### Timing

| Method | Benchmark mean (us) | Notes |
|--------|-------------------:|-------|
| nan_to_num | 7886 | Dominant cost: percentile |
| interpolate | 6306 | Gaussian filter faster than expected |

### Memory

| Metric | Value |
|--------|------:|
| Peak allocation (nan_to_num) | 5.22 MB |
| Theoretical (500x500 x 8B x 3 copies) | ~6 MB |

### cProfile Hotspots

| Function | cumtime (ms) | % of total |
|----------|-------------:|----------:|
| `clean_c2_for_visualization` | 9 | 100% |
| `ndarray.partition` (via percentile) | 6 | 67% |
| `np.percentile` (99.9th, 0.1th) | 5 | 56% |
| `np.median` | 2 | 22% |
| `np.nan_to_num` | 1 | 11% |

### Analysis

- **`np.percentile` is the bottleneck** (67% of time) due to array partitioning.
  Two percentile calls (99.9 and 0.1) each require a full array copy + partial sort.
- The `isfinite()` mask creation + boolean indexing `c2[finite_mask]` copies 250k floats
  to create `finite_values`, then partitions that array twice for percentiles.
- **Optimization opportunity:** Replace percentile-based replacement values with simpler
  statistics (e.g., max/min of finite values, or a single `np.nanpercentile` call on the
  original array to avoid the intermediate copy).

---

## 3. G2 Vectorized Operations (`xpcsviewer/module/g2mod.py`)

### Timing

| Operation | Benchmark mean (us) | Data shape |
|-----------|-------------------:|-----------|
| baseline_correction | 13.7 | (200, 50) |
| batch_normalization (5 datasets) | 69.9 | 5 x (200, 50) |
| ensemble_statistics | 1045 | 5 x (200, 50) |
| interpolation (JAX vmap) | 630 | (200, 50) -> (150, 50) |

### Cold vs Warm (interpolation)

| Metric | Value |
|--------|------:|
| Cold (ms) | 340.0 |
| Warm mean (ms) | 0.56 |
| Cold/warm ratio | **607x** |

### Memory

| Operation | Peak (MB) |
|-----------|----------:|
| ensemble_statistics | 1.67 |

### cProfile Hotspots (ensemble_statistics)

| Function | cumtime (ms) | % |
|----------|-------------:|-:|
| `compute_g2_ensemble_statistics` | 1.0 | 100% |
| `np.median` (array partition) | 1.0 | 85% |
| `np.matmul` (batched corrcoef) | <0.1 | <5% |

### Analysis

- **`baseline_correction` is already optimal** (single broadcast subtract, 14us).
- **`batch_normalization` has a Python loop** over `g2_data_list` but each iteration
  is vectorized NumPy. With 5 datasets, loop overhead is minimal.
- **`ensemble_statistics` bottleneck is `np.median`** (array partition sort). The
  batched correlation matrix computation via `np.matmul` is efficient.
- **Interpolation cold-start is extreme (607x)** due to JAX tracing `interpax.interp1d`
  + `jax.vmap`. Once JIT-cached, it runs in 0.56ms. This is expected JAX behavior.
- **Optimization opportunity:** If median is not strictly required, replacing it with
  mean (already computed) would eliminate the dominant bottleneck. For interpolation,
  consider AOT compilation or persistent JIT cache.

---

## 4. SAXS 1D Processing (`xpcsviewer/module/saxs1d.py`)

### Timing

| Operation | Benchmark mean (us) | Data size |
|-----------|-------------------:|----------|
| q-binning (1k pts, 100 bins) | 392 | 1,000 |
| q-binning (262k pts, 200 bins) | **52,174** | 262,144 |
| background subtraction | 14.9 | 1,000 |
| batch analysis (10 datasets) | 70.8 | 10 x 1,000 |

### cProfile Hotspots (q-binning, 262k points)

| Function | cumtime (ms) | % |
|----------|-------------:|-:|
| `vectorized_q_binning` | 57 | 100% |
| `vectorized_q_binning` body (Python loop) | 43 | 75% |
| `np.searchsorted` (via digitize) | 12 | 21% |
| `np.mean` (200 calls, per-bin) | 1 | 2% |

### Analysis

- **Critical bottleneck: Python `for b in range(num_bins)` loop** in `vectorized_q_binning`
  (lines 466-477). Despite the function name containing "vectorized", the per-bin mean
  computation uses a Python loop with boolean mask indexing per bin.
- The loop body `np.mean(valid_intensities[valid_indices == b])` creates a boolean mask
  per bin (262k comparisons x 200 bins = 52M comparisons total), then indexes + reduces.
- **Optimization opportunity (HIGH PRIORITY):** Replace the Python loop with
  `np.bincount(valid_indices, weights=valid_intensities)` / `np.bincount(valid_indices)`
  for a single-pass O(N) computation. Expected speedup: 20-50x for large detectors.
  The existing `vectorized_binning` in `test_saxs_baseline.py` already demonstrates this
  pattern and is known to be equivalent.
- **Background subtraction is fast** (15us) and already fully vectorized.
- **Batch analysis** is loop-bound but lightweight (71us for 10 datasets).

---

## 5. NLSQ Fitting (`xpcsviewer/fitting/nlsq.py`)

### Timing

| Metric | Value |
|--------|------:|
| Cold (ms) | 1935.9 |
| Warm mean (ms) | 524.2 |
| Warm std (ms) | 17.4 |
| Benchmark mean (us) | **557,162** |
| Cold/warm ratio | 3.7x |

### Memory

| Metric | Value |
|--------|------:|
| Peak allocation | 1.08 MB |

### cProfile Hotspots

| Function | cumtime (ms) | % |
|----------|-------------:|-:|
| `nlsq_optimize` (via `@log_timing`) | 2596 | 100% |
| `nlsq.fit` -> `curve_fit` | 2469 | 95% |
| `_run_multistart_optimization` | 2462 | 95% |
| `multi_start.fit` (5 starts) | 2447 | 94% |
| JAX `cache_miss` / tracing | 2412 | 93% |
| `least_squares` (TRF solver) | ~1139 | 44% |

### Analysis

- **Multi-start is the dominant cost:** NLSQ's "robust" preset generates 5 Latin
  Hypercube starting points, each running a full TRF optimization (6-9 iterations each).
  Total: 5 curve fits x ~100ms each (warm) = ~500ms.
- **Cold-start JIT penalty:** First call traces through JAX operations in the model
  function and TRF solver, adding ~1.4s. Subsequent calls benefit from JIT cache.
- **The first LHS start is ~1.5s** (includes JIT tracing), subsequent starts are ~60ms each.
- **Optimization opportunity:**
  - Use "fast" preset (1 start) when initial guess is good (saves 4x).
  - Pre-JIT the model function before entering the fitting loop.
  - Consider `workflow='auto'` (new API) instead of deprecated `preset='robust'`.
  - Memory is very low (1.08 MB) -- not memory-bound.

---

## 6. HDF5 I/O / FFT Cache (`xpcsviewer/xpcs_file.py`)

### Timing

| Operation | Benchmark mean (us) | Data size |
|-----------|-------------------:|----------|
| FFT (10k frames) | 145 | 10,000 |
| FFT (100k frames) | 1,152 | 100,000 |
| SAXS log10 (512x512) | 1,120 | 262,144 |

### Memory

| Operation | Peak (MB) |
|-----------|----------:|
| FFT (10k frames) | 0.52 |
| SAXS log (512x512) | 4.25 |

### Analysis

- **FFT is fast and well-optimized** by NumPy's FFTPACK/MKL. 145us for 10k frames
  is near-optimal. 100k frames scales linearly to ~1.15ms.
- **SAXS log computation** (`np.where(saxs > 0, np.log10(saxs), 0)`) allocates ~4MB
  for a 512x512 float64 array (3 intermediate arrays: comparison, log result, where output).
- The `_compute_int_t_fft_cached` method uses the unified memory manager cache, so
  repeated calls hit cache. The cache itself is not profiled here (requires HDF5 file).
- **`_load_saxs_data_batch`** is heavily I/O bound (HDF5 disk reads). The extensive
  memory management / prediction code adds Python overhead before the actual read.
  Without a real HDF5 file, this cannot be profiled with synthetic data.
- **Optimization opportunity:** The SAXS log computation could use `np.log10(saxs, out=result)`
  with pre-allocated output to reduce intermediate copies. For FFT, consider using
  `np.fft.rfft` (real-input FFT) instead of `np.fft.fft` for 2x memory savings.

---

## Summary: Top Optimization Targets

| Priority | Target | Current | Expected After | Technique |
|----------|--------|--------:|---------------:|-----------|
| P0 | SAXS q-binning loop (262k pts) | 52ms | 2-5ms | Replace Python loop with `np.bincount` |
| P0 | NLSQ multi-start overhead | 524ms | ~130ms | Use "fast" preset or pre-JIT model |
| P1 | C2 percentile computation | 7.9ms | 2-3ms | Single `nanpercentile` or max/min |
| P1 | Q-map JIT cold start | 126ms | <50ms | Pre-convert scalars to JAX arrays |
| P2 | G2 ensemble median | 1.0ms | 0.2ms | Replace with mean (already computed) |
| P2 | G2 interpolation cold start | 340ms | N/A | AOT compile or persistent cache |
| P3 | SAXS log intermediate arrays | 4.25MB | 2MB | In-place `np.log10` |
| P3 | FFT memory (rfft optimization) | 0.52MB | 0.26MB | Use `np.fft.rfft` |

---

## Benchmark Data Files

- **Benchmark script:** `tests/benchmarks/performance/test_hotpath_baseline.py` (29 tests)
- **cProfile script:** `tests/benchmarks/performance/_cprofile_hotpaths.py`
- **This report:** `tests/reports/baseline_profile.md`

### Reproducing Results

```bash
# Run pytest-benchmark timing
JAX_PLATFORMS=cpu uv run pytest tests/benchmarks/performance/test_hotpath_baseline.py \
  --benchmark-only --benchmark-columns=mean,stddev,min,max,rounds --benchmark-sort=mean \
  -m "not slow" --no-cov

# Run cold/warm and memory tests
JAX_PLATFORMS=cpu uv run pytest tests/benchmarks/performance/test_hotpath_baseline.py \
  -v -s --benchmark-disable -k "cold_warm or memory" --no-cov

# Run cProfile analysis
JAX_PLATFORMS=cpu uv run python tests/benchmarks/performance/_cprofile_hotpaths.py
```
