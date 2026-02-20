# Algorithmic Bottleneck Analysis

**Date:** 2026-02-19
**Analyst:** bottleneck-hunter
**Baseline data source:** `tests/reports/baseline_profile.md`
**Micro-benchmarks:** `tests/benchmarks/performance/test_bottleneck_analysis.py`
**Test results:** 20/20 passing

---

## Methodology

Code paths were read directly from source and validated against timing data from the
baseline profiling report (Task #1). Each bottleneck is isolated with a self-contained
micro-benchmark that:
1. Verifies correctness of the proposed fix relative to the baseline
2. Measures the speedup ratio on realistic synthetic data
3. Characterises how the speedup scales with input size

---

## Correction: SAXS q-binning is already fixed in production

The baseline profile measured `loop_based_binning` (the baseline function *inside*
`test_saxs_baseline.py`) not the production code. The current
`vectorized_q_binning` in `xpcsviewer/module/saxs1d.py:429` already uses `np.bincount`
and `np.add.at` — the correct O(N) single-pass algorithm. **No fix needed here.**

---

## Top 3 Bottlenecks (by wall-time impact)

### Bottleneck #1 — NLSQ multi-start: `preset="robust"` runs 5 TRF solves per fit call

**File:** `xpcsviewer/fitting/nlsq.py:26` (`nlsq_optimize`)
**Baseline profile:** **557ms warm, 1.9s cold** — P0 priority
**Micro-benchmark:** `TestBottleneck5NLSQMultiStart`

#### Root Cause

```python
native_result = nlsq.fit(
    model_fn, x, y, ...,
    preset=preset,   # "robust" -> 5 Latin-Hypercube TRF starts
)
```

`preset="robust"` instructs the NLSQ library to generate 5 Latin Hypercube starting
points and run a full TRF solve for each. For a 3-parameter G2 model with 50-200 data
points, the objective is convex-ish and a single start from a physically-informed
initial guess converges reliably. The multi-start overhead is pure waste for this use case.

Cold-start: 93% of the 1.9s is JAX tracing inside the model function and TRF
solver -- the first LHS start pays the full JIT compilation cost, subsequent starts
benefit from cache.

#### Complexity

| Configuration | Cost per fit |
|---|---|
| `preset="robust"` (5 starts) | 5 x TRF_cost + 4 x JIT_cache_lookup |
| `preset="fast"` (1 start) | 1 x TRF_cost |
| Speedup (well-conditioned data) | ~5-7x |

#### Measured Speedup (scipy TRF stand-in, 100 data points, 3 params)

| Run | Multi-start (5) | Single-start | Speedup |
|-----|----:|----:|----:|
| N_starts=2 | 3.6ms | 1.2ms | 3.0x |
| N_starts=5 | 9.0ms | 1.2ms | **7.3x** |
| N_starts=10 | 17.8ms | 1.2ms | 14.8x |

Linear scaling confirms each start is an independent solve.

#### Proposed Fix

```python
# In xpcs_file.py fit_g2() calls and similar callers:
# Change: preset="robust"
# To:     preset="fast"   (when initial guess is physically informed)
# Or:     workflow="auto" (new NLSQ 0.6.0 API -- auto-selects starts based on
#                          problem conditioning)

# Pre-JIT the model function before entering the fitting loop:
import jax
model_jit = jax.jit(model_fn)  # trace once, reuse for all q-bins
```

**Expected savings: 350-430ms per fit call** (from 557ms to ~100-130ms).

---

### Bottleneck #2 — `compute_g2_ensemble_statistics`: `np.median` = 85% of runtime

**File:** `xpcsviewer/module/g2mod.py:887`
**Baseline profile:** 1.0ms; `np.median` = 85% (0.85ms)
**Micro-benchmark:** `TestBottleneck1G2Ensemble`

#### Root Cause

```python
stats = {
    "ensemble_mean":   np.mean(g2_stack, axis=0),    # O(B*T*Q)  -- fast
    "ensemble_std":    np.std(g2_stack, axis=0),      # O(B*T*Q)  -- fast
    "ensemble_median": np.median(g2_stack, axis=0),   # O(B*T*Q*log B) -- BOTTLENECK
    ...
}
```

`np.median` requires a **partial sort** of B values at every (T, Q) position. For
`shape=(B=20, T=200, Q=64)` that is 12,800 partial-sort calls. The sort is
O(B log B) per cell vs O(B) for mean/std. The baseline profile confirms `ndarray.partition`
(called by np.median) dominates.

Additionally, the list comprehension at the end:
```python
stats["temporal_correlation"] = [corr_batch[q] for q in range(num_q)]
```
creates Q separate Python objects from a contiguous 3-D array -- O(Q) heap allocations.

#### Measured Speedup

| Run | Baseline (ms) | Candidate (ms) | Speedup |
|-----|----:|----:|----:|
| B=20, T=200, Q=64 | 6-10 | 2-5 | **1.7-2.8x** |

#### Proposed Fix

```python
# Remove np.median from the hot path -- not used by any known caller.
# Callers that need median can call np.median(g2_stack, axis=0) explicitly.
# Return temporal_correlation as ndarray[Q, B, B] -- not a Python list.
stats["temporal_correlation"] = corr_batch   # [Q, B, B] 3-D ndarray
```

---

### Bottleneck #3 — `clean_c2_for_visualization`: 3 separate sort passes = 67% of runtime

**File:** `xpcsviewer/module/twotime.py:52-57`
**Baseline profile:** 7.9ms total; `np.percentile` = 67% (5.3ms)
**Micro-benchmark:** `TestBottleneck4C2Percentile`

#### Root Cause

```python
finite_values = c2[finite_mask]               # copies ~250k floats
pos_replacement = np.percentile(finite_values, 99.9)  # sort pass #1 -> 2.5ms
neg_replacement = np.percentile(finite_values, 0.1)   # sort pass #2 -> 2.5ms
nan_replacement = np.median(finite_values)             # sort pass #3 -> 0.3ms
```

Three separate `np.percentile` / `np.median` calls each trigger an independent
partial sort of the `finite_values` array (~250k elements for a 500x500 C2 matrix).
NumPy does not cache the sorted partition between calls.

Additionally, `c2[finite_mask]` creates an intermediate copy of all finite elements
before any statistics are computed.

#### Complexity

| Current | After Fix |
|---------|-----------|
| 3 partial sorts of ~250k elements | 1 partial sort (nanpercentile with 3 quantiles) |
| 1 intermediate array copy (finite_values) | 0 intermediate copies |

#### Measured Speedup (500x500 C2 matrix, 5% NaN/inf)

| Run | Baseline (ms) | Candidate (ms) | Speedup |
|-----|----:|----:|----:|
| nan_to_num path | 6.8 | 4.5 | **1.51x** |

#### Proposed Fix

```python
# Replace three separate sort passes with one nanpercentile call:
pcts = np.nanpercentile(c2, [0.1, 50.0, 99.9])
neg_replacement, nan_replacement, pos_replacement = (
    float(pcts[0]), float(pcts[1]), float(pcts[2])
)
return np.nan_to_num(
    c2, nan=nan_replacement, posinf=pos_replacement, neginf=neg_replacement
)
```

`np.nanpercentile` operates directly on the full matrix (no intermediate `finite_values`
copy) and computes all three quantiles in a single sort.

---

## Additional Findings

### `MaskAssemble.get_mask()` -- per-event memcpy in GUI hot path

**File:** `xpcsviewer/simplemask/area_mask.py:470`
**Impact:** GUI responsiveness during interactive drawing (not in baseline profile)
**Micro-benchmark:** `TestBottleneck2MaskHistory`

`get_mask()` always copies the current mask: `mask_record[ptr].copy()`. For a
2048x2048 bool mask this is 4MB of memcpy per call. In the SimpleMask GUI, this is
called on every mouse-drag redraw event.

**Measured speedup:** >230x for a read-only view path vs copy path.

**Fix:** Add `_get_mask_ref()` returning a read-only view for internal use. Keep
`get_mask()` as copy for backward-compatible external API.

### `batch_g2_normalization` -- Python loop over stackable data

**File:** `xpcsviewer/module/g2mod.py:832`
**Impact:** 1.4-1.7x speedup for B=10-50 datasets
**Micro-benchmark:** `TestBottleneck3BatchNormalization`

Python loop dispatches to NumPy once per dataset. Stacking all datasets into a
single 3-D array eliminates B separate kernel calls. Gain is batch-size dependent;
break-even at very large B (stack copy dominates).

---

## Summary Table

| Priority | Bottleneck | File | Current | Fix | Measured Speedup |
|----------|-----------|------|---------|-----|-----------------|
| P0 | NLSQ multi-start (5 TRF starts) | `nlsq.py:26` | 557ms/fit | `preset="fast"` or pre-JIT | **7.3x** |
| P1 | G2 ensemble median sort | `g2mod.py:887` | O(B*T*Q*log B) | Skip median in hot path | **1.7-2.8x** |
| P1 | C2 percentile: 3 sort passes | `twotime.py:52` | 7.9ms | `nanpercentile([0.1,50,99.9])` | **1.5x** |
| P2 | MaskAssemble `get_mask()` copies | `area_mask.py:470` | 4MB/event | Read-only view | **>230x** |
| P2 | `batch_g2_normalization` loop | `g2mod.py:832` | O(B) dispatches | `np.stack` vectorize | **1.4-1.7x** |

---

## Micro-Benchmark Results Summary

```
TestBottleneck1G2Ensemble (G2 median skip)       -- all tests PASSED  ~2x speedup
TestBottleneck2MaskHistory (mask view)           -- all tests PASSED  >230x speedup (view)
TestBottleneck3BatchNormalization (batch stack)  -- all tests PASSED  1.4-1.7x speedup
TestBottleneck4C2Percentile (3->1 sort pass)     -- all tests PASSED  1.5x speedup
TestBottleneck5NLSQMultiStart (5->1 TRF start)  -- all tests PASSED  7.3x speedup

Total: 20/20 passing
```

Run with:
```bash
uv run pytest tests/benchmarks/performance/test_bottleneck_analysis.py \
  -v -p no:benchmark -k "not timing" -s
```

---

## Recommendations for Downstream Tasks

### Task #4 (JAX/vectorization optimization)
- NLSQ: pre-JIT the model function with `jax.jit` before entering q-bin fitting loop
- G2 ensemble: replace `np.median` with mean; use `jax.vmap` over Q for temporal correlation
- C2 cleaning: replace 3-call percentile with `np.nanpercentile(c2, [0.1, 50, 99.9])`

### Task #5 (HDF5 I/O pipeline)
- `_load_saxs_data_batch` calls `MemoryMonitor.get_memory_usage()` (psutil syscall) multiple
  times before every HDF5 read. Cache the result with a 1-second TTL.

### Task #6 (Python patterns)
- `MaskAssemble.get_mask()` -- add `_get_mask_ref()` internal view
- `batch_g2_normalization` -- stack-based vectorization with B > 5 threshold
- `clean_c2_for_visualization` -- single `nanpercentile` call
