# JAX Backend Audit Report

**Date:** 2026-02-19
**Scope:** `xpcsviewer/backends/`, `xpcsviewer/module/`, `xpcsviewer/simplemask/`, `xpcsviewer/fitting/`
**Status:** READ-ONLY analysis — no source modifications

---

## Executive Summary

The codebase has a solid JAX backend abstraction layer with JIT support already implemented in the most performance-critical paths (`qmap.py`, `utils.py`). However, there are several untapped opportunities in `module/saxs1d.py` and `module/g2mod.py`. The `vectorized_g2_interpolation()` function correctly uses `jax.vmap` with `interpax`. Host-device transfer patterns are largely correct (applied at I/O boundaries). The main risks are nested Python loops in Q-binning and an issue with `batch_g2_normalization` bypassing the backend abstraction.

---

## 1. JIT Compilation Audit

### 1.1 Functions With JIT — Confirmed

| File | Function / Scope | JIT Mechanism | Notes |
|------|-----------------|---------------|-------|
| `xpcsviewer/simplemask/qmap.py:139` | `_transmission_qmap_core` | `@jax.jit` inside `_get_transmission_qmap_jit()` | Dict-based cache avoids hashability issue |
| `xpcsviewer/simplemask/qmap.py:520` | `_reflection_qmap_core` | `@jax.jit` inside `_get_reflection_qmap_jit(orientation)` | Per-orientation cache key; orientation baked in |
| `xpcsviewer/simplemask/utils.py:118` | `_partition_linear_core` | `@jax.jit` inside `_get_partition_linear_jit()` | |
| `xpcsviewer/simplemask/utils.py:168` | `_partition_log_core` | `@jax.jit` inside `_get_partition_log_jit()` | |
| `xpcsviewer/simplemask/utils.py:215` | `_phi_transform_core` | `@jax.jit` inside `_get_phi_transform_jit()` | |
| `xpcsviewer/simplemask/calibration.py:79` | `loss_fn` gradient + eval | `jax.jit(jax.grad(...))` and `jax.jit(loss_fn)` | Two separate JIT calls for the same function |
| `xpcsviewer/simplemask/calibration.py:268` | `objective` gradient + eval | `jax.jit(jax.grad(...))` and `jax.jit(objective)` | Same pattern as above |
| `xpcsviewer/fitting/models.py:44` | `single_exp_func`, `double_exp_func`, `stretched_exp_func`, `power_law_func` | `@_maybe_jit` decorator applies `jax.jit` | Applied at module load time; good practice |

### 1.2 Functions Without JIT — Opportunities

| File | Function | Lines | Computation Type | JIT Feasibility | Priority |
|------|----------|-------|-----------------|-----------------|----------|
| `xpcsviewer/module/saxs1d.py` | `vectorized_q_binning()` | 430-481 | Q-space binning with nested Python loops | **Low** — inner loops are Python; would need restructuring to `jnp.digitize` + `jax.ops.segment_sum` | HIGH |
| `xpcsviewer/module/saxs1d.py` | `vectorized_intensity_normalization()` | 532-580 | Pure element-wise array ops | **High** — already vectorized with NumPy; wrapping with `backend.jit()` would give XLA fusion | MEDIUM |
| `xpcsviewer/module/g2mod.py` | `vectorized_g2_baseline_correction()` | 818-830 | Simple broadcast subtraction | **High** — trivial JIT candidate | LOW (already near-trivial) |
| `xpcsviewer/module/g2mod.py` | `batch_g2_normalization()` | 833-869 | Per-dataset normalization loop | **Medium** — outer Python loop over datasets is unavoidable; inner math could use `backend.jit()` | MEDIUM |
| `xpcsviewer/module/g2mod.py` | `compute_g2_ensemble_statistics()` | 872-919 | Batched matmul, stack, mean over large arrays | **High** — all ops are `np.*`; wrapping core computation with `backend.jit()` would benefit GPU | HIGH |
| `xpcsviewer/module/twotime_utils.py` | (not read — check separately) | — | — | Needs audit | — |

### 1.3 XLA Recompilation Risk — `static_argnums` Gaps

| File | Location | Issue | Risk |
|------|----------|-------|------|
| `xpcsviewer/simplemask/utils.py:118` | `_partition_linear_core` | `num_pts` is derived from `v_span.shape[0]-1` inside the JIT body — shape is dynamic but traced. `linspace` uses `num_pts` as a concrete int for length, computed from a traced array dimension. This is safe since shapes are static in JAX tracing. | LOW — shape tracing is correct |
| `xpcsviewer/simplemask/qmap.py:139` | `_transmission_qmap_core` | All args are JAX arrays (no Python scalars). No `static_argnums` needed. | SAFE |
| `xpcsviewer/fitting/models.py:44` | `_maybe_jit` on model funcs | Python scalar args (e.g., `tau`, `baseline`) passed to JIT functions. In NumPyro NUTS these are traced as JAX arrays, not Python scalars. | SAFE within NUTS context |
| `xpcsviewer/simplemask/calibration.py:79,268` | `jax.jit(jax.grad(loss_fn))` | `grad_fn` and `loss_fn_jit` are two separate JIT compilations of the same function body — causes two compilation traces unnecessarily. Use `jax.value_and_grad` + single JIT. | LOW performance overhead, MEDIUM code quality issue |

---

## 2. vmap Audit

### 2.1 vmap Usage — Confirmed

| File | Function | Lines | Vectorization | Notes |
|------|----------|-------|--------------|-------|
| `xpcsviewer/module/g2mod.py` | `vectorized_g2_interpolation()` | 960-1013 | `jax.vmap(_interp_single_q, in_axes=1, out_axes=1)` | Correctly vmaps over Q-axis (axis=1). Falls back to sequential Python loop for NumPy path (lines 1003-1012). |
| `xpcsviewer/utils/vectorized_roi.py` | (internal) | 338 | `jax.vmap(process_single_frame)` | Present; not part of primary audit scope. |

### 2.2 Sequential Loops — vmap Opportunities

| File | Function | Lines | Loop Pattern | vmap Feasibility | Priority |
|------|----------|-------|-------------|-----------------|----------|
| `xpcsviewer/module/saxs1d.py` | `vectorized_q_binning()` — 1D branch | 466-470 | `for b in range(num_bins): np.mean(...)` over bins | **Medium** — `jax.ops.segment_mean` or `jnp.bincount` + masked mean would replace; shape-static binning needed | HIGH |
| `xpcsviewer/module/saxs1d.py` | `vectorized_q_binning()` — 2D branch | 473-479 | `for b in range(num_bins): for phi_idx in range(num_phi): np.mean(...)` | **Medium** — double loop; `jax.ops.segment_mean` vectorized over phi axis | HIGH |
| `xpcsviewer/module/saxs1d.py` | `vectorized_intensity_normalization()` — "area" branch | 573-578 | `np.array([np.trapezoid(...) for i in ...])` | **Low** — `jnp.trapezoid` is not yet in interpax; use `jnp.trapz` (deprecated) or sum-based approximation | MEDIUM |
| `xpcsviewer/module/g2mod.py` | `batch_g2_normalization()` | 844-866 | `for g2_data in g2_data_list: np.max(...)` | **Low** — outer loop is over files (different shapes possible); not vmappable unless padded | LOW |
| `xpcsviewer/module/g2mod.py` | `optimize_g2_error_propagation()` | 922-957 | `for op in operations:` with branching | **Low** — operation type branching makes direct vmap difficult; `jax.lax.switch` could replace | LOW |
| `xpcsviewer/module/saxs1d.py` | `batch_saxs_analysis()` | 583-625 | `for q, intensity in data_list:` | **Low** — datasets may have different shapes; padding + vmap possible but complex | LOW |
| `xpcsviewer/module/saxs1d.py` | `optimize_roi_extraction()` | 628-686 | `for i, roi in enumerate(roi_definitions):` | **Medium** — rectangular ROI branch: `jax.vmap` over ROI batch, circular ROI: precompute distance mask per ROI then vmap | MEDIUM |

---

## 3. Host-Device Transfer Audit (`ensure_numpy` Usage)

### 3.1 Correct I/O Boundary Usage — Confirmed

All `ensure_numpy()` calls in `module/` and `simplemask/` are correctly placed **at** the final I/O boundary (PyQtGraph `setImage`, `plot`, scatter calls). No mid-computation transfers found in the primary hot paths.

| File | Call Site | Context | Assessment |
|------|-----------|---------|------------|
| `module/g2mod.py:761-763` | `pg_plot_one_g2()` — x, y, dy | Right before `pg.ErrorBarItem` / `ax.plot` | CORRECT — I/O boundary |
| `module/g2mod.py:996` | `vectorized_g2_interpolation()` | After vmap, before return | CORRECT — function output boundary |
| `module/saxs1d.py:232,246,247` | `plot_line_with_marker()` | `plot_item.plot()` / `ScatterPlotItem` | CORRECT — I/O boundary |
| `module/saxs2d.py:85` | `pg_hdl.setImage()` | PyQtGraph image display | CORRECT — I/O boundary |
| `module/twotime.py:219,220,252` | `hdl["saxs"].setImage()` etc. | PyQtGraph image display | CORRECT — I/O boundary |
| `simplemask/qmap.py:425-697` | Return dict construction | After JIT computation, before dict return | CORRECT — output boundary |
| `simplemask/utils.py:354-362` | HDF5 write path | Before `np.uint32` cast and HDF5 write | CORRECT — storage boundary |
| `simplemask/simplemask_window.py:677-892` | `setImage()` calls | PyQtGraph display | CORRECT — I/O boundary |
| `simplemask/calibration.py:304` | Return from calibration | Before returning params | CORRECT — API boundary |

### 3.2 Potential Mid-Computation Transfer Issues

| File | Function | Lines | Issue | Priority |
|------|----------|-------|-------|----------|
| `module/g2mod.py` | `batch_g2_normalization()` | 844-869 | Uses raw `np.*` (NumPy) instead of `backend.*`, so when JAX is active, arrays passed in may be JAX arrays that trigger implicit device→host copies via `np.max`, `np.mean`, etc. | HIGH |
| `module/g2mod.py` | `compute_g2_ensemble_statistics()` | 872-919 | Uses `np.stack`, `np.mean`, `np.std`, `np.median`, `np.matmul` directly. If called with JAX arrays (e.g. after `vectorized_g2_interpolation()`), this forces host copy. | HIGH |
| `module/g2mod.py` | `optimize_g2_error_propagation()` | 922-957 | Uses `np.abs`, `np.power`, `np.isfinite` directly. No `ensure_numpy` guard at entry — implicit copy if JAX arrays passed. | MEDIUM |
| `module/saxs1d.py` | `vectorized_q_binning()` | 430-481 | Uses `np.digitize`, `np.bincount`, `np.mean` throughout. JAX arrays not converted at entry. | MEDIUM |

---

## 4. XLA Recompilation Audit

### 4.1 Static vs. Dynamic Arguments

| File | Function | Issue | Risk Level |
|------|----------|-------|------------|
| `simplemask/calibration.py:79` | `grad_fn = jax.jit(jax.grad(loss_fn))` + `loss_fn_jit = jax.jit(loss_fn)` | Two separate `jax.jit` compilations of overlapping functions. The gradient function will be recompiled separately from the value function. Use `jax.value_and_grad` wrapped in a single `jax.jit` instead. | MEDIUM |
| `simplemask/calibration.py:268` | Same pattern as above | Same issue in second calibration path | MEDIUM |
| `simplemask/qmap.py` | `_JIT_CACHE` dict-based cache | Cache uses string keys (`"transmission_qmap_jit"`, `"reflection_qmap_jit_{orientation}"`). Orientation is baked into the compiled function. Different orientations produce separate compiled functions — this is correct. | SAFE |
| `simplemask/utils.py` | `_PARTITION_JIT_CACHE` | Same dict-based pattern. Functions have no Python-value arguments inside JIT body. `v_span` shape determines dynamic behavior but is always a concrete array. | SAFE |
| `fitting/models.py` | `_maybe_jit` on model functions | JIT applied at module load time. Functions receive JAX arrays during NUTS tracing — no Python scalar args that would cause retrace. | SAFE |

### 4.2 Shape-Induced Recompilation Risk

| File | Function | Risk |
|------|----------|------|
| `simplemask/qmap.py:139` | `_transmission_qmap_core(k0, v, h, ...)` | `v` and `h` vary by detector shape. Different detector configurations (different shapes) will cause XLA recompilation. Acceptable since cache is bounded at 32 entries. |
| `simplemask/utils.py:118` | `_partition_linear_core(mask_b, xmap_b, v_min, v_max, v_span)` | `xmap_b` is always full detector shape — stable. `v_span` length varies by `num_pts` — but only the first call compiles; subsequent calls with same `num_pts` reuse. Different `num_pts` → recompile. |

---

## 5. Backend Abstraction Compliance

### 5.1 Files Using `np.*` Directly Instead of `backend.*`

| File | Functions | Direct NumPy Usage | Issue |
|------|-----------|--------------------|-------|
| `module/g2mod.py` | `batch_g2_normalization()`, `compute_g2_ensemble_statistics()`, `optimize_g2_error_propagation()`, `vectorized_g2_baseline_correction()` | `np.max`, `np.mean`, `np.std`, `np.stack`, `np.matmul`, etc. | Not backend-agnostic; breaks when inputs are JAX arrays |
| `module/saxs1d.py` | `vectorized_q_binning()`, `vectorized_background_subtraction()`, `vectorized_intensity_normalization()`, `batch_saxs_analysis()`, `optimize_roi_extraction()` | All use `np.*` directly | Not backend-agnostic |
| `module/twotime.py` | `clean_c2_for_visualization()`, `calculate_safe_levels()` | `np.*` directly | Acceptable — these work on NumPy arrays at display time |
| `module/g2mod.py` | `pg_plot`, `pg_plot_stability`, `pg_plot_from_data` | `np.*` for plotting support | Acceptable — plotting code should use NumPy |
| `fitting/sampler.py` | All samplers | `np.asarray(x)` at entry, then `jnp.asarray(x)` for JAX ops | CORRECT pattern — converts to NumPy at API boundary, JAX internally |

---

## 6. Prioritized Recommendations

### Priority: HIGH

1. **`saxs1d.py:vectorized_q_binning()` — Replace Nested Python Loops** (lines 466-479)
   The double `for b in range(num_bins): for phi_idx in range(num_phi)` loop is the most glaring issue. For `num_bins=100` and `num_phi=8`, this is 800 Python iterations. Replace with `jnp.digitize` + `jax.ops.segment_sum` / `jax.ops.segment_mean`. This function is called `vectorized_q_binning` but has non-vectorized inner loops.
   **Estimated speedup:** 10-50x for large binning operations.

2. **`g2mod.py:batch_g2_normalization()` and `compute_g2_ensemble_statistics()` — Backend Compliance** (lines 833-919)
   These functions use raw `np.*` ops on arrays that may already be JAX arrays (e.g., output of `vectorized_g2_interpolation()`). This forces silent host transfers. Fix: either add `ensure_numpy()` at function entry (if NumPy semantics are intended) or migrate to `backend.*` calls.
   **Impact:** Prevents silent mid-pipeline device→host transfers.

3. **`compute_g2_ensemble_statistics()` — JIT Opportunity** (lines 872-919)
   This function already uses batched matmul (`np.matmul`), stack, and statistical reductions over potentially large 3D arrays. With `backend.jit()` wrapping the core computation, the XLA compiler can fuse these ops.
   **Estimated speedup:** 2-5x on GPU, 1.2-1.5x on CPU (XLA fusion).

### Priority: MEDIUM

4. **`simplemask/calibration.py:79,268` — Consolidate Dual JIT Calls**
   Replace `jax.jit(jax.grad(loss_fn))` + `jax.jit(loss_fn)` with:
   ```python
   value_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))
   ```
   This reduces compilation overhead from 2 JIT traces to 1.

5. **`saxs1d.py:optimize_roi_extraction()` — vmap for Rectangular ROIs** (lines 628-686)
   The loop over ROI definitions with rectangular ROIs can be batched: pre-compute bounding box indices as a stacked array and use `jax.vmap` over the ROI axis. For circular ROIs, precompute the distance mask stack.
   **Condition:** Only beneficial when `len(roi_definitions) > 4`.

6. **`saxs1d.py:vectorized_intensity_normalization()` — JIT Wrapping** (lines 532-580)
   Already uses vectorized NumPy ops. Add `backend.jit()` for the 2D normalization paths to enable XLA fusion.

### Priority: LOW

7. **`g2mod.py:vectorized_g2_baseline_correction()` — Trivially JIT-Able** (lines 818-830)
   Single broadcast subtraction. Already very fast; JIT overhead may exceed benefit for typical array sizes. Only JIT if called in a tight loop.

8. **`g2mod.py:batch_g2_normalization()` — vmap Infeasible**
   Datasets have different shapes; cannot directly vmap. Sequential loop is appropriate. Fix the NumPy compliance issue (Priority HIGH #2) instead.

9. **`module/stability.py` — No JAX Opportunity**
   Pure plotting delegation to `saxs1d.plot_line_with_marker`; no computation.

10. **`fitting/` — Well-Optimized**
    NLSQ model functions are JIT-compiled at import time via `@_maybe_jit`. NumPyro models are traced by NUTS directly. No additional JIT/vmap opportunities needed.

---

## 7. Summary Table

| File | Function | JIT Status | vmap Status | Backend Compliant | Priority |
|------|----------|-----------|------------|------------------|----------|
| `simplemask/qmap.py` | `_transmission_qmap_core` | JIT (dict cache) | N/A | Yes | — |
| `simplemask/qmap.py` | `_reflection_qmap_core` | JIT (dict cache) | N/A | Yes | — |
| `simplemask/utils.py` | `_partition_*_core` | JIT (dict cache) | N/A | Yes | — |
| `simplemask/calibration.py` | `loss_fn`, `objective` | JIT (2x redundant) | N/A | Yes | MEDIUM — consolidate |
| `fitting/models.py` | `*_func` functions | JIT (`@_maybe_jit`) | N/A | Yes | — |
| `module/g2mod.py` | `vectorized_g2_interpolation` | None | vmap (correct) | Partial | — |
| `module/g2mod.py` | `batch_g2_normalization` | None | Not feasible | **No** (np.*) | HIGH |
| `module/g2mod.py` | `compute_g2_ensemble_statistics` | None — opportunity | N/A | **No** (np.*) | HIGH |
| `module/g2mod.py` | `optimize_g2_error_propagation` | None | N/A | **No** (np.*) | MEDIUM |
| `module/g2mod.py` | `vectorized_g2_baseline_correction` | None — trivial | N/A | No (np.*) | LOW |
| `module/saxs1d.py` | `vectorized_q_binning` | None | Nested loops | **No** (np.*) | HIGH |
| `module/saxs1d.py` | `vectorized_intensity_normalization` | None — opportunity | Partial loop | **No** (np.*) | MEDIUM |
| `module/saxs1d.py` | `optimize_roi_extraction` | None | Loop — partial vmap | **No** (np.*) | MEDIUM |
| `module/saxs1d.py` | `batch_saxs_analysis` | None | Not feasible | **No** (np.*) | LOW |
| `module/twotime.py` | `clean_c2_for_visualization` | None | N/A | No (np.*) | LOW (display only) |
| `module/stability.py` | `plot` | N/A | N/A | N/A | — |
| `module/intt.py` | `smooth_data` | None | N/A | No (np.*) | LOW |

---

## 8. Architecture Observations

1. **The dict-based JIT cache pattern** in `qmap.py` and `utils.py` is the correct solution for JAX arrays not being hashable for `functools.lru_cache`. This pattern should be adopted whenever new JIT functions are needed.

2. **The `_PARTITION_JIT_CACHE_MAXSIZE=32` bound** prevents unbounded memory growth. However, both `_JIT_CACHE` in `qmap.py` and `_PARTITION_JIT_CACHE` in `utils.py` are module-level globals without thread safety. For multi-threaded contexts, a lock is needed.

3. **The NumPy fallback in `vectorized_g2_interpolation()`** (lines 1000-1012) uses a Python loop over Q-indices. This is the exact pattern that `jax.vmap` is designed to replace — and the JAX path (lines 976-996) correctly does so.

4. **`ensure_numpy()` is correctly applied only at true I/O boundaries** (PyQtGraph, HDF5). There are no detected cases of mid-computation `ensure_numpy()` followed by re-import to JAX, which would indicate a waste pattern.

5. **`compute_g2_ensemble_statistics()` contains a notable optimization**: the correlation matrix computation was already refactored from a per-Q Python loop to a batched `np.matmul` (the comment confirms this). However, it still uses `np.*` and would benefit from JIT compilation.
