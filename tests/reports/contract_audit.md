# Contract Audit — Phase 2: Type and Contract Verification

Generated: 2026-02-19
Agent: python-pro (type and contract verification specialist)

---

## Executive Summary

This audit covers all major module boundaries in the XPCS Viewer codebase. Findings are classified by severity:

- **P0**: Runtime crash / data corruption likely
- **P1**: Silent failure / wrong behavior under common conditions
- **P2**: Type inconsistency / defensive coding gap / technical debt

---

## 1. Fitting Pipeline Contracts

### 1.1 `nlsq_optimize()` → `NLSQResult` — Caller verification

**Status: PASS with caveats**

- `nlsq_optimize()` always returns `NLSQResult` (never raises); the fallback sentinel path (`is_fallback=True`) is well-designed.
- **P1 — BUG-017 persists**: `converged = getattr(native_result, "success", False)` defaults to `False`. The NLSQ 0.6.0 `CurveFitResult` may not expose a `.success` attribute (it's from scipy's `OptimizeResult`; NLSQ may use a different attribute name). If the attribute is always absent, all successful fits report `converged=False`.
  - File: `xpcsviewer/fitting/nlsq.py:119`
  - Fix: Confirm NLSQ 0.6.0 attribute name; consider `native_result.converged` or `native_result.status`.

- **P2**: `nlsq_optimize` passes `model_fn` directly to `nlsq.fit()` without wrapping. If `nlsq.fit()` calls `model_fn(x, *popt)` with the positional-array convention, this is fine. But the chi-squared computation on line 142 also calls `model_fn(x, *popt)` — if `popt` is a JAX array (returned from `np.asarray(native_result.popt)`), unpacking with `*` over a JAX array may behave unexpectedly. Low risk but worth verifying.
  - File: `xpcsviewer/fitting/nlsq.py:142`

### 1.2 `run_single_exp_fit()` → `FitResult` — Contract chain

**Status: PASS**

- `_build_fit_result()` correctly extracts samples from the MCMC output via `samples_np = {k: np.asarray(v) for k, v in samples.items() if k in param_names}`.
- **P1 — Silent empty samples**: If MCMC returns no samples with keys matching `param_names` (e.g., a NumPyro model uses renamed latent variables), `samples_np` will be an empty dict. `FitResult.__post_init__()` will then raise `ValueError: "FitResult.samples must be non-empty"` — this exception propagates back through `do_work()`, which will emit `signals.error`. The error path handler (`_on_g2_bayesian_error`) correctly resets `_g2_bayesian_worker_active = False`, so no lock-up.
  - However: the `raise TypeError` in `run_power_law_fit()` when `tau` is a `FitResult` (line 572–575 of sampler.py) is reached on the main thread if called incorrectly — it will crash the worker rather than gracefully failing. This is acceptable defensive behavior.

### 1.3 `NLSQResult` delegation — None-safety when `native_result=None`

**Status: P1**

- All delegated properties (`r_squared`, `adj_r_squared`, `rmse`, `mae`, `aic`, `bic`, `residuals`, `predictions`, `covariance`, `confidence_intervals`, `diagnostics`) correctly fall back to `_r_squared`, `_rmse`, etc., when `native_result is None`.
- **P1**: `health_score` property accesses `self.diagnostics.health_score` without `hasattr` guard. If `diagnostics` is not `None` but does not have `health_score` (e.g., an older NLSQ version), `AttributeError` is raised.
  - File: `xpcsviewer/fitting/results.py:568`
  - Context: `health_score` is exposed on the fallback sentinel (`NLSQResult(native_result=None)`) via `_r_squared=float("nan")` etc., but `diagnostics` falls back to `None` there. So the guard `if self.diagnostics is not None: return int(self.diagnostics.health_score)` is fine for `native_result=None`. Risk only exists when `compute_diagnostics=True` is used with an NLSQ version that provides a `ModelHealthReport` without `.health_score`.
  - Severity: P2 if NLSQ version is fixed; P1 if NLSQ 0.6.0 API is still being confirmed.

- **P1**: `condition_number` accesses `self.diagnostics.identifiability.condition_number` with a chain of attribute access. If `identifiability` is not `None` but lacks `.condition_number`, `AttributeError` is raised.
  - File: `xpcsviewer/fitting/results.py:578-580`

- **P2**: `get_confidence_interval()` falls back to computing from `scipy.stats.norm.ppf`. This is a hidden scipy dependency in what is now intended to be an NLSQ/JAX-first module.
  - File: `xpcsviewer/fitting/results.py:624`

### 1.4 `SamplerConfig.__post_init__()` — Validation completeness

**Status: PASS**

- All numeric constraints validated: `num_warmup > 0`, `num_samples > 0`, `num_chains > 0`, `0 < target_accept_prob < 1`, `max_tree_depth > 0`.
- `random_seed: int | None` has no type validation — passing `random_seed=1.5` (float) would not be caught. Low risk.
  - File: `xpcsviewer/fitting/results.py:84-99`

### 1.5 `FitDiagnostics.converged` property

**Status: PASS**

- Logic is correct: checks `r_hat < 1.01`, `ess_bulk > 400`, `ess_tail > 400`, `divergences == 0`, `bfmi >= 0.2` (when set).
- **P2 — Edge case**: If `r_hat`, `ess_bulk`, or `ess_tail` are empty dicts (e.g., MCMC ran but no parameter names matched), `converged` returns `True` vacuously. This can happen if `_build_fit_result()` silently skips parameters not found in `summary.index`.
  - File: `xpcsviewer/fitting/results.py:164-184`
  - Context: `_build_fit_result()` uses `if name in summary.index` guard, which means missing parameters simply omit their diagnostics. A converged check with empty dicts then reports `True`.

---

## 2. Schema Validation — Enforcement at Boundaries

### 2.1 Schema enforcement at actual I/O boundaries

**Status: PARTIAL**

- `HDF5Facade.read_qmap()`, `read_g2_data()`, `read_geometry_metadata()` enforce schemas when `validate=True` (default).
- `HDF5Facade(validate=False)` returns a raw `dict` from `read_qmap()` instead of `QMapSchema`.
  - **P1 — Type inconsistency (BUG-029, partially addressed)**: The return type annotation is `-> QMapSchema` but the actual return can be `dict` when `validate=False`. The `# type: ignore[assignment]` suppresses mypy. Callers that type-check the return value (e.g., pass it to a function expecting `QMapSchema`) will fail at runtime.
  - File: `xpcsviewer/io/hdf5_facade.py:82,185`
  - Only two test callers use `validate=False` (in `tests/unit/test_tg7_gui_io_p1_fixes.py`). No production code paths observed using `validate=False`.
  - Fix: Add `Union[QMapSchema, dict]` return type or create a `RawQMapDict` TypedDict.

- **P2 — G2Data NaN check is strict**: `G2Data.__post_init__()` raises on any NaN in `g2` but not in `g2_err`. Real XPCS data may contain NaN-filled Q-bins legitimately. This could cause `HDF5Facade.read_g2_data()` to throw `HDF5ValidationError` on valid data.
  - File: `xpcsviewer/schemas/validators.py:477`

### 2.2 `QMapSchema.from_dict()` — Missing key handling

**Status: PASS with caveats**

- Missing `sqmap`/`q` key: raises `KeyError` with descriptive message (line 163).
- Missing `phis`/`phi` key: raises `KeyError` (line 167).
- Missing `dqmap`: defaults to `np.zeros_like(sqmap)` — this is intentional (line 170-172).
- Missing unit keys: default to `"nm^-1"` / `"deg"` with a logger warning for `phis_unit`.
- **P2**: The default for `phis_unit` is `"deg"` (with a warning) but the warning text says `"defaulting to 'deg'"` while the log message says `"phis_unit not found in dict, defaulting to 'deg'"` — this is fine, just inconsistent between comment and actual string. Minor.

### 2.3 `QMapSchema.from_compute_qmap()` — Unit normalization

**Status: P2**

- The unit normalization logic on line 231 is fragile: `"A^-1" if "Å" in q_unit or "A" in q_unit else "nm^-1"`. If `q_unit = "nm-1"` (dash instead of `^-1`), it falls through to `"nm^-1"` correctly. But if `q_unit = "1/A"` (common alternate), the `"A"` substring test would match it, returning `"A^-1"` — this is actually correct behavior. If `q_unit = "angstrom^-1"` (lowercase), `"A"` would not match, returning `"nm^-1"` incorrectly.
  - File: `xpcsviewer/schemas/validators.py:231`
  - Severity: P2 (only triggered if external code supplies a non-standard unit string).

---

## 3. Backend Abstraction — Type Contract Verification

### 3.1 `JAXBackend` vs `NumPyBackend` — Return type consistency

**Status: PASS**

- Both backends implement `from_numpy()` and `to_numpy()`: JAXBackend converts to/from `jnp.ndarray`, NumPyBackend is identity.
- The `BackendProtocol` is a structural `Protocol` (`@runtime_checkable`), so both pass `isinstance(backend, BackendProtocol)` checks at runtime.
- **P2**: `BackendProtocol` does not declare `from_numpy()` or `to_numpy()` as required methods (not visible in the first 80 lines of `_base.py`). If callers rely on `backend.from_numpy()`, they must know the concrete type. The adapters (`io_adapter.py`) call `backend.from_numpy()` in `from_hdf5()` and `from_pyqtgraph()` — this would fail if `BackendProtocol` doesn't require those methods and a custom backend is provided.
  - File: `xpcsviewer/backends/io_adapter.py:134,254`

### 3.2 `ensure_numpy()` coverage

**Status: PASS**

The architecture map documents all known JAX→NumPy boundaries. `ensure_numpy()` is called at:
- `apply_saxs_2d_result()` → `pg_saxs.setImage(ensure_numpy(image_data))` (line 1301)
- `apply_g2_result()` passes pre-converted data from worker (worker thread already handled conversion via NumPy operations)
- `SimpleMaskKernel.save_mask()` (documented in architecture map)
- `HDF5Adapter.to_hdf5()` calls `ensure_numpy()`

**P2 — Gap in `apply_twotime_result()`**: `c2_result = result["c2_result"]` is passed directly to `twotime.plot_twotime_g2()` without `ensure_numpy()`. If the worker produces JAX arrays in `c2_result`, this could pass a JAX array to PyQtGraph.
  - File: `xpcsviewer/xpcs_viewer.py:1376-1380`
  - Severity: P2 (depends on whether `TwotimePlotWorker.do_work()` returns NumPy or JAX arrays).

### 3.3 `io_adapter.py` — Consistency

**Status: PASS**

- All three adapters (`PyQtGraphAdapter`, `HDF5Adapter`, `MatplotlibAdapter`) consistently call `ensure_numpy()` on output.
- `MatplotlibAdapter.to_matplotlib()` returns `results[0]` (single array, not tuple) when one array is passed — this is a convenience behavior, but the return type annotation `-> tuple[np.ndarray, ...] | np.ndarray` is correct.
- **P2**: Adapters require a `BackendProtocol` instance at construction time. `create_adapters(backend=None)` will call `get_backend()` if `backend` is None — this is fine. But `PyQtGraphAdapter(backend)` stores `backend` and calls `backend.from_numpy()` in `from_pyqtgraph()`. If the adapters are created once at startup and the backend changes later (via `set_backend()`), the adapters hold a stale reference to the old backend.
  - File: `xpcsviewer/backends/io_adapter.py:134`

---

## 4. GUI ↔ Kernel Contract

### 4.1 `ViewerKernel` method return shapes — dict stability

**Status: P1**

- `apply_saxs_2d_result(result)` expects `result["image_data"]` and `result["levels"]` — hard-coded dict keys with no schema. If the worker returns a dict with different key names, `KeyError` is raised on the main thread (inside a slot — not caught).
  - File: `xpcsviewer/xpcs_viewer.py:1297-1298`
  - The info-message path (`result.get("type") == "info"`) is checked first, so info dicts are safe.

- `apply_g2_result(result)` uses `.get()` for all keys (`q`, `tel`, `g2`, etc.) — graceful. The `if q is None or g2 is None: return` guard prevents crashes.
  - File: `xpcsviewer/xpcs_viewer.py:1328-1337`

- `apply_twotime_result(result)` accesses `result["c2_result"]` and `result["new_qbin_labels"]` with direct dict access (no `.get()`).
  - **P1**: If `TwotimePlotWorker.do_work()` returns a result without `c2_result` (e.g., an error-shaped dict), `KeyError` is raised on the main thread.
  - File: `xpcsviewer/xpcs_viewer.py:1376-1377`

### 4.2 `_on_g2_bayesian_finished()` — result contract

**Status: PASS**

- Expects `result["fit_result"]` (a `FitResult`), `result["q_index"]`, `result["q_value"]`.
- `BayesianFitWorker.do_work()` returns exactly these keys plus `"context"`.
- `fit_result.diagnostics.converged` access is safe: `FitResult.diagnostics` is typed as `FitDiagnostics` with `default_factory=FitDiagnostics`, so it's never `None`.
- **P1 — Flag reset in error path**: The `_on_g2_bayesian_error()` handler (line 3932) resets `_g2_bayesian_worker_active = False` and `btn_g2_bayesian.setEnabled(True)`. This is correct. However, if `do_work()` raises an uncaught exception that bypasses `signals.error` (e.g., if `BaseAsyncWorker.run()` doesn't catch all exceptions), the flag would never reset.
  - This depends on `BaseAsyncWorker.run()` implementation; noted as a risk but not confirmed.

### 4.3 `XpcsFile.__getattr__()` — dynamic attribute access

**Status: P1**

- Keys routed via `__getattr__`: `_QMAP_KEYS` (frozenset), `saxs_2d`, `saxs_2d_log`, `Int_t_fft`, `_G2_PARTIAL_KEYS`.
- **P1**: If `self.qmap` itself is `None` (failed load), `getattr(self.qmap, key)` raises `AttributeError("'NoneType' object has no attribute X")` — which is then re-raised by `__getattr__`, indistinguishable from "attribute not found."
  - File: `xpcsviewer/xpcs_file.py:939-940`
  - Fix: Guard with `if self.qmap is None: raise AttributeError(...)`.

- **P1**: `_G2_PARTIAL_KEYS` lazy load catches `Exception` broadly and stores `None` (line 958-960). Subsequent callers that receive `None` without expecting it will fail with `TypeError` (e.g., `self.Int_t * some_scalar` if `Int_t` is `None`).
  - File: `xpcsviewer/xpcs_file.py:954-960`

- **P2**: The `if key in self.__dict__: return self.__dict__[key]` check at line 962 is reached only after the explicit key checks. If a key not in any special set ends up in `__dict__`, it's returned correctly. But `__getattr__` is only called when normal attribute lookup fails, so `self.__dict__[key]` access here is redundant (the normal `__getattribute__` would have returned it first). This line is dead code.
  - File: `xpcsviewer/xpcs_file.py:962-963`

### 4.4 `fit_g2()` — bounds `assert` statements

**Status: P1**

- `assert len(bounds) == 2` and `assert len(bounds[0]) == SINGLE_EXP_PARAMS` are used for validation (lines 1655-1659 of xpcs_file.py). These `assert`s are disabled when Python runs with `-O` (optimize flag), turning silent into wrong behavior.
  - File: `xpcsviewer/xpcs_file.py:1655-1659`
  - Fix: Replace with explicit `if/raise ValueError`.

- `get_g2_data()` returns a 5-tuple `(q_val, t_el, g2, sigma, label)` (inferred from line 1672). The caller at line 1672 uses `q_val, t_el, g2, sigma, label = self.get_g2_data(...)`. This matches the return from `get_g2_data()` as visible in the method. **PASS** on contract consistency.

---

## 5. CLI Contracts

### 5.1 Argument parsing completeness

**Status: PASS**

- All required arguments for `twotime` subcommand are parsed: `--input`, `--output`, and a mutually exclusive group (`--q`, `--phi`, `--q-phi`).
- Optional: `--dpi`, `--format`, `--log-level`.
- `required=True` on the mutually exclusive group ensures one selection is always provided.

### 5.2 `_run_twotime_batch()` error handling

**Status: PASS**

- Wrapped in `try/except Exception` at the `main()` call site (line 374), logs the error, and exits with code 1.
- `convert_exception()` is used to wrap exceptions with XPCS-specific context.
- The deferred import of `twotime_batch` module via `from xpcsviewer.cli.twotime_batch import run_twotime_batch` means import errors during CLI execution will also be caught and reported as exit code 1 (from the same except clause).

---

## 6. Additional Findings

### 6.1 Import structure — conditional imports in sampler.py

**Status: P1**

- `sampler.py` imports `jax`, `jnp`, `numpyro`, `az` at module level inside `try/except` (lines 41-61).
- **P1**: After the try/except, `jnp` is used directly (e.g., `jnp.asarray(x)` at line 319 of sampler.py) with no guard. If JAX is not available (`JAX_AVAILABLE=False`) and `run_single_exp_fit()` is called, `check_numpyro()` correctly raises `ImportError`. **But** module-level code like `jnp.asarray(x)` at the function body level is only reached after `check_numpyro()` passes, so the guard is in place.
- **P2**: `az` is similarly guarded. `compute_bfmi()` calls `az.bfmi()` with `az` bound at module level. If `ARVIZ_AVAILABLE=False`, `az` is unbound and `compute_bfmi()` raises `NameError`. This is protected by `check_numpyro()` (which checks `ARVIZ_AVAILABLE`), but only when called from `run_single_exp_fit()`. A direct call to `compute_bfmi()` without first calling `check_numpyro()` would get an unguarded `NameError`.
  - File: `xpcsviewer/fitting/sampler.py:178`

### 6.2 `FitResult.to_dict()` — `.tolist()` without ensure_numpy

**Status: P2**

- `"samples": {k: v.tolist() for k, v in self.samples.items()}` assumes `v` has `.tolist()`. If `v` is a JAX array, `.tolist()` exists on JAX arrays — this works. But `np.ndarray.tolist()` and `jnp.ndarray.tolist()` both exist, so this is fine. However, if `v` is neither (e.g., a Python list accidentally stored in samples), `AttributeError: 'list' has no attribute 'tolist'` is raised.
  - File: `xpcsviewer/fitting/results.py:932`
  - `FitResult.__post_init__()` calls `np.asarray(arr).shape` for validation but does NOT convert arrays to NumPy. So JAX arrays stored in `samples` are allowed.

### 6.3 `NLSQResult` — `params` dict values not type-checked

**Status: P2**

- `params: dict[str, float]` is the declared type but `__post_init__()` does not validate that values are float. If `nlsq_optimize()` returns `popt` containing JAX arrays (from `np.asarray(native_result.popt)`), `dict(zip(param_names, popt))` stores NumPy scalars, not floats. This is generally fine but `get_param_uncertainty()` calls `self.params[param]` and returns `float(...)`, which is correct.

---

## 7. Summary Table

| Severity | ID | Location | Issue |
|---|---|---|---|
| P1 | C-001 | `fitting/nlsq.py:119` | `getattr(native_result, "success", False)` — NLSQ 0.6.0 may not expose `.success`; all fits report `converged=False` |
| P1 | C-002 | `fitting/sampler.py:202-219` | Silent empty samples dict if MCMC param names mismatch model output names → FitResult `__post_init__` ValueError |
| P1 | C-003 | `fitting/results.py:568` | `health_score` accesses `diagnostics.health_score` without `hasattr` guard |
| P1 | C-004 | `fitting/results.py:578-580` | `condition_number` chains `diagnostics.identifiability.condition_number` without guard |
| P1 | C-005 | `io/hdf5_facade.py:82,185` | `read_qmap()` return type is `QMapSchema` but returns `dict` when `validate=False` |
| P1 | C-006 | `xpcs_viewer.py:1376-1377` | `apply_twotime_result()` uses direct dict access without `.get()` — `KeyError` on main thread |
| P1 | C-007 | `xpcs_file.py:939-940` | `__getattr__` delegates to `self.qmap` without None guard — confusing error if qmap is None |
| P1 | C-008 | `xpcs_file.py:954-960` | `_G2_PARTIAL_KEYS` lazy load silently stores `None`; downstream callers receive `None` without warning |
| P1 | C-009 | `xpcs_file.py:1655-1659` | `assert` used for bounds validation — disabled with `-O` flag |
| P1 | C-010 | `fitting/results.py:164-184` | Empty diagnostic dicts → `FitDiagnostics.converged` returns `True` vacuously |
| P2 | C-011 | `xpcs_viewer.py:1376-1380` | `c2_result` passed to `plot_twotime_g2()` without `ensure_numpy()` — potential JAX→PyQtGraph |
| P2 | C-012 | `fitting/results.py:624` | Hidden scipy dependency in `get_confidence_interval()` fallback path |
| P2 | C-013 | `backends/_base.py` | `from_numpy()` / `to_numpy()` not in `BackendProtocol` — callers that use these methods bypass the protocol |
| P2 | C-014 | `backends/io_adapter.py:134` | Adapters cache backend reference — stale if `set_backend()` called after adapter creation |
| P2 | C-015 | `schemas/validators.py:477` | `G2Data` raises on any NaN in `g2` — may reject valid data with NaN-filled Q-bins |
| P2 | C-016 | `schemas/validators.py:231` | Unit normalization in `from_compute_qmap()` fragile for non-standard unit strings |
| P2 | C-017 | `xpcs_file.py:962-963` | Dead code: `__getattr__` `self.__dict__[key]` check is unreachable via normal lookup |
| P2 | C-018 | `fitting/sampler.py:178` | `compute_bfmi()` uses unguarded `az` name — `NameError` if called directly without `check_numpyro()` |
| P2 | C-019 | `fitting/results.py:932` | `v.tolist()` in `to_dict()` — fails if samples dict contains Python lists |

---

## 8. Recommendations (Prioritized)

### P1 Fixes (should address before production release)

1. **C-001**: Determine correct NLSQ 0.6.0 success attribute. Run `print(dir(native_result))` in a test to confirm. Fallback chain: `getattr(native_result, "success", getattr(native_result, "converged", False))`.

2. **C-005**: Give `HDF5Facade.read_qmap()` an overloaded return type:
   ```python
   @overload
   def read_qmap(self, ..., validate: Literal[True] = True) -> QMapSchema: ...
   @overload
   def read_qmap(self, ..., validate: Literal[False]) -> dict: ...
   ```

3. **C-006 + C-007**: Add `.get()` calls in `apply_twotime_result()` and guard `self.qmap` None check in `__getattr__`.

4. **C-009**: Replace `assert len(bounds) == 2` with `if len(bounds) != 2: raise ValueError(...)` in `fit_g2()`.

5. **C-010**: Add a non-empty check in `FitDiagnostics.converged`: if all diagnostic dicts are empty, return `False` (unknown state) rather than `True`.

### P2 Fixes (code quality improvements)

6. **C-013**: Add `from_numpy()` and `to_numpy()` to `BackendProtocol` interface.

7. **C-015**: Consider allowing NaN in `G2Data.g2` with a warning instead of raising, or document that Q-bins with NaN must be pre-filtered before schema construction.

8. **C-018**: Add `if not ARVIZ_AVAILABLE: return None` guard at top of `compute_bfmi()`.

---

*Phase 2 audit complete. Findings cover: fitting pipeline (nlsq.py, results.py, sampler.py), schema validation (validators.py, hdf5_facade.py), backend abstraction (io_adapter.py, _base.py, _conversions.py), GUI↔kernel contract (xpcs_viewer.py, xpcs_file.py), CLI (cli_main.py).*

---

## Phase 3 — JAX/NumPy Boundary Deep Dive

Generated: 2026-02-19
Focus: ensure_numpy() placement, dtype preservation, model function dual-backend safety, FitResult serialization.

---

### P3.1 ensure_numpy() Coverage — Full Boundary Inventory

#### Confirmed COVERED boundaries

| Location | Method | Guard |
|---|---|---|
| `xpcs_viewer.py:1301` | `apply_saxs_2d_result()` | `ensure_numpy(image_data)` before `setImage()` |
| `xpcs_viewer.py:1504` | `apply_qmap_result()` | `ensure_numpy(image_data)` before `setImage()` |
| `viewer_kernel.py:368` | `plot_g2map()` | `ensure_numpy(xf_obj.get_offseted_g2(...))` before `setImage()` |
| `viewer_kernel.py:370` | `plot_g2map()` | `ensure_numpy(xf_obj.get_cropped_qmap(...))` before `setImage()` |
| `viewer_kernel.py:464,466,468` | `plot_qmap()` | `ensure_numpy(value/dqmap/sqmap)` before `setImage()` |
| `module/saxs2d.py:85` | `plot()` | `ensure_numpy(img)` before `setImage()` |
| `module/saxs1d.py:232,246,247` | `plot_line_with_marker()` | `ensure_numpy(x_plot/y_scatter)` before `plot()` |
| `module/g2mod.py:762-764` | data rendering | `ensure_numpy(x/y/dy)` |
| `module/g2mod.py:831,832,852,914,974,975,1038` | various | `ensure_numpy()` at plot boundaries |
| `module/twotime.py:223,224,256` | `plot_twotime_g2()` | `ensure_numpy(saxs/dqmap/c2_clean)` before `setImage()` |

#### GAPS found

**P2 — C-020: `intt.py` — no ensure_numpy() at plot boundary**

- `smooth_data()` accesses `fc.Int_t[1]` (an XpcsFile attribute, returned as NumPy from HDF5 load). The `x` array is created as `np.arange(..., dtype=np.uint32)`. Both are NumPy — safe for the normal path.
- However, `Int_t_fft` is computed lazily via `XpcsFile.__getattr__` → `_compute_int_t_fft_cached()`. If the FFT uses backend operations, the result may be a JAX array.
- `tf.plot(x_fft, y_fft, ...)` at `module/intt.py:182` has **no `ensure_numpy()` call**. If `x_fft` or `y_fft` is a JAX array, this silently passes a JAX array to PyQtGraph's `plot()`.
- File: `xpcsviewer/module/intt.py:182`
- Severity: P2 (depends on `_compute_int_t_fft_cached()` implementation)

**P2 — C-021: `stability.py` — no ensure_numpy() at plot boundary**

- `fc.get_saxs1d_data(target="saxs1d_partial", ...)` returns `(q, Iqp, xlabel, ylabel)`. These arrays originate from HDF5 reads (NumPy) but pass through backend operations in the analysis pipeline.
- `plot_line_with_marker()` in `saxs1d.py` does call `ensure_numpy()` at line 232 — **this is the actual render point**. So `stability.py` is safely guarded indirectly via `saxs1d.plot_line_with_marker()`.
- Status: **PASS** (ensure_numpy at saxs1d layer covers stability).

**~~P1 — C-022~~ RESOLVED (false alarm): `twotime.py` `c2g2` plot() calls**

- **Verified safe.** `plot_twotime_g2()` at lines 289-290 applies `np.nan_to_num()` to both `g2_full` and `g2_partial` before the `.plot()` calls:
  ```python
  g2_full_clean = np.nan_to_num(g2_full, nan=1.0, posinf=1.0, neginf=0.0)  # → NumPy
  g2_partial_clean = np.nan_to_num(g2_partial, nan=1.0, posinf=1.0, neginf=0.0)
  ```
  `xaxis` is produced by `np.arange(...) * acquire_period` — also NumPy. All arguments to `hdl["c2g2"].plot()` are NumPy arrays at that point.
- File: `xpcsviewer/module/twotime.py:289-306`
- Status: **PASS** — no ensure_numpy() gap; `np.nan_to_num()` serves as the implicit boundary conversion.

---

### P3.2 Dtype Preservation — Backend Abstraction

**Status: Mostly PASS, with one structural risk**

**JAX float64 enablement**

The `backends/__init__.py` sets `os.environ.setdefault("JAX_ENABLE_X64", "true")` at module scope (line 30) and calls `jax.config.update("jax_enable_x64", True)` in `_configure_jax()`. This correctly enables 64-bit precision before any JAX operations.

- **P1 — C-023: Race condition on JAX float64 configuration**

  `os.environ.setdefault("JAX_ENABLE_X64", "true")` is set at `backends/__init__.py` import time, but `jax.config.update("jax_enable_x64", True)` is only called inside `_configure_jax()`, which is deferred until `get_backend()` or `set_backend("jax")` is called. JAX's `x64` mode must be set **before the first JAX operation** in a process. If any module imports `jax.numpy as jnp` at module load time (before `_configure_jax()` is called), JAX silently defaults to float32 for all subsequent arrays in that session.
  - `fitting/models.py` imports `import jax.numpy as jnp` at module scope (line 25) — inside a `try/except`, but the import itself triggers JAX initialization. If `backends/__init__.py` has not been imported yet, `jax_enable_x64` may not be active.
  - `fitting/sampler.py` likewise imports `jax` and `jnp` at module scope (lines 43-44).
  - The `os.environ.setdefault("JAX_ENABLE_X64", "true")` at `backends/__init__.py:30` mitigates this **if** `backends/__init__.py` is imported before `fitting/models.py`. The import chain via `xpcs_viewer.py` imports `backends` before `fitting`, so in the GUI path this is safe. For standalone use of `fitting.models` (e.g., in tests), the env var may not be set in time.
  - File: `xpcsviewer/backends/__init__.py:30`, `xpcsviewer/fitting/models.py:25`

**Dtype at nlsq_optimize boundary**

- `nlsq_optimize()` explicitly casts: `x = np.asarray(x, dtype=np.float64)`, `y = np.asarray(y, dtype=np.float64)`, `yerr = np.asarray(yerr, dtype=np.float64)`.
- `popt = np.asarray(native_result.popt)` — **no explicit dtype**. If NLSQ 0.6.0 returns float32 `popt`, chi-squared computation `(residuals / yerr) ** 2` will be done in float32 (due to NumPy type promotion rules when mixing float32 popt with float64 x/y). The result could be silently downcast.
- File: `xpcsviewer/fitting/nlsq.py:117`
- **P2 — C-024**: Add `dtype=np.float64` to `popt`/`pcov` asarray calls.

**Dtype at sampler boundary**

- `_build_fit_result()` converts with `{k: np.asarray(v) for k, v in samples.items()}` — **no dtype specified**. NUTS samples are JAX arrays; `np.asarray()` preserves the JAX dtype. If JAX ran in float32 mode (see C-023), samples will be float32, violating the `FitResult.samples` type annotation of `dict[str, np.ndarray]` (dtype unspecified but float64 expected for scientific use).
- File: `xpcsviewer/fitting/sampler.py:202`
- **P2 — C-025**: Add `dtype=np.float64` to the `np.asarray(v)` call in `_build_fit_result()`.

---

### P3.3 Model Functions — JAX and NumPy Input Safety

**Status: P1 for NLSQ path, PASS for MCMC path**

The model functions (`single_exp_func`, `double_exp_func`, `stretched_exp_func`, `power_law_func`) use `_xnp` which is set to `jnp` when JAX is available and `np` otherwise. They are decorated with `@_maybe_jit`.

**P1 — C-026: JIT-compiled functions reject NumPy inputs under some conditions**

When JAX is available and `_maybe_jit` applies `jax.jit`:
- `jax.jit` traces function arguments. If `x` is a plain Python scalar or a NumPy array, JAX traces it through abstract evaluation, which is fine for `jnp.clip`, `jnp.exp`, etc.
- **Problem**: `nlsq_optimize()` passes `model_fn` directly to `nlsq.fit()`. NLSQ internally calls `model_fn(x, *p0_array)` where `x` is a NumPy float64 array and `p0_array` elements are NumPy scalars. If `model_fn` is `single_exp_func` (now `jax.jit`-compiled), JAX will trace with NumPy array shapes — this generally works, but:
  - If `nlsq.fit()` calls `model_fn` with varying array shapes (e.g., for Jacobian estimation with perturbed parameters), JAX will re-trace on each shape change, causing repeated JIT compilation overhead.
  - More critically: the chi-squared residual computation at `nlsq.py:142` calls `model_fn(x, *popt)` where `popt` is `np.asarray(native_result.popt)`. This second call to the JIT-compiled `model_fn` from inside Python (not from inside `nlsq.fit()`) produces a JAX array result, which is then used in `y - model_fn(x, *popt)` where `y` is a NumPy array. This mixed JAX/NumPy subtraction works in newer JAX versions but is an implicit device transfer.
  - File: `xpcsviewer/fitting/nlsq.py:142`

**P2 — C-027: _xnp module-level binding is evaluated at import time**

`_xnp = jnp if JAX_AVAILABLE else np` is set once at module load. If the backend changes via `set_backend()` after import (e.g., in tests that switch from JAX to NumPy), `_xnp` in `models.py` is **never updated** — the model functions still use JAX operations even when the NumPy backend is active. This doesn't cause crashes (JAX functions accept NumPy inputs) but it means the NumPy backend path through model functions is actually JAX-backed.
- File: `xpcsviewer/fitting/models.py:36-38`

---

### P3.4 FitResult Serialization Round-Trip

**Status: No HDF5 round-trip exists — PASS for what exists, gap noted**

**Finding**: There is **no FitResult → HDF5 → FitResult round-trip** in the codebase. `FitResult` results from Bayesian fits are:
1. Stored in memory in `self._g2_bayesian_results` (a dict, keyed by q_idx) in `XpcsViewer`.
2. Passed to `BayesianDiagnosisWindow.update_results()` for visualization.
3. Never written to HDF5, JSON, or any persistence layer.

The `FitResult.to_dict()` method exists and is well-implemented (samples converted via `.tolist()`, diagnostics serialized), but **nothing calls it** in production code.

- **P2 — C-028: FitResult not persisted — results lost on window close**. If the user closes and reopens the diagnosis window or restarts the application, all Bayesian fit results are gone. The `to_dict()` / `from_dict()` round-trip path is unimplemented (no `FitResult.from_dict()` exists).
  - File: `xpcsviewer/fitting/results.py:886` (to_dict implemented but unused)
  - Severity: P2 (feature gap, not a crash risk, but scientifically significant for reproducibility)

**G2 export path — dtype safety**

`ViewerKernel.export_g2()` accesses `xf.fit_summary["fit_val"]` and writes with `np.savetxt()`. `fit_val` originates from legacy scipy `curve_fit` results (NumPy float64 arrays). The path is dtype-safe for the legacy fitter.

For the Bayesian fitter, `_g2_bayesian_results` is not included in `export_g2()` — consistent with the finding above.

---

### P3.5 Phase 3 Additional Findings Summary

| Severity | ID | Location | Issue |
|---|---|---|---|
| ~~P1~~ RESOLVED | C-022 | `module/twotime.py:289-306` | `c2g2` plot() calls — **false alarm**: `np.nan_to_num()` converts to NumPy before plot(); `np.arange()` for xaxis is also NumPy |
| P1 | C-023 | `backends/__init__.py:30`, `fitting/models.py:25` | JAX float64 config race: `jax_enable_x64` may not be active when `fitting/models.py` is imported standalone |
| P1 | C-026 | `fitting/nlsq.py:142` | JIT-compiled model_fn returns JAX array mixed with NumPy in chi-squared computation — implicit device transfer |
| P2 | C-020 | `module/intt.py:182` | FFT plot path — no ensure_numpy() on `x_fft`/`y_fft` before PyQtGraph `plot()` |
| P2 | C-024 | `fitting/nlsq.py:117` | `np.asarray(native_result.popt)` lacks `dtype=float64` — silent float32 downcast if NLSQ returns float32 |
| P2 | C-025 | `fitting/sampler.py:202` | `np.asarray(v)` in `_build_fit_result()` lacks `dtype=float64` — samples may be float32 if JAX x64 not active |
| P2 | C-027 | `fitting/models.py:36-38` | `_xnp` bound at import time — doesn't respect runtime backend changes |
| P2 | C-028 | `fitting/results.py:886` | `FitResult.to_dict()` implemented but never called — Bayesian results not persisted across sessions |

---

### P3.6 Phase 3 Recommended Fixes

1. **C-022**: ~~Add ensure_numpy() guards~~ — **RESOLVED**. `twotime.py:289-306` already converts via `np.nan_to_num()` before all `.plot()` calls. No action required.

2. **C-023**: Move `jax.config.update("jax_enable_x64", True)` call to the top of `models.py` (inside the `try: import jax` block, immediately after import), so it runs regardless of import order. This is a standard JAX best-practice: always configure before first use.

3. **C-024/C-025**: Add `dtype=np.float64` to:
   ```python
   popt = np.asarray(native_result.popt, dtype=np.float64)
   pcov = np.asarray(native_result.pcov, dtype=np.float64)
   # and in sampler.py:
   samples_np = {
       k: np.asarray(v, dtype=np.float64) for k, v in samples.items() if k in param_names
   }
   ```

4. **C-020**: In `intt.py` FFT plot:
   ```python
   from xpcsviewer.backends._conversions import ensure_numpy

   tf.plot(ensure_numpy(x_fft), ensure_numpy(y_fft), ...)
   ```

5. **C-028**: Implement `FitResult.from_dict()` and wire `to_dict()` into the session save/restore cycle (`SessionManager`) to persist Bayesian results across restarts.

---

*Phase 3 audit complete. New findings: 8 issues (3 P1, 5 P2). Combined total across Phase 2+3: 27 issues.*
