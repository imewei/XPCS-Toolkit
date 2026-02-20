# Numerical and JAX Audit Report

**Date:** 2026-02-19  
**Scope:** JAX-specific and numerical correctness issues across the xpcsviewer codebase  
**Auditor:** jax-pro (Phase 3)

---

## Executive Summary

10 findings across P0–P2 severity. The most critical issue (P0) is a JIT-tracing
correctness bug in the reflection Q-map: Python `if/elif` control flow that depends on
the traced `orientation` string is embedded inside `@jax.jit`, causing the compiler to
bake the orientation at first-trace time and silently return wrong Q-maps for any other
orientation. Four P1 issues cover NaN propagation, a converged-flag bug, model-function
unpacking with JAX arrays, and a warm-start parameter mismatch. Five P2 issues cover
minor efficiency and numerical hygiene concerns.

---

## P0 — Critical (Silent Correctness Failure)

### P0-001: Python control flow inside `@jax.jit` in reflection Q-map

**File:** `xpcsviewer/simplemask/qmap.py`, lines 334–378 (`_reflection_qmap_core`)  
**Root cause:**  
The inner function `_reflection_qmap_core` is decorated with `@jax.jit` and captures
`orientation` (a Python string) as a closed-over variable. Inside the compiled function
the `if/elif` branches selecting the orientation transform (`west`, `south`, `east`) are
treated as **Python-level control flow evaluated once at trace time**. After the first
JIT compilation for a given `orientation`, any call with a different orientation still
executes the cached XLA computation compiled for the original orientation. The Q-map
coordinate transformation is therefore silently wrong for every orientation except the
one used at first trace.

```python
@jax.jit
def _reflection_qmap_core(k0, v, h, pix_dim, det_dist, alpha_i_deg):
    ...
    if orientation == "west":      # BUG: captured Python constant, not traced value
        vg, hg = -hg, vg
    elif orientation == "south":
        vg, hg = -vg, -hg
    elif orientation == "east":
        vg, hg = hg, -vg
```

The cache key `f"reflection_qmap_jit_{orientation}"` does create one compiled variant
per orientation, so this bug only manifests if the _same_ Python process reuses an
existing variant for a different orientation string. Under the current call path from
`_compute_reflection_qmap_backend`, a new inner function is created and JIT-compiled
for each distinct `orientation` cache key, which **partially** mitigates the problem.
However, if `_JIT_CACHE` evicts the "west" entry due to the 32-entry cap (FIFO), and
the next call for "west" finds the "north" entry still cached under a **collision** or
under any future refactoring that reuses the key incorrectly, the wrong compiled XLA
kernel is dispatched. Additionally, the pattern is dangerous: it conflates Python
control flow with JAX tracing in a way that will silently break under any code path that
reuses the compiled function object across orientations.

**Proposed fix:**  
Use `jax.lax.switch` or `jax.lax.cond` for differentiable branching, or pass
orientation as a `static_argnums` integer index to `jax.jit`, so XLA compiles a
separate specialization per orientation value in a well-defined, traceable way:

```python
@functools.partial(jax.jit, static_argnums=(6,))
def _reflection_qmap_core(k0, v, h, pix_dim, det_dist, alpha_i_deg, orientation_idx):
    ...
    # Use lax.switch(orientation_idx, [north_fn, west_fn, south_fn, east_fn], vg, hg)
```

---

## P1 — High (Likely Runtime Failures or Incorrect Results)

### P1-001: `converged` always `False` due to missing `success` attribute

**File:** `xpcsviewer/fitting/nlsq.py`, line 107  
**Root cause:**  
```python
converged = getattr(native_result, "success", False)  # BUG-017: default False
```
The NLSQ 0.6.0 `CurveFitResult` does not expose a `.success` attribute (the NLSQ API
uses `.converged` or checks `.status`). The comment itself flags this as BUG-017 but
the fix was not applied. As a result `converged` is always `False` regardless of whether
the solver actually converged. Every downstream check of `NLSQResult.converged` silently
reports failure even for well-converged fits, causing the Bayesian warm-start to log an
erroneous "NLSQ warm-start may be unreliable" warning for every fit, and potentially
triggering fallback paths that skip valid results.

**Proposed fix:**
```python
converged = bool(getattr(native_result, "converged",
                         getattr(native_result, "success", False)))
```
Check `.converged` first; fall back to `.success`; finally default `False`.

---

### P1-002: NaN gradient from `jnp.clip(tau, 1e-30)` inside NumPyro model

**File:** `xpcsviewer/fitting/models.py`, line 79 (`single_exp_model`)  
**Root cause:**  
```python
tau = jnp.clip(tau, 1e-30)
```
`jnp.clip` is not differentiable at the boundary: `d/dtau clip(tau, eps) = 0` when
`tau <= eps`, which zero-kills the gradient for `tau`. NUTS uses the gradient of the
log-probability w.r.t. `tau` to propose new states. When the sampler proposes `tau`
near `1e-30`, the gradient is `0` (not NaN), which causes the leapfrog integrator to
stop updating `tau`, effectively trapping the chain at the lower bound. This is distinct
from a hard NaN but produces divergences and pathological R-hat.

Additionally, `jnp.clip` inside a `numpyro.sample` site means the clipping is applied
**after** the log-prior is evaluated, so the prior `LogNormal(0, 2)` still assigns
probability mass to `tau < 1e-30`. The sampled value is clamped post-hoc, making the
effective prior different from the declared prior. The proper fix is a `Constraints`-
based prior or a log-parameterization:

**Proposed fix:**  
Reparameterize: sample `log_tau ~ Normal(0, 2)` and compute `tau = jnp.exp(log_tau)`,
which guarantees positivity without gradient issues:
```python
log_tau = numpyro.sample("log_tau", dist.Normal(0.0, 2.0))
tau = jnp.exp(log_tau)
```
The same issue exists in `double_exp_model`, `stretched_exp_model`, and `power_law_model`.

---

### P1-003: `model_fn(x, *popt)` unpacking JAX arrays inside NLSQ residual computation

**File:** `xpcsviewer/fitting/nlsq.py`, lines 121–127  
**Root cause:**  
```python
residuals = y - model_fn(x, *popt)
```
`popt = np.asarray(native_result.popt)` is a NumPy array. The `*popt` unpacking
iterates over NumPy array rows, passing each element as a Python float. However,
`single_exp_func` is decorated with `@_maybe_jit`, which means it is JIT-compiled. A
JIT-compiled JAX function called with `(numpy_array, float, float, float)` will trigger
a host-device transfer and retracing on every call with different scalar values
(JAX traces separate compilations for different concrete Python values unless
`static_argnums` is used). In practice this causes a fresh XLA compilation for every
Q-bin's residual evaluation, which is very slow and defeats the purpose of JIT.

Additionally, if `model_fn` is `double_exp_func` or `stretched_exp_func`, the parameter
order expected by the JIT-compiled function may not match the NLSQ parameter ordering,
producing silently wrong residuals and chi-squared values.

**Proposed fix:**  
Compute residuals with explicit NumPy (not JIT-compiled) evaluation, or use a dedicated
non-JIT wrapper for the residual path:
```python
residuals = y - np.asarray(model_fn.__wrapped__(x, *popt))  # bypass JIT
```
Or use a `model_fn_numpy` variant that avoids JIT decoration.

---

### P1-004: MCMC warm-start init_params parameter name mismatch

**File:** `xpcsviewer/fitting/sampler.py`, lines 130–148 (`_run_mcmc`)  
**Root cause:**  
`nlsq_init` dict keys are `{"tau": ..., "baseline": ..., "contrast": ...}` from
`NLSQResult.params`. These are passed directly as `init_params` to
`mcmc.run(rng_key, *model_args, init_params=init_params_jax)`. NumPyro's MCMC
`init_params` dict must use the **exact NumPyro site names** declared in
`numpyro.sample("tau", ...)`.

This works correctly for `single_exp_model` because the NLSQ param names happen to
match the NumPyro site names. However, for `run_double_exp_fit`, the NLSQ parameter
names include `"tau1"`, `"tau2"`, `"contrast1"`, `"contrast2"` while the NumPyro model
declares sites `"tau1"`, `"tau2"`, `"contrast1"`, `"contrast2"` — this is consistent.
The silent failure case is: if any model has a mismatch between NLSQ and NumPyro
parameter names (e.g. due to future refactoring), NumPyro silently ignores the
unrecognized `init_params` keys and falls back to its own initialization, invalidating
the warm-start without any warning.

**Proposed fix:**  
Add an explicit validation step that checks all keys in `init_params` appear as
declared NumPyro sample sites before running MCMC:
```python
model_sites = numpyro.infer.util.get_parameter_transforms(model, *model_args)
unknown = set(init_params) - set(model_sites)
if unknown:
    logger.warning(f"init_params keys {unknown} do not match NumPyro sites; warm-start ignored")
```

---

### P1-005: `_active` flag in Bayesian sampler never reset on cancel

**File:** `xpcsviewer/fitting/sampler.py` (noted in threading audit)  
**Numerical impact:** When NUTS sampling is cancelled mid-chain (e.g. user closes the
dialog), `_active = True` is never cleared. On the next invocation the sampler guard
blocks the new run, leaving the fitting pipeline permanently stalled until restart. This
is primarily a threading issue (already documented in `threading_audit.md`) but has a
numerical consequence: the partially-computed JAX XLA compilation cache may retain stale
compiled functions that reference the cancelled `jax.random.PRNGKey`, causing
non-reproducible results on the next run if the process is not restarted.

**Proposed fix:** Reset `_active = False` in a `finally` block wrapping the MCMC run.

---

## P2 — Medium (Inefficiency or Minor Correctness Risk)

### P2-001: Unnecessary host-device sync inside `refine_beam_center` convergence check

**File:** `xpcsviewer/simplemask/calibration.py`, lines 76–93  
**Root cause:**  
```python
if i % 10 == 0 or i == max_iterations - 1:
    loss_val = float(loss)   # triggers host-device sync every 10 steps
```
The pattern is intentional for convergence checking, but `float(loss)` on a JAX array
forces a device-to-host transfer and blocks until the XLA computation completes. Since
`val_and_grad_fn` is JIT-compiled, JAX can pipeline multiple iterations asynchronously.
The sync every 10 steps partially defeats this asynchronous execution. For typical
`max_iterations=100`, this causes 10 forced syncs that serialize the computation.

**Proposed fix:** Use `jax.lax.while_loop` to keep the entire optimization loop on-device,
or accept the current 10-step sync cadence as a deliberate trade-off (the comment
mentions "reduce host-device sync overhead" so the intent is understood).

---

### P2-002: `lru_cache` caches mutable dicts — aliased numpy arrays risk silent mutation

**File:** `xpcsviewer/simplemask/qmap.py`, lines 278–303 (`_compute_transmission_qmap_cached`)  
**Root cause:**  
The cached function returns a tuple `(qmap_dict, qmap_unit_dict)` where `qmap_dict`
contains NumPy arrays. The comment in `compute_transmission_qmap` (BUG-055) acknowledges
this:

> "Callers that need their own writable copy must call np.copy() themselves."

However, if any caller mutates a returned array in-place (e.g. `qmap["q"][mask] = 0`),
the cached dict is permanently corrupted for all future callers. NumPy arrays from
`ensure_numpy()` are writable by construction.

**Proposed fix:** Either mark all returned arrays as read-only:
```python
for arr in qmap.values():
    if isinstance(arr, np.ndarray):
        arr.flags.writeable = False
```
Or return a shallow copy of the dict (arrays as views, not copies) to prevent accidental
dict mutations while preserving array aliasing:
```python
return dict(qmap), dict(qmap_unit)
```

---

### P2-003: `single_exp_func` JIT-compiles with `_xnp = jnp` but NLSQ calls with numpy

**File:** `xpcsviewer/fitting/models.py`, lines 238–246  
**Root cause:**  
```python
@_maybe_jit
def single_exp_func(x, tau, baseline, contrast):
    tau = _xnp.clip(tau, 1e-30, None)
    return baseline + contrast * _xnp.exp(-2 * x / tau)
```
`_xnp` is set to `jnp` at import time when JAX is available. `@_maybe_jit` wraps the
function with `jax.jit`. NLSQ's `nlsq.fit` calls `model_fn(x, *p0_array)` internally
during optimization with NumPy arrays. JAX JIT accepts NumPy array inputs by converting
them to JAX device arrays, but this triggers an implicit transfer on every NLSQ
iteration. For large-scale fitting (many Q-bins), this is an avoidable overhead.

Furthermore, if NLSQ calls the model with a 1-D NumPy `x` array and scalar floats for
`tau`, `baseline`, `contrast`, JAX will retrace for each unique Python scalar value
(these are not static argnums), generating hundreds of recompilations for a single fit.

**Proposed fix:** Provide a separate non-JIT `_single_exp_func_numpy` for use in NLSQ
warm-start:
```python
def single_exp_func_numpy(x, tau, baseline, contrast):
    tau = max(tau, 1e-30)
    return baseline + contrast * np.exp(-2 * x / tau)
```
Use the JIT version only for the JAX/NumPyro model.

---

### P2-004: `compute_q_at_pixel` always converts to `float` — breaks differentiability

**File:** `xpcsviewer/simplemask/qmap.py`, lines 194–222  
**Root cause:**  
```python
return float(q)  # Always returns a Python float for consistent display behavior.
```
The docstring example shows using `jax.grad(compute_q_at_pixel, ...)`, but the function
returns `float(q)`, which strips JAX tracing information. `jax.grad` through a
`float()` call will raise `TypeError: Argument 'x' of type <class 'float'> is not a
valid JAX type.` The function is therefore not actually differentiable despite the
documentation claim.

**Proposed fix:** Remove the `float()` conversion from the return path, or document that
`compute_q_at_pixel` is not differentiable and that callers should use
`compute_q_sum_squared` (which correctly returns a JAX array) for gradient use cases:
```python
# Return JAX array to preserve differentiability; caller can do float(q) if needed
return q
```

---

### P2-005: `_maybe_jit` applied to NumPyro model functions causes double-tracing

**File:** `xpcsviewer/fitting/models.py`, lines 40–44  
**Root cause:**  
`_maybe_jit` applies `jax.jit` to the deterministic scalar functions (`single_exp_func`,
etc.) at module import time. NumPyro's NUTS already JIT-compiles the full model
log-probability (including the likelihood) internally via `jax.jit(potential_fn)`.
The `@_maybe_jit` decoration on the model functions creates a nested JIT situation:
the outer NUTS JIT traces into the inner JIT-compiled `single_exp_func`. While JAX
handles nested JIT correctly (the inner compilation is a no-op inside the outer), it
adds unnecessary compilation overhead at warm-up.

Note: This does not cause incorrect results, but it wastes ~1 compilation cycle per
inner function and may produce confusing XLA profiler traces.

**Proposed fix:** Remove `@_maybe_jit` from functions called inside NumPyro models;
keep it only for standalone NLSQ warm-start use where standalone JIT is beneficial.

---

## Summary Table

| ID | Severity | File | Line(s) | Title |
|----|----------|------|---------|-------|
| P0-001 | P0 | `simplemask/qmap.py` | 334–378 | Python if/elif inside `@jax.jit` for orientation |
| P1-001 | P1 | `fitting/nlsq.py` | 107 | `converged` always False (missing `.converged` attr) |
| P1-002 | P1 | `fitting/models.py` | 79 | `jnp.clip` kills gradient for tau in NUTS model |
| P1-003 | P1 | `fitting/nlsq.py` | 121–127 | `model_fn(x, *popt)` retriggers JIT trace per call |
| P1-004 | P1 | `fitting/sampler.py` | 130–148 | Warm-start init_params silently ignored on name mismatch |
| P1-005 | P1 | `fitting/sampler.py` | (various) | `_active` flag not reset on cancel — stale XLA cache |
| P2-001 | P2 | `simplemask/calibration.py` | 76–93 | Forced host-device sync every 10 steps in gradient loop |
| P2-002 | P2 | `simplemask/qmap.py` | 278–303 | Mutable NumPy arrays returned from `lru_cache` |
| P2-003 | P2 | `fitting/models.py` | 238–246 | JIT model called with NumPy by NLSQ → per-call retracing |
| P2-004 | P2 | `simplemask/qmap.py` | 194–222 | `float()` in `compute_q_at_pixel` breaks differentiability |
| P2-005 | P2 | `fitting/models.py` | 40–44 | `@_maybe_jit` on NumPyro model functions causes double JIT |

---

## Non-Issues (Investigated but Dismissed)

- **`lru_cache` on `_compute_transmission_qmap_cached`**: The `@lru_cache(maxsize=4)`
  works correctly because all arguments are Python primitives (float, tuple, int). JAX
  arrays are only used internally and are not part of the cache key. No issue.

- **`_JIT_CACHE` dict with FIFO eviction**: The manual `dict`-based JIT cache is
  unusual but correct. Dict ordering in Python 3.7+ is insertion-order, so `next(iter())`
  correctly returns the oldest key. The pattern is safe.

- **`val_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))`**: This is the standard
  JAX pattern. Wrapping `value_and_grad` in a single `jit` is correct and produces one
  XLA compilation for both forward and backward passes. No issue.

- **`FitResult.samples` shape validation in `__post_init__`**: The shape check
  `all(s == shape_values[0] for s in shape_values)` is correct for multi-chain samples
  where NumPyro returns `(num_chains * num_samples,)` shaped arrays. No issue.
