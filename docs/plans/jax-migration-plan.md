# JAX Migration Plan for xpcsviewer

## Overview

**Goal**: Full migration of xpcsviewer from NumPy to JAX for GPU acceleration, JIT compilation, and automatic differentiation.

**Scope**: Entire package (102 Python files, ~134k lines)

**Strategy**: Full NumPy replacement with graceful CPU/GPU fallback

**Environment**: Mixed (GPU and CPU-only deployments)

---

## Phase 1: Foundation (Files to Create)

### 1.1 Backend Abstraction Layer

Create `xpcsviewer/backends/` module:

| File | Purpose |
|------|---------|
| `__init__.py` | Backend selection, initialization, environment config |
| `_base.py` | `BackendProtocol` - abstract interface for array operations |
| `_numpy_backend.py` | NumPy implementation (fallback) |
| `_jax_backend.py` | JAX implementation with JIT support |
| `_device.py` | `DeviceManager` - CPU/GPU selection and placement |
| `_conversions.py` | `ensure_numpy()`, `ensure_backend_array()` utilities |

### 1.2 SciPy Replacements

Create `xpcsviewer/backends/scipy_replacements/`:

| File | Replaces |
|------|----------|
| `ndimage.py` | `scipy.ndimage.gaussian_filter` → JAX convolution |
| `interpolate.py` | `scipy.interpolate.interp1d` → `jnp.interp` + custom spline |

### 1.3 Bayesian Fitting Module

Create `xpcsviewer/fitting/` module (replaces `helper/fitting.py`):

| File | Purpose |
|------|---------|
| `__init__.py` | Public API for fitting functions |
| `nlsq.py` | JAX-based nonlinear least squares (warm-start generator) |
| `models.py` | Probabilistic models for NumPyro (single_exp, double_exp, etc.) |
| `sampler.py` | NumPyro NUTS sampler with NLSQ warm-start |
| `results.py` | `FitResult` dataclass with posterior samples, diagnostics |

**Fitting Pipeline:**
1. **NLSQ Phase**: Fast initial parameter estimation via JAX-accelerated NLSQ
2. **NUTS Phase**: Full Bayesian posterior sampling using NumPyro NUTS
3. **Warm-Start**: NLSQ point estimates initialize NUTS chains

**No backward compatibility** with `scipy.optimize.curve_fit` - all fitting migrates to new pipeline.

### 1.4 Configuration

- Environment variables: `XPCS_USE_JAX`, `XPCS_USE_GPU`, `XPCS_GPU_FALLBACK`
- Enable float64: `jax.config.update('jax_enable_x64', True)`
- Add to `pyproject.toml`:
  ```toml
  [project.optional-dependencies]
  jax = [
    "jax>=0.4.35",
    "jaxlib>=0.4.35",
    "numpyro>=0.15.0",
    "arviz>=0.18.0",  # Diagnostics and visualization
  ]
  jax-cuda = [
    "jax[cuda12]>=0.4.35",
    "numpyro>=0.15.0",
    "arviz>=0.18.0",
  ]
  ```

---

## Phase 2: SimpleMask Module Migration

### Critical Files (in order)

1. **`xpcsviewer/simplemask/qmap.py`**
   - Replace: `np.meshgrid`, `np.hypot`, `np.arctan2`, `np.sin`, `np.cos`
   - Add: JIT-compiled `_compute_transmission_qmap_jit()`
   - Keep: LRU caching via dict-based cache (JAX arrays not hashable)

2. **`xpcsviewer/simplemask/utils.py`**
   - Replace: `np.digitize`, `np.bincount`, `np.unique`, `np.linspace`, `np.logspace`
   - Fix: `np.unique(return_inverse=True)` requires `size` param for JIT
   - Fix: `check_consistency()` - vectorize Python dict loop

3. **`xpcsviewer/simplemask/area_mask.py`**
   - Replace in-place ops: `mask[idx] = val` → `mask.at[idx].set(val)`
   - Replace: `xmap[xmap > vend] -= 360` → `jnp.where(xmap > vend, xmap - 360, xmap)`
   - Replace: `np.logical_and`, `np.logical_or`, `np.nonzero`

4. **`xpcsviewer/simplemask/simplemask_kernel.py`**
   - Replace: slice assignment with `lax.dynamic_update_slice`
   - Keep: PyQtGraph ROI operations (require NumPy at boundary)

---

## Phase 3: Bayesian Fitting & Analysis Modules

### 3.1 New Fitting Module (Replace `helper/fitting.py`)

**Delete**: `xpcsviewer/helper/fitting.py` (no backward compatibility)

**Create**: `xpcsviewer/fitting/` with NLSQ → NumPyro NUTS pipeline:

1. **`xpcsviewer/fitting/nlsq.py`** - JAX-accelerated NLSQ
   ```python
   def nlsq_fit(model_fn, x, y, yerr, p0, bounds):
       """Fast NLSQ for warm-start parameter estimation."""
       # Returns point estimates and approximate covariance
   ```

2. **`xpcsviewer/fitting/models.py`** - NumPyro probabilistic models
   ```python
   def single_exp_model(x, y, yerr):
       """NumPyro model: y = baseline + contrast * exp(-2*x/tau)"""
       tau = numpyro.sample("tau", dist.LogNormal(0, 1))
       baseline = numpyro.sample("baseline", dist.Normal(1, 0.1))
       contrast = numpyro.sample("contrast", dist.HalfNormal(1))
       # ...

   def double_exp_model(x, y, yerr):
       """NumPyro model for double exponential decay."""

   def power_law_model(x, y, yerr):
       """NumPyro model for power law Q-dependence."""
   ```

3. **`xpcsviewer/fitting/sampler.py`** - NUTS with warm-start
   ```python
   def fit_with_nuts(model, x, y, yerr, nlsq_init, num_warmup=500, num_samples=1000):
       """Run NUTS sampling initialized from NLSQ estimates."""
       kernel = NUTS(model)
       mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples)
       mcmc.run(rng_key, x, y, yerr, init_params=nlsq_init)
       return mcmc
   ```

4. **`xpcsviewer/fitting/results.py`** - Fit results container
   ```python
   @dataclass
   class FitResult:
       samples: dict[str, jnp.ndarray]  # Posterior samples
       summary: pd.DataFrame             # Mean, std, HDI
       diagnostics: dict                 # R-hat, ESS, divergences
       nlsq_init: dict                   # Initial NLSQ estimates
       arviz_data: az.InferenceData      # For plotting
   ```

### 3.2 Update Callers

1. **`xpcsviewer/module/g2mod.py`**
   - Replace: `fit_g2()` calls to use new `xpcsviewer.fitting` module
   - Replace: `np.mean`, `np.std`, `np.median`, `np.corrcoef` with JAX
   - Replace: `scipy.interpolate.interp1d` with JAX interpolation
   - Add: JIT for batch G2 normalization
   - Return: `FitResult` with full posterior instead of just point estimates

2. **`xpcsviewer/module/twotime.py`**
   - Replace: `scipy.ndimage.gaussian_filter` with JAX convolution
   - Replace: `np.nan_to_num`, `np.percentile`, `np.isfinite`

3. **`xpcsviewer/module/saxs1d.py`**, **`saxs2d.py`**
   - Replace: statistical operations with JAX
   - Keep: Matplotlib plotting (requires NumPy conversion)

---

## Phase 4: Utilities Migration

### Files to Migrate

1. **`xpcsviewer/utils/vectorized_roi.py`**
   - Replace: `np.bincount` with JIT-compiled version
   - Add: `size` parameter for dynamic output shapes
   - Parallelize: Use `jax.vmap` for batch ROI processing

2. **`xpcsviewer/utils/streaming_processor.py`**
   - Add: Device-aware chunking
   - Add: GPU memory monitoring

3. **`xpcsviewer/fileIO/qmap_utils.py`**
   - Replace: Q-map computation calls
   - Keep: HDF5 I/O with NumPy (h5py requirement)

---

## Phase 5: Testing Infrastructure

### New Test Structure

```
tests/jax_migration/
  conftest.py              # Backend fixtures, tolerances
  backend/
    test_backend_detection.py
    test_device_transfer.py
  numerical/
    test_qmap_equivalence.py
    test_partition_equivalence.py
    test_fitting_equivalence.py
  precision/
    test_float32_vs_float64.py
    test_angular_computations.py
  performance/
    test_benchmarks.py
  integration/
    test_qt_jax_interop.py
    test_hdf5_jax_io.py
```

### Tolerance Standards

| Operation | atol | rtol |
|-----------|------|------|
| Q-map transmission | 1e-7 | 1e-6 |
| Q-map reflection | 1e-6 | 1e-5 |
| Phi angles | 1e-5 | 1e-4 |
| Partition indices | 0 | 0 (exact) |
| Boolean masks | 0 | 0 (exact) |

### Bayesian Fitting Validation

| Metric | Acceptance Criteria |
|--------|---------------------|
| R-hat (convergence) | < 1.01 for all parameters |
| ESS (effective samples) | > 400 for all parameters |
| Divergences | 0 divergent transitions |
| NLSQ vs posterior mean | Within 2σ of posterior |
| Posterior predictive | 95% HDI covers data |

### CI/CD Updates

Add JAX test job to `.github/workflows/ci.yml`:
- Test both CPU and GPU backends
- Run equivalence tests against NumPy reference
- Performance benchmarks on GPU runner

---

## Key Migration Patterns

### In-Place Operations
```python
# NumPy
arr[idx] = value
# JAX
arr = arr.at[idx].set(value)
```

### Conditional Updates
```python
# NumPy
xmap[xmap > vend] -= 360.0
# JAX
xmap = jnp.where(xmap > vend, xmap - 360.0, xmap)
```

### Dynamic Shapes (nonzero, unique)
```python
# NumPy
indices = np.nonzero(mask)
# JAX (must specify max size)
indices = jnp.nonzero(mask, size=max_elements, fill_value=-1)
```

### Control Flow in JIT
```python
# NumPy
if condition:
    result = a
else:
    result = b
# JAX
result = lax.cond(condition, lambda: a, lambda: b)
```

---

## Boundary Handling

### Must Convert to NumPy at:
1. **HDF5 I/O** - h5py requires NumPy arrays
2. **PyQtGraph display** - `ImageItem`, `PlotWidget`
3. **Matplotlib** - All plotting functions
4. **Qt Signals** - Data passing through signal/slot

### Conversion Pattern
```python
from xpcsviewer.backends import ensure_numpy

# At boundary
numpy_array = ensure_numpy(jax_array)
imageview.setImage(numpy_array)
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Numerical precision | Enable float64, comprehensive tolerance tests |
| GPU unavailable | Automatic CPU fallback via DeviceManager |
| Memory exhaustion | Chunked processing, memory monitoring |
| API breakage | Maintain 100% API compatibility |
| Performance regression | Benchmark CI, keep NumPy fast path |

---

## Critical Files Summary

### Phase 1 (Create)
- `xpcsviewer/backends/__init__.py`
- `xpcsviewer/backends/_base.py`
- `xpcsviewer/backends/_numpy_backend.py`
- `xpcsviewer/backends/_jax_backend.py`
- `xpcsviewer/backends/_device.py`
- `xpcsviewer/backends/_conversions.py`

### Phase 2 (Modify - SimpleMask)
- `xpcsviewer/simplemask/qmap.py`
- `xpcsviewer/simplemask/utils.py`
- `xpcsviewer/simplemask/area_mask.py`
- `xpcsviewer/simplemask/simplemask_kernel.py`

### Phase 3 (Create - Fitting Module)
- `xpcsviewer/fitting/__init__.py`
- `xpcsviewer/fitting/nlsq.py`
- `xpcsviewer/fitting/models.py`
- `xpcsviewer/fitting/sampler.py`
- `xpcsviewer/fitting/results.py`

### Phase 3 (Delete)
- `xpcsviewer/helper/fitting.py` (no backward compatibility)

### Phase 3-4 (Modify - Analysis & Utilities)
- `xpcsviewer/module/g2mod.py`
- `xpcsviewer/module/twotime.py`
- `xpcsviewer/utils/vectorized_roi.py`
- `xpcsviewer/fileIO/qmap_utils.py`

### Phase 5 (Create)
- `tests/jax_migration/conftest.py`
- `tests/jax_migration/numerical/*.py`
- `tests/jax_migration/precision/*.py`
- `tests/jax_migration/performance/*.py`
- `pyproject.toml` (update dependencies)
- `.github/workflows/ci.yml` (add JAX jobs)
