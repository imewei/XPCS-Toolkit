# ADR-002: Migration from scipy.optimize to NLSQ 0.6.0

## Status

Accepted

## Context

The XPCS Viewer fitting module originally used `scipy.optimize.curve_fit` for nonlinear least squares (NLSQ) fitting of G2 correlation functions. This approach had several limitations:

1. **No JAX integration**: `scipy.optimize.curve_fit` operates on NumPy arrays only. When the backend is JAX, data must be copied to CPU, fit with scipy, then results copied back -- defeating the purpose of GPU acceleration.
2. **No native diagnostics**: scipy provides only `popt` and `pcov`. Computing R-squared, AIC, BIC, prediction intervals, or model health diagnostics required manual implementation (hundreds of LOC of boilerplate).
3. **No automatic stability fixes**: Users encountering ill-conditioned problems had to manually adjust bounds, scaling, or initial guesses.
4. **No fallback strategies**: When fitting failed, scipy raised an exception with no recovery path.
5. **Duplicated statistics code**: The codebase contained ~400 LOC of custom statistical computation that replicated what a proper fitting library should provide natively.

NLSQ 0.6.0 was selected because it provides:
- Native `CurveFitResult` with built-in statistical properties (R-squared, AIC, BIC, RMSE, MAE, prediction intervals).
- Preset configurations (`fast`, `robust`, `global`, `streaming`, `large`) for different use cases.
- `auto_bounds=True` for automatic bound inference from data.
- `stability='auto'` for automatic numerical fixes.
- `fallback=True` for automatic fallback strategies on failure.
- `compute_diagnostics=True` for `ModelHealthReport` with health scores.

## Decision

We migrated the fitting module to use NLSQ 0.6.0's native API with a **delegation pattern** for the `NLSQResult` class.

### Architecture

```
xpcsviewer/fitting/
  __init__.py         # Public API: fit_single_exp, nlsq_fit, etc.
  models.py           # NumPyro probabilistic models + JIT-compiled curve functions
  nlsq.py             # nlsq_optimize(): NLSQ 0.6.0 curve fitting wrapper
  sampler.py          # NumPyro NUTS sampling with NLSQ warm-start
  results.py          # NLSQResult (delegation), FitResult (Bayesian), SamplerConfig
  visualization.py    # Plotting: posterior predictive, NLSQ fit, ArviZ diagnostics
  legacy.py           # scipy.optimize compatibility layer (deprecated)
```

### Key Design Choices

1. **Property delegation pattern**: `NLSQResult` stores the native `CurveFitResult` object and delegates statistical property access to it. When the native result is `None` (legacy code path or fitting failure), private fallback fields (`_r_squared`, `_adj_r_squared`, etc.) are used instead.

   ```python
   @property
   def r_squared(self) -> float:
       if self.native_result is not None:
           return self.native_result.r_squared  # Delegation
       return self._r_squared  # Legacy fallback
   ```

2. **Setter no-ops when delegation is active (BUG-039)**: Property setters on `NLSQResult` log a warning and do nothing when `native_result` is set. This prevents accidental overwriting of the authoritative native values.

3. **Sentinel fallback for failed fits (BUG-003)**: When NLSQ fitting fails, `nlsq_optimize()` returns a sentinel `NLSQResult` with `is_fallback=True`, `converged=False`, and all metrics set to `float('nan')`. Callers detect failure via `math.isnan(result.r_squared)` rather than catching exceptions.

4. **NaN defaults for failed fits (BUG-023)**: All statistical metrics default to `float('nan')` rather than `0.0`, making failed fits distinguishable from poor-but-converged fits.

5. **Covariance validation (FR-021)**: The `validate_pcov()` function checks the covariance matrix for infinities, NaN values, and negative diagonal elements, setting `pcov_valid` and `pcov_message` accordingly.

6. **Warm-start pipeline**: The complete fitting workflow is:
   - `nlsq_optimize()` runs NLSQ 0.6.0 to get point estimates.
   - Point estimates initialize NumPyro NUTS sampling as the warm-start.
   - `run_single_exp_fit()` in `sampler.py` orchestrates the full pipeline.
   - BFMI (Bayesian Fraction of Missing Information) is computed for convergence diagnostics per the Technical Guidelines.

7. **JIT-compiled model functions**: Curve functions (`single_exp_func`, `double_exp_func`, etc.) in `models.py` are decorated with `@_maybe_jit`, which applies `jax.jit` when JAX is available and is a no-op otherwise.

### Statistical Properties (Delegated from CurveFitResult)

| Property | Description | Source |
|----------|-------------|--------|
| `r_squared` | Coefficient of determination | `native_result.r_squared` |
| `adj_r_squared` | Adjusted R-squared | `native_result.adj_r_squared` |
| `rmse` | Root mean squared error | `native_result.rmse` |
| `mae` | Mean absolute error | `native_result.mae` |
| `aic` | Akaike Information Criterion | `native_result.aic` |
| `bic` | Bayesian Information Criterion | `native_result.bic` |
| `residuals` | Fit residuals | `native_result.residuals` |
| `predictions` | Model predictions | `native_result.predictions` |
| `covariance` | Parameter covariance matrix | `native_result.pcov` |
| `confidence_intervals` | 95% CI per parameter | `native_result.confidence_intervals` |
| `diagnostics` | ModelHealthReport | `native_result.diagnostics` |
| `health_score` | 0-100 health score | `native_result.diagnostics.health_score` |
| `condition_number` | Identifiability metric | `native_result.diagnostics.identifiability.condition_number` |

## Consequences

### What became easier

- **~400 LOC reduction**: Eliminated custom statistical computation code. All metrics come from the native `CurveFitResult`.
- **Robust fitting**: `stability='auto'` and `fallback=True` handle ill-conditioned problems automatically.
- **Model selection**: AIC and BIC are available natively, enabling principled model comparison (single vs. double vs. stretched exponential).
- **Health diagnostics**: `compute_diagnostics=True` provides a `ModelHealthReport` with health scores and identifiability analysis, surfaced directly through `NLSQResult.health_score` and `NLSQResult.condition_number`.
- **Prediction intervals**: `get_prediction_interval(x_new)` delegates to the native result, providing proper prediction intervals rather than the rough `+/- 2*RMSE` approximation.

### What became more difficult

- **Additional dependency**: NLSQ 0.6.0 is now a required dependency for fitting. If NLSQ is not installed, the fitting module raises an `ImportError`.
- **Delegation complexity**: The property delegation pattern in `NLSQResult` requires careful handling of the dual path (native vs. legacy). New properties must implement both paths.
- **Version coupling**: The codebase depends on NLSQ 0.6.0's specific API surface (`CurveFitResult` properties, `ModelHealthReport` structure). Major NLSQ API changes would require updates to the delegation layer.
- **Health check fragility (BUG-040)**: Determining `is_healthy` requires checking multiple possible API patterns (`is_healthy` attribute, `health_score >= 70`, enum value comparison) to handle different NLSQ versions gracefully.
