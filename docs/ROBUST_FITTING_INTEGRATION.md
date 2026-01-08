# Robust Fitting Integration Guide

## Overview

This document describes the robust fitting integration in XPCSviewer, including the JAX-accelerated NLSQ (Nonlinear Least Squares) solver and its integration with the Bayesian MCMC sampler.

## 🎯 Task Overview

The fitting system provides:
- **NLSQ Optimization**: Fast curve fitting using the `nlsq` library
- **Warm-start for MCMC**: NLSQ results initialize the Bayesian sampler
- **Large Dataset Support**: Automatic streaming and batching for large data

## 🔧 Technical Implementation

### NLSQ Solver

The core fitting function is `nlsq_optimize()` in `xpcsviewer/fitting/nlsq.py`:

```python
from xpcsviewer.fitting.nlsq import nlsq_optimize
from xpcsviewer.fitting.models import single_exp_func

result = nlsq_optimize(
    model_fn=single_exp_func,
    x=tau_values,
    y=g2_values,
    yerr=g2_errors,
    p0={"tau": 1.0, "baseline": 1.0, "contrast": 0.3},
    bounds={
        "tau": (1e-6, 1e6),
        "baseline": (0.0, 2.0),
        "contrast": (0.0, 1.0),
    },
    preset="robust",  # Options: fast, robust, global, streaming, large
)
```

### Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `fast` | Single-start optimization | Quick fits, good initial guess |
| `robust` | Multi-start with 5 starts | Default, reliable convergence |
| `global` | 20 starts | Finding global minimum |
| `streaming` | Streaming optimization | Large datasets with multi-start |
| `large` | Auto-detect strategy | Automatic dataset size handling |

### Model Functions

Available model functions in `xpcsviewer/fitting/models.py`:

- `single_exp_func(x, tau, baseline, contrast)` - Single exponential decay
- `stretched_exp_func(x, tau, baseline, contrast, beta)` - Stretched exponential
- `double_exp_func(x, tau1, tau2, baseline, contrast1, contrast2)` - Double exponential
- `power_law_func(q, tau0, alpha)` - Power law for Q-dependence

All model functions use `jax.numpy` for GPU acceleration and are JIT-compiled.

### Result Structure

```python
@dataclass
class NLSQResult:
    params: dict[str, float]      # Fitted parameter values
    covariance: np.ndarray        # Parameter covariance matrix
    residuals: np.ndarray         # Fit residuals
    chi_squared: float            # Reduced chi-squared
    converged: bool               # Convergence status
    pcov_valid: bool              # Covariance validity flag
    pcov_message: str             # Covariance validation message
```

## 📊 Validation Results

### Performance Benchmarks

| Component | Speedup | Notes |
|-----------|---------|-------|
| SAXS binning | 73x | Loop → vectorized |
| C2 statistics | 448x | Loop → vectorized |
| Model functions | Amortized | JIT compilation overhead ~500ms |

### Test Coverage

- Unit tests: `tests/jax_migration/fitting/test_nlsq.py`
- Equivalence tests: `tests/jax_migration/numerical/test_fitting_equivalence.py`
- Benchmark tests: `tests/benchmarks/performance/test_fitting_jit.py`

## 🏆 Achievement Summary

1. **Unified Solver**: Replaced optimistix with NLSQ library
2. **Large Dataset Support**: Built-in streaming and batching
3. **JAX Integration**: Full GPU acceleration when available
4. **Robust Convergence**: Multi-start optimization presets
5. **Clean API**: Simple dict-based parameter interface

## Troubleshooting

### Common Issues

1. **Slow first fit**: JIT compilation overhead. Subsequent fits are faster.
2. **Poor convergence**: Try `preset="global"` for difficult data.
3. **Out of memory**: Use `preset="streaming"` for large datasets.

### Error Messages

| Error | Solution |
|-------|----------|
| `nlsq fitting failed` | Check bounds and initial guess |
| `covariance invalid` | Data may have too few points |
| `chi_squared > 10` | Poor model fit, check data quality |

## Integration with MCMC

NLSQ results are used to warm-start Bayesian MCMC sampling:

```python
# 1. Get NLSQ MAP estimate
nlsq_result = nlsq_optimize(model_fn, x, y, yerr, p0, bounds)

# 2. Use as MCMC initialization
initial_theta = list(nlsq_result.params.values())
```

This provides faster MCMC convergence by starting near the posterior mode.
