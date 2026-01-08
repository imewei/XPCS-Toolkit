# Robust Fitting Integration

This document describes the integration of robust fitting capabilities into XPCSViewer using the `nlsq` library.

## Overview

XPCSViewer uses the `nlsq` library (version 0.6.0+) for JAX-accelerated non-linear least squares fitting. This provides:

- **Large dataset support**: Efficient handling of datasets with hundreds of thousands of points
- **Multiple presets**: `fast`, `robust`, `global`, `streaming`, and `large` configurations
- **MCMC warm-start**: Optimized initial estimates for Bayesian inference

## API

### `nlsq_optimize`

```python
from xpcsviewer.fitting.nlsq import nlsq_optimize

result = nlsq_optimize(
    model_func,      # JAX-compatible model function
    x_data,          # Independent variable data
    y_data,          # Dependent variable (observations)
    p0,              # Initial parameter guess
    sigma=None,      # Optional uncertainties
    bounds=None,     # Optional (lower, upper) bounds tuple
    preset="robust"  # Fitting preset
)
```

### Result Object

The result contains:
- `popt`: Optimized parameters
- `pcov`: Parameter covariance matrix
- `residuals`: Fit residuals
- `chi_squared`: Chi-squared statistic
- `converged`: Convergence status

## Presets

| Preset | Use Case |
|--------|----------|
| `fast` | Quick fits, small datasets |
| `robust` | Default, handles outliers |
| `global` | Multiple minima exploration |
| `streaming` | Memory-limited environments |
| `large` | Datasets > 100k points |

## Integration with MCMC

The NLSQ results serve as warm-start estimates for NumPyro/Blackjax MCMC sampling:

```python
result = nlsq_optimize(model, x, y, p0)
# Use result.popt to initialize MCMC chains
```

## Requirements

- `nlsq >= 0.6.0`
- `jax >= 0.4.0`
- `jaxlib >= 0.4.0`
