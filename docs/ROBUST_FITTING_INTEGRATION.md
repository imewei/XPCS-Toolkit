# Robust G2 Diffusion Fitting Integration Guide

## Overview

The XPCS Toolkit now includes a comprehensive robust fitting framework specifically designed for G2 correlation function analysis. This integration provides enhanced reliability, performance optimizations, and detailed diagnostics while maintaining full backward compatibility with existing analysis workflows.

## Key Features

### 🔧 Core Capabilities
- **Multi-strategy robust optimization** - Automatic fallback between Levenberg-Marquardt, Trust Region Reflective, and Differential Evolution
- **Bootstrap uncertainty estimation** - Non-parametric confidence intervals with parallel processing
- **Outlier detection and handling** - Intelligent data quality assessment and cleanup
- **Advanced diagnostics** - Comprehensive residual analysis, goodness-of-fit metrics, and convergence monitoring

### ⚡ Performance Optimizations
- **Adaptive memory management** - Intelligent chunking for datasets >2GB
- **Parallel processing** - Optimized CPU utilization with conservative resource management
- **Intelligent caching** - Joblib-based caching with LRU cleanup and usage tracking
- **Vectorized operations** - NumPy-optimized algorithms throughout

### 🔒 Production Ready
- **100% backward compatibility** - Existing code continues to work unchanged
- **Comprehensive error handling** - Graceful degradation with detailed logging
- **Memory pressure monitoring** - Integration with XPCS memory management system
- **Scientific validation** - Maintains numerical precision for research accuracy

## API Reference

### XpcsFile Integration

#### Standard G2 Fitting (Enhanced)
```python
# Existing method now supports robust fitting
xf.fit_g2(
    q_range=(0.001, 0.01),
    t_range=(1e-6, 1.0),
    bounds=bounds,
    fit_flag=fit_flag,
    fit_func="single",
    robust_fitting=True,           # NEW: Enable robust fitting
    diagnostic_level="standard",   # NEW: Control diagnostic detail
    bootstrap_samples=500          # NEW: Bootstrap uncertainty estimation
)
```

#### Dedicated Robust Fitting
```python
# Dedicated robust fitting method
results = xf.fit_g2_robust(
    q_range=(0.001, 0.01),
    t_range=(1e-6, 1.0),
    bounds=bounds,
    fit_flag=fit_flag,
    fit_func="single",
    diagnostic_level="comprehensive",  # 'basic', 'standard', 'comprehensive'
    bootstrap_samples=1000            # Number of bootstrap resamples
)

# Access enhanced results
params = results['fit_val'][0]['params']
uncertainties = results['fit_val'][0]['param_errors']
diagnostics = results['fit_val'][0]['diagnostics']
```

#### High-Performance Fitting
```python
# For large datasets (>10MB correlation data)
results = xf.fit_g2_high_performance(
    q_range=(0.001, 0.01),
    bounds=bounds,
    fit_func="single",
    bootstrap_samples=500,
    diagnostic_level="standard",
    max_memory_mb=4096  # Memory limit for adaptive processing
)

# Performance metrics included
performance_info = results['performance_info']
timing_info = results['timing']
optimization_summary = results['optimization_summary']
```

### Direct API Access

#### Robust Curve Fitting (scipy.curve_fit replacement)
```python
from xpcs_toolkit.helper.fitting import robust_curve_fit

# Drop-in replacement for scipy.optimize.curve_fit
popt, pcov = robust_curve_fit(
    func=lambda x, a, b: a * np.exp(-b * x),
    xdata=time_delays,
    ydata=g2_values,
    p0=[1.0, 0.1],
    bounds=([0, 0], [np.inf, np.inf]),
    sigma=g2_errors
)
```

#### Comprehensive Analysis Framework
```python
from xpcs_toolkit.helper.fitting import ComprehensiveDiffusionAnalyzer

# Advanced diffusion analysis
analyzer = ComprehensiveDiffusionAnalyzer(diagnostic_level='comprehensive')
results = analyzer.analyze_diffusion(
    tau=time_delays,
    g2_data=correlation_data,
    g2_errors=error_estimates,
    q_value=0.005,
    models_to_test=['simple_diffusion', 'stretched_exponential'],
    bootstrap_samples=1000
)

# Rich diagnostic information
diagnostics = results['diagnostics']
model_comparison = results['model_comparison']
parameter_uncertainties = results['diffusion_parameters']
```

## Performance Guidelines

### Memory Management
- **Small datasets (<100MB)**: Use standard `fit_g2_robust()`
- **Medium datasets (100MB-1GB)**: Use `fit_g2_high_performance()` with default settings
- **Large datasets (>1GB)**: Set `max_memory_mb` to 50% of available RAM

### Bootstrap Recommendations
- **Exploratory analysis**: 100-500 samples
- **Publication results**: 1000-5000 samples
- **High-precision studies**: 5000+ samples

### Diagnostic Levels
- **'basic'**: Essential fit metrics only (fastest)
- **'standard'**: Includes residual analysis and outlier detection (recommended)
- **'comprehensive'**: Full bootstrap CI, model comparison, advanced diagnostics

## Migration Guide

### From Standard Fitting
```python
# OLD: Standard fitting
results = xf.fit_g2(q_range=q_range, bounds=bounds, fit_func="single")

# NEW: Enhanced with robust fitting
results = xf.fit_g2(
    q_range=q_range,
    bounds=bounds,
    fit_func="single",
    robust_fitting=True,      # Enable robust methods
    bootstrap_samples=500     # Add uncertainty estimation
)
# All existing result access patterns work unchanged
```

### From scipy.optimize.curve_fit
```python
# OLD: scipy curve_fit
from scipy.optimize import curve_fit
popt, pcov = curve_fit(func, xdata, ydata, p0=p0, bounds=bounds)

# NEW: Robust curve_fit (drop-in replacement)
from xpcs_toolkit.helper.fitting import robust_curve_fit
popt, pcov = robust_curve_fit(func, xdata, ydata, p0=p0, bounds=bounds)
# Enhanced reliability with identical interface
```

## Configuration

### Environment Variables
```bash
# Suppress Qt warnings during fitting
export PYXPCS_SUPPRESS_QT_WARNINGS=1

# Set custom cache directory
export XPCS_CACHE_DIR=/path/to/cache
```

### Memory Pressure Thresholds
```python
# Custom memory management
from xpcs_toolkit.helper.fitting import XPCSPerformanceOptimizer

optimizer = XPCSPerformanceOptimizer(
    max_memory_mb=2048,    # Memory limit
    cache_size=1000        # Cache entry limit
)
```

## Troubleshooting

### Common Issues

**Issue**: Fitting fails with "Insufficient data" error
**Solution**: Check data quality - ensure >3 valid points per q-value

**Issue**: Memory errors with large datasets
**Solution**: Reduce `max_memory_mb` or enable chunking in high-performance mode

**Issue**: Slow bootstrap analysis
**Solution**: Reduce `bootstrap_samples` or use `diagnostic_level='basic'`

### Performance Optimization

1. **Use caching** for repeated analysis of identical datasets
2. **Adjust parallelization** based on system capabilities
3. **Monitor memory usage** with built-in tracking
4. **Choose appropriate diagnostic level** for analysis needs

### Logging and Monitoring
```python
import logging

# Enable detailed fitting logs
logging.getLogger('xpcs_toolkit.helper.fitting').setLevel(logging.DEBUG)

# Monitor performance metrics
results = xf.fit_g2_high_performance(...)
print(f"Memory usage: {results['performance_info']['estimated_memory_mb']:.1f} MB")
print(f"Processing time: {results['timing']['time_per_q']*1000:.1f} ms/q")
```

## Scientific Validation

The robust fitting framework has been validated to:
- Maintain numerical precision equivalent to original methods
- Provide more reliable parameter estimates in noisy conditions
- Correctly handle edge cases (insufficient data, poor initial guesses)
- Scale efficiently to datasets with 1000+ q-values

All optimizations preserve the scientific integrity required for publication-quality XPCS analysis.

## Support

For questions or issues:
1. Check the [troubleshooting section](#troubleshooting)
2. Review test cases in `tests/unit/test_robust_fitting_integration.py`
3. Enable debug logging for detailed diagnostic information
4. Submit issues with complete error logs and data characteristics

## Version Compatibility

- **Minimum Python**: 3.8+
- **Required Dependencies**: scipy≥1.7, numpy≥1.20, joblib≥1.0
- **XPCS Toolkit**: Compatible with all versions ≥2.0
- **Backward Compatibility**: 100% with existing analysis scripts
