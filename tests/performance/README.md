# XPCS Toolkit Performance Testing Framework

This comprehensive performance testing and benchmarking framework provides automated performance regression detection, scalability testing, and optimization validation for XPCS Toolkit.

## Overview

The performance testing framework consists of:

- **Comprehensive Benchmarks**: HDF5 I/O, scientific analysis algorithms, memory management, threading, and GUI performance
- **Regression Detection**: Automated performance regression detection with CI/CD integration
- **Scalability Testing**: Performance scaling analysis with data size and concurrent operations
- **Optimization Validation**: Before/after comparisons for performance optimizations
- **Trend Analysis**: Historical performance tracking and anomaly detection

## Directory Structure

```
tests/performance/
├── __init__.py                 # Framework configuration and constants
├── conftest.py                 # Shared fixtures and configuration
├── README.md                   # This documentation
├── benchmarks/                 # Benchmark test modules
│   ├── test_hdf5_performance.py           # HDF5 I/O benchmarks
│   ├── test_scientific_analysis_performance.py  # Algorithm benchmarks
│   ├── test_memory_threading_performance.py     # Memory/threading benchmarks
│   ├── test_gui_performance.py            # GUI performance benchmarks
│   ├── test_scalability.py               # Scalability testing
│   └── test_optimization_validation.py    # Optimization validation
├── config/
│   └── baseline_performance.json         # Performance baselines
├── data/                       # Test data (created dynamically)
├── reports/                    # Generated reports
├── utils/
│   ├── synthetic_data.py       # Synthetic data generation
│   └── regression_detection.py # Regression analysis tools
```

## Quick Start

### Running Benchmarks

```bash
# Run all performance benchmarks
pytest tests/performance/ --benchmark-only

# Run specific category
pytest tests/performance/benchmarks/test_hdf5_performance.py --benchmark-only

# Run with baseline comparison
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

# Generate detailed reports
pytest tests/performance/ --benchmark-only --benchmark-save=results --benchmark-save-data
```

### Running with Memory Profiling

```bash
# Install memory profiling dependencies
pip install memory_profiler psutil

# Run memory-intensive tests
pytest tests/performance/ -m memory_intensive --profile-mem
```

### Scalability Testing

```bash
# Run scalability tests
pytest tests/performance/benchmarks/test_scalability.py -v

# Run with specific data sizes
pytest tests/performance/benchmarks/test_scalability.py -k "large" -v
```

## Benchmark Categories

### 1. HDF5 I/O Performance (`test_hdf5_performance.py`)

Tests performance of HDF5 file operations:

- **Basic Operations**: File open/close, metadata access, dataset info
- **Data Reading**: Full datasets, partial reads, chunked reading
- **Connection Pooling**: Pool creation, acquisition, concurrent usage
- **Memory Efficiency**: Large dataset handling, cleanup performance

```python
# Example: Benchmark file reading performance
@pytest.mark.benchmark(group="hdf5_basic")
@pytest.mark.parametrize("file_size", ["tiny", "small", "medium"])
def test_file_open_close_performance(benchmark, synthetic_hdf5_file):
    result = benchmark.pedantic(open_close_file, rounds=20, iterations=5)
```

### 2. Scientific Analysis Performance (`test_scientific_analysis_performance.py`)

Tests core scientific algorithms:

- **G2 Analysis**: Data extraction, exponential fitting, multi-tau correlation
- **SAXS Analysis**: 2D analysis, radial averaging, peak finding
- **Two-Time Analysis**: Correlation matrix calculation, parallel processing
- **Fitting Algorithms**: Different optimization methods, robustness testing

```python
# Example: Benchmark G2 fitting performance
@pytest.mark.benchmark(group="g2_analysis")
@pytest.mark.parametrize("fit_type", ["single_exp", "double_exp"])
def test_g2_fitting_performance(benchmark, fit_type):
    result = benchmark.pedantic(fit_g2, rounds=15, iterations=3)
```

### 3. Memory and Threading Performance (`test_memory_threading_performance.py`)

Tests system-level performance:

- **Memory Management**: Array allocation, garbage collection, weak references
- **Cache Performance**: LRU cache, connection pool caching, memory pressure
- **Threading**: Thread pool scaling, producer-consumer patterns, synchronization
- **Memory Pressure**: Large dataset processing, leak detection, concurrent usage

```python
# Example: Benchmark thread pool scaling
@pytest.mark.benchmark(group="threading_performance")
@pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
def test_thread_pool_performance(benchmark, num_threads):
    result = benchmark.pedantic(test_thread_pool, rounds=5, iterations=1)
```

### 4. GUI Performance (`test_gui_performance.py`)

Tests user interface performance:

- **Plotting**: PyQtGraph vs matplotlib, plot creation/updates
- **Widgets**: Creation, tab switching, content updates
- **Responsiveness**: GUI under CPU load, event processing
- **Memory Usage**: Widget cleanup, plot memory efficiency

```python
# Example: Benchmark plot creation
@pytest.mark.benchmark(group="plotting_performance")
@pytest.mark.gui
def test_pyqtgraph_plot_creation_performance(benchmark, qapp, num_points):
    result = benchmark.pedantic(create_pyqtgraph_plot, rounds=10, iterations=3)
```

### 5. Scalability Testing (`test_scalability.py`)

Tests performance scaling characteristics:

- **Data Size Scaling**: Performance vs dataset size
- **Concurrency Scaling**: Parallel processing efficiency
- **Memory Scaling**: Memory usage patterns
- **Algorithm Complexity**: Computational complexity validation

```python
# Example: Test data size scaling
@pytest.mark.benchmark(group="data_size_scaling")
@pytest.mark.parametrize("size_category", ["tiny", "small", "medium", "large"])
def test_hdf5_read_scaling(benchmark, test_data_dir, size_category):
    result = benchmark.pedantic(read_full_dataset, rounds=5, iterations=2)
```

### 6. Optimization Validation (`test_optimization_validation.py`)

Validates performance optimization effectiveness:

- **HDF5 Optimizations**: Connection pooling, batch operations, memory mapping
- **Memory Optimizations**: Streaming vs batch processing, caching benefits
- **Threading Optimizations**: Parallel vs sequential processing
- **Algorithm Optimizations**: Vectorized operations, fitting improvements

```python
# Example: Validate connection pooling optimization
@pytest.mark.benchmark(group="optimization_validation")
def test_connection_pooling_optimization(benchmark, synthetic_hdf5_file):
    baseline_time = benchmark.pedantic(test_without_pooling, rounds=5)
    optimized_time = benchmark.pedantic(test_with_pooling, rounds=5)

    improvement = (baseline_time - optimized_time) / baseline_time
    assert improvement >= 0  # Should not degrade performance
```

## Regression Detection System

### Automated Regression Analysis

The framework automatically detects performance regressions by comparing current results with established baselines:

```python
from tests.performance.utils.regression_detection import (
    PerformanceRegressionDetector,
    BenchmarkResult
)

# Initialize detector with baseline
detector = PerformanceRegressionDetector(
    baseline_file=Path("tests/performance/config/baseline_performance.json"),
    regression_threshold=0.2  # 20% degradation threshold
)

# Analyze results
analyses = detector.analyze_all_results()
report = detector.generate_report(analyses)

# Check if CI should fail
should_fail, reason = detector.check_ci_failure_conditions(analyses)
```

### Regression Severity Levels

- **Critical**: >100% performance degradation
- **Major**: 50-100% performance degradation
- **Minor**: 20-50% performance degradation

### Statistical Analysis

The system performs statistical significance testing using:

- Coefficient of variation for result reliability
- Effect size calculations (Cohen's d)
- Confidence interval analysis
- Trend detection algorithms

## CI/CD Integration

### GitHub Actions Workflow

The framework integrates with GitHub Actions for automated testing:

```yaml
name: Performance Regression Testing

on:
  pull_request:
    branches: [ master, main ]
  push:
    branches: [ master, main ]
  schedule:
    - cron: '0 2 * * *'  # Nightly tests

jobs:
  performance-regression-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        test-category: ["hdf5", "scientific", "memory", "scalability"]
```

### Automated Reporting

The system generates multiple report formats:

- **JSON**: Machine-readable regression analysis
- **Markdown**: Human-readable GitHub PR comments
- **JUnit XML**: Integration with CI systems
- **HTML**: Visual performance dashboards

### Performance Gates

CI fails automatically on:

- Critical regressions (>100% slower)
- More than 3 major regressions
- Regression rate >30% of total tests

## Configuration

### Performance Test Configuration

```python
PERFORMANCE_CONFIG = {
    "default_rounds": 10,           # Number of benchmark rounds
    "warmup_rounds": 3,             # Number of warmup rounds
    "max_time": 30.0,               # Maximum time per benchmark (seconds)
    "regression_threshold": 0.2,     # 20% degradation threshold
    "statistical_confidence": 0.95,  # 95% confidence level
    "memory_threshold_mb": 1024,     # Memory usage threshold
}
```

### Test Data Sizes

```python
DATA_SIZES = {
    "tiny": {"width": 64, "height": 64, "frames": 100},
    "small": {"width": 128, "height": 128, "frames": 500},
    "medium": {"width": 256, "height": 256, "frames": 1000},
    "large": {"width": 512, "height": 512, "frames": 2000},
    "xlarge": {"width": 1024, "height": 1024, "frames": 5000},
}
```

## Advanced Usage

### Custom Benchmark Creation

```python
@pytest.mark.benchmark(group="custom_analysis")
@pytest.mark.parametrize("parameter", [1, 2, 4])
def test_custom_performance(benchmark, parameter):
    """Custom performance test."""

    def custom_operation():
        # Your code here
        result = expensive_computation(parameter)
        return result

    result = benchmark.pedantic(custom_operation, rounds=10, iterations=3)

    # Add metadata for analysis
    benchmark.extra_info = {
        'parameter_value': parameter,
        'custom_metric': result
    }

    assert result > 0
```

### Memory Profiling

```python
def test_memory_usage(benchmark, memory_profiler):
    """Test with memory monitoring."""

    def memory_intensive_operation():
        memory_profiler.start()

        # Your memory-intensive code
        data = np.random.random((1000, 1000))
        result = np.sum(data)

        memory_stats = memory_profiler.stop()
        return result, memory_stats['peak_mb']

    result, peak_memory = benchmark.pedantic(memory_intensive_operation, rounds=3)

    assert peak_memory < 100  # Memory threshold
```

### Scalability Analysis

```python
def test_scaling_analysis(benchmark, scalability_analyzer):
    """Analyze performance scaling."""

    for size in [100, 1000, 10000]:
        def scale_test():
            return algorithm_performance_test(size)

        time_result = benchmark.pedantic(scale_test, rounds=5)
        scalability_analyzer.add_result('algorithm_test', size, time_result)

    # Analyze scaling trend
    analysis = scalability_analyzer.analyze_scaling_trend('algorithm_test')
    assert analysis['best_fit'] in ['linear', 'logarithmic']  # Acceptable scaling
```

## Monitoring and Alerts

### Performance Trend Analysis

```python
from tests.performance.utils.regression_detection import PerformanceTrendAnalyzer

analyzer = PerformanceTrendAnalyzer(history_file="performance_history.json")

# Analyze trends
trend_analysis = analyzer.analyze_trends("test_hdf5_read_performance", days_back=30)

# Detect anomalies
anomalies = analyzer.detect_anomalies("test_hdf5_read_performance", sensitivity=2.0)

# Generate trend report
trend_report = analyzer.generate_trend_report()
```

### Setting Up Alerts

Configure GitHub Actions to notify on regressions:

```yaml
- name: Notify on Regression
  if: steps.regression-analysis.outputs.analysis_pass == 'false'
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    text: 'Performance regression detected in XPCS Toolkit'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

## Troubleshooting

### Common Issues

1. **Qt GUI Tests Failing**:
   ```bash
   # Install virtual display for headless testing
   sudo apt-get install xvfb
   xvfb-run -a pytest tests/performance/benchmarks/test_gui_performance.py
   ```

2. **Memory Tests Out of Memory**:
   ```bash
   # Reduce test data sizes or skip memory-intensive tests
   pytest tests/performance/ -m "not memory_intensive"
   ```

3. **Baseline Not Found**:
   ```bash
   # Create initial baseline
   pytest tests/performance/ --benchmark-only --benchmark-save=baseline
   ```

### Performance Debugging

```python
# Enable detailed logging
export XPCS_PERFORMANCE_DEBUG=1
export PYTHONPATH=/path/to/xpcs-toolkit

# Run with profiling
python -m cProfile -o profile.stats -m pytest tests/performance/benchmarks/test_hdf5_performance.py::test_file_open_close_performance
```

## Contributing

### Adding New Benchmarks

1. Create test in appropriate benchmark file
2. Use `@pytest.mark.benchmark` decorator
3. Add meaningful parametrization
4. Include `benchmark.extra_info` for analysis
5. Update baseline after validation

### Performance Optimization Guidelines

1. Always benchmark before and after optimizations
2. Use statistical significance testing
3. Consider multiple data sizes and conditions
4. Document optimization rationale
5. Update baselines when improvements are validated

## Integration with XPCS Toolkit

The performance tests are designed to work seamlessly with XPCS Toolkit's existing architecture:

- **File I/O**: Tests `xpcs_toolkit.fileIO` modules
- **Analysis**: Tests `xpcs_toolkit.module` scientific algorithms
- **GUI**: Tests `xpcs_toolkit.xpcs_viewer` components
- **Utils**: Tests `xpcs_toolkit.utils` performance utilities

## License

This performance testing framework is part of XPCS Toolkit and follows the same MIT license terms.