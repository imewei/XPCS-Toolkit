# XPCS Toolkit Enhanced Test Framework

This document provides comprehensive guidance for using the enhanced test framework for the XPCS Toolkit project. The enhanced framework includes advanced reliability testing, performance optimization, CI/CD integration, and sophisticated test data management.

## Table of Contents

1. [Framework Overview](#framework-overview)
2. [Installation and Setup](#installation-and-setup)
3. [Test Organization](#test-organization)
4. [Advanced Features](#advanced-features)
5. [Reliability Testing](#reliability-testing)
6. [Performance Optimization](#performance-optimization)
7. [CI/CD Integration](#cicd-integration)
8. [Test Data Management](#test-data-management)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Framework Overview

The XPCS Toolkit test framework has been enhanced with several advanced capabilities:

### Core Components

- **Test Isolation and Reliability** (`tests/utils/test_isolation.py`)
  - Comprehensive resource cleanup
  - Test environment isolation
  - Performance monitoring
  - Flaky test detection

- **Reliability Framework** (`tests/utils/test_reliability.py`)
  - Flakiness detection and analysis
  - Test stabilization with retry mechanisms
  - Resource lock management
  - Environment validation

- **Performance Optimization** (`tests/utils/test_performance.py`)
  - Real-time performance monitoring
  - Memory usage tracking
  - NumPy/scientific computing optimizations
  - Test benchmarking utilities

- **CI/CD Integration** (`tests/utils/ci_integration.py`)
  - Multi-platform CI environment detection
  - Automated report generation (JUnit XML, JSON, Markdown)
  - GitHub Actions integration
  - Artifact management

- **Advanced Data Management** (`tests/utils/test_data_management.py`)
  - Intelligent test data caching
  - Synthetic XPCS dataset generation
  - HDF5 test file management
  - Parametrized data fixtures

## Installation and Setup

### Prerequisites

```bash
# Install required dependencies
pip install pytest pytest-qt psutil h5py numpy scipy
```

### Environment Configuration

The test framework automatically detects and configures itself for different environments:

- **Local Development**: Full feature set enabled
- **CI/CD Environments**: Optimized for automated testing
- **Headless Environments**: GUI tests automatically skipped

### Framework Initialization

The framework is automatically initialized through `tests/conftest.py`. Key features are conditionally enabled based on available dependencies:

```python
# Framework availability flags
RELIABILITY_FRAMEWORKS_AVAILABLE
ADVANCED_DATA_MANAGEMENT_AVAILABLE
PERFORMANCE_OPTIMIZATION_AVAILABLE
CI_INTEGRATION_AVAILABLE
```

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                    # Main test configuration
├── FRAMEWORK_GUIDE.md            # This documentation
├── utils/                         # Framework utilities
│   ├── test_isolation.py         # Isolation and cleanup
│   ├── test_reliability.py       # Reliability framework
│   ├── test_performance.py       # Performance optimization
│   ├── ci_integration.py         # CI/CD integration
│   └── test_data_management.py   # Data management
├── unit/                          # Unit tests
├── integration/                   # Integration tests
├── performance/                   # Performance tests
├── error_handling/                # Error handling tests
├── gui_interactive/               # GUI tests
└── end_to_end/                    # End-to-end tests
```

### Test Markers

The framework supports comprehensive test categorization:

```python
# Core categories
@pytest.mark.unit               # Fast unit tests
@pytest.mark.integration        # Integration tests
@pytest.mark.performance        # Performance benchmarks
@pytest.mark.gui               # GUI tests (requires display)
@pytest.mark.slow              # Tests taking >1 second
@pytest.mark.scientific        # Scientific accuracy tests

# Reliability categories
@pytest.mark.flaky             # Known flaky tests
@pytest.mark.stress            # Stress tests
@pytest.mark.system_dependent  # System resource dependent
@pytest.mark.reliable          # Using reliability framework
```

### Running Tests

```bash
# Run all tests with full reporting
pytest -v

# Run specific test categories
pytest -m "unit or integration"      # Core tests only
pytest -m "not (stress or flaky)"    # Stable tests only
pytest -m "gui" --display=:0         # GUI tests with display

# Performance-focused runs
pytest -m "not slow"                 # Skip slow tests
pytest --benchmark-only              # Only benchmarks

# CI-safe execution
pytest -m "not (stress or system_dependent or flaky)"
```

## Advanced Features

### 1. Automated Performance Monitoring

All tests are automatically monitored for performance:

```python
# Automatic monitoring (no code changes needed)
def test_data_processing():
    # Test implementation
    pass

# Manual monitoring for detailed analysis
@monitor_performance
def test_intensive_operation():
    # Performance metrics automatically collected
    pass
```

### 2. Test Environment Validation

Validate system requirements before running tests:

```python
@validate_test_environment(memory_mb=1000, disk_mb=100)
def test_large_dataset():
    # Test only runs if system meets requirements
    pass

# Convenience decorators
@require_memory(500)  # Requires 500MB RAM
@require_network()    # Requires network connectivity
@require_display()    # Requires GUI display
def test_gui_feature():
    pass
```

### 3. Resource Management

Prevent test interference with resource locking:

```python
@reliable_test(required_resources=['gpu', 'large_memory'])
def test_gpu_computation():
    # Exclusive access to specified resources
    pass
```

### 4. Flaky Test Handling

Built-in flaky test detection and stabilization:

```python
@reliable_test(max_retries=3, timeout=10.0, deterministic_timing=True)
def test_timing_sensitive_operation():
    # Automatic retry with backoff
    # Deterministic timing mocking
    # Timeout protection
    pass
```

## Reliability Testing

### Flakiness Detection

The framework automatically tracks test reliability:

```python
# Access flakiness detector
def test_with_flakiness_tracking(flakiness_detector):
    detector = flakiness_detector

    # Test results are automatically tracked
    # View flaky tests at session end
    flaky_tests = detector.get_flaky_tests()
```

### Test Stabilization

Use the `@reliable_test` decorator for comprehensive stabilization:

```python
from tests.utils.test_reliability import reliable_test

@reliable_test(
    max_retries=3,                    # Retry failed tests
    timeout=30.0,                     # Timeout protection
    deterministic_timing=True,        # Mock time functions
    required_resources=['database']   # Resource locking
)
def test_database_integration():
    # Highly reliable test execution
    pass
```

### Manual Retry Logic

For custom retry scenarios:

```python
from tests.utils.test_isolation import retry_on_failure

@retry_on_failure(max_retries=5, delay=0.1)
def test_external_service():
    # Custom retry logic with exponential backoff
    pass
```

## Performance Optimization

### Automatic Optimizations

The framework automatically applies performance optimizations:

- **NumPy Threading**: Optimal thread counts for test environment
- **Memory Management**: Aggressive garbage collection
- **I/O Optimization**: Reduced buffer sizes for faster tests

### Manual Performance Control

```python
from tests.utils.test_performance import (
    optimize_for_speed, memory_limit, timeout_limit
)

@optimize_for_speed
@memory_limit(100)  # 100MB limit
@timeout_limit(5)   # 5 second timeout
def test_performance_critical():
    # Optimized test execution
    pass
```

### Benchmarking

```python
def test_algorithm_performance(benchmark_tool):
    def algorithm_to_test():
        # Implementation to benchmark
        pass

    results = benchmark_tool(algorithm_to_test, iterations=100)
    assert results['mean_time'] < 0.001  # 1ms requirement
```

### Performance Monitoring

Access detailed performance metrics:

```python
def test_with_monitoring(performance_monitor):
    monitor = performance_monitor

    initial_state = monitor.start_monitoring('custom_test')
    # Test implementation
    metrics = monitor.stop_monitoring(initial_state)

    print(f"Duration: {metrics.duration:.3f}s")
    print(f"Peak Memory: {metrics.memory_peak_mb:.1f}MB")
```

## CI/CD Integration

### Automatic CI Detection

The framework automatically detects CI environments:

- GitHub Actions
- GitLab CI
- Jenkins
- CircleCI
- Travis CI
- Azure Pipelines

### CI Configuration

```python
def test_ci_aware(ci_environment):
    if ci_environment['is_ci']:
        # CI-specific test behavior
        pytest.mark.timeout(30)  # Shorter timeout in CI
    else:
        # Local development behavior
        pass
```

### GitHub Actions Integration

```python
def test_github_actions(github_actions_integration):
    if github_actions_integration:
        # Set custom outputs
        github_actions_integration.set_output('test_status', 'passed')

        # Add to step summary
        github_actions_integration.add_step_summary('## Custom Results\\nAll tests passed!')
```

### Report Generation

The framework automatically generates CI reports:

- **JUnit XML**: For test result parsing
- **JSON Reports**: Detailed test analytics
- **Markdown Summaries**: Human-readable results
- **Coverage Badges**: Visual status indicators

## Test Data Management

### Intelligent Data Generation

Create sophisticated test datasets:

```python
from tests.utils.test_data_management import TestDataSpec

# Define data specification
spec = TestDataSpec(
    data_type='xpcs',
    shape=(512, 512),
    size_mb=10.0,
    noise_level=0.02,
    seed=42,
    metadata={'n_frames': 100}
)

def test_with_custom_data(advanced_data_factory):
    factory = advanced_data_factory
    data = factory.create_data(spec)
    # Data is cached automatically
```

### HDF5 Test Files

Generate comprehensive test files:

```python
def test_hdf5_processing(advanced_xpcs_hdf5):
    hdf5_file = advanced_xpcs_hdf5  # Comprehensive test file
    # File includes qmap, correlation, SAXS, two-time data
    # Automatic cleanup after test
```

### Parametrized Data

Test with various data configurations:

```python
def test_data_sizes(sized_test_data):
    # Automatically parametrized with different sizes:
    # small (1MB), medium (5MB), large (10MB)
    data = sized_test_data
    # Test implementation
```

### Data Caching

The framework intelligently caches test data:

- **Hash-based Caching**: Identical specifications reuse data
- **Size Management**: Automatic cleanup when cache exceeds limits
- **Performance Optimization**: Faster test execution

## Best Practices

### 1. Test Organization

```python
# Good: Descriptive test names with categories
@pytest.mark.unit
@pytest.mark.scientific
def test_correlation_function_monotonic_decay():
    \"\"\"Test that correlation function exhibits monotonic decay.\"\"\"
    pass

# Good: Use appropriate markers
@pytest.mark.slow
@pytest.mark.integration
def test_full_xpcs_analysis_pipeline():
    \"\"\"Integration test for complete XPCS analysis.\"\"\"
    pass
```

### 2. Reliability

```python
# Good: Use reliability decorators for flaky tests
@reliable_test(max_retries=2, timeout=10.0)
def test_external_dependency():
    \"\"\"Test that may fail due to external factors.\"\"\"
    pass

# Good: Validate environment requirements
@require_memory(1000)
@require_network()
def test_large_dataset_download():
    \"\"\"Test requiring significant resources.\"\"\"
    pass
```

### 3. Performance

```python
# Good: Set appropriate performance limits
@memory_limit(500)  # 500MB limit
@timeout_limit(30)  # 30 second limit
def test_memory_intensive_operation():
    \"\"\"Test with defined resource limits.\"\"\"
    pass

# Good: Use performance monitoring for optimization
@monitor_performance
def test_algorithm_efficiency():
    \"\"\"Monitor performance for optimization opportunities.\"\"\"
    pass
```

### 4. Data Management

```python
# Good: Use advanced fixtures for complex data needs
def test_xpcs_analysis(realistic_xpcs_dataset):
    data = realistic_xpcs_dataset  # Cached, comprehensive dataset
    # Test implementation

# Good: Specify data requirements clearly
def test_correlation_fitting(minimal_test_dataset):
    data = minimal_test_dataset   # Fast, minimal dataset
    # Test implementation
```

## Troubleshooting

### Common Issues

#### 1. Framework Not Available

**Problem**: Reliability frameworks not available
```
RELIABILITY_FRAMEWORKS_AVAILABLE = False
```

**Solution**: Check for import conflicts, especially with `logging` module
```bash
# Rename conflicting directories
mv tests/logging tests/logging_tests
```

#### 2. Memory Errors

**Problem**: Tests failing with memory errors
```python
MemoryError: Test exceeded memory limit: 150.0MB > 100.0MB
```

**Solution**: Adjust memory limits or optimize test data
```python
@memory_limit(200)  # Increase limit
def test_large_operation():
    pass

# Or use smaller test datasets
def test_with_minimal_data(minimal_test_dataset):
    pass
```

#### 3. Flaky Tests

**Problem**: Tests passing individually but failing in suite

**Solution**: Use reliability framework
```python
@reliable_test(max_retries=3, required_resources=['exclusive_resource'])
def test_flaky_operation():
    pass
```

#### 4. CI/CD Issues

**Problem**: Tests behaving differently in CI

**Solution**: Use CI-aware testing
```python
def test_ci_aware(ci_environment):
    if ci_environment['is_ci']:
        # Use stable, deterministic behavior
        pytest.skip_if_no_resource()
    else:
        # Full testing in development
        pass
```

### Performance Issues

#### Slow Test Execution

1. **Use appropriate test categories**:
   ```bash
   pytest -m "not slow"  # Skip slow tests during development
   ```

2. **Monitor performance**:
   ```python
   @monitor_performance
   def test_slow_operation():
       pass
   ```

3. **Optimize data sizes**:
   ```python
   # Use smaller datasets for unit tests
   def test_algorithm(minimal_test_dataset):
       pass
   ```

#### Memory Issues

1. **Set memory limits**:
   ```python
   @memory_limit(100)  # Prevent runaway memory usage
   def test_memory_operation():
       pass
   ```

2. **Use isolation manager**:
   ```python
   def test_with_cleanup(isolation_manager):
       # Automatic resource cleanup
       pass
   ```

### Getting Help

1. **Check framework status**:
   ```python
   def test_framework_status():
       print(f"Reliability: {RELIABILITY_FRAMEWORKS_AVAILABLE}")
       print(f"Performance: {PERFORMANCE_OPTIMIZATION_AVAILABLE}")
       print(f"Data Management: {ADVANCED_DATA_MANAGEMENT_AVAILABLE}")
       print(f"CI Integration: {CI_INTEGRATION_AVAILABLE}")
   ```

2. **Enable verbose reporting**:
   ```bash
   pytest -v --tb=long
   ```

3. **Generate performance reports**:
   ```bash
   pytest --performance-report
   ```

## Framework Development

### Contributing to the Framework

When extending the test framework:

1. **Add new utilities to appropriate modules**:
   - `test_reliability.py`: Reliability and stabilization features
   - `test_performance.py`: Performance monitoring and optimization
   - `test_data_management.py`: Data generation and management
   - `ci_integration.py`: CI/CD platform support

2. **Update `conftest.py`** for new fixtures and configuration

3. **Document new features** in this README

4. **Add tests** for framework components in `tests/framework/`

### Custom Extensions

Create project-specific extensions:

```python
# tests/utils/custom_extensions.py
from tests.utils.test_reliability import reliable_test

def xpcs_reliable_test(**kwargs):
    \"\"\"XPCS-specific reliable test decorator.\"\"\"
    defaults = {
        'max_retries': 2,
        'timeout': 30.0,
        'required_resources': ['hdf5_file']
    }
    defaults.update(kwargs)
    return reliable_test(**defaults)
```

This enhanced test framework provides a robust, scalable foundation for testing the XPCS Toolkit while maintaining scientific accuracy and reliability standards.