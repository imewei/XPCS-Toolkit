# XPCS Toolkit Test Framework Quick Reference

## Common Test Patterns

### Basic Test Structure
```python
import pytest

@pytest.mark.unit
def test_basic_functionality():
    """Test basic functionality with clear description."""
    assert True
```

### Reliability Testing
```python
from tests.utils.test_reliability import reliable_test

@reliable_test(max_retries=3, timeout=10.0)
def test_flaky_operation():
    """Test with automatic retry and timeout."""
    pass
```

### Performance Testing
```python
from tests.utils.test_performance import monitor_performance, memory_limit

@monitor_performance
@memory_limit(100)  # 100MB limit
def test_performance_critical():
    """Performance-monitored test with memory limit."""
    pass
```

### Data Testing
```python
def test_with_data(advanced_xpcs_hdf5, realistic_xpcs_dataset):
    """Test with comprehensive XPCS data."""
    hdf5_file = advanced_xpcs_hdf5
    dataset = realistic_xpcs_dataset
    # Test implementation
```

### CI-Aware Testing
```python
def test_ci_behavior(ci_environment):
    """Test that behaves differently in CI."""
    if ci_environment['is_ci']:
        # Simplified CI behavior
        pass
    else:
        # Full local testing
        pass
```

## Test Markers

| Marker | Purpose | Usage |
|--------|---------|-------|
| `@pytest.mark.unit` | Unit tests | Fast, isolated tests |
| `@pytest.mark.integration` | Integration tests | Component interaction |
| `@pytest.mark.performance` | Performance tests | Benchmarks and timing |
| `@pytest.mark.gui` | GUI tests | Requires display |
| `@pytest.mark.slow` | Slow tests | Takes >1 second |
| `@pytest.mark.flaky` | Flaky tests | Known reliability issues |
| `@pytest.mark.stress` | Stress tests | Resource intensive |
| `@pytest.mark.system_dependent` | System tests | Requires specific resources |

## Running Tests

### Development
```bash
# Fast feedback loop
pytest -m "unit"

# Skip slow tests
pytest -m "not slow"

# Only stable tests
pytest -m "not (flaky or stress)"
```

### CI/CD
```bash
# CI-safe test run
pytest -m "not (stress or system_dependent or flaky)"

# With reporting
pytest --junitxml=test-results.xml
```

### Performance Analysis
```bash
# Monitor performance
pytest -v --tb=short

# Generate reports
pytest --performance-report
```

## Fixtures Quick Guide

### Core Fixtures
- `temp_dir`: Temporary directory
- `random_seed`: Fixed random seed (42)
- `performance_timer`: Time measurement
- `test_logger`: Test logging

### Data Fixtures
- `minimal_test_dataset`: Small, fast dataset
- `realistic_xpcs_dataset`: Realistic XPCS data
- `advanced_xpcs_hdf5`: Comprehensive HDF5 file
- `sized_test_data`: Parametrized data sizes

### Advanced Fixtures
- `isolation_manager`: Resource cleanup
- `performance_monitor`: Performance tracking
- `ci_environment`: CI environment info
- `flakiness_detector`: Flaky test detection

## Decorators

### Reliability
```python
from tests.utils.test_reliability import (
    reliable_test, require_memory, require_network
)

@reliable_test(max_retries=3)
@require_memory(500)  # 500MB
@require_network()
def test_reliable():
    pass
```

### Performance
```python
from tests.utils.test_performance import (
    optimize_for_speed, memory_limit, timeout_limit
)

@optimize_for_speed
@memory_limit(100)    # 100MB
@timeout_limit(30)    # 30 seconds
def test_optimized():
    pass
```

### Environment Validation
```python
from tests.utils.test_reliability import validate_test_environment

@validate_test_environment(
    memory_mb=1000,
    disk_mb=100,
    network=True
)
def test_with_requirements():
    pass
```

## Framework Status Check

```python
def test_framework_status():
    from tests.conftest import (
        RELIABILITY_FRAMEWORKS_AVAILABLE,
        PERFORMANCE_OPTIMIZATION_AVAILABLE,
        ADVANCED_DATA_MANAGEMENT_AVAILABLE,
        CI_INTEGRATION_AVAILABLE
    )

    print(f"Reliability: {RELIABILITY_FRAMEWORKS_AVAILABLE}")
    print(f"Performance: {PERFORMANCE_OPTIMIZATION_AVAILABLE}")
    print(f"Data Management: {ADVANCED_DATA_MANAGEMENT_AVAILABLE}")
    print(f"CI Integration: {CI_INTEGRATION_AVAILABLE}")
```

## Error Troubleshooting

| Error | Solution |
|-------|----------|
| `RELIABILITY_FRAMEWORKS_AVAILABLE = False` | Check for `logging` module conflicts |
| `MemoryError: exceeded limit` | Increase `@memory_limit()` or use smaller data |
| `TimeoutError: exceeded time limit` | Increase `@timeout_limit()` or optimize code |
| `Test failed in CI but passes locally` | Use `@reliable_test()` or check CI environment |
| `ImportError: cannot import h5py` | Install dependencies: `pip install h5py` |

## Performance Optimization Tips

1. **Use appropriate data sizes**:
   ```python
   # Good for unit tests
   def test_algorithm(minimal_test_dataset):
       pass

   # Good for integration tests
   def test_workflow(realistic_xpcs_dataset):
       pass
   ```

2. **Set resource limits**:
   ```python
   @memory_limit(100)
   @timeout_limit(10)
   def test_bounded():
       pass
   ```

3. **Monitor performance**:
   ```python
   @monitor_performance
   def test_to_optimize():
       pass
   ```

4. **Use caching**:
   ```python
   # Data is automatically cached
   def test_repeated_data(advanced_data_factory):
       # Same spec = cached data
       pass
   ```

## Best Practices Summary

1. **Always use descriptive test names and docstrings**
2. **Apply appropriate markers for test categorization**
3. **Use reliability decorators for flaky tests**
4. **Set performance limits for resource-intensive tests**
5. **Validate environment requirements upfront**
6. **Use advanced fixtures for complex data needs**
7. **Monitor performance for optimization opportunities**
8. **Write CI-aware tests for better portability**

## Getting More Help

- **Full Documentation**: `tests/FRAMEWORK_GUIDE.md`
- **Framework Source**: `tests/utils/`
- **Example Tests**: `tests/unit/`, `tests/integration/`
- **Performance Reports**: Generated automatically during test runs