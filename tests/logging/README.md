# XPCS Toolkit Logging Tests

This directory contains all logging-related tests for the XPCS Toolkit, organized by test category to provide clear separation of concerns and easier maintenance.

## Directory Structure

```
tests/logging/
├── __init__.py                 # Package initialization
├── README.md                   # This documentation
├── conftest.py                 # Common fixtures and utilities
├── functional/                 # Comprehensive functional tests
│   ├── __init__.py
│   └── test_logging_system.py  # Main functional test suite (1,036 lines)
├── performance/                # Performance benchmarks and optimization tests
│   ├── __init__.py
│   ├── test_logging_benchmarks.py  # Performance benchmark suite (1,106 lines)
│   └── run_benchmarks.py           # Benchmark runner script (125 lines)
├── properties/                 # Property-based testing with Hypothesis
│   ├── __init__.py
│   └── test_logging_properties.py  # Property-based test suite (1,532 lines)
├── unit/                       # Unit tests for logging components
│   ├── __init__.py
│   └── test_logging_config.py      # Logging configuration unit tests
└── utils/                      # Common test utilities (reserved for future use)
    └── __init__.py
```

## Test Categories

### Functional Tests (`functional/`)
- **File**: `test_logging_system.py`
- **Purpose**: Comprehensive functional validation of the logging system
- **Features**:
  - Environment-based configuration testing
  - Multi-threading and concurrency validation
  - File and console handler testing
  - Memory leak detection
  - Error handling and recovery
  - Integration with XPCS Toolkit components

### Performance Tests (`performance/`)
- **Files**: `test_logging_benchmarks.py`, `run_benchmarks.py`
- **Purpose**: Performance benchmarks and optimization validation
- **Features**:
  - Throughput and latency measurements
  - Memory usage profiling
  - Concurrent logging performance
  - Statistical analysis with confidence intervals
  - Performance regression detection
  - Scientific computing workload simulation

### Property-Based Tests (`properties/`)
- **File**: `test_logging_properties.py`
- **Purpose**: Mathematical property validation using Hypothesis
- **Features**:
  - Message integrity and ordering properties
  - Format consistency validation
  - Configuration invariant testing
  - Edge case generation and testing
  - Statistical property validation
  - Thread-safety property verification

### Unit Tests (`unit/`)
- **File**: `test_logging_config.py`
- **Purpose**: Isolated unit tests for logging configuration components
- **Features**:
  - Configuration class testing
  - Environment variable handling
  - Singleton pattern validation
  - Thread-safe initialization

## Running Tests

### Run All Logging Tests
```bash
# From project root
python -m pytest tests/logging/ -v

# With coverage
python -m pytest tests/logging/ --cov=xpcs_toolkit.utils.logging_config --cov-report=html
```

### Run Specific Test Categories

#### Functional Tests
```bash
python -m pytest tests/logging/functional/ -v
```

#### Performance Benchmarks
```bash
python -m pytest tests/logging/performance/test_logging_benchmarks.py --benchmark-only

# Or use the benchmark runner
python tests/logging/performance/run_benchmarks.py --quick
python tests/logging/performance/run_benchmarks.py --report
```

#### Property-Based Tests
```bash
python -m pytest tests/logging/properties/ -v

# With increased examples for thorough testing
python -m pytest tests/logging/properties/ -v --hypothesis-show-statistics
```

#### Unit Tests
```bash
python -m pytest tests/logging/unit/ -v
```

### Performance Testing

The performance tests include statistical analysis and can generate detailed reports:

```bash
# Quick benchmark run (fewer iterations)
python tests/logging/performance/run_benchmarks.py --quick

# Full benchmark with report generation
python tests/logging/performance/run_benchmarks.py --report

# Benchmark with pytest-benchmark
python -m pytest tests/logging/performance/ --benchmark-only --benchmark-sort=mean
```

## Common Fixtures

The `conftest.py` file provides shared fixtures used across all logging tests:

- **`clean_logging_environment`**: Cleans environment variables and logging state
- **`clean_logging_state`**: Ensures clean logging state with temporary directory
- **`logging_temp_dir`**: Session-scoped temporary directory for log files
- **`isolated_logger`**: Creates isolated logger instances for testing
- **`log_capture`**: Utility for capturing and analyzing log messages
- **`memory_handler`**: Memory-based handler for message collection
- **`temp_log_file`**: Temporary log file for file handler testing

## Test Configuration

### Environment Variables
The tests respect and validate these environment variables:
- `PYXPCS_LOG_LEVEL`: Logging level configuration
- `PYXPCS_LOG_DIR`: Log directory location
- `PYXPCS_LOG_FILE`: Log file name
- `PYXPCS_LOG_FORMAT`: Log message format
- `PYXPCS_LOG_MAX_SIZE`: Maximum log file size
- `PYXPCS_LOG_BACKUP_COUNT`: Number of backup files
- `PYXPCS_LOG_DISABLE_FILE`: Disable file logging
- `PYXPCS_LOG_DISABLE_CONSOLE`: Disable console logging
- `PYXPCS_SUPPRESS_QT_WARNINGS`: Suppress Qt connection warnings

### Pytest Marks
Common pytest marks used in logging tests:
- `@pytest.mark.slow`: For long-running tests (especially benchmarks)
- `@pytest.mark.benchmark`: For performance benchmark tests
- `@pytest.mark.hypothesis`: For property-based tests
- `@pytest.mark.integration`: For integration tests
- `@pytest.mark.memory`: For memory usage tests

## Development Guidelines

### Adding New Tests

1. **Choose the appropriate category**:
   - Functional: End-to-end behavior testing
   - Performance: Benchmark and optimization testing
   - Properties: Mathematical invariant testing
   - Unit: Isolated component testing

2. **Use common fixtures**: Leverage the fixtures in `conftest.py` to maintain consistency

3. **Follow naming conventions**:
   - Test files: `test_*.py`
   - Test classes: `Test*`
   - Test methods: `test_*`

4. **Add appropriate pytest marks** for test categorization

### Test Isolation

- Each test should be completely isolated and not depend on other tests
- Use the `clean_logging_environment` fixture for environment isolation
- Use temporary directories for file-based tests
- Reset logging configuration after each test

### Performance Considerations

- Performance tests should include statistical analysis
- Use appropriate sample sizes for reliable measurements
- Include confidence intervals in benchmark results
- Document expected performance characteristics

## Maintenance

### Regular Tasks
- Review test coverage periodically
- Update benchmarks when performance characteristics change
- Ensure property-based tests cover new edge cases
- Validate integration with new XPCS Toolkit components

### When Adding New Logging Features
1. Add corresponding functional tests
2. Include performance impact assessment
3. Validate mathematical properties if applicable
4. Update unit tests for configuration changes
5. Update this documentation

## Dependencies

Key testing dependencies for logging tests:
- `pytest`: Test framework
- `pytest-benchmark`: Performance benchmarking
- `hypothesis`: Property-based testing
- `psutil`: Memory and system monitoring
- `numpy`: Scientific computing simulation

## Integration with CI/CD

The logging tests are integrated with the project's continuous integration:
- Functional tests run on every commit
- Performance benchmarks run nightly
- Property-based tests with extended examples run weekly
- All tests must pass before merge to main branch

For more information about the XPCS Toolkit logging system, see the main project documentation.