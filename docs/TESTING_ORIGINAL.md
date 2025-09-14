# XPCS Toolkit Testing System

This document provides comprehensive guidance for testing in XPCS Toolkit, including unit tests, property-based testing, performance benchmarks, and the validation framework.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Test Commands](#test-commands)
3. [Property-Based Testing](#property-based-testing)
4. [Performance Benchmarks](#performance-benchmarks)
5. [Validation Framework](#validation-framework)
6. [Writing Tests](#writing-tests)
7. [Best Practices](#best-practices)

## Quick Start

### Running Basic Tests

```bash
# Run basic unit tests
make test

# Run all tests across Python versions
make test-all

# Run with coverage
make coverage
```

### Running Specific Test Categories

```bash
# Property-based tests
make test-properties

# Performance benchmarks  
make test-benchmarks

# Test validation
make test-validation
```

## Test Commands

### Basic Test Commands

```bash
make test                # Run basic unit tests quickly
make test-all           # Run tests on every Python version with tox
make test-unit          # Run unit and integration tests for logging system
```

### Property-Based Testing

```bash
make test-properties        # Run property-based tests using Hypothesis
make test-quick            # Run quick property tests (50 examples per property)
make test-comprehensive    # Run comprehensive property tests (500 examples per property)
```

### Performance & Benchmarks

```bash
make test-benchmarks       # Run performance benchmarks and regression tests
make test-performance      # Run performance benchmarks with detailed reporting
make test-memory          # Run memory usage and performance tests
```

### Specialized Test Categories

```bash
make test-scientific      # Run scientific computing specific tests
make test-concurrency     # Run concurrency and thread safety tests
make test-logging         # Run complete logging system test suite
```

### Quality Assurance & Validation

```bash
make test-validation      # Validate test suite quality and completeness
make test-mutation        # Run mutation testing to validate test effectiveness
make test-ci              # Run tests suitable for CI/CD environments
make test-full            # Run complete test suite including validation
```

### Coverage Analysis

```bash
make coverage            # Generate test coverage report
make coverage-html       # Generate HTML coverage report
make coverage-report     # Display coverage report in terminal
```

### Individual Test Execution

```bash
# Run specific test file
python -m pytest tests/test_logging_system.py -v

# Run specific test method
python -m pytest tests/test_xpcs_toolkit.py::TestXpcsToolkit::test_package_version -v

# Run tests with specific markers
python -m pytest -m "not slow" -v
python -m pytest -k "benchmark" --benchmark-only
```

## Property-Based Testing

### Overview

Property-based testing validates fundamental mathematical properties and invariants using Hypothesis to generate test cases automatically.

### Property Categories Tested

#### 1. Message Integrity Properties
- **Message Preservation**: Every logged message appears in output exactly once
- **Chronological Ordering**: Messages maintain temporal order within single thread
- **Character Encoding**: All valid UTF-8 strings are handled correctly
- **Size Bounds**: Arbitrary message lengths within reasonable limits are supported

#### 2. Mathematical Properties
- **Associativity**: (A + B) + C = A + (B + C) for log message sequences
- **Monotonicity**: Timestamp ordering is always increasing
- **Homogeneity**: Scaling properties of performance metrics

#### 3. Concurrency Properties
- **Thread Safety**: No data races under arbitrary thread interleavings
- **Atomicity**: Individual log operations are atomic
- **Progress**: System makes progress under contention

#### 4. Performance Properties
- **Latency Distribution**: Response times follow expected statistical properties
- **Throughput Scaling**: Performance scales predictably with load
- **Memory Usage**: Memory consumption remains within bounds

### Writing Property-Based Tests

```python
from hypothesis import given, strategies as st
import pytest

@given(messages=st.lists(st.text(), min_size=1, max_size=100))
def test_message_preservation_property(self, messages):
    """Test that all messages are preserved in logging."""
    logger = get_logger("test")
    
    # Log all messages
    for msg in messages:
        logger.info(msg)
    
    # Verify all messages appear in output
    log_output = capture_logs()
    for msg in messages:
        assert msg in log_output

@given(timestamps=st.lists(st.floats(min_value=0, max_value=1e9), min_size=2))
def test_monotonicity_property(self, timestamps):
    """Test that timestamps are monotonically increasing."""
    sorted_timestamps = sorted(timestamps)
    
    # Verify monotonic property
    for i in range(1, len(sorted_timestamps)):
        assert sorted_timestamps[i] >= sorted_timestamps[i-1]
```

### Property Test Configuration

```python
# Configure Hypothesis for scientific computing
from hypothesis import settings, HealthCheck

@settings(
    max_examples=500,           # Comprehensive testing
    deadline=2000,             # 2 second deadline per example
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True           # Reproducible tests
)
@given(data=st.floats(min_value=0.0, max_value=1e6))
def test_numerical_stability(self, data):
    # Test numerical stability properties
    pass
```

## Performance Benchmarks

### Benchmark Categories

#### 1. Throughput Benchmarks
Measure message processing rates under different scenarios:
- Console vs file logging
- Different log levels
- JSON vs text formatting
- Multi-threaded scenarios

#### 2. Latency Benchmarks  
Per-message timing analysis with statistical rigor:
- Message formatting time
- I/O operation latency
- Handler processing time
- Statistical distribution analysis

#### 3. Memory Benchmarks
Memory usage patterns and leak detection:
- Baseline memory consumption
- Memory scaling with load
- Garbage collection behavior
- Memory leak detection

#### 4. Scientific Computing Benchmarks
Domain-specific performance validation:
- Large array logging
- Scientific notation formatting
- Complex data structure serialization
- NumPy integration performance

### Running Benchmarks

```bash
# Run all benchmarks
python run_logging_benchmarks.py

# Quick benchmark subset
python run_logging_benchmarks.py --quick

# Generate detailed reports
python run_logging_benchmarks.py --report

# Run specific benchmark categories
python -m pytest tests/test_logging_benchmarks.py -k "throughput" --benchmark-only
python -m pytest tests/test_logging_benchmarks.py -k "latency" --benchmark-only
python -m pytest tests/test_logging_benchmarks.py -k "memory" --benchmark-only
```

### Benchmark Results Interpretation

Key metrics to understand:
- **Min/Max**: Fastest and slowest execution times
- **Mean/Median**: Average and middle values  
- **StdDev**: Performance consistency (lower is better)
- **OPS**: Operations per second (throughput)
- **IQR**: Interquartile range (statistical spread)

### Writing Performance Tests

```python
import pytest
from xpcs_toolkit.utils.logging_config import get_logger

def test_throughput_console_logging(benchmark):
    """Benchmark console logging throughput."""
    logger = get_logger("benchmark")
    
    def log_messages():
        for i in range(1000):
            logger.info("Test message %d", i)
    
    # Run benchmark
    result = benchmark(log_messages)
    
    # Validate performance requirements
    assert result.stats.mean < 0.1  # Less than 100ms for 1000 messages

@pytest.mark.parametrize("message_count", [100, 1000, 10000])
def test_memory_scaling(benchmark, message_count):
    """Test memory usage scaling with message count."""
    logger = get_logger("memory_test")
    
    def log_batch():
        for i in range(message_count):
            logger.info("Memory test message %d with data", i)
    
    # Measure memory usage
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss
    
    benchmark(log_batch)
    
    memory_after = process.memory_info().rss
    memory_delta_mb = (memory_after - memory_before) / 1024 / 1024
    
    # Validate memory usage is reasonable
    assert memory_delta_mb < message_count * 0.001  # Less than 1KB per message
```

## Validation Framework

### Overview

The validation framework provides meta-testing that validates the quality, completeness, and effectiveness of the entire test suite.

### Validation Categories

#### 1. Test Coverage Analysis (Weight: 30%)
- **Function Coverage**: Every public function has tests
- **Branch Coverage**: All code paths tested
- **Edge Case Coverage**: Boundary conditions validated  
- **Integration Coverage**: Module interactions tested

#### 2. Test Quality Metrics (Weight: 25%)
- **Assertion Density**: Adequate assertions per test
- **Test Independence**: No inter-test dependencies
- **Test Determinism**: Consistent, reproducible results
- **Test Performance**: Reasonable execution times
- **Test Maintainability**: Readable, well-documented tests

#### 3. Scientific Rigor Validation (Weight: 25%)
- **Statistical Significance**: Proper statistical validation in benchmarks
- **Numerical Precision**: Appropriate floating-point tolerances
- **Hypothesis Testing**: Comprehensive property-based testing
- **Baseline Management**: Current, valid performance baselines
- **Reproducibility**: Full cross-environment reproducibility

#### 4. Test Suite Completeness (Weight: 20%)
- **Requirement Coverage**: All requirements have corresponding tests
- **Scenario Coverage**: All usage scenarios tested
- **Error Path Coverage**: All error conditions tested
- **Documentation Coverage**: All examples in docs are tested

### Running Validation

```bash
# Run complete test suite validation
make test-validation

# Generate validation report
python tests/run_validation.py --report

# Validate specific aspects
python tests/run_validation.py --coverage-only
python tests/run_validation.py --quality-only
```

### Validation Results

The framework generates detailed reports including:
- **Overall Score**: Weighted composite score (0-100)
- **Category Breakdown**: Scores for each validation category
- **Recommendations**: Specific improvements to make
- **Trends**: Historical progression of test quality

### Test Infrastructure Validation

The framework also validates:
- **Fixture Quality**: Test fixtures are well-designed and reusable
- **Mock Objects**: Mocks accurately represent real objects
- **Test Data**: Test data is representative and comprehensive
- **Test Environment**: Test environment is correctly configured

## Writing Tests

### Unit Test Template

```python
import unittest
import tempfile
import os
from xpcs_toolkit.utils.logging_config import get_logger

class TestMyModule(unittest.TestCase):
    """Tests for MyModule functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.logger = get_logger(__name__)
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info("Test setup completed for %s", self._testMethodName)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.logger.info("Test cleanup completed for %s", self._testMethodName)

    def test_basic_functionality(self):
        """Test basic functionality."""
        self.logger.debug("Testing basic functionality")
        
        # Test implementation
        result = my_function("test input")
        
        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.status, "success")
        
        self.logger.info("Basic functionality test passed")

    def test_error_conditions(self):
        """Test error handling."""
        self.logger.debug("Testing error conditions")
        
        with self.assertRaises(ValueError):
            my_function(invalid_input)
        
        self.logger.info("Error conditions test passed")
```

### Integration Test Pattern

```python
class TestModuleIntegration(unittest.TestCase):
    """Integration tests for module interactions."""

    def test_logging_system_integration(self):
        """Test integration with logging system."""
        from xpcs_toolkit.utils.logging_config import get_logger
        
        logger = get_logger("integration_test")
        
        # Test that logging works correctly
        logger.info("Integration test message")
        
        # Verify log output
        # (Implementation depends on your log capture mechanism)
        
    def test_gui_integration(self):
        """Test GUI component integration."""
        # Set up headless GUI testing
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'
        
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        
        # Test GUI components
        # ...
        
        app.quit()
```

### Test Utilities

```python
# Common test utilities
def create_test_data():
    """Create standardized test data."""
    import numpy as np
    return np.random.rand(100, 100)

def capture_log_output(logger_name):
    """Capture log output for testing."""
    import logging
    import io
    
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    
    return log_capture

def assert_log_contains(log_output, expected_message):
    """Assert that log output contains expected message."""
    log_content = log_output.getvalue()
    assert expected_message in log_content, f"Expected '{expected_message}' not found in logs"
```

## Best Practices

### 1. Test Organization
- Group related tests in classes
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern

### 2. Test Independence
- Each test should be independent and not rely on other tests
- Use setUp() and tearDown() methods for test fixtures
- Avoid shared mutable state between tests

### 3. Test Data Management
- Use temporary directories for file-based tests
- Create minimal test data that exercises the functionality
- Use parameterized tests for multiple input scenarios

### 4. Assertion Best Practices
```python
# Good: Specific assertions with helpful messages
self.assertEqual(result.count, 5, "Expected 5 items in result")
self.assertAlmostEqual(result.value, 3.14159, places=5)

# Good: Context managers for exception testing
with self.assertRaises(ValueError) as context:
    invalid_operation()
self.assertIn("invalid input", str(context.exception))
```

### 5. Performance Test Guidelines
- Use appropriate sample sizes for statistical significance
- Measure what matters (don't over-optimize micro-benchmarks)
- Include baseline comparisons
- Account for system variability

### 6. Property-Based Test Design
- Focus on invariants and mathematical properties
- Use appropriate data generation strategies
- Include shrinking for minimal failing examples
- Set reasonable example counts and deadlines

### 7. Mock Usage
```python
from unittest.mock import Mock, patch

# Good: Mock external dependencies
@patch('requests.get')
def test_api_call(self, mock_get):
    mock_get.return_value.json.return_value = {'status': 'success'}
    
    result = api_function()
    
    mock_get.assert_called_once_with('expected_url')
    self.assertEqual(result['status'], 'success')
```

### 8. Test Documentation
- Include docstrings explaining what each test validates
- Document any special setup or preconditions
- Explain complex test logic or edge cases

---

Remember: Good tests are your safety net for refactoring and your documentation for how code should behave. Invest in test quality - it pays dividends in code confidence and maintainability.