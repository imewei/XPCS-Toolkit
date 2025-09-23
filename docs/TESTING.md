:orphan:

# XPCS Toolkit Comprehensive Testing Framework

This document provides complete guidance for testing in XPCS Toolkit, covering all aspects of the comprehensive test suite including architecture, best practices, quality standards, and developer workflows.

## Table of Contents

- Overview and Philosophy
- Test Architecture
- Quick Start Guide
- Test Categories and Commands
- Writing Quality Tests
- Test Quality Standards
- Scientific Testing Guidelines
- Performance Testing
- Property-Based Testing
- GUI Testing
- Integration and End-to-End Testing
- Test Utilities and Helpers
- Debugging and Profiling
- IDE Integration
- Maintenance and Evolution
- Troubleshooting Guide

## Overview and Philosophy

The XPCS Toolkit testing framework is designed with the following principles:

### Core Principles
- **Scientific Rigor**: All tests must validate scientific accuracy and maintain numerical precision
- **Comprehensive Coverage**: Every public interface and critical code path must be tested
- **Maintainable Quality**: Tests should be clear, well-documented, and easy to maintain
- **Performance Awareness**: Tests must verify performance requirements and detect regressions
- **Cross-Platform Compatibility**: Tests run reliably across different environments

### Testing Philosophy
The framework follows a multi-layered approach:
1. **Unit Tests**: Validate individual components in isolation
2. **Integration Tests**: Verify component interactions and data flow
3. **Scientific Tests**: Validate mathematical accuracy and physical constraints
4. **Performance Tests**: Ensure performance requirements are met
5. **GUI Tests**: Validate user interface functionality
6. **End-to-End Tests**: Test complete workflows from user perspective

## Test Architecture

### Directory Structure

```
tests/
├── conftest.py                 # Global pytest configuration and fixtures
├── utils.py                   # Test utilities and helper functions
├── quality_standards.py       # Automated quality checking
│
├── unit/                      # Unit tests for individual modules
│   ├── core/                  # Core functionality tests
│   ├── fileio/                # File I/O operation tests
│   ├── analysis/              # Analysis algorithm tests
│   ├── utils/                 # Utility function tests
│   └── threading/             # Threading and async tests
│
├── integration/               # Integration tests
│   ├── test_component_integration.py
│   ├── test_data_pipeline_integration.py
│   └── test_performance_integration.py
│
├── scientific/                # Scientific validation tests
│   ├── algorithms/            # Algorithm accuracy tests
│   ├── properties/            # Physical property validation
│   └── reference_validation/  # Cross-validation against references
│
├── performance/               # Performance and benchmark tests
│   └── benchmarks/           # Specific performance benchmarks
│
├── gui_interactive/           # GUI and interactive tests
│   ├── test_main_window.py
│   ├── test_analysis_tabs.py
│   └── test_user_scenarios.py
│
├── end_to_end/               # Complete workflow tests
│   └── test_xpcs_workflow.py
│
├── error_handling/           # Error handling and edge case tests
│   ├── test_error_recovery.py
│   └── test_edge_cases.py
│
└── fixtures/                 # Test data and fixtures
    └── hdf5_fixtures.py
```

### Test Categories and Markers

The framework uses pytest markers to categorize tests:

```python
@pytest.mark.unit           # Unit tests
@pytest.mark.integration    # Integration tests
@pytest.mark.performance    # Performance tests
@pytest.mark.gui           # GUI tests
@pytest.mark.slow          # Tests taking > 1 second
@pytest.mark.scientific    # Scientific accuracy tests
```

## Quick Start Guide

### Environment Setup

```bash
# Create and activate conda environment
conda create -n xpcs-toolkit-testing python==3.10.16
conda activate xpcs-toolkit-testing

# Install in development mode with test dependencies
pip install -e ".[test]"
```

### Running Tests

```bash
# Quick test run (unit tests only)
make test

# Full test suite
make test-full

# Specific categories
make test-unit          # Unit tests
make test-integration   # Integration tests
make test-scientific    # Scientific validation
make test-performance   # Performance benchmarks
make test-gui          # GUI tests (requires display)

# With coverage reporting
make coverage

# Quality assessment
python tests/framework/quality_standards.py --check-all
```

### IDE Integration

#### VS Code Configuration

Create `.vscode/settings.json`:
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/",
        "-v",
        "--tb=short"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true
}
```

#### PyCharm Configuration
1. Go to Settings → Tools → Python Integrated Tools
2. Set Default test runner to "pytest"
3. Set pytest arguments to `-v --tb=short`
4. Configure test runner to use project Python interpreter

## Test Categories and Commands

### Unit Tests

**Purpose**: Test individual functions, classes, and modules in isolation.

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific module tests
pytest tests/unit/core/test_xpcs_file.py -v

# Run with markers
pytest -m "unit and not slow" -v
```

**Example Unit Test Structure**:
```python
class TestXpcsFile:
    """Unit tests for XpcsFile class."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_loading_success(self, minimal_xpcs_hdf5):
        """Test successful HDF5 file loading."""
        xf = XpcsFile(minimal_xpcs_hdf5)

        assert xf.fname == minimal_xpcs_hdf5
        assert xf.is_valid()

    def test_invalid_file_raises_error(self):
        """Test that invalid file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            XpcsFile("nonexistent_file.hdf")
```

### Integration Tests

**Purpose**: Test component interactions and data flow between modules.

```bash
# Run integration tests
pytest tests/integration/ -v

# Specific integration suites
pytest tests/integration/test_data_pipeline_integration.py -v
```

### Scientific Validation Tests

**Purpose**: Validate mathematical accuracy and physical constraints.

```bash
# Run scientific tests
pytest tests/scientific/ -v

# Algorithm validation
pytest tests/scientific/algorithms/ -v

# Property validation
pytest tests/scientific/properties/ -v
```

**Scientific Test Example**:
```python
@scientific_test(tolerance=1e-6)
def test_g2_calculation_accuracy(self, synthetic_correlation_data):
    """Test G2 calculation maintains scientific accuracy."""
    data = synthetic_correlation_data

    # Calculate G2 using our implementation
    calculated_g2 = calculate_g2(data['raw_intensity'])

    # Compare with theoretical expectation
    ScientificAssertions.assert_arrays_close(
        calculated_g2, data['g2_theory'],
        rtol=1e-6, atol=1e-12,
        msg="G2 calculation accuracy validation"
    )

    # Validate physical constraints
    ScientificAssertions.assert_correlation_properties(
        data['tau'], calculated_g2
    )
```

### Performance Tests

**Purpose**: Ensure performance requirements are met and detect regressions.

```bash
# Run performance tests
pytest tests/performance/ -v --benchmark-only

# Memory performance tests
pytest tests/performance/ -k "memory" -v

# Generate performance report
python tests/run_logging_benchmarks.py --report
```

### GUI Tests

**Purpose**: Test user interface functionality and interactions.

```bash
# Run GUI tests (requires display)
pytest tests/gui_interactive/ -v

# Headless GUI tests
QT_QPA_PLATFORM=offscreen pytest tests/gui_interactive/ -v

# Interactive GUI tests (manual)
python tests/gui_interactive/run_gui_tests.py
```

## Writing Quality Tests

### Test Quality Checklist

Before submitting any test, ensure it meets these quality standards:

- [ ] **Descriptive docstring** explaining what the test validates
- [ ] **Clear test structure** following Arrange-Act-Assert pattern
- [ ] **Specific assertions** with meaningful error messages
- [ ] **Proper fixture usage** for test data and setup
- [ ] **Appropriate tolerance** for numerical comparisons
- [ ] **Error case testing** for expected exceptions
- [ ] **Cleanup handling** in teardown methods
- [ ] **Performance awareness** for potentially slow operations

### Test Template

```python
def test_feature_behavior(self, fixture_name):
    """Test that feature behaves correctly under specific conditions.

    This test validates:
    - Expected behavior for valid input
    - Proper error handling for invalid input
    - Performance within acceptable limits
    """
    # Arrange - Set up test data and expected results
    test_data = create_test_data()
    expected_result = calculate_expected_result(test_data)

    # Act - Execute the functionality being tested
    with PerformanceTimer("feature_execution") as timer:
        actual_result = feature_function(test_data)

    # Assert - Verify results and constraints
    assert actual_result is not None, "Result should not be None"
    ScientificAssertions.assert_arrays_close(
        actual_result, expected_result,
        rtol=1e-7, atol=1e-14,
        msg="Feature calculation accuracy"
    )

    # Performance assertion
    assert timer.elapsed < 1.0, f"Feature too slow: {timer.elapsed:.3f}s"
```

### Common Anti-Patterns to Avoid

**❌ Bad Test Practices:**
```python
def test_something(self):
    result = do_something()
    assert result  # Too vague

def test_numbers(self):
    assert abs(result - expected) < 0.001  # Hardcoded tolerance

def test_without_cleanup(self):
    create_temporary_files()
    # No cleanup - files left behind
```

**✅ Good Test Practices:**
```python
def test_correlation_calculation_accuracy(self, correlation_data):
    """Test that correlation calculation produces accurate results."""
    result = calculate_correlation(correlation_data['raw'])

    ScientificAssertions.assert_arrays_close(
        result, correlation_data['expected'],
        rtol=1e-7, atol=1e-14,
        msg="Correlation calculation accuracy validation"
    )
```

## Test Quality Standards

### Automated Quality Checking

Use the built-in quality checker to assess test quality:

```bash
# Check all test files
python tests/framework/quality_standards.py --check-all

# Generate quality report
python tests/framework/quality_standards.py --check-all --format json --output quality_report.json

# Fix automatically fixable issues
python tests/framework/quality_standards.py --check-all --fix-issues
```

### Quality Metrics

The quality checker evaluates tests on these dimensions:

1. **Docstring Coverage** (20 points)
   - All test classes have descriptive docstrings
   - All test methods explain what they validate

2. **Assertion Quality** (25 points)
   - Use specific assertions (assertEqual vs assertTrue)
   - Provide meaningful assertion messages
   - Appropriate tolerance for numerical comparisons

3. **Scientific Rigor** (20 points)
   - Explicit tolerances for floating-point comparisons
   - Validation of physical constraints
   - Proper array comparison methods

4. **Test Patterns** (20 points)
   - Proper setup/teardown usage
   - Exception testing for error conditions
   - Use of fixtures and parametrization

5. **Code Quality** (15 points)
   - Clear naming conventions
   - Absence of magic numbers
   - Minimal code duplication

### Target Quality Scores

- **Excellent** (85-100): Production-ready tests
- **Good** (70-84): Acceptable with minor improvements
- **Fair** (50-69): Needs significant improvement
- **Poor** (0-49): Major rewrite required

## Scientific Testing Guidelines

### Numerical Precision Standards

Always specify explicit tolerances for numerical comparisons:

```python
# For basic floating-point comparisons
assert abs(actual - expected) < 1e-12

# For array comparisons
np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-14)

# Using scientific assertion helpers
ScientificAssertions.assert_arrays_close(actual, expected, rtol=1e-7, atol=1e-14)
```

### Physical Constraint Validation

Always validate that results satisfy physical constraints:

```python
def test_correlation_function_properties(self, correlation_data):
    """Test that correlation function satisfies physical constraints."""
    tau, g2, g2_err = correlation_data

    # Use built-in validation
    ScientificAssertions.assert_correlation_properties(tau, g2, g2_err)

    # Additional domain-specific constraints
    assert np.all(g2[0] >= g2), "G2 should be monotonically decreasing"
```

### Statistical Validation

For tests involving statistics or random processes:

```python
@pytest.mark.parametrize("n_samples", [100, 1000, 10000])
def test_statistical_property(self, n_samples):
    """Test statistical property with different sample sizes."""
    samples = generate_samples(n_samples)

    # Statistical test with appropriate confidence level
    statistic, p_value = statistical_test(samples)

    assert p_value > 0.05, f"Statistical test failed: p={p_value:.3f}"
```

## Performance Testing

### Performance Test Structure

```python
@performance_test(max_time=1.0, memory_limit_mb=50.0)
def test_algorithm_performance(self, large_dataset):
    """Test that algorithm meets performance requirements."""

    with PerformanceTimer("algorithm_execution") as timer:
        result = expensive_algorithm(large_dataset)

    # Automatic validation by decorator
    # Additional custom performance assertions
    assert timer.mean_time < 0.5, "Algorithm too slow for typical usage"
```

### Benchmark Tests

```python
def test_g2_calculation_benchmark(benchmark, correlation_data):
    """Benchmark G2 calculation performance."""

    def calculate():
        return calculate_g2(correlation_data)

    result = benchmark(calculate)

    # Validate performance requirements
    assert benchmark.stats.mean < 0.1, "G2 calculation too slow"
    assert benchmark.stats.min < 0.05, "Best case performance insufficient"
```

### Memory Usage Testing

```python
def test_memory_usage_bounds(self, large_dataset):
    """Test that memory usage stays within bounds."""
    import psutil

    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024

    result = process_large_dataset(large_dataset)

    memory_after = process.memory_info().rss / 1024 / 1024
    memory_delta = memory_after - memory_before

    assert memory_delta < 100, f"Excessive memory usage: {memory_delta:.1f}MB"
```

## Property-Based Testing

### Using Hypothesis for Mathematical Properties

```python
from hypothesis import given, strategies as st

@given(tau=st.lists(st.floats(min_value=1e-6, max_value=1e6), min_size=10, max_size=100))
def test_correlation_monotonicity_property(self, tau):
    """Test that correlation function is monotonically decreasing."""
    tau_sorted = sorted(tau)
    g2 = calculate_correlation_function(tau_sorted)

    # Property: correlation should be monotonically decreasing
    for i in range(len(g2) - 1):
        assert g2[i] >= g2[i + 1], f"Monotonicity violated at index {i}"
```

### Complex Property Testing

```python
@given(
    data=st.builds(
        dict,
        intensity=st.lists(st.floats(min_value=0, max_value=1e6), min_size=100),
        background=st.floats(min_value=0, max_value=1000),
        noise_level=st.floats(min_value=0, max_value=0.1)
    )
)
@settings(max_examples=200, deadline=5000)
def test_scattering_analysis_properties(self, data):
    """Test scattering analysis maintains physical properties."""
    result = analyze_scattering(data)

    # Properties that should always hold
    assert np.all(result.q >= 0), "Q values must be non-negative"
    assert np.all(result.intensity >= 0), "Intensity must be non-negative"
    assert result.fit_quality > 0, "Fit quality must be positive"
```

## GUI Testing

### Headless GUI Testing

```python
@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """Set up Qt application for GUI testing."""
    import sys
    from PySide6.QtWidgets import QApplication

    # Ensure headless operation
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    if not QApplication.instance():
        app = QApplication(sys.argv)
    else:
        app = QApplication.instance()

    yield app

    app.quit()

@gui_test(requires_display=False)
def test_main_window_creation(self, qt_application):
    """Test that main window creates successfully."""
    from xpcs_toolkit.xpcs_viewer import XpcsViewer

    window = XpcsViewer()
    assert window is not None
    assert window.windowTitle() == "XPCS Toolkit"
```

### Widget Testing

```python
def test_plot_widget_functionality(self, qt_application, synthetic_correlation_data):
    """Test plot widget with synthetic data."""
    from xpcs_toolkit.plotting import CorrelationPlotWidget

    widget = CorrelationPlotWidget()

    # Test data loading
    widget.set_data(synthetic_correlation_data)
    assert widget.has_data()

    # Test plot updates
    widget.update_plot()
    assert len(widget.plot_items) > 0
```

## Integration and End-to-End Testing

### Data Pipeline Integration

```python
def test_complete_data_pipeline(self, comprehensive_xpcs_hdf5):
    """Test complete data processing pipeline."""
    # Load data
    xf = XpcsFile(comprehensive_xpcs_hdf5)

    # Process through pipeline
    processor = DataProcessor(xf)
    results = processor.process_all()

    # Validate end-to-end results
    assert 'g2' in results
    assert 'fitting' in results
    assert 'saxs' in results

    # Validate data integrity throughout pipeline
    ScientificAssertions.assert_correlation_properties(
        results['g2']['tau'], results['g2']['values']
    )
```

### Workflow Testing

```python
def test_user_workflow_g2_analysis(self, sample_data_directory):
    """Test complete user workflow for G2 analysis."""
    # Simulate user actions
    file_locator = FileLocator(sample_data_directory)
    files = file_locator.find_xpcs_files()

    assert len(files) > 0, "No XPCS files found"

    # Load and analyze
    for file_path in files[:3]:  # Test first 3 files
        xf = XpcsFile(file_path)

        # Perform analysis
        g2_results = xf.get_g2_analysis()

        # Validate results
        assert g2_results is not None
        assert 'tau' in g2_results
        assert 'g2' in g2_results
```

## Test Utilities and Helpers

### Using Test Utilities

The `tests/utils.py` module provides comprehensive utilities:

```python
from tests.utils import (
    ScientificAssertions, MockFactory, TestDataGenerator,
    PerformanceTimer, TestDebugger, scientific_test
)

class TestMyFeature:
    def test_with_utilities(self):
        # Generate test data
        data = TestDataGenerator.generate_correlation_data(
            tau_range=(1e-6, 1e2), n_points=50
        )

        # Use scientific assertions
        ScientificAssertions.assert_correlation_properties(
            data['tau'], data['g2']
        )

        # Mock objects
        mock_file = MockFactory.create_mock_xpcs_file()

        # Performance monitoring
        with PerformanceTimer("feature_test") as timer:
            result = my_feature(data)

        print(timer.report())
```

### Custom Fixtures

```python
@pytest.fixture
def realistic_xpcs_data():
    """Generate realistic XPCS data for testing."""
    return TestDataGenerator.generate_correlation_data(
        tau_range=(1e-6, 1e2),
        n_points=50,
        beta=0.8,
        tau_c=1e-3,
        noise_level=0.02
    )

@pytest.fixture
def mock_detector_setup():
    """Mock detector geometry and configuration."""
    return MockFactory.create_mock_detector_geometry()
```

## Debugging and Profiling

### Test Debugging

```python
@TestDebugger.capture_context
@TestDebugger.log_test_steps
def test_complex_calculation(self):
    """Test complex calculation with debugging support."""
    # Test implementation
    # Automatic context capture on failure
    # Automatic step logging
```

### Profiling Test Performance

```python
def test_with_profiling(self):
    """Test with detailed profiling information."""
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    # Test execution
    result = complex_operation()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 functions
```

### Memory Profiling

```python
def test_memory_profiling(self):
    """Test with memory usage profiling."""
    from memory_profiler import profile

    @profile
    def memory_intensive_operation():
        return process_large_data()

    result = memory_intensive_operation()
```

## IDE Integration

### VS Code Setup

**Settings for .vscode/settings.json:**
```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/",
        "-v",
        "--tb=short",
        "--strict-markers"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"]
}
```

**Launch configuration for .vscode/launch.json:**
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Current Test",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v", "-s"],
            "console": "integratedTerminal",
            "justMyCode": false
        }
    ]
}
```

### PyCharm Configuration

1. **Test Runner Configuration:**
   - File → Settings → Tools → Python Integrated Tools
   - Default test runner: pytest
   - pytest arguments: `-v --tb=short --strict-markers`

2. **Code Quality Integration:**
   - Enable inspections for test code
   - Configure code coverage to show test coverage
   - Set up run configurations for different test categories

## Maintenance and Evolution

### Test Suite Maintenance Procedures

#### Weekly Maintenance
- [ ] Run full test suite across all Python versions
- [ ] Check test execution time trends
- [ ] Review failing tests and flaky test reports
- [ ] Update test data if needed

#### Monthly Maintenance
- [ ] Run test quality assessment
- [ ] Review test coverage reports
- [ ] Update test dependencies
- [ ] Clean up obsolete test code

#### Release Preparation
- [ ] Run comprehensive validation suite
- [ ] Performance regression testing
- [ ] Cross-platform compatibility testing
- [ ] Documentation example testing

### Adding New Tests

**Checklist for New Test Files:**
- [ ] Follow naming convention: `test_*.py`
- [ ] Include comprehensive module docstring
- [ ] Use appropriate test class organization
- [ ] Add appropriate pytest markers
- [ ] Include in relevant test categories
- [ ] Update documentation if needed

**Checklist for New Test Cases:**
- [ ] Descriptive docstring explaining validation purpose
- [ ] Follow Arrange-Act-Assert pattern
- [ ] Use specific assertions with messages
- [ ] Include error case testing
- [ ] Add performance considerations
- [ ] Use appropriate fixtures

### Deprecating Tests

When removing or changing tests:
1. **Document the reason** for deprecation
2. **Provide migration path** for test patterns
3. **Update related documentation**
4. **Communicate changes** to team
5. **Archive old test code** if needed for reference

## Troubleshooting Guide

### Common Issues and Solutions

#### Test Discovery Issues
```bash
# Problem: Tests not discovered by pytest
# Solution: Check naming conventions and __init__.py files
find tests/ -name "*.py" -not -name "test_*" -not -name "*_test.py"

# Verify pytest can discover tests
pytest --collect-only tests/
```

#### Import Errors in Tests
```bash
# Problem: Module import errors
# Solution: Check PYTHONPATH and installation
python -c "import xpcs_toolkit; print(xpcs_toolkit.__file__)"

# Install in development mode
pip install -e .
```

#### GUI Test Failures
```bash
# Problem: GUI tests fail on headless systems
# Solution: Use offscreen platform
export QT_QPA_PLATFORM=offscreen
pytest tests/gui_interactive/ -v

# Alternative: Skip GUI tests
pytest tests/ -v -m "not gui"
```

#### Memory Issues in Tests
```bash
# Problem: Tests consume too much memory
# Solution: Use smaller test datasets or test chunking
pytest tests/ -v --maxfail=1  # Stop after first failure

# Monitor memory usage
python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

#### Slow Test Execution
```bash
# Problem: Tests run too slowly
# Solution: Profile and optimize

# Run fastest tests first
pytest tests/ -v -m "not slow"

# Profile test execution
pytest tests/ -v --durations=10

# Use parallel execution (if available)
pytest tests/ -v -n auto  # Requires pytest-xdist
```

### Performance Debugging

#### Identifying Slow Tests
```bash
# Find tests taking > 1 second
pytest tests/ --durations=0 | grep -E "s (test_|FAILED|ERROR)"

# Profile specific test
python -m cProfile -s cumulative -m pytest tests/unit/test_slow_module.py -v
```

#### Memory Leak Detection
```python
def test_for_memory_leaks(self):
    """Test for memory leaks in long-running operations."""
    import gc
    import psutil

    process = psutil.Process()

    # Baseline measurement
    gc.collect()
    memory_start = process.memory_info().rss

    # Repeat operation multiple times
    for i in range(100):
        result = potentially_leaky_operation()

        # Force cleanup
        del result
        if i % 10 == 0:
            gc.collect()

    # Final measurement
    gc.collect()
    memory_end = process.memory_info().rss
    memory_growth = (memory_end - memory_start) / 1024 / 1024

    assert memory_growth < 10, f"Potential memory leak: {memory_growth:.1f}MB growth"
```

### Test Data Management

#### Generating Test Data
```python
# Use fixtures for consistent test data
@pytest.fixture(scope="session")
def large_test_dataset():
    """Generate large test dataset once per session."""
    if not Path("test_cache/large_dataset.hdf").exists():
        generate_and_cache_large_dataset()
    return load_cached_dataset()

# Clean up test data
@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Automatically clean up temporary test files."""
    yield
    # Clean up after test
    for temp_file in Path(".").glob("test_*.tmp"):
        temp_file.unlink()
```

#### Test Environment Issues
```bash
# Problem: Tests behave differently across environments
# Solution: Standardize test environment

# Check environment consistency
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
import numpy; print(f'NumPy: {numpy.__version__}')
"

# Use environment markers
@pytest.mark.skipif(sys.platform == "win32", reason="Unix-specific test")
def test_unix_specific_functionality():
    pass
```

---

## Summary

This comprehensive testing framework ensures that XPCS Toolkit maintains the highest standards of scientific software quality. The multi-layered approach provides confidence in both the correctness and performance of the software, while the quality standards and automation ensure that the test suite itself remains maintainable and effective.

Key takeaways:
- **Use appropriate test categories** for different validation needs
- **Follow scientific rigor standards** for numerical accuracy
- **Maintain high test quality** through automated checking
- **Leverage utilities and helpers** for consistent patterns
- **Monitor performance** and prevent regressions
- **Keep tests maintainable** through clear structure and documentation

The framework evolves with the codebase, ensuring that testing practices remain effective as the software grows in complexity and capability.
