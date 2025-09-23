:orphan:

# XPCS Toolkit Testing Developer Workflow Guide

This guide provides step-by-step workflows for developers working with the XPCS Toolkit test suite, covering daily development practices, IDE integration, and testing workflows.

## Table of Contents

- Daily Development Workflow
- IDE Setup and Integration
- Test-Driven Development Workflow
- Debugging Test Failures
- Performance Testing Workflow
- Code Review Testing Checklist
- CI/CD Integration
- Team Collaboration

## Daily Development Workflow

### Morning Routine

```bash
# 1. Sync with latest changes
git pull origin master

# 2. Quick health check (< 30 seconds)
make test

# 3. Check test quality status
python tests/quality_standards.py --check-all --format text | head -20
```

### Before Starting New Feature

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Run full test suite to establish baseline
make test-full

# 3. Check current coverage
make coverage-report | grep "TOTAL"

# 4. Identify relevant test categories for your feature
# - Scientific algorithm? Run: make test-scientific
# - GUI feature? Run: make test-gui
# - Performance critical? Run: make test-performance
```

### During Development

#### Test-First Approach (Recommended)

```bash
# 1. Write failing test first
# Edit: tests/unit/analysis/test_my_feature.py

# 2. Run specific test to confirm it fails
pytest tests/unit/analysis/test_my_feature.py::TestMyFeature::test_new_functionality -v

# 3. Implement feature to make test pass
# Edit: xpcs_toolkit/analysis/my_feature.py

# 4. Run test again to confirm it passes
pytest tests/unit/analysis/test_my_feature.py::TestMyFeature::test_new_functionality -v

# 5. Run related tests to check for regressions
pytest tests/unit/analysis/ -v

# 6. Refactor and optimize (test should still pass)
```

#### Continuous Testing During Development

```bash
# Use pytest-watch for automatic test running (optional)
pip install pytest-watch
cd /path/to/project

# Watch specific test file
ptw tests/unit/analysis/test_my_feature.py

# Watch all tests related to module
ptw tests/unit/analysis/
```

### Before Committing Changes

```bash
# 1. Run relevant test categories
make test-unit          # Always run unit tests
make test-integration   # If you modified multiple components
make test-scientific    # If you modified algorithms
make test-performance   # If you modified performance-critical code

# 2. Check test quality for new/modified tests
python tests/quality_standards.py --check-all | grep -A 5 -B 5 "test_my_feature"

# 3. Update test coverage
make coverage

# 4. Run linting
make lint

# 5. Check that all tests pass
make test-ci
```

### End of Day Routine

```bash
# 1. Commit your changes with descriptive message
git add -A
git commit -m "feat: add correlation analysis optimization

- Implement vectorized G2 calculation
- Add comprehensive test coverage
- Optimize memory usage by 40%
- Add performance benchmarks"

# 2. Push to feature branch
git push origin feature/my-new-feature

# 3. Run overnight validation (optional for major changes)
nohup make test-full > test_results_$(date +%Y%m%d).log 2>&1 &
```

## IDE Setup and Integration

### VS Code Complete Setup

#### 1. Required Extensions
```bash
# Install VS Code extensions
code --install-extension ms-python.python
code --install-extension ms-python.flake8
code --install-extension ms-python.black-formatter
code --install-extension ms-python.pylint
code --install-extension charliermarsh.ruff
```

#### 2. Workspace Configuration

Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "~/anaconda3/envs/xpcs-toolkit/bin/python",

    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/",
        "-v",
        "--tb=short",
        "--strict-markers",
        "--maxfail=3"
    ],
    "python.testing.autoTestDiscoverOnSaveEnabled": true,
    "python.testing.cwd": "${workspaceFolder}",

    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintArgs": ["--load-plugins=pylint_pytest"],

    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],

    "python.sortImports.args": ["--profile", "black"],

    "files.associations": {
        "*.hdf": "plaintext",
        "conftest.py": "python"
    },

    "python.analysis.typeCheckingMode": "basic",
    "python.analysis.autoImportCompletions": true,

    "editor.rulers": [88],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### 3. Debug Configuration

Create `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Current Test File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v", "-s", "--no-header"],
            "console": "integratedTerminal",
            "justMyCode": false,
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "PYXPCS_LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Debug Current Test Method",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}::${selectedText}", "-v", "-s"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Debug Test with Coverage",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v", "-s", "--cov=xpcs_toolkit", "--cov-report=html"],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Run GUI Tests Interactively",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/tests/gui_interactive/run_gui_tests.py",
            "console": "integratedTerminal",
            "env": {
                "QT_QPA_PLATFORM": "xcb"
            }
        }
    ]
}
```

#### 4. Tasks Configuration

Create `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Unit Tests",
            "type": "shell",
            "command": "make test-unit",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Run All Tests",
            "type": "shell",
            "command": "make test-full",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Generate Coverage Report",
            "type": "shell",
            "command": "make coverage",
            "group": "test"
        },
        {
            "label": "Check Test Quality",
            "type": "shell",
            "command": "python tests/quality_standards.py --check-all",
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Run Scientific Tests",
            "type": "shell",
            "command": "make test-scientific",
            "group": "test"
        },
        {
            "label": "Run Performance Tests",
            "type": "shell",
            "command": "make test-performance",
            "group": "test"
        }
    ]
}
```

### PyCharm Professional Setup

#### 1. Project Configuration
- **File** → **Settings** → **Project: XPCS-Toolkit** → **Python Interpreter**
  - Set to conda environment: `~/anaconda3/envs/xpcs-toolkit/bin/python`

#### 2. Test Runner Configuration
- **File** → **Settings** → **Tools** → **Python Integrated Tools**
  - **Default test runner**: pytest
  - **pytest arguments**: `-v --tb=short --strict-markers`

#### 3. Run Configurations

Create run configurations for:
- **Unit Tests**: `pytest tests/unit/ -v`
- **Integration Tests**: `pytest tests/integration/ -v`
- **Scientific Tests**: `pytest tests/scientific/ -v`
- **Current Test File**: `pytest $FilePath$ -v`
- **Test with Coverage**: `pytest tests/ --cov=xpcs_toolkit --cov-report=html`

#### 4. Code Quality Integration
- **File** → **Settings** → **Editor** → **Inspections**
  - Enable Python inspections
  - Add custom inspection for test quality patterns

## Test-Driven Development Workflow

### TDD Cycle for New Features

#### Step 1: Write the Test
```python
# File: tests/unit/analysis/test_new_feature.py

class TestNewFeature:
    """Tests for new correlation analysis feature."""

    def test_new_correlation_method_accuracy(self, synthetic_correlation_data):
        """Test that new correlation method produces accurate results."""
        # Arrange
        input_data = synthetic_correlation_data['raw_intensity']
        expected_result = synthetic_correlation_data['expected_g2']

        # Act
        # This will fail initially - feature doesn't exist yet
        result = new_correlation_method(input_data)

        # Assert
        ScientificAssertions.assert_arrays_close(
            result, expected_result,
            rtol=1e-6, atol=1e-12,
            msg="New correlation method accuracy"
        )

    def test_new_correlation_method_performance(self, large_dataset):
        """Test that new method meets performance requirements."""
        with PerformanceTimer("new_correlation") as timer:
            result = new_correlation_method(large_dataset)

        assert timer.elapsed < 1.0, f"Too slow: {timer.elapsed:.3f}s"

    def test_new_correlation_method_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            new_correlation_method(np.array([]))
```

#### Step 2: Run Test (Should Fail)
```bash
# Run the test - it should fail because feature doesn't exist
pytest tests/unit/analysis/test_new_feature.py::TestNewFeature::test_new_correlation_method_accuracy -v

# Expected output: ImportError or AttributeError
```

#### Step 3: Implement Minimum Feature
```python
# File: xpcs_toolkit/analysis/correlation.py

def new_correlation_method(intensity_data):
    """New correlation analysis method.

    Args:
        intensity_data: Input intensity time series

    Returns:
        Correlation function values
    """
    if len(intensity_data) == 0:
        raise ValueError("Input data cannot be empty")

    # Minimal implementation to make test pass
    # TODO: Implement actual algorithm
    return np.ones_like(intensity_data) + 0.5 * np.random.rand(len(intensity_data))
```

#### Step 4: Run Test Again (Should Pass)
```bash
pytest tests/unit/analysis/test_new_feature.py::TestNewFeature::test_new_correlation_method_accuracy -v
```

#### Step 5: Refactor and Optimize
```python
def new_correlation_method(intensity_data):
    """Optimized correlation analysis method using vectorized operations."""
    if len(intensity_data) == 0:
        raise ValueError("Input data cannot be empty")

    # Proper implementation with scientific algorithms
    # ... (implement actual correlation calculation)

    return correlation_result
```

#### Step 6: Add More Tests
```python
def test_new_correlation_method_edge_cases(self):
    """Test edge cases and boundary conditions."""
    # Single data point
    single_point = np.array([1.0])
    result = new_correlation_method(single_point)
    assert len(result) == 1

    # Very large dataset
    large_data = np.random.rand(10000)
    result = new_correlation_method(large_data)
    assert len(result) == len(large_data)
```

### TDD Cycle for Bug Fixes

#### Step 1: Write Test Reproducing Bug
```python
def test_correlation_bug_with_negative_values(self):
    """Test that correlation handles negative intensity values correctly.

    This test reproduces bug #123 where negative values cause crashes.
    """
    # Data that reproduces the bug
    problematic_data = np.array([1.0, -0.1, 2.0, 3.0])  # Contains negative value

    # This should not crash (currently does)
    result = correlation_function(problematic_data)

    # Should handle negative values gracefully
    assert result is not None
    assert len(result) == len(problematic_data)
```

#### Step 2: Confirm Test Fails
```bash
pytest tests/unit/analysis/test_correlation_fixes.py::test_correlation_bug_with_negative_values -v
# Should fail with the original bug
```

#### Step 3: Fix the Bug
```python
def correlation_function(intensity_data):
    """Fixed correlation function that handles negative values."""
    # Add validation and sanitization
    if np.any(intensity_data < 0):
        logging.warning("Negative intensity values detected, clipping to zero")
        intensity_data = np.maximum(intensity_data, 0)

    # Rest of implementation
    return calculate_correlation(intensity_data)
```

#### Step 4: Confirm Fix Works
```bash
pytest tests/unit/analysis/test_correlation_fixes.py::test_correlation_bug_with_negative_values -v
# Should now pass
```

## Debugging Test Failures

### Systematic Debugging Approach

#### 1. Identify the Failure Type
```bash
# Run single failing test with maximum verbosity
pytest tests/unit/analysis/test_failing.py::TestClass::test_method -vvv -s --tb=long

# Check if it's an environment issue
python -c "import xpcs_toolkit; print('Import successful')"

# Check if it's data-related
ls -la tests/fixtures/
```

#### 2. Use Debugging Decorators
```python
from tests.utils import TestDebugger

class TestProblematicFeature:

    @TestDebugger.capture_context
    @TestDebugger.log_test_steps
    def test_failing_method(self, test_data):
        """Test with automatic debugging support."""
        # Test implementation
        # Failure will automatically capture context
```

#### 3. Interactive Debugging
```python
def test_with_debugging(self, test_data):
    """Test with embedded debugging capability."""
    result = problematic_function(test_data)

    # Add debugging breakpoint
    import pdb; pdb.set_trace()

    # Or use pytest's built-in debugging
    # Run with: pytest --pdb tests/path/test_file.py::test_method

    assert result is not None
```

#### 4. Isolate the Problem
```bash
# Test in isolation
pytest tests/unit/analysis/test_failing.py::TestClass::test_method -v --no-header

# Test with fresh environment
python -m pytest tests/unit/analysis/test_failing.py::TestClass::test_method -v

# Test with different data
pytest tests/unit/analysis/test_failing.py::TestClass::test_method -v --fixtures-per-test
```

### Common Debugging Scenarios

#### Scientific Accuracy Failures
```python
def debug_numerical_precision(self):
    """Debug numerical precision issues."""
    expected = 1.23456789
    actual = complex_calculation()

    print(f"Expected: {expected:.15f}")
    print(f"Actual:   {actual:.15f}")
    print(f"Diff:     {abs(actual - expected):.2e}")
    print(f"Rel diff: {abs(actual - expected) / expected:.2e}")

    # Check if it's a tolerance issue
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-15)
```

#### Performance Test Failures
```python
def debug_performance_issues(self):
    """Debug performance test failures."""
    import cProfile

    # Profile the slow operation
    profiler = cProfile.Profile()
    profiler.enable()

    result = slow_function()

    profiler.disable()
    profiler.dump_stats('performance_debug.prof')

    # Analyze with snakeviz: pip install snakeviz
    # snakeviz performance_debug.prof
```

#### Memory Issues
```python
def debug_memory_usage(self):
    """Debug memory-related test failures."""
    import tracemalloc
    import psutil

    # Start tracing
    tracemalloc.start()
    process = psutil.Process()

    # Baseline
    memory_start = process.memory_info().rss / 1024 / 1024

    # Operation that might leak memory
    for i in range(100):
        result = potentially_leaky_operation()
        if i % 10 == 0:
            current, peak = tracemalloc.get_traced_memory()
            print(f"Iteration {i}: Current={current/1024/1024:.1f}MB, Peak={peak/1024/1024:.1f}MB")

    # Final measurement
    memory_end = process.memory_info().rss / 1024 / 1024
    print(f"Memory growth: {memory_end - memory_start:.1f}MB")

    # Get top memory consumers
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    for stat in top_stats[:10]:
        print(stat)
```

## Performance Testing Workflow

### Setting Up Performance Baselines

```python
# File: tests/performance/baselines.py

PERFORMANCE_BASELINES = {
    "g2_calculation": {
        "max_time": 0.5,  # seconds
        "max_memory": 50,  # MB
        "dataset_size": 10000
    },
    "hdf5_loading": {
        "max_time": 0.1,
        "max_memory": 20,
        "file_size": "1MB"
    },
    "gui_startup": {
        "max_time": 2.0,
        "max_memory": 100
    }
}

def get_baseline(operation_name):
    """Get performance baseline for operation."""
    return PERFORMANCE_BASELINES.get(operation_name, {})
```

### Performance Testing Workflow

#### 1. Establish Baseline
```bash
# Run performance tests to establish baseline
pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Save baseline results
cp .benchmarks/baseline.json tests/performance/baseline_results.json
```

#### 2. Regular Performance Testing
```bash
# Run against baseline
pytest tests/performance/ --benchmark-only --benchmark-compare=baseline

# Check for regressions
pytest tests/performance/ --benchmark-only --benchmark-compare-fail=mean:5%
```

#### 3. Performance Test Development
```python
@pytest.mark.performance
def test_correlation_performance(benchmark, large_correlation_dataset):
    """Benchmark correlation calculation performance."""

    def run_correlation():
        return calculate_g2_correlation(large_correlation_dataset)

    # Run benchmark
    result = benchmark.pedantic(
        run_correlation,
        rounds=10,
        iterations=5
    )

    # Validate result quality
    assert result is not None
    assert len(result) > 0

    # Check against baseline
    baseline = get_baseline("g2_calculation")
    assert benchmark.stats.mean < baseline["max_time"]
```

#### 4. Memory Performance Testing
```python
def test_memory_efficiency(large_dataset):
    """Test memory usage efficiency."""
    import psutil

    process = psutil.Process()
    baseline_memory = process.memory_info().rss / 1024 / 1024

    # Operation under test
    with memory_limit(100):  # 100MB limit
        result = memory_intensive_operation(large_dataset)

    peak_memory = process.memory_info().rss / 1024 / 1024
    memory_used = peak_memory - baseline_memory

    assert memory_used < 50, f"Used {memory_used:.1f}MB, limit is 50MB"
```

## Code Review Testing Checklist

### Pre-Review Checklist (Author)

- [ ] **All tests pass locally**
  ```bash
  make test-ci
  ```

- [ ] **Test coverage maintained or improved**
  ```bash
  make coverage | grep "TOTAL"
  ```

- [ ] **New tests follow quality standards**
  ```bash
  python tests/quality_standards.py --check-all | grep "new_feature"
  ```

- [ ] **Performance tests pass**
  ```bash
  make test-performance
  ```

- [ ] **Documentation updated**
  - Test docstrings are clear
  - TESTING.md updated if needed
  - Examples work correctly

### Review Checklist (Reviewer)

#### Test Quality Review
- [ ] **Test names are descriptive**
  - Test name explains what is being validated
  - Follows naming convention: `test_[feature]_[condition]_[expected_result]`

- [ ] **Test docstrings explain purpose**
  - What the test validates
  - Why the test is important
  - Any special setup or conditions

- [ ] **Assertions are specific and meaningful**
  - No bare `assert result` statements
  - Assertion messages provide context
  - Appropriate tolerance for numerical comparisons

#### Scientific Rigor Review
- [ ] **Numerical comparisons use explicit tolerances**
```python
# Good
np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-14)

# Bad
assert actual == expected
```

- [ ] **Physical constraints are validated**
```python
# Validate correlation function properties
ScientificAssertions.assert_correlation_properties(tau, g2)
```

- [ ] **Edge cases are tested**
  - Empty inputs
  - Single data points
  - Very large datasets
  - Boundary conditions

#### Test Structure Review
- [ ] **Tests follow AAA pattern** (Arrange-Act-Assert)
- [ ] **Appropriate use of fixtures**
- [ ] **Proper cleanup in teardown**
- [ ] **Tests are independent** (can run in any order)

### Review Commands
```bash
# Review test changes
git diff origin/master -- tests/

# Check test quality for changed files
python tests/quality_standards.py --check-all | grep -f <(git diff --name-only origin/master -- tests/ | sed 's|tests/||')

# Run tests affected by changes
pytest $(git diff --name-only origin/master -- tests/) -v

# Check coverage impact
pytest --cov=xpcs_toolkit --cov-report=term-missing $(git diff --name-only origin/master -- "*.py" | grep -v "test_")
```

## CI/CD Integration

### GitHub Actions Workflow

Create `.github/workflows/tests.yml`:
```yaml
name: Test Suite

on:
  push:
    branches: [ master, develop ]
  pull_request:
    branches: [ master ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.9', '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[test]"

    - name: Run test quality check
      run: |
        python tests/quality_standards.py --check-all --format json --output quality_report.json

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --junitxml=junit/test-results-unit.xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --junitxml=junit/test-results-integration.xml

    - name: Run scientific tests
      run: |
        pytest tests/scientific/ -v --junitxml=junit/test-results-scientific.xml

    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-json=benchmark-results.json

    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.os }}-${{ matrix.python-version }}
        path: |
          junit/
          quality_report.json
          benchmark-results.json

  coverage:
    runs-on: ubuntu-latest
    needs: test

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -e ".[test]"

    - name: Generate coverage report
      run: |
        pytest tests/ --cov=xpcs_toolkit --cov-report=xml --cov-report=html

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
```

### Pre-commit Hooks

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: '23.3.0'
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: 'v0.0.270'
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: local
    hooks:
      - id: test-quality-check
        name: Test Quality Check
        entry: python tests/quality_standards.py --check-all
        language: system
        pass_filenames: false
        always_run: true

      - id: unit-tests
        name: Unit Tests
        entry: pytest tests/unit/ -x
        language: system
        pass_filenames: false
        types: [python]
```

## Team Collaboration

### Test Ownership and Responsibilities

#### Individual Developer Responsibilities
- Write tests for all new features
- Maintain existing tests when modifying code
- Run relevant test suites before committing
- Fix failing tests promptly
- Update test documentation

#### Team Lead Responsibilities
- Review test architecture and patterns
- Ensure test quality standards are maintained
- Coordinate test infrastructure updates
- Resolve test environment issues
- Monitor test suite performance and coverage

#### Release Manager Responsibilities
- Run comprehensive test validation before releases
- Coordinate cross-platform testing
- Manage test baselines and benchmarks
- Ensure CI/CD pipeline health
- Document test results for release notes

### Communication Protocols

#### Test Failure Communication
1. **Immediate notification** for master branch test failures
2. **Daily digest** of test quality metrics
3. **Weekly summary** of test coverage and performance trends
4. **Release summary** of test validation results

#### Test Infrastructure Changes
1. **Propose changes** in team meeting
2. **Document impact** on existing tests
3. **Coordinate migration** timeline
4. **Validate changes** across all environments

### Knowledge Sharing

#### Test Documentation Standards
- All test patterns documented in team wiki
- Common debugging scenarios shared
- Performance optimization techniques documented
- Regular test architecture reviews

#### Training and Onboarding
- New team members shadow experienced developers
- Test writing workshops and code reviews
- Documentation of testing best practices
- Regular updates on new testing tools and techniques

---

This developer workflow guide ensures that all team members can effectively contribute to and maintain the high-quality test suite that supports the XPCS Toolkit's scientific computing requirements.
