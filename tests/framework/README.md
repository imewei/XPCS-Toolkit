# XPCS Toolkit Testing Framework

This directory contains the testing framework tools and utilities for the XPCS Toolkit project. The framework provides comprehensive quality assessment, performance monitoring, integration validation, and test maintenance automation.

## Framework Components

### Core Framework Tools

#### `quality_standards.py` (24KB)
Test quality assessment and standards enforcement tool.

**Purpose**: Analyzes test files for quality metrics and enforces coding standards.

**Key Features**:
- Automated test quality scoring
- Docstring completeness analysis
- Assertion strength evaluation
- Code complexity analysis
- Best practices enforcement

**Usage**:
```python
from tests.framework.quality_standards import TestQualityChecker

checker = TestQualityChecker(test_directory_path)
metrics = checker.check_all_tests()
```

#### `performance_monitor.py` (36KB)
Performance monitoring and regression detection tool.

**Purpose**: Monitors test performance and detects performance regressions.

**Key Features**:
- Real-time performance monitoring
- Historical performance tracking
- Regression detection algorithms
- Resource usage analysis
- Performance trend analysis
- Automated alerting for performance degradation

**Usage**:
```python
from tests.framework.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()
# ... run tests ...
report = monitor.get_performance_report()
```

#### `integration_validator.py` (28KB)
Cross-platform validation and integration testing tool.

**Purpose**: Validates test integration across different platforms and environments.

**Key Features**:
- Cross-platform compatibility validation
- Environment-specific testing
- Integration test coordination
- Dependency validation
- Platform-specific test execution

**Usage**:
```python
from tests.framework.integration_validator import IntegrationValidator

validator = IntegrationValidator()
validation_results = validator.validate_integration()
```

#### `maintenance_framework.py` (32KB)
Test maintenance automation and health monitoring.

**Purpose**: Automates test maintenance tasks and monitors test suite health.

**Key Features**:
- Automated test maintenance
- Test suite health monitoring
- Dead code detection
- Test coverage analysis
- Maintenance recommendations
- Automated fix suggestions

**Usage**:
```python
from tests.framework.maintenance_framework import TestMaintenanceFramework

maintenance = TestMaintenanceFramework(test_directory)
health_report = maintenance.generate_health_report()
```

#### `utils.py` (20KB)
Framework utilities and helper functions.

**Purpose**: Provides common utilities used across framework components.

**Key Features**:
- Scientific test assertions
- Performance timing utilities
- Test data generation helpers
- Common test decorators
- Framework configuration utilities

**Usage**:
```python
from tests.framework.utils import ScientificAssertions, PerformanceTimer

with PerformanceTimer() as timer:
    # ... test code ...
    pass
```

## Runner Scripts

The `runners/` subdirectory contains test execution and validation runner scripts.

### `runners/run_validation.py` (16KB)
General test validation runner with configuration support.

**Purpose**: Orchestrates comprehensive test suite validation.

**Features**:
- Configuration-driven validation
- Multiple validation modes
- Detailed reporting
- Integration with framework tools
- Custom validation workflows

**Usage**:
```bash
# From project root
python -m tests.framework.runners.run_validation

# With configuration file
python -m tests.framework.runners.run_validation --config validation_config.yaml
```

### `runners/run_scientific_validation.py` (20KB)
Scientific validation runner for XPCS algorithms.

**Purpose**: Specialized runner for scientific algorithm validation.

**Features**:
- XPCS-specific algorithm validation
- Scientific accuracy testing
- Cross-validation frameworks
- Analytical benchmarking
- Reference data validation

**Usage**:
```bash
# Run scientific validation
python -m tests.framework.runners.run_scientific_validation

# Run with specific test categories
python -m tests.framework.runners.run_scientific_validation --tests g2,saxs,twotime
```

## Framework Usage Patterns

### 1. Quality Assessment Workflow
```python
from tests.framework.quality_standards import TestQualityChecker
from tests.framework.maintenance_framework import TestMaintenanceFramework

# Assess test quality
checker = TestQualityChecker("tests/")
quality_metrics = checker.check_all_tests()

# Get maintenance recommendations
maintenance = TestMaintenanceFramework("tests/")
recommendations = maintenance.get_maintenance_recommendations()
```

### 2. Performance Monitoring Workflow
```python
from tests.framework.performance_monitor import PerformanceMonitor

# Start monitoring
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Run your tests here...

# Generate performance report
report = monitor.generate_report()
regression_alerts = monitor.check_for_regressions()
```

### 3. Integration Validation Workflow
```python
from tests.framework.integration_validator import IntegrationValidator

# Validate cross-platform compatibility
validator = IntegrationValidator()
validation_results = validator.run_full_validation()

# Check specific platform compatibility
platform_results = validator.validate_platform("linux")
```

## Framework Configuration

### Configuration Files
The framework supports YAML configuration files for customizing behavior:

```yaml
# framework_config.yaml
quality_standards:
  min_docstring_coverage: 0.8
  max_complexity_score: 10
  required_assertions_per_test: 2

performance_monitoring:
  regression_threshold: 0.15
  memory_threshold_mb: 500
  cpu_threshold_percent: 80

integration_validation:
  platforms: ["linux", "darwin", "win32"]
  python_versions: ["3.9", "3.10", "3.11"]
  test_timeout: 300
```

### Environment Variables
- `XPCS_TEST_FRAMEWORK_CONFIG`: Path to framework configuration file
- `XPCS_PERFORMANCE_THRESHOLD`: Override performance regression threshold
- `XPCS_TEST_TIMEOUT`: Override default test timeout

## Integration with Test Suite

### Importing Framework Components
```python
# Import framework tools
from tests.framework import (
    quality_standards,
    performance_monitor,
    integration_validator,
    maintenance_framework,
    utils
)

# Import runners
from tests.framework.runners import run_validation, run_scientific_validation
```

### Using Framework Utilities in Tests
```python
from tests.framework.utils import scientific_test, PerformanceTimer

@scientific_test
def test_g2_analysis():
    with PerformanceTimer() as timer:
        # ... test implementation ...
        pass

    assert timer.elapsed < 5.0  # Performance assertion
```

## Framework Development

### Adding New Framework Components
1. Create new module in `tests/framework/`
2. Follow existing patterns for configuration and initialization
3. Add comprehensive docstrings and type hints
4. Update `__init__.py` to expose new components
5. Add usage examples to this README

### Best Practices for Framework Tools
- Use type hints for all public APIs
- Provide comprehensive error handling
- Support configuration through YAML files
- Include performance monitoring capabilities
- Write comprehensive unit tests
- Document all public methods and classes

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're importing from the correct paths after the reorganization
2. **Configuration Issues**: Check YAML syntax and file paths
3. **Permission Errors**: Ensure framework has read/write access to test directories
4. **Performance Issues**: Monitor memory usage during large test suite runs

### Debug Mode
Enable debug logging for framework components:

```python
import logging
logging.getLogger('tests.framework').setLevel(logging.DEBUG)
```

## Contributing

When contributing to the testing framework:

1. Follow the established patterns for error handling and logging
2. Add comprehensive tests for new framework components
3. Update this README with new features and usage patterns
4. Ensure backward compatibility with existing test suites
5. Performance test any new monitoring or validation features

## Framework Dependencies

The framework relies on several external packages:
- `psutil` for system monitoring
- `pyyaml` for configuration parsing
- `numpy` and `scipy` for scientific computations
- `matplotlib` for performance visualizations
- Standard library modules for core functionality

For the complete list, see the project's `pyproject.toml` dependencies.