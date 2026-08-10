# XPCS Toolkit Test Suite

A comprehensive, multi-layered testing framework designed for scientific computing applications, ensuring accuracy, performance, and maintainability.

## Quick Start

```bash
# Install test dependencies
pip install -e ".[test]"

# Run basic tests
make test

# Run with coverage
make coverage

# Quality check
python tests/framework/quality_standards.py --check-all
```

## Architecture Overview

### Test Categories

```
tests/
├── unit/                    # Component isolation tests
│   ├── core/               # Core system components
│   │   ├── test_xpcs_file.py             # Existing XpcsFile tests
│   │   ├── test_xpcs_file_data_access.py # G2, TwoTime data access (XF-001→XF-006)
│   │   ├── test_xpcs_file_fitting.py     # G2 fitting tests (XF-009→XF-012)
│   │   ├── test_xpcs_file_roi.py         # ROI extraction tests (XF-007, XF-008)
│   │   ├── test_viewer_kernel.py         # Existing ViewerKernel tests
│   │   ├── test_viewer_kernel_plotting.py # Plotting tests (VK-001→VK-005)
│   │   └── test_viewer_kernel_export.py  # Export tests (VK-006→VK-009)
│   ├── schemas/            # Schema validation tests
│   │   └── test_validators.py            # QMapSchema, G2Data, MaskSchema validation
│   ├── analysis/           # Analysis module tests
│   ├── fileio/             # File I/O system tests
│   ├── threading/          # Threading and async tests
│   ├── utils/              # Utility function tests
│   └── test_package_basics.py # Basic package functionality
├── integration/            # Component interaction tests
│   └── test_hdf5_facade.py # HDF5 facade read/write tests
├── scientific/             # Scientific accuracy validation
│   ├── algorithms/         # Algorithm validation
│   ├── properties/         # Mathematical property tests
│   └── reference_validation/ # Cross-validation framework
├── logging/               # Comprehensive logging tests
│   ├── unit/              # Logging component tests
│   ├── functional/        # End-to-end logging tests
│   ├── performance/       # Logging performance tests
│   └── properties/        # Logging property tests
├── performance/           # Performance benchmarks
│   ├── benchmarks/        # Core performance tests
│   ├── config/           # Performance configuration
│   └── utils/            # Performance utilities
├── framework/             # Test framework tools
│   └── runners/          # Test execution runners
├── fixtures/              # Test data and fixtures
│   ├── reference_data/   # Reference datasets
│   └── xpcs_synthetic.py # XPCS synthetic data generators
├── gui_interactive/       # User interface tests
├── end_to_end/           # Complete workflow tests
└── error_handling/        # Error scenarios and edge cases
```

### Quality Assurance Framework

- **Automated Quality Checking**: `framework/quality_standards.py`
- **Test Suite Validation**: `test_suite_validation.py`
- **Performance Monitoring**: `framework/performance_monitor.py`
- **Test Execution Runners**: `framework/runners/`
- **Reference Data Management**: `fixtures/reference_data/`

## Key Features

### 🔬 Scientific Rigor
- Explicit numerical tolerances
- Physical constraint validation
- Property-based testing with Hypothesis
- Cross-validation against reference implementations

### ⚡ Performance Monitoring
- Real-time execution monitoring
- Regression detection
- Resource usage tracking (CPU, memory, I/O)
- Automated optimization recommendations

### 🛠️ Developer Experience
- IDE integration guides (VS Code, PyCharm)
- Test debugging tools and profiling
- Quality standards enforcement
- Comprehensive documentation

### 🔄 Maintenance & Evolution
- Automated maintenance scheduling
- Technical debt tracking
- Health metrics monitoring
- Cross-platform compatibility validation

## Usage Examples

### Running Specific Test Categories

```bash
# Scientific accuracy tests
make test-scientific

# Performance benchmarks
make test-performance

# Logging system tests
python -m pytest tests/logging/ -v

# Framework validation
python tests/framework/runners/run_validation.py

# Test quality assessment
python tests/framework/quality_standards.py --check-all --format json
```

### Writing Quality Tests

```python
from tests.framework.utils import (
    ScientificAssertions,
    scientific_test,
    PerformanceTimer,
)


class TestCorrelationAnalysis:
    @scientific_test(tolerance=1e-6)
    def test_g2_calculation_accuracy(self, synthetic_correlation_data):
        """Test G2 calculation maintains scientific accuracy."""
        result = calculate_g2(synthetic_correlation_data["raw"])

        ScientificAssertions.assert_arrays_close(
            result,
            synthetic_correlation_data["expected"],
            rtol=1e-6,
            atol=1e-12,
            msg="G2 calculation accuracy validation",
        )

        # Validate physical constraints
        ScientificAssertions.assert_correlation_properties(
            synthetic_correlation_data["tau"], result
        )
```

### Performance Monitoring

```bash
# Run performance profiling
python tests/framework/performance_monitor.py --profile

# View performance dashboard
python tests/framework/performance_monitor.py --dashboard

# Get optimization recommendations
python tests/framework/performance_monitor.py --optimize
```

## Documentation

- **[Comprehensive Testing Guide](../docs/TESTING_COMPREHENSIVE.md)**: Complete testing framework documentation
- **[Developer Workflow Guide](../docs/TESTING_DEVELOPER_GUIDE.md)**: Daily development practices and IDE setup
- **[Maintenance Procedures](maintenance_framework.py)**: Automated maintenance and evolution

## Quality Standards

The framework enforces quality standards through automated checking:

- **Docstring Coverage**: All tests documented
- **Assertion Quality**: Specific assertions with meaningful messages
- **Scientific Rigor**: Explicit tolerances for numerical comparisons
- **Test Patterns**: Proper setup/teardown and fixture usage
- **Performance Awareness**: Resource usage monitoring

Target quality scores:
- **Excellent** (85-100): Production-ready
- **Good** (70-84): Minor improvements needed
- **Fair** (50-69): Significant improvement required
- **Poor** (0-49): Major rewrite needed

## Continuous Integration

The framework supports CI/CD with:
- Cross-platform testing (Windows, macOS, Linux)
- Multiple Python versions (3.9, 3.10, 3.11)
- Performance regression detection
- Quality gate enforcement
- Automated reporting

## Contributing

When adding new tests:

1. **Follow the quality checklist** in the developer guide
2. **Use appropriate test category** and markers
3. **Include comprehensive docstrings**
4. **Add performance considerations**
5. **Run quality assessment** before committing

## Maintenance

The framework includes automated maintenance:

```bash
# Daily maintenance
python tests/maintenance_framework.py --daily

# Weekly deep analysis
python tests/maintenance_framework.py --weekly

# Release preparation
python tests/maintenance_framework.py --release v1.0.0
```

## Support

- **Issues**: Use the quality checker to identify problems
- **Performance**: Check the performance monitor for bottlenecks
- **Cross-platform**: Run integration validator for compatibility
- **Documentation**: Refer to comprehensive guides for detailed information

---

This testing framework ensures the XPCS Toolkit maintains the highest standards of scientific software quality while remaining maintainable and efficient for daily development.
