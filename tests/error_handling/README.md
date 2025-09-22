# XPCS Toolkit Error Handling & Edge Case Test Suite

This directory contains comprehensive error handling and edge case tests for the XPCS Toolkit, designed to ensure robust operation under all error conditions and boundary scenarios.

## Overview

The error handling test suite provides systematic validation of:

- **File I/O Error Conditions**: Corrupted files, permission issues, disk space problems
- **Data Validation Errors**: Invalid formats, missing datasets, type mismatches
- **Memory Management Errors**: Out-of-memory conditions, allocation failures, memory pressure
- **GUI Error Handling**: Widget failures, threading issues, display problems
- **Edge Cases**: Boundary values, extreme inputs, special numerical values
- **Resource Exhaustion**: Memory, disk, file handles, CPU, network limitations
- **Error Recovery**: Cleanup procedures, graceful degradation, system stability
- **Error Injection**: Systematic error injection and validation framework

## Test Modules

### Core Error Handling Tests

#### `test_file_io_errors.py`
Comprehensive file I/O error condition testing:
- **Corrupted HDF5 files**: Handles truncated, malformed, and corrupted files
- **Permission errors**: Tests read/write permission denied scenarios
- **Missing files**: Validates non-existent file handling
- **Connection pool errors**: Tests HDF5 connection pool resilience
- **Batch operations**: Error handling in bulk file operations
- **Resource cleanup**: Ensures proper cleanup after I/O failures

Key test classes:
- `TestFileIOErrors`: Basic I/O error scenarios
- `TestXpcsFileErrors`: XPCS-specific file handling
- `TestFileLocatorErrors`: Directory and path validation
- `TestResourceExhaustionErrors`: File handle and disk space limits
- `TestErrorRecoveryAndCleanup`: Post-error system stability

#### `test_data_validation_errors.py`
Data format and validation error testing:
- **Invalid NeXus structure**: Malformed HDF5/NeXus files
- **Missing datasets**: Required XPCS datasets absent
- **Wrong dimensions**: Incorrect array shapes and sizes
- **Data type mismatches**: String vs numeric, complex vs real
- **Special values**: Infinite, NaN, and extreme value handling
- **Range violations**: Negative values where positive expected
- **Consistency checks**: Mismatched array dimensions across datasets

Key test classes:
- `TestDataFormatValidationErrors`: File format validation
- `TestDataRangeValidation`: Value range and boundary checks
- `TestModuleSpecificValidation`: Analysis module data requirements
- `TestMetadataValidation`: Metadata consistency and completeness
- `TestDataConsistencyValidation`: Cross-dataset validation

#### `test_memory_errors.py`
Memory-related error handling and resource management:
- **Allocation failures**: Out-of-memory conditions and recovery
- **Memory pressure detection**: Monitoring and adaptive behavior
- **Leak detection**: Resource cleanup and garbage collection
- **Large dataset handling**: Memory-efficient processing strategies
- **Concurrent access**: Memory management under threading
- **Cache management**: Memory-aware caching systems

Key test classes:
- `TestMemoryAllocationErrors`: Basic allocation failure scenarios
- `TestMemoryLeakDetection`: Resource leak prevention
- `TestMemoryPressureHandling`: Adaptive behavior under pressure
- `TestMemoryMonitoringSystem`: Memory monitoring accuracy
- `TestMemoryErrorRecovery`: Recovery mechanisms

#### `test_gui_errors.py`
GUI error handling and user interface resilience:
- **Widget creation failures**: UI component initialization errors
- **Threading errors**: GUI thread safety and signal/slot issues
- **Display errors**: Plot rendering and image display problems
- **User input validation**: Invalid parameter and text input handling
- **Error message display**: User-friendly error reporting
- **GUI recovery**: Interface stability after errors

Key test classes:
- `TestWidgetCreationErrors`: UI component failure handling
- `TestThreadingErrors`: Thread safety and synchronization
- `TestDisplayErrors`: Graphics and visualization errors
- `TestUserInputValidation`: Input validation and sanitization
- `TestPlotHandlerErrors`: Plotting system error handling
- `TestXpcsViewerErrors`: Application-specific GUI errors

### Edge Case and Boundary Testing

#### `test_edge_cases.py`
Comprehensive boundary condition and edge case testing:
- **Boundary values**: Zero, minimum, and maximum values
- **Special numerical values**: Infinity, NaN, precision limits
- **Extreme array shapes**: Very long, wide, or thin arrays
- **Data type edges**: Integer overflow, floating-point precision
- **Parameter validation**: Correlation times, q-values, geometry limits
- **Concurrency edges**: Race conditions and thread safety

Key test classes:
- `TestBoundaryValues`: Numerical boundary conditions
- `TestSpecialNumericalValues`: Infinite, NaN, and extreme values
- `TestExtremeArrayShapes`: Unusual array dimensions
- `TestDataTypeEdgeCases`: Type conversion and overflow
- `TestParameterValidationEdgeCases`: Scientific parameter limits
- `TestConcurrencyEdgeCases`: Multi-threading edge cases

#### `test_resource_exhaustion.py`
Resource exhaustion simulation and handling:
- **Memory exhaustion**: Gradual and sudden memory depletion
- **Disk space limits**: Full disk and cleanup procedures
- **File handle limits**: Operating system handle exhaustion
- **CPU resource limits**: High-load and timeout scenarios
- **Network failures**: Connectivity and timeout handling
- **Recovery mechanisms**: Automatic cleanup and adaptation

Key test classes:
- `TestMemoryExhaustion`: Memory depletion scenarios
- `TestDiskSpaceExhaustion`: Storage limitation handling
- `TestFileHandleExhaustion`: File descriptor management
- `TestCPUResourceExhaustion`: Processing load limits
- `TestNetworkResourceExhaustion`: Network failure simulation
- `TestResourceRecoveryMechanisms`: Automatic recovery systems

### Recovery and Stability Testing

#### `test_error_recovery.py`
Error recovery and system stability validation:
- **Recovery mechanisms**: System resilience after failures
- **Graceful degradation**: Partial functionality maintenance
- **State consistency**: Data integrity after errors
- **Resource cleanup**: Memory and handle management
- **Cleanup procedures**: Automatic and manual cleanup validation

Key test classes:
- `TestErrorRecoveryMechanisms`: Post-error system recovery
- `TestGracefulDegradation`: Reduced functionality modes
- `TestSystemStabilityAfterErrors`: Long-term stability
- `TestCleanupProcedures`: Resource management validation

#### `test_error_injection_framework.py`
Systematic error injection and validation framework:
- **Controlled error injection**: Systematic failure simulation
- **Scenario-based testing**: Predefined error conditions
- **Recovery validation**: Post-injection system health
- **Comprehensive coverage**: Multiple error types and conditions
- **Statistical analysis**: Error handling effectiveness metrics

Key classes:
- `SystematicErrorInjector`: Advanced error injection system
- `ErrorInjectionTestSuite`: Comprehensive test suite runner
- `TestErrorInjectionFramework`: Framework validation
- `TestSystematicErrorScenarios`: Scenario-based testing
- `TestComprehensiveErrorInjection`: Full suite execution

## Test Configuration and Fixtures

### `conftest.py`
Comprehensive fixture library for error testing:
- **Error simulation fixtures**: Corrupted files, limited resources
- **Mock environments**: Memory pressure, disk limitations
- **Error injection utilities**: Controlled failure introduction
- **Edge case data**: Boundary values and special cases
- **Resource monitoring**: Memory and performance tracking

Key fixtures:
- `corrupted_hdf5_file`: Simulated file corruption
- `memory_limited_environment`: Memory pressure simulation
- `error_injector`: Systematic error injection utility
- `edge_case_data`: Boundary and special value datasets
- `resource_exhaustion`: Resource limitation simulation

## Running Error Handling Tests

### Basic Test Execution
```bash
# Run all error handling tests
python -m pytest tests/error_handling/ -v

# Run specific test modules
python -m pytest tests/error_handling/test_file_io_errors.py -v
python -m pytest tests/error_handling/test_memory_errors.py -v
python -m pytest tests/error_handling/test_edge_cases.py -v

# Run with coverage
python -m pytest tests/error_handling/ --cov=xpcs_toolkit --cov-report=html
```

### Targeted Test Categories
```bash
# File I/O and corruption tests
python -m pytest tests/error_handling/test_file_io_errors.py::TestFileIOErrors -v

# Memory management tests
python -m pytest tests/error_handling/test_memory_errors.py::TestMemoryAllocationErrors -v

# GUI error handling (requires display)
python -m pytest tests/error_handling/test_gui_errors.py -m gui -v

# Edge case boundary testing
python -m pytest tests/error_handling/test_edge_cases.py::TestBoundaryValues -v

# Resource exhaustion simulation
python -m pytest tests/error_handling/test_resource_exhaustion.py -m slow -v
```

### Advanced Test Options
```bash
# Run error injection framework tests
python -m pytest tests/error_handling/test_error_injection_framework.py -v

# Run with specific markers
python -m pytest tests/error_handling/ -m "not slow" -v  # Skip slow tests
python -m pytest tests/error_handling/ -m "not gui" -v   # Skip GUI tests

# Stress testing with error injection
python -m pytest tests/error_handling/test_error_injection_framework.py::TestComprehensiveErrorInjection -v
```

## Test Categories and Markers

The test suite uses pytest markers for categorization:

- `@pytest.mark.unit`: Unit-level error handling tests
- `@pytest.mark.integration`: Component integration error tests
- `@pytest.mark.gui`: GUI error handling tests (requires display)
- `@pytest.mark.slow`: Long-running stress and exhaustion tests
- `@pytest.mark.parametrize`: Parameterized edge case testing

## Expected Behaviors

### Error Handling Principles
1. **Graceful Degradation**: System provides partial functionality when components fail
2. **Resource Cleanup**: All resources properly released after errors
3. **State Consistency**: System state remains valid after error conditions
4. **User Communication**: Clear, actionable error messages for users
5. **Recovery Capability**: System can recover and continue operation after errors

### Performance Under Stress
- Memory usage should stabilize after allocation failures
- File handle counts should not grow unbounded
- System should remain responsive under resource pressure
- Error handling should not significantly impact normal operation performance

### Validation Criteria
- **Error Detection**: All error conditions properly detected and handled
- **Recovery Success**: System recovers to functional state after errors
- **Resource Stability**: No memory leaks or handle exhaustion
- **User Experience**: Errors reported clearly without system crashes
- **Data Integrity**: Scientific data remains valid through error conditions

## Integration with Main Test Suite

The error handling tests integrate with the main XPCS Toolkit test suite:

```bash
# Run all tests including error handling
python -m pytest tests/ -v

# Include in continuous integration
python -m pytest tests/error_handling/ --junitxml=error_handling_results.xml
```

## Extending Error Handling Tests

### Adding New Error Scenarios
1. **Identify Error Condition**: Specific failure mode to test
2. **Create Test Fixture**: Simulate the error condition reliably
3. **Implement Test Cases**: Validate error detection and handling
4. **Add Recovery Tests**: Ensure system recovers properly
5. **Document Expected Behavior**: Update test documentation

### Custom Error Injection
```python
# Example: Custom error injection
from tests.error_handling.conftest import ErrorInjector

def test_custom_error_scenario(error_injector):
    # Inject specific error condition
    error_injector.inject_io_error('target.function', CustomError)

    # Test system behavior under error
    with pytest.raises(CustomError):
        # Operation that should fail
        perform_operation()

    # Validate recovery
    error_injector.cleanup()
    assert system_is_functional()
```

## Troubleshooting Common Issues

### Test Environment Setup
- Ensure sufficient disk space for test file creation
- Verify memory availability for allocation failure tests
- Check file permissions for corruption simulation tests
- Install GUI dependencies for display error tests

### Platform-Specific Considerations
- Windows: File permission tests may behave differently
- macOS: Memory monitoring might have different characteristics
- Linux: File handle limits vary by distribution
- Headless systems: GUI tests will be automatically skipped

### Performance Tuning
- Adjust memory limits for slower systems
- Reduce iteration counts for resource-constrained environments
- Use selective test execution for faster development cycles
- Enable parallel execution where thread-safe

This comprehensive error handling test suite ensures the XPCS Toolkit maintains robust operation under all error conditions, providing scientists with reliable data analysis capabilities even when facing unexpected failures or extreme data scenarios.
