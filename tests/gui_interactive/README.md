# XPCS Toolkit GUI Interactive Tests

This directory contains comprehensive automated GUI testing infrastructure for the XPCS Toolkit using pytest-qt framework. The tests validate user interface functionality, interactions, performance, and error handling across all analysis modules.

## Overview

The GUI testing framework provides:

- **Comprehensive Tab Testing**: All analysis tabs (SAXS 2D/1D, G2, diffusion, two-time, stability)
- **Plot Widget Testing**: PyQtGraph and Matplotlib integration validation
- **File Operation Testing**: Directory selection, file loading, data validation
- **User Scenario Testing**: Complete workflow automation and validation
- **Error Handling Testing**: Robustness under various failure conditions
- **Performance Testing**: Responsiveness, memory usage, and scalability validation

## Test Structure

```
tests/gui_interactive/
├── README.md                    # This file
├── conftest.py                  # Pytest configuration and fixtures
├── run_gui_tests.py            # Test runner script
├── test_main_window.py         # Main window and tab management tests
├── test_analysis_tabs.py       # Analysis tab-specific functionality tests
├── test_plot_interactions.py   # Plot widget interaction and visualization tests
├── test_file_operations.py     # File loading, data management tests
├── test_user_scenarios.py      # Complete user workflow scenario tests
├── test_error_handling.py      # Error conditions and edge case tests
└── test_performance.py         # Performance and responsiveness tests
```

## Quick Start

### Prerequisites

Ensure you have the required testing dependencies:

```bash
pip install pytest pytest-qt psutil
```

### Running Tests

Use the provided test runner for various testing scenarios:

```bash
# Quick validation tests (recommended for development)
python tests/gui_interactive/run_gui_tests.py quick

# Full GUI test suite
python tests/gui_interactive/run_gui_tests.py full

# Performance tests
python tests/gui_interactive/run_gui_tests.py performance

# Error handling tests
python tests/gui_interactive/run_gui_tests.py errors

# CI-friendly headless tests
python tests/gui_interactive/run_gui_tests.py ci
```

### Running Specific Test Files

```bash
# Run specific test file
python tests/gui_interactive/run_gui_tests.py test_main_window.py

# Run with pytest directly
python -m pytest tests/gui_interactive/test_analysis_tabs.py -v
```

## Test Categories

### 1. Main Window Tests (`test_main_window.py`)

- **Window Initialization**: Main window creation, title, visibility
- **Tab Management**: Tab switching, state persistence, navigation
- **Menu Functionality**: Menu bar operations, keyboard shortcuts
- **Status Updates**: Status bar messages, progress indication
- **Window Behavior**: Resizing, multiple instances, close handling

### 2. Analysis Tab Tests (`test_analysis_tabs.py`)

- **SAXS 2D Tab**: Image display, colorbar controls, zoom/pan functionality
- **SAXS 1D Tab**: Plot controls, log/linear scaling, axis management
- **G2 Analysis Tab**: Correlation plotting, fitting controls, parameter adjustment
- **Diffusion Tab**: Diffusion analysis interface validation
- **Two-Time Tab**: Correlation matrix display, parameter controls
- **Stability Tab**: Time series plotting, analysis controls
- **Cross-Tab Integration**: Data consistency, state preservation

### 3. Plot Interaction Tests (`test_plot_interactions.py`)

- **PyQtGraph Integration**: Plot creation, zoom, pan, data updates
- **Matplotlib Integration**: Canvas interaction, plot customization
- **XPCS-Specific Plotting**: G2 curves, SAXS images, stability plots
- **Plot Customization**: Axis labels, grid, color schemes, legends
- **Performance**: Large dataset plotting, rapid updates, memory usage

### 4. File Operation Tests (`test_file_operations.py`)

- **Directory Selection**: Browser dialogs, path display, recent directories
- **File Loading**: Single/multiple file loading, validation, progress indication
- **File List Management**: Display, selection, context menus
- **Drag and Drop**: File drop handling, multiple files, invalid files
- **Error Handling**: Corrupted files, missing files, permission errors

### 5. User Scenario Tests (`test_user_scenarios.py`)

- **Complete Analysis Workflows**: SAXS → G2 → fitting workflows
- **Multi-Tab Navigation**: Sequential exploration, rapid switching
- **Parameter Adjustment**: Systematic control testing, validation
- **Data Visualization**: Plot customization workflows, multi-dataset comparison
- **Error Recovery**: Analysis failures, memory issues, UI stability

### 6. Error Handling Tests (`test_error_handling.py`)

- **Data Loading Errors**: Corrupted files, malformed data, permission issues
- **Memory/Resource Errors**: Large datasets, disk space, thread exhaustion
- **Calculation Errors**: Fitting failures, numerical overflow, division by zero
- **UI Boundary Conditions**: Extreme window sizes, rapid interactions, parameter limits
- **Concurrency Issues**: Thread safety, race conditions, resource cleanup

### 7. Performance Tests (`test_performance.py`)

- **Response Times**: Tab switching, button clicks, parameter adjustments
- **Memory Usage**: Tab switching memory, plot memory, widget cleanup
- **Rendering Performance**: Plot rendering, image display, rapid updates
- **Scalability**: Large files, many widgets, concurrent operations
- **Resource Utilization**: CPU usage, thread efficiency, baseline metrics

## Test Fixtures and Utilities

### Core Fixtures (`conftest.py`)

- **`gui_main_window`**: Main application window with mocked backend
- **`mock_xpcs_file`**: Mock XPCS data file for testing
- **`mock_xpcs_data`**: Synthetic XPCS datasets
- **`temp_hdf5_files`**: Temporary HDF5 files for file operation tests

### Advanced Testing Utilities

- **`gui_widget_inspector`**: Widget property analysis and state validation
- **`gui_interaction_recorder`**: Record and replay user interactions
- **`gui_state_validator`**: Comprehensive GUI state consistency checking
- **`gui_accessibility_helper`**: Keyboard navigation and accessibility testing
- **`gui_error_simulator`**: Controlled error injection for robustness testing
- **`gui_performance_monitor`**: Performance timing and memory monitoring

## Testing Best Practices

### 1. Test Isolation

Each test is isolated using pytest fixtures that create fresh instances:

```python
def test_tab_switching(self, gui_main_window, qtbot, gui_test_helpers):
    window = gui_main_window  # Fresh window instance
    # Test implementation
```

### 2. Asynchronous Operations

Use `qtbot.wait()` for GUI operations that need time to complete:

```python
qtbot.mouseClick(button, QtCore.Qt.MouseButton.LeftButton)
qtbot.wait(100)  # Allow GUI update time
```

### 3. Mock Data Usage

Use mock data to avoid dependency on real XPCS files:

```python
mock_xpcs_file.get_g2.return_value = (taus, g2_data, g2_err)
with patch.object(window.vk, 'current_file', mock_xpcs_file):
    # Test with controlled data
```

### 4. Error Simulation

Test error conditions with controlled injection:

```python
gui_error_simulator.simulate_analysis_error(mock_file)
# Test error handling without actual errors
```

## Performance Benchmarks

The performance tests establish baseline metrics for regression detection:

- **Tab Switching**: < 100ms per switch
- **Button Clicks**: < 50ms response time
- **Plot Updates**: < 50ms per update
- **Memory Growth**: < 100MB during normal operations
- **Window Operations**: < 200ms for resize/move

## Continuous Integration

### Headless Testing

Tests run in headless mode for CI environments:

```bash
export QT_QPA_PLATFORM=offscreen
python tests/gui_interactive/run_gui_tests.py ci
```

### Test Markers

Use pytest markers for selective test execution:

- `@pytest.mark.gui` - GUI tests (requires display)
- `@pytest.mark.slow` - Slow tests (skip in quick runs)
- `@pytest.mark.performance` - Performance benchmarks
- `@pytest.mark.integration` - Integration tests

## Debugging GUI Tests

### Test Failures

1. **Increase Wait Times**: Add longer `qtbot.wait()` for slow operations
2. **Check Mock Setup**: Verify mock objects return expected data
3. **Validate Widget State**: Use `gui_widget_inspector` to examine widgets
4. **Test Isolation**: Ensure tests don't interfere with each other

### Interactive Debugging

Run tests with visible windows for debugging:

```bash
# Remove headless mode
unset QT_QPA_PLATFORM
python -m pytest tests/gui_interactive/test_main_window.py::test_specific_function -v -s
```

## Extending the Test Suite

### Adding New Tests

1. **Choose Appropriate Test File**: Based on functionality being tested
2. **Use Existing Fixtures**: Leverage `gui_main_window`, `mock_xpcs_file`, etc.
3. **Follow Test Patterns**: Use established patterns for consistency
4. **Add Performance Tests**: Include timing for new UI operations

### Custom Fixtures

Create specialized fixtures for specific testing needs:

```python
@pytest.fixture
def custom_test_setup(gui_main_window, qtbot):
    # Custom setup logic
    window = gui_main_window
    # Configure for specific test scenario
    yield window, custom_data
    # Cleanup if needed
```

### Mock Data Generation

Extend mock data for new analysis types:

```python
@pytest.fixture
def mock_new_analysis_data():
    return {
        'new_analysis/result': np.random.random(100),
        'new_analysis/parameters': {'param1': 1.0}
    }
```

## Troubleshooting

### Common Issues

1. **Qt Platform Issues**: Set `QT_QPA_PLATFORM=offscreen` for headless systems
2. **Widget Not Found**: Check if widgets are visible and properly initialized
3. **Timing Issues**: Increase wait times for slow operations
4. **Memory Errors**: Use smaller test datasets or mock large data operations
5. **Permission Errors**: Ensure test directories are writable

### Platform-Specific Notes

- **Linux**: May need `xvfb` for GUI tests in containers
- **macOS**: Tests run natively with Cocoa backend
- **Windows**: Use `QT_QPA_PLATFORM=windows` if needed

## Contributing

When contributing GUI tests:

1. **Follow Naming Conventions**: `test_feature_description`
2. **Document Test Purpose**: Clear docstrings explaining what is tested
3. **Use Appropriate Markers**: Mark slow/performance/integration tests
4. **Maintain Isolation**: Tests should not depend on each other
5. **Test Both Success and Failure**: Include error condition testing
6. **Performance Considerations**: Add timing checks for new UI operations

## Resources

- [pytest-qt Documentation](https://pytest-qt.readthedocs.io/)
- [PySide6 Testing Guide](https://doc.qt.io/qtforpython/tutorials/index.html)
- [PyQtGraph Documentation](https://pyqtgraph.readthedocs.io/)
- [XPCS Toolkit Development Guide](../../docs/DEVELOPMENT.md)
