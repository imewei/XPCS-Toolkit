:orphan:

# XPCS Toolkit Logging System

This document provides comprehensive guidance for using the logging system in XPCS Toolkit, designed specifically for scientific computing applications with demanding performance requirements.

## Table of Contents

- Quick Start Guide
- API Reference
- Configuration
- Integration Patterns
- Performance and Benchmarks
- Best Practices
- Troubleshooting

## Quick Start Guide

### Basic Usage

```python
# At the top of any module
from xpcs_toolkit.utils.logging_config import get_logger

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.info("Module initialized successfully")
logger.debug("Processing data with shape: %s", data.shape)
logger.warning("Using default parameters - consider specifying them explicitly")
logger.error("Failed to process file: %s", filename)

# Exception logging
try:
    result = risky_operation()
except Exception as e:
    logger.exception("Operation failed")  # Automatically includes traceback
    raise
```

### Module Integration Template

```python
"""
Your module docstring here.
"""
import numpy as np
from typing import Optional, Dict, Any

# Import logging at the top
from xpcs_toolkit.utils.logging_config import get_logger

# Create module logger
logger = get_logger(__name__)

class YourClass:
    """Your class with integrated logging."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logger.debug("Initializing %s with config: %s", self.__class__.__name__, config)
        self.config = config or {}
        logger.info("%s initialized successfully", self.__class__.__name__)

    def process_data(self, data: np.ndarray) -> np.ndarray:
        """Process data with comprehensive logging."""
        logger.info("Processing data with shape: %s, dtype: %s", data.shape, data.dtype)

        if data.size == 0:
            logger.warning("Empty data array provided")
            return data

        try:
            # Your processing logic here
            result = data * 2  # Example operation
            logger.debug("Processing completed, output shape: %s", result.shape)
            return result

        except Exception as e:
            logger.error("Data processing failed for input shape %s", data.shape)
            logger.exception("Full exception details")
            raise
```

## API Reference

### Core Functions

#### `get_logger(name: str = None) -> logging.Logger`
Get a configured logger instance for your module.

**Parameters:**
- `name` (str, optional): Logger name, typically `__name__`

**Returns:**
- `logging.Logger`: Configured logger instance

**Example:**
```python
logger = get_logger(__name__)
logger.info("This is an info message")
```

#### `set_log_level(level: Union[str, int])`
Set the global logging level.

**Parameters:**
- `level`: Log level (e.g., 'DEBUG', 'INFO', logging.DEBUG)

**Example:**
```python
from xpcs_toolkit.utils.logging_config import set_log_level
set_log_level('DEBUG')  # Enable debug logging
```

#### `get_log_file_path() -> Path`
Get the current log file path.

#### `log_system_info()`
Log useful system information for debugging.

### Logger Methods

Standard Python logging methods are available:

```python
logger.debug("Detailed information for debugging")
logger.info("General information about program execution")
logger.warning("Something unexpected happened")
logger.error("A serious problem occurred")
logger.critical("A very serious error occurred")
logger.exception("Log an exception with traceback")
```

### Structured Logging

Add structured data to your logs:

```python
# Using extra parameter
logger.info("Processing file", extra={
    'filename': 'data.h5',
    'file_size': 1024000,
    'processing_time': 2.5
})

# Using string formatting for performance
logger.debug("Data shape: %s, processing time: %.2fs", data.shape, elapsed_time)
```

## Configuration

The logging system is configured via environment variables:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `PYXPCS_LOG_LEVEL` | Logging level | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `PYXPCS_LOG_FILE` | Custom log file path | `~/.xpcs_toolkit/logs/xpcs_toolkit.log` | Any valid file path |
| `PYXPCS_LOG_DIR` | Log directory | `~/.xpcs_toolkit/logs` | Any valid directory path |
| `PYXPCS_LOG_FORMAT` | Log format type | `TEXT` | `TEXT`, `JSON` |
| `PYXPCS_LOG_MAX_SIZE` | Max log file size (MB) | `10` | Any positive integer |
| `PYXPCS_LOG_BACKUP_COUNT` | Number of backup files | `5` | Any positive integer |
| `PYXPCS_SUPPRESS_QT_WARNINGS` | Suppress Qt warnings | `0` | `0`, `1` |

### Configuration Examples

```bash
# Enable debug logging
export PYXPCS_LOG_LEVEL=DEBUG

# Use JSON format for structured logging
export PYXPCS_LOG_FORMAT=JSON

# Custom log file location
export PYXPCS_LOG_FILE=/path/to/custom/logfile.log

# Suppress Qt warnings for cleaner logs
export PYXPCS_SUPPRESS_QT_WARNINGS=1
```

### Runtime Configuration

```python
from xpcs_toolkit.utils.logging_config import get_logging_config, set_log_level

# Get current configuration
config = get_logging_config()
info = config.get_logger_info()
print(f"Current log level: {info['log_level']}")
print(f"Log file: {info['log_file']}")

# Change log level at runtime
set_log_level('DEBUG')
```

## Integration Patterns

### New Module Integration

1. **Import logging at the top of your module:**
```python
from xpcs_toolkit.utils.logging_config import get_logger
logger = get_logger(__name__)
```

2. **Log module initialization:**
```python
logger.info("Module %s loaded", __name__)
```

3. **Log function entry/exit for complex operations:**
```python
def complex_calculation(data):
    logger.debug("Starting complex calculation with %d data points", len(data))
    # ... processing ...
    logger.debug("Complex calculation completed")
    return result
```

4. **Log errors and exceptions:**
```python
try:
    result = operation()
except ValueError as e:
    logger.error("Invalid input: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error in operation")
    raise
```

### GUI Integration Pattern

```python
from PySide6.QtWidgets import QWidget
from xpcs_toolkit.utils.logging_config import get_logger

class MyWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("Widget %s initialized", self.__class__.__name__)

    def on_button_clicked(self):
        self.logger.debug("Button clicked, processing request")
        try:
            # Process the request
            self.process_data()
            self.logger.info("Request processed successfully")
        except Exception as e:
            self.logger.exception("Failed to process request")
            # Show error dialog to user
```

### Test Integration Pattern

```python
import pytest
from xpcs_toolkit.utils.logging_config import get_logger, set_log_level

logger = get_logger(__name__)

class TestMyModule:
    def setup_method(self):
        """Setup for each test method."""
        set_log_level('DEBUG')  # Enable detailed logging for tests
        logger.info("Starting test: %s", self._testMethodName)

    def test_data_processing(self):
        logger.debug("Testing data processing functionality")
        # Test implementation
        assert result is not None
        logger.info("Data processing test passed")
```

## Performance and Benchmarks

### Performance Logging

Use the performance logging utilities for timing operations:

```python
from xpcs_toolkit.utils.log_templates import log_performance
import time

@log_performance
def slow_operation(data):
    """This decorator will log execution time."""
    time.sleep(1)
    return data * 2

# Manual performance logging
with log_performance("custom_operation"):
    # Your code here
    pass
```

### Benchmark System

The logging system includes comprehensive benchmarks that validate:

- **Throughput**: Message processing rates under different conditions
- **Latency**: Per-message timing with statistical rigor  
- **Memory Usage**: Memory consumption patterns and leak detection
- **Concurrency**: Thread safety and parallel performance
- **Scientific Computing**: Domain-specific performance validation

Run benchmarks with:

```bash
# Quick benchmark
python run_logging_benchmarks.py --quick

# Comprehensive benchmark suite  
python run_logging_benchmarks.py

# Generate detailed reports
python run_logging_benchmarks.py --report
```

### Key Performance Characteristics

- **Console Logging**: ~50,000 messages/second
- **File Logging**: ~30,000 messages/second  
- **JSON Format**: ~15,000 messages/second
- **Memory Usage**: <2MB baseline, scales linearly
- **Latency**: <0.1ms per message (99th percentile)

## Best Practices

### 1. Logger Naming
- Use `__name__` for module loggers: `logger = get_logger(__name__)`
- For classes, consider: `logger = get_logger(f"{__name__}.{self.__class__.__name__}")`

### 2. Log Levels
- **DEBUG**: Detailed diagnostic information
- **INFO**: General operational information
- **WARNING**: Something unexpected but the program continues
- **ERROR**: Serious problem, program may not continue
- **CRITICAL**: Very serious error, program may be unable to continue

### 3. Message Formatting
```python
# Good: Use % formatting for performance
logger.debug("Processing %d items with config %s", len(items), config)

# Avoid: String concatenation or f-strings in log calls
logger.debug("Processing " + str(len(items)) + " items")  # Don't do this
logger.debug(f"Processing {len(items)} items")  # Don't do this
```

### 4. Exception Logging
```python
# Good: Use logger.exception() in exception handlers
try:
    risky_operation()
except Exception:
    logger.exception("Operation failed")  # Includes full traceback
    raise

# Alternative: Use exc_info=True
logger.error("Operation failed", exc_info=True)
```

### 5. Avoid Logging in Hot Paths
```python
# Good: Use appropriate log level
for item in large_dataset:
    if logger.isEnabledFor(logging.DEBUG):  # Check before expensive operations
        logger.debug("Processing item: %s", expensive_repr(item))

# Better: Use higher log level for summary
logger.info("Processing %d items", len(large_dataset))
# Process items...
logger.info("Completed processing %d items in %.2fs", len(large_dataset), elapsed)
```

### 6. Structured Logging
```python
# Good: Use extra parameter for structured data
logger.info("File processed", extra={
    'filename': filename,
    'size_bytes': file_size,
    'processing_time': elapsed
})

# Also good: Use consistent key naming
logger.info("Analysis started", extra={'operation': 'analysis', 'type': 'correlation'})
logger.info("Analysis completed", extra={'operation': 'analysis', 'status': 'success'})
```

## Troubleshooting

### Common Issues

#### 1. No Log Output
**Problem**: Logger not producing output
**Solutions**:
```python
# Check current log level
from xpcs_toolkit.utils.logging_config import get_logging_config
config = get_logging_config()
print(config.get_logger_info())

# Set more verbose level
from xpcs_toolkit.utils.logging_config import set_log_level
set_log_level('DEBUG')
```

#### 2. Log File Not Created
**Problem**: Log file not being written
**Solutions**:
- Check directory permissions
- Verify log directory exists and is writable
- Check `PYXPCS_LOG_FILE` and `PYXPCS_LOG_DIR` environment variables

```python
# Check log file path
from xpcs_toolkit.utils.logging_config import get_log_file_path
print(f"Log file: {get_log_file_path()}")

# Check if directory exists
log_path = get_log_file_path()
print(f"Directory exists: {log_path.parent.exists()}")
print(f"Directory writable: {os.access(log_path.parent, os.W_OK)}")
```

#### 3. Poor Performance
**Problem**: Logging is slowing down the application
**Solutions**:
- Reduce log level (`INFO` instead of `DEBUG`)
- Use lazy evaluation with % formatting
- Check for expensive operations in log messages
- Consider using queue handlers for high-volume logging

#### 4. JSON Format Issues
**Problem**: JSON logs are malformed
**Solutions**:
- Ensure all extra data is JSON-serializable
- Check for circular references in logged objects
- Use simple data types (str, int, float, list, dict)

#### 5. Qt Warnings Cluttering Logs
**Problem**: Too many Qt-related log messages
**Solution**:
```bash
export PYXPCS_SUPPRESS_QT_WARNINGS=1
```

### Debugging the Logging System

Enable verbose logging configuration:

```python
import logging
logging.getLogger('xpcs_toolkit.utils.logging_config').setLevel(logging.DEBUG)
```

Check handler configuration:

```python
import logging
root_logger = logging.getLogger()
print(f"Root logger level: {root_logger.level}")
print(f"Handlers: {len(root_logger.handlers)}")
for i, handler in enumerate(root_logger.handlers):
    print(f"Handler {i}: {type(handler).__name__} - Level: {handler.level}")
```

## Advanced Features

### Remote Logging

Configure logging to remote destinations:

```python
# Syslog integration (example)
import logging.handlers
syslog_handler = logging.handlers.SysLogHandler(address=('localhost', 514))
logger.addHandler(syslog_handler)

# HTTP webhook logging (example)
import requests
class WebhookHandler(logging.Handler):
    def emit(self, record):
        requests.post('https://your-webhook-url', json={'message': self.format(record)})
```

### Custom Formatters

```python
from xpcs_toolkit.utils.log_formatters import create_formatter

# Create a custom performance formatter
perf_formatter = create_formatter('performance')
handler = logging.FileHandler('performance.log')
handler.setFormatter(perf_formatter)

# Add to specific logger
perf_logger = get_logger('performance')
perf_logger.addHandler(handler)
```

### Context Managers

```python
from xpcs_toolkit.utils.log_templates import log_context

with log_context("Processing batch of files", logger=logger) as ctx:
    for filename in filenames:
        process_file(filename)
        ctx.update(f"Processed {filename}")
```

---

Remember: The logging system is designed to be helpful, not intrusive. When in doubt, log more rather than less - you can always adjust the log level later.
