"""
Logging templates and utility functions for XPCS Toolkit.

This module provides reusable templates, decorators, and utilities to make
logging integration easier and more consistent across the codebase.

Features:
- Performance monitoring decorators
- Context managers for operation logging
- Template functions for common patterns
- Test logging helpers
- Structured logging utilities
- Error logging decorators
"""

import functools
import logging
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Union

from .logging_config import get_logger

# =============================================================================
# Performance Logging Templates
# =============================================================================


def log_performance(
    func: Callable = None,
    *,
    logger_name: str = None,
    level: str = "INFO",
    include_args: bool = False,
    include_result: bool = False,
) -> Callable:
    """
    Decorator to log function execution time and optionally arguments/results.

    Args:
        func: Function to decorate (when used as @log_performance)
        logger_name: Custom logger name (defaults to function's module)
        level: Log level for performance messages
        include_args: Whether to log function arguments (DEBUG level)
        include_result: Whether to log function result (DEBUG level)

    Usage:
        @log_performance
        def my_function():
            pass

        @log_performance(level='DEBUG', include_args=True)
        def detailed_function(data):
            return processed_data
    """

    def decorator(f):
        perf_logger = get_logger(logger_name or f.__module__)
        log_level = getattr(logging, level.upper())

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Create operation ID for tracking
            operation_id = f"{f.__name__}_{int(time.time() * 1000) % 10000}"

            # Log function start with optional arguments
            if include_args and perf_logger.isEnabledFor(logging.DEBUG):
                perf_logger.debug(
                    "Starting %s [%s] with args=%s, kwargs=%s",
                    f.__name__,
                    operation_id,
                    args,
                    kwargs,
                )
            else:
                perf_logger.log(log_level, "Starting %s [%s]", f.__name__, operation_id)

            try:
                result = f(*args, **kwargs)
                elapsed = time.time() - start_time

                # Log successful completion with performance metrics
                perf_logger.log(
                    log_level,
                    "Completed %s [%s] in %.3fs",
                    f.__name__,
                    operation_id,
                    elapsed,
                )

                # Log structured performance data
                if perf_logger.isEnabledFor(logging.DEBUG):
                    perf_logger.debug(
                        "Performance metrics",
                        extra={
                            "function": f.__name__,
                            "operation_id": operation_id,
                            "duration_seconds": elapsed,
                            "performance_category": "timing",
                        },
                    )

                # Log result if requested
                if include_result and perf_logger.isEnabledFor(logging.DEBUG):
                    perf_logger.debug(
                        "Function %s [%s] result: %s", f.__name__, operation_id, result
                    )

                return result

            except Exception as e:
                elapsed = time.time() - start_time
                perf_logger.error(
                    "Function %s [%s] failed after %.3fs: %s",
                    f.__name__,
                    operation_id,
                    elapsed,
                    str(e),
                )
                perf_logger.debug("Exception details", exc_info=True)
                raise

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


class LogPerformanceContext:
    """Context manager for logging operation performance with progress tracking."""

    def __init__(
        self,
        operation_name: str,
        logger: logging.Logger = None,
        level: str = "INFO",
        log_memory: bool = False,
    ):
        self.operation_name = operation_name
        self.logger = logger or get_logger(__name__)
        self.level = getattr(logging, level.upper())
        self.log_memory = log_memory
        self.start_time = None
        self.operation_id = f"{operation_name}_{int(time.time() * 1000) % 10000}"

        # Memory tracking
        if log_memory:
            try:
                import psutil

                self.process = psutil.Process()
                self.start_memory = None
            except ImportError:
                self.log_memory = False
                self.logger.debug("psutil not available, memory logging disabled")

    def __enter__(self):
        self.start_time = time.time()

        # Log memory at start
        if self.log_memory:
            self.start_memory = self.process.memory_info().rss / 1024 / 1024
            self.logger.debug(
                "Starting %s [%s] - Memory: %.1f MB",
                self.operation_name,
                self.operation_id,
                self.start_memory,
            )
        else:
            self.logger.log(
                self.level, "Starting %s [%s]", self.operation_name, self.operation_id
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time

        if exc_type is None:
            # Success
            if self.log_memory and self.start_memory is not None:
                end_memory = self.process.memory_info().rss / 1024 / 1024
                memory_delta = end_memory - self.start_memory

                self.logger.log(
                    self.level,
                    "Completed %s [%s] in %.3fs - Memory: %.1f MB (Δ%.1f MB)",
                    self.operation_name,
                    self.operation_id,
                    elapsed,
                    end_memory,
                    memory_delta,
                )
            else:
                self.logger.log(
                    self.level,
                    "Completed %s [%s] in %.3fs",
                    self.operation_name,
                    self.operation_id,
                    elapsed,
                )

            # Log structured data
            extra_data = {
                "operation": self.operation_name,
                "operation_id": self.operation_id,
                "duration_seconds": elapsed,
                "status": "success",
            }

            if self.log_memory and self.start_memory is not None:
                extra_data.update(
                    {
                        "start_memory_mb": self.start_memory,
                        "end_memory_mb": end_memory,
                        "memory_delta_mb": memory_delta,
                    }
                )

            self.logger.debug("Operation completed", extra=extra_data)
        else:
            # Failure
            self.logger.error(
                "Failed %s [%s] after %.3fs: %s",
                self.operation_name,
                self.operation_id,
                elapsed,
                str(exc_val),
            )

            # Log structured error data
            self.logger.error(
                "Operation failed",
                extra={
                    "operation": self.operation_name,
                    "operation_id": self.operation_id,
                    "duration_seconds": elapsed,
                    "status": "error",
                    "error_type": exc_type.__name__ if exc_type else "Unknown",
                    "error_message": str(exc_val) if exc_val else "Unknown error",
                },
            )

    def update(self, message: str, progress: float = None):
        """Update progress message with optional progress percentage."""
        elapsed = time.time() - self.start_time

        if progress is not None:
            self.logger.log(
                self.level,
                "%s [%s] [%.1fs]: %s (%.1f%%)",
                self.operation_name,
                self.operation_id,
                elapsed,
                message,
                progress,
            )
        else:
            self.logger.log(
                self.level,
                "%s [%s] [%.1fs]: %s",
                self.operation_name,
                self.operation_id,
                elapsed,
                message,
            )


# Convenience function for performance context
def log_context(
    operation_name: str,
    logger: logging.Logger = None,
    level: str = "INFO",
    log_memory: bool = False,
) -> LogPerformanceContext:
    """
    Create a performance logging context manager.

    Usage:
        with log_context("Processing data") as ctx:
            # do work
            ctx.update("Phase 1 complete", 25.0)
            # more work
    """
    return LogPerformanceContext(operation_name, logger, level, log_memory)


# =============================================================================
# Error Handling Templates
# =============================================================================


def log_exceptions(
    func: Callable = None,
    *,
    logger_name: str = None,
    logger: logging.Logger = None,
    reraise: bool = True,
    default_return: Any = None,
    return_value: Any = None,
) -> Callable:
    """
    Decorator to log exceptions with full context.

    Args:
        func: Function to decorate
        logger_name: Custom logger name
        reraise: Whether to reraise the exception after logging
        default_return: Value to return if exception occurs and reraise=False

    Usage:
        @log_exceptions
        def risky_operation():
            pass

        @log_exceptions(reraise=False, default_return=[])
        def safe_operation():
            pass
    """

    def decorator(f):
        exc_logger = logger or get_logger(logger_name or f.__module__)

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                # Log the exception with full context
                exc_logger.exception("Exception in %s: %s", f.__name__, str(e))

                # Log structured exception data
                exc_logger.error(
                    "Function exception details",
                    extra={
                        "function_name": f.__name__,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "function_args": str(args) if args else None,
                        "function_kwargs": str(kwargs) if kwargs else None,
                    },
                )

                if reraise:
                    raise
                else:
                    exc_logger.warning(
                        "Returning default value due to exception in %s", f.__name__
                    )
                    return return_value if return_value is not None else default_return

        return wrapper

    if func is None:
        return decorator
    else:
        return decorator(func)


def retry_with_logging(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    logger_name: str = None,
) -> Callable:
    """
    Decorator to retry function execution with exponential backoff and logging.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exception types to catch and retry
        logger_name: Custom logger name
    """

    def decorator(func: Callable) -> Callable:
        retry_logger = get_logger(logger_name or func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    if attempt > 0:
                        retry_logger.info(
                            "Retry attempt %d/%d for %s",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                        )

                    result = func(*args, **kwargs)

                    if attempt > 0:
                        retry_logger.info(
                            "Function %s succeeded on attempt %d",
                            func.__name__,
                            attempt + 1,
                        )

                    return result

                except exceptions as e:
                    last_exception = e
                    retry_logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt + 1,
                        max_retries,
                        func.__name__,
                        str(e),
                    )

                    if attempt < max_retries - 1:
                        retry_logger.info(
                            "Waiting %.1fs before retry...", current_delay
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        retry_logger.error(
                            "All %d attempts failed for %s", max_retries, func.__name__
                        )
                except Exception as e:
                    # Non-retryable exception
                    retry_logger.error(
                        "Non-retryable exception in %s: %s", func.__name__, str(e)
                    )
                    raise

            # All retries exhausted
            raise last_exception

        return wrapper

    return decorator


# =============================================================================
# Module Integration Templates
# =============================================================================


def create_module_logger(module_name: str, level: str = None) -> logging.Logger:
    """
    Create a properly configured logger for a module.

    This is the standard way to create loggers in XPCS Toolkit modules.
    """
    logger = get_logger(module_name)
    if level:
        logger.setLevel(getattr(logging, level.upper()))
    logger.info("Module logger initialized: %s", module_name)
    return logger


class ModuleLoggerMixin:
    """
    Mixin class to add logging capabilities to any class.

    Usage:
        class MyClass(ModuleLoggerMixin):
            def __init__(self):
                super().__init__()
                self.logger.info("MyClass initialized")
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create class-specific logger
        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        self.logger = get_logger(f"{module_name}.{class_name}")


def log_method_calls(include_args: bool = False, include_return: bool = False):
    """
    Class decorator to log all method calls.

    Args:
        include_args: Whether to log method arguments (DEBUG level)
        include_return: Whether to log return values (DEBUG level)

    Usage:
        @log_method_calls(include_args=True)
        class MyClass:
            pass
    """

    def class_decorator(cls):
        class_logger = get_logger(f"{cls.__module__}.{cls.__name__}")

        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)

            if (
                callable(attr)
                and not attr_name.startswith("__")
                and not isinstance(attr, type)
            ):
                # Create logging wrapper for method
                def create_wrapper(method_name, original_method):
                    @functools.wraps(original_method)
                    def wrapper(self, *args, **kwargs):
                        if include_args and class_logger.isEnabledFor(logging.DEBUG):
                            class_logger.debug(
                                "Calling %s.%s with args=%s, kwargs=%s",
                                cls.__name__,
                                method_name,
                                args,
                                kwargs,
                            )
                        else:
                            class_logger.debug(
                                "Calling %s.%s", cls.__name__, method_name
                            )

                        try:
                            result = original_method(self, *args, **kwargs)

                            if include_return and class_logger.isEnabledFor(
                                logging.DEBUG
                            ):
                                class_logger.debug(
                                    "Method %s.%s returned: %s",
                                    cls.__name__,
                                    method_name,
                                    result,
                                )

                            return result

                        except Exception as e:
                            class_logger.error(
                                "Method %s.%s failed: %s",
                                cls.__name__,
                                method_name,
                                str(e),
                            )
                            raise

                    return wrapper

                setattr(cls, attr_name, create_wrapper(attr_name, attr))

        return cls

    return class_decorator


# =============================================================================
# Test Logging Helpers
# =============================================================================


class TestLogCapture:
    """
    Context manager to capture log messages during testing.

    Usage:
        with TestLogCapture() as log_capture:
            # code that logs messages
            pass

        assert "Expected message" in log_capture.messages
        assert log_capture.has_error()
    """

    def __init__(self, logger_name: str = None, level: str = "DEBUG"):
        self.logger_name = logger_name
        self.level = getattr(logging, level.upper())
        self.messages = []
        self.records = []
        self.handler = None
        self.original_level = None

    def __enter__(self):
        # Create custom handler to capture log messages
        self.handler = logging.Handler()
        self.handler.setLevel(self.level)
        self.handler.emit = self._capture_record

        # Add handler to appropriate logger
        if self.logger_name:
            logger = logging.getLogger(self.logger_name)
        else:
            logger = logging.getLogger()

        # Store original level and set to capture level
        self.original_level = logger.level
        logger.setLevel(min(self.level, logger.level) if logger.level != logging.NOTSET else self.level)

        logger.addHandler(self.handler)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            # Remove handler from logger and restore original level
            if self.logger_name:
                logger = logging.getLogger(self.logger_name)
            else:
                logger = logging.getLogger()

            logger.removeHandler(self.handler)
            # Restore original logger level
            if hasattr(self, 'original_level'):
                logger.setLevel(self.original_level)

    def _capture_record(self, record):
        """Capture a log record."""
        self.records.append(record)
        self.messages.append(record.getMessage())

    def has_level(self, level: str) -> bool:
        """Check if any messages were logged at the specified level."""
        level_no = getattr(logging, level.upper())
        return any(record.levelno >= level_no for record in self.records)

    def has_error(self) -> bool:
        """Check if any error or critical messages were logged."""
        return self.has_level("ERROR")

    def has_warning(self) -> bool:
        """Check if any warning messages were logged."""
        return self.has_level("WARNING")

    def get_messages_containing(self, text: str) -> List[str]:
        """Get all messages containing the specified text."""
        return [msg for msg in self.messages if text in msg]

    def get_logs(self) -> List[str]:
        """Get all captured log messages."""
        return self.messages

    def assert_logged(self, text: str, level: str = None):
        """Assert that a message containing text was logged."""
        matching_messages = self.get_messages_containing(text)
        if not matching_messages:
            raise AssertionError(f"No log message found containing: '{text}'")

        if level:
            level_no = getattr(logging, level.upper())
            matching_records = [
                r
                for r in self.records
                if text in r.getMessage() and r.levelno >= level_no
            ]
            if not matching_records:
                raise AssertionError(f"No {level} message found containing: '{text}'")


@contextmanager
def temp_log_level(level: str):
    """
    Temporarily change the log level.

    Usage:
        with temp_log_level('DEBUG'):
            # Debug logging is enabled here
            logger.debug("This will be shown")
        # Original log level restored
    """
    from .logging_config import get_logging_config

    config = get_logging_config()
    original_level = config.log_level

    try:
        config.update_log_level(level)
        yield
    finally:
        config.update_log_level(original_level)


# =============================================================================
# Structured Logging Helpers
# =============================================================================


class StructuredLogger:
    """
    Helper class for consistent structured logging with automatic context.

    Usage:
        structured_logger = StructuredLogger(__name__, {
            'component': 'data_processor',
            'version': '1.0.0'
        })

        structured_logger.log_event('INFO', 'Processing started',
                                  operation='analysis', file_count=10)
    """

    def __init__(
        self,
        logger_name: Union[str, logging.Logger],
        base_context: Dict[str, Any] = None,
        **kwargs,
    ):
        if isinstance(logger_name, str):
            self.logger = get_logger(logger_name)
        else:
            self.logger = logger_name
        self.base_context = base_context or {}
        self.base_context.update(kwargs)  # Add any additional keyword arguments
        self.session_id = str(uuid.uuid4())[:8]

    def log_event(self, level: str, message: str, **context):
        """Log an event with structured context."""
        full_context = {"session_id": self.session_id, **self.base_context, **context}
        log_level = getattr(logging, level.upper())
        self.logger.log(log_level, message, extra=full_context)

    def log_business_event(self, event_type: str, **context):
        """Log a business/application event."""
        self.log_event(
            "INFO",
            f"Business event: {event_type}",
            event_type=event_type,
            category="business",
            **context,
        )

    def log_performance_metric(
        self, metric_name: str, value: float, unit: str = "seconds", **context
    ):
        """Log a performance metric."""
        self.log_event(
            "INFO",
            f"Performance metric: {metric_name}",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            category="performance",
            **context,
        )

    def log_user_action(self, action: str, user_id: str = None, **context):
        """Log a user action."""
        self.log_event(
            "INFO",
            f"User action: {action}",
            action=action,
            user_id=user_id,
            category="user_action",
            **context,
        )

    def log_system_event(self, event: str, **context):
        """Log a system event."""
        self.log_event(
            "INFO", f"System event: {event}", event=event, category="system", **context
        )


# =============================================================================
# Template Generation Functions
# =============================================================================


def generate_module_template(module_name: str, class_name: str = None) -> str:
    """
    Generate a template for a new module with logging integration.

    Args:
        module_name: Name of the module
        class_name: Optional class name to include in template

    Returns:
        String containing the module template
    """
    template = f'''"""
{module_name.replace("_", " ").title()} module for XPCS Toolkit.

This module provides functionality for...

Example:
    from xpcs_toolkit.{module_name} import MyClass
    
    processor = MyClass()
    result = processor.process(data)
"""
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Import logging at the top
from xpcs_toolkit.utils.logging_config import get_logger

# Create module logger
logger = get_logger(__name__)

'''

    if class_name:
        template += f'''
class {class_name}:
    """Main class for {module_name} functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger(f"{{__name__}}.{{self.__class__.__name__}}")
        self.config = config or {{}}
        
        self.logger.info("Initializing %s with config: %s", 
                        self.__class__.__name__, self.config)
        
        # Initialize your class here
        
        self.logger.info("%s initialized successfully", self.__class__.__name__)
    
    def process(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Process data with comprehensive logging."""
        self.logger.info("Processing data: shape=%s, dtype=%s", data.shape, data.dtype)
        
        if data.size == 0:
            self.logger.warning("Processing empty data array")
            return data
        
        try:
            # Your processing logic here
            result = data.copy()  # Placeholder
            
            self.logger.info("Processing completed: output shape=%s", result.shape)
            return result
            
        except Exception as e:
            self.logger.error("Processing failed for data shape %s", data.shape)
            self.logger.exception("Processing exception details")
            raise

# Module initialization logging
logger.info("Module %s loaded successfully", __name__)
'''

    return template


def generate_test_template(module_name: str, class_name: str = None) -> str:
    """
    Generate a template for test files with logging integration.

    Args:
        module_name: Name of the module being tested
        class_name: Optional class name being tested

    Returns:
        String containing the test template
    """
    template = f'''"""
Test module for {module_name}.

This module contains unit tests with integrated logging for debugging
and test analysis.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch

from xpcs_toolkit.utils.logging_config import get_logger, set_log_level
from xpcs_toolkit.utils.log_templates import TestLogCapture, temp_log_level
'''

    if class_name:
        template += f"""from xpcs_toolkit.{module_name} import {class_name}"""

    template += f'''

logger = get_logger(__name__)

class Test{class_name or module_name.replace("_", " ").title().replace(" ", "")}:
    """Test class with logging integration."""
    
    @classmethod
    def setup_class(cls):
        """Setup for entire test class."""
        set_log_level('DEBUG')  # Enable detailed logging for tests
        logger.info("Starting test class: %s", cls.__name__)
    
    def setup_method(self, method):
        """Setup for each test method."""
        logger.info("Starting test: %s", method.__name__)'''

    if class_name:
        template += f"""
        self.instance = {class_name}()"""

    template += '''
    
    def teardown_method(self, method):
        """Teardown for each test method."""
        logger.info("Completed test: %s", method.__name__)'''

    if class_name:
        template += """
        if hasattr(self, 'instance'):
            # Cleanup if needed
            pass"""

    template += '''
    
    def test_basic_functionality(self):
        """Test basic functionality with logging."""
        logger.debug("Testing basic functionality")
        
        # Create test data
        test_data = np.random.random((10, 10))
        logger.debug("Created test data: shape=%s", test_data.shape)
        
        # Test the functionality
        with TestLogCapture() as log_capture:'''

    if class_name:
        template += """
            result = self.instance.process(test_data)"""
    else:
        template += """
            # Your test code here
            result = test_data * 2  # Placeholder"""

    template += '''
            
            # Verify no errors were logged
            assert not log_capture.has_error(), "Unexpected errors in logs"
            
            # Verify expected log messages
            log_capture.assert_logged("Processing")
        
        # Verify results
        assert result is not None
        logger.info("Basic functionality test passed")
    
    def test_error_handling(self):
        """Test error handling with logging."""
        logger.debug("Testing error handling")
        
        with TestLogCapture() as log_capture:
            with pytest.raises(ValueError):
                # Test code that should raise an error
                pass  # Replace with actual error-inducing code
            
            # Verify error was logged
            assert log_capture.has_error(), "Expected error was not logged"
        
        logger.info("Error handling test passed")
    
    @pytest.mark.parametrize("test_size", [0, 1, 100])
    def test_various_sizes(self, test_size):
        """Test with various input sizes."""
        logger.debug("Testing with size: %d", test_size)
        
        if test_size == 0:
            test_data = np.array([])
        else:
            test_data = np.random.random(test_size)
        
        with temp_log_level('DEBUG'):'''

    if class_name:
        template += """
            result = self.instance.process(test_data)"""
    else:
        template += """
            # Your size-specific test code
            result = test_data  # Placeholder"""

    template += '''
            assert len(result) == test_size
        
        logger.info("Size %d test passed", test_size)

# Performance test
@pytest.mark.performance
def test_performance():
    """Test performance with timing and logging."""
    logger.info("Starting performance test")
    
    import time'''

    if class_name:
        template += f"""
    instance = {class_name}()"""

    template += """
    
    # Large test data
    large_data = np.random.random((1000, 1000))
    
    start_time = time.time()"""

    if class_name:
        template += """
    result = instance.process(large_data)"""
    else:
        template += """
    result = large_data * 2  # Placeholder operation"""

    template += """
    elapsed = time.time() - start_time
    
    logger.info("Performance test completed in %.3fs", elapsed)
    
    # Assert performance requirements
    assert elapsed < 10.0, f"Processing took too long: {elapsed:.3f}s"
    
    # Log performance metrics
    throughput = large_data.size / elapsed
    logger.info("Processing throughput: %.0f elements/second", throughput)
"""

    return template


# =============================================================================
# Utility Functions
# =============================================================================


def setup_module_logging(
    module_name: str, extra_context: Dict[str, Any] = None
) -> logging.Logger:
    """
    Setup logging for a module with optional extra context.

    This is a convenience function for modules that need structured logging
    with consistent context information.
    """
    logger = get_logger(module_name)

    # Add any extra context as a filter if provided
    if extra_context:

        class ContextFilter(logging.Filter):
            def filter(self, record):
                for key, value in extra_context.items():
                    setattr(record, key, value)
                return True

        logger.addFilter(ContextFilter())

    logger.info("Module logging setup completed: %s", module_name)
    return logger


def log_function_signature(func: Callable, args: tuple, kwargs: dict) -> str:
    """
    Create a string representation of a function call for logging.

    Useful for debugging complex function calls with many parameters.
    """
    arg_strs = [repr(arg) for arg in args]
    kwarg_strs = [f"{k}={repr(v)}" for k, v in kwargs.items()]
    all_args = arg_strs + kwarg_strs

    return f"{func.__name__}({', '.join(all_args)})"


# =============================================================================
# Compatibility Aliases for Existing Tests
# =============================================================================

# Aliases for compatibility with existing tests
LogPerformanceMonitor = log_performance
LogExceptionHandler = log_exceptions

# =============================================================================
# Export commonly used items
# =============================================================================

__all__ = [
    # Performance logging
    "log_performance",
    "LogPerformanceContext",
    "log_context",
    # Error handling
    "log_exceptions",
    "retry_with_logging",
    # Module integration
    "create_module_logger",
    "ModuleLoggerMixin",
    "log_method_calls",
    # Test helpers
    "TestLogCapture",
    "temp_log_level",
    # Structured logging
    "StructuredLogger",
    # Template generation
    "generate_module_template",
    "generate_test_template",
    # Utilities
    "setup_module_logging",
    "log_function_signature",
]
