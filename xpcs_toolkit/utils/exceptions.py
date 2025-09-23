"""
XPCS Toolkit Exception Hierarchy for Enhanced Error Handling and Reliability.

This module provides a comprehensive exception hierarchy that enables:
- Better error categorization and handling
- Zero-overhead performance (exceptions only cost when raised)
- Detailed error context and recovery suggestions
- Exception chaining for full error traceability
- Automated error reporting and classification
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any


class XPCSBaseError(Exception):
    """
    Base exception for all XPCS Toolkit errors.

    Provides common functionality for error context, recovery suggestions,
    and performance-aware error handling.
    """

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        context: dict[str, Any] | None = None,
        recovery_suggestions: list[str] | None = None,
        severity: str = "ERROR",
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.severity = severity
        self.timestamp = None  # Set by error handler when needed

    def add_context(self, key: str, value: Any) -> "XPCSBaseError":
        """Add context information to the error (fluent interface)."""
        self.context[key] = value
        return self

    def add_recovery_suggestion(self, suggestion: str) -> "XPCSBaseError":
        """Add recovery suggestion (fluent interface)."""
        self.recovery_suggestions.append(suggestion)
        return self

    def get_error_details(self) -> dict[str, Any]:
        """Get comprehensive error details for logging/reporting."""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity,
            "context": self.context,
            "recovery_suggestions": self.recovery_suggestions,
            "timestamp": self.timestamp,
        }


class XPCSDataError(XPCSBaseError):
    """
    Exceptions related to data validation, corruption, or format issues.

    Used for:
    - Invalid data formats or structures
    - Data corruption detection
    - Missing required data fields
    - Data consistency violations
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, severity="ERROR", **kwargs)


class XPCSFileError(XPCSBaseError):
    """
    Exceptions related to file operations and I/O issues.

    Used for:
    - File not found or access issues
    - HDF5 file corruption or format problems
    - File permission or disk space issues
    - Network file access problems
    """

    def __init__(
        self, message: str, file_path: str | Path | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, severity="ERROR", **kwargs)
        if file_path:
            self.add_context("file_path", str(file_path))


class XPCSComputationError(XPCSBaseError):
    """
    Exceptions related to scientific computations and analysis.

    Used for:
    - Numerical computation failures
    - Fitting algorithm convergence issues
    - Invalid computation parameters
    - Mathematical domain errors
    """

    def __init__(
        self, message: str, operation: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, severity="ERROR", **kwargs)
        if operation:
            self.add_context("operation", operation)


class XPCSMemoryError(XPCSBaseError):
    """
    Exceptions related to memory management and resource exhaustion.

    Used for:
    - Out of memory conditions
    - Memory allocation failures
    - Cache overflow situations
    - Resource limit exceeded
    """

    def __init__(
        self, message: str, requested_mb: float | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, severity="CRITICAL", **kwargs)
        if requested_mb is not None:
            self.add_context("requested_memory_mb", requested_mb)


class XPCSConfigurationError(XPCSBaseError):
    """
    Exceptions related to configuration and setup issues.

    Used for:
    - Invalid configuration values
    - Missing required configuration
    - Configuration validation failures
    - Environment setup problems
    """

    def __init__(
        self, message: str, config_key: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, severity="ERROR", **kwargs)
        if config_key:
            self.add_context("config_key", config_key)


class XPCSGUIError(XPCSBaseError):
    """
    Exceptions related to GUI operations and Qt interactions.

    Used for:
    - Qt signal/slot connection issues
    - GUI component initialization failures
    - Threading/concurrency problems in GUI
    - Display or rendering issues
    """

    def __init__(
        self, message: str, component: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(message, severity="WARNING", **kwargs)
        if component:
            self.add_context("gui_component", component)


class XPCSValidationError(XPCSDataError):
    """
    Specific validation failures with detailed field information.

    Used for:
    - Input parameter validation
    - Data schema validation
    - Range and constraint validation
    - Type validation failures
    """

    def __init__(
        self, message: str, field: str | None = None, value: Any = None, **kwargs: Any
    ) -> None:
        super().__init__(message, **kwargs)
        if field:
            self.add_context("field", field)
        if value is not None:
            self.add_context("invalid_value", str(value))


class XPCSNetworkError(XPCSBaseError):
    """
    Exceptions related to network operations and remote access.

    Used for:
    - Network connectivity issues
    - Remote file access failures
    - API call failures
    - Timeout conditions
    """

    def __init__(self, message: str, url: str | None = None, **kwargs: Any) -> None:
        super().__init__(message, severity="WARNING", **kwargs)
        if url:
            self.add_context("url", url)


class XPCSCriticalError(XPCSBaseError):
    """
    Critical system errors that require immediate attention.

    Used for:
    - System state corruption
    - Unrecoverable failures
    - Security violations
    - Data integrity failures
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, severity="CRITICAL", **kwargs)


class XPCSWarning(XPCSBaseError):
    """
    Non-fatal issues that should be reported but don't stop execution.

    Used for:
    - Performance degradation warnings
    - Deprecated feature usage
    - Partial operation failures
    - Resource usage warnings
    """

    def __init__(self, message: str, **kwargs: Any) -> None:
        super().__init__(message, severity="WARNING", **kwargs)


# Convenience function for exception chaining
def chain_exception(
    new_exception: XPCSBaseError, original_exception: Exception
) -> XPCSBaseError:
    """
    Chain exceptions to preserve full error context with zero overhead.

    Usage::

        try:
            risky_operation()
        except ValueError as e:
            raise chain_exception(
                XPCSDataError("Failed to process data"),
                e
            )
    """
    # Add original exception context
    new_exception.add_context("original_exception", str(original_exception))
    new_exception.add_context("original_type", type(original_exception).__name__)

    # Chain the exceptions
    new_exception.__cause__ = original_exception
    return new_exception


# Exception mapping for automatic conversion
EXCEPTION_MAPPING = {
    # File and I/O related
    FileNotFoundError: XPCSFileError,
    PermissionError: XPCSFileError,
    OSError: XPCSFileError,
    IOError: XPCSFileError,
    # Data and validation related
    ValueError: XPCSValidationError,
    TypeError: XPCSValidationError,
    KeyError: XPCSDataError,
    IndexError: XPCSDataError,
    # Memory related
    MemoryError: XPCSMemoryError,
    OverflowError: XPCSMemoryError,
    # Computation related
    ArithmeticError: XPCSComputationError,
    ZeroDivisionError: XPCSComputationError,
    FloatingPointError: XPCSComputationError,
}


def convert_exception(
    original_exception: Exception, message: str | None = None
) -> XPCSBaseError:
    """
    Convert standard Python exceptions to XPCS exceptions with zero overhead.

    Args:
        original_exception: The original exception to convert
        message: Optional custom message (uses original message if not provided)

    Returns:
        Appropriate XPCS exception with original exception chained
    """
    exc_type = type(original_exception)
    xpcs_exception_class = EXCEPTION_MAPPING.get(exc_type, XPCSBaseError)

    final_message = message or str(original_exception)
    xpcs_exception = xpcs_exception_class(final_message)

    return chain_exception(xpcs_exception, original_exception)


# Decorator for automatic exception conversion
def handle_exceptions(
    default_exception: type = XPCSBaseError,
    convert_exceptions: bool = True,
    add_context: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for automatic exception handling and conversion.

    Zero overhead when no exceptions occur.

    Usage::

        @handle_exceptions(XPCSDataError, add_context={"operation": "data_loading"})
        def load_data(file_path):
            # Function implementation
            pass
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except XPCSBaseError:
                # Already an XPCS exception, re-raise as-is
                raise
            except Exception as e:
                if convert_exceptions:
                    xpcs_exception = convert_exception(e)
                else:
                    xpcs_exception = default_exception(str(e))
                    xpcs_exception = chain_exception(xpcs_exception, e)

                if add_context:
                    for key, value in add_context.items():
                        xpcs_exception.add_context(key, value)

                xpcs_exception.add_context("function", func.__name__)
                raise xpcs_exception

        return wrapper

    return decorator


# Exception context manager for resource safety
class exception_context:
    """
    Context manager for safe exception handling with resource cleanup.

    Zero overhead when no exceptions occur.

    Usage:
        with exception_context(XPCSFileError, "Failed to process file"):
            # Operations that might fail
            pass
    """

    def __init__(
        self,
        exception_type: type = XPCSBaseError,
        message: str = "Operation failed",
        cleanup_func: Callable[[], None] | None = None,
    ):
        self.exception_type = exception_type
        self.message = message
        self.cleanup_func = cleanup_func

    def __enter__(self) -> "exception_context":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback_obj: Any) -> None:
        if exc_type is not None:
            # Perform cleanup if provided
            if self.cleanup_func:
                try:
                    self.cleanup_func()
                except Exception:
                    pass  # Don't let cleanup failures mask original exception

            # Convert exception if needed
            if not isinstance(exc_value, XPCSBaseError):
                xpcs_exception = convert_exception(exc_value, self.message)
                raise xpcs_exception from exc_value

        # Don't suppress exceptions - no return needed for None
