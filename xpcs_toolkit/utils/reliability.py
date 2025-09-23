"""
Enhanced Reliability Framework for XPCS Toolkit.

This module provides zero-overhead reliability mechanisms including:
- Fail-fast validation decorators
- Exception result caching
- Smart fallback strategies
- Performance-preserving reliability checks
"""

import functools
import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import numpy as np

from .exceptions import XPCSBaseError, XPCSValidationError, convert_exception
from .logging_config import get_logger

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels for performance tuning."""

    MINIMAL = "minimal"  # Only critical checks, maximum performance
    STANDARD = "standard"  # Balanced checks, good performance
    STRICT = "strict"  # Comprehensive checks, moderate performance
    PARANOID = "paranoid"  # All possible checks, thorough validation


@dataclass
class ValidationResult:
    """Result of validation operation with caching support."""

    is_valid: bool
    error_message: str | None = None
    warnings: list[str] = None
    validation_time: float = 0.0
    cached: bool = False

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class ValidationCache:
    """Thread-safe cache for validation results with TTL support."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 300.0):
        self._cache: dict[str, tuple[ValidationResult, float]] = {}
        self._lock = threading.RLock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._access_times: dict[str, float] = {}

    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function call signature."""
        # Create deterministic key from arguments
        key_data = {
            "function": func_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_string = str(key_data)
        return hashlib.md5(key_string.encode(), usedforsecurity=False).hexdigest()

    def get(self, key: str) -> ValidationResult | None:
        """Get cached validation result if still valid."""
        with self._lock:
            if key not in self._cache:
                return None

            result, timestamp = self._cache[key]
            current_time = time.time()

            # Check TTL
            if current_time - timestamp > self._default_ttl:
                del self._cache[key]
                self._access_times.pop(key, None)
                return None

            # Update access time for LRU
            self._access_times[key] = current_time
            result.cached = True
            return result

    def put(self, key: str, result: ValidationResult) -> None:
        """Store validation result in cache."""
        with self._lock:
            current_time = time.time()

            # Evict oldest entries if cache is full
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            self._cache[key] = (result, current_time)
            self._access_times[key] = current_time

    def _evict_lru(self) -> None:
        """Evict least recently used entries."""
        if not self._access_times:
            return

        # Remove 20% of oldest entries
        sorted_items = sorted(self._access_times.items(), key=lambda x: x[1])
        evict_count = max(1, len(sorted_items) // 5)

        for key, _ in sorted_items[:evict_count]:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)

    def clear(self) -> None:
        """Clear all cached results."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


# Global validation cache instance
_validation_cache = ValidationCache()


def validate_input(
    *,
    check_types: bool = True,
    check_ranges: bool = True,
    check_shapes: bool = True,
    check_values: bool = True,
    level: ValidationLevel = ValidationLevel.STANDARD,
    cache_results: bool = True,
    fast_fail: bool = True,
):
    """
    Zero-overhead input validation decorator with intelligent caching.

    Args:
        check_types: Validate parameter types
        check_ranges: Validate numerical ranges
        check_shapes: Validate array shapes
        check_values: Validate for NaN, infinity, etc.
        level: Validation strictness level
        cache_results: Cache validation results for performance
        fast_fail: Fail immediately on first validation error

    Example::

        @validate_input(check_ranges=True, level=ValidationLevel.STRICT)
        def process_data(data: np.ndarray, threshold: float = 0.1):
            # Function implementation
            pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip validation in minimal mode for performance
            if level == ValidationLevel.MINIMAL:
                return func(*args, **kwargs)

            # Generate cache key if caching enabled
            cache_key = None
            if cache_results:
                cache_key = _validation_cache._generate_key(func.__name__, args, kwargs)
                cached_result = _validation_cache.get(cache_key)
                if cached_result and cached_result.is_valid:
                    logger.debug(f"Using cached validation for {func.__name__}")
                    return func(*args, **kwargs)

            # Perform validation
            start_time = time.time()
            validation_errors = []
            validation_warnings = []

            try:
                # Get function signature for parameter validation
                import inspect

                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()

                for param_name, param_value in bound_args.arguments.items():
                    param_info = sig.parameters[param_name]

                    # Type checking
                    if check_types and param_info.annotation != inspect.Parameter.empty:
                        if not _validate_type(
                            param_value, param_info.annotation, param_name
                        ):
                            error_msg = f"Parameter '{param_name}' type mismatch"
                            if fast_fail:
                                raise XPCSValidationError(
                                    error_msg, field=param_name, value=param_value
                                )
                            validation_errors.append(error_msg)

                    # Range checking for numerical values
                    if (
                        check_ranges
                        and isinstance(param_value, (int, float, np.number))
                        and not _validate_range(param_value, param_name)
                    ):
                        error_msg = f"Parameter '{param_name}' outside valid range"
                        if fast_fail:
                            raise XPCSValidationError(
                                error_msg, field=param_name, value=param_value
                            )
                        validation_errors.append(error_msg)

                    # Array validation
                    if check_shapes and isinstance(param_value, np.ndarray):
                        shape_errors, shape_warnings = _validate_array(
                            param_value, param_name, level
                        )
                        if shape_errors:
                            if fast_fail:
                                raise XPCSValidationError(
                                    shape_errors[0],
                                    field=param_name,
                                    value=param_value.shape,
                                )
                            validation_errors.extend(shape_errors)
                        validation_warnings.extend(shape_warnings)

                    # Value checking
                    if check_values:
                        value_errors, value_warnings = _validate_values(
                            param_value, param_name, level
                        )
                        if value_errors:
                            if fast_fail:
                                raise XPCSValidationError(
                                    value_errors[0], field=param_name, value=param_value
                                )
                            validation_errors.extend(value_errors)
                        validation_warnings.extend(value_warnings)

                # Create validation result
                validation_time = time.time() - start_time
                is_valid = len(validation_errors) == 0

                result = ValidationResult(
                    is_valid=is_valid,
                    error_message="; ".join(validation_errors)
                    if validation_errors
                    else None,
                    warnings=validation_warnings,
                    validation_time=validation_time,
                )

                # Cache result if enabled
                if cache_results and cache_key:
                    _validation_cache.put(cache_key, result)

                # Raise exception if validation failed
                if not is_valid:
                    raise XPCSValidationError(result.error_message)

                # Log warnings
                for warning in validation_warnings:
                    logger.warning(f"Validation warning in {func.__name__}: {warning}")

                return func(*args, **kwargs)

            except Exception as e:
                if isinstance(e, XPCSValidationError):
                    raise
                # Convert unexpected validation errors
                xpcs_error = convert_exception(
                    e, f"Validation failed for {func.__name__}"
                )
                raise xpcs_error

        return wrapper

    return decorator


def _validate_type(value: Any, expected_type: type, param_name: str) -> bool:
    """Validate parameter type with support for complex types."""
    try:
        # Handle Union types and generic types
        if hasattr(expected_type, "__origin__"):
            if expected_type.__origin__ is Union:
                return any(isinstance(value, t) for t in expected_type.__args__)
            # Handle other generic types as needed
            return isinstance(value, expected_type.__origin__)
        return isinstance(value, expected_type)
    except Exception:
        # If type checking fails, assume it's valid to avoid breaking functionality
        return True


def _validate_range(value: int | float | np.number, param_name: str) -> bool:
    """Validate numerical ranges with domain-specific checks."""
    # Check for NaN and infinity
    if np.isnan(value) or np.isinf(value):
        return False

    # Domain-specific range checks
    if "threshold" in param_name.lower() or "alpha" in param_name.lower():
        return 0.0 <= value <= 1.0
    if "q_" in param_name.lower() or param_name.lower().endswith("_q"):
        return value > 0.0  # Q-values should be positive
    if "time" in param_name.lower() or "t_" in param_name.lower():
        return value >= 0.0  # Time values should be non-negative

    return True  # Default: accept all finite values


def _validate_array(
    array: np.ndarray, param_name: str, level: ValidationLevel
) -> tuple[list[str], list[str]]:
    """Validate numpy array properties."""
    errors = []
    warnings = []

    # Check for empty arrays
    if array.size == 0:
        errors.append(f"Array '{param_name}' is empty")
        return errors, warnings

    # Check for reasonable array sizes to prevent memory issues
    array_size_mb = array.nbytes / (1024 * 1024)
    if array_size_mb > 1000:  # > 1GB
        if level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            warnings.append(f"Large array '{param_name}': {array_size_mb:.1f}MB")

    # Check for appropriate data types
    if array.dtype == np.object_:
        warnings.append(f"Object array '{param_name}' may cause performance issues")

    # Domain-specific shape validation
    if "saxs" in param_name.lower():
        if array.ndim not in [2, 3]:
            errors.append(
                f"SAXS data '{param_name}' should be 2D or 3D, got {array.ndim}D"
            )
    elif "g2" in param_name.lower() or "correlation" in param_name.lower():
        if array.ndim != 2:
            errors.append(
                f"Correlation data '{param_name}' should be 2D, got {array.ndim}D"
            )

    return errors, warnings


def _validate_values(
    value: Any, param_name: str, level: ValidationLevel
) -> tuple[list[str], list[str]]:
    """Validate data values for scientific correctness."""
    errors = []
    warnings = []

    if isinstance(value, np.ndarray):
        # Check for NaN and infinity
        if np.any(np.isnan(value)):
            if level == ValidationLevel.PARANOID:
                errors.append(f"Array '{param_name}' contains NaN values")
            else:
                warnings.append(f"Array '{param_name}' contains NaN values")

        if np.any(np.isinf(value)):
            errors.append(f"Array '{param_name}' contains infinite values")

        # Check for reasonable value ranges
        if np.issubdtype(value.dtype, np.number) and value.size > 0:
            min_val, max_val = np.min(value), np.max(value)
            value_range = max_val - min_val

            if value_range == 0 and level in [
                ValidationLevel.STRICT,
                ValidationLevel.PARANOID,
            ]:
                warnings.append(f"Array '{param_name}' has zero variance")

            # Domain-specific value checks
            if "intensity" in param_name.lower() or "saxs" in param_name.lower():
                if min_val < 0:
                    warnings.append(f"Negative intensity values in '{param_name}'")

    elif isinstance(value, (int, float, np.number)):
        if np.isnan(value):
            errors.append(f"Parameter '{param_name}' is NaN")
        elif np.isinf(value):
            errors.append(f"Parameter '{param_name}' is infinite")

    return errors, warnings


class SmartFallbackManager:
    """Manager for smart fallback strategies with pre-computed paths."""

    def __init__(self):
        self._fallback_strategies: dict[str, list[Callable]] = {}
        self._performance_history: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def register_fallback_chain(
        self, operation_name: str, strategies: list[Callable]
    ) -> None:
        """Register a chain of fallback strategies for an operation."""
        with self._lock:
            self._fallback_strategies[operation_name] = strategies
            self._performance_history[operation_name] = []

    def execute_with_fallback(self, operation_name: str, *args, **kwargs) -> Any:
        """Execute operation with automatic fallback on failure."""
        strategies = self._fallback_strategies.get(operation_name, [])
        if not strategies:
            raise XPCSBaseError(
                f"No fallback strategies registered for '{operation_name}'"
            )

        last_exception = None
        performance_start = time.time()

        for i, strategy in enumerate(strategies):
            try:
                logger.debug(
                    f"Attempting {operation_name} strategy {i + 1}/{len(strategies)}: {strategy.__name__}"
                )
                result = strategy(*args, **kwargs)

                # Record successful performance
                execution_time = time.time() - performance_start
                with self._lock:
                    self._performance_history[operation_name].append(execution_time)
                    # Keep only recent history
                    if len(self._performance_history[operation_name]) > 100:
                        self._performance_history[operation_name] = (
                            self._performance_history[operation_name][-50:]
                        )

                logger.debug(
                    f"Successfully executed {operation_name} with strategy {i + 1}"
                )
                return result

            except Exception as e:
                logger.debug(f"Strategy {i + 1} failed for {operation_name}: {e}")
                last_exception = e
                continue

        # All strategies failed
        if last_exception:
            raise convert_exception(
                last_exception, f"All fallback strategies failed for '{operation_name}'"
            )
        raise XPCSBaseError(
            f"All strategies failed for '{operation_name}' with no exceptions"
        )

    def get_performance_stats(self, operation_name: str) -> dict[str, float]:
        """Get performance statistics for an operation."""
        with self._lock:
            history = self._performance_history.get(operation_name, [])
            if not history:
                return {}

            return {
                "mean_time": np.mean(history),
                "std_time": np.std(history),
                "min_time": np.min(history),
                "max_time": np.max(history),
                "success_count": len(history),
            }


# Global fallback manager instance
_fallback_manager = SmartFallbackManager()


def with_fallback(operation_name: str, strategies: list[Callable] | None = None):
    """
    Decorator for automatic fallback execution with pre-computed strategies.

    Args:
        operation_name: Name of the operation for strategy registration
        strategies: List of fallback functions (if not already registered)

    Example::

        @with_fallback("data_loading", [load_enhanced, load_standard])
        def load_data(file_path):
            # Primary implementation - automatically falls back if it fails
            pass
    """

    def decorator(func: Callable) -> Callable:
        # Register strategies if provided
        if strategies:
            _fallback_manager.register_fallback_chain(
                operation_name, [func, *strategies]
            )
        else:
            # Use just the function itself
            _fallback_manager.register_fallback_chain(operation_name, [func])

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return _fallback_manager.execute_with_fallback(
                operation_name, *args, **kwargs
            )

        return wrapper

    return decorator


class ReliabilityContext:
    """Context manager for enhanced reliability with retries and exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        exponential_backoff: bool = True,
        acceptable_exceptions: tuple[type, ...] | None = None,
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.acceptable_exceptions = acceptable_exceptions or (
            OSError,
            IOError,
            TimeoutError,
            ConnectionError,
        )
        self.retry_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            return False  # No exception, normal exit

        if issubclass(exc_type, self.acceptable_exceptions):
            self.retry_count += 1

            if self.retry_count <= self.max_retries:
                # Calculate delay with optional exponential backoff
                if self.exponential_backoff:
                    delay = self.retry_delay * (2 ** (self.retry_count - 1))
                else:
                    delay = self.retry_delay

                logger.debug(
                    f"Retry {self.retry_count}/{self.max_retries} after {delay:.2f}s delay: {exc_val}"
                )
                time.sleep(delay)
                return True  # Suppress the exception to allow retry
            # Max retries exceeded, let exception propagate
            return False
        # Non-retryable exception - let it propagate
        return False


def reliability_context(
    max_retries: int = 3,
    retry_delay: float = 0.1,
    exponential_backoff: bool = True,
    acceptable_exceptions: tuple[type, ...] | None = None,
):
    """
    Context manager for enhanced reliability with retries and exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        retry_delay: Base delay between retries (seconds)
        exponential_backoff: Use exponential backoff for delays
        acceptable_exceptions: Exception types that should be retried

    Example::

        attempt = 0
        while attempt <= max_retries:
            try:
                with reliability_context(max_retries=3, retry_delay=0.5):
                    risky_operation()
                break  # Success
            except (OSError, ConnectionError) as e:
                attempt += 1
                if attempt > max_retries:
                    raise
    """
    return ReliabilityContext(
        max_retries, retry_delay, exponential_backoff, acceptable_exceptions
    )


def get_validation_cache() -> ValidationCache:
    """Get the global validation cache instance."""
    return _validation_cache


def get_fallback_manager() -> SmartFallbackManager:
    """Get the global fallback manager instance."""
    return _fallback_manager


def clear_reliability_caches() -> None:
    """Clear all reliability-related caches for testing or memory management."""
    _validation_cache.clear()
    logger.debug("Reliability caches cleared")


# Performance monitoring for reliability overhead
class ReliabilityProfiler:
    """Lightweight profiler for reliability overhead measurement."""

    def __init__(self):
        self._stats: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def record_overhead(self, operation: str, overhead_time: float) -> None:
        """Record reliability overhead for an operation."""
        with self._lock:
            if operation not in self._stats:
                self._stats[operation] = []
            self._stats[operation].append(overhead_time)
            # Keep only recent samples
            if len(self._stats[operation]) > 1000:
                self._stats[operation] = self._stats[operation][-500:]

    def get_overhead_stats(self, operation: str | None = None) -> dict[str, Any]:
        """Get overhead statistics for operations."""
        with self._lock:
            if operation:
                times = self._stats.get(operation, [])
                if not times:
                    return {}
                return {
                    "mean_overhead_ms": np.mean(times) * 1000,
                    "max_overhead_ms": np.max(times) * 1000,
                    "total_calls": len(times),
                }
            return {op: self.get_overhead_stats(op) for op in self._stats}


# Global profiler instance
_reliability_profiler = ReliabilityProfiler()


def get_reliability_profiler() -> ReliabilityProfiler:
    """Get the global reliability profiler instance."""
    return _reliability_profiler
