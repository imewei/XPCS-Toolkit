"""
Common utility functions for consistent validation and checking patterns.

This module provides standardized utility functions to reduce code duplication
across validation, error checking, and common operations in the XPCS toolkit.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def is_empty_list(obj: Any) -> bool:
    """
    Standardized check for empty lists or list-like objects.

    Args:
        obj: Object to check for emptiness

    Returns:
        True if object is empty or None, False otherwise
    """
    if obj is None:
        return True

    try:
        return len(obj) == 0
    except TypeError:
        # Object doesn't support len(), so it's not list-like
        return False


def is_empty_or_none(obj: Any) -> bool:
    """
    Check if object is None or empty (for various container types).

    Args:
        obj: Object to check

    Returns:
        True if object is None or empty, False otherwise
    """
    if obj is None:
        return True

    # Handle different container types
    try:
        if hasattr(obj, "__len__"):
            return len(obj) == 0
        if isinstance(obj, (str, bytes)):
            return len(obj) == 0
        if hasattr(obj, "size"):  # NumPy arrays
            return obj.size == 0
        return False
    except:
        return False


def safe_array_access(arr: Any, index: int | slice, default: Any = None) -> Any:
    """
    Safely access array elements with bounds checking.

    Args:
        arr: Array-like object to access
        index: Index or slice to access
        default: Default value to return if access fails

    Returns:
        Array element/slice or default value
    """
    try:
        if arr is None:
            return default

        # Check bounds for integer indices
        if isinstance(index, int):
            if hasattr(arr, "__len__") and (index >= len(arr) or index < -len(arr)):
                return default

        return arr[index]
    except (IndexError, TypeError, KeyError):
        return default


def safe_dict_access(d: dict, key: str, default: Any = None) -> Any:
    """
    Safely access dictionary values with None checking.

    Args:
        d: Dictionary to access
        key: Key to access
        default: Default value if key not found or dict is None

    Returns:
        Dictionary value or default
    """
    if d is None:
        return default
    return d.get(key, default)


def validate_numeric_range(
    value: Any,
    min_val: float | None = None,
    max_val: float | None = None,
    param_name: str = "value",
) -> tuple[bool, str | None]:
    """
    Validate that a numeric value is within specified range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        param_name: Parameter name for error messages

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        num_value = float(value)
    except (TypeError, ValueError):
        return (
            False,
            f"{param_name} must be a numeric value, got {type(value).__name__}",
        )

    if not np.isfinite(num_value):
        return False, f"{param_name} must be finite, got {num_value}"

    if min_val is not None and num_value < min_val:
        return False, f"{param_name} must be >= {min_val}, got {num_value}"

    if max_val is not None and num_value > max_val:
        return False, f"{param_name} must be <= {max_val}, got {num_value}"

    return True, None


def ensure_list(obj: Any) -> list[Any]:
    """
    Ensure object is a list, converting if necessary.

    Args:
        obj: Object to convert to list

    Returns:
        List representation of object
    """
    if obj is None:
        return []
    if isinstance(obj, list):
        return obj
    if isinstance(obj, (tuple, set)):
        return list(obj)
    if hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
        return list(obj)
    return [obj]


def safe_min_max(
    values: list[Any], default_min: float = 0.0, default_max: float = 1.0
) -> tuple[float, float]:
    """
    Safely compute min and max of a list with fallback defaults.

    Args:
        values: List of values to compute min/max for
        default_min: Default minimum if computation fails
        default_max: Default maximum if computation fails

    Returns:
        Tuple of (min_value, max_value)
    """
    try:
        if not values:
            return default_min, default_max

        # Filter out non-numeric and non-finite values
        numeric_values = []
        for val in values:
            try:
                num_val = float(val)
                if np.isfinite(num_val):
                    numeric_values.append(num_val)
            except (TypeError, ValueError):
                continue

        if not numeric_values:
            return default_min, default_max

        return min(numeric_values), max(numeric_values)

    except Exception:
        return default_min, default_max


def check_required_attributes(
    obj: Any, required_attrs: list[str], obj_name: str = "object"
) -> tuple[bool, list[str]]:
    """
    Check if an object has all required attributes.

    Args:
        obj: Object to check
        required_attrs: List of required attribute names
        obj_name: Name of object for error messages

    Returns:
        Tuple of (all_present, missing_attributes)
    """
    if obj is None:
        return False, required_attrs

    missing_attrs = []
    for attr in required_attrs:
        if not hasattr(obj, attr):
            missing_attrs.append(attr)

    return len(missing_attrs) == 0, missing_attrs


def log_validation_warning(message: str, category: str = "validation") -> None:
    """
    Log a standardized validation warning.

    Args:
        message: Warning message
        category: Category for logging filtering
    """
    logger.warning(f"[{category.upper()}] {message}")


def log_validation_error(message: str, category: str = "validation") -> None:
    """
    Log a standardized validation error.

    Args:
        message: Error message
        category: Category for logging filtering
    """
    logger.error(f"[{category.upper()}] {message}")


def standardize_file_rows(rows: Any) -> list[int]:
    """
    Standardize file row selection to a list of integers.

    Args:
        rows: Row selection (can be None, int, or list)

    Returns:
        List of integer row indices
    """
    if rows is None:
        return []
    if isinstance(rows, int):
        return [rows]
    if isinstance(rows, (list, tuple)):
        # Filter to valid integers
        valid_rows = []
        for row in rows:
            try:
                valid_rows.append(int(row))
            except (TypeError, ValueError):
                log_validation_warning(f"Invalid row index: {row}, skipping")
        return valid_rows
    log_validation_warning(f"Unexpected rows type: {type(rows)}, returning empty list")
    return []


def create_progress_steps(
    base_steps: list[str], prefix: str | None = None
) -> list[str]:
    """
    Create standardized progress step descriptions.

    Args:
        base_steps: Base list of step descriptions
        prefix: Optional prefix to add to each step

    Returns:
        List of formatted step descriptions
    """
    if prefix:
        return [f"{prefix}: {step}" for step in base_steps]
    return base_steps.copy()


# Common validation patterns as constants
COMMON_REQUIRED_FIELDS = {
    "fit_summary": ["q_val", "fit_val"],
    "saxs_data": ["intensity", "q_values"],
    "g2_data": ["time", "correlation", "error"],
    "twotime_data": ["time_1", "time_2", "correlation"],
}
