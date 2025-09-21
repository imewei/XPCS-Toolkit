"""
Validation utilities for XPCS data processing.

This module provides centralized validation functions to reduce code duplication
and ensure consistent error handling across the XPCS toolkit.
"""

import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def get_file_label_safe(xf) -> str:
    """
    Safely extract a label from an XPCS file object.

    Args:
        xf: XPCS file object that may or may not have a 'label' attribute

    Returns:
        str: The file label if available, otherwise 'unknown'
    """
    return getattr(xf, 'label', 'unknown')


def validate_xf_fit_summary(xf) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
    """
    Validate that an XPCS file object has a valid fit_summary with required fields.

    Args:
        xf: XPCS file object to validate

    Returns:
        Tuple containing:
        - bool: True if validation passed, False otherwise
        - dict or None: The fit_summary if valid, None otherwise
        - str or None: Error message if validation failed, None if passed
    """
    file_label = get_file_label_safe(xf)

    # Check if fit_summary exists
    if "fit_summary" not in xf.__dict__ or xf.fit_summary is None:
        error_msg = f"Skipping file {file_label} - no fit_summary"
        logger.debug(error_msg)
        return False, None, error_msg

    fit_summary = xf.fit_summary

    # Check for required fields
    if "q_val" not in fit_summary or "fit_val" not in fit_summary:
        error_msg = f"Skipping file {file_label} - missing q_val or fit_val"
        logger.debug(error_msg)
        return False, None, error_msg

    return True, fit_summary, None


def validate_xf_has_fit_summary(xf) -> Tuple[bool, Optional[str]]:
    """
    Simple validation that an XPCS file object has a fit_summary attribute.

    Args:
        xf: XPCS file object to validate

    Returns:
        Tuple containing:
        - bool: True if has fit_summary, False otherwise
        - str or None: Error message if validation failed, None if passed
    """
    file_label = get_file_label_safe(xf)

    if not hasattr(xf, "fit_summary") or xf.fit_summary is None:
        error_msg = f"Skipping file {file_label} - no fit_summary"
        logger.debug(error_msg)
        return False, error_msg

    return True, None


def validate_fit_summary_fields(fit_summary: Dict[str, Any], required_fields: list, file_label: str = "unknown") -> Tuple[bool, Optional[str]]:
    """
    Validate that a fit_summary dictionary contains required fields.

    Args:
        fit_summary: Dictionary to validate
        required_fields: List of required field names
        file_label: Label for error messaging

    Returns:
        Tuple containing:
        - bool: True if all fields present, False otherwise
        - str or None: Error message if validation failed, None if passed
    """
    missing_fields = [field for field in required_fields if field not in fit_summary]

    if missing_fields:
        error_msg = f"Skipping file {file_label} - missing fields: {', '.join(missing_fields)}"
        logger.debug(error_msg)
        return False, error_msg

    return True, None


def log_array_size_mismatch(file_label: str, array_info: Dict[str, int], min_length: int) -> None:
    """
    Log a standardized array size mismatch warning.

    Args:
        file_label: Label of the file with the mismatch
        array_info: Dictionary mapping array names to their lengths
        min_length: The minimum length arrays will be trimmed to
    """
    array_desc = ", ".join([f"{name}={length}" for name, length in array_info.items()])
    logger.warning(
        f"Array size mismatch in file {file_label}: {array_desc}. Trimming to {min_length}"
    )


def validate_array_compatibility(*arrays, file_label: str = "unknown") -> Tuple[bool, int, Optional[str]]:
    """
    Validate that arrays have compatible sizes and return the minimum length.

    Args:
        *arrays: Variable number of arrays to check
        file_label: Label for error messaging

    Returns:
        Tuple containing:
        - bool: True if arrays are compatible (after trimming), False if empty
        - int: Minimum length of arrays
        - str or None: Warning message if sizes differ, None if all same size
    """
    if not arrays:
        return False, 0, "No arrays provided for validation"

    lengths = [len(arr) for arr in arrays if arr is not None]

    if not lengths:
        return False, 0, f"All arrays are None for file {file_label}"

    min_length = min(lengths)
    max_length = max(lengths)

    warning_msg = None
    if min_length != max_length:
        array_info = {f"array_{i}": len(arr) for i, arr in enumerate(arrays) if arr is not None}
        log_array_size_mismatch(file_label, array_info, min_length)
        warning_msg = f"Array sizes differ, trimmed to {min_length}"

    if min_length == 0:
        return False, 0, f"Empty arrays for file {file_label}"

    return True, min_length, warning_msg