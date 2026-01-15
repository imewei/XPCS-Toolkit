"""Tests for validation utility data integrity features.

Tests for Technical Guidelines compliance:
- T019: ValidationError on array mismatch
"""

import numpy as np
import pytest

from xpcsviewer.utils.exceptions import XPCSValidationError


class TestValidateArrayCompatibility:
    """Test validate_array_compatibility raises ValidationError on mismatch (T019)."""

    def test_raises_on_length_mismatch(self):
        """Verify ValidationError raised when arrays have different lengths."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([1, 2, 3])  # Different length

        with pytest.raises(XPCSValidationError) as exc_info:
            validate_array_compatibility(arr1, arr2)

        assert "Array length mismatch" in str(exc_info.value)
        assert "silent truncation is not allowed" in str(exc_info.value)

    def test_raises_on_length_mismatch_with_names(self):
        """Verify error message includes array names when provided."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3])

        with pytest.raises(XPCSValidationError) as exc_info:
            validate_array_compatibility(x, y, names=["x_data", "y_data"])

        error_msg = str(exc_info.value)
        assert "x_data" in error_msg
        assert "y_data" in error_msg

    def test_returns_true_on_compatible_arrays(self):
        """Verify returns True when arrays have same length."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([1, 2, 3, 4, 5])
        arr2 = np.array([5, 4, 3, 2, 1])

        result = validate_array_compatibility(arr1, arr2)
        assert result is True

    def test_raises_on_empty_arrays(self):
        """Verify ValidationError raised for empty arrays."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([])
        arr2 = np.array([1, 2, 3])

        with pytest.raises(XPCSValidationError) as exc_info:
            validate_array_compatibility(arr1, arr2)

        assert "Empty arrays" in str(exc_info.value)

    def test_raises_on_no_arrays(self):
        """Verify ValidationError raised when no arrays provided."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        with pytest.raises(XPCSValidationError) as exc_info:
            validate_array_compatibility()

        assert "No arrays provided" in str(exc_info.value)

    def test_raises_on_all_none_arrays(self):
        """Verify ValidationError raised when all arrays are None."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        with pytest.raises(XPCSValidationError) as exc_info:
            validate_array_compatibility(None, None)

        assert "All arrays are None" in str(exc_info.value)

    def test_handles_none_in_array_list(self):
        """Verify None values are filtered out before comparison."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])

        # Should succeed - None is filtered out
        result = validate_array_compatibility(arr1, None, arr2)
        assert result is True

    def test_multiple_arrays_all_same_length(self):
        """Verify multiple arrays with same length pass validation."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([5, 6, 7, 8])
        arr3 = np.array([9, 10, 11, 12])

        result = validate_array_compatibility(arr1, arr2, arr3)
        assert result is True

    def test_multiple_arrays_mismatch(self):
        """Verify ValidationError on any length mismatch in multiple arrays."""
        from xpcsviewer.utils.validation import validate_array_compatibility

        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([5, 6, 7, 8])
        arr3 = np.array([9, 10, 11])  # Different length

        with pytest.raises(XPCSValidationError):
            validate_array_compatibility(arr1, arr2, arr3)


class TestLegacyValidation:
    """Test legacy validation function is deprecated."""

    def test_legacy_function_shows_deprecation_warning(self):
        """Verify legacy function emits DeprecationWarning."""
        from xpcsviewer.utils.validation import validate_array_compatibility_legacy

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([1, 2, 3])

        with pytest.warns(DeprecationWarning):
            validate_array_compatibility_legacy(arr1, arr2)
