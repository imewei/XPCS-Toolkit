"""NumPy backend implementation (CPU fallback).

This backend provides the baseline implementation using NumPy.
It's used when JAX is not available or when explicitly selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable


class NumPyBackend:
    """NumPy-based backend for array operations.

    This backend provides CPU-only computation using NumPy.
    JIT compilation and gradient computation are not supported.
    """

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "numpy"

    @property
    def supports_gpu(self) -> bool:
        """NumPy does not support GPU."""
        return False

    @property
    def supports_jit(self) -> bool:
        """NumPy does not support JIT compilation."""
        return False

    @property
    def supports_grad(self) -> bool:
        """NumPy does not support auto-differentiation."""
        return False

    @property
    def pi(self) -> float:
        """Mathematical constant π."""
        return float(np.pi)

    # =========================================================================
    # Array Creation
    # =========================================================================

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Create array filled with zeros."""
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> np.ndarray:
        """Create array filled with ones."""
        return np.ones(shape, dtype=dtype)

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1,
        dtype: Any = None,
    ) -> np.ndarray:
        """Create array with evenly spaced values."""
        return np.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int) -> np.ndarray:
        """Create array with linearly spaced values."""
        return np.linspace(start, stop, num)

    def logspace(self, start: float, stop: float, num: int) -> np.ndarray:
        """Create array with logarithmically spaced values."""
        return np.logspace(start, stop, num)

    def meshgrid(self, *xi: np.ndarray, indexing: str = "xy") -> tuple[np.ndarray, ...]:
        """Create coordinate matrices from coordinate vectors."""
        return tuple(np.meshgrid(*xi, indexing=indexing))

    def zeros_like(self, x: np.ndarray, dtype: Any = None) -> np.ndarray:
        """Create zero-filled array with same shape as input."""
        return np.zeros_like(x, dtype=dtype)

    def ones_like(self, x: np.ndarray, dtype: Any = None) -> np.ndarray:
        """Create ones-filled array with same shape as input."""
        return np.ones_like(x, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any = None
    ) -> np.ndarray:
        """Create array filled with specified value."""
        return np.full(shape, fill_value, dtype=dtype)

    def array(self, data: Any, dtype: Any = None) -> np.ndarray:
        """Create array from data."""
        return np.array(data, dtype=dtype)

    # =========================================================================
    # Trigonometric Functions
    # =========================================================================

    def sin(self, x: np.ndarray) -> np.ndarray:
        """Element-wise sine."""
        return np.sin(x)

    def cos(self, x: np.ndarray) -> np.ndarray:
        """Element-wise cosine."""
        return np.cos(x)

    def arctan(self, x: np.ndarray) -> np.ndarray:
        """Element-wise arctangent."""
        return np.arctan(x)

    def arctan2(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Element-wise arctangent of y/x, handling quadrants."""
        return np.arctan2(y, x)

    def hypot(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise sqrt(x^2 + y^2)."""
        return np.hypot(x, y)

    def deg2rad(self, x: np.ndarray) -> np.ndarray:
        """Convert degrees to radians."""
        return np.deg2rad(x)

    def rad2deg(self, x: np.ndarray) -> np.ndarray:
        """Convert radians to degrees."""
        return np.rad2deg(x)

    def mod(self, x: np.ndarray, y: np.ndarray | float) -> np.ndarray:
        """Element-wise modulo."""
        return np.mod(x, y)

    def floor(self, x: np.ndarray) -> np.ndarray:
        """Element-wise floor."""
        return np.floor(x)

    def ceil(self, x: np.ndarray) -> np.ndarray:
        """Element-wise ceiling."""
        return np.ceil(x)

    def round(self, x: np.ndarray, decimals: int = 0) -> np.ndarray:
        """Round to given number of decimals."""
        return np.round(x, decimals=decimals)

    # =========================================================================
    # Statistical Functions
    # =========================================================================

    def mean(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute mean along axis."""
        return np.mean(x, axis=axis)

    def std(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute standard deviation along axis."""
        return np.std(x, axis=axis)

    def nanmean(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute mean, ignoring NaN values."""
        return np.nanmean(x, axis=axis)

    def nanmin(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute minimum, ignoring NaN values."""
        return np.nanmin(x, axis=axis)

    def nanmax(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute maximum, ignoring NaN values."""
        return np.nanmax(x, axis=axis)

    def percentile(
        self, x: np.ndarray, q: float, axis: int | None = None
    ) -> np.ndarray:
        """Compute percentile along axis."""
        return np.percentile(x, q, axis=axis)

    def sum(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute sum along axis."""
        return np.sum(x, axis=axis)

    def min(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute minimum along axis."""
        return np.min(x, axis=axis)

    def max(self, x: np.ndarray, axis: int | None = None) -> np.ndarray:
        """Compute maximum along axis."""
        return np.max(x, axis=axis)

    # =========================================================================
    # Binning Functions
    # =========================================================================

    def digitize(self, x: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """Return indices of bins to which each value belongs."""
        return np.digitize(x, bins)

    def bincount(
        self,
        x: np.ndarray,
        weights: np.ndarray | None = None,
        minlength: int = 0,
    ) -> np.ndarray:
        """Count number of occurrences of each value."""
        return np.bincount(x.astype(np.int64), weights=weights, minlength=minlength)

    def unique(
        self,
        x: np.ndarray,
        return_inverse: bool = False,
        size: int | None = None,
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """Find unique elements of array.

        Note: size parameter is ignored for NumPy (used by JAX for JIT).
        """
        return np.unique(x, return_inverse=return_inverse)

    # =========================================================================
    # Boolean/Masking Functions
    # =========================================================================

    def logical_and(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise logical AND."""
        return np.logical_and(x, y)

    def logical_or(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Element-wise logical OR."""
        return np.logical_or(x, y)

    def logical_not(self, x: np.ndarray) -> np.ndarray:
        """Element-wise logical NOT."""
        return np.logical_not(x)

    def where(self, condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return elements chosen from x or y depending on condition."""
        return np.where(condition, x, y)

    def nonzero(self, x: np.ndarray, size: int | None = None) -> tuple[np.ndarray, ...]:
        """Return indices of non-zero elements.

        Note: size parameter is ignored for NumPy (used by JAX for JIT).
        """
        return np.nonzero(x)

    def isnan(self, x: np.ndarray) -> np.ndarray:
        """Test element-wise for NaN."""
        return np.isnan(x)

    def isfinite(self, x: np.ndarray) -> np.ndarray:
        """Test element-wise for finite values."""
        return np.isfinite(x)

    # =========================================================================
    # Array Manipulation
    # =========================================================================

    def clip(self, x: np.ndarray, a_min: float, a_max: float) -> np.ndarray:
        """Clip array values to specified range."""
        return np.clip(x, a_min, a_max)

    def stack(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Stack arrays along new axis."""
        return np.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        """Concatenate arrays along existing axis."""
        return np.concatenate(arrays, axis=axis)

    def copy(self, x: np.ndarray) -> np.ndarray:
        """Return copy of array."""
        return np.copy(x)

    def reshape(self, x: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Reshape array to specified shape."""
        return np.reshape(x, shape)

    def transpose(
        self, x: np.ndarray, axes: tuple[int, ...] | None = None
    ) -> np.ndarray:
        """Permute array dimensions."""
        return np.transpose(x, axes=axes)

    def flatten(self, x: np.ndarray) -> np.ndarray:
        """Flatten array to 1D."""
        return x.flatten()

    # =========================================================================
    # Mathematical Functions
    # =========================================================================

    def exp(self, x: np.ndarray) -> np.ndarray:
        """Element-wise exponential."""
        return np.exp(x)

    def log(self, x: np.ndarray) -> np.ndarray:
        """Element-wise natural logarithm."""
        return np.log(x)

    def log10(self, x: np.ndarray) -> np.ndarray:
        """Element-wise base-10 logarithm."""
        return np.log10(x)

    def sqrt(self, x: np.ndarray) -> np.ndarray:
        """Element-wise square root."""
        return np.sqrt(x)

    def abs(self, x: np.ndarray) -> np.ndarray:
        """Element-wise absolute value."""
        return np.abs(x)

    def power(self, x: np.ndarray, y: float | np.ndarray) -> np.ndarray:
        """Element-wise power."""
        return np.power(x, y)

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        """Convert array to NumPy ndarray (identity for NumPy)."""
        return np.asarray(x)

    def from_numpy(self, x: np.ndarray) -> np.ndarray:
        """Convert NumPy ndarray to backend array (identity)."""
        return x

    def astype(self, x: np.ndarray, dtype: Any) -> np.ndarray:
        """Cast array to specified dtype."""
        return x.astype(dtype)

    # =========================================================================
    # JIT Compilation (no-op for NumPy)
    # =========================================================================

    def jit(
        self,
        func: Callable,
        static_argnums: tuple[int, ...] | None = None,
    ) -> Callable:
        """JIT compile function (no-op for NumPy)."""
        return func

    # =========================================================================
    # Gradient Computation (not supported)
    # =========================================================================

    def grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return gradient function.

        Raises
        ------
        NotImplementedError
            NumPy does not support automatic differentiation.
        """
        raise NotImplementedError(
            "NumPy backend does not support automatic differentiation. "
            "Use JAX backend for gradient computation."
        )

    def value_and_grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return function computing both value and gradient.

        Raises
        ------
        NotImplementedError
            NumPy does not support automatic differentiation.
        """
        raise NotImplementedError(
            "NumPy backend does not support automatic differentiation. "
            "Use JAX backend for gradient computation."
        )
