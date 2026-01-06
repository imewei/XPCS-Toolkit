"""JAX backend implementation with GPU and JIT support.

This backend provides GPU-accelerated computation using JAX with
JIT compilation and automatic differentiation support.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax.numpy as jnp

# Import JAX lazily to allow fallback when not installed
_jax = None
_jnp = None


def _ensure_jax():
    """Ensure JAX is imported and configured."""
    global _jax, _jnp
    if _jax is None:
        import jax
        import jax.numpy as jnp_module

        _jax = jax
        _jnp = jnp_module


class JAXBackend:
    """JAX-based backend for array operations.

    This backend provides GPU-accelerated computation with JIT
    compilation and automatic differentiation support.
    """

    def __init__(self):
        """Initialize JAX backend."""
        _ensure_jax()

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "jax"

    @property
    def supports_gpu(self) -> bool:
        """JAX can support GPU if available."""
        _ensure_jax()
        try:
            devices = _jax.devices("gpu")
            return len(devices) > 0
        except RuntimeError:
            return False

    @property
    def supports_jit(self) -> bool:
        """JAX supports JIT compilation."""
        return True

    @property
    def supports_grad(self) -> bool:
        """JAX supports automatic differentiation."""
        return True

    @property
    def pi(self) -> float:
        """Mathematical constant π."""
        _ensure_jax()
        return float(_jnp.pi)

    # =========================================================================
    # Array Creation
    # =========================================================================

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> jnp.ndarray:
        """Create array filled with zeros."""
        _ensure_jax()
        return _jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> jnp.ndarray:
        """Create array filled with ones."""
        _ensure_jax()
        return _jnp.ones(shape, dtype=dtype)

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1,
        dtype: Any = None,
    ) -> jnp.ndarray:
        """Create array with evenly spaced values."""
        _ensure_jax()
        return _jnp.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int) -> jnp.ndarray:
        """Create array with linearly spaced values."""
        _ensure_jax()
        return _jnp.linspace(start, stop, num)

    def logspace(self, start: float, stop: float, num: int) -> jnp.ndarray:
        """Create array with logarithmically spaced values."""
        _ensure_jax()
        return _jnp.logspace(start, stop, num)

    def meshgrid(
        self, *xi: jnp.ndarray, indexing: str = "xy"
    ) -> tuple[jnp.ndarray, ...]:
        """Create coordinate matrices from coordinate vectors."""
        _ensure_jax()
        return tuple(_jnp.meshgrid(*xi, indexing=indexing))

    def zeros_like(self, x: jnp.ndarray, dtype: Any = None) -> jnp.ndarray:
        """Create zero-filled array with same shape as input."""
        _ensure_jax()
        return _jnp.zeros_like(x, dtype=dtype)

    def ones_like(self, x: jnp.ndarray, dtype: Any = None) -> jnp.ndarray:
        """Create ones-filled array with same shape as input."""
        _ensure_jax()
        return _jnp.ones_like(x, dtype=dtype)

    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any = None
    ) -> jnp.ndarray:
        """Create array filled with specified value."""
        _ensure_jax()
        return _jnp.full(shape, fill_value, dtype=dtype)

    def array(self, data: Any, dtype: Any = None) -> jnp.ndarray:
        """Create array from data."""
        _ensure_jax()
        return _jnp.array(data, dtype=dtype)

    # =========================================================================
    # Trigonometric Functions
    # =========================================================================

    def sin(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise sine."""
        _ensure_jax()
        return _jnp.sin(x)

    def cos(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise cosine."""
        _ensure_jax()
        return _jnp.cos(x)

    def arctan(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise arctangent."""
        _ensure_jax()
        return _jnp.arctan(x)

    def arctan2(self, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise arctangent of y/x, handling quadrants."""
        _ensure_jax()
        return _jnp.arctan2(y, x)

    def hypot(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Element-wise sqrt(x^2 + y^2)."""
        _ensure_jax()
        return _jnp.hypot(x, y)

    def deg2rad(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert degrees to radians."""
        _ensure_jax()
        return _jnp.deg2rad(x)

    def rad2deg(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convert radians to degrees."""
        _ensure_jax()
        return _jnp.rad2deg(x)

    def mod(self, x: jnp.ndarray, y: jnp.ndarray | float) -> jnp.ndarray:
        """Element-wise modulo."""
        _ensure_jax()
        return _jnp.mod(x, y)

    def floor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise floor."""
        _ensure_jax()
        return _jnp.floor(x)

    def ceil(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise ceiling."""
        _ensure_jax()
        return _jnp.ceil(x)

    def round(self, x: jnp.ndarray, decimals: int = 0) -> jnp.ndarray:
        """Round to given number of decimals."""
        _ensure_jax()
        return _jnp.round(x, decimals=decimals)

    # =========================================================================
    # Statistical Functions
    # =========================================================================

    def mean(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute mean along axis."""
        _ensure_jax()
        return _jnp.mean(x, axis=axis)

    def std(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute standard deviation along axis."""
        _ensure_jax()
        return _jnp.std(x, axis=axis)

    def nanmean(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute mean, ignoring NaN values."""
        _ensure_jax()
        return _jnp.nanmean(x, axis=axis)

    def nanmin(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute minimum, ignoring NaN values."""
        _ensure_jax()
        return _jnp.nanmin(x, axis=axis)

    def nanmax(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute maximum, ignoring NaN values."""
        _ensure_jax()
        return _jnp.nanmax(x, axis=axis)

    def percentile(
        self, x: jnp.ndarray, q: float, axis: int | None = None
    ) -> jnp.ndarray:
        """Compute percentile along axis."""
        _ensure_jax()
        return _jnp.percentile(x, q, axis=axis)

    def sum(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute sum along axis."""
        _ensure_jax()
        return _jnp.sum(x, axis=axis)

    def min(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute minimum along axis."""
        _ensure_jax()
        return _jnp.min(x, axis=axis)

    def max(self, x: jnp.ndarray, axis: int | None = None) -> jnp.ndarray:
        """Compute maximum along axis."""
        _ensure_jax()
        return _jnp.max(x, axis=axis)

    # =========================================================================
    # Binning Functions
    # =========================================================================

    def digitize(self, x: jnp.ndarray, bins: jnp.ndarray) -> jnp.ndarray:
        """Return indices of bins to which each value belongs."""
        _ensure_jax()
        return _jnp.digitize(x, bins)

    def bincount(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray | None = None,
        minlength: int = 0,
    ) -> jnp.ndarray:
        """Count number of occurrences of each value."""
        _ensure_jax()
        return _jnp.bincount(x.astype(_jnp.int32), weights=weights, minlength=minlength)

    def unique(
        self,
        x: jnp.ndarray,
        return_inverse: bool = False,
        size: int | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, ...]:
        """Find unique elements of array.

        Parameters
        ----------
        x : array
            Input array
        return_inverse : bool
            If True, also return indices to reconstruct x
        size : int, optional
            Expected number of unique elements (required for JIT)
        """
        _ensure_jax()
        return _jnp.unique(x, return_inverse=return_inverse, size=size)

    # =========================================================================
    # Boolean/Masking Functions
    # =========================================================================

    def logical_and(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Element-wise logical AND."""
        _ensure_jax()
        return _jnp.logical_and(x, y)

    def logical_or(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Element-wise logical OR."""
        _ensure_jax()
        return _jnp.logical_or(x, y)

    def logical_not(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise logical NOT."""
        _ensure_jax()
        return _jnp.logical_not(x)

    def where(
        self, condition: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        """Return elements chosen from x or y depending on condition."""
        _ensure_jax()
        return _jnp.where(condition, x, y)

    def nonzero(
        self, x: jnp.ndarray, size: int | None = None
    ) -> tuple[jnp.ndarray, ...]:
        """Return indices of non-zero elements.

        Parameters
        ----------
        x : array
            Input array
        size : int, optional
            Expected number of non-zero elements (required for JIT).
            Results are padded with -1 if actual count is less.
        """
        _ensure_jax()
        if size is not None:
            return _jnp.nonzero(x, size=size, fill_value=-1)
        return _jnp.nonzero(x, size=x.size, fill_value=-1)

    def isnan(self, x: jnp.ndarray) -> jnp.ndarray:
        """Test element-wise for NaN."""
        _ensure_jax()
        return _jnp.isnan(x)

    def isfinite(self, x: jnp.ndarray) -> jnp.ndarray:
        """Test element-wise for finite values."""
        _ensure_jax()
        return _jnp.isfinite(x)

    # =========================================================================
    # Array Manipulation
    # =========================================================================

    def clip(self, x: jnp.ndarray, a_min: float, a_max: float) -> jnp.ndarray:
        """Clip array values to specified range."""
        _ensure_jax()
        return _jnp.clip(x, a_min, a_max)

    def stack(self, arrays: list[jnp.ndarray], axis: int = 0) -> jnp.ndarray:
        """Stack arrays along new axis."""
        _ensure_jax()
        return _jnp.stack(arrays, axis=axis)

    def concatenate(self, arrays: list[jnp.ndarray], axis: int = 0) -> jnp.ndarray:
        """Concatenate arrays along existing axis."""
        _ensure_jax()
        return _jnp.concatenate(arrays, axis=axis)

    def copy(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return copy of array (note: JAX arrays are immutable)."""
        _ensure_jax()
        return _jnp.array(x)

    def reshape(self, x: jnp.ndarray, shape: tuple[int, ...]) -> jnp.ndarray:
        """Reshape array to specified shape."""
        _ensure_jax()
        return _jnp.reshape(x, shape)

    def transpose(
        self, x: jnp.ndarray, axes: tuple[int, ...] | None = None
    ) -> jnp.ndarray:
        """Permute array dimensions."""
        _ensure_jax()
        return _jnp.transpose(x, axes=axes)

    def flatten(self, x: jnp.ndarray) -> jnp.ndarray:
        """Flatten array to 1D."""
        _ensure_jax()
        return x.flatten()

    # =========================================================================
    # Mathematical Functions
    # =========================================================================

    def exp(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise exponential."""
        _ensure_jax()
        return _jnp.exp(x)

    def log(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise natural logarithm."""
        _ensure_jax()
        return _jnp.log(x)

    def log10(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise base-10 logarithm."""
        _ensure_jax()
        return _jnp.log10(x)

    def sqrt(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise square root."""
        _ensure_jax()
        return _jnp.sqrt(x)

    def abs(self, x: jnp.ndarray) -> jnp.ndarray:
        """Element-wise absolute value."""
        _ensure_jax()
        return _jnp.abs(x)

    def power(self, x: jnp.ndarray, y: float | jnp.ndarray) -> jnp.ndarray:
        """Element-wise power."""
        _ensure_jax()
        return _jnp.power(x, y)

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def to_numpy(self, x: jnp.ndarray) -> np.ndarray:
        """Convert JAX array to NumPy ndarray."""
        _ensure_jax()
        return np.asarray(x)

    def from_numpy(self, x: np.ndarray) -> jnp.ndarray:
        """Convert NumPy ndarray to JAX array."""
        _ensure_jax()
        return _jnp.asarray(x)

    def astype(self, x: jnp.ndarray, dtype: Any) -> jnp.ndarray:
        """Cast array to specified dtype."""
        _ensure_jax()
        return x.astype(dtype)

    # =========================================================================
    # JIT Compilation
    # =========================================================================

    def jit(
        self,
        func: Callable,
        static_argnums: tuple[int, ...] | None = None,
    ) -> Callable:
        """JIT compile function.

        Parameters
        ----------
        func : callable
            Function to compile
        static_argnums : tuple of int, optional
            Argument indices that should be treated as static (not traced)

        Returns
        -------
        callable
            JIT-compiled function
        """
        _ensure_jax()
        return _jax.jit(func, static_argnums=static_argnums)

    # =========================================================================
    # Gradient Computation
    # =========================================================================

    def grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return gradient function.

        Parameters
        ----------
        func : callable
            Function to differentiate (must return scalar)
        argnums : int or tuple of int
            Arguments to differentiate with respect to

        Returns
        -------
        callable
            Function that computes gradients
        """
        _ensure_jax()
        return _jax.grad(func, argnums=argnums)

    def value_and_grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return function computing both value and gradient.

        Parameters
        ----------
        func : callable
            Function to differentiate (must return scalar)
        argnums : int or tuple of int
            Arguments to differentiate with respect to

        Returns
        -------
        callable
            Function returning (value, gradient) tuple
        """
        _ensure_jax()
        return _jax.value_and_grad(func, argnums=argnums)
