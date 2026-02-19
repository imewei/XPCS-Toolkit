"""Backend protocol interface for JAX/NumPy array operations.

This module defines the abstract interface that both NumPyBackend and
JAXBackend must implement, ensuring consistent API across backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

# Generic type for backend-specific arrays
ArrayType = TypeVar("ArrayType")


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol defining the backend interface for array operations.

    Both NumPyBackend and JAXBackend implement this protocol, providing
    a unified API for array computations that can run on CPU or GPU.

    Attributes
    ----------
    name : str
        Backend identifier ("numpy" or "jax")
    supports_gpu : bool
        Whether backend supports GPU computation
    supports_jit : bool
        Whether backend supports JIT compilation
    supports_grad : bool
        Whether backend supports automatic differentiation
    pi : float
        Mathematical constant π
    """

    @property
    def name(self) -> str:
        """Backend identifier ('numpy' or 'jax')."""
        ...

    @property
    def supports_gpu(self) -> bool:
        """Whether backend supports GPU computation."""
        ...

    @property
    def supports_jit(self) -> bool:
        """Whether backend supports JIT compilation."""
        ...

    @property
    def supports_grad(self) -> bool:
        """Whether backend supports automatic differentiation."""
        ...

    @property
    def pi(self) -> float:
        """Mathematical constant π."""
        ...

    # =========================================================================
    # Array Creation
    # =========================================================================

    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> ArrayType:  # type: ignore[type-var]
        """Create array filled with zeros."""
        ...

    def ones(self, shape: tuple[int, ...], dtype: Any = None) -> ArrayType:  # type: ignore[type-var]
        """Create array filled with ones."""
        ...

    def arange(
        self,
        start: float,
        stop: float | None = None,
        step: float = 1,
        dtype: Any = None,
    ) -> ArrayType:  # type: ignore[type-var]
        """Create array with evenly spaced values."""
        ...

    def linspace(self, start: float, stop: float, num: int) -> ArrayType:  # type: ignore[type-var]
        """Create array with linearly spaced values."""
        ...

    def logspace(self, start: float, stop: float, num: int) -> ArrayType:  # type: ignore[type-var]
        """Create array with logarithmically spaced values."""
        ...

    def meshgrid(self, *xi: ArrayType, indexing: str = "xy") -> tuple[ArrayType, ...]:
        """Create coordinate matrices from coordinate vectors."""
        ...

    def zeros_like(self, x: ArrayType, dtype: Any = None) -> ArrayType:
        """Create zero-filled array with same shape as input."""
        ...

    def ones_like(self, x: ArrayType, dtype: Any = None) -> ArrayType:
        """Create ones-filled array with same shape as input."""
        ...

    def full(
        self, shape: tuple[int, ...], fill_value: float, dtype: Any = None
    ) -> ArrayType:  # type: ignore[type-var]
        """Create array filled with specified value."""
        ...

    def array(self, data: Any, dtype: Any = None) -> ArrayType:  # type: ignore[type-var]
        """Create array from data."""
        ...

    # =========================================================================
    # Trigonometric Functions
    # =========================================================================

    def sin(self, x: ArrayType) -> ArrayType:
        """Element-wise sine."""
        ...

    def cos(self, x: ArrayType) -> ArrayType:
        """Element-wise cosine."""
        ...

    def arctan(self, x: ArrayType) -> ArrayType:
        """Element-wise arctangent."""
        ...

    def arctan2(self, y: ArrayType, x: ArrayType) -> ArrayType:
        """Element-wise arctangent of y/x, handling quadrants."""
        ...

    def hypot(self, x: ArrayType, y: ArrayType) -> ArrayType:
        """Element-wise sqrt(x^2 + y^2)."""
        ...

    def deg2rad(self, x: ArrayType) -> ArrayType:
        """Convert degrees to radians."""
        ...

    def rad2deg(self, x: ArrayType) -> ArrayType:
        """Convert radians to degrees."""
        ...

    def mod(self, x: ArrayType, y: ArrayType | float) -> ArrayType:
        """Element-wise modulo."""
        ...

    def floor(self, x: ArrayType) -> ArrayType:
        """Element-wise floor."""
        ...

    def ceil(self, x: ArrayType) -> ArrayType:
        """Element-wise ceiling."""
        ...

    def round(self, x: ArrayType, decimals: int = 0) -> ArrayType:
        """Round to given number of decimals."""
        ...

    # =========================================================================
    # Statistical Functions
    # =========================================================================

    def mean(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute mean along axis."""
        ...

    def std(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute standard deviation along axis."""
        ...

    def nanmean(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute mean, ignoring NaN values."""
        ...

    def nanmin(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute minimum, ignoring NaN values."""
        ...

    def nanmax(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute maximum, ignoring NaN values."""
        ...

    def percentile(self, x: ArrayType, q: float, axis: int | None = None) -> ArrayType:
        """Compute percentile along axis."""
        ...

    def sum(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute sum along axis."""
        ...

    def min(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute minimum along axis."""
        ...

    def max(self, x: ArrayType, axis: int | None = None) -> ArrayType:
        """Compute maximum along axis."""
        ...

    # =========================================================================
    # Binning Functions
    # =========================================================================

    def digitize(self, x: ArrayType, bins: ArrayType) -> ArrayType:
        """Return indices of bins to which each value belongs."""
        ...

    def bincount(
        self,
        x: ArrayType,
        weights: ArrayType | None = None,
        minlength: int = 0,
    ) -> ArrayType:
        """Count number of occurrences of each value."""
        ...

    def unique(
        self,
        x: ArrayType,
        return_inverse: bool = False,
        size: int | None = None,
    ) -> ArrayType | tuple[ArrayType, ...]:
        """Find unique elements of array."""
        ...

    # =========================================================================
    # Boolean/Masking Functions
    # =========================================================================

    def logical_and(self, x: ArrayType, y: ArrayType) -> ArrayType:
        """Element-wise logical AND."""
        ...

    def logical_or(self, x: ArrayType, y: ArrayType) -> ArrayType:
        """Element-wise logical OR."""
        ...

    def logical_not(self, x: ArrayType) -> ArrayType:
        """Element-wise logical NOT."""
        ...

    def where(self, condition: ArrayType, x: ArrayType, y: ArrayType) -> ArrayType:
        """Return elements chosen from x or y depending on condition."""
        ...

    def nonzero(self, x: ArrayType, size: int | None = None) -> tuple[ArrayType, ...]:
        """Return indices of non-zero elements."""
        ...

    def isnan(self, x: ArrayType) -> ArrayType:
        """Test element-wise for NaN."""
        ...

    def isfinite(self, x: ArrayType) -> ArrayType:
        """Test element-wise for finite values."""
        ...

    # =========================================================================
    # Array Manipulation
    # =========================================================================

    def clip(self, x: ArrayType, a_min: float, a_max: float) -> ArrayType:
        """Clip array values to specified range."""
        ...

    def stack(self, arrays: list[ArrayType], axis: int = 0) -> ArrayType:
        """Stack arrays along new axis."""
        ...

    def concatenate(self, arrays: list[ArrayType], axis: int = 0) -> ArrayType:
        """Concatenate arrays along existing axis."""
        ...

    def copy(self, x: ArrayType) -> ArrayType:
        """Return copy of array."""
        ...

    def reshape(self, x: ArrayType, shape: tuple[int, ...]) -> ArrayType:
        """Reshape array to specified shape."""
        ...

    def transpose(self, x: ArrayType, axes: tuple[int, ...] | None = None) -> ArrayType:
        """Permute array dimensions."""
        ...

    def flatten(self, x: ArrayType) -> ArrayType:
        """Flatten array to 1D."""
        ...

    # =========================================================================
    # Mathematical Functions
    # =========================================================================

    def exp(self, x: ArrayType) -> ArrayType:
        """Element-wise exponential."""
        ...

    def log(self, x: ArrayType) -> ArrayType:
        """Element-wise natural logarithm."""
        ...

    def log10(self, x: ArrayType) -> ArrayType:
        """Element-wise base-10 logarithm."""
        ...

    def sqrt(self, x: ArrayType) -> ArrayType:
        """Element-wise square root."""
        ...

    def abs(self, x: ArrayType) -> ArrayType:
        """Element-wise absolute value."""
        ...

    def power(self, x: ArrayType, y: float | ArrayType) -> ArrayType:
        """Element-wise power."""
        ...

    # =========================================================================
    # Type Conversion
    # =========================================================================

    def to_numpy(self, x: ArrayType) -> np.ndarray:
        """Convert array to NumPy ndarray."""
        ...

    def from_numpy(self, x: np.ndarray) -> ArrayType:  # type: ignore[type-var]
        """Convert NumPy ndarray to backend array."""
        ...

    def astype(self, x: ArrayType, dtype: Any) -> ArrayType:
        """Cast array to specified dtype."""
        ...

    # =========================================================================
    # JIT Compilation
    # =========================================================================

    def jit(
        self,
        func: Callable,
        static_argnums: tuple[int, ...] | None = None,
    ) -> Callable:
        """JIT compile function (no-op for NumPy)."""
        ...

    # =========================================================================
    # Gradient Computation (JAX only)
    # =========================================================================

    def grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return gradient function (raises for NumPy)."""
        ...

    def value_and_grad(
        self,
        func: Callable,
        argnums: int | tuple[int, ...] = 0,
    ) -> Callable:
        """Return function computing both value and gradient."""
        ...

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def vmap(
        self,
        func: Callable,
        in_axes: int | tuple[int | None, ...] = 0,
        out_axes: int = 0,
    ) -> Callable:
        """Vectorize function over batch dimension."""
        ...

    def scan(
        self,
        func: Callable,
        init: ArrayType,
        xs: ArrayType,
        length: int | None = None,
    ) -> tuple[ArrayType, ArrayType]:
        """Scan over leading array dimension while carrying along state."""
        ...

    def fori_loop(
        self,
        lower: int,
        upper: int,
        body_fun: Callable,
        init_val: ArrayType,
    ) -> ArrayType:
        """Execute body function in a loop from lower to upper."""
        ...
