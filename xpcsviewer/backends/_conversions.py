"""Array conversion utilities for I/O boundaries.

This module provides functions for converting between NumPy arrays and
backend-specific arrays at I/O boundaries (file I/O, visualization, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._base import BackendProtocol


def ensure_numpy(array: ArrayLike) -> np.ndarray:
    """Convert any array type to NumPy for I/O boundaries.

    Use this function at boundaries where NumPy arrays are required:
    - HDF5 file I/O (h5py)
    - PyQtGraph visualization
    - Matplotlib plotting
    - Pandas DataFrame operations

    Parameters
    ----------
    array : array-like
        JAX array, NumPy array, list, or any array-like object

    Returns
    -------
    np.ndarray
        NumPy array

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> x = jnp.array([1, 2, 3])
    >>> np_x = ensure_numpy(x)
    >>> isinstance(np_x, np.ndarray)
    True

    >>> # Works with lists too
    >>> ensure_numpy([1, 2, 3])
    array([1, 2, 3])
    """
    # Fast path for NumPy arrays - ensure it's writable
    if isinstance(array, np.ndarray):
        if not array.flags.writeable:
            return np.array(array)  # Force copy for read-only arrays
        return array

    # Check for JAX arrays - must copy to ensure writeable numpy array
    try:
        import jax.numpy as jnp

        if isinstance(array, jnp.ndarray):
            return np.array(array)  # Use np.array to ensure copy
    except ImportError:
        pass

    # Check for arrays with __array__ method (covers most array-like objects)
    if hasattr(array, "__array__"):
        result = np.asarray(array)
        # Ensure result is writable
        if not result.flags.writeable:
            return np.array(result)
        return result

    # Final fallback: convert via np.array
    return np.array(array)


def ensure_backend_array(
    array: ArrayLike, backend: BackendProtocol | None = None
) -> Any:
    """Convert array to the backend's array type.

    Use this function when receiving external data that needs to be
    converted to the current backend's array format.

    Parameters
    ----------
    array : array-like
        NumPy array, JAX array, list, or any array-like object
    backend : BackendProtocol, optional
        Backend to convert to. If None, uses the current backend.

    Returns
    -------
    ArrayType
        Array in the backend's native format

    Examples
    --------
    >>> from xpcsviewer.backends import get_backend, ensure_backend_array
    >>> import numpy as np
    >>> x = np.array([1, 2, 3])
    >>> backend = get_backend()
    >>> bx = ensure_backend_array(x, backend)
    """
    if backend is None:
        from . import get_backend

        backend = get_backend()

    return backend.from_numpy(ensure_numpy(array))


def is_jax_array(array: Any) -> bool:
    """Check if array is a JAX array.

    Parameters
    ----------
    array : Any
        Object to check

    Returns
    -------
    bool
        True if array is a JAX array
    """
    try:
        import jax.numpy as jnp

        return isinstance(array, jnp.ndarray)
    except ImportError:
        return False


def is_numpy_array(array: Any) -> bool:
    """Check if array is a NumPy array.

    Parameters
    ----------
    array : Any
        Object to check

    Returns
    -------
    bool
        True if array is a NumPy array
    """
    return isinstance(array, np.ndarray)


def get_array_backend(array: Any) -> str:
    """Determine which backend an array belongs to.

    Parameters
    ----------
    array : Any
        Array to check

    Returns
    -------
    str
        'numpy', 'jax', or 'unknown'
    """
    if is_numpy_array(array):
        return "numpy"
    if is_jax_array(array):
        return "jax"
    return "unknown"


def arrays_compatible(a: Any, b: Any) -> bool:
    """Check if two arrays are from the same backend.

    Parameters
    ----------
    a, b : Any
        Arrays to compare

    Returns
    -------
    bool
        True if both arrays are from the same backend
    """
    return get_array_backend(a) == get_array_backend(b)
