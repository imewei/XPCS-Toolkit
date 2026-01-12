"""JAX replacements for scipy.ndimage functions.

This module provides JAX-compatible implementations of scipy.ndimage
functions used in SimpleMask for smoothing and filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from xpcsviewer.backends import ensure_numpy


def gaussian_filter(
    input_array: ArrayLike,
    sigma: float | tuple[float, ...],
    mode: str = "reflect",
    truncate: float = 4.0,
) -> np.ndarray:
    """Apply Gaussian filter to array.

    This is a JAX-compatible implementation that uses convolution with
    a Gaussian kernel. Falls back to scipy.ndimage.gaussian_filter when
    JAX is not available.

    Parameters
    ----------
    input_array : array-like
        Input array to filter
    sigma : float or tuple of floats
        Standard deviation for Gaussian kernel. Can be a single value
        for isotropic filtering or a tuple for anisotropic filtering.
    mode : str
        Boundary mode: 'reflect', 'constant', 'nearest', 'wrap'
        (default: 'reflect')
    truncate : float
        Truncate filter at this many standard deviations (default: 4.0)

    Returns
    -------
    ndarray
        Filtered array
    """
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        return _gaussian_filter_jax(input_array, sigma, mode, truncate)
    except ImportError:
        # Fall back to scipy
        from scipy.ndimage import gaussian_filter as scipy_gaussian

        return scipy_gaussian(input_array, sigma, mode=mode, truncate=truncate)


def _gaussian_filter_jax(
    input_array: ArrayLike,
    sigma: float | tuple[float, ...],
    mode: str = "reflect",
    truncate: float = 4.0,
) -> np.ndarray:
    """JAX implementation of Gaussian filter using separable convolution."""
    import jax.numpy as jnp
    from jax import lax

    arr = jnp.asarray(input_array)
    ndim = arr.ndim

    # Normalize sigma to tuple
    if isinstance(sigma, (int, float)):
        sigma = (float(sigma),) * ndim
    else:
        sigma = tuple(float(s) for s in sigma)

    if len(sigma) != ndim:
        raise ValueError(f"sigma must have {ndim} elements, got {len(sigma)}")

    # Map mode to JAX padding mode
    padding_modes = {
        "reflect": "REFLECT",
        "constant": "CONSTANT",
        "nearest": "REPLICATE",
        "wrap": "CIRCULAR",
    }
    if mode not in padding_modes:
        raise ValueError(f"Unsupported mode: {mode}")
    pad_mode = padding_modes[mode]

    # Apply 1D Gaussian filter along each axis (separable)
    result = arr
    for axis in range(ndim):
        if sigma[axis] > 0:
            result = _gaussian_filter_1d_jax(
                result, sigma[axis], axis, pad_mode, truncate
            )

    from xpcsviewer.backends import ensure_numpy

    return ensure_numpy(result)


def _gaussian_filter_1d_jax(
    arr,
    sigma: float,
    axis: int,
    pad_mode: str,
    truncate: float,
):
    """Apply 1D Gaussian filter along specified axis using JAX."""
    import jax.numpy as jnp
    from jax import lax

    # Compute kernel size
    radius = int(truncate * sigma + 0.5)
    if radius == 0:
        return arr

    # Create 1D Gaussian kernel
    x = jnp.arange(-radius, radius + 1, dtype=arr.dtype)
    kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / jnp.sum(kernel)

    # Reshape kernel for the target axis
    shape = [1] * arr.ndim
    shape[axis] = len(kernel)
    kernel = kernel.reshape(shape)

    # Pad array
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)

    if pad_mode == "CONSTANT":
        padded = jnp.pad(arr, pad_width, mode="constant", constant_values=0)
    elif pad_mode == "REFLECT":
        padded = jnp.pad(arr, pad_width, mode="reflect")
    elif pad_mode == "REPLICATE":
        padded = jnp.pad(arr, pad_width, mode="edge")
    elif pad_mode == "CIRCULAR":
        padded = jnp.pad(arr, pad_width, mode="wrap")
    else:
        padded = jnp.pad(arr, pad_width, mode="constant", constant_values=0)

    # Convolve
    # For 2D case, we use a simple approach with sum over the kernel window
    # This is less efficient than lax.conv but more general for N-D
    result = _convolve_1d(padded, kernel.flatten(), axis)

    return result


def _convolve_1d(arr, kernel, axis: int):
    """1D convolution along specified axis using JAX."""
    import jax.numpy as jnp
    from jax import lax

    kernel_size = len(kernel)
    ndim = arr.ndim

    # For 2D arrays, use lax.conv_general_dilated
    if ndim == 2:
        # Add batch and channel dimensions for lax.conv
        arr_4d = arr[jnp.newaxis, jnp.newaxis, :, :]

        if axis == 0:
            kernel_4d = kernel.reshape(1, 1, kernel_size, 1)
        else:
            kernel_4d = kernel.reshape(1, 1, 1, kernel_size)

        result = lax.conv_general_dilated(
            arr_4d,
            kernel_4d,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )
        return result[0, 0]

    # Fallback for other dimensions: use jnp.convolve approach
    # Move axis to last position, convolve, move back
    arr_moved = jnp.moveaxis(arr, axis, -1)
    original_shape = arr_moved.shape

    # Flatten to 2D for vectorized convolution
    flat = arr_moved.reshape(-1, arr_moved.shape[-1])

    def convolve_row(row):
        return jnp.convolve(row, kernel, mode="valid")

    # Use vmap for vectorized convolution
    from jax import vmap

    result_flat = vmap(convolve_row)(flat)
    result_shape = original_shape[:-1] + (result_flat.shape[-1],)
    result = result_flat.reshape(result_shape)

    return jnp.moveaxis(result, -1, axis)


def uniform_filter(
    input_array: ArrayLike,
    size: int | tuple[int, ...] = 3,
    mode: str = "reflect",
) -> np.ndarray:
    """Apply uniform (box) filter to array.

    Parameters
    ----------
    input_array : array-like
        Input array to filter
    size : int or tuple of ints
        Filter size along each axis
    mode : str
        Boundary mode (default: 'reflect')

    Returns
    -------
    ndarray
        Filtered array
    """
    try:
        import jax.numpy as jnp

        arr = jnp.asarray(input_array)
        ndim = arr.ndim

        if isinstance(size, int):
            size = (size,) * ndim

        # Use Gaussian filter with sigma that approximates uniform
        # sigma ≈ size / sqrt(12) for uniform distribution
        sigma = tuple(s / (12**0.5) for s in size)
        return gaussian_filter(arr, sigma, mode=mode)
    except ImportError:
        from scipy.ndimage import uniform_filter as scipy_uniform

        return scipy_uniform(input_array, size=size, mode=mode)
