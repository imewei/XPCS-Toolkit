"""JAX replacements for scipy.interpolate functions using interpax.

This module provides JAX-compatible implementations of scipy.interpolate
functions used in SimpleMask for interpolation operations, using the
interpax library for GPU-accelerated interpolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

# Supported interpolation methods
InterpolationKind = Literal["linear", "nearest", "cubic", "quadratic"]


class Interp1d:
    """1D interpolation class compatible with scipy.interpolate.interp1d.

    This class provides JAX-compatible 1D interpolation using interpax.
    It stores the interpolation data and provides a callable interface.

    Parameters
    ----------
    x : array-like
        1D array of x coordinates (must be monotonically increasing)
    y : array-like
        1D or ND array of y values. If ND, interpolation is performed
        along the last axis.
    kind : str
        Interpolation method: 'linear', 'nearest', 'quadratic', 'cubic'.
        Default is 'linear'.
    bounds_error : bool
        If True, raise ValueError for out-of-bounds values.
        If False, use fill_value for out-of-bounds. Default: True.
    fill_value : float or tuple
        Value for out-of-bounds points. Can be 'extrapolate' for
        linear extrapolation, or a tuple (below, above) for different
        values below and above the range.

    Examples
    --------
    >>> x = np.array([0, 1, 2, 3])
    >>> y = np.array([0, 1, 4, 9])
    >>> f = Interp1d(x, y, kind='linear')
    >>> f(1.5)
    2.5
    """

    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        kind: str = "linear",
        bounds_error: bool = True,
        fill_value: float | tuple | str = np.nan,
    ):
        """Initialize interpolator."""
        self._use_interpax = False
        self._use_jax = False

        try:
            import interpax
            import jax.numpy as jnp

            self._x = jnp.asarray(x)
            self._y = jnp.asarray(y)
            self._use_interpax = True
            self._use_jax = True
            self._jnp = jnp
            self._interpax = interpax
        except ImportError:
            self._x = np.asarray(x)  # type: ignore
            self._y = np.asarray(y)  # type: ignore

        self._kind = kind
        self._bounds_error = bounds_error
        self._fill_value = fill_value
        self._extrapolate = fill_value == "extrapolate"

        # Validate inputs
        if self._x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if len(self._x) < 2:
            raise ValueError("x must have at least 2 elements")

        # Check monotonicity
        xp = np if not self._use_jax else self._jnp
        if not xp.all(xp.diff(self._x) > 0):
            raise ValueError("x must be strictly increasing")

    def __call__(self, x_new: ArrayLike) -> np.ndarray:
        """Evaluate interpolation at new x values.

        Parameters
        ----------
        x_new : array-like
            New x values at which to interpolate

        Returns
        -------
        ndarray
            Interpolated y values
        """
        if self._use_interpax:
            return self._interp_interpax(x_new)
        return self._interp_numpy(x_new)

    def _interp_interpax(self, x_new: ArrayLike) -> np.ndarray:
        """Interpolation using interpax library."""
        jnp = self._jnp
        interpax = self._interpax
        x_new_arr = jnp.asarray(x_new)
        x_new_shape = x_new_arr.shape
        x_new_flat = x_new_arr.flatten()

        x_min, x_max = self._x[0], self._x[-1]

        # Handle bounds error check
        if self._bounds_error:
            out_of_bounds = jnp.logical_or(x_new_flat < x_min, x_new_flat > x_max)
            if jnp.any(out_of_bounds):
                raise ValueError("x_new values out of interpolation range")

        # Map kind to interpax method
        # interpax supports: "nearest", "linear", "cubic", "cubic2", "cardinal", "catmull-rom"
        method_map = {
            "linear": "linear",
            "nearest": "nearest",
            "cubic": "cubic",
            "quadratic": "cubic",  # Use cubic as approximation
            "slinear": "linear",
            "zero": "nearest",
        }
        method = method_map.get(self._kind, "linear")

        # Determine extrapolation behavior
        if self._extrapolate:
            extrap = True  # interpax will extrapolate
        elif self._bounds_error:
            extrap = False
        else:
            extrap = False  # We'll handle fill values manually

        # Use interpax.interp1d for 1D interpolation
        result = interpax.interp1d(
            x_new_flat,
            self._x,
            self._y,
            method=method,
            extrap=extrap,
        )

        # Handle fill values for out-of-bounds when not extrapolating
        if not self._bounds_error and not self._extrapolate:
            below = x_new_flat < x_min
            above = x_new_flat > x_max

            if isinstance(self._fill_value, tuple):
                fill_below, fill_above = self._fill_value
            else:
                fill_below = fill_above = self._fill_value

            result = jnp.where(below, fill_below, result)
            result = jnp.where(above, fill_above, result)

        return result.reshape(x_new_shape)

    def _interp_numpy(self, x_new: ArrayLike) -> np.ndarray:
        """NumPy/SciPy fallback implementation."""
        from scipy.interpolate import interp1d as scipy_interp1d

        f = scipy_interp1d(
            self._x,
            self._y,
            kind=self._kind,
            bounds_error=self._bounds_error,
            fill_value=self._fill_value if not self._extrapolate else "extrapolate",
        )
        return f(x_new)


def interp1d(
    x: ArrayLike,
    y: ArrayLike,
    kind: str = "linear",
    bounds_error: bool = True,
    fill_value: float | tuple | str = np.nan,
) -> Interp1d:
    """Create 1D interpolation function.

    Factory function that returns an Interp1d instance using interpax
    when JAX is available.

    Parameters
    ----------
    x : array-like
        1D array of x coordinates (must be monotonically increasing)
    y : array-like
        Array of y values
    kind : str
        Interpolation method: 'linear', 'nearest', 'cubic', etc.
    bounds_error : bool
        If True, raise ValueError for out-of-bounds values
    fill_value : float or tuple or 'extrapolate'
        Value for out-of-bounds points

    Returns
    -------
    Interp1d
        Callable interpolation function
    """
    return Interp1d(x, y, kind=kind, bounds_error=bounds_error, fill_value=fill_value)


def interp2d_jax(
    xq: ArrayLike,
    yq: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    f: ArrayLike,
    method: str = "linear",
    extrap: bool = False,
    fill_value: float = np.nan,
) -> np.ndarray:
    """2D interpolation using interpax.

    Parameters
    ----------
    xq, yq : array-like
        Query points for interpolation
    x, y : array-like
        Original grid coordinates (1D arrays)
    f : array-like
        Values on original grid (2D array)
    method : str
        Interpolation method: 'linear', 'cubic', etc.
    extrap : bool
        Whether to extrapolate outside bounds
    fill_value : float
        Value for out-of-bounds points when extrap=False

    Returns
    -------
    ndarray
        Interpolated values at query points
    """
    try:
        import interpax
        import jax.numpy as jnp

        xq = jnp.asarray(xq)
        yq = jnp.asarray(yq)
        x = jnp.asarray(x)
        y = jnp.asarray(y)
        f = jnp.asarray(f)

        result = interpax.interp2d(xq, yq, x, y, f, method=method, extrap=extrap)

        if not extrap:
            # Apply fill value for out-of-bounds
            x_oob = jnp.logical_or(xq < x[0], xq > x[-1])
            y_oob = jnp.logical_or(yq < y[0], yq > y[-1])
            oob = jnp.logical_or(x_oob, y_oob)
            result = jnp.where(oob, fill_value, result)

        return result
    except ImportError:
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (np.asarray(x), np.asarray(y)),
            np.asarray(f),
            method=method if method in ("linear", "nearest") else "linear",
            bounds_error=False,
            fill_value=fill_value,
        )
        points = np.stack([np.asarray(xq).ravel(), np.asarray(yq).ravel()], axis=-1)
        return interp(points).reshape(np.asarray(xq).shape)
