"""JAX replacements for SciPy functions used in the SimpleMask module.

Provides JAX-compatible implementations backed by interpax:

Available modules:
    ndimage: gaussian_filter, gaussian_filter1d, zoom, etc.
    interpolate: interp1d, interp2d_jax, etc.
"""

from __future__ import annotations

from xpcsviewer.backends.scipy_replacements import interpolate, ndimage
from xpcsviewer.backends.scipy_replacements.interpolate import (
    Interp1d,
    interp1d,
    interp2d_jax,
)
from xpcsviewer.backends.scipy_replacements.ndimage import (
    gaussian_filter,
    gaussian_filter1d,
    zoom,
)

__all__ = [
    # Modules
    "ndimage",
    "interpolate",
    # Interpolation functions
    "Interp1d",
    "interp1d",
    "interp2d_jax",
    # ndimage functions
    "gaussian_filter",
    "gaussian_filter1d",
    "zoom",
]
