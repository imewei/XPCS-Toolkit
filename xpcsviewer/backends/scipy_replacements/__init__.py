"""JAX replacements for SciPy functions using interpax, optimistix, and optax.

This module provides JAX-compatible implementations of SciPy functions
that are used in the SimpleMask module:
- interpax for interpolation
- optimistix for optimization and root-finding
- optax for gradient-based optimization

Available modules:
    ndimage: gaussian_filter, binary_dilation, etc.
    interpolate: interp1d, interp2d_jax, etc.
    optimize: minimize, curve_fit, least_squares, root
"""

from __future__ import annotations

from xpcsviewer.backends.scipy_replacements.interpolate import (
    Interp1d,
    interp1d,
    interp2d_jax,
)
from xpcsviewer.backends.scipy_replacements.optimize import (
    OptimizeResult,
    curve_fit,
    least_squares,
    minimize,
    root,
)

__all__ = [
    # Modules
    "ndimage",
    "interpolate",
    "optimize",
    # Interpolation functions
    "Interp1d",
    "interp1d",
    "interp2d_jax",
    # Optimization functions
    "OptimizeResult",
    "minimize",
    "curve_fit",
    "least_squares",
    "root",
]
