"""
Helper utilities for XPCS data processing.

Provides common data transformation functions used across analysis modules:

- get_min_max: Percentile-based intensity range calculation
- norm_saxs_data: SAXS intensity normalization (I, I*q^2, I*q^4)
- create_slice: Slice creation for array subsetting
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_min_max(
    data: ArrayLike,
    min_percent: float = 0,
    max_percent: float = 100,
    **kwargs: Any,
) -> tuple[float, float]:
    """Calculate intensity min/max values using percentiles.

    Args:
        data: Input array.
        min_percent: Lower percentile (0-100).
        max_percent: Upper percentile (0-100).
        **kwargs: Optional plot_norm and plot_type for symmetric scaling.

    Returns:
        Tuple of (vmin, vmax) values.
    """
    logger.debug(
        f"Calculating min/max: min_percent={min_percent}, max_percent={max_percent}"
    )
    arr = np.asarray(data)
    vmin = float(np.percentile(arr.ravel(), min_percent))
    vmax = float(np.percentile(arr.ravel(), max_percent))
    logger.debug(f"Percentile values: vmin={vmin}, vmax={vmax}")
    if "plot_norm" in kwargs and "plot_type" in kwargs and kwargs["plot_norm"] == 3:
        if kwargs["plot_type"] == "log":
            t = max(abs(vmin), abs(vmax))
            vmin, vmax = -t, t
        else:
            t = max(abs(1 - vmin), abs(vmax - 1))
            vmin, vmax = 1 - t, 1 + t

    return vmin, vmax


def norm_saxs_data(
    Iq: NDArray[np.floating[Any]],
    q: NDArray[np.floating[Any]],
    plot_norm: int = 0,
) -> tuple[NDArray[np.floating[Any]], str, str]:
    """Normalize SAXS intensity data.

    Args:
        Iq: Intensity array.
        q: Q-values array.
        plot_norm: Normalization mode (0=none, 1=I*q^2, 2=I*q^4, 3=I/I_0).

    Returns:
        Tuple of (normalized_Iq, xlabel, ylabel).
    """
    logger.debug(f"Normalizing SAXS data with plot_norm={plot_norm}")
    ylabel = "Intensity"
    if plot_norm == 1:
        Iq = Iq * np.square(q)
        ylabel = ylabel + " * q^2"
    elif plot_norm == 2:
        Iq = Iq * np.square(np.square(q))
        ylabel = ylabel + " * q^4"
    elif plot_norm == 3:
        baseline = Iq[0]
        Iq = Iq / baseline
        ylabel = ylabel + " / I_0"

    xlabel = "$q (\\AA^{-1})$"
    return Iq, xlabel, ylabel


def create_slice(
    arr: NDArray[np.floating[Any]],
    x_range: tuple[float, float],
) -> slice:
    """Create a slice for array subsetting based on value range.

    Args:
        arr: 1D sorted array of values.
        x_range: Tuple of (min_value, max_value) defining the range.

    Returns:
        Slice object for the array subset.
    """
    logger.debug(f"Creating slice for range {x_range} on array of size {arr.size}")
    start, end = 0, arr.size - 1
    while arr[start] < x_range[0]:
        start += 1
        if start == arr.size:
            break

    while arr[end] >= x_range[1]:
        end -= 1
        if end == 0:
            break

    return slice(start, end + 1)
