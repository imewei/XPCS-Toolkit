"""Two-time correlation analysis module.

Provides two-time correlation map visualization and analysis for studying
temporal dynamics beyond traditional multi-tau analysis.

Functions:
    get_twotime_data: Extract two-time correlation matrices
    plot_twotime: Generate two-time correlation maps
"""

import numpy as np
import pyqtgraph as pg

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.utils.logging_config import get_logger

from ..xpcs_file import MemoryMonitor

logger = get_logger(__name__)

PG_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def clean_c2_for_visualization(c2, method="nan_to_num"):
    """
    Clean C2 data to remove NaN/inf values that cause white lines in visualization.

    Args:
        c2: C2 correlation matrix
        method: Cleaning method ('nan_to_num', 'interpolate', 'median')

    Returns:
        Cleaned C2 matrix with NaN/inf values replaced
    """
    if not np.any(np.isnan(c2)) and not np.any(np.isinf(c2)):
        # No cleaning needed
        return c2

    nan_count = np.sum(np.isnan(c2))
    inf_count = np.sum(np.isinf(c2))

    if nan_count > 0 or inf_count > 0:
        logger.debug(f"Cleaning C2 matrix: {nan_count} NaN and {inf_count} inf values")

    if method == "nan_to_num":
        # Replace NaN/inf with numeric values based on finite data statistics.
        # Use a single np.nanpercentile call with three quantiles to avoid two
        # separate array sorts + an intermediate finite_values copy.
        # np.nanpercentile ignores NaN/inf natively; no boolean masking needed.
        finite_vals = c2[np.isfinite(c2)]
        has_finite = finite_vals.size > 0
        if has_finite:
            # Single pass over finite-only values: [0.1th, 50th, 99.9th] percentiles.
            # np.nanpercentile cannot handle inf (produces nan from inf-inf),
            # so we pre-filter to finite values only.
            neg_replacement, nan_replacement, pos_replacement = np.percentile(
                finite_vals, [0.1, 50.0, 99.9]
            )
        else:
            # Fallback if all values are non-finite
            pos_replacement, neg_replacement, nan_replacement = 1.0, 0.0, 0.5

        return np.nan_to_num(
            c2, nan=nan_replacement, posinf=pos_replacement, neginf=neg_replacement
        )

    if method == "interpolate":
        # Use interpolation for better continuity
        try:
            from xpcsviewer.backends.scipy_replacements import gaussian_filter

            c2_clean = np.copy(c2)
            finite_mask = np.isfinite(c2)

            if np.any(finite_mask):
                # Replace non-finite values with median, then apply Gaussian smoothing
                median_val = np.median(c2[finite_mask])
                c2_clean[~finite_mask] = median_val

                # Apply light Gaussian filter to smooth transitions (backend-agnostic)
                # gaussian_filter already returns numpy at its API boundary
                c2_clean = gaussian_filter(c2_clean, sigma=0.5, mode="nearest")

                # Preserve original finite values
                c2_clean[finite_mask] = c2[finite_mask]

            return c2_clean
        except ImportError:
            logger.warning(
                "gaussian_filter not available for interpolation, falling back to nan_to_num"
            )
            return clean_c2_for_visualization(c2, method="nan_to_num")

    elif method == "median":
        # Replace with median of valid data (simple but effective)
        finite_mask = np.isfinite(c2)
        if np.any(finite_mask):
            valid_median = np.median(c2[finite_mask])
            return np.where(finite_mask, c2, valid_median)
        return np.zeros_like(c2)

    # Fallback to original data if method not recognized
    logger.warning(f"Unknown cleaning method '{method}', returning original data")
    return c2


def calculate_safe_levels(c2):
    """
    Calculate vmin/vmax levels while safely handling NaN/inf values.

    Args:
        c2: C2 correlation matrix

    Returns:
        Tuple of (vmin, vmax) computed from finite values only
    """
    # Only use finite values for percentile calculation
    finite_mask = np.isfinite(c2)

    if np.any(finite_mask):
        finite_values = c2[finite_mask]
        if len(finite_values) > 0:
            vmin, vmax = np.percentile(finite_values, [0.5, 99.5])
            # Ensure valid range
            if vmin >= vmax:
                vmax = vmin + 1e-6
        else:
            vmin, vmax = 0.0, 1.0
    else:
        # Fallback if all values are NaN/inf
        logger.warning("All C2 values are non-finite, using default levels")
        vmin, vmax = 0.0, 1.0

    logger.debug(f"C2 display levels: vmin={vmin:.3e}, vmax={vmax:.3e}")
    return vmin, vmax


def plot_twotime(
    xfile,
    hdl,
    scale="log",
    auto_crop=True,
    highlight_xy=None,
    cmap="jet",
    vmin=None,
    vmax=None,
    autolevel=True,
    correct_diag=False,
    selection=0,
):
    """Render a two-time correlation map with associated SAXS and G2 panels.

    Displays the C2 two-time correlation matrix alongside the dynamic Q-map,
    SAXS pattern, and extracted G2 correlation curves (full and partial).

    Args:
        xfile: XpcsFile object with Twotime analysis data.
        hdl: Dictionary of PyQtGraph ImageView/PlotItem handles with keys
            ``'saxs'``, ``'dqmap'``, ``'tt'``, ``'c2g2'``.
        scale: Intensity scale for the SAXS/dqmap display: ``'log'`` or
            ``'linear'``.
        auto_crop: If True, automatically crop the Q-map display.
        highlight_xy: Optional (x, y) pixel coordinate to highlight on maps.
        cmap: Matplotlib colormap name for the C2 matrix (default ``'jet'``).
        vmin: Minimum color level for C2 display. Ignored when autolevel=True.
        vmax: Maximum color level for C2 display. Ignored when autolevel=True.
        autolevel: If True, auto-compute color levels from the data.
        correct_diag: If True, apply diagonal correction to C2 matrix.
        selection: Q-bin index to display (default 0).

    Raises:
        AssertionError: If ``xfile`` does not contain Twotime analysis.

    Example:
        >>> plot_twotime(xfile, hdl, cmap='viridis', autolevel=True)
    """
    assert "Twotime" in xfile.atype, "Not a twotime file"

    # Monitor memory before processing large twotime data
    memory_mb, _ = MemoryMonitor.get_memory_usage()
    logger.debug(
        f"Plotting twotime data for {xfile.label}, memory usage: {memory_mb:.1f}MB"
    )

    # display dqmap and saxs
    dqmap_disp, saxs, selection_xy = xfile.get_twotime_maps(
        scale=scale,
        auto_crop=auto_crop,
        highlight_xy=highlight_xy,
        selection=selection,
    )

    if selection_xy is not None:
        selection = selection_xy

    # Check if saxs data is empty before setting image
    if saxs.size == 0:
        logger.warning(
            f"SAXS data is empty for {xfile.label}, cannot display twotime plot"
        )
        return

    # Check if dqmap_disp data is empty before setting image
    if dqmap_disp.size == 0:
        logger.warning(
            f"DQMAP data is empty for {xfile.label}, cannot display twotime plot"
        )
        return

    # Ensure NumPy arrays at PyQtGraph boundary
    hdl["saxs"].setImage(ensure_numpy(np.flipud(saxs)))
    hdl["dqmap"].setImage(ensure_numpy(dqmap_disp))

    # Monitor memory before loading potentially large c2 data
    if MemoryMonitor.is_memory_pressure_high(0.8):
        logger.warning(
            f"High memory pressure detected before loading c2 data for {xfile.label}"
        )

    c2_result = xfile.get_twotime_c2(selection=selection, correct_diag=correct_diag)
    if c2_result is None:
        return

    c2, delta_t = c2_result["c2_mat"], c2_result["delta_t"]

    # Log memory usage after loading c2 data
    c2_memory_mb = MemoryMonitor.estimate_array_memory(c2.shape, c2.dtype)
    logger.debug(f"Loaded c2 data ({c2.shape}), estimated size: {c2_memory_mb:.1f}MB")

    # Check if c2 data is empty or has zero size
    if c2.size == 0:
        logger.warning(f"C2 data is empty for {xfile.label}, selection={selection}")
        return

    # ✅ FIX: Clean C2 data to remove NaN/inf values that cause white lines
    c2_clean = clean_c2_for_visualization(c2, method="nan_to_num")

    # Additional check for empty array after cleaning
    if c2_clean.size == 0:
        logger.warning(f"C2 data became empty after cleaning for {xfile.label}")
        return

    hdl["tt"].imageItem.setScale(delta_t)
    hdl["tt"].setImage(ensure_numpy(c2_clean), autoRange=True)  # Use cleaned data

    cmap = pg.colormap.getFromMatplotlib(cmap)
    hdl["tt"].setColorMap(cmap)
    hdl["tt"].ui.histogram.setHistogramRange(mn=0, mx=3)
    if not autolevel and vmin is not None and vmax is not None:
        hdl["tt"].setLevels(min=vmin, max=vmax)
    else:
        # ✅ FIX: Use safe level calculation that handles NaN/inf values
        vmin, vmax = calculate_safe_levels(c2_clean)
        hdl["tt"].setLevels(min=vmin, max=vmax)
    plot_twotime_g2(hdl, c2_result)


def plot_twotime_g2(hdl, c2_result):
    """Plot G2 correlation curves extracted from the two-time correlation matrix.

    Displays both the full G2 (averaged over all time sections) and partial
    G2 curves (individual time sections) on a log-x axis.

    Args:
        hdl: Dictionary of PyQtGraph handles; uses ``hdl['c2g2']`` PlotItem.
        c2_result: Dictionary from ``XpcsFile.get_twotime_c2()`` with keys
            ``'g2_full'``, ``'g2_partial'``, and ``'acquire_period'``.
    """
    g2_full, g2_partial = c2_result["g2_full"], c2_result["g2_partial"]

    hdl["c2g2"].clear()
    hdl["c2g2"].setLabel("left", "g2")
    hdl["c2g2"].setLabel("bottom", "t (s)")
    acquire_period = c2_result["acquire_period"]

    # ✅ FIX: Clean G2 data to handle NaN values in time series plots
    g2_full_clean = np.nan_to_num(g2_full, nan=1.0, posinf=1.0, neginf=0.0)
    g2_partial_clean = np.nan_to_num(g2_partial, nan=1.0, posinf=1.0, neginf=0.0)

    xaxis = np.arange(g2_full_clean.size) * acquire_period
    hdl["c2g2"].plot(
        x=xaxis[1:],
        y=g2_full_clean[1:],
        pen=pg.mkPen(color=PG_COLORS[-1], width=4),
        name="g2_full",
    )
    for n in range(g2_partial_clean.shape[0]):
        xaxis = np.arange(g2_partial_clean.shape[1]) * acquire_period
        hdl["c2g2"].plot(
            x=xaxis[1:],
            y=g2_partial_clean[n][1:],
            pen=pg.mkPen(color=PG_COLORS[n], width=1),
            name=f"g2_partial_{n}",
        )
    hdl["c2g2"].setLogMode(x=True, y=False)
    hdl["c2g2"].autoRange()
