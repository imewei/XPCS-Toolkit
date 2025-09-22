import numpy as np
import pyqtgraph as pg

from xpcs_toolkit.utils.logging_config import get_logger

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
        # Replace NaN/inf with numeric values based on finite data statistics
        finite_mask = np.isfinite(c2)
        if np.any(finite_mask):
            finite_values = c2[finite_mask]
            # Use percentiles of finite data for inf replacement
            pos_replacement = np.percentile(finite_values, 99.9)
            neg_replacement = np.percentile(finite_values, 0.1)
            # Use median for NaN replacement to avoid bias
            nan_replacement = np.median(finite_values)
        else:
            # Fallback if all values are non-finite
            pos_replacement, neg_replacement, nan_replacement = 1.0, 0.0, 0.5

        return np.nan_to_num(
            c2, nan=nan_replacement, posinf=pos_replacement, neginf=neg_replacement
        )

    if method == "interpolate":
        # Use interpolation for better continuity
        try:
            from scipy import ndimage

            c2_clean = np.copy(c2)
            finite_mask = np.isfinite(c2)

            if np.any(finite_mask):
                # Replace non-finite values with median, then apply Gaussian smoothing
                median_val = np.median(c2[finite_mask])
                c2_clean[~finite_mask] = median_val

                # Apply light Gaussian filter to smooth transitions
                c2_clean = ndimage.gaussian_filter(c2_clean, sigma=0.5, mode="nearest")

                # Preserve original finite values
                c2_clean[finite_mask] = c2[finite_mask]

            return c2_clean
        except ImportError:
            logger.warning(
                "scipy not available for interpolation, falling back to nan_to_num"
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

    hdl["saxs"].setImage(np.flipud(saxs))
    hdl["dqmap"].setImage(dqmap_disp)

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
    hdl["tt"].setImage(c2_clean, autoRange=True)  # Use cleaned data

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
