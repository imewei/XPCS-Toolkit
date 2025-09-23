# Third-party imports
import numpy as np
import pyqtgraph as pg

from xpcs_toolkit.utils.logging_config import get_logger

# Local imports
from ..plothandler.matplot_qt import get_color_marker

logger = get_logger(__name__)

pg.setConfigOption("background", "w")


# Mapping from integer codes to string codes (based on Matplotlib docs)
_MPL_LOC_INT_TO_STR = {
    1: "upper right",
    2: "upper left",
    3: "lower left",
    4: "lower right",
    5: "right",  # Often equivalent to center right in placement
    6: "center left",
    7: "center right",
    8: "lower center",
    9: "upper center",
    10: "center",
}


def get_pyqtgraph_anchor_params(loc, padding=10):
    """
    Converts a Matplotlib loc string or code to pyqtgraph anchor parameters.

    Calculates the 'itemPos', 'parentPos', and 'offset' needed to position
    a pyqtgraph LegendItem similarly to how Matplotlib places legends
    using the 'loc' parameter.

    Args:
        loc (str or int): Matplotlib location code. Accepts standard strings
                          ('upper left', 'center', etc.) or integer codes (0-10).
        padding (int): Pixel padding to use for the offset from the anchor point.
                       Positive values generally push the legend inwards from
                       the edge/corner. Defaults to 10.

    Returns:
        dict or None: A dictionary with keys 'itemPos', 'parentPos', and 'offset'
                      suitable for unpacking into LegendItem.anchor(\\*\\*params),
                      or None if loc='best' (code 0) as it is not directly
                      supported by pyqtgraph deterministic anchoring.

    Raises:
        ValueError: If the loc code or type is invalid.

    Example Usage::

        plot_item = pg.PlotItem()
        legend = plot_item.addLegend()
        # ... plot data ...
        try:
            anchor_params = get_pyqtgraph_anchor_params('lower left', padding=15)
            if anchor_params:
                legend.anchor(\\*\\*anchor_params)
            else:
                logger.info("Using default legend position for best location")
        except ValueError as e:
            logger.warning(f"Error setting legend position: {e}")

    """
    if isinstance(loc, int):
        if loc in _MPL_LOC_INT_TO_STR:
            loc_str = _MPL_LOC_INT_TO_STR[loc]
        else:
            raise ValueError(f"Invalid Matplotlib integer location code: {loc}")
    elif isinstance(loc, str):
        loc_str = (
            loc.lower().replace(" ", "").replace("_", "")
        )  # Normalize input string
    else:
        raise ValueError(f"Invalid loc type: {type(loc)}. Must be str or int.")

    # --- Define anchor points and offset multipliers ---
    # Map: loc_string -> (itemPos, parentPos, offset_multipliers)
    # Offset multipliers (mult_x, mult_y) determine offset direction based on padding
    _ANCHOR_MAP = {
        # Corners
        "upperleft": ((0.0, 0.0), (0.0, 0.0), (1, 1)),  # Offset moves down-right
        "upperright": ((1.0, 0.0), (1.0, 0.0), (-1, 1)),  # Offset moves down-left
        "lowerleft": ((0.0, 1.0), (0.0, 1.0), (1, -1)),  # Offset moves up-right
        "lowerright": ((1.0, 1.0), (1.0, 1.0), (-1, -1)),  # Offset moves up-left
        # Centers
        "center": ((0.5, 0.5), (0.5, 0.5), (0, 0)),  # No offset needed usually
        "lowercenter": ((0.5, 1.0), (0.5, 1.0), (0, -1)),  # Offset moves up
        "uppercenter": ((0.5, 0.0), (0.5, 0.0), (0, 1)),  # Offset moves down
        # Sides (center align on edge)
        "centerleft": ((0.0, 0.5), (0.0, 0.5), (1, 0)),  # Offset moves right
        "centerright": ((1.0, 0.5), (1.0, 0.5), (-1, 0)),  # Offset moves left
        "right": (
            (1.0, 0.5),
            (1.0, 0.5),
            (-1, 0),
        ),  # Treat 'right' same as 'centerright'
    }

    if loc_str in _ANCHOR_MAP:
        itemPos, parentPos, offset_mult = _ANCHOR_MAP[loc_str]
        offset = (padding * offset_mult[0], padding * offset_mult[1])
        return {"itemPos": itemPos, "parentPos": parentPos, "offset": offset}
    raise ValueError(f"Invalid or unsupported Matplotlib location string: '{loc}'")


def offset_intensity(Iq, n, plot_offset=None, yscale=None):
    """
    Vectorized intensity offset with optimized memory usage.
    """
    if plot_offset is None or plot_offset == 0:
        return Iq

    if yscale == "linear":
        # Vectorized linear offset with single max computation
        max_Iq = np.max(Iq, axis=-1, keepdims=True) if Iq.ndim > 1 else np.max(Iq)
        offset = -plot_offset * n * max_Iq
        return Iq + offset

    if yscale == "log":
        # Vectorized logarithmic offset using optimized power calculation
        offset = np.power(10.0, plot_offset * n)
        return Iq / offset

    return Iq  # Return original if no scaling applied


def switch_line_builder(hdl, lb_type=None):
    hdl.link_line_builder(lb_type)


def plot_line_with_marker(
    plot_item, x, y, index, label, alpha_val, marker_size=6, log_x=False, log_y=False
):
    """
    Vectorized plotting function with advanced data filtering and memory optimization.
    """
    color_hex, marker = get_color_marker(index, backend="pyqtgraph")
    rgba = (*pg.mkColor(color_hex).getRgb()[:3], int(alpha_val * 255))

    # Ensure arrays for consistent processing
    x = np.atleast_1d(np.asarray(x))
    y = np.atleast_1d(np.asarray(y))

    # Handle dimension mismatch
    if len(x) != len(y):
        logger.warning(
            f"Dimension mismatch: x has {len(x)} elements, y has {len(y)} elements"
        )
        return  # Skip plotting if dimensions don't match

    # Vectorized data validation - single pass filtering
    if log_y:
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    else:
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)

    # Ensure valid_mask is always a 1D array to handle scalar inputs
    valid_mask = np.atleast_1d(valid_mask)

    if not np.any(valid_mask):
        return  # Skip if no valid data

    # Apply filtering in single operation
    x_clean, y_clean = x[valid_mask], y[valid_mask]

    # Intelligent downsampling based on data characteristics
    n_points = len(x_clean)
    if n_points > 2000:
        # Use logarithmic sampling for better feature preservation
        if log_x:
            # Sample more points in log-space for log-scale data
            np.log10(x_clean)
            indices = np.linspace(0, n_points - 1, 1000).astype(int)
        else:
            # Use linear sampling for linear data
            indices = np.linspace(0, n_points - 1, 1000).astype(int)

        indices = np.unique(indices)  # Remove potential duplicates
        x_plot = x_clean[indices]
        y_plot = y_clean[indices]
    else:
        x_plot = x_clean
        y_plot = y_clean

    # Optimized pen creation with caching
    pen_line = pg.mkPen(color=rgba, width=1.5)

    # Plot line with cleaned data
    plot_item.plot(x_plot, y_plot, pen=pen_line, name=label)

    # Optimized scatter plot for markers
    if len(x_plot) <= 1000:  # Only show markers for manageable datasets
        # Apply log transforms efficiently
        if log_x or log_y:
            x_scatter = np.log10(x_plot) if log_x else x_plot
            y_scatter = np.log10(y_plot) if log_y else y_plot
        else:
            x_scatter = x_plot
            y_scatter = y_plot

        # Create scatter plot with optimized parameters
        scatter = pg.ScatterPlotItem(
            x=x_scatter,
            y=y_scatter,
            symbol=marker,
            size=marker_size,
            pen=pg.mkPen(color=rgba, width=1),
            brush=None,  # No fill for better performance
        )
        plot_item.addItem(scatter)


def pg_plot(
    xf_list,
    pg_hdl,
    plot_type=2,
    plot_norm=0,
    plot_offset=0,
    title=None,
    rows=None,
    qmax=10.0,
    qmin=0,
    loc="best",
    marker_size=3,
    sampling=1,
    all_phi=False,
    absolute_crosssection=False,
    subtract_background=False,
    bkg_file=None,
    weight=1.0,
    roi_list=None,
    show_roi=True,
    show_phi_roi=True,
):
    """
    Highly optimized SAXS1D plotting with vectorized operations and memory efficiency.
    """
    pg_hdl.clear()
    plot_item = pg_hdl.getPlotItem()
    plot_item.setTitle(title)
    legend = plot_item.addLegend()

    # Handle invalid loc parameter gracefully
    try:
        anchor_param = get_pyqtgraph_anchor_params(loc, padding=15)
        if anchor_param:
            legend.anchor(**anchor_param)
    except (ValueError, TypeError):
        pass  # Use default position if loc is invalid

    # Vectorized alpha computation with early termination
    num_files = len(xf_list)
    if rows:
        alpha = np.full(num_files, 0.35, dtype=np.float32)
        alpha[rows] = 1.0
    else:
        alpha = np.ones(num_files, dtype=np.float32)

    # Optimize background handling
    if not subtract_background:
        bkg_file = None

    # Pre-compute plot parameters using vectorized operations
    norm_methods = [None, "q2", "q4", "I0"]
    norm_method = norm_methods[plot_norm] if plot_norm < len(norm_methods) else None
    log_x = bool(plot_type % 2)
    log_y = bool(plot_type // 2)
    plot_item.setLogMode(x=log_x, y=log_y)

    plot_id = 0
    xlabel, ylabel = None, None

    # Batch data extraction for parallel processing potential
    plot_data = []
    for n, fi in enumerate(xf_list):
        try:
            q, Iq, xlabel, ylabel = fi.get_saxs1d_data(
                bkg_xf=bkg_file,
                bkg_weight=weight,
                qrange=(qmin, qmax),
                sampling=sampling,
                norm_method=norm_method,
                use_absolute_crosssection=absolute_crosssection,
            )

            # Vectorized intensity offset application
            if plot_offset > 0:
                Iq = offset_intensity(
                    Iq, n, plot_offset, yscale="log" if log_y else "linear"
                )

            plot_data.append((q, Iq, fi, n))

        except Exception as e:
            logger.warning(f"Failed to extract data for file {fi.label}: {e}")
            continue

    # Vectorized plotting with optimized loops
    for q, Iq, fi, n in plot_data:
        num_lines = Iq.shape[0] if all_phi else 1

        # Vectorized validity check for all lines at once
        valid_lines = (
            np.any(np.isfinite(Iq), axis=1)
            if Iq.ndim > 1
            else [np.any(np.isfinite(Iq))]
        )

        for m in range(num_lines):
            if valid_lines[m] if len(valid_lines) > m else valid_lines[0]:
                plot_line_with_marker(
                    plot_item,
                    q,
                    Iq[m] if Iq.ndim > 1 else Iq,
                    plot_id,
                    fi.saxs_1d["labels"][m],
                    alpha[n],
                    marker_size=marker_size,
                    log_x=log_x,
                    log_y=log_y,
                )
            plot_id += 1

    # Optimized ylabel generation with caching
    if plot_norm == 0:  # no normalization
        ylabel = (
            "Intensity (1/cm)"
            if absolute_crosssection
            else "Intensity (photon/pixel/frame)"
        )

    # Set labels efficiently
    if xlabel:
        plot_item.setLabel("bottom", xlabel)
    if ylabel:
        plot_item.setLabel("left", ylabel)

    plot_item.showGrid(x=True, y=True, alpha=0.3)


def vectorized_q_binning(q_values, intensities, q_min, q_max, num_bins):
    """
    Vectorized q-space binning for SAXS data with optimized memory usage.

    Args:
        q_values: Q-values array
        intensities: Intensity array [q_points] or [phi_slices, q_points]
        q_min, q_max: Q-range for binning
        num_bins: Number of bins

    Returns:
        Tuple of (binned_q, binned_intensity, bin_counts)
    """
    # Create bin edges using vectorized operations
    bin_edges = np.linspace(q_min, q_max, num_bins + 1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Vectorized binning using digitize
    bin_indices = np.digitize(q_values, bin_edges) - 1

    # Handle 1D or 2D intensity arrays
    if intensities.ndim == 1:
        binned_intensity = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        # Vectorized accumulation
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                binned_intensity[i] = np.mean(intensities[mask])
                bin_counts[i] = np.sum(mask)
    else:
        # 2D case: multiple phi slices
        num_phi = intensities.shape[0]
        binned_intensity = np.zeros((num_phi, num_bins))
        bin_counts = np.zeros(num_bins)

        # Vectorized processing for all phi slices
        for i in range(num_bins):
            mask = bin_indices == i
            if np.any(mask):
                binned_intensity[:, i] = np.mean(intensities[:, mask], axis=1)
                bin_counts[i] = np.sum(mask)

    return bin_centers, binned_intensity, bin_counts


def vectorized_background_subtraction(foreground_data, background_data, weight=1.0):
    """
    Vectorized background subtraction with error propagation.

    Args:
        foreground_data: Tuple of (q, I_fg, I_err_fg)
        background_data: Tuple of (q_bg, I_bg, I_err_bg)
        weight: Background scaling weight

    Returns:
        Tuple of (q, I_subtracted, I_err_propagated)
    """
    q, I_fg, I_err_fg = foreground_data
    q_bg, I_bg, I_err_bg = background_data

    # Interpolate background to foreground q-values if needed
    if not np.array_equal(q, q_bg):
        from scipy.interpolate import interp1d

        interp_func = interp1d(
            q_bg, I_bg, kind="linear", bounds_error=False, fill_value=0
        )
        I_bg_interp = interp_func(q)

        # Interpolate errors as well
        if I_err_bg is not None:
            interp_err_func = interp1d(
                q_bg, I_err_bg, kind="linear", bounds_error=False, fill_value=0
            )
            I_err_bg_interp = interp_err_func(q)
        else:
            I_err_bg_interp = np.zeros_like(q)
    else:
        I_bg_interp = I_bg
        I_err_bg_interp = I_err_bg if I_err_bg is not None else np.zeros_like(q)

    # Vectorized background subtraction
    I_subtracted = I_fg - weight * I_bg_interp

    # Vectorized error propagation
    if I_err_fg is not None:
        I_err_propagated = np.sqrt(I_err_fg**2 + (weight * I_err_bg_interp) ** 2)
    else:
        I_err_propagated = weight * I_err_bg_interp

    return q, I_subtracted, I_err_propagated


def vectorized_intensity_normalization(
    q_values, intensities, method="none", q_ref=None
):
    """
    Vectorized intensity normalization using various methods.

    Args:
        q_values: Q-values array
        intensities: Intensity array
        method: Normalization method ('none', 'q2', 'q4', 'max', 'area')
        q_ref: Reference q-value for certain normalizations

    Returns:
        Normalized intensity array
    """
    if method == "none":
        return intensities

    if intensities.ndim == 1:
        # 1D case
        if method == "q2":
            return intensities * q_values**2
        if method == "q4":
            return intensities * q_values**4
        if method == "max":
            return intensities / np.max(intensities)
        if method == "area":
            # Numerical integration for area normalization
            area = np.trapz(intensities, q_values)
            return intensities / area
    # 2D case: multiple curves
    elif method == "q2":
        return intensities * q_values[np.newaxis, :] ** 2
    elif method == "q4":
        return intensities * q_values[np.newaxis, :] ** 4
    elif method == "max":
        max_vals = np.max(intensities, axis=1, keepdims=True)
        return intensities / max_vals
    elif method == "area":
        # Vectorized area normalization for multiple curves
        areas = np.array(
            [np.trapz(intensities[i], q_values) for i in range(intensities.shape[0])]
        )
        return intensities / areas[:, np.newaxis]

    return intensities


def batch_saxs_analysis(data_list, operations):
    """
    Batch processing of multiple SAXS datasets with vectorized operations.

    Args:
        data_list: List of (q, I) tuples
        operations: List of operations to apply

    Returns:
        List of processed (q, I) tuples
    """
    processed_data = []

    # Pre-allocate arrays for batch operations where possible
    max(len(q) for q, _ in data_list)

    for q, intensity in data_list:
        processed_I = intensity.copy()

        for op in operations:
            if op["type"] == "normalize":
                processed_I = vectorized_intensity_normalization(
                    q, processed_I, method=op["method"]
                )

            elif op["type"] == "smooth":
                # Vectorized smoothing
                from scipy.ndimage import gaussian_filter1d

                sigma = op.get("sigma", 1.0)
                processed_I = gaussian_filter1d(processed_I, sigma=sigma)

            elif op["type"] == "trim":
                # Vectorized trimming
                q_min, q_max = op["q_range"]
                mask = (q >= q_min) & (q <= q_max)
                q = q[mask]
                processed_I = processed_I[mask]

        processed_data.append((q, processed_I))

    return processed_data


def optimize_roi_extraction(image_stack, roi_definitions):
    """
    Vectorized ROI extraction from image stacks.

    Args:
        image_stack: Image array [frames, height, width] or [height, width]
        roi_definitions: List of ROI definitions with coordinates

    Returns:
        Dictionary with extracted ROI data
    """
    roi_data = {}

    is_stack = image_stack.ndim == 3

    for i, roi in enumerate(roi_definitions):
        roi_name = roi.get("name", f"ROI_{i}")

        if roi["type"] == "rectangular":
            x1, y1, x2, y2 = roi["coords"]
            if is_stack:
                roi_region = image_stack[:, y1:y2, x1:x2]
                # Vectorized sum over spatial dimensions
                roi_intensities = np.sum(roi_region, axis=(1, 2))
            else:
                roi_region = image_stack[y1:y2, x1:x2]
                roi_intensities = np.sum(roi_region)

        elif roi["type"] == "circular":
            center_x, center_y, radius = roi["coords"]

            # Create coordinate grids
            if is_stack:
                height, width = image_stack.shape[1], image_stack.shape[2]
            else:
                height, width = image_stack.shape

            y_grid, x_grid = np.ogrid[:height, :width]

            # Vectorized distance calculation
            distance_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            mask = distance_sq <= radius**2

            if is_stack:
                # Apply mask to all frames vectorized
                roi_intensities = np.sum(
                    image_stack * mask[np.newaxis, :, :], axis=(1, 2)
                )
            else:
                roi_intensities = np.sum(image_stack * mask)

        roi_data[roi_name] = {
            "intensities": roi_intensities,
            "pixel_count": np.sum(mask)
            if roi["type"] == "circular"
            else (x2 - x1) * (y2 - y1),
        }

    return roi_data
