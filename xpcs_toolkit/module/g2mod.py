# Third-party imports
import numpy as np
import pyqtgraph as pg

# Local imports
from xpcs_toolkit.utils.logging_config import get_logger

pg.setConfigOption("foreground", pg.mkColor(80, 80, 80))
# pg.setConfigOption("background", 'w')
logger = get_logger(__name__)

# colors converted from
# https://matplotlib.org/stable/tutorials/colors/colors.html
# colors = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf')

colors = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
)


# https://www.geeksforgeeks.org/pyqtgraph-symbols/
symbols = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d", "x"]


def get_data(xf_list, q_range=None, t_range=None):
    # Early validation - check all files have correlation analysis (Multitau or Twotime)
    analysis_types = [xf.atype for xf in xf_list]
    if not all(
        any(atype_part in ["Multitau", "Twotime"] for atype_part in atype)
        for atype in analysis_types
    ):
        return False, None, None, None, None

    # Pre-allocate lists with known size for better memory efficiency
    num_files = len(xf_list)
    q = [None] * num_files
    tel = [None] * num_files
    g2 = [None] * num_files
    g2_err = [None] * num_files
    labels = [None] * num_files

    # Process all files - can potentially be parallelized
    for i, fc in enumerate(xf_list):
        _q, _tel, _g2, _g2_err, _labels = fc.get_g2_data(qrange=q_range, trange=t_range)
        q[i] = _q
        tel[i] = _tel
        g2[i] = _g2
        g2_err[i] = _g2_err
        labels[i] = _labels

    return q, tel, g2, g2_err, labels


def compute_geometry(g2, plot_type):
    """
    compute the number of figures and number of plot lines for a given type
    and dataset;
    :param g2: input g2 data; 2D array; dim0: t_el; dim1: q_vals
    :param plot_type: string in ['multiple', 'single', 'single-combined']
    :return: tuple of (number_of_figures, number_of_lines)
    """
    if plot_type == "multiple":
        num_figs = g2[0].shape[1]
        num_lines = len(g2)
    elif plot_type == "single":
        num_figs = len(g2)
        num_lines = g2[0].shape[1]
    elif plot_type == "single-combined":
        num_figs = 1
        num_lines = g2[0].shape[1] * len(g2)
    else:
        raise ValueError("plot_type not support.")
    return num_figs, num_lines


def pg_plot(
    hdl,
    xf_list,
    q_range,
    t_range,
    y_range,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    rows=None,
    offset=0,
    show_fit=False,
    show_label=False,
    bounds=None,
    fit_flag=None,
    plot_type="multiple",
    subtract_baseline=True,
    marker_size=5,
    label_size=4,
    fit_func="single",
    **kwargs,
):
    if q_auto:
        q_range = None
    if t_auto:
        t_range = None
    if y_auto:
        y_range = None

    _q, tel, g2, g2_err, labels = get_data(xf_list, q_range=q_range, t_range=t_range)
    num_figs, _num_lines = compute_geometry(g2, plot_type)

    num_data, num_qval = len(g2), g2[0].shape[1]
    # col and rows for the 2d layout
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    if rows is None or len(rows) == 0:
        rows = list(range(len(xf_list)))

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()
    # a bug in pyqtgraph; the log scale in x-axis doesn't apply
    if t_range:
        # Handle log10 of zero or negative values
        with np.errstate(divide='ignore', invalid='ignore'):
            # Only take log10 of positive values
            t_range_positive = np.asarray(t_range)
            t_range_positive = np.where(t_range_positive > 0, t_range_positive, np.finfo(float).eps)
            t0_range = np.log10(t_range_positive)
    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            t.addLegend(offset=(-1, 1), labelTextSize="9pt", verSpacing=-10)

        t.setMouseEnabled(x=False, y=y_auto)

    for m in range(num_data):
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)
        if show_fit:
            fit_summary = xf_list[m].fit_g2(
                q_range, t_range, bounds, fit_flag, fit_func
            )
            if fit_summary is not None and subtract_baseline:
                # make sure the fitting is successful
                if fit_summary["fit_line"][n].get("success", False):
                    baseline_offset = fit_summary["fit_val"][:, 0, 3]

        for n in range(num_qval):
            color = colors[rows[m] % len(colors)]
            label = None
            if plot_type == "multiple":
                ax = axes[n]
                title = labels[m][n]
                label = xf_list[m].label
                if m == 0:
                    ax.setTitle(title)
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % len(colors)]
                title = xf_list[m].label
                # label = labels[m][n]
                ax.setTitle(title)
            elif plot_type == "single-combined":
                ax = axes[0]
                label = xf_list[m].label + labels[m][n]

            ax.setLabel("bottom", "tau (s)")
            ax.setLabel("left", "g2")

            symbol = symbols[rows[m] % len(symbols)]

            x = tel[m]
            # normalize baseline
            y = g2[m][:, n] - baseline_offset[n] + 1.0 + m * offset
            y_err = g2_err[m][:, n]

            pg_plot_one_g2(
                ax,
                x,
                y,
                y_err,
                color,
                label=label,
                symbol=symbol,
                symbol_size=marker_size,
            )
            # if t_range is not None:
            if not y_auto:
                ax.setRange(yRange=y_range)
            if not t_auto:
                ax.setRange(xRange=t0_range)

            if show_fit and fit_summary is not None:
                if fit_summary["fit_line"][n].get("success", False):
                    y_fit = fit_summary["fit_line"][n]["fit_y"] + m * offset
                    # normalize baseline
                    y_fit = y_fit - baseline_offset[n] + 1.0
                    ax.plot(
                        fit_summary["fit_line"][n]["fit_x"],
                        y_fit,
                        pen=pg.mkPen(color, width=2.5),
                    )


def pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5):
    """
    Optimized G2 plotting with improved data validation and performance.
    """
    # Validate input data
    if len(x) == 0 or len(y) == 0:
        return

    # Filter out invalid data points (NaN, inf, non-positive x for log scale)
    valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dy) & (x > 0)
    if not np.any(valid_mask):
        return  # Skip if no valid data

    x_clean = x[valid_mask]
    y_clean = y[valid_mask]
    dy_clean = dy[valid_mask]

    # Optimize pen creation
    pen_line = pg.mkPen(color=color, width=2)
    pen_symbol = pg.mkPen(color=color, width=1)

    # Create error bars more efficiently
    try:
        log_x = np.log10(x_clean)
        line = pg.ErrorBarItem(
            x=log_x, y=y_clean, top=dy_clean, bottom=dy_clean, pen=pen_line
        )
    except (ValueError, RuntimeWarning):
        # Handle edge cases in logarithm calculation
        return

    # Downsample data if too many points for better performance
    if len(x_clean) > 500:
        step = len(x_clean) // 250
        x_plot = x_clean[::step]
        y_plot = y_clean[::step]
    else:
        x_plot = x_clean
        y_plot = y_clean

    # Plot symbols with optimized parameters
    ax.plot(
        x_plot,
        y_plot,
        pen=None,
        symbol=symbol,
        name=label,
        symbolSize=symbol_size,
        symbolPen=pen_symbol,
        symbolBrush=pg.mkBrush(color=(*color, 0)),
    )

    ax.setLogMode(x=True, y=None)
    ax.addItem(line)
    return


def vectorized_g2_baseline_correction(g2_data, baseline_values):
    """
    Vectorized baseline correction for G2 data.

    Args:
        g2_data: G2 data array [time, q_values]
        baseline_values: Baseline values [q_values]

    Returns:
        Baseline-corrected G2 data
    """
    # Broadcast baseline subtraction across all time points
    return g2_data - baseline_values[np.newaxis, :] + 1.0


def batch_g2_normalization(g2_data_list, method="max"):
    """
    Batch normalization of multiple G2 datasets using vectorized operations.

    Args:
        g2_data_list: List of G2 data arrays
        method: Normalization method ('max', 'mean', 'std')

    Returns:
        List of normalized G2 data arrays
    """
    normalized_data = []

    for g2_data in g2_data_list:
        if method == "max":
            # Vectorized max normalization
            max_vals = np.max(g2_data, axis=0, keepdims=True)
            # Avoid division by zero
            max_vals = np.where(max_vals == 0, 1.0, max_vals)
            normalized = g2_data / max_vals
        elif method == "mean":
            # Vectorized mean normalization
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            mean_vals = np.where(mean_vals == 0, 1.0, mean_vals)
            normalized = g2_data / mean_vals
        elif method == "std":
            # Vectorized standard score normalization
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            std_vals = np.std(g2_data, axis=0, keepdims=True)
            std_vals = np.where(std_vals == 0, 1.0, std_vals)
            normalized = (g2_data - mean_vals) / std_vals
        else:
            normalized = g2_data

        normalized_data.append(normalized)

    return normalized_data


def compute_g2_ensemble_statistics(g2_data_list):
    """
    Compute ensemble statistics for multiple G2 datasets using vectorized operations.

    Args:
        g2_data_list: List of G2 data arrays [time, q_values]

    Returns:
        Dictionary with ensemble statistics
    """
    # Stack all data for vectorized operations
    g2_stack = np.stack(g2_data_list, axis=0)  # [batch, time, q_values]

    # Vectorized statistical computations
    stats = {
        "ensemble_mean": np.mean(g2_stack, axis=0),
        "ensemble_std": np.std(g2_stack, axis=0),
        "ensemble_median": np.median(g2_stack, axis=0),
        "ensemble_min": np.min(g2_stack, axis=0),
        "ensemble_max": np.max(g2_stack, axis=0),
        "ensemble_var": np.var(g2_stack, axis=0),
        "q_mean_values": np.mean(g2_stack, axis=(0, 1)),  # Mean across time and batch
        "temporal_correlation": [],
    }

    # Compute temporal correlations for each q-value
    for q_idx in range(g2_stack.shape[2]):
        q_data = g2_stack[:, :, q_idx]  # [batch, time]
        # Vectorized correlation matrix computation
        corr_matrix = np.corrcoef(q_data)
        stats["temporal_correlation"].append(corr_matrix)

    return stats


def optimize_g2_error_propagation(g2_data, g2_errors, operations):
    """
    Vectorized error propagation for G2 data operations.

    Args:
        g2_data: G2 data array [time, q_values]
        g2_errors: G2 error array [time, q_values]
        operations: List of operations applied to data

    Returns:
        Propagated errors
    """
    propagated_errors = g2_errors.copy()

    for op in operations:
        if op["type"] == "scale":
            # Error propagation for scaling: σ_new = |scale| * σ_old
            scale_factor = op["factor"]
            propagated_errors *= np.abs(scale_factor)

        elif op["type"] == "offset":
            # Error propagation for offset: σ_new = σ_old (additive operations don't change uncertainty)
            pass

        elif op["type"] == "power":
            # Error propagation for power: σ_new = |n * x^(n-1)| * σ_old
            power = op["power"]
            propagated_errors = (
                np.abs(power * np.power(g2_data, power - 1)) * propagated_errors
            )

        elif op["type"] == "log":
            # Error propagation for logarithm: σ_new = σ_old / |x|
            propagated_errors = propagated_errors / np.abs(g2_data)

    return propagated_errors


def vectorized_g2_interpolation(tel, g2_data, target_tel):
    """
    Vectorized interpolation of G2 data to new time points.

    Args:
        tel: Original time points
        g2_data: G2 data [time, q_values]
        target_tel: Target time points for interpolation

    Returns:
        Interpolated G2 data
    """
    from scipy.interpolate import interp1d

    # Vectorized interpolation for all q-values simultaneously
    interpolated_data = np.zeros((len(target_tel), g2_data.shape[1]))

    # Process all q-values in vectorized manner
    for q_idx in range(g2_data.shape[1]):
        # Create interpolation function
        interp_func = interp1d(
            tel,
            g2_data[:, q_idx],
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )

        # Vectorized interpolation
        interpolated_data[:, q_idx] = interp_func(target_tel)

    return interpolated_data
