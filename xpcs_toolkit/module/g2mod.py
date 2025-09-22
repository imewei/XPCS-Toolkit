# Third-party imports
import numpy as np
import pyqtgraph as pg

from xpcs_toolkit.plothandler.plot_constants import MATPLOTLIB_COLORS_RGB as colors

# Local imports
from xpcs_toolkit.utils.logging_config import get_logger

pg.setConfigOption("foreground", pg.mkColor(80, 80, 80))
# pg.setConfigOption("background", 'w')
logger = get_logger(__name__)


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
    robust_fitting=False,
    enable_diagnostics=False,
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
        with np.errstate(divide="ignore", invalid="ignore"):
            # Only take log10 of positive values
            t_range_positive = np.asarray(t_range)
            t_range_positive = np.where(
                t_range_positive > 0, t_range_positive, np.finfo(float).eps
            )
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
            # Extract force_refit parameter from kwargs
            force_refit = kwargs.get("force_refit", False)

            if robust_fitting:
                # Use robust fitting with diagnostics
                fit_summary = xf_list[m].fit_g2_robust(
                    q_range,
                    t_range,
                    bounds,
                    fit_flag,
                    fit_func,
                    enable_diagnostics=enable_diagnostics,
                    force_refit=force_refit,
                    **{k: v for k, v in kwargs.items() if k != "force_refit"},
                )
            else:
                # Use traditional fitting
                fit_summary = xf_list[m].fit_g2(
                    q_range,
                    t_range,
                    bounds,
                    fit_flag,
                    fit_func,
                    force_refit=force_refit,
                )

            if fit_summary is not None and subtract_baseline:
                # Check if fitting was successful by validating fit_val
                if (
                    fit_summary["fit_val"] is not None
                    and len(fit_summary["fit_val"]) > 0
                ):
                    # Extract baseline parameter (parameter index 3 for single exponential)
                    try:
                        baseline_offset = fit_summary["fit_val"][:, 0, 3]
                    except (IndexError, TypeError):
                        # Fallback to default baseline if shape doesn't match
                        baseline_offset = np.ones(num_qval)

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
                # Check if we have valid fit_line data for this q-index
                if (
                    fit_summary["fit_line"] is not None
                    and n < fit_summary["fit_line"].shape[0]
                    and fit_summary.get("fit_x") is not None
                ):
                    # Get fitted y-values from the numpy array
                    y_fit = fit_summary["fit_line"][n] + m * offset
                    # normalize baseline
                    y_fit = y_fit - baseline_offset[n] + 1.0
                    # Use the correct x-values that were used for fitting
                    fit_x = fit_summary["fit_x"]
                    ax.plot(
                        fit_x,
                        y_fit,
                        pen=pg.mkPen(color, width=2.5),
                    )

                    # Add fitted parameter text annotation
                    if (
                        fit_summary.get("fit_val") is not None
                        and n < fit_summary["fit_val"].shape[0]
                    ):
                        fit_val = fit_summary["fit_val"][
                            n
                        ]  # [2, n_params] - values and errors

                        # Format parameters based on fit function
                        if fit_func == "single":
                            param_names = ["τ", "bkg", "cts", "d"]
                        else:  # double exponential
                            param_names = ["τ1", "bkg", "cts1", "τ2", "cts2"]

                        # Create parameter text
                        param_text_lines = []
                        for p_idx in range(min(len(param_names), fit_val.shape[1])):
                            value = fit_val[0, p_idx]  # fitted value
                            error = fit_val[1, p_idx]  # error estimate

                            # Format error gracefully
                            if np.isfinite(error) and error > 0:
                                if (
                                    param_names[p_idx] == "τ"
                                    or param_names[p_idx] == "τ1"
                                ):
                                    param_text_lines.append(
                                        f"{param_names[p_idx]} = {value:.3e} ± {error:.2e}"
                                    )
                                else:
                                    param_text_lines.append(
                                        f"{param_names[p_idx]} = {value:.3f} ± {error:.3f}"
                                    )
                            elif (
                                param_names[p_idx] == "τ" or param_names[p_idx] == "τ1"
                            ):
                                param_text_lines.append(
                                    f"{param_names[p_idx]} = {value:.3e} ± --"
                                )
                            else:
                                param_text_lines.append(
                                    f"{param_names[p_idx]} = {value:.3f} ± --"
                                )

                        param_text = "\n".join(param_text_lines)

                        # Add text item to plot
                        text_item = pg.TextItem(param_text, anchor=(0, 1), color=color)

                        # Position text in data coordinates (top-left of visible area)
                        viewbox = ax.getViewBox()
                        if viewbox is not None:
                            # Get current view range
                            [[xmin, xmax], [ymin, ymax]] = viewbox.viewRange()
                            # Position at 5% from left edge, 95% from bottom (top area)
                            text_x = xmin + 0.05 * (xmax - xmin)
                            text_y = ymin + 0.95 * (ymax - ymin)
                            text_item.setPos(text_x, text_y)
                        else:
                            # Fallback position
                            text_item.setPos(-5, 1.5)

                        ax.addItem(text_item)


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
