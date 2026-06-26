"""G2 correlation analysis module.

Provides multi-tau correlation analysis with single and double exponential
fitting for XPCS time correlation functions.

Functions:
    get_data: Extract G2 correlation data from XpcsFile objects
    fit_g2: Fit G2 data with exponential models
"""

# Standard library imports
import logging

# Third-party imports
import numpy as np
import pyqtgraph as pg

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.plothandler.plot_constants import MATPLOTLIB_COLORS_RGB as colors

# Local imports
from xpcsviewer.utils.logging_config import get_logger

pg.setConfigOption("foreground", pg.mkColor(80, 80, 80))
# pg.setConfigOption("background", 'w')
logger = get_logger(__name__)


def _log_array_dims(prefix: str, arr) -> None:
    """Log array shape/dtype for data flow tracing."""
    if logger.isEnabledFor(logging.DEBUG):
        if arr is None:
            logger.debug(f"{prefix}: None")
        elif hasattr(arr, "shape"):
            logger.debug(f"{prefix}: shape={arr.shape}, dtype={arr.dtype}")
        elif isinstance(arr, (list, tuple)):
            logger.debug(f"{prefix}: list of {len(arr)} items")


# https://www.geeksforgeeks.org/pyqtgraph-symbols/
symbols = ["o", "t", "t1", "t2", "t3", "s", "p", "h", "star", "+", "d", "x"]


def get_data(xf_list, q_range=None, t_range=None):
    """Extract G2 correlation data from a list of XpcsFile objects.

    Reads G2 correlation values, errors, delay times, and Q-values from
    each file, optionally filtered by Q-range and time-range.

    Args:
        xf_list: List of XpcsFile objects with correlation analysis data.
        q_range: Optional (min, max) Q-value filter in inverse Angstroms.
        t_range: Optional (min, max) delay time filter in seconds.

    Returns:
        If all files contain correlation data, returns a 5-tuple:
        ``(q, tel, g2, g2_err, labels)`` where each element is a list
        of per-file arrays. Returns ``(False, None, None, None, None)``
        if any file lacks Multitau or Twotime analysis.

    Example:
        >>> q, tel, g2, g2_err, labels = get_data(xf_list, q_range=(0.01, 0.1))
        >>> if q is not False:
        ...     print(f"Loaded {len(q)} files, {g2[0].shape[1]} Q-bins")
    """
    logger.debug(
        f"get_data: entry with {len(xf_list)} files, q_range={q_range}, t_range={t_range}"
    )
    # Early validation - check all files have correlation analysis (Multitau or Twotime)
    analysis_types = [xf.atype for xf in xf_list]
    if not all(
        any(atype_part in ["Multitau", "Twotime"] for atype_part in atype)
        for atype in analysis_types
    ):
        logger.debug("get_data: exit early - no correlation analysis")
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

    _log_array_dims("get_data g2", g2[0] if g2 else None)
    logger.debug(f"get_data: exit with {num_files} datasets")
    return q, tel, g2, g2_err, labels


def get_g2_stability_data(xf_obj, q_range=None, t_range=None):
    """
    Extract G2 stability data from a single XpcsFile object.

    Parameters
    ----------
    xf_obj : XpcsFile
        Single XpcsFile object with g2_partial data
    q_range : tuple or None
        Q-range filter (qmin, qmax)
    t_range : tuple or None
        Time range filter (tmin, tmax)

    Returns
    -------
    tuple
        (q, tel, g2, g2_err, qbin_labels, labels)
    """
    if "Multitau" not in xf_obj.atype:
        return False, None, None, None, None, None

    q, tel, g2, g2_err, qbin_labels, labels = xf_obj.get_g2_stability_data(
        qrange=q_range, trange=t_range
    )
    return q, tel, g2, g2_err, qbin_labels, labels


def pg_plot_stability(
    hdl,
    xf_obj,
    q_range,
    t_range,
    y_range,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    offset=0,
    show_label=False,
    plot_type="multiple",
    marker_size=5,
    **kwargs,
):
    """
    Plot G2 stability data showing frame-by-frame correlation analysis.

    Parameters
    ----------
    hdl : GraphicsLayoutWidget
        PyQtGraph layout widget for plotting
    xf_obj : XpcsFile
        XpcsFile object with g2_partial data
    q_range : tuple
        Q-range for filtering
    t_range : tuple
        Time range for filtering
    y_range : tuple
        Y-axis range
    y_auto, q_auto, t_auto : bool
        Auto-range flags
    num_col : int
        Number of columns in layout
    offset : float
        Y-offset between curves
    show_label : bool
        Show legend labels
    plot_type : str
        Plot type: 'multiple', 'single', 'single-combined'
    marker_size : int
        Symbol size
    **kwargs : dict
        Additional plotting parameters
    """
    if q_auto:
        q_range = None
    if t_auto:
        t_range = None
    if y_auto:
        y_range = None

    q, tel, g2, g2_err, qbin_labels, labels = get_g2_stability_data(
        xf_obj, q_range=q_range, t_range=t_range
    )

    # Handle case where data is not available
    if g2 is False or g2 is None:
        logger.warning("G2 stability data not available for this file")
        return

    num_figs, num_lines = compute_geometry(g2, plot_type)

    num_data, num_qval = len(g2), g2[0].shape[1]
    # col and row for the 2d layout
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    # Frame indices for color/symbol cycling
    frame_indices = np.arange(num_data)

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()

    # Handle log scale for time range
    t0_range = None
    if t_range:
        with np.errstate(divide="ignore", invalid="ignore"):
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
            legend = t.addLegend(labelTextSize="6pt")
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(0, 0))

        t.setMouseEnabled(x=False, y=y_auto)
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)

        color = colors[frame_indices[m] % color_len]
        symbol = symbols[frame_indices[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                ax = axes[n]
                label = f"frame={int(labels[m])}"
                if m == 0:
                    ax.setTitle(qbin_labels[n])
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % color_len]
                if m == 0 or n == 0:
                    ax.setTitle(str(labels[m]))
            elif plot_type == "single-combined":
                ax = axes[0]
                label = str(labels[m]) + qbin_labels[n]

            x = tel
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
            if not y_auto and y_range is not None:
                ax.setRange(yRange=y_range)
            if not t_auto and t_range is not None:
                ax.setRange(xRange=t0_range)

    return


def compute_geometry(g2, plot_type):
    """Compute subplot grid dimensions for G2 plots.

    Determines how many figures (subplots) and how many overlay
    lines per figure are needed based on the layout mode.

    Args:
        g2: List of 2-D G2 arrays, each shaped
            ``(n_delay, n_q)``.
        plot_type: Layout mode. ``"multiple"`` creates one subplot
            per Q-bin, ``"single"`` one per file, and
            ``"single-combined"`` puts everything on one axes.

    Returns:
        tuple[int, int]: ``(num_figs, num_lines)`` where *num_figs*
        is the number of subplot panels and *num_lines* is the
        number of curves per panel.

    Raises:
        ValueError: If *plot_type* is not a recognised layout mode.
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
    """Plot G2 correlation data using PyQtGraph with optional curve fitting.

    Renders multi-panel G2 plots with configurable layout, fitting overlays,
    and baseline subtraction. Supports single, double, and stretched
    exponential fitting models.

    Args:
        hdl: PyQtGraph plot handler (GraphicsLayoutWidget).
        xf_list: List of XpcsFile objects containing G2 data.
        q_range: (min, max) Q-value range in inverse Angstroms, or None.
        t_range: (min, max) delay time range in seconds, or None.
        y_range: (min, max) y-axis range for G2 values, or None.
        y_auto: If True, auto-scale y-axis.
        q_auto: If True, ignore q_range and use all Q-bins.
        t_auto: If True, ignore t_range and use all delay times.
        num_col: Number of plot columns in the grid layout.
        rows: List of file indices to plot, or None for all.
        offset: Vertical offset between datasets for visibility.
        show_fit: If True, overlay fitted curves on the data.
        show_label: If True, display legend labels.
        bounds: Fitting bounds array, shape ``(2, n_params)``.
        fit_flag: Boolean array indicating which parameters to fit.
        plot_type: Layout mode: ``'multiple'`` (one panel per Q-bin),
            ``'single'`` (one panel per file), or ``'single-combined'``.
        subtract_baseline: If True, subtract fitted baseline from data.
        marker_size: Size of data point markers in pixels.
        label_size: Font size for legend labels in points.
        fit_func: Fitting model: ``'single'`` or ``'double'`` exponential.
        robust_fitting: If True, use NLSQ multistart robust fitting.
        enable_diagnostics: If True, compute model health diagnostics.
        **kwargs: Additional keyword arguments passed to fitting routines.
            Includes ``force_refit`` (bool) to force re-fitting.

    Example:
        >>> pg_plot(hdl, xf_list, q_range=(0.01, 0.1),
        ...         t_range=(1e-4, 10), y_range=(0.9, 1.5),
        ...         show_fit=True, fit_func='single')
    """
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
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        # default base line to be 1.0; used for non-fitting or fit error cases
        baseline_offset = np.ones(num_qval)
        fit_summary = None
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
                    # Extract baseline: model is a*exp(-2x/b) + c + d, so baseline = c + d (params 2 and 3)
                    try:
                        baseline_offset = (
                            fit_summary["fit_val"][:, 0, 2]
                            + fit_summary["fit_val"][:, 0, 3]
                        )
                    except (IndexError, TypeError):
                        # Fallback to default baseline if shape doesn't match
                        baseline_offset = np.ones(num_qval)

        color = colors[rows[m] % color_len]
        symbol = symbols[rows[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                ax = axes[n]
                label = xf_list[m].label
                if m == 0:
                    ax.setTitle(labels[m][n])
            elif plot_type == "single":
                ax = axes[m]
                # overwrite color; use the same color for the same set;
                color = colors[n % color_len]
                if n == 0:
                    ax.setTitle(xf_list[m].label)
            elif plot_type == "single-combined":
                ax = axes[0]
                label = xf_list[m].label + labels[m][n]

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
                        _add_fit_param_annotation(
                            ax, fit_summary["fit_val"][n], fit_func, color
                        )


def _add_fit_param_annotation(ax, fit_val, fit_func, color):
    """Add fitted parameter text annotation to plot.

    Args:
        ax: PyQtGraph plot axis
        fit_val: Fitted values array [2, n_params] - values and errors
        fit_func: Fitting function type ('single' or 'double')
        color: Color for the text
    """
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
            if param_names[p_idx] == "τ" or param_names[p_idx] == "τ1":
                param_text_lines.append(
                    f"{param_names[p_idx]} = {value:.3e} ± {error:.2e}"
                )
            else:
                param_text_lines.append(
                    f"{param_names[p_idx]} = {value:.3f} ± {error:.3f}"
                )
        elif param_names[p_idx] == "τ" or param_names[p_idx] == "τ1":
            param_text_lines.append(f"{param_names[p_idx]} = {value:.3e} ± --")
        else:
            param_text_lines.append(f"{param_names[p_idx]} = {value:.3f} ± --")

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


def pg_plot_from_data(
    hdl,
    *,
    q,
    tel,
    g2,
    g2_err,
    labels,
    num_figs,
    fit_results=None,
    y_auto=False,
    q_auto=False,
    t_auto=False,
    num_col=4,
    rows=None,
    offset=0,
    show_fit=False,
    show_label=False,
    y_range=None,
    t_range=None,
    plot_type="multiple",
    subtract_baseline=True,
    marker_size=5,
    fit_func="single",
    **_ignored_kwargs,
):
    """Render pre-fetched G2 data without re-fetching from XpcsFile objects.

    This is the rendering-only counterpart of ``pg_plot``.  It accepts the
    data structures already extracted by the async worker (``q``, ``tel``,
    ``g2``, ``g2_err``, ``labels``) and renders them directly, avoiding the
    redundant ``get_data()`` and ``get_xf_list()`` calls that ``vk.plot_g2``
    would otherwise trigger on the main thread (BUG-014).

    Parameters mirror the ``pg_plot`` signature where applicable.
    """
    if not q or g2 is None or len(g2) == 0:
        return

    num_data = len(g2)
    num_qval = g2[0].shape[1] if g2[0].ndim > 1 else 1
    col = min(num_figs, num_col)
    row = (num_figs + col - 1) // col

    if rows is None or len(rows) == 0:
        rows = list(range(num_data))

    hdl.adjust_canvas_size(num_col=col, num_row=row)
    hdl.clear()

    t0_range = None
    if t_range and not t_auto:
        with np.errstate(divide="ignore", invalid="ignore"):
            t_range_arr = np.asarray(t_range)
            t_range_arr = np.where(t_range_arr > 0, t_range_arr, np.finfo(float).eps)
            t0_range = np.log10(t_range_arr)

    axes = []
    for n in range(num_figs):
        i_col = n % col
        i_row = n // col
        t = hdl.addPlot(row=i_row, col=i_col)
        axes.append(t)
        if show_label:
            t.addLegend(offset=(-1, 1), labelTextSize="9pt", verSpacing=-10)
        t.setMouseEnabled(x=False, y=y_auto)
        # Set axis labels once during setup (hoisted out of the m×n inner loop)
        t.setLabel("bottom", "tau (s)")
        t.setLabel("left", "g2")

    color_len = len(colors)
    symbol_len = len(symbols)
    for m in range(num_data):
        baseline_offset = np.ones(num_qval)

        # Use pre-computed fit_results from worker if available
        fit_summary = None
        if show_fit and fit_results is not None and m < len(fit_results):
            fit_summary = fit_results[m]
            if fit_summary is not None and subtract_baseline:
                try:
                    if (
                        fit_summary.get("fit_val") is not None
                        and len(fit_summary["fit_val"]) > 0
                    ):
                        baseline_offset = (
                            fit_summary["fit_val"][:, 0, 2]
                            + fit_summary["fit_val"][:, 0, 3]
                        )
                except (IndexError, TypeError, KeyError):
                    baseline_offset = np.ones(num_qval)

        color = colors[rows[m] % color_len]
        symbol = symbols[rows[m] % symbol_len]

        for n in range(num_qval):
            label = None
            if plot_type == "multiple":
                ax = axes[n]
                if m == 0:
                    ax.setTitle(labels[m][n] if labels and m < len(labels) else "")
            elif plot_type == "single":
                ax = axes[m]
                color = colors[n % color_len]
                if n == 0:
                    ax.setTitle(labels[m][0] if labels and m < len(labels) else "")
            elif plot_type == "single-combined":
                ax = axes[0]
                label = labels[m][n] if labels and m < len(labels) else ""
            else:
                ax = axes[n % len(axes)]

            x = tel[m]
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

            if not y_auto and y_range is not None:
                ax.setRange(yRange=y_range)
            if not t_auto and t0_range is not None:
                ax.setRange(xRange=t0_range)

            if show_fit and fit_summary is not None:
                try:
                    if (
                        fit_summary.get("fit_line") is not None
                        and n < fit_summary["fit_line"].shape[0]
                        and fit_summary.get("fit_x") is not None
                    ):
                        y_fit = fit_summary["fit_line"][n] + m * offset
                        y_fit = y_fit - baseline_offset[n] + 1.0
                        fit_x = fit_summary["fit_x"]
                        ax.plot(fit_x, y_fit, pen=pg.mkPen(color, width=2.5))
                        if (
                            fit_summary.get("fit_val") is not None
                            and n < fit_summary["fit_val"].shape[0]
                        ):
                            _add_fit_param_annotation(
                                ax, fit_summary["fit_val"][n], fit_func, color
                            )
                except (IndexError, KeyError, TypeError):
                    pass


def pg_plot_one_g2(ax, x, y, dy, color, label, symbol, symbol_size=5):
    """Plot a single G2 correlation curve with error bars on a log-x axis.

    Filters NaN/inf values, applies log-scale x-axis, and downsamples
    dense data for rendering performance.

    Args:
        ax: PyQtGraph PlotItem to draw on.
        x: Delay times array (must be positive for log scale).
        y: G2 correlation values.
        dy: G2 error bars (standard deviation).
        color: RGB tuple for plot color, e.g. ``(255, 0, 0)``.
        label: Legend label string, or None.
        symbol: PyQtGraph symbol character (e.g. ``'o'``, ``'s'``).
        symbol_size: Marker size in pixels (default 5).
    """
    # Ensure NumPy arrays at PyQtGraph I/O boundary (JAX arrays not supported)
    from xpcsviewer.backends._conversions import ensure_numpy

    x = ensure_numpy(x)
    y = ensure_numpy(y)
    dy = ensure_numpy(dy)

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
        line = pg.ErrorBarItem(
            x=x_clean, y=y_clean, top=dy_clean, bottom=dy_clean, pen=pen_line
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
    # Ensure host-resident NumPy arrays before raw np.* operations.
    g2_data = ensure_numpy(g2_data)
    baseline_values = ensure_numpy(baseline_values)
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
    if not g2_data_list:
        return []

    # Ensure host-resident arrays; np.max/mean/std don't accept JAX arrays.
    arrays = [ensure_numpy(a) for a in g2_data_list]

    # Vectorized fast path for 'max' normalization with small-to-medium batch sizes
    # (B ≤ 100). For very large B with small T×Q the stack allocation itself becomes
    # the bottleneck, so fall back to the per-item loop in that case.
    # For 'std' and 'mean', the per-item loop is retained as it avoids materializing
    # a potentially large 3-D stack when the savings are smaller.
    n = len(arrays)
    use_vectorized = n <= 100 and method == "max"

    if use_vectorized:
        # Stack → single kernel for max + divide. The stack is unavoidable but
        # eliminates 2*n separate NumPy kernel dispatches (one max + one divide per item).
        g2_stack = np.stack(arrays, axis=0)  # [B, T, Q]
        scale = np.max(g2_stack, axis=1, keepdims=True)  # [B, 1, Q]
        scale = np.where(scale == 0, 1.0, scale)
        result = g2_stack / scale  # [B, T, Q]
        # Return list of views (no extra copy) for backward compatibility.
        return list(result)

    # Per-item path: handles 'mean', 'std', very large B, or non-uniform shapes.
    normalized_data = []
    for g2_data in arrays:
        if method == "max":
            max_vals = np.max(g2_data, axis=0, keepdims=True)
            max_vals = np.where(max_vals == 0, 1.0, max_vals)
            normalized = g2_data / max_vals
        elif method == "mean":
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            mean_vals = np.where(mean_vals == 0, 1.0, mean_vals)
            normalized = g2_data / mean_vals
        elif method == "std":
            mean_vals = np.mean(g2_data, axis=0, keepdims=True)
            std_vals = np.std(g2_data, axis=0, keepdims=True)
            std_vals = np.where(std_vals == 0, 1.0, std_vals)
            normalized = (g2_data - mean_vals) / std_vals
        else:
            normalized = g2_data
        normalized_data.append(normalized)

    return normalized_data


def compute_g2_ensemble_statistics(g2_data_list, include_median: bool = False):
    """
    Compute ensemble statistics for multiple G2 datasets using vectorized operations.

    Args:
        g2_data_list: List of G2 data arrays [time, q_values]
        include_median: If True, compute and include `ensemble_median` in the result.
            This adds an O(B·T·Q·log B) partial-sort cost on top of the O(B·T·Q) mean/std
            computation. Defaults to False for performance. Set True only when the caller
            genuinely needs the median statistic.

    Returns:
        Dictionary with ensemble statistics. Always contains:
            ensemble_mean, ensemble_std, ensemble_min, ensemble_max,
            ensemble_var, q_mean_values, temporal_correlation.
        Optionally contains:
            ensemble_median (when include_median=True).
    """
    # Ensure host-resident NumPy arrays before stacking / raw np.* statistics.
    g2_data_list = [ensure_numpy(arr) for arr in g2_data_list]
    # Stack all data for vectorized operations
    g2_stack = np.stack(g2_data_list, axis=0)  # [batch, time, q_values]
    # Vectorized statistical computations — O(B·T·Q) operations only.
    # np.median is intentionally excluded from the default path: it requires a
    # full O(B·T·Q·log B) partial sort and accounts for 85% of this function's
    # runtime (see tests/reports/baseline_profile.md).
    stats = {
        "ensemble_mean": np.mean(g2_stack, axis=0),
        "ensemble_std": np.std(g2_stack, axis=0),
        "ensemble_min": np.min(g2_stack, axis=0),
        "ensemble_max": np.max(g2_stack, axis=0),
        "ensemble_var": np.var(g2_stack, axis=0),
        "q_mean_values": np.mean(g2_stack, axis=(0, 1)),  # Mean across time and batch
    }

    if include_median:
        stats["ensemble_median"] = np.median(g2_stack, axis=0)

    # Batched temporal correlation: transpose to (q, batch, time) and compute
    # correlation matrices for all q-values without a Python loop.
    # Each q-slice is (batch, time); corrcoef produces (batch, batch).
    q_transposed = np.transpose(g2_stack, (2, 0, 1))  # [q, batch, time]

    # Vectorized batched corrcoef: center, normalize, matmul
    # mean across time axis (last)
    q_mean = np.mean(q_transposed, axis=2, keepdims=True)  # [q, batch, 1]
    q_centered = q_transposed - q_mean  # [q, batch, time]
    # Standard deviation per (q, batch)
    q_std = np.sqrt(np.sum(q_centered**2, axis=2, keepdims=True))  # [q, batch, 1]
    # Avoid division by zero
    q_std = np.where(q_std == 0, 1.0, q_std)
    q_normed = q_centered / q_std  # [q, batch, time]
    # Batched correlation: (q, batch, time) @ (q, time, batch) -> (q, batch, batch)
    n_time = q_transposed.shape[2]
    corr_batch = np.matmul(q_normed, np.transpose(q_normed, (0, 2, 1))) / n_time

    # Return as 3-D ndarray for efficiency; callers that need per-q slices
    # can index directly (corr_batch[q]) without rebuilding a Python list.
    stats["temporal_correlation"] = corr_batch  # shape [num_q, batch, batch]

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
    # Ensure host-resident NumPy arrays; np.abs/power don't accept JAX arrays
    # without triggering an untraced host-device transfer.
    g2_data = ensure_numpy(g2_data)
    g2_errors = ensure_numpy(g2_errors)
    propagated_errors = g2_errors.copy()

    for op in operations:
        if op["type"] == "scale":
            # Error propagation for scaling: sigma_new = |scale| * sigma_old
            scale_factor = op["factor"]
            propagated_errors *= np.abs(scale_factor)

        elif op["type"] == "offset":
            # Error propagation for offset: sigma_new = sigma_old (additive operations don't change uncertainty)
            pass

        elif op["type"] == "power":
            # Error propagation for power: sigma_new = |n * x^(n-1)| * sigma_old
            power = op["power"]
            propagated_errors = (
                np.abs(power * np.power(g2_data, power - 1)) * propagated_errors
            )

        elif op["type"] == "log":
            # Error propagation for logarithm: sigma_new = sigma_old / |x|
            propagated_errors = propagated_errors / np.abs(g2_data)

    return propagated_errors


def vectorized_g2_interpolation(tel, g2_data, target_tel):
    """
    Vectorized interpolation of G2 data to new time points.

    Uses JAX vmap + interpax for batch interpolation when available,
    falling back to a single scipy interp1d call with 2D y otherwise.

    Args:
        tel: Original time points
        g2_data: G2 data [time, q_values]
        target_tel: Target time points for interpolation

    Returns:
        Interpolated G2 data
    """
    try:
        import interpax
        import jax
        import jax.numpy as jnp

        # Convert to JAX arrays once
        tel_jax = jnp.asarray(tel)
        target_jax = jnp.asarray(target_tel)
        g2_jax = jnp.asarray(g2_data)  # [time, q_values]

        # Define single-column interpolation function
        def _interp_single_q(y_col):
            return interpax.interp1d(
                target_jax, tel_jax, y_col, method="cubic", extrap=True
            )

        # vmap over columns (q-values): in_axes=1, out_axes=1
        interpolated_jax = jax.vmap(_interp_single_q, in_axes=1, out_axes=1)(g2_jax)

        from xpcsviewer.backends._conversions import ensure_numpy

        return ensure_numpy(interpolated_jax)

    except ImportError:
        # NumPy fallback: use project's Interp1d wrapper (avoids direct scipy import)
        from xpcsviewer.backends.scipy_replacements.interpolate import Interp1d

        # Interpolate each q-column using the wrapper
        result = np.empty((len(target_tel), g2_data.shape[1]))
        for q_idx in range(g2_data.shape[1]):
            interp_func = Interp1d(
                tel,
                g2_data[:, q_idx],
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            result[:, q_idx] = interp_func(target_tel)
        return result
