"""
Sample stability monitoring module.

Analyzes temporal stability of XPCS measurements by comparing intensity
profiles across time sections. Helps identify beam damage, sample drift,
or other instabilities.

Functions:
    plot: Generate stability comparison plots.
"""
# Third-party imports

# Local imports
from xpcsviewer.utils.logging_config import get_logger

from .saxs1d import get_pyqtgraph_anchor_params, plot_line_with_marker

logger = get_logger(__name__)


def plot(
    fc,
    pg_hdl,
    plot_type=2,
    plot_norm=0,
    legend=None,
    title=None,
    loc="upper right",
    **kwargs,
):
    """Generate a stability comparison plot for an XPCS measurement.

    Plots partial SAXS-1D profiles from different time sections of the
    measurement to visualize temporal stability (beam damage, drift, etc.).

    Args:
        fc: XpcsFile object containing partial SAXS-1D data.
        pg_hdl: PyQtGraph PlotWidget to render into.
        plot_type: Axis scaling mode. Bit 0 controls log-x, bit 1 controls
            log-y: 0=linear-linear, 1=log-linear, 2=linear-log, 3=log-log.
        plot_norm: Normalization method index:
            0=none, 1=q^2, 2=q^4, 3=I(0).
        legend: Unused (legend is created internally).
        title: Unused (title is set from ``fc.label``).
        loc: Legend anchor position string, e.g. ``'upper right'``.
        **kwargs: Additional keyword arguments (currently unused).

    Example:
        >>> from xpcsviewer.module import stability
        >>> stability.plot(xf, pg_widget, plot_type=3, plot_norm=1)
    """
    logger.info(f"Starting stability plot for {fc.label}")
    logger.debug(
        f"Plot parameters: plot_type={plot_type}, plot_norm={plot_norm}, loc='{loc}'"
    )

    pg_hdl.clear()
    plot_item = pg_hdl.getPlotItem()

    plot_item.setTitle(fc.label)
    legend = plot_item.addLegend()
    anchor_param = get_pyqtgraph_anchor_params(loc, padding=15)
    if anchor_param:
        legend.anchor(**anchor_param)

    norm_method = [None, "q2", "q4", "I0"][plot_norm]
    log_x = (False, True)[plot_type % 2]
    log_y = (False, True)[plot_type // 2]
    plot_item.setLogMode(x=log_x, y=log_y)

    q, Iqp, xlabel, ylabel = fc.get_saxs1d_data(
        target="saxs1d_partial", norm_method=norm_method
    )
    logger.debug(
        f"Retrieved SAXS1D data: q shape={q.shape}, Iqp shape={Iqp.shape}, norm_method='{norm_method}'"
    )

    for n in range(Iqp.shape[0]):
        plot_line_with_marker(
            plot_item,
            q,
            Iqp[n],
            n,
            f"p{n}",  # label
            1.0,  # alpha
            marker_size=6,
            log_x=log_x,
            log_y=log_y,
        )

    plot_item.setLabel("bottom", xlabel)
    plot_item.setLabel("left", ylabel)
    plot_item.showGrid(x=True, y=True, alpha=0.3)

    logger.info("Stability plot completed successfully")
