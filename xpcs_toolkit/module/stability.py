# Third-party imports

# Local imports
from xpcs_toolkit.utils.logging_config import get_logger

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
    logger.info(f"Starting stability plot for {fc.label}")
    logger.debug(
        f"Plot parameters: plot_type={plot_type}, plot_norm={plot_norm}, loc='{loc}'"
    )

    pg_hdl.clear()
    plot_item = pg_hdl.getPlotItem()

    plot_item.setTitle(fc.label)
    legend = plot_item.addLegend()
    anchor_param = get_pyqtgraph_anchor_params(loc, padding=15)
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
