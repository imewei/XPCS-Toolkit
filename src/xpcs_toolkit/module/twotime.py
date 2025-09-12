import numpy as np
import pyqtgraph as pg
from ..xpcs_file import MemoryMonitor
from xpcs_toolkit.utils.logging_config import get_logger

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

    hdl["saxs"].setImage(np.flipud(saxs))
    hdl["dqmap"].setImage(dqmap_disp)

    # Monitor memory before loading potentially large c2 data
    if MemoryMonitor.is_memory_pressure_high(0.8):
        logger.warning(
            f"High memory pressure detected before loading c2 data for {xfile.label}"
        )

    c2_result = xfile.get_twotime_c2(selection=selection, correct_diag=correct_diag)
    if c2_result is None:
        return None

    c2, delta_t = c2_result["c2_mat"], c2_result["delta_t"]

    # Log memory usage after loading c2 data
    c2_memory_mb = MemoryMonitor.estimate_array_memory(c2.shape, c2.dtype)
    logger.debug(f"Loaded c2 data ({c2.shape}), estimated size: {c2_memory_mb:.1f}MB")

    hdl["tt"].imageItem.setScale(delta_t)
    hdl["tt"].setImage(c2, autoRange=True)

    cmap = pg.colormap.getFromMatplotlib(cmap)
    hdl["tt"].setColorMap(cmap)
    hdl["tt"].ui.histogram.setHistogramRange(mn=0, mx=3)
    if not autolevel and vmin is not None and vmax is not None:
        hdl["tt"].setLevels(min=vmin, max=vmax)
    else:
        vmin, vmax = np.percentile(c2, [0.5, 99.5])
        hdl["tt"].setLevels(min=vmin, max=vmax)
    plot_twotime_g2(hdl, c2_result)


def plot_twotime_g2(hdl, c2_result):
    g2_full, g2_partial = c2_result["g2_full"], c2_result["g2_partial"]

    hdl["c2g2"].clear()
    hdl["c2g2"].setLabel("left", "g2")
    hdl["c2g2"].setLabel("bottom", "t (s)")
    acquire_period = c2_result["acquire_period"]

    xaxis = np.arange(g2_full.size) * acquire_period
    hdl["c2g2"].plot(
        x=xaxis[1:],
        y=g2_full[1:],
        pen=pg.mkPen(color=PG_COLORS[-1], width=4),
        name="g2_full",
    )
    for n in range(g2_partial.shape[0]):
        xaxis = np.arange(g2_partial.shape[1]) * acquire_period
        hdl["c2g2"].plot(
            x=xaxis[1:],
            y=g2_partial[n][1:],
            pen=pg.mkPen(color=PG_COLORS[n], width=1),
            name=f"g2_partial_{n}",
        )
    hdl["c2g2"].setLogMode(x=True, y=False)
    hdl["c2g2"].autoRange()
