import numpy as np

from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_min_max(data, min_percent=0, max_percent=100, **kwargs):
    logger.debug(
        f"Calculating min/max: min_percent={min_percent}, max_percent={max_percent}"
    )
    vmin = np.percentile(data.ravel(), min_percent)
    vmax = np.percentile(data.ravel(), max_percent)
    logger.debug(f"Percentile values: vmin={vmin}, vmax={vmax}")
    if "plot_norm" in kwargs and "plot_type" in kwargs and kwargs["plot_norm"] == 3:
        if kwargs["plot_type"] == "log":
            t = max(abs(vmin), abs(vmax))
            vmin, vmax = -t, t
        else:
            t = max(abs(1 - vmin), abs(vmax - 1))
            vmin, vmax = 1 - t, 1 + t

    return vmin, vmax


def norm_saxs_data(Iq, q, plot_norm=0):
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


def create_slice(arr, x_range):
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
