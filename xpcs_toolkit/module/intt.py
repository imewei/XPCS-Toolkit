import contextlib

import numpy as np
import pyqtgraph as pg

from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)

colors = [
    (192, 0, 0),
    (0, 176, 80),
    (0, 32, 96),
    (255, 0, 0),
    (0, 176, 240),
    (0, 32, 96),
    (255, 164, 0),
    (146, 208, 80),
    (0, 112, 192),
    (112, 48, 160),
    (54, 96, 146),
    (150, 54, 52),
    (118, 147, 60),
    (96, 73, 122),
    (49, 134, 155),
    (226, 107, 10),
]


def smooth_data(fc, window=1, sampling=1):
    # some bad frames have both x and y = 0;
    # x, y = fc.Int_t[0], fc.Int_t[1]
    y = fc.Int_t[1]
    x = np.arange(y.shape[0], dtype=np.uint32)  # Use appropriate dtype

    if window > 1:
        # More memory-efficient moving average using numpy.convolve
        # This avoids creating the full cumsum array
        if window <= y.shape[0]:
            # Use valid convolution mode to avoid edge effects
            y = np.convolve(y, np.ones(window) / window, mode="valid")
            x = x[window - 1 :]  # Adjust x indices for valid convolution

    if sampling >= 2:
        # Combined slicing operation
        y = y[::sampling]
        x = x[::sampling]

    return x, y


def plot(xf_list, pg_hdl, enable_zoom=True, xlabel="Frame Index", **kwargs):
    """
    Optimized intensity plotting with improved data handling and performance.
    :param xf_list: list of xf objects
    :param pg_hdl: pyqtgraph handler to plot
    :param enable_zoom: bool, if to plot the zoom view or not
    :param xlabel:
    :param kwargs: used to define how to average/sample the data
    :return:
    """
    logger.info(f"Starting intensity plot for {len(xf_list)} files")
    logger.debug(
        f"Plot parameters: enable_zoom={enable_zoom}, xlabel='{xlabel}', kwargs={kwargs}"
    )
    # Pre-process all data and validate
    data = []
    valid_files = []

    for fc in xf_list:
        try:
            # Filter kwargs to only include parameters that smooth_data accepts
            smooth_kwargs = {
                k: v for k, v in kwargs.items() if k in ["window", "sampling"]
            }
            x, y = smooth_data(fc, **smooth_kwargs)

            # Apply time scaling if needed
            if xlabel != "Frame Index":
                x = x * fc.t0

            # Validate data
            if len(x) > 0 and len(y) > 0:
                data.append([x, y])
                valid_files.append(fc)
            else:
                logger.warning(f"No valid data for file {fc.label}")

        except Exception as e:
            logger.error(f"Failed to process file {fc.label}: {e}")
            continue

    if not data:
        logger.warning("No valid data to plot")
        return

    logger.debug(f"Successfully processed {len(data)} files for plotting")
    pg_hdl.clear()

    # Create plot layout with optimized settings
    t = pg_hdl.addPlot(colspan=2)
    t.addLegend(offset=(-1, 1), labelTextSize="8pt", verSpacing=-10)
    tf = pg_hdl.addPlot(row=1, col=0, title="Fourier Spectrum")
    tf.addLegend(offset=(-1, 1), labelTextSize="8pt", verSpacing=-10)
    tf.setLabel("bottom", "Frequency (Hz)")
    tf.setLabel("left", "FFT Intensity")

    tz = pg_hdl.addPlot(row=1, col=1, title="Zoom In")
    tz.addLegend(offset=(-1, 1), labelTextSize="8pt", verSpacing=-10)

    # Enable downsampling for better performance with large datasets
    t.setDownsampling(mode="peak")
    tf.setDownsampling(mode="peak")
    tz.setDownsampling(mode="peak")

    # Plot main intensity data
    for n, (x, y) in enumerate(data):
        color = colors[n % len(colors)]
        t.plot(
            x,
            y,
            pen=pg.mkPen(color, width=1),
            name=valid_files[n].label,
        )
    t.setTitle(f"Intensity vs {xlabel}")
    t.setLabel("bottom", xlabel)
    t.setLabel("left", "Intensity (cts / pixel)")

    # Setup zoom functionality if enabled
    lr = None
    if enable_zoom and len(data) > 0:
        try:
            x_data = data[0][0]
            if len(x_data) > 1:
                vmin, vmax = np.min(x_data), np.max(x_data)
                cen = vmin * 0.382 + vmax * 0.618
                width = (vmax - vmin) * 0.05
                lr = pg.LinearRegionItem([cen - width, cen + width])
                t.addItem(lr)
        except Exception as e:
            logger.warning(f"Failed to setup zoom: {e}")
            enable_zoom = False

    # Plot FFT data using optimized cached method
    for n, fc in enumerate(valid_files):
        try:
            # Use the optimized Int_t_fft property (now cached)
            x_fft, y_fft = fc.Int_t_fft
            color = colors[n % len(colors)]
            tf.plot(x_fft, y_fft, pen=pg.mkPen(color, width=1), name=fc.label)
        except Exception as e:
            logger.error(f"Failed to plot FFT for {fc.label}: {e}")
            continue

    # Plot zoom data
    for n, (x, y) in enumerate(data):
        color = colors[n % len(colors)]
        tz.plot(x, y, pen=pg.mkPen(color, width=1), name=valid_files[n].label)

    # Setup zoom callbacks
    if enable_zoom and lr is not None:

        def update_plot():
            with contextlib.suppress(Exception):
                tz.setXRange(*lr.getRegion(), padding=0)

        def update_region():
            with contextlib.suppress(Exception):
                lr.setRegion(tz.getViewBox().viewRange()[0])

        lr.sigRegionChanged.connect(update_plot)
        tz.sigXRangeChanged.connect(update_region)
        update_plot()

    tz.setLabel("bottom", xlabel)
    tz.setLabel("left", "Intensity (cts / pixel)")

    logger.info("Intensity plot completed successfully")
    return
