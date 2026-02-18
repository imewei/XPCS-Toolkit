"""
2D SAXS scattering pattern visualization.

Provides visualization of integrated 2D small-angle X-ray scattering patterns
with support for log/linear scaling, colormap selection, and image rotation.

Functions:
    plot: Display 2D SAXS pattern with beam center overlay.
"""

from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


def plot(
    xfile,
    pg_hdl=None,
    plot_type="log",
    cmap="jet",
    rotate=False,
    autolevel=False,
    autorange=False,
    vmin=None,
    vmax=None,
):
    """Display a 2-D SAXS detector image in a PyQtGraph ImageView.

    Renders the detector image from *xfile* with optional log scaling,
    custom colour mapping, and intensity clamping. A beam-centre ROI
    marker is drawn automatically.

    Args:
        xfile: XpcsFile containing ``saxs_2d`` and ``saxs_2d_log``
            image arrays plus beam centre coordinates (``bcx``,
            ``bcy``).
        pg_hdl: PyQtGraph ImageView handle for rendering.
        plot_type: Image intensity scaling. ``"log"`` uses the
            pre-computed log-scaled image; any other value uses the
            linear image.
        cmap: Colour-map name passed to ``pg_hdl.set_colormap``.
            ``None`` keeps the current colour map.
        rotate: If True, transpose the image (unused in current
            implementation but returned for caller state tracking).
        autolevel: Automatically scale intensity levels to the
            data range.
        autorange: Force the view to fit the full image. When
            False the previous view range is preserved unless the
            image shape changed.
        vmin: Manual lower intensity clamp. Applied only when
            *autolevel* is False.
        vmax: Manual upper intensity clamp. Applied only when
            *autolevel* is False.

    Returns:
        bool: The *rotate* flag, echoed back for caller convenience.

    Example:
        >>> rotate = plot(xfile, pg_hdl=viewer, plot_type="log",
        ...               cmap="viridis", vmin=0, vmax=100)
    """
    logger.info(f"Starting SAXS2D plot for {getattr(xfile, 'label', 'unknown file')}")
    logger.debug(
        f"Plot parameters: plot_type='{plot_type}', cmap='{cmap}', rotate={rotate}, autolevel={autolevel}, autorange={autorange}"
    )

    center = (xfile.bcx, xfile.bcy)
    img = xfile.saxs_2d_log if plot_type == "log" else xfile.saxs_2d

    logger.debug(f"Image data: shape={img.shape}, center=({xfile.bcx}, {xfile.bcy})")

    if cmap is not None:
        pg_hdl.set_colormap(cmap)

    prev_img = pg_hdl.image
    shape_changed = prev_img is None or prev_img.shape != img.shape
    do_autorange = autorange or shape_changed

    # Save view range if keeping it
    if not do_autorange:
        view_range = pg_hdl.view.viewRange()

    # Set new image - ensure NumPy at PyQtGraph boundary
    pg_hdl.setImage(ensure_numpy(img), autoLevels=autolevel, autoRange=do_autorange)

    # Restore view range if we skipped auto-ranging
    if not do_autorange:
        pg_hdl.view.setRange(xRange=view_range[0], yRange=view_range[1], padding=0)

    # Restore levels if needed
    if not autolevel and vmin is not None and vmax is not None:
        pg_hdl.setLevels(vmin, vmax)

    # Restore intensity levels (if needed)
    if not autolevel and vmin is not None and vmax is not None:
        pg_hdl.setLevels(vmin, vmax)

    if center is not None:
        pg_hdl.add_roi(sl_type="Center", center=center, label="Center")

    logger.info("SAXS2D plot completed successfully")
    return rotate
