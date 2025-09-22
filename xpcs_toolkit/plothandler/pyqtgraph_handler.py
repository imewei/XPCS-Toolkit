import numpy as np
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget, ImageView, QtCore, QtGui

from xpcs_toolkit.utils.logging_config import get_logger

from .qt_signal_fixes import (
    configure_pyqtgraph_for_qt_compatibility,
    qt_connection_context,
)

logger = get_logger(__name__)

# Configure PyQtGraph for Qt compatibility to reduce connection warnings
configure_pyqtgraph_for_qt_compatibility()

pg.setConfigOptions(imageAxisOrder="row-major")


class ImageViewPlotItem(ImageView):
    # ImageView only supports x/y tics when view is a PlotItem
    def __init__(self, *args, **kwargs) -> None:
        with qt_connection_context():
            plot_item = pg.PlotItem()
            super().__init__(*args, view=plot_item, **kwargs)


class ImageViewDev(ImageView):
    def __init__(self, *args, **kwargs) -> None:
        with qt_connection_context():
            super().__init__(*args, **kwargs)
        self.roi_record = {}
        self.roi_idx = 0
        logger.debug("ImageViewDev initialized")

    def reset_limits(self):
        """
        reset the viewbox's limits so updating image won't break the layout;
        """
        self.view.state["limits"] = {
            "xLimits": [None, None],
            "yLimits": [None, None],
            "xRange": [None, None],
            "yRange": [None, None],
        }

    def set_colormap(self, cmap):
        logger.debug(f"Setting colormap to '{cmap}'")
        pg_cmap = pg.colormap.getFromMatplotlib(cmap)
        self.setColorMap(pg_cmap)

    def clear(self):
        logger.debug("Clearing ImageViewDev")
        super().clear()

        self.remove_rois()
        self.reset_limits()
        # Safely disconnect signal if it has connections
        try:
            if hasattr(self, "scene") and self.scene is not None:
                if hasattr(self.scene, "sigMouseMoved"):
                    # Check if signal has any connections before disconnecting
                    signal = self.scene.sigMouseMoved
                    if hasattr(signal, "receivers") and signal.receivers(signal) > 0:
                        signal.disconnect()
        except (RuntimeError, TypeError, AttributeError):
            # Signal already disconnected or doesn't exist
            pass

    def add_roi(
        self,
        cen=None,
        num_edges=None,
        radius=60,
        color="r",
        sl_type="Pie",
        width=3,
        sl_mode="exclusive",
        second_point=None,
        label=None,
        center=None,
    ):
        # Handle parameter consolidation: 'center' takes precedence over 'cen'
        if center is not None:
            cen = center

        # label: label of roi; default is None, which is for roi-draw
        logger.debug(
            f"ImageViewDev: Adding ROI: type='{sl_type}', label='{label}', cen={cen}, radius={radius}"
        )
        logger.debug(
            f"ImageViewDev: ROI parameters: color='{color}', width={width}, sl_mode='{sl_mode}'"
        )

        if label is not None and label in self.roi_record:
            self.remove_roi(label)

        if sl_mode == "inclusive":
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        kwargs = {"pen": pen, "removable": True, "hoverPen": pen, "handlePen": pen}

        if sl_type == "Circle":
            if cen is None:
                logger.warning("Cannot add Circle ROI: center coordinates not provided")
                return None
            if second_point is not None:
                radius = np.sqrt(
                    (second_point[1] - cen[1]) ** 2 + (second_point[0] - cen[0]) ** 2
                )
            try:
                new_roi = pg.CircleROI(
                    pos=[cen[0] - radius, cen[1] - radius],
                    radius=radius,
                    movable=False,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Failed to create Circle ROI: {e}")
                return None

        elif sl_type == "Line":
            if cen is None or second_point is None:
                logger.warning(
                    f"Cannot add Line ROI: center ({cen}) or second_point ({second_point}) not provided"
                )
                return None
            width = kwargs.pop("width", 1)
            new_roi = pg.LineROI(cen, second_point, width, **kwargs)
        elif sl_type == "Pie":
            if cen is None:
                logger.warning("Cannot add Pie ROI: center coordinates not provided")
                return None
            width = kwargs.pop("width", 1)
            try:
                new_roi = PieROI(cen, radius, movable=False, **kwargs)
            except Exception as e:
                logger.error(f"Failed to create Pie ROI: {e}")
                return None
        elif sl_type == "Q-Wedge":
            # Q-Wedge is a wedge-shaped ROI for Q-space analysis (similar to Pie)
            logger.debug("ImageViewDev: Creating Q-Wedge ROI")
            if cen is None:
                logger.warning(
                    "Cannot add Q-Wedge ROI: center coordinates not provided"
                )
                return None
            width = kwargs.pop("width", 1)
            logger.debug(
                f"ImageViewDev: Q-Wedge parameters: cen={cen}, radius={radius}, width={width}"
            )
            try:
                # Validate parameters before creating PieROI
                if not isinstance(cen, (tuple, list)) or len(cen) != 2:
                    logger.error(
                        f"Invalid center coordinates: {cen} (expected tuple/list of length 2)"
                    )
                    return None
                if not isinstance(radius, (int, float)) or radius <= 0:
                    logger.error(f"Invalid radius: {radius} (expected positive number)")
                    return None

                logger.debug(
                    f"ImageViewDev: Creating PieROI with pos={cen}, size={radius}"
                )
                new_roi = PieROI(cen, radius, movable=False, **kwargs)
                logger.debug(
                    f"ImageViewDev: Q-Wedge PieROI created successfully, ROI state: {new_roi.getState()}"
                )
            except Exception as e:
                logger.error(f"Failed to create Q-Wedge ROI: {e}")
                logger.debug(
                    f"ImageViewDev: Q-Wedge creation exception details: {type(e).__name__}: {e}"
                )
                import traceback

                logger.debug(f"ImageViewDev: Full traceback: {traceback.format_exc()}")
                return None
        elif sl_type == "Phi-Ring":
            # Phi-Ring is a ring-shaped ROI for angular analysis (similar to Circle)
            if cen is None:
                logger.warning(
                    "Cannot add Phi-Ring ROI: center coordinates not provided"
                )
                return None
            if second_point is not None:
                radius = np.sqrt(
                    (second_point[1] - cen[1]) ** 2 + (second_point[0] - cen[0]) ** 2
                )
            try:
                new_roi = pg.CircleROI(
                    pos=[cen[0] - radius, cen[1] - radius],
                    radius=radius,
                    movable=False,
                    **kwargs,
                )
            except Exception as e:
                logger.error(f"Failed to create Phi-Ring ROI: {e}")
                return None
        elif sl_type == "Center":
            if cen is None:
                logger.warning("Cannot add Center ROI: center coordinates not provided")
                return None
            new_roi = pg.ScatterPlotItem()
            new_roi.addPoints(x=[cen[0]], y=[cen[1]], symbol="+", size=15)
        else:
            raise TypeError(f"type not implemented. {sl_type}")

        try:
            logger.debug("ImageViewDev: Setting ROI properties and adding to plot")
            new_roi.sl_mode = sl_mode

            if label is None:
                label = f"roi_{self.roi_idx:06d}"
                self.roi_idx += 1

            logger.debug(f"ImageViewDev: ROI label assigned: '{label}'")
            self.roi_record[label] = new_roi
            logger.debug(
                f"ImageViewDev: ROI added to roi_record, total ROIs: {len(self.roi_record)}"
            )

            self.addItem(new_roi)
            logger.debug("ImageViewDev: ROI added to plot via addItem()")

            if sl_type != "Center":
                new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(label))
                logger.debug(f"ImageViewDev: Connected remove signal for ROI '{label}'")

            logger.debug(
                f"ImageViewDev: Successfully added ROI: type='{sl_type}', label='{label}'"
            )
            return label

        except Exception as e:
            logger.error(f"ImageViewDev: Failed to add ROI to plot: {e}")
            logger.debug(f"ImageViewDev: Exception details: {type(e).__name__}: {e}")
            # Cleanup if ROI was partially created
            if label and label in self.roi_record:
                del self.roi_record[label]
                logger.debug(
                    f"ImageViewDev: Cleaned up partial ROI '{label}' from roi_record"
                )
            return None

    def remove_rois(self, filter_str=None):
        # if filter_str is None; then remove all rois
        keys = list(self.roi_record.keys()).copy()
        if filter_str is not None:
            keys = list(filter(lambda x: x.startswith(filter_str), keys))
        for key in keys:
            self.remove_roi(key)

    def remove_roi(self, roi_key):
        t = self.roi_record.pop(roi_key, None)
        if t is not None:
            self.removeItem(t)

    def get_roi_list(self):
        parameter = []
        for key, roi in self.roi_record.items():
            if key == "Center":
                continue
            if key.startswith("RingB"):
                temp = {
                    "sl_type": "Ring",
                    "radius": (
                        roi.getState()["size"][1] / 2.0,
                        self.roi_record["RingA"].getState()["size"][1] / 2.0,
                    ),
                }
                parameter.append(temp)
            elif key.startswith("roi"):
                temp = roi.get_parameter()
                parameter.append(temp)
        return parameter


class PlotWidgetDev(GraphicsLayoutWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setBackground("w")

    def adjust_canvas_size(self, num_col, num_row):
        t = self.parent().parent().parent()
        aspect = 1 / 1.618 if t is None else t.height() / self.width()

        min_size = t.height() - 20
        width = self.width()
        canvas_size = max(min_size, int(width / num_col * aspect * num_row))
        self.setMinimumSize(QtCore.QSize(0, canvas_size))


class PieROI(pg.ROI):
    r"""
    Equilateral triangle ROI subclass with one scale handle and one rotation handle.
    Arguments
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (float) The length of an edge of the triangle.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, pos, size, **args):
        # pos is the center of the pie ROI, but pg.ROI expects top-left corner
        # Convert center position to top-left corner position
        top_left = (pos[0] - size / 2.0, pos[1] - size / 2.0)
        logger.debug(
            f"PieROI: __init__ called with pos={pos}, size={size}, converted to top_left={top_left}"
        )
        try:
            pg.ROI.__init__(self, top_left, [size, size], aspectLocked=False, **args)
            logger.debug("PieROI: pg.ROI.__init__ completed successfully")
        except Exception as e:
            logger.error(f"PieROI: Failed in pg.ROI.__init__: {e}")
            raise
        # _updateView is a rendering method inherited; used here to force
        # update the view
        try:
            self.sigRegionChanged.connect(self._updateView)
            logger.debug("PieROI: Connected sigRegionChanged signal")

            self.poly = None
            self.half_angle = None
            self.create_poly()
            logger.debug("PieROI: Created polygon shape")

            self.addScaleRotateHandle([1.0, 0], [0, 0.5])
            self.addScaleHandle([1.0, 1.0], [0, 0.5])
            logger.debug("PieROI: Added handles, initialization complete")
        except Exception as e:
            logger.error(f"PieROI: Failed during signal/handle setup: {e}")
            raise

    def create_poly(self, width=1.0, height=1.0):
        radius = np.hypot(width, height / 2.0)
        max_angle = np.arcsin(height / 2.0 / radius)
        angle = np.linspace(-max_angle, max_angle, 16)
        self.half_angle = np.rad2deg(max_angle)

        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        # make the x[0] and x[-1] two vertices at x = width after scaling
        x = x / np.abs(x[0])
        # make the y range to be height after scaling
        y = y / (np.max(y) - np.min(y)) + 0.5
        poly = QtGui.QPolygonF()
        poly.append(QtCore.QPointF(0.0, 0.5))
        for pt in zip(x, y, strict=False):
            poly.append(QtCore.QPointF(*pt))
        self.poly = None
        self.poly = poly

    def get_parameter(self):
        state = self.getState()
        angle_range = np.array([-1, 1]) * self.half_angle + state["angle"]
        # shift angle_range's origin to 6 clock
        angle_range = angle_range - 90
        angle_range = angle_range - np.floor(angle_range / 360.0) * 360.0
        size = state["size"]
        dist = np.hypot(size[0], size[1] / 2.0)
        ret = {
            "sl_type": "Pie",
            "dist": dist,
            "angle_range": angle_range,
            "pos": tuple(self.pos()),
        }
        return ret

    def paint(self, p, *args):
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.scale(r.width(), r.height())
        p.setPen(self.currentPen)
        p.drawPolygon(self.poly)

    def shape(self):
        self.path = QtGui.QPainterPath()
        r = self.boundingRect()
        # scale the path to match whats on the screen
        t = QtGui.QTransform()
        t.scale(r.width(), r.height())

        width = r.width()
        height = r.height()
        self.create_poly(width, height)
        self.path.addPolygon(self.poly)
        return t.map(self.path)

    def getArrayRegion(self, *args, **kwds):
        return self._getArrayRegionForArbitraryShape(*args, **kwds)
