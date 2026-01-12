"""Custom PyQtGraph components for SimpleMask.

This module provides customized PyQtGraph widgets and ROI classes
for the SimpleMask editor.

Ported from pySimpleMask with minimal modifications.
"""

import numpy as np
import pyqtgraph as pg


class ImageViewROI(pg.ImageView):
    """Extended ImageView with ROI management capabilities."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.removeItem(self.roi)
        self.roi: dict[str, pg.ROI] = {}
        self.roi_idx = 0
        self.ui.roiBtn.setDisabled(True)
        self.ui.menuBtn.setDisabled(True)

    def roiClicked(self):
        """Override to disable default ROI click behavior."""

    def roiChanged(self):
        """Override to disable default ROI change behavior."""

    def adjust_viewbox(self):
        """Configure viewbox for optimal image viewing."""
        vb = self.getView()
        xmin, xmax = vb.viewRange()[0]
        ymin, ymax = vb.viewRange()[1]

        vb.setLimits(
            xMin=xmin,
            xMax=xmax,
            yMin=ymin,
            yMax=ymax,
            minXRange=(xmax - xmin) / 50,
            minYRange=(ymax - ymin) / 50,
        )
        vb.setMouseMode(vb.RectMode)
        vb.setAspectLocked(1.0)

    def reset_limits(self):
        """Reset viewbox limits for image updates."""
        self.view.state["limits"] = {
            "xLimits": [None, None],
            "yLimits": [None, None],
            "xRange": [None, None],
            "yRange": [None, None],
        }

    def set_colormap(self, cmap: str):
        """Set colormap from matplotlib name.

        Args:
            cmap: Matplotlib colormap name (e.g., "jet", "viridis")
        """
        pg_cmap = pg.colormap.getFromMatplotlib(cmap)
        self.setColorMap(pg_cmap)

    def remove_rois(self, filter_str: str | None = None):
        """Remove ROIs, optionally filtered by label prefix.

        Args:
            filter_str: If provided, only remove ROIs with labels starting with this
        """
        keys = list(self.roi.keys()).copy()
        if filter_str is not None:
            keys = list(filter(lambda x: x.startswith(filter_str), keys))
        for key in keys:
            self.remove_item(key)

    def clear(self):
        """Clear all ROIs and reset the view."""
        self.remove_rois()
        self.roi = {}
        super().clear()
        self.reset_limits()

    def add_item(self, t: pg.ROI, label: str | None = None) -> str:
        """Add an ROI item with optional label.

        Args:
            t: PyQtGraph ROI or graphics item
            label: Optional label (auto-generated if None)

        Returns:
            Label string for the added item
        """
        if label is None:
            label = f"roi_{self.roi_idx:06d}"
            self.roi_idx += 1

        if label in self.roi:
            self.remove_item(label)

        self.roi[label] = t
        self.addItem(t)
        return label

    def remove_item(self, label: str):
        """Remove an ROI by label.

        Args:
            label: Label of the ROI to remove
        """
        t = self.roi.pop(label, None)
        if t is not None:
            self.removeItem(t)

    def updateImage(self, autoHistogramRange: bool = True):
        """Update displayed image with optional histogram auto-ranging.

        Args:
            autoHistogramRange: Whether to auto-adjust histogram range
        """
        if self.image is None:
            return

        image = self.getProcessedImage()

        lmin = np.min(image[self.currentIndex])
        lmax = np.max(image[self.currentIndex])
        if autoHistogramRange:
            self.setLevels(rgba=[(lmin, lmax)])

        # Transpose image into order expected by ImageItem
        if self.imageItem.axisOrder == "col-major":
            axorder = ["t", "x", "y", "c"]
        else:
            axorder = ["t", "y", "x", "c"]
        axorder = [self.axes[ax] for ax in axorder if self.axes[ax] is not None]
        image = image.transpose(axorder)

        # Select time index
        if self.axes["t"] is not None:
            self.ui.roiPlot.show()
            image = image[self.currentIndex]

        self.imageItem.updateImage(image)


class LineROI(pg.ROI):
    """Rectangular ROI with line-like positioning.

    This ROI subclass allows positioning like a line segment while
    maintaining a rectangular area for masking.

    Args:
        pos1: Position of the center of the ROI's left edge
        pos2: Position of the center of the ROI's right edge
        width: Width of the ROI perpendicular to the line axis
        **args: Additional arguments passed to pg.ROI
    """

    def __init__(
        self,
        pos1: tuple[float, float],
        pos2: tuple[float, float],
        width: float,
        **args,
    ):
        p1 = pg.Point(pos1)
        p2 = pg.Point(pos2)
        d = p2 - p1
        length = d.length()
        ra = d.angle(pg.Point(1, 0), units="radians")
        c = pg.Point(width / 2.0 * np.sin(ra), -width / 2.0 * np.cos(ra))
        p1 = p1 + c

        pg.ROI.__init__(
            self, p1, size=pg.Point(length, width), angle=np.rad2deg(ra), **args
        )
        self.addScaleRotateHandle([1, 0.5], [0, 0])
