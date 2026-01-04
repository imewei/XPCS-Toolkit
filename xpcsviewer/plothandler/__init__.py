# Import modules gracefully for documentation builds
import os

if os.environ.get("BUILDING_DOCS"):
    # Provide placeholder classes for documentation
    class PlaceholderClass:
        """Placeholder class for documentation builds."""

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    # Matplotlib-based classes
    MplCanvas = PlaceholderClass
    MplCanvasBar = PlaceholderClass
    MplCanvasBarH = PlaceholderClass
    MplCanvasBarV = PlaceholderClass

    # PyQtGraph-based classes
    ImageViewDev = PlaceholderClass
    ImageViewPlotItem = PlaceholderClass
    PlotWidgetDev = PlaceholderClass

    # Additional modules
    pyqtgraph_handler = type("Module", (), {})()
    matplot_qt = type("Module", (), {})()
    plot_constants = type("Module", (), {})()
    qt_signal_fixes = type("Module", (), {})()
else:
    try:
        from . import matplot_qt, plot_constants, pyqtgraph_handler, qt_signal_fixes
        from .matplot_qt import MplCanvas, MplCanvasBar, MplCanvasBarH, MplCanvasBarV
        from .pyqtgraph_handler import ImageViewDev, ImageViewPlotItem, PlotWidgetDev
    except ImportError:
        # Fallback for missing dependencies
        pass

__all__ = [
    "ImageViewDev",
    "ImageViewPlotItem",
    "MplCanvas",
    "MplCanvasBar",
    "MplCanvasBarH",
    "MplCanvasBarV",
    "PlotWidgetDev",
    "matplot_qt",
    "plot_constants",
    "pyqtgraph_handler",
    "qt_signal_fixes",
]
