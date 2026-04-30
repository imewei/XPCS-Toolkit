"""
Plot rendering backends for XPCS visualization.

This package provides theme-aware plotting backends using both
Matplotlib and PyQtGraph:

- matplot_qt: Matplotlib Qt integration with MplCanvas classes
- pyqtgraph_handler: PyQtGraph widgets (ImageViewDev, PlotWidgetDev)
- plot_constants: Theme-aware color palettes and styling
- qt_signal_fixes: Qt signal/slot compatibility utilities

All backends support light/dark theme switching and maintain
consistent visual styling across the application.
"""

# Import modules gracefully for documentation builds
import os
from typing import TYPE_CHECKING

if os.environ.get("BUILDING_DOCS") and not TYPE_CHECKING:
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

    # Additional modules — need __name__ for Sphinx autodoc compatibility
    import types

    pyqtgraph_handler = types.ModuleType("xpcsviewer.plothandler.pyqtgraph_handler")
    matplot_qt = types.ModuleType("xpcsviewer.plothandler.matplot_qt")
    plot_constants = types.ModuleType("xpcsviewer.plothandler.plot_constants")
    qt_signal_fixes = types.ModuleType("xpcsviewer.plothandler.qt_signal_fixes")
else:

    def __getattr__(name):
        """Lazily load matplotlib and pyqtgraph backends to improve startup time."""
        if name in ("MplCanvas", "MplCanvasBar", "MplCanvasBarH"):
            import importlib

            module = importlib.import_module("xpcsviewer.plothandler.matplot_qt")
            return getattr(module, name)
        if name == "MplCanvasBarV":
            from .lazy_proxy import LazyMplCanvasBarV

            return LazyMplCanvasBarV
        if name in ("ImageViewDev", "ImageViewPlotItem", "PlotWidgetDev"):
            import importlib

            module = importlib.import_module("xpcsviewer.plothandler.pyqtgraph_handler")
            return getattr(module, name)
        if name in (
            "matplot_qt",
            "plot_constants",
            "pyqtgraph_handler",
            "qt_signal_fixes",
        ):
            import importlib

            return importlib.import_module(f"xpcsviewer.plothandler.{name}")

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
