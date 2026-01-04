"""
GUI widgets for XPCS Viewer.

Custom widgets for enhanced user interaction.
"""

from xpcsviewer.gui.widgets.command_palette import CommandPalette
from xpcsviewer.gui.widgets.drag_drop_list import DragDropListView
from xpcsviewer.gui.widgets.empty_state import (
    EmptyStateWidget,
    FeatureNotAvailableState,
    NoDataToPlotState,
    NoFilesLoadedState,
    NoFilesSelectedState,
)
from xpcsviewer.gui.widgets.error_dialog import (
    ErrorDialog,
    show_connection_error,
    show_data_error,
    show_file_error,
    show_plot_error,
)
from xpcsviewer.gui.widgets.toast_notification import (
    ToastManager,
    ToastType,
    ToastWidget,
)

__all__ = [
    "CommandPalette",
    "DragDropListView",
    "EmptyStateWidget",
    "ErrorDialog",
    "FeatureNotAvailableState",
    "NoDataToPlotState",
    "NoFilesLoadedState",
    "NoFilesSelectedState",
    "ToastManager",
    "ToastType",
    "ToastWidget",
    "show_connection_error",
    "show_data_error",
    "show_file_error",
    "show_plot_error",
]
