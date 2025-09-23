# Standard library imports
import json
import os
import shutil
import time
import traceback

# Third-party imports
import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from PySide6 import QtCore, QtWidgets

# Import async components
from .threading.async_kernel import AsyncDataPreloader, AsyncViewerKernel
from .threading.progress_manager import ProgressManager

# Import centralized logging
from .utils import get_logger, log_system_info, setup_exception_logging
from .viewer_kernel import ViewerKernel

# Local imports
from .viewer_ui import Ui_mainWindow as Ui

# Initialize centralized logging and get logger
logger = get_logger(__name__)

# Setup exception logging
setup_exception_logging()

# Ensure home directory exists (moved after logging setup)
home_dir = os.path.join(os.path.expanduser("~"), ".xpcs_toolkit")
if not os.path.isdir(home_dir):
    os.mkdir(home_dir)
    logger.info(f"Created home directory: {home_dir}")

# Log system information for debugging
log_system_info()


tab_mapping = {
    0: "saxs_2d",
    1: "saxs_1d",
    2: "stability",
    3: "intensity_t",
    4: "g2",
    5: "diffusion",
    6: "twotime",
    7: "qmap",
    8: "average",
    9: "metadata",
}


def create_param_tree(data_dict):
    """Convert a dictionary into PyQtGraph's ParameterTree format."""
    params = []
    for key, value in data_dict.items():
        if isinstance(value, dict):  # If value is a nested dictionary
            params.append(
                {"name": key, "type": "group", "children": create_param_tree(value)}
            )
        elif isinstance(value, (int, float, np.number)):  # Numeric types
            params.append({"name": key, "type": "float", "value": float(value)})
        elif isinstance(value, str):  # String types
            params.append({"name": key, "type": "str", "value": value})
        elif isinstance(value, np.ndarray):  # Numpy arrays
            params.append({"name": key, "type": "text", "value": str(value.tolist())})
        else:  # Default fallback
            params.append({"name": key, "type": "text", "value": str(value)})
    return params


class XpcsViewer(QtWidgets.QMainWindow, Ui):
    """
    Main XPCS Viewer application window.

    XpcsViewer provides a comprehensive GUI for analyzing X-ray Photon Correlation
    Spectroscopy (XPCS) datasets. It supports both multi-tau and two-time correlation
    analysis with interactive plotting and real-time data visualization.

    The viewer integrates multiple analysis modules including SAXS 2D/1D visualization,
    G2 correlation analysis, stability monitoring, intensity-vs-time analysis,
    two-time correlation maps, and data averaging capabilities.

    Parameters
    ----------
    path : str, optional
        Initial directory path to load XPCS data files. If None, uses home directory.
    label_style : str, optional
        Custom labeling style for file identification. Format: comma-separated indices.

    Attributes
    ----------
    vk : ViewerKernel
        Backend kernel managing data processing and file operations
    async_vk : AsyncViewerKernel
        Asynchronous processing kernel for non-blocking operations
    progress_manager : ProgressManager
        Manager for tracking and displaying operation progress
    thread_pool : QThreadPool
        Thread pool for background computations

    Notes
    -----
    The application uses a tab-based interface where each tab represents a different
    analysis mode. Heavy computations are performed in background threads to maintain
    GUI responsiveness. The viewer supports both synchronous and asynchronous plotting
    modes for optimal performance.

    Examples
    --------
    >>> viewer = XpcsViewer(path="/path/to/data")
    >>> viewer.show()
    """

    def __init__(self, path=None, label_style=None):
        super().__init__()
        self.setupUi(self)
        self.home_dir = home_dir
        self.label_style = label_style

        # Maximize the window at startup
        self.showMaximized()

        self.tabWidget.setCurrentIndex(0)  # show scattering 2d
        self.plot_kwargs_record = {}
        for _, v in tab_mapping.items():
            self.plot_kwargs_record[v] = {}

        self.thread_pool = QtCore.QThreadPool()
        logger.info("Maximal threads: %d", self.thread_pool.maxThreadCount())

        # Initialize async components
        self.progress_manager = ProgressManager(self)
        self.progress_manager.set_statusbar(self.statusbar)

        self.vk = None
        self.async_vk = None  # Will be initialized when vk is ready
        self.data_preloader = None

        # Track initialization state for performance optimization
        self._startup_complete = False

        # Track active async operations
        self.active_plot_operations = {}  # operation_id -> plot_type

        # list widget models
        self.source_model = None
        self.target_model = None
        self.timer = QtCore.QTimer()

        if path is not None:
            self.start_wd = path
            self.load_path(path)
        else:
            # use home directory
            self.start_wd = os.path.expanduser("~")

        self.start_wd = os.path.abspath(self.start_wd)
        logger.info(f"Start up directory is [{self.start_wd}]")

        self.pushButton_plot_saxs2d.clicked.connect(self.plot_saxs_2d)
        self.pushButton_plot_saxs1d.clicked.connect(self.plot_saxs_1d)
        self.pushButton_plot_stability.clicked.connect(self.plot_stability)
        self.pushButton_plot_intt.clicked.connect(self.plot_intensity_t)
        self.pushButton_8.clicked.connect(
            self.plot_diffusion
        )  # Diffusion "fit plot" button
        # self.saxs1d_lb_type.currentIndexChanged.connect(self.switch_saxs1d_line)

        self.tabWidget.currentChanged.connect(self.update_plot)
        self.list_view_target.clicked.connect(self.update_plot)

        self.mp_2t_hdls = None
        self.init_twotime_plot_handler()

        self.avg_job_pop.clicked.connect(self.remove_avg_job)
        self.btn_submit_job.clicked.connect(self.submit_job)
        self.btn_start_avg_job.clicked.connect(self.start_avg_job)
        self.btn_set_average_save_path.clicked.connect(self.set_average_save_path)
        self.btn_set_average_save_name.clicked.connect(self.set_average_save_name)
        self.btn_avg_kill.clicked.connect(self.avg_kill_job)
        self.btn_avg_jobinfo.clicked.connect(self.show_avg_jobinfo)
        self.avg_job_table.clicked.connect(self.update_avg_info)
        self.show_g2_fit_summary.clicked.connect(self.show_g2_fit_summary_func)
        self.btn_g2_export.clicked.connect(self.export_g2)
        self.btn_g2_refit.clicked.connect(self.refit_g2)
        self.pushButton_6.clicked.connect(self.saxs2d_roi_add)  # ROI Add button
        self.saxs2d_autolevel.stateChanged.connect(self.update_saxs2d_level)
        self.btn_deselect.clicked.connect(self.clear_target_selection)
        self.list_view_target.doubleClicked.connect(self.show_dataset)
        self.btn_select_bkgfile.clicked.connect(self.select_bkgfile)
        self.spinBox_saxs2d_selection.valueChanged.connect(self.plot_saxs_2d_selection)
        self.comboBox_twotime_selection.currentIndexChanged.connect(
            self.on_twotime_q_selection_changed
        )

        self.g2_fitting_function.currentIndexChanged.connect(
            self.update_g2_fitting_function
        )
        self.btn_up.clicked.connect(lambda: self.reorder_target("up"))
        self.btn_down.clicked.connect(lambda: self.reorder_target("down"))

        self.btn_export_saxs1d.clicked.connect(self.saxs1d_export)
        self.btn_export_diffusion.clicked.connect(self.export_diffusion)

        self.comboBox_qmap_target.currentIndexChanged.connect(self.update_plot)
        self.update_g2_fitting_function()

        self.pg_saxs.getView().scene().sigMouseMoved.connect(self.saxs2d_mouseMoved)

        # Connect progress manager to show/hide progress dialog
        # Add keyboard shortcut to show progress dialog
        self.setup_async_connections()
        self.setup_progress_shortcut()

        self.load_default_setting()
        self.show()

    def load_default_setting(self):
        if not os.path.isdir(self.home_dir):
            os.mkdir(self.home_dir)

        key_fname = os.path.join(self.home_dir, "default_setting.json")
        # copy the default values
        if not os.path.isfile(key_fname):
            from .default_setting import setting

            with open(key_fname, "w") as f:
                json.dump(setting, f, indent=4)

        # the display size might too big for some laptops
        with open(key_fname) as f:
            config = json.load(f)
            if "window_size_h" in config:
                new_size = (config["window_size_w"], config["window_size_h"])
                logger.info("set mainwindow to size %s", new_size)
                self.resize(*new_size)

        cache_dir = os.path.join(
            os.path.expanduser("~"), ".xpcs_toolkit", "joblib/xpcs_toolkit"
        )
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)

    def setup_async_connections(self):
        """Set up connections for async operations."""
        # Connect progress manager signals for operation cancellation
        self.progress_manager.operation_cancelled.connect(self.cancel_async_operation)

    def setup_progress_shortcut(self):
        """Set up keyboard shortcut to show progress dialog."""
        from PySide6.QtGui import QKeySequence, QShortcut

        # Ctrl+P to show progress dialog
        shortcut = QShortcut(QKeySequence("Ctrl+P"), self)
        shortcut.activated.connect(self.show_progress_dialog)

    def show_progress_dialog(self):
        """Show the progress dialog."""
        self.progress_manager.show_progress_dialog()

    def init_async_kernel(self):
        """Initialize async kernel when viewer kernel is ready."""
        if self.vk and not self.async_vk:
            self.async_vk = AsyncViewerKernel(self.vk, self.thread_pool)
            self.data_preloader = AsyncDataPreloader(self.async_vk)

            # Connect async signals
            self.async_vk.plot_ready.connect(self.on_async_plot_ready)
            self.async_vk.operation_error.connect(self.on_async_operation_error)
            self.async_vk.operation_progress.connect(
                self.progress_manager.update_progress
            )

            logger.info("Async kernel initialized")

    def cancel_async_operation(self, operation_id: str):
        """Cancel an async operation."""
        if self.async_vk:
            self.async_vk.cancel_operation(operation_id)

        # Remove from active operations
        if operation_id in self.active_plot_operations:
            del self.active_plot_operations[operation_id]

    def on_async_plot_ready(self, operation_id: str, result):
        """Handle async plot completion."""
        if operation_id not in self.active_plot_operations:
            return

        plot_type = self.active_plot_operations[operation_id]

        try:
            # Apply the plot result to the GUI
            if plot_type == "saxs_2d":
                self.apply_saxs_2d_result(result)
            elif plot_type == "g2":
                self.apply_g2_result(result)
            elif plot_type == "twotime":
                self.apply_twotime_result(result)
            elif plot_type == "intensity_t":
                self.apply_intensity_result(result)
            elif plot_type == "stability":
                self.apply_stability_result(result)
            elif plot_type == "qmap":
                self.apply_qmap_result(result)

            self.progress_manager.complete_operation(operation_id, True)

        except Exception as e:
            logger.error(f"Error applying {plot_type} plot result: {e}")
            self.progress_manager.complete_operation(
                operation_id, False, f"Plot error: {e!s}"
            )

        # Clean up
        del self.active_plot_operations[operation_id]

    def on_async_operation_error(
        self, operation_id: str, error_msg: str, traceback_str: str
    ):
        """Handle async operation errors."""
        logger.error(f"Async operation {operation_id} failed: {error_msg}")
        logger.debug(f"Traceback: {traceback_str}")

        self.progress_manager.complete_operation(operation_id, False, error_msg)

        # Clean up
        if operation_id in self.active_plot_operations:
            del self.active_plot_operations[operation_id]

    def apply_saxs_2d_result(self, result):
        """Apply SAXS 2D plot result to the GUI."""
        if result is None:
            logger.debug("No SAXS 2D data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"SAXS 2D plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        image_data = result["image_data"]
        levels = result["levels"]
        result["cmap"]

        # Update the plot
        self.pg_saxs.setImage(image_data, levels=levels)
        # Apply colormap if needed

    def apply_g2_result(self, result):
        """Apply G2 plot result to the GUI."""
        if result is None:
            logger.debug("No G2 data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(f"G2 plotting info: {result.get('message', 'Unknown info')}")
            return

        # Apply the processed G2 data to the plot widget
        try:
            # Extract data from worker result
            result["q"]
            result["tel"]
            result["g2"]
            result["g2_err"]
            result["labels"]
            plot_params = result["plot_params"]

            # Get the file list (need to recreate from viewer kernel)
            # Support both Multitau and Twotime files for G2 plotting
            rows = plot_params.get("rows", [])
            xf_list = self.vk.get_xf_list(rows=rows)

            # Filter for files that support G2 plotting (Multitau or Twotime)
            g2_compatible_files = []
            for xf in xf_list:
                if any(atype in xf.atype for atype in ["Multitau", "Twotime"]):
                    g2_compatible_files.append(xf)

            if g2_compatible_files:
                # Use the viewer kernel method with lazy loading
                self.vk.plot_g2(
                    self.mp_g2,  # Plot handler
                    plot_params.get("q_range"),
                    plot_params.get("t_range"),
                    plot_params.get("y_range"),
                    rows=self.get_selected_rows(),
                    **{
                        k: v
                        for k, v in plot_params.items()
                        if k not in ["q_range", "t_range", "y_range", "rows"]
                    },
                )
                logger.info("G2 plot applied successfully")
            else:
                logger.warning(
                    "No files with G2 data (Multitau or Twotime) available for plotting"
                )

        except Exception as e:
            logger.error(f"Failed to apply G2 result: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_twotime_result(self, result):
        """Apply two-time plot result to the GUI."""
        if result is None:
            logger.debug("No twotime data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Two-time plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        c2_result = result["c2_result"]
        new_qbin_labels = result["new_qbin_labels"]

        # Update the two-time plots using lazy loading
        self.vk.get_module("twotime").plot_twotime_g2(self.mp_2t_hdls, c2_result)

        if new_qbin_labels:
            logger.debug(
                f"[ASYNC] Repopulating ComboBox with new labels: {new_qbin_labels}"
            )
            # Preserve current selection when repopulating
            current_selection = self.comboBox_twotime_selection.currentIndex()
            logger.debug(
                f"COMBOBOX POPULATE: [ASYNC] Current selection before repopulation: {current_selection}"
            )
            logger.debug(
                f"COMBOBOX POPULATE: [ASYNC] Adding {len(new_qbin_labels)} items to ComboBox: {new_qbin_labels[:5]}..."
            )  # Show first 5 items

            # Block signals to prevent recursive updates
            self.comboBox_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.blockSignals(True)

            try:
                self.comboBox_twotime_selection.clear()
                self.comboBox_twotime_selection.addItems(new_qbin_labels)
                self.horizontalSlider_twotime_selection.setMaximum(
                    len(new_qbin_labels) - 1
                )

                # Restore selection if it's valid
                if 0 <= current_selection < len(new_qbin_labels):
                    self.comboBox_twotime_selection.setCurrentIndex(current_selection)
                    self.horizontalSlider_twotime_selection.setValue(current_selection)
                    logger.debug(f"[ASYNC] Restored selection to: {current_selection}")
                else:
                    logger.debug(
                        f"[ASYNC] Selection {current_selection} is invalid for {len(new_qbin_labels)} items, defaulting to 0"
                    )
            finally:
                # Always restore signal connections
                self.comboBox_twotime_selection.blockSignals(False)
                self.horizontalSlider_twotime_selection.blockSignals(False)

    def apply_intensity_result(self, result):
        """Apply intensity plot result to the GUI."""
        logger.debug(
            f"Apply intensity result called with result: {type(result)} - {result}"
        )

        if result is None:
            logger.debug("No intensity data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Intensity plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        try:
            # Use the lazy-loaded intensity module to plot
            logger.debug(
                f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}"
            )
            plot_params = result["plot_params"]
            rows = plot_params.get("rows", [])
            logger.debug(f"Plot params: {plot_params}")

            # Get the file list
            xf_list = self.vk.get_xf_list(rows=rows)
            logger.debug(f"Got {len(xf_list)} files for intensity plot")

            if xf_list:
                # Use the viewer kernel method with lazy loading
                # Filter out 'rows' from plot_params to avoid duplicate parameter
                filtered_params = {k: v for k, v in plot_params.items() if k != "rows"}
                self.vk.plot_intt(
                    self.pg_intt, rows=self.get_selected_rows(), **filtered_params
                )
                logger.info("Intensity plot applied successfully")
            else:
                logger.warning("No files available for intensity plotting")

        except Exception as e:
            logger.error(f"Failed to apply intensity result: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_stability_result(self, result):
        """Apply stability plot result to the GUI."""
        if result is None:
            logger.debug("No stability data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(
                f"Stability plotting info: {result.get('message', 'Unknown info')}"
            )
            return

        try:
            # Use the lazy-loaded stability module via viewer kernel
            result["xf_obj"]
            plot_params = result["plot_params"]

            # Use the viewer kernel method with lazy loading
            # Filter out 'rows' from plot_params to avoid duplicate parameter
            filtered_params = {k: v for k, v in plot_params.items() if k != "rows"}
            if self.vk:
                self.vk.plot_stability(
                    self.mp_stab, rows=self.get_selected_rows(), **filtered_params
                )
            logger.info("Stability plot applied successfully")

        except Exception as e:
            logger.error(f"Failed to apply stability result: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")

    def apply_qmap_result(self, result):
        """Apply Q-map plot result to the GUI."""
        if result is None:
            logger.debug("No Q-map data to plot")
            return

        # Handle info messages
        if isinstance(result, dict) and result.get("type") == "info":
            logger.info(f"Q-map plotting info: {result.get('message', 'Unknown info')}")
            return

        image_data = result["image_data"]
        levels = result["levels"]

        # Update the Q-map plot
        self.pg_qmap.setImage(image_data, levels=levels)

    def get_selected_rows(self):
        selected_index = self.list_view_target.selectedIndexes()
        selected_row = [x.row() for x in selected_index]
        # the selected index is ordered;
        selected_row.sort()

        # If no specific rows are selected but files exist in target list,
        # default to using all files for better user experience
        if not selected_row and self.vk and self.vk.target and len(self.vk.target) > 0:
            selected_row = list(range(len(self.vk.target)))
            logger.debug(
                f"No specific rows selected, defaulting to all {len(selected_row)} files in target list"
            )

        return selected_row

    def update_plot(self):
        """
        Update plot display based on current tab and selected files.

        Automatically determines whether to use synchronous or asynchronous
        plotting based on the current tab and available async kernel.
        Supported async tabs include SAXS 2D/1D, G2, two-time, intensity,
        stability, and Q-map visualizations.

        Notes
        -----
        - Checks plot parameters for changes before updating
        - Uses async plotting for performance-critical visualizations
        - Falls back to synchronous plotting for unsupported tabs
        - Updates plot kwargs record for caching
        """
        idx = self.tabWidget.currentIndex()
        tab_name = tab_mapping[idx]
        logger.debug(f"update_plot called for tab: {tab_name} (index: {idx})")

        if tab_name == "average":
            return

        # Check if files are selected before attempting to plot
        # Some tabs like diffusion should always update to show proper empty state
        tabs_requiring_files = [
            "saxs_2d",
            "saxs_1d",
            "stability",
            "intensity_t",
            "g2",
            "twotime",
            "qmap",
        ]
        if (
            not self.vk or not self.vk.target or len(self.vk.target) == 0
        ) and tab_name in tabs_requiring_files:
            logger.debug(
                f"No files selected for {tab_name} plotting, "
                f"clearing plot and skipping update"
            )
            # Clear the plot to show empty state
            self._clear_plot_for_tab(tab_name)
            return

        # Check if we should use async plotting
        # Note: twotime removed from async list to fix automatic plotting issues
        use_async = self.async_vk is not None and tab_name in [
            "saxs_2d",
            "g2",
            "intensity_t",
            "stability",
            "qmap",
        ]

        logger.debug(
            f"Using {'async' if use_async else 'sync'} plotting for {tab_name}"
        )

        if use_async:
            self.update_plot_async(tab_name)
        else:
            self.update_plot_sync(tab_name)

    def update_plot_sync(self, tab_name):
        """Synchronous plot update (original behavior)."""
        # Special handling for diffusion tab - always update the pre-plot
        if tab_name == "diffusion":
            try:
                self.init_diffusion()
                logger.debug("Diffusion pre-plot updated")
            except Exception as e:
                logger.error(f"Failed to update diffusion pre-plot: {e}")
                traceback.print_exc()
            # Don't return here - still allow the normal plot_diffusion to be called

        func = getattr(self, "plot_" + tab_name)
        try:
            kwargs = func(dryrun=True)
            kwargs["target_timestamp"] = self.vk.timestamp
            if self.plot_kwargs_record[tab_name] != kwargs:
                logger.debug(f"Plot parameters changed for {tab_name}")
                logger.debug(f"Old params: {self.plot_kwargs_record[tab_name]}")
                logger.debug(f"New params: {kwargs}")
                self.plot_kwargs_record[tab_name] = kwargs
                func(dryrun=False)
            else:
                logger.debug(f"No parameter changes for {tab_name}, skipping update")
        except Exception as e:
            logger.error(f"update selection in [{tab_name}] failed")
            logger.error(e)
            traceback.print_exc()

    def _clear_plot_for_tab(self, tab_name):
        """Clear plot display for specified tab when no files are selected."""
        try:
            plot_handlers = {
                "saxs_2d": self.pg_saxs if hasattr(self, "pg_saxs") else None,
                "saxs_1d": self.mp_saxs1d if hasattr(self, "mp_saxs1d") else None,
                "stability": self.mp_stab if hasattr(self, "mp_stab") else None,
                "intensity_t": self.pg_intt if hasattr(self, "pg_intt") else None,
                "g2": self.mp_g2 if hasattr(self, "mp_g2") else None,
                "twotime": self.mp_2t_hdls if hasattr(self, "mp_2t_hdls") else None,
                "qmap": self.pg_qmap if hasattr(self, "pg_qmap") else None,
            }

            handler = plot_handlers.get(tab_name)
            if handler:
                if hasattr(handler, "clear"):
                    handler.clear()
                elif hasattr(handler, "setImage"):
                    # For ImageView widgets, set empty image
                    empty_image = np.zeros((50, 50))
                    handler.setImage(empty_image)
                logger.debug(f"Cleared plot for {tab_name}")
                # Show status message to user for guidance
                self.statusbar.showMessage(
                    f"No files selected for {tab_name.replace('_', ' ').title()} - please select files from source list and click 'Add Target'",  # nosec B608
                    5000,
                )
        except Exception as e:
            logger.warning(f"Failed to clear plot for {tab_name}: {e}")

    def update_plot_async(self, tab_name):
        """Asynchronous plot update with progress indication."""
        # Check if there's already an active operation for this plot type
        for _op_id, plot_type in self.active_plot_operations.items():
            if plot_type == tab_name:
                logger.info(f"Async {tab_name} plot already in progress, skipping")
                return

        try:
            # Get plot parameters
            func = getattr(self, "plot_" + tab_name)
            kwargs = func(dryrun=True)
            kwargs["target_timestamp"] = self.vk.timestamp

            # Check if parameters changed
            if self.plot_kwargs_record[tab_name] == kwargs:
                logger.debug(f"No parameter changes for async {tab_name}, skipping")
                return  # No change needed
            logger.debug(f"Async plot parameters changed for {tab_name}")
            logger.debug(f"Old params: {self.plot_kwargs_record[tab_name]}")
            logger.debug(f"New params: {kwargs}")

            self.plot_kwargs_record[tab_name] = kwargs

            # Start async plotting
            operation_id = f"plot_{tab_name}_{int(time.time() * 1000)}"

            # Start progress tracking
            description = f"Plotting {tab_name.replace('_', ' ').title()}"
            self.progress_manager.start_operation(
                operation_id, description, show_in_statusbar=True, is_cancellable=True
            )

            # Track this operation
            self.active_plot_operations[operation_id] = tab_name

            # Map tab names to async methods
            async_method_map = {
                "saxs_2d": self.async_vk.plot_saxs_2d_async,
                "g2": self.async_vk.plot_g2_async,
                "twotime": self.async_vk.plot_twotime_async,
                "intensity_t": self.async_vk.plot_intensity_async,
                "stability": self.async_vk.plot_stability_async,
                "qmap": self.async_vk.plot_qmap_async,
            }

            # Get the appropriate plot handler
            handler_map = {
                "saxs_2d": self.pg_saxs,
                "g2": self.mp_g2,
                "twotime": self.mp_2t_hdls,
                "intensity_t": self.pg_intt,
                "stability": self.mp_stab,
                "qmap": self.pg_qmap,
            }

            if tab_name in async_method_map:
                plot_handler = handler_map[tab_name]
                async_method = async_method_map[tab_name]

                # Start the async operation
                async_method(plot_handler, operation_id=operation_id, **kwargs)

                logger.info(f"Started async {tab_name} plotting: {operation_id}")

        except Exception as e:
            logger.error(f"Failed to start async {tab_name} plotting: {e}")
            # Fall back to synchronous plotting
            self.update_plot_sync(tab_name)

    def plot_metadata(self, dryrun=False):
        kwargs = {"rows": self.get_selected_rows()}
        if dryrun:
            return kwargs

        xf_list = self.vk.get_xf_list(**kwargs)
        if not xf_list:
            logger.warning("No files available for metadata display")
            # Clear the metadata display
            empty_params = Parameter.create(name="Settings", type="group", children=[])
            self.hdf_info.setParameters(empty_params, showTop=True)
            return None

        msg = xf_list[0].get_hdf_info()
        hdf_info_data = create_param_tree(msg)
        hdf_params = Parameter.create(
            name="Settings", type="group", children=hdf_info_data
        )
        self.hdf_info.setParameters(hdf_params, showTop=True)
        return None

    def saxs2d_mouseMoved(self, pos):
        if self.pg_saxs.view.sceneBoundingRect().contains(pos):
            mouse_point = self.pg_saxs.getView().mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())
            rows = self.get_selected_rows()
            payload = self.vk.get_info_at_mouse(rows, x, y)
            if payload:
                self.saxs2d_display.setText(payload)

    def plot_saxs_2d_selection(self):
        selection = self.spinBox_saxs2d_selection.value()
        self.plot_saxs_2d(selection=selection)

    def on_twotime_q_selection_changed(self, index):
        """Handle Q value selection changes in twotime ComboBox."""
        logger.debug(f"TWOTIME Q SELECTION CHANGED: ComboBox index changed to {index}")
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: ComboBox current text = '{self.comboBox_twotime_selection.currentText()}'"
        )
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: ComboBox item count = {self.comboBox_twotime_selection.count()}"
        )
        logger.debug(
            f"TWOTIME Q SELECTION CHANGED: Current tab = {self.tabWidget.currentIndex()}"
        )

        # Also update the slider to stay in sync
        if 0 <= index < self.comboBox_twotime_selection.count():
            self.horizontalSlider_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.setValue(index)
            self.horizontalSlider_twotime_selection.blockSignals(False)

        # Now trigger the plot update
        logger.debug("TWOTIME Q SELECTION CHANGED: Calling update_plot()")
        self.update_plot()
        logger.debug("TWOTIME Q SELECTION CHANGED: update_plot() completed")

    def plot_saxs_2d(self, selection=None, dryrun=False):
        kwargs = {
            "plot_type": self.cb_saxs2D_type.currentText(),
            "cmap": self.cb_saxs2D_cmap.currentText(),
            "rotate": self.saxs2d_rotate.isChecked(),
            "autolevel": self.saxs2d_autolevel.isChecked(),
            "vmin": self.saxs2d_min.value(),
            "vmax": self.saxs2d_max.value(),
        }
        if selection and selection >= 0:
            kwargs["rows"] = [selection]
        else:
            kwargs["rows"] = self.get_selected_rows()

        if dryrun:
            return kwargs
        self.vk.plot_saxs_2d(pg_hdl=self.pg_saxs, **kwargs)
        return None

    def saxs2d_roi_add(self):
        sl_type_idx = self.cb_saxs2D_roi_type.currentIndex()
        color = ("g", "y", "b", "r", "c", "m", "k", "w")[
            self.cb_saxs2D_roi_color.currentIndex()
        ]

        # Map UI ROI types correctly: Q-Wedge (0), Phi-Ring (1)
        roi_types = ("Q-Wedge", "Phi-Ring")

        kwargs = {
            "sl_type": roi_types[sl_type_idx],
            "width": self.sb_saxs2D_roi_width.value(),
            "color": color,
        }

        logger.debug(f"GUI: Attempting to add ROI with kwargs: {kwargs}")
        result = self.vk.add_roi(self.pg_saxs, **kwargs)
        if result is None:
            logger.warning(f"GUI: ROI addition failed for type '{kwargs['sl_type']}'")
        else:
            logger.debug(f"GUI: ROI addition succeeded, returned: {result}")

    def plot_saxs_1d(self, dryrun=False):
        kwargs = {
            "plot_type": self.cb_saxs_type.currentIndex(),
            "plot_offset": self.sb_saxs_offset.value(),
            "plot_norm": self.cb_saxs_norm.currentIndex(),
            "rows": self.get_selected_rows(),
            "qmin": self.saxs1d_qmin.value(),
            "qmax": self.saxs1d_qmax.value(),
            "loc": self.saxs1d_legend_loc.currentText(),
            "marker_size": self.sb_saxs_marker_size.value(),
            "sampling": self.saxs1d_sampling.value(),
            "all_phi": self.box_all_phi.isChecked(),
            "absolute_crosssection": self.cbox_use_abs.isChecked(),
            "subtract_background": self.cb_sub_bkg.isChecked(),
            "weight": self.bkg_weight.value(),
            "show_roi": self.box_show_roi.isChecked(),
            "show_phi_roi": self.box_show_phi_roi.isChecked(),
        }
        if kwargs["qmin"] >= kwargs["qmax"]:
            self.statusbar.showMessage("check qmin and qmax")
            return None

        if dryrun:
            return kwargs
        self.vk.plot_saxs_1d(self.pg_saxs, self.mp_saxs, **kwargs)
        # adjust the line behavior
        self.switch_saxs1d_line()
        return None

    def switch_saxs1d_line(self):
        lb_type = self.saxs1d_lb_type.currentIndex()
        lb_type = [None, "slope", "hline"][lb_type]
        self.vk.switch_saxs1d_line(self.mp_saxs, lb_type)

    def saxs1d_export(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="select a folder to export SAXS profiles"
        )

        if folder in [None, ""]:
            return

        self.vk.export_saxs_1d(self.pg_saxs, folder)

    def init_twotime_plot_handler(self):
        self.mp_2t_hdls = {}
        labels = ["saxs", "dqmap"]
        titles = ["scattering", "dynamic_qmap"]
        cmaps = ["viridis", "tab20"]
        self.mp_2t_map.setBackground("w")
        for n in range(2):
            plot_item = self.mp_2t_map.addPlot(row=0, col=n)
            # Remove axes
            plot_item.hideAxis("left")
            plot_item.hideAxis("bottom")
            plot_item.getViewBox().setDefaultPadding(0)

            plot_item.setMouseEnabled(x=False, y=False)
            image_item = pg.ImageItem(np.ones((128, 128)))
            image_item.setOpts(axisOrder="row-major")  # Set to row-major order

            plot_item.setTitle(titles[n])
            plot_item.addItem(image_item)
            plot_item.setAspectLocked(True)

            cmap = pg.colormap.getFromMatplotlib(cmaps[n])
            if n == 1:
                positions = cmap.pos
                colors = cmap.color
                new_color = [0, 0, 1, 1.0]
                colors[-1] = new_color
                # need to convert to 0-255 range for pyqtgraph ColorMap
                cmap = pg.ColorMap(positions, colors * 255)
            colorbar = plot_item.addColorBar(image_item, colorMap=cmap)
            self.mp_2t_hdls[labels[n]] = image_item
            self.mp_2t_hdls[labels[n] + "_colorbar"] = colorbar

        c2g2_plot = self.mp_2t_map.addPlot(row=0, col=2)
        self.mp_2t_hdls["c2g2"] = c2g2_plot

        self.mp_2t_hdls["dqmap"].mouseClickEvent = self.pick_twotime_index
        self.mp_2t_hdls["saxs"].mouseClickEvent = self.pick_twotime_index
        self.mp_2t.ui.graphicsView.setBackground("w")
        self.mp_2t_hdls["tt"] = self.mp_2t
        self.mp_2t_hdls["tt"].view.invertY(False)
        self.mp_2t.view.setLabel("left", "t2", units="s")
        self.mp_2t.view.setLabel("bottom", "t1", units="s")

    def pick_twotime_index(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            pos = event.pos()
            x, y = int(pos.x()), int(pos.y())
            self.plot_twotime(highlight_xy=(x, y))
        event.accept()  # Mark the event as handled

    def plot_qmap(self, dryrun=False):
        kwargs = {
            "rows": self.get_selected_rows(),
            "target": self.comboBox_qmap_target.currentText(),
        }
        if dryrun:
            return kwargs
        self.vk.plot_qmap(self.pg_qmap, **kwargs)
        return None

    def plot_twotime(self, dryrun=False, highlight_xy=None):
        current_selection = self.comboBox_twotime_selection.currentIndex()
        logger.debug(
            f"plot_twotime called: dryrun={dryrun}, ComboBox currentIndex={current_selection}, count={self.comboBox_twotime_selection.count()}"
        )

        kwargs = {
            "rows": self.get_selected_rows(),
            "auto_crop": self.twotime_autocrop.isChecked(),
            "highlight_xy": highlight_xy,
            "cmap": self.cb_twotime_cmap.currentText(),
            "vmin": self.c2_min.value(),
            "vmax": self.c2_max.value(),
            "correct_diag": self.twotime_correct_diag.isChecked(),
            "autolevel": self.checkBox_twotime_autolevel.isChecked(),
            "selection": max(0, current_selection),
        }

        logger.debug(f"Twotime plot kwargs: {kwargs}")
        if dryrun:
            return kwargs

        if self.mp_2t_hdls is None:
            self.init_twotime_plot_handler()
        new_labels = self.vk.plot_twotime(self.mp_2t_hdls, **kwargs)
        if new_labels is not None:
            logger.debug(f"Repopulating ComboBox with new labels: {new_labels}")
            # Preserve current selection when repopulating
            current_selection = self.comboBox_twotime_selection.currentIndex()
            logger.debug(
                f"COMBOBOX POPULATE: [SYNC] Current selection before repopulation: {current_selection}"
            )
            logger.debug(
                f"COMBOBOX POPULATE: [SYNC] Adding {len(new_labels)} items to ComboBox: {new_labels[:5]}..."
            )  # Show first 5 items

            # Block signals to prevent recursive updates
            self.comboBox_twotime_selection.blockSignals(True)
            self.horizontalSlider_twotime_selection.blockSignals(True)

            try:
                self.comboBox_twotime_selection.clear()
                self.comboBox_twotime_selection.addItems(new_labels)
                self.horizontalSlider_twotime_selection.setMaximum(len(new_labels) - 1)

                # Restore selection if it's valid
                if 0 <= current_selection < len(new_labels):
                    self.comboBox_twotime_selection.setCurrentIndex(current_selection)
                    self.horizontalSlider_twotime_selection.setValue(current_selection)
                    logger.debug(f"Restored selection to: {current_selection}")
                else:
                    logger.debug(
                        f"Selection {current_selection} is invalid for {len(new_labels)} items, defaulting to 0"
                    )
            finally:
                # Always restore signal connections
                self.comboBox_twotime_selection.blockSignals(False)
                self.horizontalSlider_twotime_selection.blockSignals(False)
        return None

    def show_dataset(self):
        rows = self.get_selected_rows()
        self.tree = self.vk.get_pg_tree(rows)
        if self.tree:
            self.tree.show()

    def plot_stability(self, dryrun=False):
        kwargs = {
            "plot_type": self.cb_stab_type.currentIndex(),
            "plot_norm": self.cb_stab_norm.currentIndex(),
            "rows": self.get_selected_rows(),
            "loc": self.stab_legend_loc.currentText(),
        }
        if dryrun:
            return kwargs
        if self.vk:
            self.vk.plot_stability(self.mp_stab, **kwargs)
        return None

    def plot_intensity_t(self, dryrun=False):
        kwargs = {
            "sampling": max(1, self.sb_intt_sampling.value()),
            "window": self.sb_window.value(),
            "rows": self.get_selected_rows(),
            "xlabel": self.intt_xlabel.currentText(),
        }
        if dryrun:
            return kwargs
        self.vk.plot_intt(self.pg_intt, **kwargs)
        return None

    def init_diffusion(self):
        logger.info("init_diffusion called - updating left window (mp_tauq_pre)")
        rows = self.get_selected_rows()
        logger.info(f"Selected rows for pre-plot: {rows}")

        # Debug mp_tauq_pre widget state
        logger.info(
            f"mp_tauq_pre visible: {self.mp_tauq_pre.isVisible()}, size: {self.mp_tauq_pre.size()}"
        )
        logger.info(f"mp_tauq_pre canvas: {self.mp_tauq_pre.hdl}")

        self.vk.plot_tauq_pre(hdl=self.mp_tauq_pre.hdl, rows=rows)

        logger.info("init_diffusion completed")

    def plot_diffusion(self, dryrun=False):
        logger.info("plot_diffusion method called")

        try:
            keys = [self.tauq_amin, self.tauq_bmin, self.tauq_amax, self.tauq_bmax]
            bounds = np.array([float(x.text()) for x in keys]).reshape(2, 2)

            fit_flag = [self.tauq_afit.isChecked(), self.tauq_bfit.isChecked()]

            if sum(fit_flag) == 0:
                logger.warning("No fit flags selected")
                self.statusbar.showMessage("nothing to fit, really?", 1000)
                return None

            tauq = [self.tauq_qmin, self.tauq_qmax]
            q_range = [float(x.text()) for x in tauq]

            kwargs = {
                "bounds": bounds.tolist(),
                "fit_flag": fit_flag,
                "offset": self.sb_tauq_offset.value(),
                "rows": self.get_selected_rows(),
                "q_range": q_range,
                "plot_type": self.cb_tauq_type.currentIndex(),
            }

            if dryrun:
                return kwargs

            # CRITICAL FIX: Force fresh G2 fitting first, then tau-q fitting
            # This ensures tau-q gets fresh G2 results as input
            rows = self.get_selected_rows()
            xf_list = self.vk.get_xf_list(rows)

            # Force fresh G2 fitting for all files to ensure fresh input data
            for xf in xf_list:
                if hasattr(xf, "fit_summary") and xf.fit_summary is not None:
                    # Get current G2 parameters from UI
                    g2_bounds, g2_fit_flag, g2_fit_func = self.check_g2_fitting_number()
                    g2_params = self.check_g2_number()
                    g2_q_range = (g2_params[0], g2_params[1])
                    g2_t_range = (g2_params[2], g2_params[3])

                    # Force fresh G2 fitting to provide fresh input for tau-q
                    logger.info(
                        f"Forcing fresh G2 fitting for {xf.label} before tau-q analysis"
                    )
                    xf.fit_g2(
                        g2_q_range,
                        g2_t_range,
                        g2_bounds,
                        g2_fit_flag,
                        g2_fit_func,
                        force_refit=True,
                    )

            # Now force fresh tau-q fitting with the fresh G2 results
            kwargs["force_refit"] = True
            msg = self.vk.plot_tauq(hdl=self.mp_tauq.hdl, **kwargs)

            # Update BOTH display areas: right panel AND left panel
            self.tauq_msg.clear()
            self.tauq_msg.setData(msg)
            self.tauq_msg.parent().repaint()

            # ALSO update left panel parameter plots with fresh data
            logger.info("Updating diffusion parameter plots with fresh results")
            self.init_diffusion()

            logger.info("plot_diffusion completed successfully")
            return None

        except Exception as e:
            logger.error(f"Error in plot_diffusion: {e}")
            import traceback

            traceback.print_exc()
            return None

    def select_bkgfile(self):
        path = self.work_dir.text()
        f = QtWidgets.QFileDialog.getOpenFileName(
            self, caption="select the file for background subtraction", dir=path
        )[0]
        if os.path.isfile(f):
            self.le_bkg_fname.setText(f)
            self.vk.select_bkgfile(f)
        else:
            return

    def remove_avg_job(self):
        index = self.avg_job_table.currentIndex().row()
        if index < 0:
            return
        self.vk.remove_job(index)

    def set_average_save_path(self):
        save_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Open directory")
        self.avg_save_path.clear()
        self.avg_save_path.setText(save_path)

    def set_average_save_name(self):
        save_name = QtWidgets.QFileDialog.getSaveFileName(self, "Save as")
        self.avg_save_name.clear()
        self.avg_save_name.setText(os.path.basename(save_name[0]))

    def init_average(self):
        if len(self.vk.target) > 0:
            save_path = self.avg_save_path.text()
            if save_path == "":
                self.avg_save_path.setText(self.work_dir.text())
            else:
                logger.info("use the previous save path")

            save_name = self.avg_save_name.text()
            save_name = "Avg" + self.vk.target[0]
            self.avg_save_name.setText(save_name)

    def submit_job(self):
        if len(self.vk.target) < 2:
            self.statusbar.showMessage("select at least 2 files for averaging", 1000)
            return

        self.thread_pool.setMaxThreadCount(self.max_thread_count.value())

        save_path = self.avg_save_path.text()
        save_name = self.avg_save_name.text()

        if not os.path.isdir(save_path):
            logger.info("the average save_path doesn't exist; creating one")
            try:
                os.mkdir(save_path)
            except OSError as e:
                logger.error("cannot create the folder %s: %s", save_path, e)
                return

        avg_fields = []
        if self.bx_avg_G2IPIF.isChecked():
            avg_fields.extend(["G2", "IP", "IF"])
        if self.bx_avg_g2g2err.isChecked():
            avg_fields.extend(["g2", "g2_err"])
        if self.bx_avg_saxs.isChecked():
            avg_fields.extend(["saxs_1d", "saxs_2d"])

        if len(avg_fields) == 0:
            self.statusbar.showMessage("No average field is selected. quit", 1000)
            return

        kwargs = {
            "save_path": os.path.join(save_path, save_name),
            "chunk_size": int(self.cb_avg_chunk_size.currentText()),
            "avg_blmin": self.avg_blmin.value(),
            "avg_blmax": self.avg_blmax.value(),
            "avg_qindex": self.avg_qindex.value(),
            "avg_window": self.avg_window.value(),
            "fields": avg_fields,
        }

        if kwargs["avg_blmax"] <= kwargs["avg_blmin"]:
            self.statusbar.showMessage("check avg min/max values.", 1000)
            return

        self.vk.submit_job(**kwargs)
        # the target_average has been reset
        self.update_box(self.vk.target, mode="target")

    def update_avg_info(self):
        index = self.avg_job_table.currentIndex().row()
        if index < 0 or index >= len(self.vk.avg_worker):
            self.statusbar.showMessage("select a job to start", 1000)
            return

        self.timer.stop()
        self.timer.setInterval(1000)

        try:
            self.timer.timeout.disconnect()
            logger.info("disconnect previous slot")
        except RuntimeError:
            # Timer already disconnected or destroyed
            pass

        worker = self.vk.avg_worker[index]
        worker.initialize_plot(self.mp_avg_g2)

        self.timer.timeout.connect(lambda x=index: self.vk.update_avg_info(x))
        self.timer.start()

    def start_avg_job(self):
        index = self.avg_job_table.currentIndex().row()
        if index < 0 or index >= len(self.vk.avg_worker):
            self.statusbar.showMessage("select a job to start", 1000)
            return
        worker = self.vk.avg_worker[index]
        if worker.status == "finished":
            self.statusbar.showMessage("this job has finished", 1000)
            return
        if worker.status == "running":
            self.statusbar.showMessage("this job is running.", 1000)
            return

        worker.signals.values.connect(self.vk.update_avg_values)
        self.thread_pool.start(worker)
        self.vk.avg_worker_active[worker.jid] = None

    def avg_kill_job(self):
        index = self.avg_job_table.currentIndex().row()
        if index < 0 or index >= len(self.vk.avg_worker):
            self.statusbar.showMessage("select a job to kill", 1000)
            return
        worker = self.vk.avg_worker[index]
        if worker.status != "running":
            self.statusbar.showMessage("the selected job isn's running", 1000)
            return
        worker.kill()

    def show_g2_fit_summary_func(self):
        rows = self.get_selected_rows()
        self.tree = self.vk.get_fitting_tree(rows)
        self.tree.show()

    def show_avg_jobinfo(self):
        index = self.avg_job_table.currentIndex().row()
        if index < 0 or index >= len(self.vk.avg_worker):
            logger.info("select a job to show it's settting")
            return
        worker = self.vk.avg_worker[index]
        self.tree = worker.get_pg_tree()
        self.tree.show()

    def init_g2(self, qd, tel):
        if qd is None or tel is None:
            return

        q_auto = self.g2_qauto.isChecked()
        t_auto = self.g2_tauto.isChecked()

        # tel is a list of arrays, which may have diffent shape;
        t_min = np.min([t[0] for t in tel])
        t_max = np.max([t[-1] for t in tel])

        def to_e(x):
            return f"{x:.2e}"

        self.g2_bmin.setValue(t_min / 20)
        self.g2_bmax.setValue(t_max * 10)

        if t_auto:
            self.g2_tmin.setText(to_e(t_min / 1.1))
            self.g2_tmax.setText(to_e(t_max * 1.1))

        if q_auto:
            self.g2_qmin.setValue(np.min(qd) / 1.1)
            self.g2_qmax.setValue(np.max(qd) * 1.1)

    def plot_g2(self, dryrun=False):
        """
        Plot G2 correlation functions with optional fitting.

        Generates multi-tau correlation function plots for selected files
        with configurable parameters including Q-range, time-range,
        fitting functions, and display options.

        Parameters
        ----------
        dryrun : bool, optional
            If True, returns parameters without plotting (default: False).

        Returns
        -------
        dict or tuple
            If dryrun=True, returns plot parameters dictionary.
            If dryrun=False, returns (q_values, time_delays) tuple.

        Notes
        -----
        - Supports single and double exponential fitting
        - Automatically initializes fitting parameters based on data range
        - Updates diffusion analysis tab when fitting is enabled
        - Validates fitting parameters and bounds before processing
        """
        p = self.check_g2_number()
        bounds, fit_flag, fit_func = self.check_g2_fitting_number()

        kwargs = {
            "num_col": self.sb_g2_column.value(),
            "offset": self.sb_g2_offset.value(),
            "show_fit": self.g2_show_fit.isChecked(),
            "show_label": self.g2_show_label.isChecked(),
            "plot_type": self.g2_plot_type.currentText(),
            "q_range": (p[0], p[1]),
            "t_range": (p[2], p[3]),
            "y_range": (p[4], p[5]),
            "y_auto": self.g2_yauto.isChecked(),
            "q_auto": self.g2_qauto.isChecked(),
            "t_auto": self.g2_tauto.isChecked(),
            "rows": self.get_selected_rows(),
            "bounds": bounds,
            "fit_flag": fit_flag,
            "marker_size": self.g2_marker_size.value(),
            "subtract_baseline": self.g2_sub_baseline.isChecked(),
            "fit_func": fit_func,
            "robust_fitting": True,  # Always use robust sequential fitting for better reliability
        }
        if kwargs["show_fit"] and sum(kwargs["fit_flag"]) == 0:
            self.statusbar.showMessage("nothing to fit, really?", 1000)
            return None

        if dryrun:
            return kwargs
        self.pushButton_4.setDisabled(True)
        self.pushButton_4.setText("plotting")
        try:
            qd, tel = self.vk.plot_g2(handler=self.mp_g2, **kwargs)
            self.init_g2(qd, tel)
            if kwargs["show_fit"]:
                self.init_diffusion()
        except Exception:
            traceback.print_exc()
        finally:
            self.pushButton_4.setEnabled(True)
            self.pushButton_4.setText("plot")

    def refit_g2(self):
        """
        Refit G2 correlation functions with force_refit=True to bypass cache.

        This method forces a complete refitting of G2 data, ensuring that
        parameter values are updated even if fitting parameters haven't changed.
        """
        logger.info("Refit G2 button clicked - forcing new fit calculation")

        p = self.check_g2_number()
        bounds, fit_flag, fit_func = self.check_g2_fitting_number()

        kwargs = {
            "num_col": self.sb_g2_column.value(),
            "offset": self.sb_g2_offset.value(),
            "show_fit": self.g2_show_fit.isChecked(),
            "show_label": self.g2_show_label.isChecked(),
            "plot_type": self.g2_plot_type.currentText(),
            "q_range": (p[0], p[1]),
            "t_range": (p[2], p[3]),
            "y_range": (p[4], p[5]),
            "y_auto": self.g2_yauto.isChecked(),
            "q_auto": self.g2_qauto.isChecked(),
            "t_auto": self.g2_tauto.isChecked(),
            "rows": self.get_selected_rows(),
            "bounds": bounds,
            "fit_flag": fit_flag,
            "marker_size": self.g2_marker_size.value(),
            "subtract_baseline": self.g2_sub_baseline.isChecked(),
            "fit_func": fit_func,
            "robust_fitting": True,
            "force_refit": True,  # KEY: Force refitting to bypass cache
        }

        if kwargs["show_fit"] and sum(kwargs["fit_flag"]) == 0:
            self.statusbar.showMessage("nothing to fit, really?", 1000)
            return

        self.btn_g2_refit.setDisabled(True)
        self.btn_g2_refit.setText("refitting")
        try:
            qd, tel = self.vk.plot_g2(handler=self.mp_g2, **kwargs)
            self.init_g2(qd, tel)
            if kwargs["show_fit"]:
                self.init_diffusion()
            self.statusbar.showMessage("G2 refitting completed successfully", 2000)
        except Exception:
            traceback.print_exc()
            self.statusbar.showMessage("G2 refitting failed", 2000)
        finally:
            self.btn_g2_refit.setEnabled(True)
            self.btn_g2_refit.setText("refit")

    def export_g2(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="select a folder to export G2 data and fitting results"
        )

        if folder in [None, ""]:
            return

        try:
            rows = self.get_selected_rows()
            logger.info(f"G2 export: Selected rows: {rows}")

            # Check if there are any files in the target list at all
            all_available_files = self.vk.get_xf_list()
            logger.info(
                f"G2 export: Total files available in target list: {len(all_available_files)}"
            )

            if not rows:
                if not all_available_files:
                    self.statusbar.showMessage(
                        "No files loaded. Please add XPCS files to the target list first.",
                        5000,
                    )
                    logger.warning(
                        "G2 export: No files in target list. User needs to load XPCS files first."
                    )
                else:
                    # Show helpful message about file selection
                    multitau_available = self.vk.get_xf_list(filter_atype="Multitau")
                    if multitau_available:
                        self.statusbar.showMessage(
                            f"Please select files for export. {len(multitau_available)} Multitau files available in target list.",
                            5000,
                        )
                        logger.warning(
                            f"G2 export: No files selected. {len(multitau_available)} Multitau files available for export."
                        )
                    else:
                        self.statusbar.showMessage(
                            "No Multitau files found. G2 export requires Multitau analysis files.",
                            5000,
                        )
                        logger.warning(
                            f"G2 export: No Multitau files available. Found {len(all_available_files)} files but none are Multitau type."
                        )
                return

            # Additional diagnostics
            all_files = self.vk.get_xf_list(rows=rows)
            multitau_files = self.vk.get_xf_list(rows=rows, filter_atype="Multitau")
            logger.info(f"G2 export: Total files from selection: {len(all_files)}")
            logger.info(
                f"G2 export: Multitau files from selection: {len(multitau_files)}"
            )
            for i, xf in enumerate(all_files):
                logger.info(
                    f"G2 export: File {i}: {xf.label} - atype: {getattr(xf, 'atype', 'unknown')}"
                )

            # Check if selected files include any Multitau files
            if not multitau_files:
                # Check if there are any Multitau files available but not selected
                all_multitau_available = self.vk.get_xf_list(filter_atype="Multitau")
                if all_multitau_available:
                    self.statusbar.showMessage(
                        f"Selected files are not Multitau type. Please select from {len(all_multitau_available)} available Multitau files.",  # nosec B608
                        5000,
                    )
                    logger.warning(
                        f"G2 export: Selected {len(all_files)} files but none are Multitau. {len(all_multitau_available)} Multitau files available."
                    )
                else:
                    self.statusbar.showMessage(
                        "No Multitau files available for G2 export.", 5000
                    )
                    logger.warning(
                        "G2 export: No Multitau files available in entire target list."
                    )
                return

            self.vk.export_g2(folder, rows)
            self.statusbar.showMessage(f"G2 data exported to {folder}", 5000)
        except ValueError as e:
            # Handle data validation errors with specific guidance
            if "No files with G2 data found" in str(e):
                self.statusbar.showMessage(
                    "No files with G2 correlation data found. Please select multitau analysis files.",
                    5000,
                )
            elif "No valid Q indices" in str(e):
                self.statusbar.showMessage(
                    "Export failed: Q-range selection resulted in invalid indices. Try adjusting Q-range.",
                    5000,
                )
            else:
                self.statusbar.showMessage(f"Data validation error: {e}", 5000)
            logger.error(f"G2 export validation error: {e}")
        except (OSError, PermissionError) as e:
            # Handle file system errors
            self.statusbar.showMessage(
                f"File system error: Cannot write to {folder}. Check permissions.", 5000
            )
            logger.error(f"G2 export file system error: {e}")
        except Exception as e:
            # Handle unexpected errors
            self.statusbar.showMessage(
                f"Export failed with unexpected error: {type(e).__name__}: {e}", 5000
            )
            logger.error(f"G2 export unexpected error: {e}", exc_info=True)

    def export_diffusion(self):
        """Export power law fitting results from diffusion analysis."""
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, caption="Select folder to export diffusion fitting results"
        )

        if folder in [None, ""]:
            return

        try:
            rows = self.get_selected_rows()
            if not rows:
                self.statusbar.showMessage("No files selected for export", 3000)
                return

            self.vk.export_diffusion(folder, rows)
            self.statusbar.showMessage(
                f"Diffusion fitting results exported to {folder}", 5000
            )
        except ValueError as e:
            # Handle data validation errors with specific guidance
            if "No files with tau-q fitting results found" in str(e):
                self.statusbar.showMessage(
                    "No files with tau-q fitting results found. Please select multitau analysis files.",
                    5000,
                )
            elif "No files with tau-q power law fitting results found" in str(e):
                self.statusbar.showMessage(
                    "No diffusion analysis results found. Please perform diffusion analysis first in the Diffusion tab.",
                    5000,
                )
            else:
                self.statusbar.showMessage(f"Data validation error: {e}", 5000)
            logger.error(f"Diffusion export validation error: {e}")
        except (OSError, PermissionError) as e:
            # Handle file system errors
            self.statusbar.showMessage(
                f"File system error: Cannot write to {folder}. Check permissions.", 5000
            )
            logger.error(f"Diffusion export file system error: {e}")
        except Exception as e:
            # Handle unexpected errors
            self.statusbar.showMessage(
                f"Export failed with unexpected error: {type(e).__name__}: {e}", 5000
            )
            logger.error(f"Diffusion export unexpected error: {e}", exc_info=True)

    def reload_source(self):
        self.pushButton_11.setText("loading")
        self.pushButton_11.setDisabled(True)
        self.pushButton_11.parent().repaint()
        path = self.work_dir.text()
        self.vk.build(path=path, sort_method=self.sort_method.currentText())
        self.pushButton_11.setText("reload")
        self.pushButton_11.setEnabled(True)
        self.pushButton_11.parent().repaint()

        self.update_box(self.vk.source, mode="source")
        self.apply_filter_to_source()

    def load_path(self, path=None, debug=False):
        """
        Load XPCS data from specified directory path.

        This method initializes the viewer kernel with the specified path,
        builds the file list, and sets up the GUI for data analysis.

        Parameters
        ----------
        path : str, optional
            Directory path containing XPCS data files. If None, opens file dialog.
        debug : bool, optional
            Enable debug mode for additional logging (default: False).

        Notes
        -----
        - Creates new ViewerKernel instance or resets existing one
        - Initializes async kernel for background processing
        - Updates source and target file lists in GUI
        - Sets up averaging job table model
        """
        if path in [None, False]:
            # DontUseNativeDialog is used so files are shown along with dirs;
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                "Open directory",
                self.start_wd,
                QtWidgets.QFileDialog.DontUseNativeDialog,
            )
        else:
            folder = path

        if not os.path.isdir(folder):
            self.statusbar.showMessage(f"{folder} is not a folder.")
            folder = self.start_wd

        self.work_dir.setText(folder)

        if self.vk is None:
            self.vk = ViewerKernel(folder, self.statusbar)
        else:
            self.vk.set_path(folder)
            self.vk.clear()

        # Initialize async kernel after viewer kernel is ready
        self.init_async_kernel()

        self.reload_source()
        self.avg_job_table.setModel(self.vk.avg_worker)
        self.source_model = self.vk.source
        self.update_box(self.vk.source, mode="source")

        # Trigger plot update to show proper empty states since no files are auto-added
        self.update_plot()

    def update_box(self, file_list, mode="source"):
        if file_list is None:
            return
        if mode == "source":
            self.list_view_source.setModel(file_list)
            self.box_source.setTitle(f"Source: {len(file_list):5d}")
            self.box_source.parent().repaint()
            self.list_view_source.parent().repaint()
        elif mode == "target":
            self.list_view_target.setModel(file_list)
            self.box_target.setTitle(f"Target: {len(file_list):5d}")
            # on macos, the target box doesn't seem to update; force it
            file_list.layoutChanged.emit()
            self.box_target.repaint()
            self.list_view_target.repaint()
            max_size = len(file_list) - 1
            self.horizontalSlider_saxs2d_selection.setMaximum(max_size)
            self.horizontalSlider_saxs2d_selection.setValue(0)
            self.spinBox_saxs2d_selection.setMaximum(max_size)
            self.spinBox_saxs2d_selection.setValue(0)
        self.statusbar.showMessage("Target file list updated.", 1000)
        return

    def add_target(self):
        target = []
        for x in self.list_view_source.selectedIndexes():
            # in some cases, it will return None
            val = x.data()
            if val is not None:
                target.append(val)
        if target == []:
            return

        tab_id = self.tabWidget.currentIndex()
        tab_name = tab_mapping[tab_id]
        preload = tab_name != "average"
        self.vk.add_target(target, preload=preload)
        self.list_view_source.clearSelection()
        self.update_box(self.vk.target, mode="target")

        # Update tab availability based on file formats
        self.update_tab_availability()

        if tab_name == "average":
            self.init_average()
        else:
            self.update_plot()

    def update_tab_availability(self):
        """
        Update tab availability based on the file formats in the target list.

        Disables incompatible tabs to prevent application freezing:
        - Multitau format: Disables "Two Time" tab
        - Twotime format: Disables "g2" and "Diffusion" tabs
        - Mixed formats: Enables all tabs with warning
        """
        if not self.vk.target or len(self.vk.target) == 0:
            # No files selected, enable all tabs
            self._enable_all_tabs()
            self.statusbar.showMessage("No files selected - all tabs available", 3000)
            return

        # Analyze file formats in target list
        file_formats = set()
        multitau_count = 0
        twotime_count = 0

        try:
            # Get XpcsFile objects to check their analysis types
            xf_list = self.vk.get_xf_list()
            if not xf_list:
                self._enable_all_tabs()
                return

            for xf in xf_list:
                if hasattr(xf, "atype") and xf.atype:
                    for atype in xf.atype:
                        file_formats.add(atype)
                        if atype == "Multitau":
                            multitau_count += 1
                        elif atype == "Twotime":
                            twotime_count += 1

            logger.debug(
                f"File format analysis: {file_formats}, Multitau: {multitau_count}, Twotime: {twotime_count}"
            )

            # Determine tab availability based on formats
            if len(file_formats) == 0:
                # No recognized formats, enable all tabs
                self._enable_all_tabs()
                self.statusbar.showMessage(
                    "Warning: File format not recognized - proceed with caution", 5000
                )

            elif len(file_formats) == 1:
                # Single format detected
                format_type = next(iter(file_formats))
                if format_type == "Multitau":
                    self._configure_for_multitau()
                    self.statusbar.showMessage(
                        f"Multitau format detected - Two Time tab disabled ({multitau_count} files)",
                        5000,
                    )
                elif format_type == "Twotime":
                    self._configure_for_twotime()
                    self.statusbar.showMessage(
                        f"Twotime format detected - G2 and Diffusion tabs disabled ({twotime_count} files)",
                        5000,
                    )
                else:
                    self._enable_all_tabs()

            else:
                # Mixed formats detected
                self._enable_all_tabs()
                self.statusbar.showMessage(
                    f"Warning: Mixed formats detected (Multitau: {multitau_count}, Twotime: {twotime_count}) - use tabs carefully",
                    8000,
                )

        except Exception as e:
            logger.error(f"Error analyzing file formats for tab management: {e}")
            self._enable_all_tabs()
            self.statusbar.showMessage(
                "Error analyzing file formats - all tabs enabled", 5000
            )

    def _enable_all_tabs(self):
        """Enable all tabs."""
        for tab_index in range(self.tabWidget.count()):
            self.tabWidget.setTabEnabled(tab_index, True)
        self._clear_tab_tooltips()  # Clear tooltips when all tabs are enabled
        logger.debug("All tabs enabled")

    def _configure_for_multitau(self):
        """Configure tabs for multitau format: disable Two Time tab."""
        self._enable_all_tabs()  # First enable all
        self._clear_tab_tooltips()  # Clear any existing tooltips

        # Disable Two Time tab (index 6)
        twotime_tab_index = 6
        self.tabWidget.setTabEnabled(twotime_tab_index, False)
        self.tabWidget.setTabToolTip(
            twotime_tab_index,
            "Two Time analysis not available for multitau format files",
        )

        # If user is currently on Two Time tab, switch to G2 tab
        if self.tabWidget.currentIndex() == twotime_tab_index:
            self.tabWidget.setCurrentIndex(4)  # Switch to G2 tab
            logger.info("Switched from Two Time to G2 tab (multitau format)")

        logger.debug("Configured tabs for multitau format: Two Time tab disabled")

    def _configure_for_twotime(self):
        """Configure tabs for twotime format: disable G2 and Diffusion tabs."""
        self._enable_all_tabs()  # First enable all
        self._clear_tab_tooltips()  # Clear any existing tooltips

        # Disable G2 tab (index 4) and Diffusion tab (index 5)
        g2_tab_index = 4
        diffusion_tab_index = 5
        self.tabWidget.setTabEnabled(g2_tab_index, False)
        self.tabWidget.setTabEnabled(diffusion_tab_index, False)
        self.tabWidget.setTabToolTip(
            g2_tab_index, "G2 analysis not available for twotime format files"
        )
        self.tabWidget.setTabToolTip(
            diffusion_tab_index,
            "Diffusion analysis not available for twotime format files",
        )

        # If user is currently on disabled tab, switch to Two Time tab
        current_tab = self.tabWidget.currentIndex()
        if current_tab in [g2_tab_index, diffusion_tab_index]:
            self.tabWidget.setCurrentIndex(6)  # Switch to Two Time tab
            logger.info("Switched from disabled tab to Two Time tab (twotime format)")

        logger.debug(
            "Configured tabs for twotime format: G2 and Diffusion tabs disabled"
        )

    def _clear_tab_tooltips(self):
        """Clear all tab tooltips."""
        for tab_index in range(self.tabWidget.count()):
            self.tabWidget.setTabToolTip(tab_index, "")

    def reorder_target(self, direction="up"):
        rows = self.get_selected_rows()
        if len(rows) != 1 or len(self.vk.target) <= 1:
            return
        idx = self.vk.reorder_target(rows[0], direction)
        self.list_view_target.setCurrentIndex(idx)
        self.list_view_target.repaint()
        self.update_plot()
        return

    def remove_target(self):
        rmv_list = []
        for x in self.list_view_target.selectedIndexes():
            rmv_list.append(x.data())

        self.vk.remove_target(rmv_list)
        # clear selection to avoid the bug: when the last one is selected, then
        # the list will out of bounds
        self.clear_target_selection()

        # Update tab availability after removing files
        self.update_tab_availability()

        # if all files are removed; then go to state 1
        if self.vk.target in [[], None] or len(self.vk.target) == 0:
            self.reset_gui()
        self.update_box(self.vk.target, mode="target")

    def reset_gui(self):
        self.vk.reset_kernel()
        for x in [
            # self.pg_saxs,
            self.pg_intt,
            self.mp_tauq,
            self.mp_g2,
            self.mp_saxs,
            self.mp_stab,
        ]:
            x.clear()
        self.le_bkg_fname.clear()

        # Enable all tabs when no files are loaded
        self._enable_all_tabs()

    def apply_filter_to_source(self):
        min_length = 1
        val = self.filter_str.text()
        if len(val) == 0:
            self.source_model = self.vk.source
            self.update_box(self.vk.source, mode="source")
            return
        # avoid searching when the filter lister is too short
        if len(val) < min_length:
            self.statusbar.showMessage(
                f"Please enter at least {min_length} characters", 1000
            )
            return

        filter_type = ["prefix", "substr"][self.filter_type.currentIndex()]
        self.vk.search(val, filter_type)
        self.source_model = self.vk.source_search
        self.update_box(self.source_model, mode="source")
        self.list_view_source.selectAll()

    def check_g2_number(self, default_val=(0, 0.0092, 1e-8, 1, 0.95, 1.35)):
        keys = (
            self.g2_qmin,
            self.g2_qmax,
            self.g2_tmin,
            self.g2_tmax,
            self.g2_ymin,
            self.g2_ymax,
        )
        vals = [None] * len(keys)
        for n, key in enumerate(keys):
            if isinstance(key, QtWidgets.QDoubleSpinBox):
                val = key.value()
            elif isinstance(key, QtWidgets.QLineEdit):
                try:
                    val = float(key.text())
                except Exception:
                    key.setText(str(default_val[n]))
                    self.statusbar.showMessage("g2 number is invalid", 1000)
            vals[n] = val

        def swap_min_max(id1, id2):
            if vals[id1] > vals[id2]:
                keys[id1].setValue(vals[id2])
                keys[id2].setValue(vals[id1])
                vals[id1], vals[id2] = vals[id2], vals[id1]

        swap_min_max(0, 1)
        swap_min_max(4, 5)

        return vals

    def check_g2_fitting_number(self):
        fit_func = ["single", "double"][self.g2_fitting_function.currentIndex()]
        keys = (
            self.g2_amin,
            self.g2_amax,
            self.g2_bmin,
            self.g2_bmax,
            self.g2_cmin,
            self.g2_cmax,
            self.g2_dmin,
            self.g2_dmax,
            self.g2_b2min,
            self.g2_b2max,
            self.g2_c2min,
            self.g2_c2max,
            self.g2_fmin,
            self.g2_fmax,
        )

        vals = [None] * len(keys)
        for n, key in enumerate(keys):
            vals[n] = key.value()

        def swap_min_max(id1, id2):
            if vals[id1] > vals[id2]:
                keys[id1].setValue(vals[id2])
                keys[id2].setValue(vals[id1])
                vals[id1], vals[id2] = vals[id2], vals[id1]

        for n in range(7):
            swap_min_max(2 * n, 2 * n + 1)

        vals = np.array(vals).reshape(len(keys) // 2, 2)
        bounds = vals.T

        fit_keys = (
            self.g2_afit,
            self.g2_bfit,
            self.g2_cfit,
            self.g2_dfit,
            self.g2_b2fit,
            self.g2_c2fit,
            self.g2_ffit,
        )
        fit_flag = [x.isChecked() for x in fit_keys]

        if fit_func in {"single", "robust"}:
            fit_flag = fit_flag[0:4]
            bounds = bounds[:, 0:4]
        bounds = bounds.tolist()
        return bounds, fit_flag, fit_func

    def update_saxs2d_level(self, flag=True):
        if not flag:
            vmin = self.pg_saxs.levelMin
            vmax = self.pg_saxs.levelMax
            if vmin is not None:
                self.saxs2d_min.setValue(vmin)
            if vmax is not None:
                self.saxs2d_max.setValue(vmax)
            self.saxs2d_min.setEnabled(True)
            self.saxs2d_max.setEnabled(True)
        else:
            self.saxs2d_min.setDisabled(True)
            self.saxs2d_max.setDisabled(True)

        self.saxs2d_min.parent().repaint()

    def clear_target_selection(self):
        self.list_view_target.clearSelection()

    def update_g2_fitting_function(self):
        idx = self.g2_fitting_function.currentIndex()
        title = [
            "g2 fitting with Single Exp:  y = a·exp[-2(x/b)^c]+d",
            "g2 fitting with Double Exp:  y = a·[f·exp[-(x/b)^c +"
            + "(1-f)·exp[-(x/b2)^c2]^2+d",
        ]
        self.groupBox_2.setTitle(title[idx])

        pvs = [
            [self.g2_b2min, self.g2_b2max, self.g2_b2fit],
            [self.g2_c2min, self.g2_c2max, self.g2_c2fit],
            [self.g2_fmin, self.g2_fmax, self.g2_ffit],
        ]

        # change to single
        if idx == 0:  # single
            for n in range(3):
                pvs[n][0].setDisabled(True)
                pvs[n][1].setDisabled(True)
                pvs[n][2].setDisabled(True)
        # change to double
        elif idx == 1:  # double
            for n in range(3):
                pvs[n][2].setEnabled(True)
                pvs[n][1].setEnabled(True)
                if pvs[n][2].isChecked():
                    pvs[n][0].setEnabled(True)


def setup_windows_icon():
    # reference: https://stackoverflow.com/questions/1551605
    import ctypes
    from ctypes import wintypes

    lpBuffer = wintypes.LPWSTR()
    AppUserModelID = ctypes.windll.shell32.GetCurrentProcessExplicitAppUserModelID
    AppUserModelID(ctypes.cast(ctypes.byref(lpBuffer), wintypes.LPWSTR))
    appid = lpBuffer.value
    ctypes.windll.kernel32.LocalFree(lpBuffer)
    if appid is None:
        appid = "aps.xpcs_viewer.viewer.0.20"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(appid)


def main_gui(path=None, label_style=None):
    if os.name == "nt":
        setup_windows_icon()

    logger.info("Starting XPCS Toolkit GUI application")
    if path:
        logger.info(f"Starting with path: {path}")

    # Set Qt configuration before creating application
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    # Qt warnings are handled by the logging system automatically
    logger.debug("Qt application attributes configured")

    app = QtWidgets.QApplication([])
    logger.info("Qt Application created")

    try:
        XpcsViewer(path=path, label_style=label_style)
        logger.info("XpcsViewer window created successfully")
        app.exec_()
    except Exception as e:
        logger.critical(f"Failed to start application: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutting down")


if __name__ == "__main__":
    main_gui()
