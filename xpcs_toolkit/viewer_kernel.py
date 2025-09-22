# Standard library imports
import os
import weakref
from types import ModuleType
from typing import Any

# Third-party imports
import numpy as np
import pyqtgraph as pg

# Local imports
from .file_locator import FileLocator
from .helper.listmodel import TableDataModel
from .utils.logging_config import get_logger
from .xpcs_file import MemoryMonitor, XpcsFile

# Lazy-loaded modules cache
_module_cache: dict[str, ModuleType] = {}


def _get_module(module_name: str) -> ModuleType:
    """Lazy load analysis modules to improve startup time"""
    if module_name not in _module_cache:
        if module_name == "g2mod":
            from .module import g2mod

            _module_cache[module_name] = g2mod
        elif module_name == "intt":
            from .module import intt

            _module_cache[module_name] = intt
        elif module_name == "saxs1d":
            from .module import saxs1d

            _module_cache[module_name] = saxs1d
        elif module_name == "saxs2d":
            from .module import saxs2d

            _module_cache[module_name] = saxs2d
        elif module_name == "stability":
            from .module import stability

            _module_cache[module_name] = stability
        elif module_name == "tauq":
            from .module import tauq

            _module_cache[module_name] = tauq
        elif module_name == "twotime":
            from .module import twotime

            _module_cache[module_name] = twotime
        elif module_name == "average_toolbox":
            from .module.average_toolbox import AverageToolbox

            _module_cache[module_name] = AverageToolbox
        else:
            raise ImportError(f"Unknown module: {module_name}")
    return _module_cache[module_name]


logger = get_logger(__name__)


class ViewerKernel(FileLocator):
    """
    Backend kernel for XPCS data processing and analysis coordination.

    ViewerKernel serves as the central coordinator between the GUI and various
    analysis modules. It manages file collections, handles data caching,
    coordinates background processing, and provides a unified interface
    for all analysis operations with lazy-loaded modules for improved startup performance.

    The kernel inherits from FileLocator to provide file management
    capabilities and extends it with advanced caching, memory management,
    and analysis coordination features.

    Parameters
    ----------
    path : str
        Base directory path for XPCS data files
    statusbar : QStatusBar, optional
        GUI status bar for displaying operation messages

    Attributes
    ----------
    meta : dict
        Metadata storage for analysis parameters and cached results
    avg_tb : AverageToolbox
        Toolbox for file averaging operations
    avg_worker : TableDataModel
        Model for managing averaging job queue

    Notes
    -----
    The kernel implements memory-aware caching using weak references
    and automatic cleanup mechanisms to handle large XPCS datasets
    efficiently. It supports both synchronous and asynchronous
    operation modes for optimal GUI responsiveness.

    Examples
    --------
    >>> kernel = ViewerKernel("/path/to/data")
    >>> kernel.plot_saxs_2d(plot_handler, rows=[0, 1, 2])
    """

    def get_module(self, module_name: str) -> ModuleType:
        """Public interface to lazy load analysis modules"""
        return _get_module(module_name)

    def __init__(self, path: str, statusbar: Any | None = None) -> None:
        super().__init__(path)
        self.statusbar = statusbar
        self.meta = None
        self.reset_meta()
        self.path = path
        self.avg_tb = _get_module("average_toolbox")(path)
        self.avg_worker = TableDataModel()
        self.avg_jid = 0
        self.avg_worker_active = {}

        # Memory-aware caching for current_dset with weak references
        self._current_dset_cache: weakref.WeakValueDictionary[str, XpcsFile] = (
            weakref.WeakValueDictionary()
        )
        self._plot_kwargs_record: dict[str, Any] = {}
        self._memory_cleanup_threshold = 0.8  # Trigger cleanup at 80% memory usage

    def reset_meta(self) -> None:
        """
        Reset metadata dictionary to default values.

        Initializes the metadata storage with default values for
        various analysis parameters and cached results.

        Returns
        -------
        None

        Notes
        -----
        The metadata dictionary stores analysis parameters including:
        - SAXS 1D background file information
        - Averaging file lists and parameters
        - G2 analysis results
        - Other analysis-specific cached data
        """
        self.meta = {
            # saxs 1d:
            "saxs1d_bkg_fname": None,
            "saxs1d_bkg_xf": None,
            # avg
            "avg_file_list": None,
            "avg_intt_minmax": None,
            "avg_g2_avg": None,
            # g2
        }

    def reset_kernel(self) -> None:
        """
        Reset kernel state and clear all cached data.

        Performs complete kernel reset including target file list clearing,
        metadata reset, and comprehensive memory cleanup operations.

        Notes
        -----
        - Clears target file selection
        - Resets all metadata to default values
        - Triggers comprehensive memory cleanup
        - Suitable for starting fresh analysis or recovering from errors

        This method should be called when switching to a new dataset
        or when memory cleanup is required.
        """
        self.clear_target()
        self.reset_meta()
        self._cleanup_memory()

    def _cleanup_memory(self) -> None:
        """
        Clean up cached data and release memory resources.

        Performs comprehensive memory cleanup including dataset cache clearing,
        plot parameter cache management, and XpcsFile cache cleanup.
        Uses optimized cleanup strategies when available.

        Notes
        -----
        - Clears current dataset cache using weak references
        - Conditionally clears plot kwargs based on memory pressure
        - Schedules background cleanup for XpcsFile objects
        - Falls back to direct cleanup methods if optimized system unavailable
        - Implements timeout protection to prevent excessive blocking
        - Logs memory usage before and after cleanup operations

        The method is automatically called when memory pressure exceeds
        the configured threshold or when explicitly requested during
        kernel reset operations.
        """
        """Clean up cached data and release memory."""
        # Clear dataset cache
        self._current_dset_cache.clear()

        # Clear plot kwargs if memory pressure is high
        if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
            self._plot_kwargs_record.clear()
            logger.info("Memory pressure detected, cleared plot kwargs cache")

        # Schedule background cleanup for XpcsFile objects (non-blocking)
        try:
            from .threading.cleanup_optimized import (
                CleanupPriority,
                schedule_type_cleanup,
            )

            schedule_type_cleanup("XpcsFile", CleanupPriority.HIGH)
            logger.debug("Scheduled background cache clearing for XpcsFile objects")
        except ImportError:
            # Fallback to direct cleanup if optimized system not available
            try:
                from .threading.cleanup_optimized import get_object_registry

                registry = get_object_registry()
                xpcs_files = registry.get_objects_by_type("XpcsFile")
                for xf in xpcs_files:
                    xf.clear_cache()
                logger.debug(
                    f"Directly cleared cache for {len(xpcs_files)} XpcsFile objects"
                )
            except Exception as e:
                logger.debug(f"Direct cache clearing failed, using fallback: {e}")
                # Final fallback: limited traversal with timeout
                import gc
                import time

                start_time = time.time()
                cleared_count = 0
                for obj in gc.get_objects():
                    if isinstance(obj, XpcsFile):
                        obj.clear_cache()
                        cleared_count += 1
                    # Prevent excessive blocking
                    if time.time() - start_time > 0.050:  # 50ms limit
                        logger.warning("Memory cleanup timeout, stopped early")
                        break
                logger.debug(
                    f"Fallback: cleared cache for {cleared_count} XpcsFile objects"
                )

        used_mb, _available_mb = MemoryMonitor.get_memory_usage()
        logger.debug(f"Memory cleanup completed, current usage: {used_mb:.1f}MB")

    def select_bkgfile(self, fname: str) -> None:
        """
        Select background file for SAXS 1D background subtraction.

        Sets up background file for subtraction in SAXS 1D analysis
        by creating an XpcsFile object and storing metadata.

        Parameters
        ----------
        fname : str
            Full path to the background file

        Notes
        -----
        The background file should have compatible Q-range and data
        structure with the analysis files for proper subtraction.
        """
        base_fname = os.path.basename(fname)
        self.meta["saxs1d_bkg_fname"] = base_fname
        self.meta["saxs1d_bkg_xf"] = XpcsFile(fname)

    def get_pg_tree(self, rows: Any) -> Any | None:
        xf_list = self.get_xf_list(rows)
        if xf_list:
            return xf_list[0].get_pg_tree()
        return None

    def get_fitting_tree(self, rows: Any) -> Any | None:
        xf_list = self.get_xf_list(rows, filter_atype="Multitau")
        result = {}
        for x in xf_list:
            result[x.label] = x.get_fitting_info(mode="g2_fitting")
        tree = pg.DataTreeWidget(data=result)
        tree.setWindowTitle("fitting summary")
        tree.resize(1024, 800)
        return tree

    def plot_g2(self, handler, q_range, t_range, y_range, rows=None, **kwargs):
        """
        Generate G2 correlation function plots for multi-tau analysis.

        Creates correlation function plots for selected files with specified
        Q-range and time-range parameters. Supports fitting with single or
        double exponential functions.

        Parameters
        ----------
        handler : PlotHandler
            Plot handler object for rendering the correlation functions
        q_range : tuple
            Q-value range as (q_min, q_max) in inverse Angstroms
        t_range : tuple
            Time delay range as (t_min, t_max) in seconds
        y_range : tuple
            Y-axis range for correlation function values
        rows : list, optional
            List of file indices to include in analysis
        **kwargs
            Additional plotting parameters (fitting options, display settings)

        Returns
        -------
        tuple
            (q_values, time_delays) arrays for successful plots, (None, None) if no data

        Notes
        -----
        Only processes files with "Multitau" analysis type. The method
        delegates actual plotting to the g2mod module while providing
        data preprocessing and validation.
        """
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        if xf_list:
            g2_module = self.get_module("g2mod")
            g2_module.pg_plot(handler, xf_list, q_range, t_range, y_range, **kwargs)
            q, tel, *_unused = g2_module.get_data(xf_list)
            return q, tel
        return None, None

    def plot_qmap(self, hdl, rows=None, target=None):
        xf_list = self.get_xf_list(rows=rows)
        if xf_list:
            if target == "scattering":
                value = np.log10(xf_list[0].saxs_2d + 1)
                vmin, vmax = np.percentile(value, (2, 98))
                hdl.setImage(value, levels=(vmin, vmax))
            elif target == "dynamic_roi_map":
                hdl.setImage(xf_list[0].dqmap)
            elif target == "static_roi_map":
                hdl.setImage(xf_list[0].sqmap)

    def plot_tauq_pre(self, hdl=None, rows=None):
        # Support both Multitau and Twotime files for diffusion analysis
        logger.debug(f"plot_tauq_pre called with hdl={hdl is not None}, rows={rows}")

        xf_list = self.get_xf_list(rows=rows)
        logger.debug(f"Got {len(xf_list)} files from get_xf_list")

        # Filter for files that support G2 analysis (same as G2 plotting)
        g2_compatible_list = []
        for xf in xf_list:
            if any(atype in xf.atype for atype in ["Multitau", "Twotime"]):
                g2_compatible_list.append(xf)
        xf_list = g2_compatible_list
        logger.debug(f"Filtered to {len(xf_list)} G2-compatible files")

        short_list = [xf for xf in xf_list if xf.fit_summary is not None]
        logger.debug(f"Found {len(short_list)} files with fit_summary")

        if len(short_list) == 0:
            logger.warning(
                f"No files with G2 fitting results found for diffusion analysis. "
                f"Found {len(xf_list)} G2-compatible files but none have been fitted."
            )
            # Clear the plot and show a message
            if hdl is not None:
                logger.debug("Clearing plot and showing empty message")
                hdl.clear()
                ax = hdl.subplots(1, 1)
                ax.text(
                    0.5,
                    0.5,
                    'No G2 fitting results available.\n\nPlease:\n1. Go to "g2" tab\n2. Select files\n3. Click "update plot" to perform G2 fitting\n4. Return to "Diffusion" tab',
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "lightblue",
                        "alpha": 0.7,
                    },
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                hdl.fig.tight_layout()
                hdl.draw()  # Add missing draw call
                logger.debug("Empty message plot drawn")
            else:
                logger.warning("hdl is None, cannot display empty message")
        else:
            logger.info(
                f"Found {len(short_list)} files with G2 fitting results for diffusion analysis"
            )
            if hdl is not None:
                logger.debug("Calling tauq.plot_pre with files")
                _get_module("tauq").plot_pre(short_list, hdl)
                logger.debug("tauq.plot_pre completed")
            else:
                logger.warning("hdl is None, cannot call tauq.plot_pre")

    def plot_tauq(
        self,
        hdl=None,
        bounds=None,
        rows=None,
        plot_type=3,
        fit_flag=None,
        offset=None,
        q_range=None,
        force_refit=False,
    ):
        logger.debug(
            f"plot_tauq called with hdl={hdl is not None}, bounds={bounds}, rows={rows}, plot_type={plot_type}, fit_flag={fit_flag}, offset={offset}, q_range={q_range}"
        )
        if rows is None:
            rows = []
        xf_list = self.get_xf_list(
            rows=rows, filter_atype="Multitau", filter_fitted=True
        )
        result = {}
        for x in xf_list:
            if x.fit_summary is None:
                logger.info("g2 fitting is not available for %s", x.fname)
            else:
                x.fit_tauq(q_range, bounds, fit_flag, force_refit=force_refit)
                result[x.label] = x.get_fitting_info(mode="tauq_fitting")

        if len(result) > 0:
            _get_module("tauq").plot(
                xf_list, hdl=hdl, q_range=q_range, offset=offset, plot_type=plot_type
            )

        return result

    def get_info_at_mouse(self, rows, x, y):
        xf = self.get_xf_list(rows)
        if xf:
            info = xf[0].get_info_at_position(x, y)
            return info
        return None

    def plot_saxs_2d(self, *args, rows=None, **kwargs):
        """
        Generate SAXS 2D scattering pattern visualization.

        Creates 2D scattering pattern plots for the first selected file
        with configurable display parameters including colormaps,
        rotation, and intensity scaling.

        Parameters
        ----------
        *args
            Variable positional arguments passed to saxs2d.plot()
        rows : list, optional
            List of file indices (uses first file only)
        **kwargs
            Plotting parameters including plot_type, cmap, rotate, etc.

        Notes
        -----
        Only plots the first file from the selection. For multiple
        file analysis, use other visualization methods or averaging tools.
        """
        xf_list = self.get_xf_list(rows)[0:1]
        if xf_list:
            _get_module("saxs2d").plot(xf_list[0], *args, **kwargs)

    def add_roi(self, hdl, **kwargs):
        logger.debug(f"ViewerKernel: add_roi called with kwargs: {kwargs}")

        xf_list = self.get_xf_list()
        logger.debug(
            f"ViewerKernel: get_xf_list returned {len(xf_list) if xf_list else 0} files"
        )

        if not xf_list:
            logger.warning(
                "Cannot add ROI: No XPCS files loaded or selected. Please add files to the target list first."
            )
            return None

        try:
            cen = (xf_list[0].bcx, xf_list[0].bcy)
            logger.debug(f"ViewerKernel: beam center coordinates: {cen}")
        except (AttributeError, IndexError) as e:
            logger.error(f"Cannot get beam center coordinates: {e}")
            return None

        sl_type = kwargs["sl_type"]
        logger.debug(f"ViewerKernel: creating ROI of type '{sl_type}'")

        if sl_type == "Pie":
            result = hdl.add_roi(cen=cen, radius=100, **kwargs)
            logger.debug(f"ViewerKernel: Pie ROI creation result: {result}")
            return result
        if sl_type == "Q-Wedge":
            # Q-Wedge ROI for Q-space analysis (wedge-shaped)
            logger.debug(
                f"ViewerKernel: creating Q-Wedge ROI at center {cen} with radius 100"
            )
            result = hdl.add_roi(cen=cen, radius=100, **kwargs)
            logger.debug(f"ViewerKernel: Q-Wedge ROI creation result: {result}")
            return result
        if sl_type == "Circle":
            try:
                radius_v = min(xf_list[0].mask.shape[0] - cen[1], cen[1])
                radius_h = min(xf_list[0].mask.shape[1] - cen[0], cen[0])
                radius = min(radius_h, radius_v) * 0.8

                result1 = hdl.add_roi(cen=cen, radius=radius, label="RingA", **kwargs)
                result2 = hdl.add_roi(
                    cen=cen, radius=0.8 * radius, label="RingB", **kwargs
                )
                logger.debug(
                    f"ViewerKernel: Circle ROI creation results: {result1}, {result2}"
                )
                return result1  # Return first ring result
            except (AttributeError, IndexError) as e:
                logger.error(f"Cannot create Circle ROI: mask data not available - {e}")
                return None
        elif sl_type == "Phi-Ring":
            # Phi-Ring ROI for angular analysis (ring-shaped)
            try:
                radius_v = min(xf_list[0].mask.shape[0] - cen[1], cen[1])
                radius_h = min(xf_list[0].mask.shape[1] - cen[0], cen[0])
                radius = min(radius_h, radius_v) * 0.8

                result1 = hdl.add_roi(cen=cen, radius=radius, label="RingA", **kwargs)
                result2 = hdl.add_roi(
                    cen=cen, radius=0.8 * radius, label="RingB", **kwargs
                )
                logger.debug(
                    f"ViewerKernel: Phi-Ring ROI creation results: {result1}, {result2}"
                )
                return result1  # Return first ring result
            except (AttributeError, IndexError) as e:
                logger.error(
                    f"Cannot create Phi-Ring ROI: mask data not available - {e}"
                )
                return None
        else:
            logger.error(f"ViewerKernel: Unknown ROI type '{sl_type}'")
            return None

    def plot_saxs_1d(self, pg_hdl, mp_hdl, **kwargs):
        xf_list = self.get_xf_list()
        if xf_list:
            roi_list = pg_hdl.get_roi_list()
            self.get_module("saxs1d").pg_plot(
                xf_list,
                mp_hdl,
                bkg_file=self.meta["saxs1d_bkg_xf"],
                roi_list=roi_list,
                **kwargs,
            )

    def export_saxs_1d(self, pg_hdl, folder):
        xf_list = self.get_xf_list()
        roi_list = pg_hdl.get_roi_list()
        for xf in xf_list:
            xf.export_saxs1d(roi_list, folder)

    def switch_saxs1d_line(self, mp_hdl, lb_type):
        pass
        # saxs1d.switch_line_builder(mp_hdl, lb_type)

    def plot_twotime(self, hdl, rows=None, **kwargs):
        """
        Generate two-time correlation plots for dynamic analysis.

        Creates two-time correlation maps and associated visualizations
        for analyzing sample dynamics over extended time periods.
        Uses memory-aware caching to handle large correlation datasets.

        Parameters
        ----------
        hdl : dict
            Dictionary of plot handlers for different visualization components
        rows : list, optional
            List of file indices to process (uses current selection if None)
        **kwargs
            Plotting parameters including colormap, cropping, and level settings

        Returns
        -------
        list or None
            List of Q-bin labels for successful processing, None if no data

        Notes
        -----
        - Only processes files with "Twotime" analysis type
        - Implements memory pressure monitoring and cleanup
        - Uses cached datasets to improve performance for repeated access
        - Automatically triggers memory cleanup when pressure exceeds threshold
        """
        xf_list = self.get_xf_list(rows, filter_atype="Twotime")
        if len(xf_list) == 0:
            return None
        xfile = xf_list[0]
        new_qbin_labels = None

        # Use memory-aware caching for current dataset
        current_dset = self._get_cached_dataset(xfile.fname, xfile)
        logger.debug(
            f"ViewerKernel.plot_twotime: current_dset cached={current_dset is not None}, fname={xfile.fname}"
        )

        if current_dset is None or current_dset.fname != xfile.fname:
            logger.debug(f"Dataset changed, loading new qbin labels for {xfile.fname}")
            current_dset = xfile
            self._current_dset_cache[xfile.fname] = current_dset
            new_qbin_labels = xfile.get_twotime_qbin_labels()

            # Check memory pressure and cleanup if needed
            if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
                self._cleanup_memory()
        else:
            logger.debug(
                "Using cached dataset, getting qbin labels for ComboBox population"
            )
            # Always get qbin labels for ComboBox population, even for cached datasets
            new_qbin_labels = current_dset.get_twotime_qbin_labels()

        _get_module("twotime").plot_twotime(current_dset, hdl, **kwargs)
        logger.debug(f"Returning new_qbin_labels: {new_qbin_labels}")
        return new_qbin_labels

    def _get_cached_dataset(self, fname: str, fallback_xfile=None):
        """
        Retrieve cached dataset with memory management.

        Provides memory-efficient access to cached XpcsFile datasets
        using weak references to prevent memory leaks.

        Parameters
        ----------
        fname : str
            Filename identifier for the cached dataset
        fallback_xfile : XpcsFile, optional
            Fallback XpcsFile object if cache miss occurs

        Returns
        -------
        XpcsFile or None
            Cached dataset object or fallback object if available

        Notes
        -----
        Uses WeakValueDictionary for automatic garbage collection
        of unused datasets, helping to manage memory usage for
        large XPCS data collections.
        """
        """Get cached dataset with memory management."""
        return self._current_dset_cache.get(fname, fallback_xfile)

    def plot_intt(self, pg_hdl, rows=None, **kwargs):
        """
        Generate intensity vs. time plots for temporal analysis.

        Creates plots showing intensity variations over time
        for monitoring sample behavior and beam stability.

        Parameters
        ----------
        pg_hdl : PyQtGraph PlotWidget
            PyQtGraph plot widget for real-time visualization
        rows : list, optional
            List of file indices to include in analysis
        **kwargs
            Plotting parameters including sampling, window size, etc.

        Notes
        -----
        Uses the intt module for generating time-series plots
        with configurable sampling rates and analysis windows.
        """
        xf_list = self.get_xf_list(rows=rows)
        _get_module("intt").plot(xf_list, pg_hdl, **kwargs)

    def plot_stability(self, mp_hdl, rows=None, **kwargs):
        """
        Generate sample stability analysis plots.

        Creates plots showing sample stability over time using
        intensity measurements and statistical analysis.

        Parameters
        ----------
        mp_hdl : PlotHandler
            Matplotlib-based plot handler for stability visualization
        rows : list, optional
            List of file indices (uses first file only)
        **kwargs
            Plotting parameters for stability analysis

        Notes
        -----
        Uses the first selected file for stability analysis.
        The stability module provides various metrics for assessing
        sample behavior during data collection.
        """
        xf_obj = self.get_xf_list(rows)[0]
        _get_module("stability").plot(xf_obj, mp_hdl, **kwargs)

    def submit_job(self, *args, **kwargs):
        if len(self.target) <= 0:
            logger.error("no average target is selected")
            return
        worker = _get_module("average_toolbox")(
            self.path, flist=self.target, jid=self.avg_jid
        )
        worker.setup(*args, **kwargs)
        self.avg_worker.append(worker)
        logger.info("create average job, ID = %s", worker.jid)
        self.avg_jid += 1
        self.target.clear()
        return

    def remove_job(self, index):
        self.avg_worker.pop(index)

    def update_avg_info(self, jid):
        self.avg_worker.layoutChanged.emit()
        if 0 <= jid < len(self.avg_worker):
            self.avg_worker[jid].update_plot()

    def update_avg_values(self, data):
        """
        Update averaging worker values with memory management.

        Handles dynamic allocation of averaging result arrays with
        memory pressure monitoring and automatic cleanup when needed.

        Parameters
        ----------
        data : tuple
            (worker_key, value) pair for updating averaging results

        Notes
        -----
        - Dynamically resizes result arrays when capacity is exceeded
        - Monitors memory pressure before allocating additional memory
        - Triggers cleanup operations when memory pressure is high
        - Uses efficient numpy arrays for result storage

        The method implements automatic array doubling strategy
        for efficient memory allocation while monitoring system
        memory pressure to prevent out-of-memory conditions.
        """
        key, val = data[0], data[1]
        if self.avg_worker_active[key] is None:
            self.avg_worker_active[key] = [0, np.zeros(128, dtype=np.float32)]
        record = self.avg_worker_active[key]
        if record[0] == record[1].size:
            # Check memory pressure before allocating more memory
            if MemoryMonitor.is_memory_pressure_high(self._memory_cleanup_threshold):
                logger.warning(
                    "Memory pressure high during averaging, triggering cleanup"
                )
                self._cleanup_memory()

            new_g2 = np.zeros(record[1].size * 2, dtype=np.float32)
            new_g2[0 : record[0]] = record[1]
            record[1] = new_g2
        record[1][record[0]] = val
        record[0] += 1

    def get_memory_stats(self):
        """
        Get comprehensive memory usage statistics for the kernel.

        Returns detailed information about memory usage across different
        components including caches, active workers, and system memory status.

        Returns
        -------
        dict
            Memory statistics including:
            - kernel_cache_items: Number of cached datasets
            - plot_kwargs_items: Number of cached plot parameters
            - avg_worker_active_items: Number of active averaging workers
            - system_memory_used_mb: System memory usage in MB
            - system_memory_available_mb: Available system memory in MB
            - memory_pressure: Current memory pressure ratio (0-1)

        Notes
        -----
        This method is useful for monitoring memory usage during analysis
        and identifying potential memory pressure situations that may
        require cache cleanup or optimization.
        """
        """Get comprehensive memory statistics."""
        used_mb, available_mb = MemoryMonitor.get_memory_usage()
        return {
            "kernel_cache_items": len(self._current_dset_cache),
            "plot_kwargs_items": len(self._plot_kwargs_record),
            "avg_worker_active_items": len(self.avg_worker_active),
            "system_memory_used_mb": used_mb,
            "system_memory_available_mb": available_mb,
            "memory_pressure": MemoryMonitor.get_memory_pressure(),
        }

    def export_g2(self, folder, rows=None):
        """
        Export G2 correlation data and fitting results to text files.

        Parameters
        ----------
        folder : str
            Directory path where files will be saved
        rows : list, optional
            Selected row indices. If None, uses all files.
        """
        import os

        import numpy as np

        logger.info(
            f"G2 export kernel: Starting export to folder {folder} with rows {rows}"
        )
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        logger.info(f"G2 export kernel: Found {len(xf_list)} Multitau files for export")

        if not xf_list:
            logger.error("G2 export kernel: No files with G2 data found")
            raise ValueError("No files with G2 data found")

        successful_exports = 0
        total_files = len(xf_list)

        for xf in xf_list:
            # Export raw G2 data
            logger.info(
                f"G2 export kernel: Processing file {xf.label} (atype: {getattr(xf, 'atype', 'unknown')})"
            )
            try:
                q_val, t_el, g2, g2_err, _labels = xf.get_g2_data()

                # Save G2 correlation data
                g2_filename = os.path.join(folder, f"{xf.label}_g2_data.txt")

                # Create header
                header = "tau(s)"
                for i, q in enumerate(q_val):
                    header += f"\tG2_q{i:03d}(q={q:.6f})\tG2_err_q{i:03d}"

                # Prepare data array
                data_array = [t_el]
                for i in range(len(q_val)):
                    data_array.append(g2[:, i])
                    data_array.append(g2_err[:, i])

                data_matrix = np.column_stack(data_array)
                np.savetxt(g2_filename, data_matrix, header=header, delimiter="\t")
                logger.info(f"G2 export kernel: Saved G2 data file: {g2_filename}")

                # Export fitting results if available
                if xf.fit_summary is not None:
                    fit_filename = os.path.join(folder, f"{xf.label}_g2_fitting.txt")

                    with open(fit_filename, "w") as f:
                        f.write(f"# G2 Fitting Results for {xf.label}\n")
                        f.write(f"# Fitting function: {xf.fit_summary['fit_func']}\n")
                        f.write(f"# Q-range: {xf.fit_summary['q_range']}\n")
                        f.write(f"# Time-range: {xf.fit_summary['t_range']}\n")
                        f.write(f"# Bounds: {xf.fit_summary['bounds']}\n")
                        f.write(f"# Fit flags: {xf.fit_summary['fit_flag']}\n\n")

                        # Parameter names
                        if xf.fit_summary["fit_func"] == "single":
                            param_names = ["a", "b", "c", "d"]
                        else:
                            param_names = ["a", "b", "c", "d", "b2", "c2", "f"]

                        # Write header for fitting parameters
                        f.write("q_value\t")
                        for param in param_names:
                            f.write(f"{param}\t{param}_error\t")
                        f.write("\n")

                        # Write fitting parameters
                        fit_val = xf.fit_summary["fit_val"]
                        for i, q in enumerate(q_val):
                            if i < fit_val.shape[0]:
                                f.write(f"{q:.6f}\t")
                                for j in range(len(param_names)):
                                    if j < fit_val.shape[2]:
                                        f.write(
                                            f"{fit_val[i, 0, j]:.6e}\t{fit_val[i, 1, j]:.6e}\t"
                                        )
                                    else:
                                        f.write("N/A\tN/A\t")
                                f.write("\n")
                    logger.info(
                        f"G2 export kernel: Saved fitting results file: {fit_filename}"
                    )

                    # Export fitted curves if available
                    if "fit_line" in xf.fit_summary and "fit_x" in xf.fit_summary:
                        curve_filename = os.path.join(
                            folder, f"{xf.label}_g2_fitted_curves.txt"
                        )

                        fit_x = xf.fit_summary["fit_x"]
                        fit_line = xf.fit_summary["fit_line"]

                        # Create header for fitted curves
                        header = "tau(s)"
                        for i, q in enumerate(q_val):
                            header += f"\tG2_fit_q{i:03d}(q={q:.6f})"

                        # Prepare fitted curve data
                        curve_data = [fit_x]
                        for i in range(len(q_val)):
                            if i < fit_line.shape[0]:
                                curve_data.append(fit_line[i, :])
                            else:
                                curve_data.append(np.ones_like(fit_x))

                        curve_matrix = np.column_stack(curve_data)
                        np.savetxt(
                            curve_filename, curve_matrix, header=header, delimiter="\t"
                        )
                        logger.info(
                            f"G2 export kernel: Saved fitted curves file: {curve_filename}"
                        )

                successful_exports += 1
                logger.info(f"Successfully exported G2 data for {xf.label}")

            except ValueError as e:
                # Handle specific data validation errors (like invalid Q indices)
                logger.error(f"Data validation error for {xf.label}: {e}")
                continue
            except (OSError, PermissionError) as e:
                # Handle file system errors
                logger.error(f"File system error exporting {xf.label}: {e}")
                raise  # Re-raise file system errors as they affect the entire export
            except Exception as e:
                # Handle unexpected errors with more detail
                logger.error(
                    f"Unexpected error exporting G2 data for {xf.label}: {type(e).__name__}: {e}"
                )
                import traceback

                logger.debug(f"Full traceback for {xf.label}: {traceback.format_exc()}")
                continue

        # Export summary
        if successful_exports == total_files:
            logger.info(
                f"G2 export completed successfully: {successful_exports}/{total_files} files exported"
            )
        elif successful_exports > 0:
            logger.warning(
                f"G2 export partially completed: {successful_exports}/{total_files} files exported successfully"
            )
        else:
            raise ValueError(
                f"G2 export failed: 0/{total_files} files exported successfully"
            )

    def export_diffusion(self, folder, rows=None):
        """
        Export tau-q power law fitting results to text files.

        Parameters
        ----------
        folder : str
            Directory path where files will be saved
        rows : list, optional
            Selected row indices. If None, uses all files.
        """
        import os
        import time

        xf_list = self.get_xf_list(
            rows=rows, filter_atype="Multitau", filter_fitted=True
        )

        if not xf_list:
            raise ValueError("No files with tau-q fitting results found")

        # Filter files that actually have tau-q fitting data
        valid_xf_list = []
        for xf in xf_list:
            if xf.fit_summary and "tauq_fit_val" in xf.fit_summary:
                valid_xf_list.append(xf)

        if not valid_xf_list:
            raise ValueError(
                "No files with tau-q power law fitting results found. Please perform diffusion analysis first."
            )

        successful_exports = 0
        total_files = len(valid_xf_list)

        # Export individual file results
        for xf in valid_xf_list:
            try:
                filename = os.path.join(folder, f"{xf.label}_diffusion_fitting.txt")

                tauq_data = xf.fit_summary[
                    "tauq_fit_val"
                ]  # Shape: (2, 2) - [[a_val, b_val], [a_err, b_err]]

                with open(filename, "w") as f:
                    f.write(f"# Power Law Fitting Results for {xf.label}\n")
                    f.write("# Model: tau = a * q^b\n")
                    f.write(f"# File: {xf.fname}\n")
                    f.write(f"# Q-range: {xf.fit_summary.get('tauq_q_range', 'N/A')}\n")
                    f.write(f"# Bounds: {xf.fit_summary.get('tauq_bounds', 'N/A')}\n")
                    f.write(
                        f"# Fit flags: {xf.fit_summary.get('tauq_fit_flag', 'N/A')}\n\n"
                    )

                    f.write("parameter\tvalue\terror\n")
                    f.write(f"a\t{tauq_data[0, 0]:.6e}\t{tauq_data[1, 0]:.6e}\n")
                    f.write(f"b\t{tauq_data[0, 1]:.6f}\t{tauq_data[1, 1]:.6f}\n")

                successful_exports += 1
                logger.info(
                    f"Successfully exported diffusion fitting results for {xf.label}"
                )

            except (OSError, PermissionError) as e:
                logger.error(
                    f"File system error exporting diffusion data for {xf.label}: {e}"
                )
                raise  # Re-raise file system errors as they affect the entire export
            except Exception as e:
                logger.error(
                    f"Unexpected error exporting diffusion data for {xf.label}: {type(e).__name__}: {e}"
                )
                import traceback

                logger.debug(f"Full traceback for {xf.label}: {traceback.format_exc()}")
                continue

        # Export summary table
        try:
            summary_filename = os.path.join(folder, "diffusion_summary.txt")

            with open(summary_filename, "w") as f:
                f.write("# Power Law Fitting Summary\n")
                f.write("# Model: tau = a * q^b\n")
                f.write(f"# Export date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total files: {total_files}\n\n")

                f.write("file_label\ta_value\ta_error\tb_value\tb_error\tfile_name\n")

                # Debug: Check what data is available for each file
                logger.info(f"Creating summary table for {len(valid_xf_list)} files")
                for xf in valid_xf_list:
                    logger.debug(
                        f"File {xf.label}: fit_summary keys = {list(xf.fit_summary.keys()) if xf.fit_summary else 'None'}"
                    )

                    if xf.fit_summary and "tauq_fit_val" in xf.fit_summary:
                        tauq_data = xf.fit_summary["tauq_fit_val"]
                        f.write(
                            f"{xf.label}\t{tauq_data[0, 0]:.6e}\t{tauq_data[1, 0]:.6e}\t{tauq_data[0, 1]:.6f}\t{tauq_data[1, 1]:.6f}\t{xf.fname}\n"
                        )
                        logger.info(f"Added summary data for {xf.label}")
                    else:
                        logger.warning(
                            f"No tauq_fit_val found for {xf.label} - fit_summary: {xf.fit_summary is not None}"
                        )

            logger.info(f"Summary table exported to {summary_filename}")

        except Exception as e:
            logger.warning(f"Failed to create summary table: {e}")
            # Don't fail the entire export if summary fails

        # Export summary
        if successful_exports == total_files:
            logger.info(
                f"Diffusion export completed successfully: {successful_exports}/{total_files} files exported"
            )
        elif successful_exports > 0:
            logger.warning(
                f"Diffusion export partially completed: {successful_exports}/{total_files} files exported successfully"
            )
        else:
            raise ValueError(
                f"Diffusion export failed: 0/{total_files} files exported successfully"
            )


if __name__ == "__main__":
    flist = os.listdir("./data")
    dv = ViewerKernel("./data", flist)
