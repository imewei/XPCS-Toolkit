# Standard library imports
import os
import weakref

# Third-party imports
import numpy as np
import pyqtgraph as pg

# Local imports
from .file_locator import FileLocator
from .helper.listmodel import TableDataModel
from .module import g2mod, intt, saxs1d, saxs2d, stability, tauq, twotime
from .module.average_toolbox import AverageToolbox
from .utils.logging_config import get_logger
from .xpcs_file import MemoryMonitor, XpcsFile

logger = get_logger(__name__)


class ViewerKernel(FileLocator):
    """
    Backend kernel for XPCS data processing and analysis coordination.

    ViewerKernel serves as the central coordinator between the GUI and various
    analysis modules. It manages file collections, handles data caching,
    coordinates background processing, and provides a unified interface
    for all analysis operations.

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

    def __init__(self, path, statusbar=None):
        super().__init__(path)
        self.statusbar = statusbar
        self.meta = None
        self.reset_meta()
        self.path = path
        self.avg_tb = AverageToolbox(path)
        self.avg_worker = TableDataModel()
        self.avg_jid = 0
        self.avg_worker_active = {}

        # Memory-aware caching for current_dset with weak references
        self._current_dset_cache = weakref.WeakValueDictionary()
        self._plot_kwargs_record = {}
        self._memory_cleanup_threshold = 0.8  # Trigger cleanup at 80% memory usage

    def reset_meta(self):
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
        return

    def reset_kernel(self):
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

    def _cleanup_memory(self):
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

        used_mb, available_mb = MemoryMonitor.get_memory_usage()
        logger.debug(f"Memory cleanup completed, current usage: {used_mb:.1f}MB")

    def select_bkgfile(self, fname):
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

    def get_pg_tree(self, rows):
        xf_list = self.get_xf_list(rows)
        if xf_list:
            return xf_list[0].get_pg_tree()
        else:
            return None

    def get_fitting_tree(self, rows):
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
            g2mod.pg_plot(handler, xf_list, q_range, t_range, y_range, **kwargs)
            q, tel, *unused = g2mod.get_data(xf_list)
            return q, tel
        else:
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
        xf_list = self.get_xf_list(rows=rows, filter_atype="Multitau")
        short_list = [xf for xf in xf_list if xf.fit_summary is not None]

        if len(short_list) == 0:
            logger.warning(
                f"No files with G2 fitting results found for diffusion analysis. "
                f"Found {len(xf_list)} Multitau files but none have been fitted."
            )
            # Clear the plot and show a message
            if hdl is not None:
                hdl.clear()
                ax = hdl.subplots(1, 1)
                ax.text(
                    0.5,
                    0.5,
                    'No G2 fitting results available.\n\nPlease:\n1. Go to "g2" tab\n2. Select files\n3. Click "update plot" to perform G2 fitting\n4. Return to "Diffusion" tab',
                    ha="center",
                    va="center",
                    fontsize=12,
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7
                    ),
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis("off")
                hdl.tight_layout()
        else:
            logger.info(
                f"Found {len(short_list)} files with G2 fitting results for diffusion analysis"
            )
            tauq.plot_pre(short_list, hdl)

    def plot_tauq(
        self,
        hdl=None,
        bounds=None,
        rows=[],
        plot_type=3,
        fit_flag=None,
        offset=None,
        q_range=None,
    ):
        xf_list = self.get_xf_list(
            rows=rows, filter_atype="Multitau", filter_fitted=True
        )
        result = {}
        for x in xf_list:
            if x.fit_summary is None:
                logger.info("g2 fitting is not available for %s", x.fname)
            else:
                x.fit_tauq(q_range, bounds, fit_flag)
                result[x.label] = x.get_fitting_info(mode="tauq_fitting")

        if len(result) > 0:
            tauq.plot(
                xf_list, hdl=hdl, q_range=q_range, offset=offset, plot_type=plot_type
            )

        return result

    def get_info_at_mouse(self, rows, x, y):
        xf = self.get_xf_list(rows)
        if xf:
            info = xf[0].get_info_at_position(x, y)
            return info

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
            saxs2d.plot(xf_list[0], *args, **kwargs)

    def add_roi(self, hdl, **kwargs):
        xf_list = self.get_xf_list()
        cen = (xf_list[0].bcx, xf_list[0].bcy)
        if kwargs["sl_type"] == "Pie":
            hdl.add_roi(cen=cen, radius=100, **kwargs)
        elif kwargs["sl_type"] == "Circle":
            radius_v = min(xf_list[0].mask.shape[0] - cen[1], cen[1])
            radius_h = min(xf_list[0].mask.shape[1] - cen[0], cen[0])
            radius = min(radius_h, radius_v) * 0.8

            hdl.add_roi(cen=cen, radius=radius, label="RingA", **kwargs)
            hdl.add_roi(cen=cen, radius=0.8 * radius, label="RingB", **kwargs)

    def plot_saxs_1d(self, pg_hdl, mp_hdl, **kwargs):
        xf_list = self.get_xf_list()
        if xf_list:
            roi_list = pg_hdl.get_roi_list()
            saxs1d.pg_plot(
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
        return

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

        twotime.plot_twotime(current_dset, hdl, **kwargs)
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
        intt.plot(xf_list, pg_hdl, **kwargs)

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
        stability.plot(xf_obj, mp_hdl, **kwargs)

    def submit_job(self, *args, **kwargs):
        if len(self.target) <= 0:
            logger.error("no average target is selected")
            return
        worker = AverageToolbox(self.path, flist=self.target, jid=self.avg_jid)
        worker.setup(*args, **kwargs)
        self.avg_worker.append(worker)
        logger.info("create average job, ID = %s", worker.jid)
        self.avg_jid += 1
        self.target.clear()
        return

    def remove_job(self, index):
        self.avg_worker.pop(index)
        return

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

        return

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

    def export_g2(self):
        pass


if __name__ == "__main__":
    flist = os.listdir("./data")
    dv = ViewerKernel("./data", flist)
