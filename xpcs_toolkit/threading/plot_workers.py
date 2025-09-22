"""
Specific worker implementations for plotting operations in XPCS Toolkit.

This module contains specialized workers for different types of plots
that can run in background threads to maintain GUI responsiveness.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.logging_config import get_logger
from .async_workers import BaseAsyncWorker
from .base_plot_worker import BasePlotWorker

logger = get_logger(__name__)


class SaxsPlotWorker(BasePlotWorker):
    """
    Worker for SAXS 2D plotting operations.
    Handles data loading, processing, and plot generation.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(
            viewer_kernel=viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=plot_kwargs,
            worker_id=worker_id or "saxs_plot_worker",
            worker_type="saxs_2d",
        )

    def do_work(self) -> dict[str, Any]:
        """Execute SAXS 2D plotting with progress reporting."""
        self.emit_status("Loading SAXS data...")
        self.emit_step_progress(1, 4, "Getting file list")

        # Get the file list using base class method
        rows = self.get_plot_parameter("rows", [])
        xf_list = self.get_file_list(rows)[0:1]

        if not self.validate_file_list(xf_list, min_files=1):
            return self.create_error_result(
                "No files selected. Please add files to the target list first."
            )

        self.check_cancelled()
        self.emit_step_progress(2, 4, "Processing SAXS data")

        # Process the SAXS data using base class utilities
        xf = xf_list[0]
        plot_type = self.get_plot_parameter("plot_type", "log")
        rotate = self.get_plot_parameter("rotate", False)

        # Get the image data
        if hasattr(xf, "saxs_2d"):
            image_data = xf.saxs_2d.copy()
        else:
            return {
                "message": "No SAXS 2D data available in this file. Please check the file format.",
                "type": "info",
            }

        # Apply rotation if requested
        if rotate:
            image_data = np.rot90(image_data)

        self.check_cancelled()
        self.emit_progress(3, 4, "Applying scaling and normalization")

        # Apply log scaling if requested
        if plot_type == "log":
            image_data = np.log10(image_data + 1)

        # Calculate levels for auto-scaling
        autolevel = self.plot_kwargs.get("autolevel", True)
        if autolevel:
            vmin, vmax = np.percentile(image_data, (2, 98))
        else:
            vmin = self.plot_kwargs.get("vmin", image_data.min())
            vmax = self.plot_kwargs.get("vmax", image_data.max())

        self.check_cancelled()
        self.emit_progress(4, 4, "Finalizing plot data")

        # Prepare result
        result = {
            "image_data": image_data,
            "levels": (vmin, vmax),
            "cmap": self.plot_kwargs.get("cmap", "viridis"),
            "file_info": {
                "filename": xf.fname if hasattr(xf, "fname") else "unknown",
                "shape": image_data.shape,
            },
        }

        return result


class G2PlotWorker(BaseAsyncWorker):
    """
    Worker for G2 correlation function plotting.
    Handles data extraction, fitting, and multi-panel plot generation.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id or "g2_plot_worker")
        self.viewer_kernel = viewer_kernel
        self.plot_handler = plot_handler
        self.plot_kwargs = plot_kwargs

    def do_work(self) -> dict[str, Any]:
        """Execute G2 plotting with fitting and progress reporting."""
        self.emit_status("Preparing G2 data...")

        # Get parameters
        rows = self.plot_kwargs.get("rows", [])
        q_range = self.plot_kwargs.get("q_range")
        t_range = self.plot_kwargs.get("t_range")
        self.plot_kwargs.get("y_range")
        show_fit = self.plot_kwargs.get("show_fit", False)

        self.emit_progress(1, 6, "Loading G2 files")

        # Get file list - let g2mod.get_data handle the multitau filtering
        xf_list = self.viewer_kernel.get_xf_list(rows=rows)
        if not xf_list:
            return {
                "message": "No files selected. Please add files to the target list first.",
                "type": "info",
            }

        logger.info(f"G2PlotWorker: Processing {len(xf_list)} files")
        for i, xf in enumerate(xf_list):
            logger.info(f"G2PlotWorker: File {i}: {getattr(xf, 'fname', 'unknown')}")
            logger.info(f"  Analysis types: {getattr(xf, 'atype', 'No atype')}")

        self.check_cancelled()
        self.emit_progress(2, 6, "Extracting G2 data")

        # Import G2 module and extract data
        from ..module import g2mod

        q, tel, g2, g2_err, labels = g2mod.get_data(
            xf_list, q_range=q_range, t_range=t_range
        )

        logger.info(f"G2PlotWorker: g2mod.get_data returned q={q}, tel={tel}")

        if not q or q[0] is False:
            # Check what analysis types are available for better error messaging
            available_types = [getattr(xf, "atype", "unknown") for xf in xf_list]
            return {
                "message": f"No G2 correlation data found. Available analysis types: {available_types}. Files must contain Multitau or Twotime correlation analysis.",
                "type": "info",
            }

        self.check_cancelled()
        self.emit_progress(3, 6, "Computing plot geometry")

        # Compute geometry for plots
        plot_type = self.plot_kwargs.get("plot_type", "multiple")
        num_figs, num_lines = g2mod.compute_geometry(g2, plot_type)

        self.check_cancelled()

        # Perform fitting if requested
        fit_results = None
        if show_fit:
            self.emit_progress(4, 6, "Performing G2 fitting")
            fit_results = self._perform_g2_fitting(xf_list, q_range, t_range)
        else:
            self.emit_progress(4, 6, "Skipping fitting")

        self.check_cancelled()
        self.emit_progress(5, 6, "Preparing plot data structures")

        # Prepare plotting data
        plot_data = {
            "q": q,
            "tel": tel,
            "g2": g2,
            "g2_err": g2_err,
            "labels": labels,
            "num_figs": num_figs,
            "num_lines": num_lines,
            "fit_results": fit_results,
            "plot_params": self.plot_kwargs,
        }

        self.emit_progress(6, 6, "G2 data preparation complete")
        return plot_data

    def _perform_g2_fitting(self, xf_list, q_range, t_range) -> dict | None:
        """Perform G2 fitting with progress updates."""
        bounds = self.plot_kwargs.get("bounds")
        fit_flag = self.plot_kwargs.get("fit_flag")
        fit_func = self.plot_kwargs.get("fit_func", "single")

        if not bounds or not fit_flag or not any(fit_flag):
            return None

        fit_results = {}
        total_files = len(xf_list)

        for i, xf in enumerate(xf_list):
            self.check_cancelled()
            self.emit_status(f"Fitting file {i + 1}/{total_files}: {xf.label}")

            try:
                fit_summary = xf.fit_g2(q_range, t_range, bounds, fit_flag, fit_func)
                fit_results[xf.label] = fit_summary
            except Exception as e:
                logger.warning(f"G2 fitting failed for {xf.label}: {e}")
                fit_results[xf.label] = None

        return fit_results


class TwotimePlotWorker(BaseAsyncWorker):
    """
    Worker for two-time correlation function plotting.
    Handles C2 matrix loading, processing, and visualization.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handlers,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id or "twotime_plot_worker")
        self.viewer_kernel = viewer_kernel
        self.plot_handlers = plot_handlers
        self.plot_kwargs = plot_kwargs

    def do_work(self) -> dict[str, Any]:
        """Execute two-time plotting with progress reporting."""
        self.emit_status("Loading two-time data...")

        rows = self.plot_kwargs.get("rows", [])

        self.emit_progress(1, 5, "Getting two-time files")

        # Get file list - check for Twotime data after loading
        xf_list = self.viewer_kernel.get_xf_list(rows)
        if not xf_list:
            return {
                "message": "No files selected. Please add files to the target list first.",
                "type": "info",
            }

        # Check if any files have Twotime analysis data
        twotime_files = [xf for xf in xf_list if "Twotime" in getattr(xf, "atype", [])]
        if not twotime_files:
            return {
                "message": "No Twotime analysis data found. The selected files may not contain two-time correlation analysis results.",
                "type": "info",
            }

        xf_list = twotime_files

        xfile = xf_list[0]

        self.check_cancelled()
        self.emit_progress(2, 5, "Loading C2 matrices")

        # Check if we need to load new data
        new_qbin_labels = None
        try:
            current_dset = self.viewer_kernel._get_cached_dataset(xfile.fname, xfile)
            if current_dset is None or current_dset.fname != xfile.fname:
                # Cache the new dataset
                self.viewer_kernel._current_dset_cache[xfile.fname] = xfile
                new_qbin_labels = xfile.get_twotime_qbin_labels()
        except Exception as e:
            logger.warning(f"Dataset caching issue: {e}. Proceeding with current file.")
            new_qbin_labels = xfile.get_twotime_qbin_labels()

        self.check_cancelled()
        self.emit_progress(3, 5, "Processing two-time correlation data")

        # Import twotime module

        # Get selection and processing parameters
        selection = self.plot_kwargs.get("selection", 0)
        self.plot_kwargs.get("auto_crop", True)
        correct_diag = self.plot_kwargs.get("correct_diag", True)

        self.check_cancelled()
        self.emit_progress(4, 5, "Generating plot data")

        # Process the two-time data
        try:
            # Get C2 data for the selected q-bin
            c2_result = xfile.get_twotime_c2(
                selection=selection, correct_diag=correct_diag
            )

            if c2_result is None:
                return {
                    "message": "Failed to load two-time correlation data from this file.",
                    "type": "info",
                }

        except Exception as e:
            logger.error(f"Error processing two-time data: {e}")
            return {
                "message": f"Error processing two-time data: {e!s}",
                "type": "info",
            }

        self.check_cancelled()
        self.emit_progress(5, 5, "Finalizing two-time plot")

        # Prepare result
        result = {
            "c2_result": c2_result,
            "new_qbin_labels": new_qbin_labels,
            "xfile": xfile,
            "plot_params": self.plot_kwargs,
        }

        return result


class IntensityPlotWorker(BaseAsyncWorker):
    """
    Worker for intensity vs time plotting.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id or "intensity_plot_worker")
        self.viewer_kernel = viewer_kernel
        self.plot_handler = plot_handler
        self.plot_kwargs = plot_kwargs

    def do_work(self) -> dict[str, Any]:
        """Execute intensity vs time plotting."""
        self.emit_status("Loading intensity data...")

        rows = self.plot_kwargs.get("rows", [])
        sampling = self.plot_kwargs.get("sampling", 1)
        window = self.plot_kwargs.get("window", 1)

        self.emit_progress(1, 3, "Getting file list")

        xf_list = self.viewer_kernel.get_xf_list(rows=rows)
        if not xf_list:
            return {
                "message": "No files selected. Please add files to the target list first.",
                "type": "info",
            }

        self.check_cancelled()
        self.emit_progress(2, 3, "Processing intensity data")

        # Import intensity module

        # Process intensity data with sampling and windowing
        plot_data = []
        logger.info(f"IntensityPlotWorker: Processing {len(xf_list)} files")

        for i, xf in enumerate(xf_list):
            self.check_cancelled()
            logger.info(
                f"IntensityPlotWorker: Checking file {i}: {getattr(xf, 'fname', 'unknown')}"
            )
            logger.info(f"  Has Int_t: {hasattr(xf, 'Int_t')}")

            if hasattr(xf, "Int_t"):
                try:
                    logger.info(f"  Int_t shape: {xf.Int_t.shape}")
                    # Int_t[1] contains the intensity vs time data
                    int_data = xf.Int_t[1]
                    logger.info(f"  Int_t[1] shape: {int_data.shape}")
                except Exception as e:
                    logger.error(f"  Error accessing Int_t[1]: {e}")
                    continue

                # Apply sampling
                if sampling > 1:
                    int_data = int_data[::sampling]

                # Apply windowing if specified
                if window > 1:
                    # Simple moving average
                    kernel = np.ones(window) / window
                    int_data = np.convolve(int_data, kernel, mode="valid")

                plot_data.append(
                    {
                        "data": int_data,
                        "label": xf.label if hasattr(xf, "label") else f"File {i}",
                        "file_index": i,
                    }
                )

        # Check if we have any data to plot
        if not plot_data:
            return {
                "message": "No intensity vs time data available in the selected files.",
                "type": "info",
            }

        self.check_cancelled()
        self.emit_progress(3, 3, "Intensity processing complete")

        result = {"plot_data": plot_data, "plot_params": self.plot_kwargs}

        return result


class StabilityPlotWorker(BaseAsyncWorker):
    """
    Worker for stability plotting operations.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id or "stability_plot_worker")
        self.viewer_kernel = viewer_kernel
        self.plot_handler = plot_handler
        self.plot_kwargs = plot_kwargs

    def do_work(self) -> dict[str, Any]:
        """Execute stability plotting."""
        self.emit_status("Processing stability data...")

        rows = self.plot_kwargs.get("rows", [])

        self.emit_progress(1, 3, "Loading stability files")

        xf_list = self.viewer_kernel.get_xf_list(rows)
        if not xf_list:
            return {
                "message": "No files selected. Please add files to the target list first.",
                "type": "info",
            }

        # Use first file for stability analysis
        xf_obj = xf_list[0]

        self.check_cancelled()
        self.emit_progress(2, 3, "Computing stability metrics")

        # Import stability module

        # Extract stability data
        self.plot_kwargs.get("plot_type", 2)
        self.plot_kwargs.get("plot_norm", 0)

        # Process the stability data
        logger.info(
            f"StabilityPlotWorker: Processing file {getattr(xf_obj, 'fname', 'unknown')}"
        )
        logger.info(f"  Has stability_data: {hasattr(xf_obj, 'stability_data')}")
        logger.info(f"  Has Int_t: {hasattr(xf_obj, 'Int_t')}")

        if hasattr(xf_obj, "stability_data"):
            stability_data = xf_obj.stability_data
            logger.info("  Using existing stability_data")
        # Compute stability from intensity data if available
        elif hasattr(xf_obj, "Int_t"):
            try:
                logger.info(f"  Int_t shape: {xf_obj.Int_t.shape}")
                # Int_t[1] contains the intensity vs time data
                intensity_data = xf_obj.Int_t[1]
                logger.info(f"  Int_t[1] shape: {intensity_data.shape}")
                stability_data = self._compute_stability_metrics(intensity_data)
                logger.info("  Computed stability metrics")
            except Exception as e:
                logger.error(f"  Error computing stability metrics: {e}")
                return {
                    "message": f"Error processing intensity data: {e!s}",
                    "type": "info",
                }
        else:
            return {
                "message": "No intensity vs time data available. Please load a file with intensity data.",
                "type": "info",
            }

        self.check_cancelled()
        self.emit_progress(3, 3, "Stability analysis complete")

        result = {
            "stability_data": stability_data,
            "xf_obj": xf_obj,
            "plot_params": self.plot_kwargs,
        }

        return result

    def _compute_stability_metrics(self, intensity_data) -> dict[str, np.ndarray]:
        """Compute basic stability metrics from intensity data."""
        # Simple stability metrics
        mean_intensity = np.mean(intensity_data)
        std_intensity = np.std(intensity_data)
        relative_std = std_intensity / mean_intensity if mean_intensity > 0 else 0

        # Time-dependent metrics
        time_points = np.arange(len(intensity_data))

        return {
            "time": time_points,
            "intensity": intensity_data,
            "mean": mean_intensity,
            "std": std_intensity,
            "relative_std": relative_std,
            "drift": np.gradient(intensity_data),  # Simple drift calculation
        }


class QMapPlotWorker(BaseAsyncWorker):
    """
    Worker for Q-map plotting operations.
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id or "qmap_plot_worker")
        self.viewer_kernel = viewer_kernel
        self.plot_handler = plot_handler
        self.plot_kwargs = plot_kwargs

    def do_work(self) -> dict[str, Any]:
        """Execute Q-map plotting."""
        self.emit_status("Loading Q-map data...")

        rows = self.plot_kwargs.get("rows", [])
        target = self.plot_kwargs.get("target", "scattering")

        self.emit_progress(1, 3, "Getting Q-map files")

        xf_list = self.viewer_kernel.get_xf_list(rows=rows)
        if not xf_list:
            return {
                "message": "No files selected. Please add files to the target list first.",
                "type": "info",
            }

        xf = xf_list[0]

        self.check_cancelled()
        self.emit_progress(2, 3, "Processing Q-map data")

        # Get the appropriate data based on target
        if target == "scattering":
            if hasattr(xf, "saxs_2d"):
                value = np.log10(xf.saxs_2d + 1)
                vmin, vmax = np.percentile(value, (2, 98))
            else:
                return {
                    "message": "No scattering data available for Q-map visualization.",
                    "type": "info",
                }
        elif target == "dynamic_roi_map":
            if hasattr(xf, "dqmap"):
                value = xf.dqmap
                vmin, vmax = value.min(), value.max()
            else:
                return {
                    "message": "No dynamic Q-map data available in this file.",
                    "type": "info",
                }
        elif target == "static_roi_map":
            if hasattr(xf, "sqmap"):
                value = xf.sqmap
                vmin, vmax = value.min(), value.max()
            else:
                return {
                    "message": "No static Q-map data available in this file.",
                    "type": "info",
                }
        else:
            raise ValueError(f"Unknown Q-map target: {target}")

        self.check_cancelled()
        self.emit_progress(3, 3, "Q-map processing complete")

        result = {
            "image_data": value,
            "levels": (vmin, vmax),
            "target": target,
            "file_info": {
                "filename": xf.fname if hasattr(xf, "fname") else "unknown",
                "shape": value.shape,
            },
        }

        return result
