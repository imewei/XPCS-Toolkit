"""
Async-enhanced viewer kernel for XPCS Toolkit.

This module extends the ViewerKernel with async capabilities for
non-blocking data loading and processing operations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6 import QtCore
from PySide6.QtCore import QObject, Signal, Slot

from ..utils.logging_config import get_logger
from .async_workers import ComputationWorker, DataLoadWorker, WorkerManager
from .plot_workers import (
    G2PlotWorker,
    IntensityPlotWorker,
    QMapPlotWorker,
    SaxsPlotWorker,
    StabilityPlotWorker,
    TwotimePlotWorker,
)

logger = get_logger(__name__)


class AsyncViewerKernel(QObject):
    """
    Async-enhanced wrapper for ViewerKernel operations.

    Provides non-blocking versions of heavy operations like data loading,
    plotting, and computation while maintaining the original interface.
    """

    # Signals for async operations
    data_loaded = Signal(str, object)  # operation_id, result
    plot_ready = Signal(str, object)  # operation_id, result
    operation_progress = Signal(str, int, int, str)  # op_id, current, total, msg
    operation_error = Signal(str, str, str)  # op_id, error_msg, traceback
    operation_cancelled = Signal(str)  # operation_id

    def __init__(self, viewer_kernel, thread_pool: QtCore.QThreadPool):
        super().__init__()
        self.viewer_kernel = viewer_kernel
        self.worker_manager = WorkerManager(thread_pool)
        self.active_operations = {}  # operation_id -> worker_id

        # Connect worker manager signals
        self.worker_manager.worker_progress.connect(
            lambda worker_id, current, total, msg: self._emit_operation_progress(
                worker_id, current, total, msg
            )
        )

    def load_files_async(
        self,
        file_paths: list[str],
        load_func: Callable | None = None,
        operation_id: str | None = None,
    ) -> str:
        """
        Load multiple files asynchronously with progress reporting.

        Args:
            file_paths: List of file paths to load
            load_func: Optional custom loading function
            operation_id: Optional operation identifier

        Returns:
            str: Operation ID for tracking
        """
        if load_func is None:
            from ..xpcs_file import XpcsFile

            load_func = XpcsFile

        operation_id = operation_id or f"load_files_{id(file_paths)}"

        worker = DataLoadWorker(
            load_func=load_func,
            file_paths=file_paths,
            worker_id=f"data_loader_{operation_id}",
        )

        # Connect signals
        worker.signals.finished.connect(
            lambda result: self._on_data_loaded(operation_id, result)
        )
        worker.signals.error.connect(
            lambda msg, tb: self._on_operation_error(operation_id, msg, tb)
        )
        worker.signals.cancelled.connect(
            lambda: self._on_operation_cancelled(operation_id)
        )

        # Submit to worker manager
        worker_id = self.worker_manager.submit_worker(worker)
        self.active_operations[operation_id] = worker_id

        logger.info(f"Started async file loading: {operation_id}")
        return operation_id

    def plot_saxs_2d_async(
        self, plot_handler, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot SAXS 2D data asynchronously."""
        operation_id = operation_id or "plot_saxs_2d"

        worker = SaxsPlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=kwargs,
            worker_id=f"saxs_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def plot_g2_async(
        self, plot_handler, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot G2 correlation functions asynchronously."""
        operation_id = operation_id or "plot_g2"

        worker = G2PlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=kwargs,
            worker_id=f"g2_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def plot_twotime_async(
        self, plot_handlers, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot two-time correlation functions asynchronously."""
        operation_id = operation_id or "plot_twotime"

        worker = TwotimePlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handlers=plot_handlers,
            plot_kwargs=kwargs,
            worker_id=f"twotime_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def plot_intensity_async(
        self, plot_handler, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot intensity vs time asynchronously."""
        operation_id = operation_id or "plot_intensity"

        worker = IntensityPlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=kwargs,
            worker_id=f"intensity_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def plot_stability_async(
        self, plot_handler, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot stability data asynchronously."""
        operation_id = operation_id or "plot_stability"

        worker = StabilityPlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=kwargs,
            worker_id=f"stability_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def plot_qmap_async(
        self, plot_handler, operation_id: str | None = None, **kwargs
    ) -> str:
        """Plot Q-map data asynchronously."""
        operation_id = operation_id or "plot_qmap"

        worker = QMapPlotWorker(
            viewer_kernel=self.viewer_kernel,
            plot_handler=plot_handler,
            plot_kwargs=kwargs,
            worker_id=f"qmap_plot_{operation_id}",
        )

        return self._submit_plot_worker(worker, operation_id)

    def compute_async(
        self,
        compute_func: Callable,
        data: Any,
        operation_id: str | None = None,
        progress_callback: Callable | None = None,
        **kwargs,
    ) -> str:
        """
        Execute heavy computation asynchronously.

        Args:
            compute_func: Function to execute
            data: Data to process
            operation_id: Optional operation identifier
            progress_callback: Optional progress callback function
            **kwargs: Additional arguments for compute_func

        Returns:
            str: Operation ID for tracking
        """
        operation_id = operation_id or f"compute_{id(compute_func)}"

        worker = ComputationWorker(
            compute_func=compute_func,
            data=data,
            progress_callback=progress_callback,
            compute_kwargs=kwargs,
            worker_id=f"compute_{operation_id}",
        )

        # Connect signals
        worker.signals.finished.connect(
            lambda result: self._on_computation_finished(operation_id, result)
        )
        worker.signals.error.connect(
            lambda msg, tb: self._on_operation_error(operation_id, msg, tb)
        )
        worker.signals.cancelled.connect(
            lambda: self._on_operation_cancelled(operation_id)
        )

        # Submit to worker manager
        worker_id = self.worker_manager.submit_worker(worker)
        self.active_operations[operation_id] = worker_id

        logger.info(f"Started async computation: {operation_id}")
        return operation_id

    def cancel_operation(self, operation_id: str):
        """Cancel an active operation by ID."""
        if operation_id in self.active_operations:
            worker_id = self.active_operations[operation_id]
            self.worker_manager.cancel_worker(worker_id)
            logger.info(f"Cancelled operation: {operation_id}")

    def cancel_all_operations(self):
        """Cancel all active operations."""
        self.worker_manager.cancel_all_workers()
        self.active_operations.clear()
        logger.info("Cancelled all active operations")

    def is_operation_active(self, operation_id: str) -> bool:
        """Check if an operation is still active."""
        if operation_id not in self.active_operations:
            return False
        worker_id = self.active_operations[operation_id]
        return self.worker_manager.is_worker_active(worker_id)

    def get_operation_result(self, operation_id: str) -> Any | None:
        """Get the result of a completed operation."""
        if operation_id in self.active_operations:
            worker_id = self.active_operations[operation_id]
            return self.worker_manager.get_worker_result(worker_id)
        return None

    def _submit_plot_worker(self, worker, operation_id: str) -> str:
        """Submit a plot worker and set up signal connections."""
        # Connect signals
        worker.signals.finished.connect(
            lambda result: self._on_plot_ready(operation_id, result)
        )
        worker.signals.error.connect(
            lambda msg, tb: self._on_operation_error(operation_id, msg, tb)
        )
        worker.signals.cancelled.connect(
            lambda: self._on_operation_cancelled(operation_id)
        )

        # Submit to worker manager
        worker_id = self.worker_manager.submit_worker(worker)
        self.active_operations[operation_id] = worker_id

        logger.info(f"Started async plot operation: {operation_id}")
        return operation_id

    @Slot(str, object)
    def _on_data_loaded(self, operation_id: str, result: Any):
        """Handle data loading completion."""
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        self.data_loaded.emit(operation_id, result)
        logger.debug(f"Data loading completed: {operation_id}")

    @Slot(str, object)
    def _on_plot_ready(self, operation_id: str, result: Any):
        """Handle plot operation completion."""
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        self.plot_ready.emit(operation_id, result)
        logger.debug(f"Plot operation completed: {operation_id}")

    @Slot(str, object)
    def _on_computation_finished(self, operation_id: str, result: Any):
        """Handle computation completion."""
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        # For now, emit as data_loaded signal
        self.data_loaded.emit(operation_id, result)
        logger.debug(f"Computation completed: {operation_id}")

    @Slot(str, str, str)
    def _on_operation_error(
        self, operation_id: str, error_msg: str, traceback_str: str
    ):
        """Handle operation error."""
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        self.operation_error.emit(operation_id, error_msg, traceback_str)
        logger.error(f"Operation failed: {operation_id} - {error_msg}")

    @Slot(str)
    def _on_operation_cancelled(self, operation_id: str):
        """Handle operation cancellation."""
        if operation_id in self.active_operations:
            del self.active_operations[operation_id]
        self.operation_cancelled.emit(operation_id)
        logger.info(f"Operation cancelled: {operation_id}")

    def _emit_operation_progress(
        self, worker_id: str, current: int, total: int, message: str
    ):
        """Emit progress signal for operations."""
        # Find operation_id for this worker_id
        operation_id = None
        for op_id, w_id in self.active_operations.items():
            if w_id == worker_id:
                operation_id = op_id
                break

        if operation_id:
            self.operation_progress.emit(operation_id, current, total, message)


class AsyncDataPreloader(QObject):
    """
    Preloader for data files to improve user experience.

    Intelligently preloads data based on user selection patterns
    and file proximity to reduce wait times.
    """

    preload_completed = Signal(str, object)  # file_path, data
    preload_progress = Signal(int, int)  # current, total

    def __init__(self, async_kernel: AsyncViewerKernel, max_cache_size: int = 10):
        super().__init__()
        self.async_kernel = async_kernel
        self.max_cache_size = max_cache_size
        self.cache = {}  # file_path -> data
        self.cache_order = []  # LRU tracking
        self.preload_queue = []
        self.is_preloading = False

    def suggest_preload(
        self, current_files: list[str], all_files: list[str]
    ) -> list[str]:
        """
        Suggest files to preload based on current selection.

        Args:
            current_files: Currently selected files
            all_files: All available files

        Returns:
            List of files to preload
        """
        if not current_files or not all_files:
            return []

        # Get indices of current files
        current_indices = []
        for file in current_files:
            try:
                idx = all_files.index(file)
                current_indices.append(idx)
            except ValueError:
                continue

        if not current_indices:
            return []

        # Suggest adjacent files
        suggest_indices = set()
        for idx in current_indices:
            # Add previous and next files
            if idx > 0:
                suggest_indices.add(idx - 1)
            if idx < len(all_files) - 1:
                suggest_indices.add(idx + 1)

        # Convert back to file paths, excluding already cached files
        suggested_files = []
        for idx in suggest_indices:
            file_path = all_files[idx]
            if file_path not in self.cache and file_path not in current_files:
                suggested_files.append(file_path)

        return suggested_files[:5]  # Limit to 5 files

    def start_preload(self, file_paths: list[str]):
        """Start preloading the specified files."""
        if self.is_preloading:
            return

        # Filter out already cached files
        files_to_load = [f for f in file_paths if f not in self.cache]

        if not files_to_load:
            return

        self.preload_queue = files_to_load
        self.is_preloading = True

        # Start async loading
        self.async_kernel.load_files_async(files_to_load, operation_id="preload_data")

        # Connect to completion
        self.async_kernel.data_loaded.connect(self._on_preload_completed)

        logger.info(f"Started preloading {len(files_to_load)} files")

    def get_cached_data(self, file_path: str) -> Any | None:
        """Get cached data for a file if available."""
        if file_path in self.cache:
            # Move to end of LRU list
            self.cache_order.remove(file_path)
            self.cache_order.append(file_path)
            return self.cache[file_path]
        return None

    def _on_preload_completed(self, operation_id: str, result: list[Any]):
        """Handle preload completion."""
        if operation_id != "preload_data":
            return

        # Cache the loaded data
        for i, data in enumerate(result):
            if i < len(self.preload_queue) and data is not None:
                file_path = self.preload_queue[i]
                self._add_to_cache(file_path, data)

        self.is_preloading = False
        self.preload_queue = []

        # Disconnect signal
        self.async_kernel.data_loaded.disconnect(self._on_preload_completed)

        logger.info("Preloading completed")

    def _add_to_cache(self, file_path: str, data: Any):
        """Add data to cache with LRU eviction."""
        if file_path in self.cache:
            # Update existing entry
            self.cache_order.remove(file_path)
        elif len(self.cache) >= self.max_cache_size:
            # Evict oldest entry
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]

        self.cache[file_path] = data
        self.cache_order.append(file_path)

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.cache_order.clear()
        logger.info("Preload cache cleared")
