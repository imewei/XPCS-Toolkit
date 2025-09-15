"""
Main thread optimization utilities for XPCS Toolkit.

This module provides tools to keep the main GUI thread responsive by:
- Moving parameter validation to background threads
- Implementing non-blocking file operations
- Adding progressive loading with user feedback
- Optimizing plot generation and rendering
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from PySide6 import QtWidgets
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ..utils.logging_config import get_logger
from .async_workers_enhanced import (
    DataLoadWorker,
    PlotWorker,
    WorkerManager,
    WorkerPriority,
    WorkerResult,
)
from .progress_manager import ProgressManager

logger = get_logger(__name__)


@dataclass
class OperationContext:
    """Context for a main thread operation being optimized."""

    operation_id: str
    operation_type: str
    start_time: float = field(default_factory=time.perf_counter)
    params: dict[str, Any] = field(default_factory=dict)
    callbacks: dict[str, Callable] = field(default_factory=dict)
    is_critical: bool = False
    user_initiated: bool = True


class MainThreadOptimizer(QObject):
    """
    Main thread optimizer that ensures GUI responsiveness by managing
    background operations and progressive loading.
    """

    # Signals for main thread events
    operation_started = Signal(str, str)  # operation_id, operation_type
    operation_completed = Signal(str, object)  # operation_id, result
    operation_failed = Signal(str, str)  # operation_id, error_message
    ui_update_ready = Signal(str, object)  # operation_id, ui_data

    def __init__(
        self,
        parent: QtWidgets.QMainWindow,
        worker_manager: WorkerManager,
        progress_manager: ProgressManager,
    ):
        super().__init__(parent)
        self.main_window = parent
        self.worker_manager = worker_manager
        self.progress_manager = progress_manager

        # Operation tracking
        self.active_operations: dict[str, OperationContext] = {}
        self.operation_queue: list[OperationContext] = []
        self.max_concurrent_operations = 3

        # Progressive loading state
        self.progressive_loading_active = False
        self.progressive_operations: dict[str, dict] = {}

        # UI responsiveness monitoring
        self.ui_response_timer = QTimer()
        self.ui_response_timer.timeout.connect(self._check_ui_responsiveness)
        self.ui_response_timer.start(100)  # Check every 100ms
        self.last_ui_check = time.perf_counter()
        self.ui_freeze_threshold = 0.2  # 200ms threshold for UI freeze detection

        # Connect to worker manager
        self.worker_manager.worker_stats_updated.connect(self._on_worker_stats_updated)
        self.worker_manager.resource_limit_exceeded.connect(
            self._on_resource_limit_exceeded
        )

        logger.info("MainThreadOptimizer initialized")

    def optimize_file_loading(
        self,
        file_paths: list[str],
        load_function: Callable,
        progress_callback: Callable | None = None,
        completion_callback: Callable | None = None,
        batch_size: int = 5,
    ) -> str:
        """
        Optimize file loading by using background threads with progressive updates.

        Args:
            file_paths: List of files to load
            load_function: Function to load individual files
            progress_callback: Optional callback for progress updates
            completion_callback: Optional callback when loading completes
            batch_size: Number of files to load per batch

        Returns:
            Operation ID for tracking
        """
        operation_id = f"file_load_{int(time.time() * 1000)}"

        # Create operation context
        context = OperationContext(
            operation_id=operation_id,
            operation_type="file_loading",
            params={
                "file_paths": file_paths,
                "load_function": load_function,
                "batch_size": batch_size,
            },
            callbacks={
                "progress": progress_callback,
                "completion": completion_callback,
            },
        )

        # Start progress tracking
        self.progress_manager.start_operation(
            operation_id=operation_id,
            description=f"Loading {len(file_paths)} files",
            total=len(file_paths),
            show_in_statusbar=True,
        )

        # Create enhanced data load worker
        worker = DataLoadWorker(
            load_func=load_function,
            file_paths=file_paths,
            worker_id=operation_id,
            priority=WorkerPriority.HIGH,
            batch_size=batch_size,
        )

        # Connect worker signals
        worker.signals.progress.connect(self._on_loading_progress)
        worker.signals.partial_result.connect(self._on_partial_loading_result)
        worker.signals.finished.connect(self._on_loading_finished)
        worker.signals.error.connect(self._on_loading_error)

        # Track operation and submit
        self.active_operations[operation_id] = context
        self.worker_manager.submit_worker(worker)

        self.operation_started.emit(operation_id, "file_loading")
        logger.info(
            f"Started optimized file loading: {operation_id} ({len(file_paths)} files)"
        )

        return operation_id

    def optimize_plot_generation(
        self,
        plot_function: Callable,
        plot_args: tuple = (),
        plot_kwargs: dict | None = None,
        plot_type: str = "generic",
        cache_results: bool = True,
        priority: WorkerPriority = WorkerPriority.NORMAL,
    ) -> str:
        """
        Optimize plot generation by moving it to background with caching.

        Args:
            plot_function: The plotting function to execute
            plot_args: Arguments for the plot function
            plot_kwargs: Keyword arguments for the plot function
            plot_type: Type of plot for tracking
            cache_results: Whether to cache plot results
            priority: Worker priority level

        Returns:
            Operation ID for tracking
        """
        operation_id = f"plot_{plot_type}_{int(time.time() * 1000)}"

        # Create operation context
        context = OperationContext(
            operation_id=operation_id,
            operation_type="plot_generation",
            params={
                "plot_function": plot_function,
                "plot_args": plot_args,
                "plot_kwargs": plot_kwargs or {},
                "plot_type": plot_type,
            },
            is_critical=False,
            user_initiated=True,
        )

        # Start progress tracking
        self.progress_manager.start_operation(
            operation_id=operation_id,
            description=f"Generating {plot_type} plot",
            total=100,
            show_in_statusbar=False,
        )

        # Create enhanced plot worker
        worker = PlotWorker(
            plot_func=plot_function,
            plot_args=plot_args,
            plot_kwargs=plot_kwargs or {},
            worker_id=operation_id,
            priority=priority,
            cache_results=cache_results,
        )

        # Connect worker signals
        worker.signals.progress.connect(self._on_plot_progress)
        worker.signals.finished.connect(self._on_plot_finished)
        worker.signals.error.connect(self._on_plot_error)

        # Track operation and submit
        self.active_operations[operation_id] = context
        self.worker_manager.submit_worker(worker)

        self.operation_started.emit(operation_id, "plot_generation")
        logger.info(f"Started optimized plot generation: {operation_id} ({plot_type})")

        return operation_id

    def implement_progressive_loading(
        self,
        total_items: int,
        loader_function: Callable,
        ui_update_function: Callable,
        batch_size: int = 10,
        update_interval: float = 0.1,
    ) -> str:
        """
        Implement progressive loading with immediate UI feedback.

        Args:
            total_items: Total number of items to load
            loader_function: Function to load items (should accept start, end indices)
            ui_update_function: Function to update UI with loaded items
            batch_size: Items to load per batch
            update_interval: Minimum time between UI updates (seconds)

        Returns:
            Operation ID for tracking
        """
        operation_id = f"progressive_{int(time.time() * 1000)}"

        # Initialize progressive loading state
        self.progressive_operations[operation_id] = {
            "total_items": total_items,
            "loaded_items": 0,
            "batch_size": batch_size,
            "loader_function": loader_function,
            "ui_update_function": ui_update_function,
            "update_interval": update_interval,
            "last_update_time": 0,
            "batch_results": [],
        }

        # Start progress tracking
        self.progress_manager.start_operation(
            operation_id=operation_id,
            description=f"Progressive loading ({total_items} items)",
            total=total_items,
            show_in_statusbar=True,
        )

        # Start progressive loading
        self._start_progressive_batch(operation_id)

        self.operation_started.emit(operation_id, "progressive_loading")
        logger.info(
            f"Started progressive loading: {operation_id} ({total_items} items)"
        )

        return operation_id

    def _start_progressive_batch(self, operation_id: str):
        """Start loading the next batch for progressive loading."""
        if operation_id not in self.progressive_operations:
            return

        prog_state = self.progressive_operations[operation_id]

        # Check if we're done
        if prog_state["loaded_items"] >= prog_state["total_items"]:
            self._finish_progressive_loading(operation_id)
            return

        # Calculate batch range
        start_idx = prog_state["loaded_items"]
        end_idx = min(start_idx + prog_state["batch_size"], prog_state["total_items"])

        # Create worker for this batch
        batch_operation_id = f"{operation_id}_batch_{start_idx}"

        class ProgressiveBatchWorker(DataLoadWorker):
            def __init__(self, operation_id, start_idx, end_idx, loader_func):
                self.operation_id = operation_id
                self.start_idx = start_idx
                self.end_idx = end_idx
                self.loader_func = loader_func
                super().__init__(
                    load_func=self._load_batch,
                    file_paths=[f"batch_{start_idx}_{end_idx}"],  # Dummy path
                    worker_id=batch_operation_id,
                    priority=WorkerPriority.HIGH,
                )

            def _load_batch(self, dummy_path):
                return self.loader_func(self.start_idx, self.end_idx)

        worker = ProgressiveBatchWorker(
            operation_id, start_idx, end_idx, prog_state["loader_function"]
        )

        # Connect signals
        worker.signals.finished.connect(
            lambda result: self._on_progressive_batch_finished(
                operation_id, start_idx, end_idx, result
            )
        )
        worker.signals.error.connect(
            lambda w_id, msg, tb, retry: self._on_progressive_batch_error(
                operation_id, msg
            )
        )

        # Submit worker
        self.worker_manager.submit_worker(worker)

    def _on_progressive_batch_finished(
        self, operation_id: str, start_idx: int, end_idx: int, result: WorkerResult
    ):
        """Handle completion of a progressive loading batch."""
        if operation_id not in self.progressive_operations:
            return

        if not result.success:
            self._on_progressive_batch_error(operation_id, result.error_message)
            return

        prog_state = self.progressive_operations[operation_id]

        # Update state
        prog_state["loaded_items"] = end_idx
        prog_state["batch_results"].append(result.result)

        # Update progress
        self.progress_manager.update_progress(
            operation_id,
            end_idx,
            prog_state["total_items"],
            f"Loaded {end_idx}/{prog_state['total_items']} items",
        )

        # Update UI if enough time has passed
        current_time = time.perf_counter()
        if (
            current_time - prog_state["last_update_time"]
            >= prog_state["update_interval"]
        ):
            try:
                prog_state["ui_update_function"](prog_state["batch_results"])
                prog_state["last_update_time"] = current_time

                # Emit UI update signal
                self.ui_update_ready.emit(
                    operation_id,
                    {
                        "loaded_items": end_idx,
                        "total_items": prog_state["total_items"],
                        "batch_results": prog_state["batch_results"],
                    },
                )
            except Exception as e:
                logger.error(
                    f"Error updating UI for progressive loading {operation_id}: {e}"
                )

        # Start next batch
        self._start_progressive_batch(operation_id)

    def _on_progressive_batch_error(self, operation_id: str, error_message: str):
        """Handle error in progressive loading batch."""
        logger.error(
            f"Progressive loading batch failed for {operation_id}: {error_message}"
        )

        # Mark operation as failed
        self.progress_manager.complete_operation(
            operation_id,
            success=False,
            final_message=f"Progressive loading failed: {error_message}",
        )

        if operation_id in self.progressive_operations:
            del self.progressive_operations[operation_id]

        self.operation_failed.emit(operation_id, error_message)

    def _finish_progressive_loading(self, operation_id: str):
        """Finish progressive loading operation."""
        if operation_id not in self.progressive_operations:
            return

        prog_state = self.progressive_operations[operation_id]

        # Final UI update
        try:
            prog_state["ui_update_function"](prog_state["batch_results"])
        except Exception as e:
            logger.error(
                f"Error in final UI update for progressive loading {operation_id}: {e}"
            )

        # Complete operation
        self.progress_manager.complete_operation(
            operation_id,
            success=True,
            final_message=f"Loaded {prog_state['loaded_items']} items",
        )

        # Clean up
        del self.progressive_operations[operation_id]

        # Emit completion
        self.operation_completed.emit(operation_id, prog_state["batch_results"])
        logger.info(f"Progressive loading completed: {operation_id}")

    def validate_parameters_async(
        self,
        params: dict[str, Any],
        validation_function: Callable,
        validation_callback: Callable | None = None,
    ) -> str:
        """
        Validate parameters in background thread to avoid blocking main thread.

        Args:
            params: Parameters to validate
            validation_function: Function that validates the parameters
            validation_callback: Optional callback for validation results

        Returns:
            Operation ID for tracking
        """
        operation_id = f"validation_{int(time.time() * 1000)}"

        from .async_workers_enhanced import EnhancedComputationWorker

        worker = EnhancedComputationWorker(
            compute_func=validation_function,
            data=params,
            worker_id=operation_id,
            priority=WorkerPriority.HIGH,  # High priority for validation
        )

        # Connect signals
        worker.signals.finished.connect(
            lambda result: self._on_validation_finished(
                operation_id, result, validation_callback
            )
        )
        worker.signals.error.connect(
            lambda w_id, msg, tb, retry: self._on_validation_error(
                operation_id, msg, validation_callback
            )
        )

        # Submit worker
        self.worker_manager.submit_worker(worker)

        logger.debug(f"Started parameter validation: {operation_id}")
        return operation_id

    def _on_validation_finished(
        self, operation_id: str, result: WorkerResult, callback: Callable | None
    ):
        """Handle parameter validation completion."""
        if callback:
            try:
                callback(
                    result.result if result.success else None, result.error_message
                )
            except Exception as e:
                logger.error(f"Error in validation callback for {operation_id}: {e}")

        logger.debug(f"Parameter validation completed: {operation_id}")

    def _on_validation_error(
        self, operation_id: str, error_message: str, callback: Callable | None
    ):
        """Handle parameter validation error."""
        if callback:
            try:
                callback(None, error_message)
            except Exception as e:
                logger.error(
                    f"Error in validation error callback for {operation_id}: {e}"
                )

        logger.warning(f"Parameter validation failed: {operation_id} - {error_message}")

    def cancel_operation(
        self, operation_id: str, reason: str = "User requested"
    ) -> bool:
        """Cancel an active operation."""
        # Cancel in worker manager
        if self.worker_manager.cancel_worker(operation_id, reason):
            # Clean up local tracking
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]

            if operation_id in self.progressive_operations:
                del self.progressive_operations[operation_id]

            # Complete progress tracking
            self.progress_manager.cancel_operation_ui(operation_id)

            logger.info(f"Cancelled operation {operation_id}: {reason}")
            return True

        return False

    def get_operation_status(self, operation_id: str) -> dict[str, Any]:
        """Get status information for an operation."""
        status = {
            "exists": False,
            "active": False,
            "type": None,
            "progress": None,
            "worker_active": False,
        }

        # Check active operations
        if operation_id in self.active_operations:
            context = self.active_operations[operation_id]
            status.update(
                {
                    "exists": True,
                    "active": True,
                    "type": context.operation_type,
                    "start_time": context.start_time,
                    "elapsed_time": time.perf_counter() - context.start_time,
                }
            )

        # Check progressive operations
        if operation_id in self.progressive_operations:
            prog_state = self.progressive_operations[operation_id]
            status.update(
                {
                    "exists": True,
                    "active": True,
                    "type": "progressive_loading",
                    "progress": {
                        "loaded": prog_state["loaded_items"],
                        "total": prog_state["total_items"],
                        "percentage": (
                            prog_state["loaded_items"] / prog_state["total_items"]
                        )
                        * 100,
                    },
                }
            )

        # Check worker manager
        status["worker_active"] = self.worker_manager.is_worker_active(operation_id)

        return status

    # Signal handlers
    @Slot(str, int, int, str, float)
    def _on_loading_progress(
        self, worker_id: str, current: int, total: int, message: str, eta: float
    ):
        """Handle loading progress updates."""
        self.progress_manager.update_progress(worker_id, current, total, message)

    @Slot(str, object, bool)
    def _on_partial_loading_result(
        self, worker_id: str, partial_result: Any, is_final: bool
    ):
        """Handle partial loading results for immediate UI updates."""
        if worker_id in self.active_operations:
            context = self.active_operations[worker_id]
            if context.callbacks.get("progress"):
                try:
                    context.callbacks["progress"](partial_result, is_final)
                except Exception as e:
                    logger.error(f"Error in progress callback for {worker_id}: {e}")

    @Slot(object)
    def _on_loading_finished(self, result: WorkerResult):
        """Handle loading completion."""
        operation_id = result.worker_id

        if operation_id in self.active_operations:
            context = self.active_operations[operation_id]

            # Call completion callback
            if context.callbacks.get("completion"):
                try:
                    context.callbacks["completion"](
                        result.result if result.success else None
                    )
                except Exception as e:
                    logger.error(
                        f"Error in completion callback for {operation_id}: {e}"
                    )

            # Clean up
            del self.active_operations[operation_id]

        # Complete progress tracking
        self.progress_manager.complete_operation(
            operation_id,
            result.success,
            f"Loading completed in {result.execution_time:.2f}s",
        )

        self.operation_completed.emit(operation_id, result.result)

    @Slot(str, str, str, int)
    def _on_loading_error(
        self, worker_id: str, error_message: str, traceback_str: str, retry_count: int
    ):
        """Handle loading errors."""
        if worker_id in self.active_operations:
            del self.active_operations[worker_id]

        self.progress_manager.complete_operation(
            worker_id, success=False, final_message=f"Loading failed: {error_message}"
        )

        self.operation_failed.emit(worker_id, error_message)

    @Slot(str, int, int, str, float)
    def _on_plot_progress(
        self, worker_id: str, current: int, total: int, message: str, eta: float
    ):
        """Handle plot generation progress."""
        self.progress_manager.update_progress(worker_id, current, total, message)

    @Slot(object)
    def _on_plot_finished(self, result: WorkerResult):
        """Handle plot generation completion."""
        operation_id = result.worker_id

        if operation_id in self.active_operations:
            del self.active_operations[operation_id]

        self.progress_manager.complete_operation(
            operation_id,
            result.success,
            f"Plot generated in {result.execution_time:.2f}s",
        )

        self.operation_completed.emit(operation_id, result.result)

    @Slot(str, str, str, int)
    def _on_plot_error(
        self, worker_id: str, error_message: str, traceback_str: str, retry_count: int
    ):
        """Handle plot generation errors."""
        if worker_id in self.active_operations:
            del self.active_operations[worker_id]

        self.progress_manager.complete_operation(
            worker_id,
            success=False,
            final_message=f"Plot generation failed: {error_message}",
        )

        self.operation_failed.emit(worker_id, error_message)

    @Slot(object)
    def _on_worker_stats_updated(self, stats):
        """Handle worker statistics updates for optimization decisions."""
        # Adjust max concurrent operations based on system performance
        if stats.avg_execution_time > 5.0:  # If average execution is > 5 seconds
            self.max_concurrent_operations = max(2, self.max_concurrent_operations - 1)
        elif stats.avg_execution_time < 1.0:  # If average execution is < 1 second
            self.max_concurrent_operations = min(5, self.max_concurrent_operations + 1)

    @Slot(str, float)
    def _on_resource_limit_exceeded(self, resource_type: str, current_value: float):
        """Handle resource limit exceeded events."""
        logger.warning(f"Resource limit exceeded: {resource_type} = {current_value}")

        # Cancel low priority operations if resources are constrained
        if resource_type in ["memory", "system_memory"]:
            cancelled = self.worker_manager.cancel_workers_by_priority(
                WorkerPriority.LOW, "Resource limit exceeded"
            )
            if cancelled > 0:
                logger.info(
                    f"Cancelled {cancelled} low priority operations due to memory pressure"
                )

    @Slot()
    def _check_ui_responsiveness(self):
        """Check if the UI is remaining responsive."""
        current_time = time.perf_counter()
        elapsed = current_time - self.last_ui_check

        if elapsed > self.ui_freeze_threshold:
            logger.warning(
                f"UI responsiveness issue detected: {elapsed:.3f}s since last check"
            )

            # If UI is freezing, try to reduce concurrent operations
            if len(self.active_operations) > 1:
                # Cancel the lowest priority operation
                lowest_priority_op = None
                lowest_priority = WorkerPriority.CRITICAL

                for op_id, context in self.active_operations.items():
                    if context.is_critical:
                        continue  # Don't cancel critical operations

                    # For now, assume NORMAL priority for existing operations
                    if WorkerPriority.NORMAL.value < lowest_priority.value:
                        lowest_priority = WorkerPriority.NORMAL
                        lowest_priority_op = op_id

                if lowest_priority_op:
                    self.cancel_operation(
                        lowest_priority_op, "UI responsiveness optimization"
                    )

        self.last_ui_check = current_time

    def shutdown(self):
        """Graceful shutdown of the optimizer."""
        logger.info("Shutting down MainThreadOptimizer...")

        # Stop UI monitoring
        self.ui_response_timer.stop()

        # Cancel all active operations
        for operation_id in list(self.active_operations.keys()):
            self.cancel_operation(operation_id, "Application shutdown")

        # Clear progressive operations
        self.progressive_operations.clear()

        logger.info("MainThreadOptimizer shutdown complete")
