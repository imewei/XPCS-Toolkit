"""
Asynchronous worker classes for XPCS Toolkit GUI threading and concurrency.

This module provides base classes and specific implementations for running
heavy plotting and data processing operations in background threads to
maintain GUI responsiveness.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from PySide6 import QtCore
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class WorkerState(Enum):
    """Worker execution states."""

    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerResult:
    """Container for worker execution results."""

    success: bool
    data: Any
    error: str
    execution_time: float

    # Optional fields for backward compatibility
    worker_id: str = ""
    memory_usage: float = 0.0


@dataclass
class WorkerStats:
    """Statistics for worker execution monitoring."""

    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    average_execution_time: float

    # Additional fields for enhanced monitoring
    active_workers: int = 0
    cancelled_workers: int = 0
    total_memory_usage: float = 0.0


class WorkerSignals(QObject):
    """
    Enhanced signals for communicating between worker threads and the main GUI thread.
    """

    # Signal emitted when work starts (worker_id, priority)
    started = Signal(str, int)

    # Signal emitted when work finishes successfully (worker_result)
    finished = Signal(object)

    # Signal emitted on error (worker_id, error_message, traceback_string, retry_count)
    error = Signal(str, str, str, int)

    # Signal emitted for progress updates (worker_id, current, total, message, eta_seconds)
    progress = Signal(str, int, int, str, float)

    # Signal emitted for status updates (worker_id, status_message, detail_level)
    status = Signal(str, str, int)

    # Signal emitted when operation is cancelled (worker_id, reason)
    cancelled = Signal(str, str)

    # Signal for partial results during long operations (worker_id, partial_result, is_final)
    partial_result = Signal(str, object, bool)

    # Signal for resource usage updates (worker_id, cpu_percent, memory_mb)
    resource_usage = Signal(str, float, float)

    # Signal for worker state changes (worker_id, old_state, new_state)
    state_changed = Signal(str, str, str)

    # Signal for retry attempts (worker_id, attempt_number, max_attempts)
    retry_attempt = Signal(str, int, int)


class BaseAsyncWorker(QRunnable):
    """
    Base class for asynchronous workers that run plotting and data operations
    in background threads.

    Features:
    - Cancellation support
    - Progress reporting with caching and rate limiting
    - Error handling with traceback
    - Partial result reporting for long operations
    """

    def __init__(self, worker_id: str | None = None):
        super().__init__()
        self.signals = WorkerSignals()
        self.worker_id = worker_id or f"worker_{id(self)}"
        self._is_cancelled = False
        self._start_time = None

        # Performance optimization: Progress emission caching and rate limiting
        self._last_progress_time = 0.0
        self._last_eta_calculation_time = 0.0
        self._cached_eta = 0.0
        self._cached_progress_data = None  # (current, total, message)
        self._progress_emission_interval = 0.1  # Max 10 emissions per second
        self._eta_calculation_interval = 0.1  # Recalculate ETA every 100ms

        # Performance measurement
        self._progress_call_count = 0
        self._progress_emission_count = 0
        self._eta_cache_hits = 0
        self._eta_calculations = 0
        self._total_progress_time = 0.0

        # Set auto-delete to False so we can reuse workers if needed
        self.setAutoDelete(True)

    @property
    def is_cancelled(self) -> bool:
        """Check if the worker has been cancelled."""
        return self._is_cancelled

    def cancel(self):
        """Cancel the worker operation."""
        self._is_cancelled = True
        logger.info(f"Worker {self.worker_id} cancelled")

    def emit_progress(self, current: int, total: int, message: str = ""):
        """
        Emit progress signal with current/total and optional message.
        Optimized with caching and rate limiting to improve GUI responsiveness.
        """
        if self._is_cancelled:
            return

        # Performance measurement - start timing
        method_start = time.perf_counter()
        self._progress_call_count += 1

        current_time = method_start

        # Rate limiting: Only emit progress updates at most every 100ms (10/second)
        time_since_last_emission = current_time - self._last_progress_time
        if time_since_last_emission < self._progress_emission_interval:
            # Cache the data for potential later emission
            self._cached_progress_data = (current, total, message)
            self._total_progress_time += time.perf_counter() - method_start
            return

        # ETA calculation with caching: Only recalculate every 100ms
        eta_seconds = 0.0
        if self._start_time and current > 0:
            time_since_last_eta = current_time - self._last_eta_calculation_time

            if (
                time_since_last_eta >= self._eta_calculation_interval
                or self._cached_eta == 0.0
            ):
                # Recalculate ETA
                elapsed = current_time - self._start_time
                eta_seconds = (elapsed / current) * (total - current)
                self._cached_eta = eta_seconds
                self._last_eta_calculation_time = current_time
                self._eta_calculations += 1
            else:
                # Use cached ETA
                eta_seconds = self._cached_eta
                self._eta_cache_hits += 1

        # Emit progress signal
        self.signals.progress.emit(self.worker_id, current, total, message, eta_seconds)
        self._last_progress_time = current_time
        self._cached_progress_data = None
        self._progress_emission_count += 1

        # Performance measurement - end timing
        self._total_progress_time += time.perf_counter() - method_start

    def get_performance_stats(self) -> dict[str, Any]:
        """
        Get performance statistics for the worker's progress handling.

        Returns:
            Dictionary with performance metrics
        """
        total_calls = max(self._progress_call_count, 1)  # Avoid division by zero

        return {
            "progress_calls": self._progress_call_count,
            "progress_emissions": self._progress_emission_count,
            "emission_rate": (self._progress_emission_count / total_calls) * 100,
            "eta_calculations": self._eta_calculations,
            "eta_cache_hits": self._eta_cache_hits,
            "eta_cache_hit_rate": (
                self._eta_cache_hits
                / max(self._eta_cache_hits + self._eta_calculations, 1)
            )
            * 100,
            "total_progress_time_ms": self._total_progress_time * 1000,
            "avg_progress_time_ms": (self._total_progress_time / total_calls) * 1000,
        }

    def log_performance_stats(self):
        """Log performance statistics to help validate optimizations."""
        stats = self.get_performance_stats()
        logger.info(f"Worker {self.worker_id} performance stats:")
        logger.info(f"  Progress calls: {stats['progress_calls']}")
        logger.info(
            f"  Actual emissions: {stats['progress_emissions']} ({stats['emission_rate']:.1f}%)"
        )
        logger.info(f"  ETA cache hit rate: {stats['eta_cache_hit_rate']:.1f}%")
        logger.info(f"  Average progress time: {stats['avg_progress_time_ms']:.3f}ms")

    def _emit_final_progress(self):
        """
        Emit any cached progress data and force final progress emission.
        Called at completion to ensure the last progress update is sent.
        """
        if self._is_cancelled:
            return

        if self._cached_progress_data:
            current, total, message = self._cached_progress_data
            # Force emission by bypassing rate limiting
            current_time = time.perf_counter()

            # Calculate final ETA (should be 0 for completion)
            eta_seconds = 0.0
            if self._start_time and current > 0 and current < total:
                elapsed = current_time - self._start_time
                eta_seconds = (elapsed / current) * (total - current)

            self.signals.progress.emit(
                self.worker_id, current, total, message, eta_seconds
            )
            self._cached_progress_data = None

    def emit_status(self, status: str):
        """Emit status update signal."""
        if not self._is_cancelled:
            # Emit status signal with worker_id, status_message, detail_level (1 = INFO)
            self.signals.status.emit(self.worker_id, status, 1)

    def emit_partial_result(self, result: Any):
        """Emit partial result for incremental updates."""
        if not self._is_cancelled:
            # Emit partial_result signal with worker_id, result, is_final (False for partial)
            self.signals.partial_result.emit(self.worker_id, result, False)

    def check_cancelled(self):
        """Check if cancelled and raise exception if so."""
        if self._is_cancelled:
            raise InterruptedError("Worker operation was cancelled")

    def do_work(self) -> Any:
        """
        Method to be implemented by subclasses.
        This method should contain the actual work to be done.

        Returns:
            Any: The result of the work

        Raises:
            InterruptedError: If the operation was cancelled
            Exception: Any other exception that occurs during work
        """
        raise NotImplementedError("Subclasses must implement do_work() method")

    @Slot()
    def run(self):
        """
        Main entry point for the worker thread.
        Handles error catching, timing, and signal emission.
        """
        try:
            self._start_time = time.perf_counter()
            # Emit started signal with worker_id and default priority (2 = NORMAL)
            self.signals.started.emit(self.worker_id, 2)

            logger.info(f"Worker {self.worker_id} started")

            # Do the actual work
            result = self.do_work()

            # Check if we were cancelled during work
            self.check_cancelled()

            # Check if result is an info message rather than actual results
            if isinstance(result, dict) and result.get("type") == "info":
                # Handle info messages differently
                elapsed = time.perf_counter() - self._start_time
                info_msg = result.get("message", "No data available")
                logger.warning(
                    f"Worker {self.worker_id} completed with info message: {info_msg}"
                )
                # Emit status signal with worker_id, info_message, detail_level (2 = WARNING)
                self.signals.status.emit(self.worker_id, info_msg, 2)
                self.signals.finished.emit(None)  # No actual plot data
                return

            # Emit any final cached progress
            self._emit_final_progress()

            # Log performance statistics if there were progress calls
            if self._progress_call_count > 0:
                self.log_performance_stats()

            # Emit success
            elapsed = time.perf_counter() - self._start_time
            logger.info(f"Worker {self.worker_id} completed in {elapsed:.2f}s")
            self.signals.finished.emit(result)

        except InterruptedError:
            # Operation was cancelled
            logger.info(f"Worker {self.worker_id} was cancelled")
            self.signals.cancelled.emit(self.worker_id, "User cancelled")

        except Exception as e:
            # Handle any other exception
            error_msg = f"Error in worker {self.worker_id}: {e!s}"
            tb_str = traceback.format_exc()
            logger.error(f"{error_msg}\n{tb_str}")
            # Emit error signal with worker_id, error_message, traceback_string, retry_count (0)
            self.signals.error.emit(self.worker_id, error_msg, tb_str, 0)


class PlotWorker(BaseAsyncWorker):
    """
    Worker for plot operations that take plot function and arguments.
    """

    def __init__(
        self,
        plot_func: Callable,
        plot_args: tuple = (),
        plot_kwargs: dict | None = None,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id)
        self.plot_func = plot_func
        self.plot_args = plot_args
        self.plot_kwargs = plot_kwargs or {}

    def do_work(self) -> Any:
        """Execute the plot function with provided arguments."""
        self.emit_status("Starting plot operation...")

        # Call the plotting function
        result = self.plot_func(*self.plot_args, **self.plot_kwargs)

        self.emit_status("Plot operation completed")
        return result


class DataLoadWorker(BaseAsyncWorker):
    """
    Worker for data loading operations with progress reporting.
    """

    def __init__(
        self,
        load_func: Callable,
        file_paths: list,
        load_kwargs: dict | None = None,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id)
        self.load_func = load_func
        self.file_paths = file_paths
        self.load_kwargs = load_kwargs or {}
        self.loaded_data = []

    def do_work(self) -> list:
        """Load data from multiple files with progress reporting."""
        total_files = len(self.file_paths)
        self.emit_status(f"Loading {total_files} files...")

        for i, file_path in enumerate(self.file_paths):
            self.check_cancelled()

            self.emit_progress(i, total_files, f"Loading {file_path}")

            # Load individual file
            try:
                data = self.load_func(file_path, **self.load_kwargs)
                self.loaded_data.append(data)

                # Emit partial result for immediate GUI updates
                self.emit_partial_result(
                    {"file_index": i, "file_path": file_path, "data": data}
                )

            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
                self.loaded_data.append(None)

        self.emit_progress(total_files, total_files, "Loading complete")
        return self.loaded_data


class ComputationWorker(BaseAsyncWorker):
    """
    Worker for heavy computational tasks with periodic progress checks.
    """

    def __init__(
        self,
        compute_func: Callable,
        data: Any,
        progress_callback: Callable | None = None,
        progress_interval: float = 1.0,
        compute_kwargs: dict | None = None,
        worker_id: str | None = None,
    ):
        super().__init__(worker_id)
        self.compute_func = compute_func
        self.data = data
        self.progress_callback = progress_callback
        # Use the base class rate limiting, but allow override for compute workers
        if progress_interval != 1.0:
            self._progress_emission_interval = progress_interval
        self.compute_kwargs = compute_kwargs or {}

    def report_progress(self, current: int, total: int, message: str = ""):
        """Report progress using the base class optimized emission."""
        # Base class handles rate limiting and caching automatically
        self.emit_progress(current, total, message)

    def do_work(self) -> Any:
        """Execute computation with progress monitoring."""
        self.emit_status("Starting computation...")

        # If progress callback is provided, wrap it to include cancellation checks
        if self.progress_callback:
            original_callback = self.progress_callback

            def wrapped_callback(*args, **kwargs):
                self.check_cancelled()
                result = original_callback(*args, **kwargs)
                if len(args) >= 2:
                    self.report_progress(
                        args[0], args[1], kwargs.get("message", "Computing...")
                    )
                return result

            self.compute_kwargs["progress_callback"] = wrapped_callback

        # Execute the computation
        result = self.compute_func(self.data, **self.compute_kwargs)

        self.emit_status("Computation completed")
        return result


class WorkerManager(QObject):
    """
    Manager class for coordinating multiple async workers.
    Provides centralized control over worker lifecycle and progress tracking.
    """

    # Signals for overall progress tracking
    all_workers_started = Signal()
    all_workers_finished = Signal()
    worker_progress = Signal(str, int, int, str)  # worker_id, current, total, message

    def __init__(self, thread_pool: QtCore.QThreadPool):
        super().__init__()
        self.thread_pool = thread_pool
        self.active_workers = {}  # worker_id -> worker
        self.worker_results = {}  # worker_id -> result
        self.worker_errors = {}  # worker_id -> (error_msg, traceback)

        # Register for optimized cleanup
        try:
            from .cleanup_optimized import register_for_cleanup

            register_for_cleanup(f"WorkerManager_{id(self)}", self)
        except ImportError:
            pass  # Optimized cleanup system not available

    def submit_worker(self, worker: BaseAsyncWorker) -> str:
        """
        Submit a worker to the thread pool and track it.

        Returns:
            str: The worker ID for tracking
        """
        worker_id = worker.worker_id

        # Connect signals
        worker.signals.started.connect(lambda: self._on_worker_started(worker_id))
        worker.signals.finished.connect(
            lambda result: self._on_worker_finished(worker_id, result)
        )
        worker.signals.error.connect(
            lambda msg, tb: self._on_worker_error(worker_id, msg, tb)
        )
        worker.signals.progress.connect(
            lambda current, total, message: self._on_worker_progress(
                worker_id, current, total, message
            )
        )
        worker.signals.cancelled.connect(lambda: self._on_worker_cancelled(worker_id))

        # Track the worker
        self.active_workers[worker_id] = worker

        # Submit to thread pool
        self.thread_pool.start(worker)

        logger.info(f"Submitted worker {worker_id} to thread pool")
        return worker_id

    def cancel_worker(self, worker_id: str):
        """Cancel a specific worker by ID."""
        if worker_id in self.active_workers:
            self.active_workers[worker_id].cancel()

    def cancel_all_workers(self):
        """Cancel all active workers."""
        for worker in self.active_workers.values():
            worker.cancel()

    def get_worker_result(self, worker_id: str) -> Any:
        """Get the result of a completed worker."""
        return self.worker_results.get(worker_id)

    def get_worker_error(self, worker_id: str) -> tuple[str, str] | None:
        """Get the error information for a failed worker."""
        return self.worker_errors.get(worker_id)

    def is_worker_active(self, worker_id: str) -> bool:
        """Check if a worker is still active."""
        return worker_id in self.active_workers

    @Slot(str)
    def _on_worker_started(self, worker_id: str):
        """Handle worker started signal."""
        logger.debug(f"Worker {worker_id} started")

    @Slot(str, object)
    def _on_worker_finished(self, worker_id: str, result: Any):
        """Handle worker finished signal."""
        self.worker_results[worker_id] = result
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        logger.debug(f"Worker {worker_id} finished")

        if not self.active_workers:
            self.all_workers_finished.emit()

    @Slot(str, str, str)
    def _on_worker_error(self, worker_id: str, error_msg: str, traceback_str: str):
        """Handle worker error signal."""
        self.worker_errors[worker_id] = (error_msg, traceback_str)
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        logger.error(f"Worker {worker_id} failed: {error_msg}")

    @Slot(str, int, int, str)
    def _on_worker_progress(
        self, worker_id: str, current: int, total: int, message: str
    ):
        """Handle worker progress signal."""
        self.worker_progress.emit(worker_id, current, total, message)

    @Slot(str)
    def _on_worker_cancelled(self, worker_id: str):
        """Handle worker cancelled signal."""
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
        logger.info(f"Worker {worker_id} was cancelled")

    def shutdown(self):
        """
        Shutdown the WorkerManager gracefully.

        Cancels all active workers and cleans up resources.
        This method provides compatibility with the enhanced WorkerManager.
        """
        logger.debug(
            f"WorkerManager shutdown: cancelling {len(self.active_workers)} active workers"
        )

        # Cancel all active workers
        for worker_id, worker in list(self.active_workers.items()):
            try:
                if hasattr(worker, "cancel"):
                    worker.cancel()
            except Exception as e:
                logger.debug(f"Error cancelling worker {worker_id}: {e}")

        # Clear collections
        self.active_workers.clear()
        self.worker_results.clear()
        self.worker_errors.clear()

        logger.debug("WorkerManager shutdown complete")
