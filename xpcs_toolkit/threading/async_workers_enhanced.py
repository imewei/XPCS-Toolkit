"""
Enhanced asynchronous worker classes for XPCS Toolkit GUI threading and concurrency.

This module provides advanced base classes and specific implementations for running
heavy plotting and data processing operations in background threads with:
- Priority-based execution
- Result caching with TTL
- Automatic retry with exponential backoff
- Resource usage monitoring
- Graceful cancellation with cleanup
- Progress estimation and ETA calculation
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Tuple

import psutil
from PySide6 import QtCore
from PySide6.QtCore import QMutex, QObject, QRunnable, QTimer, Signal, Slot

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class WorkerPriority(Enum):
    """Priority levels for worker operations."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkerState(Enum):
    """Worker execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerResult:
    """Container for worker execution results."""

    worker_id: str
    result: Any
    execution_time: float
    memory_usage: float
    success: bool = True
    error_message: str = ""


@dataclass
class WorkerStats:
    """Statistics for worker execution monitoring."""

    total_workers: int = 0
    active_workers: int = 0
    completed_workers: int = 0
    failed_workers: int = 0
    cancelled_workers: int = 0
    avg_execution_time: float = 0.0
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
    Enhanced base class for asynchronous workers with advanced features:
    - Priority-based execution
    - Result caching with TTL
    - Automatic retry with exponential backoff
    - Resource usage monitoring
    - Graceful cancellation with cleanup
    - Progress estimation and ETA calculation
    """

    def __init__(
        self,
        worker_id: str | None = None,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        cache_results: bool = False,
        cache_ttl: float = 300.0,
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ):
        super().__init__()
        self.signals = WorkerSignals()
        self.worker_id = worker_id or f"worker_{id(self)}"
        self.priority = priority
        self.cache_results = cache_results
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Execution state
        self._state = WorkerState.PENDING
        self._is_cancelled = False
        self._start_time = None
        self._end_time = None
        self._retry_count = 0
        self._cancel_reason = ""

        # Resource monitoring
        self._process = psutil.Process()
        self._initial_memory = 0
        self._peak_memory = 0

        # Progress tracking
        self._last_progress_time = 0
        self._progress_history = deque(maxlen=10)

        # Thread safety
        self._state_mutex = QMutex()

        # Set auto-delete to True for proper cleanup
        self.setAutoDelete(True)

    @property
    def is_cancelled(self) -> bool:
        """Check if the worker has been cancelled."""
        with self._state_mutex:
            return self._is_cancelled

    @property
    def state(self) -> WorkerState:
        """Get the current worker state."""
        with self._state_mutex:
            return self._state

    @property
    def execution_time(self) -> float:
        """Get the execution time in seconds."""
        if self._start_time is None:
            return 0.0
        end_time = self._end_time or time.perf_counter()
        return end_time - self._start_time

    def cancel(self, reason: str = "User requested"):
        """Cancel the worker operation with cleanup."""
        with self._state_mutex:
            if self._state in (
                WorkerState.COMPLETED,
                WorkerState.FAILED,
                WorkerState.CANCELLED,
            ):
                return  # Already finished

            old_state = self._state.value
            self._is_cancelled = True
            self._cancel_reason = reason
            self._state = WorkerState.CANCELLED

        self.signals.state_changed.emit(
            self.worker_id, old_state, WorkerState.CANCELLED.value
        )
        logger.info(f"Worker {self.worker_id} cancelled: {reason}")

        # Perform cleanup if needed
        self._cleanup()

    def _cleanup(self):
        """Cleanup resources. Override in subclasses if needed."""
        pass

    def _set_state(self, new_state: WorkerState):
        """Thread-safe state setter."""
        with self._state_mutex:
            if self._is_cancelled:
                return  # Don't change state if cancelled
            old_state = self._state.value
            self._state = new_state

        self.signals.state_changed.emit(self.worker_id, old_state, new_state.value)

    def emit_progress(self, current: int, total: int, message: str = ""):
        """Emit progress signal with ETA calculation."""
        if self._is_cancelled:
            return

        # Calculate ETA based on progress history
        eta_seconds = self._calculate_eta(current, total)

        # Track progress for better ETA estimation
        current_time = time.perf_counter()
        if self._start_time:
            elapsed = current_time - self._start_time
            self._progress_history.append((elapsed, current, total))

        self.signals.progress.emit(self.worker_id, current, total, message, eta_seconds)

    def emit_status(self, status: str, detail_level: int = 1):
        """Emit status update signal with detail level."""
        if not self._is_cancelled:
            self.signals.status.emit(self.worker_id, status, detail_level)

    def emit_partial_result(self, result: Any, is_final: bool = False):
        """Emit partial result for incremental updates."""
        if not self._is_cancelled:
            self.signals.partial_result.emit(self.worker_id, result, is_final)

    def emit_resource_usage(self):
        """Emit current resource usage."""
        if self._is_cancelled:
            return

        try:
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)

            if memory_mb > self._peak_memory:
                self._peak_memory = memory_mb

            self.signals.resource_usage.emit(self.worker_id, cpu_percent, memory_mb)
        except Exception as e:
            logger.debug(f"Failed to get resource usage for {self.worker_id}: {e}")

    def _calculate_eta(self, current: int, total: int) -> float:
        """Calculate estimated time to completion based on progress history."""
        if not self._progress_history or current <= 0 or total <= current:
            return 0.0

        # Use weighted average of recent progress rates
        recent_rates = []
        for i in range(1, len(self._progress_history)):
            prev_time, prev_current, _ = self._progress_history[i - 1]
            curr_time, curr_current, _ = self._progress_history[i]

            time_delta = curr_time - prev_time
            progress_delta = curr_current - prev_current

            if time_delta > 0 and progress_delta > 0:
                rate = progress_delta / time_delta
                recent_rates.append(rate)

        if not recent_rates:
            return 0.0

        # Weight recent rates more heavily
        weights = [2**i for i in range(len(recent_rates))]
        weighted_rate = sum(
            rate * weight for rate, weight in zip(recent_rates, weights)
        ) / sum(weights)

        remaining = total - current
        return remaining / weighted_rate if weighted_rate > 0 else 0.0

    def check_cancelled(self):
        """Check if cancelled and raise exception if so."""
        if self._is_cancelled:
            raise InterruptedError(
                f"Worker operation was cancelled: {self._cancel_reason}"
            )

    def _should_retry(self, exception: Exception) -> bool:
        """Determine if the operation should be retried based on the exception type."""
        # Don't retry on cancellation or certain critical errors
        if isinstance(exception, (InterruptedError, KeyboardInterrupt)):
            return False

        # Don't retry if max retries reached
        if self._retry_count >= self.max_retries:
            return False

        # Retry on common transient errors
        retry_exceptions = (IOError, OSError, ConnectionError, TimeoutError)
        return isinstance(exception, retry_exceptions)

    def _get_cache_key(self) -> str:
        """Generate a cache key for this worker's operation."""
        # Override in subclasses to implement caching
        return f"{self.__class__.__name__}_{self.worker_id}"

    def _get_cached_result(self) -> Tuple[bool, Any]:
        """Check for cached result. Returns (found, result)."""
        if not self.cache_results:
            return False, None

        # Implementation would depend on the caching backend
        # For now, return no cache hit
        return False, None

    def _cache_result(self, result: Any):
        """Cache the result if caching is enabled."""
        if not self.cache_results:
            return

        # Implementation would depend on the caching backend
        # For now, just log that we would cache
        logger.debug(f"Would cache result for {self.worker_id}")

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
        Enhanced main entry point with retry logic, caching, and resource monitoring.
        """
        self._start_time = time.perf_counter()
        self._initial_memory = self._process.memory_info().rss / (1024 * 1024)

        try:
            self._set_state(WorkerState.RUNNING)
            self.signals.started.emit(self.worker_id, self.priority.value)

            logger.info(
                f"Worker {self.worker_id} started (priority: {self.priority.name})"
            )

            # Check for cached result first
            if self.cache_results:
                found, cached_result = self._get_cached_result()
                if found:
                    logger.info(f"Worker {self.worker_id} using cached result")
                    self._handle_success(cached_result)
                    return

            # Execute with retry logic
            result = self._execute_with_retry()
            self._handle_success(result)

        except InterruptedError:
            self._handle_cancellation()
        except Exception as e:
            self._handle_error(e)
        finally:
            self._end_time = time.perf_counter()
            self._cleanup()

    def _execute_with_retry(self) -> Any:
        """Execute the work with retry logic."""
        last_exception = None

        while self._retry_count <= self.max_retries:
            try:
                self.check_cancelled()

                if self._retry_count > 0:
                    self.emit_status(
                        f"Retry attempt {self._retry_count}/{self.max_retries}", 2
                    )
                    self.signals.retry_attempt.emit(
                        self.worker_id, self._retry_count, self.max_retries
                    )
                    time.sleep(
                        self.retry_delay * (2 ** (self._retry_count - 1))
                    )  # Exponential backoff

                # Emit resource usage periodically
                self.emit_resource_usage()

                # Do the actual work
                result = self.do_work()

                # Check if we were cancelled during work
                self.check_cancelled()

                return result

            except Exception as e:
                last_exception = e
                self._retry_count += 1

                if not self._should_retry(e):
                    raise e

                logger.warning(
                    f"Worker {self.worker_id} attempt {self._retry_count} failed: {e}"
                )

        # Max retries exceeded
        raise last_exception

    def _handle_success(self, result: Any):
        """Handle successful completion."""
        self._set_state(WorkerState.COMPLETED)

        # Check if result is an info message rather than actual results
        if isinstance(result, dict) and result.get("type") == "info":
            info_msg = result.get("message", "No data available")
            logger.warning(
                f"Worker {self.worker_id} completed with info message: {info_msg}"
            )
            self.emit_status(info_msg, 1)
            result = None  # No actual data
        else:
            # Cache successful result
            if self.cache_results:
                self._cache_result(result)

        # Create result object with metrics
        worker_result = WorkerResult(
            worker_id=self.worker_id,
            result=result,
            execution_time=self.execution_time,
            memory_usage=self._peak_memory - self._initial_memory,
            success=True,
        )

        elapsed = self.execution_time
        logger.info(f"Worker {self.worker_id} completed successfully in {elapsed:.2f}s")
        self.signals.finished.emit(worker_result)

    def _handle_cancellation(self):
        """Handle operation cancellation."""
        logger.info(f"Worker {self.worker_id} was cancelled: {self._cancel_reason}")
        self.signals.cancelled.emit(self.worker_id, self._cancel_reason)

    def _handle_error(self, exception: Exception):
        """Handle operation error."""
        self._set_state(WorkerState.FAILED)

        error_msg = f"Error in worker {self.worker_id}: {str(exception)}"
        tb_str = traceback.format_exc()
        logger.error(f"{error_msg}\n{tb_str}")

        # Create failed result object
        worker_result = WorkerResult(
            worker_id=self.worker_id,
            result=None,
            execution_time=self.execution_time,
            memory_usage=self._peak_memory - self._initial_memory,
            success=False,
            error_message=error_msg,
        )

        self.signals.error.emit(self.worker_id, error_msg, tb_str, self._retry_count)
        self.signals.finished.emit(worker_result)


class PlotWorker(BaseAsyncWorker):
    """
    Enhanced worker for plot operations with caching and progress monitoring.
    """

    def __init__(
        self,
        plot_func: Callable,
        plot_args: tuple = (),
        plot_kwargs: dict = None,
        worker_id: str | None = None,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        cache_results: bool = True,
        cache_ttl: float = 300.0,
    ):
        super().__init__(worker_id, priority, cache_results, cache_ttl)
        self.plot_func = plot_func
        self.plot_args = plot_args
        self.plot_kwargs = plot_kwargs or {}

    def _get_cache_key(self) -> str:
        """Generate cache key based on function and arguments."""
        try:
            # Create a hash of the function name and arguments
            func_name = getattr(self.plot_func, "__name__", str(self.plot_func))
            args_hash = hashlib.md5(
                pickle.dumps((self.plot_args, self.plot_kwargs)), usedforsecurity=False
            ).hexdigest()[:8]
            return f"plot_{func_name}_{args_hash}"
        except Exception:
            return super()._get_cache_key()

    def do_work(self) -> Any:
        """Execute the plot function with progress monitoring."""
        self.emit_status("Preparing plot operation...", 2)
        self.emit_progress(0, 100, "Initializing plot")

        # Emit resource usage before heavy operation
        self.emit_resource_usage()

        self.emit_progress(20, 100, "Processing data")
        self.check_cancelled()

        self.emit_status("Executing plot function...", 1)
        self.emit_progress(40, 100, "Rendering plot")

        # Call the plotting function
        result = self.plot_func(*self.plot_args, **self.plot_kwargs)

        self.emit_progress(80, 100, "Finalizing plot")
        self.check_cancelled()

        # Emit final resource usage
        self.emit_resource_usage()

        self.emit_progress(100, 100, "Plot completed")
        self.emit_status("Plot operation completed successfully", 1)
        return result


class DataLoadWorker(BaseAsyncWorker):
    """
    Enhanced worker for data loading operations with smart progress reporting and error recovery.
    """

    def __init__(
        self,
        load_func: Callable,
        file_paths: list,
        load_kwargs: dict = None,
        worker_id: str | None = None,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        max_retries: int = 2,
        batch_size: int = 5,
    ):
        super().__init__(worker_id, priority, max_retries=max_retries)
        self.load_func = load_func
        self.file_paths = file_paths
        self.load_kwargs = load_kwargs or {}
        self.batch_size = batch_size
        self.loaded_data = []
        self.failed_files = []
        self.load_times = []

    def do_work(self) -> dict:
        """Load data from multiple files with smart batching and detailed progress."""
        total_files = len(self.file_paths)
        self.emit_status(
            f"Loading {total_files} files in batches of {self.batch_size}...", 1
        )

        # Process files in batches for better memory management
        for batch_start in range(0, total_files, self.batch_size):
            self.check_cancelled()

            batch_end = min(batch_start + self.batch_size, total_files)
            batch_files = self.file_paths[batch_start:batch_end]

            self.emit_status(
                f"Processing batch {batch_start // self.batch_size + 1}...", 2
            )

            # Load files in current batch
            for i, file_path in enumerate(batch_files):
                global_index = batch_start + i
                self.check_cancelled()

                file_name = os.path.basename(file_path)
                self.emit_progress(global_index, total_files, f"Loading {file_name}")

                # Load individual file with timing
                load_start = time.perf_counter()
                try:
                    data = self.load_func(file_path, **self.load_kwargs)
                    load_time = time.perf_counter() - load_start
                    self.load_times.append(load_time)

                    self.loaded_data.append(data)

                    # Emit partial result for immediate GUI updates
                    self.emit_partial_result(
                        {
                            "file_index": global_index,
                            "file_path": file_path,
                            "data": data,
                            "load_time": load_time,
                            "batch_index": batch_start // self.batch_size,
                        },
                        is_final=False,
                    )

                    if global_index % 10 == 0:  # Emit resource usage every 10 files
                        self.emit_resource_usage()

                except Exception as e:
                    load_time = time.perf_counter() - load_start
                    error_msg = f"Failed to load {file_path}: {e}"
                    logger.warning(error_msg)

                    self.loaded_data.append(None)
                    self.failed_files.append((file_path, str(e)))

                    # Emit partial result with error info
                    self.emit_partial_result(
                        {
                            "file_index": global_index,
                            "file_path": file_path,
                            "data": None,
                            "error": str(e),
                            "load_time": load_time,
                            "batch_index": batch_start // self.batch_size,
                        },
                        is_final=False,
                    )

        # Final progress and summary
        self.emit_progress(total_files, total_files, "Loading complete")

        # Emit final resource usage
        self.emit_resource_usage()

        # Create comprehensive result
        successful_loads = sum(1 for data in self.loaded_data if data is not None)
        avg_load_time = (
            sum(self.load_times) / len(self.load_times) if self.load_times else 0
        )

        result = {
            "loaded_data": self.loaded_data,
            "successful_count": successful_loads,
            "failed_count": len(self.failed_files),
            "failed_files": self.failed_files,
            "total_files": total_files,
            "average_load_time": avg_load_time,
            "total_load_time": sum(self.load_times),
        }

        self.emit_status(
            f"Loaded {successful_loads}/{total_files} files successfully "
            f"(avg: {avg_load_time:.2f}s per file)",
            1,
        )

        return result


class ComputationWorker(BaseAsyncWorker):
    """
    Enhanced worker for heavy computational tasks with smart resource management.
    """

    def __init__(
        self,
        compute_func: Callable,
        data: Any,
        progress_callback: Callable | None = None,
        progress_interval: float = 0.5,
        compute_kwargs: dict = None,
        worker_id: str | None = None,
        priority: WorkerPriority = WorkerPriority.NORMAL,
        cache_results: bool = False,
        max_retries: int = 1,
    ):
        super().__init__(worker_id, priority, cache_results, max_retries=max_retries)
        self.compute_func = compute_func
        self.data = data
        self.progress_callback = progress_callback
        self.progress_interval = progress_interval
        self.compute_kwargs = compute_kwargs or {}
        self._last_progress_time = 0
        self._last_resource_check = 0
        self._resource_check_interval = 2.0

    def report_progress(self, current: int, total: int, message: str = ""):
        """Report progress and resource usage if enough time has elapsed."""
        current_time = time.perf_counter()
        if current_time - self._last_progress_time >= self.progress_interval:
            self.emit_progress(current, total, message)
            self._last_progress_time = current_time

            # Check resource usage periodically
            if (
                current_time - self._last_resource_check
                >= self._resource_check_interval
            ):
                self.emit_resource_usage()
                self._last_resource_check = current_time

    def _get_cache_key(self) -> str:
        """Generate cache key for computation results."""
        try:
            func_name = getattr(self.compute_func, "__name__", str(self.compute_func))
            data_hash = hashlib.md5(pickle.dumps(self.data, protocol=2), usedforsecurity=False).hexdigest()[:8]
            kwargs_hash = hashlib.md5(
                pickle.dumps(self.compute_kwargs, protocol=2), usedforsecurity=False
            ).hexdigest()[:8]
            return f"compute_{func_name}_{data_hash}_{kwargs_hash}"
        except Exception:
            return super()._get_cache_key()

    def do_work(self) -> Any:
        """Execute computation with enhanced monitoring and error handling."""
        self.emit_status("Initializing computation...", 2)
        self.emit_progress(0, 100, "Setting up computation")

        # Initial resource usage
        self.emit_resource_usage()

        # If progress callback is provided, wrap it with enhanced features
        if self.progress_callback:
            original_callback = self.progress_callback

            def wrapped_callback(*args, **kwargs):
                self.check_cancelled()

                # Call original callback
                try:
                    result = original_callback(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
                    result = None

                # Report progress with detailed information
                if len(args) >= 2:
                    current, total = args[0], args[1]
                    message = kwargs.get("message", "Computing...")

                    # Add performance info to message
                    if hasattr(self, "_start_time"):
                        elapsed = time.perf_counter() - self._start_time
                        rate = current / elapsed if elapsed > 0 else 0
                        message += f" (rate: {rate:.1f}/s)"

                    self.report_progress(current, total, message)

                return result

            self.compute_kwargs["progress_callback"] = wrapped_callback

        self.emit_status("Executing computation...", 1)
        self.emit_progress(10, 100, "Starting computation")

        # Execute the computation with timeout protection if needed
        try:
            result = self.compute_func(self.data, **self.compute_kwargs)
        except Exception:
            # Enhanced error context
            error_context = {
                "function": getattr(self.compute_func, "__name__", "unknown"),
                "data_type": type(self.data).__name__,
                "data_size": len(self.data)
                if hasattr(self.data, "__len__")
                else "unknown",
                "kwargs_keys": list(self.compute_kwargs.keys()),
            }
            logger.error(f"Computation failed with context: {error_context}")
            raise

        # Final resource usage and completion
        self.emit_resource_usage()
        self.emit_progress(100, 100, "Computation completed")
        self.emit_status("Computation completed successfully", 1)

        return result


class EnhancedThreadPool(QtCore.QThreadPool):
    """Enhanced thread pool with smart resource management and priority queuing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._priority_queue = defaultdict(deque)
        self._active_priorities = set()
        self._queue_mutex = threading.Lock()

        # Smart thread count based on system resources
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Adjust thread count based on available resources
        if memory_gb < 4:
            max_threads = max(2, cpu_count // 2)
        elif memory_gb < 8:
            max_threads = max(4, cpu_count)
        else:
            max_threads = min(16, cpu_count + 2)  # Cap at reasonable limit

        self.setMaxThreadCount(max_threads)
        logger.info(
            f"Enhanced thread pool initialized with {max_threads} threads (CPU: {cpu_count}, RAM: {memory_gb:.1f}GB)"
        )

        # Start background thread for priority management
        self._priority_manager_active = True
        self._priority_thread = threading.Thread(
            target=self._priority_manager, daemon=True
        )
        self._priority_thread.start()

    def start_with_priority(self, worker: BaseAsyncWorker):
        """Start worker with priority-based queuing."""
        with self._queue_mutex:
            priority_value = worker.priority.value
            self._priority_queue[priority_value].append(worker)
            self._active_priorities.add(priority_value)

    def _priority_manager(self):
        """Background thread to manage priority-based execution."""
        while self._priority_manager_active:
            try:
                with self._queue_mutex:
                    # Find highest priority with waiting workers
                    highest_priority = (
                        max(self._active_priorities)
                        if self._active_priorities
                        else None
                    )

                    if highest_priority and self._priority_queue[highest_priority]:
                        if self.activeThreadCount() < self.maxThreadCount():
                            worker = self._priority_queue[highest_priority].popleft()
                            if not self._priority_queue[highest_priority]:
                                self._active_priorities.discard(highest_priority)

                            # Start the worker
                            super().start(worker)

                time.sleep(0.1)  # Small delay to prevent busy waiting
            except Exception as e:
                logger.error(f"Error in priority manager: {e}")
                time.sleep(1.0)

    def stop_priority_manager(self):
        """Stop the priority manager thread."""
        self._priority_manager_active = False
        if self._priority_thread.is_alive():
            self._priority_thread.join(timeout=1.0)


class WorkerManager(QObject):
    """
    Enhanced manager class with advanced features:
    - Priority-based worker execution
    - Resource usage monitoring and limits
    - Worker result caching
    - Comprehensive statistics and health monitoring
    - Automatic load balancing
    """

    # Enhanced signals for comprehensive monitoring
    all_workers_started = Signal()
    all_workers_finished = Signal()
    worker_progress = Signal(
        str, int, int, str, float
    )  # id, current, total, message, eta
    worker_stats_updated = Signal(object)  # WorkerStats object
    resource_limit_exceeded = Signal(str, float)  # resource_type, current_value
    priority_queue_changed = Signal(dict)  # priority -> count mapping

    def __init__(self, thread_pool: QtCore.QThreadPool | None = None):
        super().__init__()

        # Use enhanced thread pool if none provided
        if thread_pool is None:
            self.thread_pool = EnhancedThreadPool()
        else:
            self.thread_pool = thread_pool

        # Worker tracking
        self.active_workers: Dict[str, BaseAsyncWorker] = {}
        self.worker_results: Dict[str, WorkerResult] = {}
        self.worker_errors: Dict[str, tuple] = {}
        self.worker_stats = WorkerStats()

        # Register for optimized cleanup
        try:
            from .cleanup_optimized import register_for_cleanup

            register_for_cleanup(self, ["shutdown"])
        except ImportError:
            pass  # Optimized cleanup system not available

        # Resource monitoring
        self._max_memory_mb = 2048  # 2GB default limit
        self._max_cpu_percent = 80.0
        self._resource_check_timer = QTimer()
        self._resource_check_timer.timeout.connect(self._check_resource_limits)
        self._resource_check_timer.start(5000)  # Check every 5 seconds

        # Result caching
        self._result_cache: Dict[
            str, tuple
        ] = {}  # cache_key -> (result, timestamp, ttl)
        self._cache_cleanup_timer = QTimer()
        self._cache_cleanup_timer.timeout.connect(self._cleanup_cache)
        self._cache_cleanup_timer.start(60000)  # Cleanup every minute

        logger.info(
            f"Enhanced WorkerManager initialized with {self.thread_pool.maxThreadCount()} max threads"
        )

    def submit_worker(self, worker: BaseAsyncWorker, force_start: bool = False) -> str:
        """
        Submit a worker with enhanced tracking and resource checking.

        Args:
            worker: The worker to submit
            force_start: If True, bypass resource limits (use carefully)

        Returns:
            str: The worker ID for tracking

        Raises:
            RuntimeError: If resource limits would be exceeded
        """
        worker_id = worker.worker_id

        # Check resource limits before submitting (unless forced)
        if not force_start and not self._can_submit_worker():
            raise RuntimeError(
                f"Cannot submit worker {worker_id}: resource limits exceeded"
            )

        # Connect enhanced signals
        worker.signals.started.connect(
            lambda w_id, priority: self._on_worker_started(w_id, priority)
        )
        worker.signals.finished.connect(lambda result: self._on_worker_finished(result))
        worker.signals.error.connect(
            lambda w_id, msg, tb, retry: self._on_worker_error(w_id, msg, tb, retry)
        )
        worker.signals.progress.connect(
            lambda w_id, current, total, message, eta: self._on_worker_progress(
                w_id, current, total, message, eta
            )
        )
        worker.signals.cancelled.connect(
            lambda w_id, reason: self._on_worker_cancelled(w_id, reason)
        )
        worker.signals.state_changed.connect(
            lambda w_id, old, new: self._on_worker_state_changed(w_id, old, new)
        )
        worker.signals.resource_usage.connect(
            lambda w_id, cpu, mem: self._on_worker_resource_usage(w_id, cpu, mem)
        )
        worker.signals.retry_attempt.connect(
            lambda w_id, attempt, max_attempts: self._on_worker_retry(
                w_id, attempt, max_attempts
            )
        )

        # Track the worker
        self.active_workers[worker_id] = worker
        self.worker_stats.total_workers += 1
        self.worker_stats.active_workers += 1

        # Submit based on thread pool type
        if isinstance(self.thread_pool, EnhancedThreadPool):
            self.thread_pool.start_with_priority(worker)
        else:
            self.thread_pool.start(worker)

        logger.info(f"Submitted worker {worker_id} (priority: {worker.priority.name})")
        return worker_id

    def _can_submit_worker(self) -> bool:
        """Check if we can submit another worker based on resource limits."""
        try:
            # Check memory usage
            memory_usage = psutil.virtual_memory().percent
            if memory_usage > self._max_cpu_percent:
                return False

            # Check if we have available thread slots with some buffer
            active_threads = self.thread_pool.activeThreadCount()
            max_threads = self.thread_pool.maxThreadCount()
            if active_threads >= max_threads * 0.9:  # Leave 10% buffer
                return False

            return True
        except Exception as e:
            logger.warning(f"Error checking resource limits: {e}")
            return True  # Allow submission if check fails

    def cancel_worker(self, worker_id: str, reason: str = "User requested") -> bool:
        """Cancel a specific worker by ID with reason."""
        if worker_id in self.active_workers:
            self.active_workers[worker_id].cancel(reason)
            return True
        return False

    def cancel_all_workers(self, reason: str = "Shutdown requested"):
        """Cancel all active workers with reason."""
        cancelled_count = 0
        for worker in list(self.active_workers.values()):
            worker.cancel(reason)
            cancelled_count += 1

        logger.info(f"Cancelled {cancelled_count} active workers: {reason}")
        return cancelled_count

    def cancel_workers_by_priority(
        self, max_priority: WorkerPriority, reason: str = "Priority-based cancellation"
    ) -> int:
        """Cancel all workers with priority <= max_priority."""
        cancelled_count = 0
        for worker in list(self.active_workers.values()):
            if worker.priority.value <= max_priority.value:
                worker.cancel(reason)
                cancelled_count += 1

        logger.info(
            f"Cancelled {cancelled_count} workers with priority <= {max_priority.name}"
        )
        return cancelled_count

    def get_worker_result(self, worker_id: str) -> WorkerResult | None:
        """Get the result object of a completed worker."""
        return self.worker_results.get(worker_id)

    def get_worker_error(self, worker_id: str) -> tuple | None:
        """Get the error information for a failed worker."""
        return self.worker_errors.get(worker_id)

    def is_worker_active(self, worker_id: str) -> bool:
        """Check if a worker is still active."""
        return worker_id in self.active_workers

    def get_worker_stats(self) -> WorkerStats:
        """Get current worker statistics."""
        return self.worker_stats

    def get_active_worker_count(self) -> int:
        """Get count of currently active workers."""
        return len(self.active_workers)

    def get_priority_queue_status(self) -> Dict[str, int]:
        """Get status of priority queues if using enhanced thread pool."""
        if isinstance(self.thread_pool, EnhancedThreadPool):
            with self.thread_pool._queue_mutex:
                return {
                    str(priority): len(queue)
                    for priority, queue in self.thread_pool._priority_queue.items()
                }
        return {}

    def set_resource_limits(
        self, max_memory_mb: float = None, max_cpu_percent: float = None
    ):
        """Set resource limits for worker submission."""
        if max_memory_mb is not None:
            self._max_memory_mb = max_memory_mb
        if max_cpu_percent is not None:
            self._max_cpu_percent = max_cpu_percent

        logger.info(
            f"Resource limits updated: Memory={self._max_memory_mb}MB, CPU={self._max_cpu_percent}%"
        )

    # Signal handlers
    @Slot(str, int)
    def _on_worker_started(self, worker_id: str, priority: int):
        """Handle worker started signal with priority info."""
        logger.debug(f"Worker {worker_id} started (priority: {priority})")

    @Slot(object)
    def _on_worker_finished(self, worker_result: WorkerResult):
        """Handle worker finished signal with comprehensive result."""
        worker_id = worker_result.worker_id
        self.worker_results[worker_id] = worker_result

        if worker_id in self.active_workers:
            del self.active_workers[worker_id]

        # Update statistics
        self.worker_stats.active_workers = len(self.active_workers)
        if worker_result.success:
            self.worker_stats.completed_workers += 1
        else:
            self.worker_stats.failed_workers += 1

        # Update average execution time
        total_completed = (
            self.worker_stats.completed_workers + self.worker_stats.failed_workers
        )
        if total_completed > 0:
            self.worker_stats.avg_execution_time = (
                self.worker_stats.avg_execution_time * (total_completed - 1)
                + worker_result.execution_time
            ) / total_completed

        self.worker_stats.total_memory_usage += worker_result.memory_usage

        logger.debug(
            f"Worker {worker_id} finished (success: {worker_result.success}, "
            f"time: {worker_result.execution_time:.2f}s, memory: {worker_result.memory_usage:.1f}MB)"
        )

        # Emit statistics update
        self.worker_stats_updated.emit(self.worker_stats)

        if not self.active_workers:
            self.all_workers_finished.emit()

    @Slot(str, str, str, int)
    def _on_worker_error(
        self, worker_id: str, error_msg: str, traceback_str: str, retry_count: int
    ):
        """Handle worker error signal with retry information."""
        self.worker_errors[worker_id] = (error_msg, traceback_str, retry_count)

        if worker_id in self.active_workers:
            # Don't remove from active workers if retrying
            worker = self.active_workers[worker_id]
            if retry_count >= worker.max_retries:
                del self.active_workers[worker_id]
                self.worker_stats.active_workers = len(self.active_workers)
                self.worker_stats.failed_workers += 1

        logger.error(f"Worker {worker_id} error (attempt {retry_count}): {error_msg}")

    @Slot(str, int, int, str, float)
    def _on_worker_progress(
        self, worker_id: str, current: int, total: int, message: str, eta_seconds: float
    ):
        """Handle worker progress signal with ETA."""
        self.worker_progress.emit(worker_id, current, total, message, eta_seconds)

    @Slot(str, str)
    def _on_worker_cancelled(self, worker_id: str, reason: str):
        """Handle worker cancelled signal with reason."""
        if worker_id in self.active_workers:
            del self.active_workers[worker_id]
            self.worker_stats.active_workers = len(self.active_workers)
            self.worker_stats.cancelled_workers += 1
            self.worker_stats_updated.emit(self.worker_stats)

        logger.info(f"Worker {worker_id} was cancelled: {reason}")

    @Slot(str, str, str)
    def _on_worker_state_changed(self, worker_id: str, old_state: str, new_state: str):
        """Handle worker state change signal."""
        logger.debug(f"Worker {worker_id} state: {old_state} -> {new_state}")

    @Slot(str, float, float)
    def _on_worker_resource_usage(
        self, worker_id: str, cpu_percent: float, memory_mb: float
    ):
        """Handle worker resource usage signal."""
        # Check if resource limits are exceeded
        if memory_mb > self._max_memory_mb:
            self.resource_limit_exceeded.emit("memory", memory_mb)
            logger.warning(
                f"Worker {worker_id} exceeded memory limit: {memory_mb:.1f}MB > {self._max_memory_mb}MB"
            )

    @Slot(str, int, int)
    def _on_worker_retry(self, worker_id: str, attempt_number: int, max_attempts: int):
        """Handle worker retry attempt signal."""
        logger.info(f"Worker {worker_id} retry attempt {attempt_number}/{max_attempts}")

    def _check_resource_limits(self):
        """Periodically check system resource limits."""
        try:
            # Check system memory
            memory = psutil.virtual_memory()
            if memory.percent > 90:  # 90% memory usage
                self.resource_limit_exceeded.emit("system_memory", memory.percent)

            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 95:  # 95% CPU usage
                self.resource_limit_exceeded.emit("system_cpu", cpu_percent)

        except Exception as e:
            logger.debug(f"Error checking resource limits: {e}")

    def _cache_result(self, cache_key: str, result: Any, ttl: float):
        """Cache a result with TTL."""
        timestamp = time.time()
        self._result_cache[cache_key] = (result, timestamp, ttl)

    def _get_cached_result(self, cache_key: str) -> Tuple[bool, Any]:
        """Get cached result if valid. Returns (found, result)."""
        if cache_key not in self._result_cache:
            return False, None

        result, timestamp, ttl = self._result_cache[cache_key]
        if time.time() - timestamp > ttl:
            del self._result_cache[cache_key]
            return False, None

        return True, result

    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        expired_keys = []

        for key, (result, timestamp, ttl) in self._result_cache.items():
            if current_time - timestamp > ttl:
                expired_keys.append(key)

        for key in expired_keys:
            del self._result_cache[key]

        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def clear_cache(self):
        """Clear all cached results."""
        self._result_cache.clear()
        logger.info("Result cache cleared")

    def shutdown(self):
        """Graceful shutdown of the worker manager."""
        logger.info("Shutting down WorkerManager...")

        # Cancel all active workers
        self.cancel_all_workers("Manager shutdown")

        # Stop timers
        self._resource_check_timer.stop()
        self._cache_cleanup_timer.stop()

        # Stop enhanced thread pool priority manager if applicable
        if isinstance(self.thread_pool, EnhancedThreadPool):
            self.thread_pool.stop_priority_manager()

        # Wait for threads to finish
        self.thread_pool.waitForDone(5000)  # 5 second timeout

        # Clear caches
        self.clear_cache()

        logger.info(
            f"WorkerManager shutdown complete. Final stats: {self.worker_stats}"
        )
