"""
Optimized worker system that integrates all performance optimizations for XPCS Toolkit.

This module provides:
- Workers that use optimized signal emission
- Cached attribute access for hot paths
- Integration with enhanced thread pools
- Smart resource management
- Performance monitoring and analytics
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, Signal

from ..utils.logging_config import get_logger
from .enhanced_thread_pool import get_thread_pool_manager

# Import our optimization systems
from .signal_optimization import SignalPriority, get_signal_optimizer

logger = get_logger(__name__)


@dataclass
class WorkerPerformanceMetrics:
    """Performance metrics for individual workers."""

    worker_id: str
    total_signal_emissions: int = 0
    batched_signal_emissions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    execution_time: float = 0.0
    peak_memory_usage: float = 0.0
    start_time: float = field(default_factory=time.perf_counter)

    @property
    def signal_batching_efficiency(self) -> float:
        """Calculate signal batching efficiency (0-100%)."""
        total = self.total_signal_emissions + self.batched_signal_emissions
        if total == 0:
            return 0.0
        return (self.batched_signal_emissions / total) * 100

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate (0-100%)."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100


class OptimizedWorkerSignals(QObject):
    """
    Optimized worker signals that use the signal optimization system.
    """

    def __init__(self, worker_id: str, parent=None):
        super().__init__(parent)
        self.worker_id = worker_id
        self._signal_optimizer = get_signal_optimizer()

        # Define signals (these will be optimized automatically)
        self.started = Signal(str, int)
        self.finished = Signal(object)
        self.error = Signal(str, str, str, int)
        self.progress = Signal(str, int, int, str, float)
        self.status = Signal(str, str, int)
        self.cancelled = Signal(str, str)
        self.partial_result = Signal(str, object, bool)
        self.resource_usage = Signal(str, float, float)
        self.state_changed = Signal(str, str, str)
        self.retry_attempt = Signal(str, int, int)

    def emit_optimized(
        self, signal_name: str, *args, priority: SignalPriority = SignalPriority.NORMAL
    ):
        """Emit signal through optimization pipeline."""
        signal_obj = getattr(self, signal_name, None)
        if signal_obj:
            self._signal_optimizer.emit_optimized(
                signal_obj, signal_name, args, priority
            )


class OptimizedBaseWorker(QRunnable):
    """
    Base worker class with integrated performance optimizations.

    Features:
    - Optimized signal emission with batching
    - Cached attribute access for frequently used properties
    - Performance monitoring and metrics collection
    - Integration with enhanced thread pools
    """

    def __init__(self, worker_id: str | None = None, pool_id: str = "default"):
        super().__init__(self)
        self.worker_id = worker_id or f"worker_{id(self)}"
        self.pool_id = pool_id

        # Initialize optimized signals
        self.signals = OptimizedWorkerSignals(self.worker_id)

        # Get optimization systems
        self._signal_optimizer = get_signal_optimizer()
        self._thread_pool_manager = get_thread_pool_manager()

        # State management
        self._is_cancelled = False
        self._start_time = None
        self._end_time = None

        # Performance tracking
        self._metrics = WorkerPerformanceMetrics(worker_id=self.worker_id)

        # Cached attributes (frequently accessed properties)
        self._cached_attributes = {}
        self._cache_timestamps = {}
        self._cache_mutex = threading.Lock()

        # Frequently cached attributes with TTL (Time To Live)
        self._cacheable_attributes = {
            "execution_time": 0.1,  # Cache for 100ms
            "is_cancelled": 0.05,  # Cache for 50ms (frequently checked)
            "worker_state": 0.1,  # Cache for 100ms
            "resource_usage": 0.5,  # Cache for 500ms
        }

        self.setAutoDelete(True)

    def get_cached_attribute(
        self, attr_name: str, compute_func: Optional[Callable] = None
    ) -> Any:
        """
        Get cached attribute value with automatic cache management.

        Args:
            attr_name: Name of the attribute to cache
            compute_func: Optional function to compute the attribute value

        Returns:
            The cached or computed attribute value
        """
        current_time = time.perf_counter()

        with self._cache_mutex:
            # Check if we have a valid cached value
            if (
                attr_name in self._cached_attributes
                and attr_name in self._cache_timestamps
            ):
                cache_age = current_time - self._cache_timestamps[attr_name]
                cache_ttl = self._cacheable_attributes.get(attr_name, 1.0)

                if cache_age < cache_ttl:
                    self._metrics.cache_hits += 1
                    return self._cached_attributes[attr_name]

            # Cache miss - compute new value
            self._metrics.cache_misses += 1

            if compute_func:
                value = compute_func()
            else:
                value = getattr(self, f"_{attr_name}", None)

            # Cache the value
            self._cached_attributes[attr_name] = value
            self._cache_timestamps[attr_name] = current_time

            return value

    def invalidate_cache(self, attr_name: Optional[str] = None):
        """Invalidate cached attributes."""
        with self._cache_mutex:
            if attr_name:
                self._cached_attributes.pop(attr_name, None)
                self._cache_timestamps.pop(attr_name, None)
            else:
                self._cached_attributes.clear()
                self._cache_timestamps.clear()

    @property
    def is_cancelled(self) -> bool:
        """Cached property for cancellation status."""
        return self.get_cached_attribute("is_cancelled", lambda: self._is_cancelled)

    @property
    def execution_time(self) -> float:
        """Cached property for execution time."""

        def compute_execution_time():
            if self._start_time is None:
                return 0.0
            end_time = self._end_time or time.perf_counter()
            return end_time - self._start_time

        return self.get_cached_attribute("execution_time", compute_execution_time)

    def cancel(self, reason: str = "User requested"):
        """Cancel the worker operation."""
        self._is_cancelled = True
        self.invalidate_cache("is_cancelled")

        # Emit cancellation with high priority
        self.signals.emit_optimized(
            "cancelled", self.worker_id, reason, SignalPriority.HIGH
        )
        logger.info(f"Worker {self.worker_id} cancelled: {reason}")

    def emit_progress(
        self, current: int, total: int, message: str = "", eta_seconds: float = 0.0
    ):
        """Emit progress update through optimized pipeline."""
        if self.is_cancelled:
            return

        # Progress signals are batched by default
        self.signals.emit_optimized(
            "progress",
            self.worker_id,
            current,
            total,
            message,
            eta_seconds,
            SignalPriority.NORMAL,
        )
        self._metrics.total_signal_emissions += 1

    def emit_status(self, status: str, detail_level: int = 1):
        """Emit status update through optimized pipeline."""
        if not self.is_cancelled:
            # Status signals can be batched
            self.signals.emit_optimized(
                "status", self.worker_id, status, detail_level, SignalPriority.NORMAL
            )
            self._metrics.total_signal_emissions += 1

    def emit_partial_result(self, result: Any, is_final: bool = False):
        """Emit partial result through optimized pipeline."""
        if not self.is_cancelled:
            # Partial results should not be batched heavily
            priority = SignalPriority.HIGH if is_final else SignalPriority.NORMAL
            self.signals.emit_optimized(
                "partial_result", self.worker_id, result, is_final, priority
            )
            self._metrics.total_signal_emissions += 1

    def emit_resource_usage(self, cpu_percent: float, memory_mb: float):
        """Emit resource usage through optimized pipeline."""
        if not self.is_cancelled:
            # Resource usage signals are perfect for batching
            self.signals.emit_optimized(
                "resource_usage",
                self.worker_id,
                cpu_percent,
                memory_mb,
                SignalPriority.LOW,
            )
            self._metrics.total_signal_emissions += 1

    def check_cancelled(self):
        """Check if cancelled and raise exception if so."""
        if self.is_cancelled:
            raise InterruptedError("Worker operation was cancelled")

    def do_work(self) -> Any:
        """
        Method to be implemented by subclasses.
        This method should contain the actual work to be done.
        """
        raise NotImplementedError("Subclasses must implement do_work() method")

    def run(self):
        """
        Main entry point with integrated optimizations.
        """
        try:
            self._start_time = time.perf_counter()
            self._metrics.start_time = self._start_time

            # Emit started signal with high priority (don't batch)
            self.signals.emit_optimized(
                "started", self.worker_id, 2, SignalPriority.HIGH
            )

            logger.debug(f"Optimized worker {self.worker_id} started")

            # Execute work
            result = self.do_work()

            # Check cancellation
            self.check_cancelled()

            # Handle completion
            self._handle_success(result)

        except InterruptedError:
            self._handle_cancellation()
        except Exception as e:
            self._handle_error(e)
        finally:
            self._end_time = time.perf_counter()
            self._finalize_metrics()

    def _handle_success(self, result: Any):
        """Handle successful completion."""
        # Emit finished signal with high priority
        self.signals.emit_optimized("finished", result, SignalPriority.HIGH)

        elapsed = self.execution_time
        logger.info(
            f"Optimized worker {self.worker_id} completed successfully in {elapsed:.2f}s"
        )

    def _handle_cancellation(self):
        """Handle cancellation."""
        logger.info(f"Optimized worker {self.worker_id} was cancelled")
        # Cancellation signal already emitted in cancel() method

    def _handle_error(self, exception: Exception):
        """Handle execution error."""
        import traceback

        error_msg = f"Error in worker {self.worker_id}: {str(exception)}"
        tb_str = traceback.format_exc()

        # Emit error with critical priority (never batch)
        self.signals.emit_optimized(
            "error", self.worker_id, error_msg, tb_str, 0, SignalPriority.CRITICAL
        )

        logger.error(f"{error_msg}\n{tb_str}")

    def _finalize_metrics(self):
        """Finalize performance metrics."""
        self._metrics.execution_time = self.execution_time

        # Log performance summary if debug enabled
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Worker {self.worker_id} performance summary:")
            logger.debug(f"  Execution time: {self._metrics.execution_time:.2f}s")
            logger.debug(
                f"  Signal batching efficiency: {self._metrics.signal_batching_efficiency:.1f}%"
            )
            logger.debug(f"  Cache hit rate: {self._metrics.cache_hit_rate:.1f}%")

    def get_performance_metrics(self) -> WorkerPerformanceMetrics:
        """Get worker performance metrics."""
        # Update final metrics
        self._metrics.execution_time = self.execution_time
        return self._metrics


class OptimizedPlotWorker(OptimizedBaseWorker):
    """
    Optimized plot worker with smart caching and batched progress updates.
    """

    def __init__(
        self,
        plot_func: Callable,
        plot_args: tuple = (),
        plot_kwargs: dict = None,
        worker_id: str | None = None,
        cache_results: bool = True,
        pool_id: str = "plotting",
    ):
        super().__init__(worker_id, pool_id)
        self.plot_func = plot_func
        self.plot_args = plot_args
        self.plot_kwargs = plot_kwargs or {}
        self.cache_results = cache_results

    def do_work(self) -> Any:
        """Execute plot function with optimized progress reporting."""
        self.emit_status("Initializing plot generation", 2)
        self.emit_progress(0, 100, "Starting plot", 0.0)

        # Check for cancellation early
        self.check_cancelled()

        self.emit_progress(20, 100, "Processing data", 1.0)
        self.emit_resource_usage(15.0, 50.0)  # Example values

        self.check_cancelled()

        self.emit_progress(60, 100, "Rendering plot", 0.5)

        # Execute the actual plotting function
        result = self.plot_func(*self.plot_args, **self.plot_kwargs)

        self.emit_progress(90, 100, "Finalizing", 0.1)
        self.emit_resource_usage(5.0, 45.0)  # Lower resource usage

        self.emit_progress(100, 100, "Complete", 0.0)
        self.emit_status("Plot generation completed", 1)

        return result


class OptimizedDataLoadWorker(OptimizedBaseWorker):
    """
    Optimized data loading worker with smart progress batching.
    """

    def __init__(
        self,
        load_func: Callable,
        file_paths: list,
        load_kwargs: dict = None,
        worker_id: str | None = None,
        batch_size: int = 10,
        pool_id: str = "data_loading",
    ):
        super().__init__(worker_id, pool_id)
        self.load_func = load_func
        self.file_paths = file_paths
        self.load_kwargs = load_kwargs or {}
        self.batch_size = batch_size

    def do_work(self) -> dict:
        """Load data with optimized progress reporting."""
        total_files = len(self.file_paths)
        loaded_data = []
        failed_files = []

        self.emit_status(f"Loading {total_files} files in batches", 1)

        for i, file_path in enumerate(self.file_paths):
            self.check_cancelled()

            # Only emit progress every few files to reduce signal overhead
            if i % 5 == 0 or i == total_files - 1:
                self.emit_progress(
                    i, total_files, f"Loading file {i + 1}/{total_files}"
                )

            try:
                data = self.load_func(file_path, **self.load_kwargs)
                loaded_data.append(data)

                # Emit partial results every batch
                if i % self.batch_size == 0:
                    self.emit_partial_result(
                        {
                            "batch_complete": i // self.batch_size,
                            "files_loaded": i + 1,
                            "total_files": total_files,
                        }
                    )

            except Exception as e:
                failed_files.append((file_path, str(e)))
                loaded_data.append(None)
                logger.warning(f"Failed to load {file_path}: {e}")

        # Final progress and resource update
        self.emit_progress(total_files, total_files, "Loading complete")
        self.emit_resource_usage(10.0, 100.0)  # Example final resource usage

        return {
            "loaded_data": loaded_data,
            "failed_files": failed_files,
            "success_count": len(loaded_data) - len(failed_files),
            "total_count": total_files,
        }


class OptimizedComputationWorker(OptimizedBaseWorker):
    """
    Optimized computation worker with intelligent progress management.
    """

    def __init__(
        self,
        compute_func: Callable,
        data: Any,
        progress_callback: Callable | None = None,
        compute_kwargs: dict = None,
        worker_id: str | None = None,
        pool_id: str = "computation",
    ):
        super().__init__(worker_id, pool_id)
        self.compute_func = compute_func
        self.data = data
        self.progress_callback = progress_callback
        self.compute_kwargs = compute_kwargs or {}

    def do_work(self) -> Any:
        """Execute computation with optimized monitoring."""
        self.emit_status("Initializing computation", 2)
        self.emit_progress(0, 100, "Setting up")

        # Wrap progress callback to use our optimized system
        if self.progress_callback:
            original_callback = self.progress_callback

            def optimized_callback(*args, **kwargs):
                self.check_cancelled()

                # Call original callback
                result = original_callback(*args, **kwargs)

                # Report through our optimized system
                if len(args) >= 2:
                    current, total = args[0], args[1]
                    message = kwargs.get("message", "Computing...")
                    self.emit_progress(current, total, message)

                return result

            self.compute_kwargs["progress_callback"] = optimized_callback

        self.emit_progress(10, 100, "Starting computation")
        self.emit_resource_usage(20.0, 75.0)

        # Execute computation
        result = self.compute_func(self.data, **self.compute_kwargs)

        self.emit_progress(100, 100, "Computation complete")
        self.emit_resource_usage(5.0, 60.0)
        self.emit_status("Computation completed successfully", 1)

        return result


def create_optimized_worker(worker_type: str, *args, **kwargs) -> OptimizedBaseWorker:
    """
    Factory function to create optimized workers.

    Args:
        worker_type: Type of worker ('plot', 'data_load', 'computation')
        *args: Arguments for the worker constructor
        **kwargs: Keyword arguments for the worker constructor

    Returns:
        An optimized worker instance
    """
    worker_classes = {
        "plot": OptimizedPlotWorker,
        "data_load": OptimizedDataLoadWorker,
        "computation": OptimizedComputationWorker,
    }

    worker_class = worker_classes.get(worker_type)
    if not worker_class:
        raise ValueError(f"Unknown worker type: {worker_type}")

    return worker_class(*args, **kwargs)


def submit_optimized_worker(worker: OptimizedBaseWorker, priority: int = 5) -> bool:
    """
    Submit an optimized worker to the appropriate thread pool.

    Args:
        worker: The optimized worker to submit
        priority: Task priority (lower numbers = higher priority)

    Returns:
        True if successfully submitted, False otherwise
    """
    thread_pool_manager = get_thread_pool_manager()
    return thread_pool_manager.submit_task(
        worker,
        pool_id=worker.pool_id,
        priority=priority,
        estimated_duration=1.0,  # Default estimate
    )
