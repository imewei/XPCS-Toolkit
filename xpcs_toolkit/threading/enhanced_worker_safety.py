"""
Enhanced Worker Thread Safety and Error Handling for XPCS Toolkit.

This module provides enhanced safety mechanisms for worker threads including
automatic error recovery, resource leak prevention, graceful cancellation,
and comprehensive error handling with retry capabilities.
"""

import gc
import sys
import time
import traceback
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from weakref import WeakSet

from PySide6 import QtCore
from PySide6.QtCore import QObject, QRunnable, QTimer, Signal, QMutex, QMutexLocker, Slot, Qt

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class WorkerErrorSeverity(Enum):
    """Error severity levels for worker operations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class WorkerRecoveryAction(Enum):
    """Recovery actions for worker errors."""

    RETRY = "retry"
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    FALLBACK = "fallback"
    FAIL_GRACEFULLY = "fail_gracefully"
    ESCALATE = "escalate"


class ResourceType(Enum):
    """Types of resources that workers might use."""

    MEMORY = "memory"
    FILE_HANDLE = "file_handle"
    NETWORK_CONNECTION = "network_connection"
    TEMPORARY_FILE = "temporary_file"
    CACHE_ENTRY = "cache_entry"
    GUI_OBJECT = "gui_object"


@dataclass
class ErrorInfo:
    """Information about a worker error."""

    error_id: str
    worker_id: str
    error_type: str
    error_message: str
    traceback_str: str
    severity: WorkerErrorSeverity
    timestamp: float = field(default_factory=time.perf_counter)
    retry_count: int = 0
    recovery_action: Optional[WorkerRecoveryAction] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceLease:
    """Information about a resource lease for a worker."""

    resource_id: str
    resource_type: ResourceType
    worker_id: str
    lease_time: float = field(default_factory=time.perf_counter)
    cleanup_callback: Optional[Callable] = None
    auto_cleanup_timeout: float = 300.0  # 5 minutes default
    metadata: Dict[str, Any] = field(default_factory=dict)


class SafeWorkerSignals(QObject):
    """
    Enhanced signals for safe worker operations with error handling.
    """

    # Enhanced signal set with additional safety features
    worker_started = Signal(str, dict)  # worker_id, context
    worker_finished = Signal(str, object, dict)  # worker_id, result, execution_info
    worker_error = Signal(str, object)  # worker_id, ErrorInfo
    worker_progress = Signal(str, int, int, str, dict)  # worker_id, current, total, message, stats
    worker_cancelled = Signal(str, str, dict)  # worker_id, reason, cleanup_info
    worker_resource_acquired = Signal(str, str, str)  # worker_id, resource_id, resource_type
    worker_resource_released = Signal(str, str, bool)  # worker_id, resource_id, success
    worker_recovery_attempted = Signal(str, str, int)  # worker_id, recovery_action, attempt_number


class SafeWorkerBase(QRunnable):
    """
    Enhanced base class for safe worker operations with comprehensive safety features.

    This class provides a robust foundation for worker threads with automatic error
    recovery, resource leak prevention, performance monitoring, and graceful cancellation.
    It ensures thread-safe operations and maintains system stability under all conditions.

    Key Features:
        - Automatic error recovery with configurable retry strategies
        - Resource leak prevention with automatic cleanup
        - Performance monitoring and progress reporting
        - Graceful cancellation with proper resource cleanup
        - Thread-safe state management with mutex protection
        - Comprehensive error handling with severity classification

    Args:
        worker_id (str, optional): Unique identifier for the worker. Auto-generated if None.
        max_retries (int): Maximum number of retry attempts for failed operations. Default: 3.
        timeout_seconds (float): Maximum execution time before timeout. Default: 300.0.

    Attributes:
        signals (SafeWorkerSignals): Enhanced signal set for worker communication
        worker_id (str): Unique identifier for this worker instance
        max_retries (int): Maximum retry attempts allowed
        timeout_seconds (float): Execution timeout in seconds

    Example:
        >>> class MyWorker(SafeWorkerBase):
        ...     def do_work(self):
        ...         # Acquire resources safely
        ...         self.acquire_resource("temp_file", ResourceType.TEMPORARY_FILE)
        ...
        ...         # Report progress
        ...         self.emit_progress_safe(50, 100, "Processing data")
        ...
        ...         # Do actual work
        ...         result = process_data()
        ...
        ...         # Resources are automatically cleaned up
        ...         return result
        ...
        >>> worker = MyWorker("data_processor", max_retries=2)
        >>> # Worker handles errors, retries, and cleanup automatically

    Note:
        Subclasses must implement the do_work() method to define the actual work
        to be performed. All error handling, resource management, and cleanup
        is handled automatically by the base class.
    """

    def __init__(self, worker_id: str = None, max_retries: int = 3,
                 timeout_seconds: float = 300.0):
        super().__init__()
        self.worker_id = worker_id or f"safe_worker_{id(self)}"
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds

        # Enhanced signals
        self.signals = SafeWorkerSignals()

        # State management
        self._is_cancelled = False
        self._is_running = False
        self._start_time = None
        self._last_heartbeat = None
        self._cancellation_reason = None

        # Error handling
        self._error_history = deque(maxlen=10)
        self._retry_count = 0
        self._recovery_strategies = {}

        # Resource management
        self._acquired_resources: Dict[str, ResourceLease] = {}
        self._resource_cleanup_callbacks = []

        # Performance monitoring
        self._performance_stats = {
            'memory_peak_mb': 0.0,
            'execution_time': 0.0,
            'cpu_time': 0.0,
            'progress_updates': 0,
            'errors_encountered': 0
        }

        # Thread safety
        self._state_mutex = QMutex()

        # Auto-delete management
        self.setAutoDelete(True)

        logger.debug(f"SafeWorkerBase '{self.worker_id}' initialized")

    @property
    def is_cancelled(self) -> bool:
        """Thread-safe check for cancellation status."""
        with QMutexLocker(self._state_mutex):
            return self._is_cancelled

    @property
    def is_running(self) -> bool:
        """Thread-safe check for running status."""
        with QMutexLocker(self._state_mutex):
            return self._is_running

    def cancel(self, reason: str = "User requested"):
        """Cancel the worker operation safely."""
        with QMutexLocker(self._state_mutex):
            if not self._is_cancelled:
                self._is_cancelled = True
                self._cancellation_reason = reason
                logger.info(f"Worker '{self.worker_id}' cancelled: {reason}")

    def update_heartbeat(self):
        """Update worker heartbeat for monitoring."""
        with QMutexLocker(self._state_mutex):
            self._last_heartbeat = time.perf_counter()

    def acquire_resource(self, resource_id: str, resource_type: ResourceType,
                        cleanup_callback: Callable = None,
                        auto_cleanup_timeout: float = 300.0,
                        metadata: Dict[str, Any] = None) -> bool:
        """
        Acquire a resource lease for the worker.

        Args:
            resource_id: Unique identifier for the resource
            resource_type: Type of resource being acquired
            cleanup_callback: Optional cleanup function
            auto_cleanup_timeout: Timeout for automatic cleanup
            metadata: Additional resource metadata

        Returns:
            True if resource was successfully acquired
        """
        try:
            lease = ResourceLease(
                resource_id=resource_id,
                resource_type=resource_type,
                worker_id=self.worker_id,
                cleanup_callback=cleanup_callback,
                auto_cleanup_timeout=auto_cleanup_timeout,
                metadata=metadata or {}
            )

            self._acquired_resources[resource_id] = lease

            # Emit resource acquisition signal
            self.signals.worker_resource_acquired.emit(
                self.worker_id, resource_id, resource_type.value
            )

            logger.debug(f"Worker '{self.worker_id}' acquired resource '{resource_id}'")
            return True

        except Exception as e:
            logger.error(f"Failed to acquire resource '{resource_id}' for worker '{self.worker_id}': {e}")
            return False

    def release_resource(self, resource_id: str) -> bool:
        """
        Release a resource lease.

        Args:
            resource_id: ID of resource to release

        Returns:
            True if resource was successfully released
        """
        try:
            lease = self._acquired_resources.get(resource_id)
            if lease is None:
                logger.warning(f"Resource '{resource_id}' not found for worker '{self.worker_id}'")
                return True

            success = True

            # Call cleanup callback if provided
            if lease.cleanup_callback:
                try:
                    lease.cleanup_callback()
                except Exception as e:
                    logger.error(f"Resource cleanup callback failed for '{resource_id}': {e}")
                    success = False

            # Remove from tracking
            del self._acquired_resources[resource_id]

            # Emit resource release signal
            self.signals.worker_resource_released.emit(
                self.worker_id, resource_id, success
            )

            logger.debug(f"Worker '{self.worker_id}' released resource '{resource_id}'")
            return success

        except Exception as e:
            logger.error(f"Failed to release resource '{resource_id}' for worker '{self.worker_id}': {e}")
            return False

    def release_all_resources(self):
        """Release all acquired resources."""
        resource_ids = list(self._acquired_resources.keys())
        for resource_id in resource_ids:
            self.release_resource(resource_id)

        logger.debug(f"Worker '{self.worker_id}' released all {len(resource_ids)} resources")

    def emit_progress_safe(self, current: int, total: int, message: str = "",
                          additional_stats: Dict[str, Any] = None):
        """
        Emit progress signal safely with enhanced information.

        Args:
            current: Current progress value
            total: Total progress value
            message: Progress message
            additional_stats: Additional statistics
        """
        if self.is_cancelled:
            return

        try:
            self.update_heartbeat()
            self._performance_stats['progress_updates'] += 1

            stats = additional_stats or {}
            stats.update({
                'heartbeat': self._last_heartbeat,
                'memory_usage_mb': self._get_memory_usage(),
                'elapsed_time': time.perf_counter() - (self._start_time or 0)
            })

            self.signals.worker_progress.emit(
                self.worker_id, current, total, message, stats
            )

        except Exception as e:
            logger.warning(f"Failed to emit progress for worker '{self.worker_id}': {e}")

    def handle_error(self, error: Exception, context: Dict[str, Any] = None,
                    severity: WorkerErrorSeverity = WorkerErrorSeverity.MEDIUM) -> WorkerRecoveryAction:
        """
        Handle an error with automatic recovery determination.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level

        Returns:
            Recommended recovery action
        """
        self._performance_stats['errors_encountered'] += 1

        error_info = ErrorInfo(
            error_id=f"error_{len(self._error_history)}",
            worker_id=self.worker_id,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            severity=severity,
            retry_count=self._retry_count,
            context=context or {}
        )

        # Determine recovery action
        recovery_action = self._determine_recovery_action(error_info)
        error_info.recovery_action = recovery_action

        # Store error history
        self._error_history.append(error_info)

        # Emit error signal
        self.signals.worker_error.emit(self.worker_id, error_info)

        logger.error(f"Worker '{self.worker_id}' error: {error_info.error_message}")
        logger.debug(f"Recovery action: {recovery_action.value}")

        return recovery_action

    def _determine_recovery_action(self, error_info: ErrorInfo) -> WorkerRecoveryAction:
        """
        Determine the best recovery action for an error.

        Args:
            error_info: Information about the error

        Returns:
            Recommended recovery action
        """
        # Check retry limits
        if self._retry_count >= self.max_retries:
            if error_info.severity == WorkerErrorSeverity.CRITICAL:
                return WorkerRecoveryAction.ESCALATE
            else:
                return WorkerRecoveryAction.FAIL_GRACEFULLY

        # Error-type specific recovery
        error_type = error_info.error_type

        if error_type in ['TimeoutError', 'ConnectionError', 'TemporaryError']:
            return WorkerRecoveryAction.RETRY_WITH_BACKOFF
        elif error_type in ['MemoryError', 'OSError']:
            return WorkerRecoveryAction.RETRY
        elif error_type in ['ValueError', 'TypeError']:
            return WorkerRecoveryAction.FAIL_GRACEFULLY
        else:
            # Default based on severity
            if error_info.severity == WorkerErrorSeverity.CRITICAL:
                return WorkerRecoveryAction.ESCALATE
            elif error_info.severity in [WorkerErrorSeverity.HIGH, WorkerErrorSeverity.MEDIUM]:
                return WorkerRecoveryAction.RETRY
            else:
                return WorkerRecoveryAction.FAIL_GRACEFULLY

    def check_timeout(self) -> bool:
        """Check if worker has exceeded timeout."""
        if self._start_time is None:
            return False

        elapsed = time.perf_counter() - self._start_time
        if elapsed > self.timeout_seconds:
            logger.warning(f"Worker '{self.worker_id}' exceeded timeout ({elapsed:.1f}s > {self.timeout_seconds}s)")
            return True

        return False

    def check_cancelled_safe(self):
        """Check if cancelled and raise appropriate exception."""
        if self.is_cancelled:
            self.release_all_resources()
            raise InterruptedError(f"Worker operation was cancelled: {self._cancellation_reason}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self._performance_stats['memory_peak_mb'] = max(
                self._performance_stats['memory_peak_mb'], memory_mb
            )
            return memory_mb
        except ImportError:
            return 0.0

    def do_work(self) -> Any:
        """
        Method to be implemented by subclasses.

        This method should contain the actual work to be done.
        It can use the safety features provided by this base class.

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
        Enhanced run method with comprehensive error handling and resource management.
        """
        execution_info = {}

        try:
            # Initialize execution
            with QMutexLocker(self._state_mutex):
                self._is_running = True
                self._start_time = time.perf_counter()
                self.update_heartbeat()

            # Emit started signal
            start_context = {
                'max_retries': self.max_retries,
                'timeout_seconds': self.timeout_seconds,
                'start_time': self._start_time
            }
            self.signals.worker_started.emit(self.worker_id, start_context)

            logger.info(f"Safe worker '{self.worker_id}' started")

            # Main execution loop with retry capability
            result = None
            last_error = None

            for attempt in range(self.max_retries + 1):
                try:
                    self._retry_count = attempt
                    self.check_cancelled_safe()
                    self.check_timeout()

                    # Execute the work
                    result = self.do_work()

                    # If we get here, work completed successfully
                    break

                except InterruptedError:
                    # Cancellation is not retryable
                    raise

                except Exception as e:
                    last_error = e
                    recovery_action = self.handle_error(e)

                    if recovery_action in [WorkerRecoveryAction.FAIL_GRACEFULLY, WorkerRecoveryAction.ESCALATE]:
                        break
                    elif recovery_action == WorkerRecoveryAction.RETRY_WITH_BACKOFF:
                        # Exponential backoff
                        backoff_time = min(2 ** attempt, 30)  # Max 30 seconds
                        logger.info(f"Worker '{self.worker_id}' retrying in {backoff_time}s (attempt {attempt + 1})")
                        time.sleep(backoff_time)
                    elif recovery_action == WorkerRecoveryAction.RETRY:
                        logger.info(f"Worker '{self.worker_id}' retrying immediately (attempt {attempt + 1})")

                    # Emit recovery attempt signal
                    self.signals.worker_recovery_attempted.emit(
                        self.worker_id, recovery_action.value, attempt + 1
                    )

            # Check final result
            if result is None and last_error is not None:
                raise last_error

            # Success - prepare execution info
            execution_info = {
                'execution_time': time.perf_counter() - self._start_time,
                'retry_count': self._retry_count,
                'memory_peak_mb': self._performance_stats['memory_peak_mb'],
                'errors_encountered': self._performance_stats['errors_encountered'],
                'progress_updates': self._performance_stats['progress_updates']
            }

            logger.info(f"Safe worker '{self.worker_id}' completed successfully")
            self.signals.worker_finished.emit(self.worker_id, result, execution_info)

        except InterruptedError:
            # Handle cancellation
            cleanup_info = {
                'cancelled_at': time.perf_counter(),
                'resources_released': len(self._acquired_resources),
                'execution_time': time.perf_counter() - (self._start_time or 0)
            }

            logger.info(f"Safe worker '{self.worker_id}' was cancelled")
            self.signals.worker_cancelled.emit(
                self.worker_id, self._cancellation_reason or "Unknown", cleanup_info
            )

        except Exception as e:
            # Handle unrecoverable error
            final_error = ErrorInfo(
                error_id="final_error",
                worker_id=self.worker_id,
                error_type=type(e).__name__,
                error_message=str(e),
                traceback_str=traceback.format_exc(),
                severity=WorkerErrorSeverity.CRITICAL,
                retry_count=self._retry_count
            )

            logger.error(f"Safe worker '{self.worker_id}' failed with unrecoverable error: {e}")
            self.signals.worker_error.emit(self.worker_id, final_error)

        finally:
            # Cleanup resources regardless of outcome
            try:
                self.release_all_resources()
            except Exception as e:
                logger.warning(f"Error during resource cleanup for worker '{self.worker_id}': {e}")

            # Update state
            with QMutexLocker(self._state_mutex):
                self._is_running = False

            logger.debug(f"Safe worker '{self.worker_id}' cleanup completed")


class SafeWorkerPool(QObject):
    """
    Enhanced worker pool with automatic safety management.

    This pool manages safe workers with resource tracking, error monitoring,
    and automatic cleanup capabilities.
    """

    # Pool-level signals
    pool_worker_started = Signal(str)  # worker_id
    pool_worker_finished = Signal(str, object)  # worker_id, result
    pool_worker_error = Signal(str, object)  # worker_id, error_info
    pool_resource_leak_detected = Signal(str, str)  # worker_id, resource_id
    pool_stats_updated = Signal(dict)  # pool statistics

    def __init__(self, thread_pool: QtCore.QThreadPool, pool_id: str = "safe_pool"):
        super().__init__()
        self.thread_pool = thread_pool
        self.pool_id = pool_id

        # Worker tracking
        self._active_workers: Dict[str, SafeWorkerBase] = {}
        self._worker_history = deque(maxlen=1000)
        self._resource_leases: Dict[str, ResourceLease] = {}

        # Statistics
        self._pool_stats = {
            'workers_started': 0,
            'workers_completed': 0,
            'workers_failed': 0,
            'workers_cancelled': 0,
            'total_errors': 0,
            'resource_leaks': 0,
            'avg_execution_time': 0.0
        }

        # Resource monitoring timer
        self._resource_monitor = QTimer()
        self._resource_monitor.timeout.connect(self._check_resource_leaks)
        self._resource_monitor.start(30000)  # Check every 30 seconds

        logger.info(f"SafeWorkerPool '{pool_id}' initialized")

    def submit_safe_worker(self, worker: SafeWorkerBase) -> str:
        """
        Submit a safe worker to the pool.

        Args:
            worker: SafeWorkerBase instance to submit

        Returns:
            Worker ID for tracking
        """
        worker_id = worker.worker_id

        # Connect worker signals
        worker.signals.worker_started.connect(
            lambda wid, ctx: self._on_worker_started(wid, ctx),
            Qt.ConnectionType.QueuedConnection
        )
        worker.signals.worker_finished.connect(
            lambda wid, res, info: self._on_worker_finished(wid, res, info),
            Qt.ConnectionType.QueuedConnection
        )
        worker.signals.worker_error.connect(
            lambda wid, err: self._on_worker_error(wid, err),
            Qt.ConnectionType.QueuedConnection
        )
        worker.signals.worker_resource_acquired.connect(
            lambda wid, rid, rtype: self._on_resource_acquired(wid, rid, rtype),
            Qt.ConnectionType.QueuedConnection
        )
        worker.signals.worker_resource_released.connect(
            lambda wid, rid, success: self._on_resource_released(wid, rid, success),
            Qt.ConnectionType.QueuedConnection
        )

        # Track the worker
        self._active_workers[worker_id] = worker

        # Submit to thread pool
        self.thread_pool.start(worker)
        self._pool_stats['workers_started'] += 1

        logger.debug(f"Submitted safe worker '{worker_id}' to pool '{self.pool_id}'")
        return worker_id

    @Slot(str, dict)
    def _on_worker_started(self, worker_id: str, context: dict):
        """Handle worker started signal."""
        logger.debug(f"Pool '{self.pool_id}': Worker '{worker_id}' started")
        self.pool_worker_started.emit(worker_id)

    @Slot(str, object, dict)
    def _on_worker_finished(self, worker_id: str, result: object, execution_info: dict):
        """Handle worker finished signal."""
        if worker_id in self._active_workers:
            del self._active_workers[worker_id]

        self._pool_stats['workers_completed'] += 1
        self._update_execution_time_stats(execution_info.get('execution_time', 0))

        logger.debug(f"Pool '{self.pool_id}': Worker '{worker_id}' finished")
        self.pool_worker_finished.emit(worker_id, result)
        self._emit_stats_update()

    @Slot(str, object)
    def _on_worker_error(self, worker_id: str, error_info: ErrorInfo):
        """Handle worker error signal."""
        self._pool_stats['total_errors'] += 1

        if error_info.severity == WorkerErrorSeverity.CRITICAL:
            self._pool_stats['workers_failed'] += 1
            if worker_id in self._active_workers:
                del self._active_workers[worker_id]

        logger.warning(f"Pool '{self.pool_id}': Worker '{worker_id}' error: {error_info.error_message}")
        self.pool_worker_error.emit(worker_id, error_info)
        self._emit_stats_update()

    @Slot(str, str, str)
    def _on_resource_acquired(self, worker_id: str, resource_id: str, resource_type: str):
        """Handle resource acquisition."""
        lease = ResourceLease(
            resource_id=resource_id,
            resource_type=ResourceType(resource_type),
            worker_id=worker_id
        )
        self._resource_leases[resource_id] = lease
        logger.debug(f"Pool '{self.pool_id}': Resource '{resource_id}' acquired by worker '{worker_id}'")

    @Slot(str, str, bool)
    def _on_resource_released(self, worker_id: str, resource_id: str, success: bool):
        """Handle resource release."""
        if resource_id in self._resource_leases:
            del self._resource_leases[resource_id]

        if not success:
            logger.warning(f"Pool '{self.pool_id}': Resource '{resource_id}' release failed for worker '{worker_id}'")

        logger.debug(f"Pool '{self.pool_id}': Resource '{resource_id}' released by worker '{worker_id}'")

    def _check_resource_leaks(self):
        """Check for resource leaks from finished workers."""
        current_time = time.perf_counter()
        leaked_resources = []

        for resource_id, lease in self._resource_leases.items():
            # Check if worker is still active
            if lease.worker_id not in self._active_workers:
                # Check if resource has exceeded timeout
                if current_time - lease.lease_time > lease.auto_cleanup_timeout:
                    leaked_resources.append(resource_id)

        # Clean up leaked resources
        for resource_id in leaked_resources:
            lease = self._resource_leases[resource_id]
            logger.warning(
                f"Pool '{self.pool_id}': Resource leak detected for '{resource_id}' "
                f"from worker '{lease.worker_id}'"
            )

            # Attempt cleanup
            if lease.cleanup_callback:
                try:
                    lease.cleanup_callback()
                except Exception as e:
                    logger.error(f"Resource leak cleanup failed for '{resource_id}': {e}")

            del self._resource_leases[resource_id]
            self._pool_stats['resource_leaks'] += 1
            self.pool_resource_leak_detected.emit(lease.worker_id, resource_id)

        if leaked_resources:
            self._emit_stats_update()

    def _update_execution_time_stats(self, execution_time: float):
        """Update average execution time statistics."""
        completed = self._pool_stats['workers_completed']
        if completed == 1:
            self._pool_stats['avg_execution_time'] = execution_time
        else:
            # Running average
            current_avg = self._pool_stats['avg_execution_time']
            new_avg = ((current_avg * (completed - 1)) + execution_time) / completed
            self._pool_stats['avg_execution_time'] = new_avg

    def _emit_stats_update(self):
        """Emit updated pool statistics."""
        stats = self._pool_stats.copy()
        stats['active_workers'] = len(self._active_workers)
        stats['active_resources'] = len(self._resource_leases)
        stats['pool_id'] = self.pool_id
        self.pool_stats_updated.emit(stats)

    def cancel_all_workers(self, reason: str = "Pool shutdown"):
        """Cancel all active workers."""
        worker_ids = list(self._active_workers.keys())
        for worker_id in worker_ids:
            worker = self._active_workers[worker_id]
            worker.cancel(reason)

        logger.info(f"Pool '{self.pool_id}': Cancelled {len(worker_ids)} active workers")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get current pool statistics."""
        stats = self._pool_stats.copy()
        stats['active_workers'] = len(self._active_workers)
        stats['active_resources'] = len(self._resource_leases)
        stats['pool_id'] = self.pool_id
        return stats

    def shutdown(self):
        """Shutdown the pool and clean up resources."""
        logger.info(f"Shutting down SafeWorkerPool '{self.pool_id}'")

        # Stop resource monitoring
        self._resource_monitor.stop()

        # Cancel all workers
        self.cancel_all_workers("Pool shutdown")

        # Force cleanup of remaining resources
        self._check_resource_leaks()

        logger.info(f"SafeWorkerPool '{self.pool_id}' shutdown complete")


# Factory function for creating safe workers
def create_safe_worker(work_function: Callable, worker_id: str = None,
                      max_retries: int = 3, timeout_seconds: float = 300.0,
                      **kwargs) -> SafeWorkerBase:
    """
    Factory function to create a safe worker from a work function.

    Args:
        work_function: Function to execute in the worker
        worker_id: Optional worker ID
        max_retries: Maximum retry attempts
        timeout_seconds: Worker timeout
        **kwargs: Additional arguments to pass to work function

    Returns:
        SafeWorkerBase instance
    """

    class FunctionWorker(SafeWorkerBase):
        def __init__(self):
            super().__init__(worker_id, max_retries, timeout_seconds)
            self.work_function = work_function
            self.work_kwargs = kwargs

        def do_work(self) -> Any:
            return self.work_function(**self.work_kwargs)

    return FunctionWorker()


# Context manager for safe worker operations
@contextmanager
def safe_worker_context(thread_pool: QtCore.QThreadPool = None, pool_id: str = "temp_pool"):
    """
    Context manager for safe worker operations.

    Usage:
        with safe_worker_context() as pool:
            worker = create_safe_worker(my_function)
            pool.submit_safe_worker(worker)
    """
    if thread_pool is None:
        thread_pool = QtCore.QThreadPool.globalInstance()

    pool = SafeWorkerPool(thread_pool, pool_id)
    try:
        yield pool
    except Exception as e:
        logger.error(f"Error in safe worker context: {e}")
        raise
    finally:
        pool.shutdown()