"""
Qt-Compliant Cleanup System

This module provides a thread-safe cleanup system that properly uses Qt threading
patterns to avoid QTimer threading violations and other Qt-related errors.
"""

import gc
import threading
import time
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar, Optional

from PySide6.QtCore import QObject, QTimer, QThread, Signal, QMetaObject, Qt

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CleanupPriority(Enum):
    """Priority levels for cleanup operations."""

    LOW = 1  # Nice-to-have cleanup, can wait
    NORMAL = 2  # Standard cleanup operations
    HIGH = 3  # Important cleanup for performance
    CRITICAL = 4  # Must execute for stability


@dataclass
class CleanupTask:
    """Represents a cleanup task with metadata."""

    target_ref: weakref.ReferenceType
    cleanup_method: str
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: CleanupPriority = CleanupPriority.NORMAL
    max_retry_count: int = 2
    retry_count: int = 0
    created_at: float = field(default_factory=time.time)

    def is_expired(self, max_age: float = 300.0) -> bool:
        """Check if task has expired (default 5 minutes)."""
        return time.time() - self.created_at > max_age

    def get_target(self):
        """Get the target object if it still exists."""
        return self.target_ref() if self.target_ref is not None else None


class CleanupWorkerThread(QThread):
    """
    Qt-compliant worker thread for cleanup operations.

    This thread handles the actual cleanup work and uses proper Qt threading
    patterns to avoid timer threading violations.
    """

    # Signals for communication with main thread
    cleanup_completed = Signal(dict)  # Statistics
    cleanup_error = Signal(str, str)  # task_type, error_message
    cleanup_progress = Signal(int, int)  # completed, total

    def __init__(self, max_tasks_per_batch: int = 10):
        super().__init__()
        self.max_tasks_per_batch = max_tasks_per_batch
        self.shutdown_requested = False

        # Task queues by priority - accessed only from this thread
        self.task_queues: dict[CleanupPriority, deque] = {
            priority: deque() for priority in CleanupPriority
        }

        # Thread-safe queue for receiving new tasks
        self._incoming_tasks = deque()
        self._incoming_lock = threading.Lock()

        # Statistics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_cleanup_time = 0.0

        # Timer for periodic processing (created in this thread)
        self.cleanup_timer: Optional[QTimer] = None
        self.cleanup_interval_ms = 1000

    def run(self):
        """Main thread execution loop."""
        logger.debug("CleanupWorkerThread started")

        # Create QTimer in this thread (Qt-compliant)
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._process_cleanup_batch)
        self.cleanup_timer.setSingleShot(False)
        self.cleanup_timer.start(self.cleanup_interval_ms)

        # Start the Qt event loop for this thread
        self.exec()

        logger.debug("CleanupWorkerThread stopped")

    def stop_cleanup(self):
        """Stop the cleanup thread gracefully."""
        self.shutdown_requested = True

        # Stop timer if it exists
        if self.cleanup_timer and self.cleanup_timer.isActive():
            self.cleanup_timer.stop()

        # Quit the event loop
        self.quit()

    def add_tasks(self, tasks: list[CleanupTask]):
        """
        Add tasks to the cleanup queue in a thread-safe manner.

        Args:
            tasks: List of cleanup tasks to add
        """
        if not tasks or self.shutdown_requested:
            return

        with self._incoming_lock:
            self._incoming_tasks.extend(tasks)

    def _transfer_incoming_tasks(self):
        """Transfer tasks from thread-safe queue to internal queues."""
        if not self._incoming_tasks:
            return

        with self._incoming_lock:
            while self._incoming_tasks:
                task = self._incoming_tasks.popleft()
                self.task_queues[task.priority].append(task)

    def _process_cleanup_batch(self):
        """Process a batch of cleanup tasks."""
        if self.shutdown_requested:
            return

        # Transfer any new tasks
        self._transfer_incoming_tasks()

        # Get tasks to process (highest priority first)
        tasks_to_process = []
        total_tasks = 0

        for priority in sorted(CleanupPriority, key=lambda p: p.value, reverse=True):
            queue = self.task_queues[priority]
            total_tasks += len(queue)

            while queue and len(tasks_to_process) < self.max_tasks_per_batch:
                task = queue.popleft()

                # Skip expired tasks
                if task.is_expired():
                    continue

                # Skip tasks with dead references
                if task.get_target() is None:
                    continue

                tasks_to_process.append(task)

        if not tasks_to_process:
            return

        start_time = time.time()
        completed = 0
        failed = 0

        # Process tasks
        for task in tasks_to_process:
            try:
                target = task.get_target()
                if target is None:
                    continue

                if hasattr(target, task.cleanup_method):
                    method = getattr(target, task.cleanup_method)
                    method(*task.args, **task.kwargs)
                    completed += 1
                else:
                    logger.warning(f"Cleanup method {task.cleanup_method} not found on {type(target)}")
                    failed += 1

            except Exception as e:
                failed += 1
                error_msg = f"Cleanup failed for {task.cleanup_method}: {e}"
                logger.warning(error_msg)
                self.cleanup_error.emit(task.cleanup_method, str(e))

                # Retry if possible
                if task.retry_count < task.max_retry_count:
                    task.retry_count += 1
                    self.task_queues[task.priority].append(task)

        # Update statistics
        self.tasks_completed += completed
        self.tasks_failed += failed
        cleanup_time = time.time() - start_time
        self.total_cleanup_time += cleanup_time

        # Emit progress signal
        self.cleanup_progress.emit(completed, len(tasks_to_process))

        # Emit completion statistics
        if completed > 0 or failed > 0:
            stats = {
                'completed': completed,
                'failed': failed,
                'cleanup_time': cleanup_time,
                'total_completed': self.tasks_completed,
                'total_failed': self.tasks_failed,
                'total_cleanup_time': self.total_cleanup_time
            }
            self.cleanup_completed.emit(stats)

        logger.debug(f"Processed cleanup batch: {completed} completed, {failed} failed, {cleanup_time:.3f}s")


class QtCompliantCleanupManager(QObject):
    """
    Qt-compliant cleanup manager that uses proper threading patterns.

    This manager ensures all Qt timers are created and used in appropriate thread
    contexts to avoid threading violations while maintaining the same functionality
    as the original BackgroundCleanupManager.
    """

    # Signals for monitoring cleanup progress
    cleanup_started = Signal()
    cleanup_completed = Signal(dict)  # Statistics
    cleanup_progress = Signal(int, int)  # completed, total
    cleanup_error = Signal(str, str)  # task_type, error_message

    def __init__(self, max_workers: int = 2, max_tasks_per_batch: int = 10):
        super().__init__()

        self.max_workers = max_workers
        self.max_tasks_per_batch = max_tasks_per_batch
        self.shutdown_requested = False

        # Worker thread for cleanup operations
        self.cleanup_worker: Optional[CleanupWorkerThread] = None

        # Thread pool for CPU-intensive cleanup tasks
        self.executor: Optional[ThreadPoolExecutor] = None

        # Statistics
        self.total_tasks_scheduled = 0

        logger.debug(f"QtCompliantCleanupManager initialized with {max_workers} workers")

    def start(self, interval_ms: int = 1000) -> None:
        """
        Start the cleanup manager with Qt-compliant threading.

        Args:
            interval_ms: Interval between cleanup batches in milliseconds
        """
        if self.shutdown_requested:
            return

        # Start thread pool for CPU-intensive tasks
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="cleanup"
            )

        # Start cleanup worker thread if not already running
        if self.cleanup_worker is None or not self.cleanup_worker.isRunning():
            self.cleanup_worker = CleanupWorkerThread(self.max_tasks_per_batch)
            self.cleanup_worker.cleanup_interval_ms = interval_ms

            # Connect signals
            self.cleanup_worker.cleanup_completed.connect(self.cleanup_completed.emit)
            self.cleanup_worker.cleanup_error.connect(self.cleanup_error.emit)
            self.cleanup_worker.cleanup_progress.connect(self.cleanup_progress.emit)

            # Start the worker thread
            self.cleanup_worker.start()

            # Emit startup signal
            self.cleanup_started.emit()
            logger.info(f"Qt-compliant cleanup started with {interval_ms}ms interval")

    def stop(self) -> None:
        """Stop the cleanup manager gracefully."""
        self.shutdown_requested = True

        # Stop worker thread
        if self.cleanup_worker and self.cleanup_worker.isRunning():
            self.cleanup_worker.stop_cleanup()
            self.cleanup_worker.wait(5000)  # Wait up to 5 seconds
            if self.cleanup_worker.isRunning():
                logger.warning("Cleanup worker thread did not stop gracefully")
                self.cleanup_worker.terminate()
            self.cleanup_worker = None

        # Stop thread pool
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

        logger.info("Qt-compliant cleanup manager stopped")

    def schedule_cleanup(self, tasks: list[CleanupTask]) -> None:
        """
        Schedule cleanup tasks for background execution.

        Args:
            tasks: List of cleanup tasks to schedule
        """
        if self.shutdown_requested or not tasks:
            return

        if self.cleanup_worker and self.cleanup_worker.isRunning():
            self.cleanup_worker.add_tasks(tasks)
            self.total_tasks_scheduled += len(tasks)
            logger.debug(f"Scheduled {len(tasks)} cleanup tasks")
        else:
            logger.warning("Cannot schedule cleanup tasks: worker thread not running")

    def schedule_object_cleanup(
        self,
        obj_type: type | str,
        priority: CleanupPriority = CleanupPriority.NORMAL,
    ) -> None:
        """
        Schedule cleanup for all objects of a specific type.

        Args:
            obj_type: Object type to clean up
            priority: Priority level for cleanup
        """
        if self.shutdown_requested:
            return

        # Get objects from registry
        registry = get_object_registry()
        cleanup_methods = registry.get_cleanup_methods(obj_type)

        if not cleanup_methods:
            logger.debug(f"No cleanup methods found for type {obj_type}")
            return

        # Create tasks for all registered objects of this type
        tasks = []
        type_name = obj_type if isinstance(obj_type, str) else obj_type.__name__

        for obj_ref in registry.get_objects_by_type(type_name):
            obj = obj_ref() if obj_ref else None
            if obj is None:
                continue

            for method_name in cleanup_methods:
                task = CleanupTask(
                    target_ref=obj_ref,
                    cleanup_method=method_name,
                    priority=priority
                )
                tasks.append(task)

        if tasks:
            self.schedule_cleanup(tasks)
            logger.debug(f"Scheduled cleanup for {len(tasks)} objects of type {type_name}")

    def is_running(self) -> bool:
        """Check if the cleanup manager is currently running."""
        return (
            not self.shutdown_requested and
            self.cleanup_worker is not None and
            self.cleanup_worker.isRunning()
        )

    def get_statistics(self) -> dict:
        """Get cleanup statistics."""
        if self.cleanup_worker:
            return {
                'total_scheduled': self.total_tasks_scheduled,
                'total_completed': self.cleanup_worker.tasks_completed,
                'total_failed': self.cleanup_worker.tasks_failed,
                'total_cleanup_time': self.cleanup_worker.total_cleanup_time,
                'is_running': self.is_running()
            }
        else:
            return {
                'total_scheduled': self.total_tasks_scheduled,
                'total_completed': 0,
                'total_failed': 0,
                'total_cleanup_time': 0.0,
                'is_running': False
            }


# Global instance management
_qt_compliant_cleanup_manager: Optional[QtCompliantCleanupManager] = None
_object_registry: Optional[Any] = None  # Import from original module


def get_qt_compliant_cleanup_manager() -> QtCompliantCleanupManager:
    """Get the global Qt-compliant cleanup manager instance."""
    global _qt_compliant_cleanup_manager
    if _qt_compliant_cleanup_manager is None:
        _qt_compliant_cleanup_manager = QtCompliantCleanupManager()
    return _qt_compliant_cleanup_manager


def get_object_registry():
    """Get the object registry from the original cleanup module."""
    global _object_registry
    if _object_registry is None:
        # Import here to avoid circular imports
        from .cleanup_optimized import get_object_registry as get_orig_registry
        _object_registry = get_orig_registry()
    return _object_registry


def initialize_qt_compliant_cleanup() -> None:
    """Initialize the Qt-compliant cleanup system."""
    # Start the Qt-compliant cleanup manager
    cleanup_manager = get_qt_compliant_cleanup_manager()
    cleanup_manager.start()

    logger.info("Qt-compliant cleanup system initialized")


def schedule_qt_safe_cleanup(
    obj_type: type | str,
    priority: CleanupPriority = CleanupPriority.NORMAL
) -> None:
    """
    Convenience function to schedule Qt-safe cleanup for objects of a type.

    Args:
        obj_type: Object type to clean up
        priority: Priority level for cleanup
    """
    cleanup_manager = get_qt_compliant_cleanup_manager()
    cleanup_manager.schedule_object_cleanup(obj_type, priority)


def stop_qt_compliant_cleanup() -> None:
    """Stop the Qt-compliant cleanup system."""
    global _qt_compliant_cleanup_manager
    if _qt_compliant_cleanup_manager:
        _qt_compliant_cleanup_manager.stop()
        _qt_compliant_cleanup_manager = None

    logger.info("Qt-compliant cleanup system stopped")


# Context manager for safe cleanup operations
class QtSafeCleanupContext:
    """Context manager for Qt-safe cleanup operations."""

    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self.cleanup_manager = None

    def __enter__(self):
        self.cleanup_manager = get_qt_compliant_cleanup_manager()
        self.cleanup_manager.start(self.interval_ms)
        return self.cleanup_manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_manager:
            self.cleanup_manager.stop()


# Utility function to ensure Qt-compliant timer operations
def ensure_qt_compliant_timer(func):
    """
    Decorator to ensure timer operations are Qt-compliant.

    This decorator can be used to wrap functions that create or manipulate
    QTimer objects to ensure they're created in appropriate thread contexts.
    """
    def wrapper(*args, **kwargs):
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QThread

        current_thread = QThread.currentThread()
        app = QApplication.instance()

        if app is None:
            logger.warning("No QApplication available for timer operation")
            return func(*args, **kwargs)

        app_thread = app.thread()

        if current_thread != app_thread and not isinstance(current_thread, QThread):
            logger.warning(
                f"Timer operation in non-Qt thread: {type(current_thread)}. "
                "Consider using Qt-compliant cleanup manager instead."
            )

        return func(*args, **kwargs)

    return wrapper