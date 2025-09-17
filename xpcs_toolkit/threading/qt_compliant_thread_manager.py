"""
Qt-Compliant Thread Lifecycle Management for XPCS Toolkit.

This module provides comprehensive Qt-compliant thread management with proper
lifecycle handling, signal/slot safety, and resource management to prevent
threading violations and ensure stable operation.
"""

import gc
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from weakref import WeakSet

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import (
    QObject, QThread, QThreadPool, QRunnable, QTimer, Signal, QMutex,
    QMutexLocker, QMetaObject, Qt, Slot
)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ThreadState(Enum):
    """Thread lifecycle states for tracking."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class ThreadSafety(Enum):
    """Thread safety levels for operations."""

    MAIN_THREAD_ONLY = "main_thread_only"
    QT_THREAD_SAFE = "qt_thread_safe"
    WORKER_THREAD_SAFE = "worker_thread_safe"
    THREAD_AGNOSTIC = "thread_agnostic"


@dataclass
class ThreadInfo:
    """Information about a managed thread."""

    thread_id: str
    thread_object: QThread
    state: ThreadState = ThreadState.CREATED
    created_time: float = field(default_factory=time.perf_counter)
    started_time: Optional[float] = None
    stopped_time: Optional[float] = None
    parent_thread: Optional[str] = None
    thread_purpose: str = "general"
    resource_usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkerInfo:
    """Information about a managed worker."""

    worker_id: str
    worker_object: QRunnable
    thread_pool: QThreadPool
    submitted_time: float = field(default_factory=time.perf_counter)
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    priority: int = 0
    thread_safety: ThreadSafety = ThreadSafety.WORKER_THREAD_SAFE


class QtThreadSafetyValidator:
    """
    Validates Qt thread safety for operations and objects.

    This class provides methods to ensure that Qt operations are performed
    in the correct thread context to prevent threading violations.
    """

    def __init__(self):
        self._main_thread = QtCore.QThread.currentThread()
        self._gui_application = None
        self._thread_local = threading.local()

    def is_main_thread(self) -> bool:
        """Check if current thread is the main GUI thread."""
        current_thread = QtCore.QThread.currentThread()
        return current_thread == self._main_thread

    def is_qt_thread(self) -> bool:
        """Check if current thread is a Qt-managed thread."""
        current_thread = QtCore.QThread.currentThread()
        return isinstance(current_thread, QThread)

    def ensure_main_thread(self, operation_name: str):
        """Ensure operation is running in main thread."""
        if not self.is_main_thread():
            raise RuntimeError(
                f"Operation '{operation_name}' must be called from main thread. "
                f"Current thread: {QtCore.QThread.currentThread()}"
            )

    def ensure_qt_thread(self, operation_name: str):
        """Ensure operation is running in a Qt thread."""
        if not self.is_qt_thread():
            raise RuntimeError(
                f"Operation '{operation_name}' must be called from a Qt thread. "
                f"Current thread: {threading.current_thread()}"
            )

    def validate_timer_creation(self, parent: QObject = None) -> bool:
        """
        Validate that QTimer creation is safe in current context.

        Args:
            parent: Optional parent object for the timer

        Returns:
            True if timer creation is safe
        """
        if not self.is_qt_thread():
            logger.warning(
                "QTimer creation attempted in non-Qt thread. "
                "This may cause threading violations."
            )
            return False

        # Check if we have a Qt application
        app = QtWidgets.QApplication.instance()
        if app is None:
            logger.warning("No Qt application instance found for timer creation")
            return False

        return True

    def validate_signal_connection(self, sender: QObject, signal: Signal,
                                 receiver: QObject, slot: Callable,
                                 connection_type: Qt.ConnectionType = Qt.ConnectionType.AutoConnection) -> bool:
        """
        Validate signal/slot connection safety.

        Args:
            sender: Signal sender object
            signal: Signal to connect
            receiver: Signal receiver object
            slot: Slot to connect to
            connection_type: Qt connection type

        Returns:
            True if connection is safe
        """
        # Check sender thread affinity
        sender_thread = sender.thread()
        receiver_thread = receiver.thread()

        if sender_thread != receiver_thread:
            if connection_type == Qt.ConnectionType.DirectConnection:
                logger.warning(
                    "Direct connection between objects in different threads. "
                    "Consider using QueuedConnection."
                )
                return False

        return True


class QtCompliantTimerManager:
    """
    Manager for Qt-compliant timer creation and lifecycle.

    Ensures that all QTimer objects are created in appropriate thread
    contexts and properly managed throughout their lifecycle.
    """

    def __init__(self, safety_validator: QtThreadSafetyValidator):
        self.safety_validator = safety_validator
        self._timers: Dict[str, QTimer] = {}
        self._timer_contexts: Dict[str, Dict[str, Any]] = {}
        self._mutex = QMutex()

    def create_timer(self, timer_id: str, parent: QObject = None,
                    interval: int = 1000, single_shot: bool = False) -> Optional[QTimer]:
        """
        Create a Qt-compliant timer in the current thread.

        Args:
            timer_id: Unique identifier for the timer
            parent: Parent object for the timer
            interval: Timer interval in milliseconds
            single_shot: Whether timer should fire only once

        Returns:
            QTimer instance or None if creation failed
        """
        with QMutexLocker(self._mutex):
            # Validate timer creation context
            if not self.safety_validator.validate_timer_creation(parent):
                logger.error(f"Cannot create timer '{timer_id}' in current context")
                return None

            # Check if timer already exists
            if timer_id in self._timers:
                logger.warning(f"Timer '{timer_id}' already exists")
                return self._timers[timer_id]

            try:
                # Create timer in current thread
                timer = QTimer(parent)
                timer.setInterval(interval)
                timer.setSingleShot(single_shot)

                # Store timer and context information
                self._timers[timer_id] = timer
                self._timer_contexts[timer_id] = {
                    'creation_thread': QtCore.QThread.currentThread(),
                    'creation_time': time.perf_counter(),
                    'parent': parent,
                    'interval': interval,
                    'single_shot': single_shot
                }

                logger.debug(f"Created Qt-compliant timer '{timer_id}' in thread {QtCore.QThread.currentThread()}")
                return timer

            except Exception as e:
                logger.error(f"Failed to create timer '{timer_id}': {e}")
                return None

    def get_timer(self, timer_id: str) -> Optional[QTimer]:
        """Get an existing timer by ID."""
        with QMutexLocker(self._mutex):
            return self._timers.get(timer_id)

    def start_timer(self, timer_id: str, interval: Optional[int] = None) -> bool:
        """Start a timer with optional interval override."""
        timer = self.get_timer(timer_id)
        if timer is None:
            logger.error(f"Timer '{timer_id}' not found")
            return False

        try:
            if interval is not None:
                timer.setInterval(interval)
            timer.start()
            logger.debug(f"Started timer '{timer_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to start timer '{timer_id}': {e}")
            return False

    def stop_timer(self, timer_id: str) -> bool:
        """Stop a timer."""
        timer = self.get_timer(timer_id)
        if timer is None:
            logger.error(f"Timer '{timer_id}' not found")
            return False

        try:
            timer.stop()
            logger.debug(f"Stopped timer '{timer_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to stop timer '{timer_id}': {e}")
            return False

    def destroy_timer(self, timer_id: str) -> bool:
        """Destroy a timer and clean up resources."""
        with QMutexLocker(self._mutex):
            timer = self._timers.get(timer_id)
            if timer is None:
                logger.warning(f"Timer '{timer_id}' not found for destruction")
                return True

            try:
                # Stop timer if running
                timer.stop()

                # Disconnect all signals safely
                try:
                    timer.timeout.disconnect()
                except (TypeError, RuntimeError):
                    # Timer may already be disconnected or destroyed
                    pass

                # Schedule for deletion
                timer.deleteLater()

                # Remove from tracking
                del self._timers[timer_id]
                self._timer_contexts.pop(timer_id, None)

                logger.debug(f"Destroyed timer '{timer_id}'")
                return True

            except Exception as e:
                logger.error(f"Failed to destroy timer '{timer_id}': {e}")
                return False

    def cleanup_all_timers(self):
        """Clean up all managed timers."""
        with QMutexLocker(self._mutex):
            timer_ids = list(self._timers.keys())
            for timer_id in timer_ids:
                self.destroy_timer(timer_id)

            logger.info(f"Cleaned up {len(timer_ids)} timers")


class QtCompliantWorkerManager(QObject):
    """
    Qt-compliant worker management with proper signal/slot handling.

    This manager ensures that all worker operations are performed in
    Qt-compliant ways with proper thread safety and resource management.
    """

    # Signals for worker lifecycle
    worker_started = Signal(str)  # worker_id
    worker_finished = Signal(str, object)  # worker_id, result
    worker_error = Signal(str, str, str)  # worker_id, error_msg, traceback
    worker_progress = Signal(str, int, int, str)  # worker_id, current, total, message

    def __init__(self, thread_pool: QThreadPool, safety_validator: QtThreadSafetyValidator):
        super().__init__()
        self.thread_pool = thread_pool
        self.safety_validator = safety_validator
        self._workers: Dict[str, WorkerInfo] = {}
        self._mutex = QMutex()

        # Ensure we're in the main thread for signal handling
        if not safety_validator.is_main_thread():
            logger.warning("QtCompliantWorkerManager should be created in main thread")

    def submit_worker(self, worker: QRunnable, worker_id: str,
                     priority: int = 0, thread_safety: ThreadSafety = ThreadSafety.WORKER_THREAD_SAFE) -> bool:
        """
        Submit a worker to the thread pool with Qt compliance validation.

        Args:
            worker: QRunnable worker object
            worker_id: Unique identifier for the worker
            priority: Worker priority (higher numbers = higher priority)
            thread_safety: Thread safety level of the worker

        Returns:
            True if worker was successfully submitted
        """
        with QMutexLocker(self._mutex):
            # Validate worker submission context
            if not self.safety_validator.is_qt_thread():
                logger.error(f"Worker '{worker_id}' submission must be from Qt thread")
                return False

            # Check if worker already exists
            if worker_id in self._workers:
                logger.warning(f"Worker '{worker_id}' already exists")
                return False

            try:
                # Create worker info
                worker_info = WorkerInfo(
                    worker_id=worker_id,
                    worker_object=worker,
                    thread_pool=self.thread_pool,
                    priority=priority,
                    thread_safety=thread_safety
                )

                # Connect worker signals if available
                if hasattr(worker, 'signals'):
                    self._connect_worker_signals(worker, worker_id)

                # Submit to thread pool
                self.thread_pool.start(worker, priority)

                # Store worker info
                self._workers[worker_id] = worker_info
                worker_info.started_time = time.perf_counter()

                # Emit started signal
                self.worker_started.emit(worker_id)

                logger.debug(f"Submitted Qt-compliant worker '{worker_id}' with priority {priority}")
                return True

            except Exception as e:
                logger.error(f"Failed to submit worker '{worker_id}': {e}")
                return False

    def _connect_worker_signals(self, worker: QRunnable, worker_id: str):
        """Connect worker signals with Qt-compliant connections."""
        signals = worker.signals

        # Use queued connections for thread safety
        connection_type = Qt.ConnectionType.QueuedConnection

        if hasattr(signals, 'finished'):
            signals.finished.connect(
                lambda result: self._on_worker_finished(worker_id, result),
                connection_type
            )

        if hasattr(signals, 'error'):
            signals.error.connect(
                lambda msg, tb: self._on_worker_error(worker_id, msg, tb),
                connection_type
            )

        if hasattr(signals, 'progress'):
            signals.progress.connect(
                lambda current, total, msg: self._on_worker_progress(worker_id, current, total, msg),
                connection_type
            )

    @Slot(str, object)
    def _on_worker_finished(self, worker_id: str, result: object):
        """Handle worker finished signal."""
        with QMutexLocker(self._mutex):
            worker_info = self._workers.get(worker_id)
            if worker_info:
                worker_info.completed_time = time.perf_counter()
                logger.debug(f"Worker '{worker_id}' completed")

            self.worker_finished.emit(worker_id, result)

    @Slot(str, str, str)
    def _on_worker_error(self, worker_id: str, error_msg: str, traceback_str: str):
        """Handle worker error signal."""
        logger.error(f"Worker '{worker_id}' failed: {error_msg}")
        self.worker_error.emit(worker_id, error_msg, traceback_str)

    @Slot(str, int, int, str)
    def _on_worker_progress(self, worker_id: str, current: int, total: int, message: str):
        """Handle worker progress signal."""
        self.worker_progress.emit(worker_id, current, total, message)

    def get_worker_info(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get information about a worker."""
        with QMutexLocker(self._mutex):
            return self._workers.get(worker_id)

    def cancel_worker(self, worker_id: str) -> bool:
        """Cancel a worker if it supports cancellation."""
        worker_info = self.get_worker_info(worker_id)
        if worker_info is None:
            logger.warning(f"Worker '{worker_id}' not found for cancellation")
            return False

        worker = worker_info.worker_object
        if hasattr(worker, 'cancel'):
            try:
                worker.cancel()
                logger.debug(f"Cancelled worker '{worker_id}'")
                return True
            except Exception as e:
                logger.error(f"Failed to cancel worker '{worker_id}': {e}")
                return False
        else:
            logger.warning(f"Worker '{worker_id}' does not support cancellation")
            return False

    def cleanup_completed_workers(self):
        """Clean up information for completed workers."""
        with QMutexLocker(self._mutex):
            completed_workers = [
                worker_id for worker_id, worker_info in self._workers.items()
                if worker_info.completed_time is not None
            ]

            for worker_id in completed_workers:
                del self._workers[worker_id]

            if completed_workers:
                logger.debug(f"Cleaned up {len(completed_workers)} completed workers")


class QtCompliantThreadManager(QObject):
    """
    Main Qt-compliant thread lifecycle manager.

    This is the central manager that coordinates all Qt threading operations
    to ensure compliance with Qt's threading model and prevent violations.
    """

    # Signals for thread lifecycle events
    thread_started = Signal(str)  # thread_id
    thread_finished = Signal(str)  # thread_id
    thread_error = Signal(str, str)  # thread_id, error_message

    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        # Core components
        self.safety_validator = QtThreadSafetyValidator()
        self.timer_manager = QtCompliantTimerManager(self.safety_validator)

        # Thread pool and worker management
        self.thread_pool = QThreadPool.globalInstance()
        self.worker_manager = QtCompliantWorkerManager(self.thread_pool, self.safety_validator)

        # Thread tracking
        self._threads: Dict[str, ThreadInfo] = {}
        self._thread_pool_metrics = {}
        self._mutex = QMutex()

        # Initialize in main thread
        self.safety_validator.ensure_main_thread("QtCompliantThreadManager.__init__")

        logger.info("Qt-compliant thread manager initialized")

    def create_managed_thread(self, thread_id: str, target: Callable = None,
                             thread_purpose: str = "general") -> Optional[QThread]:
        """
        Create a managed Qt thread with proper lifecycle tracking.

        Args:
            thread_id: Unique identifier for the thread
            target: Optional target function to run in thread
            thread_purpose: Description of thread purpose

        Returns:
            QThread instance or None if creation failed
        """
        with QMutexLocker(self._mutex):
            # Ensure main thread context
            self.safety_validator.ensure_main_thread("create_managed_thread")

            # Check if thread already exists
            if thread_id in self._threads:
                logger.warning(f"Thread '{thread_id}' already exists")
                return self._threads[thread_id].thread_object

            try:
                # Create Qt thread
                thread = QThread()

                # Create thread info
                thread_info = ThreadInfo(
                    thread_id=thread_id,
                    thread_object=thread,
                    thread_purpose=thread_purpose,
                    parent_thread=QtCore.QThread.currentThread().objectName()
                )

                # Connect thread signals
                thread.started.connect(lambda: self._on_thread_started(thread_id))
                thread.finished.connect(lambda: self._on_thread_finished(thread_id))

                # Store thread info
                self._threads[thread_id] = thread_info

                logger.debug(f"Created managed thread '{thread_id}' for {thread_purpose}")
                return thread

            except Exception as e:
                logger.error(f"Failed to create thread '{thread_id}': {e}")
                return None

    @Slot(str)
    def _on_thread_started(self, thread_id: str):
        """Handle thread started signal."""
        with QMutexLocker(self._mutex):
            thread_info = self._threads.get(thread_id)
            if thread_info:
                thread_info.state = ThreadState.RUNNING
                thread_info.started_time = time.perf_counter()
                logger.debug(f"Thread '{thread_id}' started")

            self.thread_started.emit(thread_id)

    @Slot(str)
    def _on_thread_finished(self, thread_id: str):
        """Handle thread finished signal."""
        with QMutexLocker(self._mutex):
            thread_info = self._threads.get(thread_id)
            if thread_info:
                thread_info.state = ThreadState.STOPPED
                thread_info.stopped_time = time.perf_counter()
                logger.debug(f"Thread '{thread_id}' finished")

            self.thread_finished.emit(thread_id)

    def start_thread(self, thread_id: str) -> bool:
        """Start a managed thread."""
        thread_info = self._threads.get(thread_id)
        if thread_info is None:
            logger.error(f"Thread '{thread_id}' not found")
            return False

        try:
            thread_info.state = ThreadState.STARTING
            thread_info.thread_object.start()
            logger.debug(f"Starting thread '{thread_id}'")
            return True
        except Exception as e:
            thread_info.state = ThreadState.FAILED
            logger.error(f"Failed to start thread '{thread_id}': {e}")
            return False

    def stop_thread(self, thread_id: str, timeout_ms: int = 5000) -> bool:
        """Stop a managed thread gracefully."""
        thread_info = self._threads.get(thread_id)
        if thread_info is None:
            logger.error(f"Thread '{thread_id}' not found")
            return False

        try:
            thread = thread_info.thread_object
            thread_info.state = ThreadState.STOPPING

            # Request interruption
            thread.requestInterruption()

            # Wait for thread to finish
            if thread.wait(timeout_ms):
                logger.debug(f"Thread '{thread_id}' stopped gracefully")
                return True
            else:
                # Force termination if needed
                thread.terminate()
                thread.wait(1000)  # Wait up to 1 second for termination
                logger.warning(f"Thread '{thread_id}' was forcefully terminated")
                return True

        except Exception as e:
            logger.error(f"Failed to stop thread '{thread_id}': {e}")
            return False

    def cleanup_finished_threads(self):
        """Clean up resources for finished threads."""
        with QMutexLocker(self._mutex):
            finished_threads = [
                thread_id for thread_id, thread_info in self._threads.items()
                if thread_info.state == ThreadState.STOPPED
            ]

            for thread_id in finished_threads:
                thread_info = self._threads[thread_id]
                try:
                    # Ensure thread is properly finished
                    thread_info.thread_object.wait(100)
                    thread_info.thread_object.deleteLater()
                    del self._threads[thread_id]
                    logger.debug(f"Cleaned up finished thread '{thread_id}'")
                except Exception as e:
                    logger.warning(f"Error cleaning up thread '{thread_id}': {e}")

    def get_thread_info(self, thread_id: str) -> Optional[ThreadInfo]:
        """Get information about a managed thread."""
        with QMutexLocker(self._mutex):
            return self._threads.get(thread_id)

    def get_all_thread_info(self) -> Dict[str, ThreadInfo]:
        """Get information about all managed threads."""
        with QMutexLocker(self._mutex):
            return self._threads.copy()

    def shutdown(self):
        """
        Shutdown the thread manager and clean up all resources.

        This method should be called before application exit to ensure
        proper cleanup of all managed threads and resources.
        """
        logger.info("Shutting down Qt-compliant thread manager")

        # Stop all managed threads
        thread_ids = list(self._threads.keys())
        for thread_id in thread_ids:
            self.stop_thread(thread_id)

        # Clean up timers
        self.timer_manager.cleanup_all_timers()

        # Clean up workers
        self.worker_manager.cleanup_completed_workers()

        # Final cleanup
        self.cleanup_finished_threads()

        logger.info("Qt-compliant thread manager shutdown complete")


# Context manager for Qt-compliant operations
@contextmanager
def qt_compliant_context(thread_manager: QtCompliantThreadManager = None):
    """
    Context manager for Qt-compliant thread operations.

    Usage:
        with qt_compliant_context() as manager:
            thread = manager.create_managed_thread("my_thread")
            manager.start_thread("my_thread")
    """
    if thread_manager is None:
        thread_manager = QtCompliantThreadManager()

    try:
        yield thread_manager
    except Exception as e:
        logger.error(f"Error in Qt-compliant context: {e}")
        raise
    finally:
        # Ensure cleanup on exit
        try:
            thread_manager.cleanup_finished_threads()
        except Exception as e:
            logger.warning(f"Error during Qt-compliant context cleanup: {e}")


# Singleton instance for global access
_global_thread_manager: Optional[QtCompliantThreadManager] = None


def get_qt_compliant_thread_manager() -> QtCompliantThreadManager:
    """
    Get the global Qt-compliant thread manager instance.

    Returns:
        QtCompliantThreadManager instance
    """
    global _global_thread_manager

    if _global_thread_manager is None:
        _global_thread_manager = QtCompliantThreadManager()

    return _global_thread_manager


def initialize_qt_thread_management():
    """
    Initialize the global Qt-compliant thread management system.

    This function should be called early in application initialization
    to set up proper Qt thread management.
    """
    manager = get_qt_compliant_thread_manager()
    logger.info("Global Qt-compliant thread management initialized")
    return manager


def shutdown_qt_thread_management():
    """
    Shutdown the global Qt-compliant thread management system.

    This function should be called before application exit to ensure
    proper cleanup of all threading resources.
    """
    global _global_thread_manager

    if _global_thread_manager is not None:
        _global_thread_manager.shutdown()
        _global_thread_manager = None
        logger.info("Global Qt-compliant thread management shutdown")