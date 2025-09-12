"""
Optimized cleanup system to eliminate expensive gc.get_objects() traversal.

This module provides lightweight object registration and background cleanup
to replace the GUI-blocking operations that iterate through all Python objects.
"""

from __future__ import annotations

import gc
import time
import threading
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, QTimer, Signal

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


class ObjectRegistry:
    """
    Lightweight registry to track objects without expensive traversal.

    Uses weak references to prevent memory leaks while providing fast
    lookup for cleanup operations. Thread-safe and singleton pattern.
    """

    _instance: Optional["ObjectRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ObjectRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._registry_lock = threading.RLock()

        # Registry maps: type_name -> set of weak references
        self._objects_by_type: Dict[str, Set[weakref.ReferenceType]] = defaultdict(set)

        # Cleanup callbacks for each type
        self._cleanup_callbacks: Dict[str, List[str]] = defaultdict(list)

        # Statistics
        self._registration_count = 0
        self._cleanup_count = 0
        self._last_cleanup_time = time.time()

        logger.debug("ObjectRegistry initialized")

    def register(self, obj: Any, cleanup_methods: Optional[List[str]] = None) -> None:
        """
        Register an object for tracking and cleanup.

        Parameters
        ----------
        obj : Any
            Object to register
        cleanup_methods : List[str], optional
            Method names to call during cleanup
        """
        if obj is None:
            return

        obj_type = type(obj).__name__

        with self._registry_lock:
            # Create weak reference with cleanup callback
            weak_ref = weakref.ref(obj, self._on_object_deleted)
            self._objects_by_type[obj_type].add(weak_ref)

            if cleanup_methods:
                self._cleanup_callbacks[obj_type] = cleanup_methods

            self._registration_count += 1

        logger.debug(
            f"Registered {obj_type} object (total registered: {self._registration_count})"
        )

    def get_objects_by_type(self, obj_type: Union[Type, str]) -> List[Any]:
        """
        Get all registered objects of a specific type.

        Parameters
        ----------
        obj_type : Type or str
            Type to look up

        Returns
        -------
        List[Any]
            Live objects of the specified type
        """
        if isinstance(obj_type, type):
            type_name = obj_type.__name__
        else:
            type_name = obj_type

        live_objects = []

        with self._registry_lock:
            weak_refs = self._objects_by_type.get(type_name, set())
            dead_refs = set()

            for weak_ref in weak_refs:
                obj = weak_ref()
                if obj is not None:
                    live_objects.append(obj)
                else:
                    dead_refs.add(weak_ref)

            # Clean up dead references
            if dead_refs:
                self._objects_by_type[type_name] -= dead_refs

        return live_objects

    def get_cleanup_tasks(
        self,
        obj_type: Union[Type, str],
        priority: CleanupPriority = CleanupPriority.NORMAL,
    ) -> List[CleanupTask]:
        """
        Generate cleanup tasks for objects of a specific type.

        Parameters
        ----------
        obj_type : Type or str
            Type to generate tasks for
        priority : CleanupPriority
            Priority level for the cleanup tasks

        Returns
        -------
        List[CleanupTask]
            Cleanup tasks for live objects
        """
        if isinstance(obj_type, type):
            type_name = obj_type.__name__
        else:
            type_name = obj_type

        cleanup_methods = self._cleanup_callbacks.get(type_name, [])
        if not cleanup_methods:
            return []

        tasks = []

        with self._registry_lock:
            weak_refs = self._objects_by_type.get(type_name, set())
            dead_refs = set()

            for weak_ref in weak_refs:
                obj = weak_ref()
                if obj is not None:
                    for method_name in cleanup_methods:
                        task = CleanupTask(
                            target_ref=weak_ref,
                            cleanup_method=method_name,
                            priority=priority,
                        )
                        tasks.append(task)
                else:
                    dead_refs.add(weak_ref)

            # Clean up dead references
            if dead_refs:
                self._objects_by_type[type_name] -= dead_refs

        return tasks

    def _on_object_deleted(self, weak_ref: weakref.ReferenceType) -> None:
        """Callback when a registered object is garbage collected."""
        with self._registry_lock:
            # Remove from all type sets
            for type_name, weak_refs in self._objects_by_type.items():
                weak_refs.discard(weak_ref)

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        with self._registry_lock:
            type_counts = {}
            total_objects = 0

            for type_name, weak_refs in self._objects_by_type.items():
                live_count = sum(1 for ref in weak_refs if ref() is not None)
                type_counts[type_name] = live_count
                total_objects += live_count

            return {
                "total_objects": total_objects,
                "types": type_counts,
                "registrations": self._registration_count,
                "cleanups": self._cleanup_count,
                "last_cleanup": self._last_cleanup_time,
            }

    def clear(self) -> None:
        """Clear all registered objects (for testing)."""
        with self._registry_lock:
            self._objects_by_type.clear()
            self._cleanup_callbacks.clear()
            self._registration_count = 0
            self._cleanup_count = 0


class BackgroundCleanupManager(QObject):
    """
    Background cleanup manager for non-blocking operations.

    Manages cleanup tasks in background threads with priorities,
    rate limiting, and progressive execution to avoid GUI freezes.
    """

    # Signals for monitoring cleanup progress
    cleanup_started = Signal()
    cleanup_completed = Signal(dict)  # Statistics
    cleanup_progress = Signal(int, int)  # completed, total
    cleanup_error = Signal(str, str)  # task_type, error_message

    def __init__(self, max_workers: int = 2, max_tasks_per_batch: int = 10):
        super().__init__()

        self._max_workers = max_workers
        self._max_tasks_per_batch = max_tasks_per_batch
        self._executor: Optional[ThreadPoolExecutor] = None
        self._shutdown_requested = False

        # Task queues by priority
        self._task_queues: Dict[CleanupPriority, deque] = {
            priority: deque() for priority in CleanupPriority
        }
        self._queue_lock = threading.RLock()

        # Rate limiting
        self._last_batch_time = time.time()
        self._min_batch_interval = 0.1  # 100ms between batches

        # Statistics
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._total_cleanup_time = 0.0

        # Timer for periodic cleanup
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._process_cleanup_batch)
        self._cleanup_timer.setSingleShot(False)

        logger.debug(f"BackgroundCleanupManager initialized with {max_workers} workers")

    def start(self, interval_ms: int = 1000) -> None:
        """Start the background cleanup manager."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self._max_workers, thread_name_prefix="cleanup"
            )

        if not self._cleanup_timer.isActive():
            self._cleanup_timer.start(interval_ms)
            logger.info(f"Background cleanup started with {interval_ms}ms interval")

    def stop(self) -> None:
        """Stop the background cleanup manager."""
        self._shutdown_requested = True

        if self._cleanup_timer.isActive():
            self._cleanup_timer.stop()

        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("Background cleanup manager stopped")

    def schedule_cleanup(self, tasks: List[CleanupTask]) -> None:
        """
        Schedule cleanup tasks for background execution.

        Parameters
        ----------
        tasks : List[CleanupTask]
            Tasks to schedule
        """
        if self._shutdown_requested or not tasks:
            return

        with self._queue_lock:
            for task in tasks:
                self._task_queues[task.priority].append(task)

        logger.debug(f"Scheduled {len(tasks)} cleanup tasks")

    def schedule_object_cleanup(
        self,
        obj_type: Union[Type, str],
        priority: CleanupPriority = CleanupPriority.NORMAL,
    ) -> None:
        """
        Schedule cleanup for all objects of a specific type.

        Parameters
        ----------
        obj_type : Type or str
            Object type to clean up
        priority : CleanupPriority
            Priority level for cleanup
        """
        registry = ObjectRegistry()
        tasks = registry.get_cleanup_tasks(obj_type, priority)
        if tasks:
            self.schedule_cleanup(tasks)

    def _process_cleanup_batch(self) -> None:
        """Process a batch of cleanup tasks."""
        if self._shutdown_requested:
            return

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_batch_time < self._min_batch_interval:
            return

        # Get tasks from priority queues
        batch_tasks = self._get_next_batch()
        if not batch_tasks:
            return

        self._last_batch_time = current_time

        # Submit batch to thread pool
        if self._executor is not None:
            self._executor.submit(self._execute_batch, batch_tasks)
            # Don't wait for completion to avoid blocking GUI

    def _get_next_batch(self) -> List[CleanupTask]:
        """Get the next batch of tasks to process, respecting priorities."""
        batch = []

        with self._queue_lock:
            # Process in priority order (highest first)
            for priority in sorted(
                CleanupPriority, key=lambda p: p.value, reverse=True
            ):
                queue = self._task_queues[priority]

                while queue and len(batch) < self._max_tasks_per_batch:
                    task = queue.popleft()

                    # Skip expired tasks
                    if task.is_expired():
                        continue

                    # Skip tasks with dead references
                    if task.get_target() is None:
                        continue

                    batch.append(task)

        return batch

    def _execute_batch(self, tasks: List[CleanupTask]) -> None:
        """Execute a batch of cleanup tasks in background thread."""
        if not tasks:
            return

        start_time = time.time()
        completed = 0
        failed = 0

        self.cleanup_started.emit()

        for i, task in enumerate(tasks):
            try:
                target = task.get_target()
                if target is None:
                    continue

                # Call the cleanup method
                if hasattr(target, task.cleanup_method):
                    method = getattr(target, task.cleanup_method)
                    method(*task.args, **task.kwargs)
                    completed += 1
                else:
                    logger.warning(
                        f"Cleanup method '{task.cleanup_method}' not found on {type(target).__name__}"
                    )
                    failed += 1

                # Emit progress
                self.cleanup_progress.emit(i + 1, len(tasks))

            except Exception as e:
                failed += 1
                task.retry_count += 1
                error_msg = f"Cleanup failed for {task.cleanup_method}: {e}"
                logger.warning(error_msg)
                self.cleanup_error.emit(task.cleanup_method, str(e))

                # Retry if under limit
                if task.retry_count <= task.max_retry_count:
                    with self._queue_lock:
                        self._task_queues[task.priority].append(task)

        # Update statistics
        self._tasks_completed += completed
        self._tasks_failed += failed
        self._total_cleanup_time += time.time() - start_time

        # Emit completion signal
        stats = {
            "completed": completed,
            "failed": failed,
            "batch_time": time.time() - start_time,
            "total_completed": self._tasks_completed,
            "total_failed": self._tasks_failed,
        }

        self.cleanup_completed.emit(stats)

        logger.debug(f"Cleanup batch completed: {completed} success, {failed} failed")

    def get_queue_stats(self) -> Dict[str, int]:
        """Get current queue statistics."""
        with self._queue_lock:
            return {
                priority.name: len(queue)
                for priority, queue in self._task_queues.items()
            }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "total_cleanup_time": self._total_cleanup_time,
            "average_task_time": (
                self._total_cleanup_time / max(1, self._tasks_completed)
            ),
            "queue_lengths": self.get_queue_stats(),
        }


class SmartGarbageCollector:
    """
    Smart garbage collection scheduler to replace manual gc.collect() calls.

    Uses idle time detection and memory pressure monitoring to schedule
    garbage collection without blocking user interactions.
    """

    _instance: Optional["SmartGarbageCollector"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "SmartGarbageCollector":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self._last_gc_time = time.time()
        self._last_activity_time = time.time()
        self._gc_interval = 30.0  # Minimum 30 seconds between GC
        self._idle_threshold = 5.0  # 5 seconds of no activity = idle
        self._memory_threshold = 0.85  # Trigger GC at 85% memory usage
        self._executor: Optional[ThreadPoolExecutor] = None

        logger.debug("SmartGarbageCollector initialized")

    def start(self) -> None:
        """Start the smart garbage collector."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="smart_gc"
            )

        logger.info("Smart garbage collector started")

    def stop(self) -> None:
        """Stop the smart garbage collector."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("Smart garbage collector stopped")

    def mark_activity(self) -> None:
        """Mark that user activity has occurred."""
        self._last_activity_time = time.time()

    def should_run_gc(self, force_memory_check: bool = False) -> bool:
        """Check if garbage collection should be run."""
        current_time = time.time()

        # Check minimum interval
        if current_time - self._last_gc_time < self._gc_interval:
            return False

        # Check if system is idle
        is_idle = current_time - self._last_activity_time > self._idle_threshold

        # Check memory pressure if forced or idle
        if force_memory_check or is_idle:
            try:
                import psutil

                memory_percent = psutil.virtual_memory().percent / 100.0
                if memory_percent > self._memory_threshold:
                    return True
            except Exception:
                pass  # Fallback to time-based GC if memory check fails

        # Run GC if idle and enough time has passed
        return is_idle and current_time - self._last_gc_time > self._gc_interval * 2

    def schedule_gc(self, reason: str = "scheduled") -> None:
        """Schedule garbage collection in background."""
        if not self.should_run_gc():
            return

        if self._executor is not None:
            self._executor.submit(self._run_gc, reason)

    def _run_gc(self, reason: str) -> None:
        """Run garbage collection in background thread."""
        start_time = time.time()

        try:
            collected = gc.collect()
            gc_time = time.time() - start_time

            self._last_gc_time = time.time()

            logger.debug(
                f"Smart GC completed ({reason}): collected {collected} objects in {gc_time:.3f}s"
            )

        except Exception as e:
            logger.warning(f"Smart GC failed: {e}")


# Global instances
_object_registry = None
_background_cleanup_manager = None
_smart_gc = None


def get_object_registry() -> ObjectRegistry:
    """Get the global object registry instance."""
    global _object_registry
    if _object_registry is None:
        _object_registry = ObjectRegistry()
    return _object_registry


def get_background_cleanup_manager() -> BackgroundCleanupManager:
    """Get the global background cleanup manager instance."""
    global _background_cleanup_manager
    if _background_cleanup_manager is None:
        _background_cleanup_manager = BackgroundCleanupManager()
    return _background_cleanup_manager


def get_smart_gc() -> SmartGarbageCollector:
    """Get the global smart garbage collector instance."""
    global _smart_gc
    if _smart_gc is None:
        _smart_gc = SmartGarbageCollector()
    return _smart_gc


def register_for_cleanup(obj: Any, cleanup_methods: Optional[List[str]] = None) -> None:
    """
    Convenience function to register an object for cleanup.

    Parameters
    ----------
    obj : Any
        Object to register
    cleanup_methods : List[str], optional
        Method names to call during cleanup
    """
    registry = get_object_registry()
    registry.register(obj, cleanup_methods)


def schedule_type_cleanup(
    obj_type: Union[Type, str], priority: CleanupPriority = CleanupPriority.NORMAL
) -> None:
    """
    Convenience function to schedule cleanup for all objects of a type.

    Parameters
    ----------
    obj_type : Type or str
        Object type to clean up
    priority : CleanupPriority
        Priority level for cleanup
    """
    cleanup_manager = get_background_cleanup_manager()
    cleanup_manager.schedule_object_cleanup(obj_type, priority)


def smart_gc_collect(reason: str = "manual") -> None:
    """
    Convenience function to request smart garbage collection.

    Parameters
    ----------
    reason : str
        Reason for the GC request (for logging)
    """
    smart_gc = get_smart_gc()
    smart_gc.schedule_gc(reason)


def initialize_optimized_cleanup() -> None:
    """Initialize the optimized cleanup system."""
    # Start background managers
    cleanup_manager = get_background_cleanup_manager()
    cleanup_manager.start()

    smart_gc = get_smart_gc()
    smart_gc.start()

    logger.info("Optimized cleanup system initialized")


def shutdown_optimized_cleanup() -> None:
    """Shutdown the optimized cleanup system."""
    global _background_cleanup_manager, _smart_gc

    if _background_cleanup_manager is not None:
        _background_cleanup_manager.stop()
        _background_cleanup_manager = None

    if _smart_gc is not None:
        _smart_gc.stop()
        _smart_gc = None

    logger.info("Optimized cleanup system shut down")
