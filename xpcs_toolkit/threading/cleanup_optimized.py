"""Optimized cleanup system for XPCS Toolkit threading operations.

This module provides optimized cleanup and resource management for
background threads, worker processes, and system resources.
"""

import gc
import threading
import time
from enum import Enum
from typing import Any

from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


class CleanupPriority(Enum):
    """Priority levels for cleanup operations."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ObjectRegistry:
    """Registry for tracking objects that need cleanup."""

    def __init__(self):
        self.objects: dict[str, Any] = {}
        self._lock = threading.Lock()

    def register(self, key: str, obj: Any):
        """Register an object for tracking."""
        with self._lock:
            self.objects[key] = obj

    def unregister(self, key: str):
        """Unregister an object."""
        with self._lock:
            self.objects.pop(key, None)

    def get_object(self, key: str) -> Any | None:
        """Get a registered object."""
        with self._lock:
            return self.objects.get(key)

    def clear_all(self):
        """Clear all registered objects."""
        with self._lock:
            self.objects.clear()

    def get_objects_by_type(self, obj_type: str) -> list[Any]:
        """Get all registered objects of a specific type.

        Args:
            obj_type: Type name to filter by (e.g., 'XpcsFile', 'WorkerManager')

        Returns:
            List of objects matching the specified type
        """
        with self._lock:
            matching_objects = []
            for key, obj in self.objects.items():
                # Check if the key contains the type name or if the object's class name matches
                if obj_type in key or (
                    hasattr(obj, "__class__") and obj.__class__.__name__ == obj_type
                ):
                    matching_objects.append(obj)
            return matching_objects


# Global object registry
_object_registry = None


def shutdown_threads(threads: list[threading.Thread], timeout: float = 5.0):
    """Utility function to shutdown a list of threads."""
    logger.debug(f"Shutting down {len(threads)} threads")
    for thread in threads:
        if thread.is_alive():
            thread.join(timeout=timeout)
    logger.debug("Thread shutdown complete")


class CleanupScheduler:
    """Schedules and manages cleanup operations."""

    def __init__(self):
        self.cleanup_tasks: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def schedule_cleanup(self, task_name: str, cleanup_func, delay: float = 0.0):
        """Schedule a cleanup task."""
        with self._lock:
            self.cleanup_tasks.append(
                {
                    "name": task_name,
                    "func": cleanup_func,
                    "delay": delay,
                    "scheduled_time": time.time() + delay,
                }
            )
        logger.debug(f"Scheduled cleanup task: {task_name}")

    def execute_pending_cleanup(self):
        """Execute all pending cleanup tasks."""
        current_time = time.time()
        executed_tasks = []

        with self._lock:
            for task in self.cleanup_tasks[
                :
            ]:  # Copy to avoid modification during iteration
                if current_time >= task["scheduled_time"]:
                    try:
                        task["func"]()
                        executed_tasks.append(task["name"])
                        self.cleanup_tasks.remove(task)
                    except Exception as e:
                        logger.warning(f"Cleanup task {task['name']} failed: {e}")
                        self.cleanup_tasks.remove(task)

        if executed_tasks:
            logger.debug(f"Executed cleanup tasks: {executed_tasks}")


class SmartGarbageCollector:
    """Smart garbage collection with memory pressure awareness."""

    def __init__(self):
        self.collection_threshold = 0.8  # 80% memory usage
        self.last_collection = time.time()
        self.min_collection_interval = 30.0  # 30 seconds minimum between collections

    def should_collect(self) -> bool:
        """Determine if garbage collection should be triggered."""
        # Check time since last collection
        if time.time() - self.last_collection < self.min_collection_interval:
            return False

        # In a real implementation, we would check memory pressure here
        # For now, we'll use a simple timer-based approach
        return True

    def collect(self):
        """Perform smart garbage collection."""
        if not self.should_collect():
            return

        logger.debug("Performing smart garbage collection")
        collected = gc.collect()
        self.last_collection = time.time()
        logger.debug(f"Garbage collection completed, freed {collected} objects")


class OptimizedCleanupSystem:
    """Main cleanup system coordinating all cleanup operations."""

    def __init__(self):
        self.active_threads: list[threading.Thread] = []
        self.threads_lock = threading.Lock()
        self.cleanup_scheduler = CleanupScheduler()
        self.garbage_collector = SmartGarbageCollector()
        self.is_initialized = True
        logger.info("Optimized cleanup system initialized")

    def shutdown(self, timeout: float = 5.0):
        """Shutdown the entire cleanup system."""
        logger.info("Starting optimized cleanup system shutdown")

        # Execute any pending cleanup tasks
        self.cleanup_scheduler.execute_pending_cleanup()

        # Shutdown all active threads
        with self.threads_lock:
            shutdown_threads(self.active_threads[:], timeout)
            self.active_threads.clear()

        # Perform final garbage collection
        self.garbage_collector.collect()

        self.is_initialized = False
        logger.info("Optimized cleanup system shutdown complete")


# Global cleanup system instance
_cleanup_system: OptimizedCleanupSystem | None = None


def get_cleanup_system() -> OptimizedCleanupSystem:
    """Get or create the global cleanup system instance."""
    global _cleanup_system
    if _cleanup_system is None:
        _cleanup_system = OptimizedCleanupSystem()
    return _cleanup_system


def shutdown_worker_managers():
    """Shutdown all active threads in the cleanup system."""
    try:
        cleanup_system = get_cleanup_system()
        with cleanup_system.threads_lock:
            shutdown_threads(cleanup_system.active_threads[:])
            cleanup_system.active_threads.clear()
    except Exception as e:
        logger.error(f"Error shutting down threads: {e}")


def schedule_cleanup(task_name: str, cleanup_func, delay: float = 0.0):
    """Schedule a cleanup task."""
    try:
        cleanup_system = get_cleanup_system()
        cleanup_system.cleanup_scheduler.schedule_cleanup(
            task_name, cleanup_func, delay
        )
    except Exception as e:
        logger.error(f"Error scheduling cleanup: {e}")


def smart_garbage_collection():
    """Trigger smart garbage collection."""
    try:
        cleanup_system = get_cleanup_system()
        cleanup_system.garbage_collector.collect()
    except Exception as e:
        logger.error(f"Error during smart garbage collection: {e}")


def cleanup_system_shutdown(timeout: float = 5.0):
    """Shutdown the cleanup system."""
    global _cleanup_system
    try:
        if _cleanup_system is not None:
            _cleanup_system.shutdown(timeout)
            _cleanup_system = None
    except Exception as e:
        logger.error(f"Error during cleanup system shutdown: {e}")


# Additional functions expected by the codebase


def get_object_registry() -> ObjectRegistry:
    """Get or create the global object registry."""
    global _object_registry
    if _object_registry is None:
        _object_registry = ObjectRegistry()
    return _object_registry


def register_for_cleanup(key: str, obj: Any):
    """Register an object for cleanup tracking."""
    try:
        registry = get_object_registry()
        registry.register(key, obj)
        logger.debug(f"Registered object for cleanup: {key}")
    except Exception as e:
        logger.error(f"Error registering object for cleanup: {e}")


def shutdown_optimized_cleanup(timeout: float = 5.0):
    """Shutdown optimized cleanup system (alias for cleanup_system_shutdown)."""
    try:
        cleanup_system_shutdown(timeout)
    except Exception as e:
        logger.error(f"Error during optimized cleanup shutdown: {e}")


def initialize_optimized_cleanup():
    """Initialize the optimized cleanup system."""
    try:
        cleanup_system = get_cleanup_system()
        logger.info("Optimized cleanup system initialized")
        return cleanup_system
    except Exception as e:
        logger.error(f"Error initializing optimized cleanup: {e}")
        return None


def schedule_type_cleanup(obj_type: str, priority_or_func, delay: float = 0.0):
    """Schedule cleanup for objects of a specific type.

    Args:
        obj_type: Type of object to clean up
        priority_or_func: Either a CleanupPriority enum value or a callable cleanup function
        delay: Delay before executing cleanup
    """
    try:
        cleanup_system = get_cleanup_system()
        task_name = f"type_cleanup_{obj_type}"

        # Handle both priority-based and function-based cleanup
        if isinstance(priority_or_func, CleanupPriority):
            # Create a priority-based cleanup function
            def priority_cleanup():
                """Priority-based cleanup function."""
                # Perform garbage collection based on priority
                if priority_or_func == CleanupPriority.CRITICAL:
                    # Force immediate garbage collection
                    import gc

                    collected = gc.collect()
                    logger.debug(
                        f"Critical cleanup for {obj_type}: freed {collected} objects"
                    )
                elif priority_or_func == CleanupPriority.HIGH:
                    # Trigger smart garbage collection
                    cleanup_system.garbage_collector.collect()
                    logger.debug(f"High priority cleanup for {obj_type} completed")
                elif priority_or_func == CleanupPriority.MEDIUM:
                    # Schedule garbage collection if needed
                    if cleanup_system.garbage_collector.should_collect():
                        cleanup_system.garbage_collector.collect()
                    logger.debug(f"Medium priority cleanup for {obj_type} completed")
                else:  # LOW priority
                    # Just log the cleanup attempt
                    logger.debug(f"Low priority cleanup for {obj_type} scheduled")

            cleanup_func = priority_cleanup
        elif callable(priority_or_func):
            # It's a callable function
            cleanup_func = priority_or_func
        else:
            # Invalid parameter type
            raise TypeError(
                f"priority_or_func must be CleanupPriority enum or callable, got {type(priority_or_func)}"
            )

        cleanup_system.cleanup_scheduler.schedule_cleanup(
            task_name, cleanup_func, delay
        )
        logger.debug(f"Scheduled type cleanup for: {obj_type}")
    except Exception as e:
        logger.error(f"Error scheduling type cleanup: {e}")


def smart_gc_collect(*args):
    """Perform smart garbage collection (with optional arguments for compatibility)."""
    try:
        smart_garbage_collection()
    except Exception as e:
        logger.error(f"Error during smart garbage collection: {e}")
