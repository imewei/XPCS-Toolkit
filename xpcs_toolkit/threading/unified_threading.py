"""
Unified Threading Manager for XPCS Toolkit

This module consolidates all threading patterns into a single, optimized system
that provides intelligent resource allocation, priority-based execution, and
seamless integration with the XPCS GUI and memory management systems.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any

import psutil
from PySide6.QtCore import QObject, Signal

from ..utils.logging_config import get_logger
from ..utils.memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


class TaskPriority(Enum):
    """Unified priority system for all task types."""

    CRITICAL = 0  # UI-blocking operations (must complete immediately)
    HIGH = 1  # User-triggered operations (plotting, analysis)
    NORMAL = 2  # Background processing (preloading, caching)
    LOW = 3  # Maintenance tasks (cleanup, optimization)
    DEFERRED = 4  # Can be postponed during high load


class TaskType(Enum):
    """Categories of tasks for optimized execution."""

    GUI_UPDATE = "gui_update"  # GUI updates, widget changes
    COMPUTATION = "computation"  # Heavy calculations, fitting
    DATA_LOADING = "data_loading"  # File I/O, data loading
    PLOT_GENERATION = "plotting"  # Plot creation, rendering
    MEMORY_OPERATION = "memory"  # Cache operations, cleanup


@dataclass
class UnifiedTask:
    """Unified task container for all worker types."""

    task_id: str
    priority: TaskPriority
    task_type: TaskType
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)

    # Execution control
    timeout_seconds: float | None = None
    memory_limit_mb: float | None = None
    requires_gui: bool = False

    # State tracking
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None

    # Results
    result: Any = None
    error: Exception | None = None

    def __lt__(self, other):
        """Priority queue comparison."""
        return self.priority.value < other.priority.value

    @property
    def execution_time(self) -> float:
        """Get execution time if completed."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return 0.0

    @property
    def wait_time(self) -> float:
        """Get time spent waiting in queue."""
        if self.started_at:
            return self.started_at - self.created_at
        return time.time() - self.created_at


class UnifiedThreadingManager(QObject):
    """
    Consolidated threading manager that unifies all threading patterns.

    Features:
    - Priority-based task execution
    - Adaptive thread pool sizing based on system resources
    - Memory-aware task scheduling
    - Integrated progress tracking
    - Intelligent load balancing
    """

    # Unified signals for all task types
    task_started = Signal(str, str)  # task_id, task_type
    task_progress = Signal(str, int, int, str)  # task_id, current, total, message
    task_completed = Signal(str, object)  # task_id, result
    task_failed = Signal(str, str)  # task_id, error_message

    # System-level signals
    load_changed = Signal(float)  # system load percentage
    memory_pressure_changed = Signal(str)  # pressure level

    def __init__(self, max_workers: int | None = None):
        super().__init__()

        # System resource detection
        self.cpu_count = psutil.cpu_count()
        self.total_memory_gb = psutil.virtual_memory().total / (1024**3)

        # Adaptive thread pool configuration
        if max_workers is None:
            max_workers = min(32, (self.cpu_count or 1) * 2)

        self.max_workers = max_workers
        self.memory_manager = get_memory_manager()

        # Specialized thread pools for different task types
        self._thread_pools = {
            TaskType.GUI_UPDATE: ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="gui-"
            ),
            TaskType.COMPUTATION: ThreadPoolExecutor(
                max_workers=max(1, self.cpu_count - 1), thread_name_prefix="compute-"
            ),
            TaskType.DATA_LOADING: ThreadPoolExecutor(
                max_workers=4, thread_name_prefix="io-"
            ),
            TaskType.PLOT_GENERATION: ThreadPoolExecutor(
                max_workers=3, thread_name_prefix="plot-"
            ),
            TaskType.MEMORY_OPERATION: ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="memory-"
            ),
        }

        # Task management
        self._task_queue = PriorityQueue()
        self._active_tasks: dict[str, UnifiedTask] = {}
        self._completed_tasks: dict[str, UnifiedTask] = {}
        self._task_lock = threading.RLock()

        # Performance tracking
        self._stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "total_wait_time": 0.0,
            "memory_rejected": 0,
            "timeout_cancelled": 0,
        }

        # Adaptive configuration
        self._last_pressure_check = 0
        self._adaptive_config = {
            "queue_size_limit": 100,
            "memory_check_interval": 5.0,
            "load_balance_interval": 10.0,
        }

        # System monitoring
        self._monitoring_active = True
        self._monitor_thread = None
        self._start_monitoring()

        logger.info(
            f"UnifiedThreadingManager initialized: {max_workers} workers, {self.cpu_count} CPUs, {self.total_memory_gb:.1f}GB RAM"
        )

    def submit_task(
        self,
        task_id: str,
        function: Callable,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_type: TaskType = TaskType.COMPUTATION,
        *args,
        **kwargs,
    ) -> bool:
        """
        Submit a task for execution with unified priority handling.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task
        function : Callable
            Function to execute
        priority : TaskPriority
            Task priority level
        task_type : TaskType
            Type of task for optimal scheduling
        *args, **kwargs
            Function arguments

        Returns
        -------
        bool
            True if task was accepted for execution
        """
        # Check system limits
        if not self._can_accept_task(task_type):
            logger.warning(f"Task {task_id} rejected due to system limits")
            self._stats["memory_rejected"] += 1
            return False

        # Create unified task
        task = UnifiedTask(
            task_id=task_id,
            priority=priority,
            task_type=task_type,
            function=function,
            args=args,
            kwargs=kwargs,
        )

        # Add to queue
        with self._task_lock:
            self._task_queue.put(task)
            self._stats["tasks_submitted"] += 1

        # Schedule execution
        self._schedule_task_execution()

        logger.debug(f"Task {task_id} submitted with priority {priority.name}")
        return True

    def _can_accept_task(self, task_type: TaskType) -> bool:
        """Check if system can accept new tasks."""
        # Memory pressure check
        memory_pressure = self.memory_manager.get_memory_pressure()
        if memory_pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            if task_type in [TaskType.DATA_LOADING, TaskType.MEMORY_OPERATION]:
                return memory_pressure != MemoryPressure.CRITICAL
            return False

        # Queue size check
        if self._task_queue.qsize() > self._adaptive_config["queue_size_limit"]:
            return task_type == TaskType.GUI_UPDATE  # Always accept critical GUI tasks

        return True

    def _schedule_task_execution(self):
        """Schedule the next task for execution."""
        try:
            task = self._task_queue.get_nowait()
        except Empty:
            return

        # Get appropriate thread pool
        pool = self._thread_pools[task.task_type]

        # Submit to thread pool
        future = pool.submit(self._execute_task, task)

        # Track active task
        with self._task_lock:
            self._active_tasks[task.task_id] = task

        # Add completion callback
        future.add_done_callback(lambda f: self._handle_task_completion(task, f))

    def _execute_task(self, task: UnifiedTask) -> Any:
        """Execute a task with full tracking and error handling."""
        task.started_at = time.time()

        try:
            # Emit started signal
            self.task_started.emit(task.task_id, task.task_type.value)

            # Memory check before execution
            if task.memory_limit_mb:
                current_pressure = self.memory_manager.get_memory_pressure()
                if current_pressure == MemoryPressure.CRITICAL:
                    raise RuntimeError("Task cancelled due to critical memory pressure")

            # Execute the function
            result = task.function(*task.args, **task.kwargs)
            task.result = result
            task.completed_at = time.time()

            return result

        except Exception as e:
            task.error = e
            task.completed_at = time.time()
            logger.error(f"Task {task.task_id} failed: {e}")
            raise

    def _handle_task_completion(self, task: UnifiedTask, future):
        """Handle task completion and cleanup."""
        with self._task_lock:
            # Move from active to completed
            self._active_tasks.pop(task.task_id, None)
            self._completed_tasks[task.task_id] = task

            # Update statistics
            if task.error:
                self._stats["tasks_failed"] += 1
                self.task_failed.emit(task.task_id, str(task.error))
            else:
                self._stats["tasks_completed"] += 1
                self.task_completed.emit(task.task_id, task.result)

            self._stats["total_execution_time"] += task.execution_time
            self._stats["total_wait_time"] += task.wait_time

        # Clean up old completed tasks (keep last 50)
        if len(self._completed_tasks) > 50:
            oldest_tasks = sorted(
                self._completed_tasks.items(), key=lambda x: x[1].completed_at or 0
            )
            for task_id, _ in oldest_tasks[:-50]:
                self._completed_tasks.pop(task_id, None)

        # Schedule next task if queue not empty
        if not self._task_queue.empty():
            self._schedule_task_execution()

    def _start_monitoring(self):
        """Start system monitoring thread."""
        import os

        # Skip starting background threads in test mode to prevent threading issues
        if os.environ.get("XPCS_TEST_MODE") == "1":
            return

        self._monitor_thread = threading.Thread(
            target=self._monitor_system, daemon=True, name="unified-monitor"
        )
        self._monitor_thread.start()

    def _monitor_system(self):
        """Monitor system resources and adapt thread pool sizes."""
        last_memory_check = 0
        last_load_balance = 0

        while self._monitoring_active:
            try:
                current_time = time.time()

                # Check memory pressure
                if (
                    current_time - last_memory_check
                    > self._adaptive_config["memory_check_interval"]
                ):
                    memory_pressure = self.memory_manager.get_memory_pressure()
                    self.memory_pressure_changed.emit(memory_pressure.value)
                    last_memory_check = current_time

                # Load balancing
                if (
                    current_time - last_load_balance
                    > self._adaptive_config["load_balance_interval"]
                ):
                    self._balance_thread_pools()
                    last_load_balance = current_time

                time.sleep(1.0)  # Check every second

            except Exception as e:
                logger.warning(f"System monitoring error: {e}")

    def _balance_thread_pools(self):
        """Dynamically balance thread pool sizes based on load."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Emit load signal
        self.load_changed.emit(cpu_percent)

        # Adjust thread pools based on system load
        if cpu_percent > 80:
            # High CPU load - reduce computation threads
            self._adjust_pool_size(TaskType.COMPUTATION, -1)
        elif cpu_percent < 30:
            # Low CPU load - can increase computation threads
            self._adjust_pool_size(TaskType.COMPUTATION, 1)

        if memory_percent > 85:
            # High memory usage - reduce I/O threads
            self._adjust_pool_size(TaskType.DATA_LOADING, -1)

    def _adjust_pool_size(self, task_type: TaskType, delta: int):
        """Safely adjust thread pool size."""
        pool = self._thread_pools[task_type]
        current_workers = pool._max_workers
        new_size = max(1, min(self.max_workers, current_workers + delta))

        if new_size != current_workers:
            # Note: ThreadPoolExecutor doesn't support runtime resizing
            # This is more of a monitoring/logging feature for now
            logger.debug(
                f"Would adjust {task_type.value} pool from {current_workers} to {new_size}"
            )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        with self._task_lock:
            active_count = len(self._active_tasks)
            queue_size = self._task_queue.qsize()

            # Calculate averages
            completed = self._stats["tasks_completed"]
            avg_execution_time = self._stats["total_execution_time"] / max(1, completed)
            avg_wait_time = self._stats["total_wait_time"] / max(1, completed)

        # System resources
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        return {
            # Task statistics
            "tasks_submitted": self._stats["tasks_submitted"],
            "tasks_completed": self._stats["tasks_completed"],
            "tasks_failed": self._stats["tasks_failed"],
            "active_tasks": active_count,
            "queued_tasks": queue_size,
            # Performance metrics
            "avg_execution_time_ms": avg_execution_time * 1000,
            "avg_wait_time_ms": avg_wait_time * 1000,
            "memory_rejected": self._stats["memory_rejected"],
            "timeout_cancelled": self._stats["timeout_cancelled"],
            # System resources
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            # Thread pool status
            "thread_pools": {
                task_type.value: {
                    "max_workers": pool._max_workers,
                    "active_workers": len([t for t in pool._threads if t.is_alive()]),
                }
                for task_type, pool in self._thread_pools.items()
            },
        }

    def shutdown(self):
        """Gracefully shutdown all thread pools and monitoring."""
        logger.info("Shutting down UnifiedThreadingManager")

        # Stop monitoring
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Shutdown thread pools
        for task_type, pool in self._thread_pools.items():
            logger.debug(f"Shutting down {task_type.value} thread pool")
            pool.shutdown(wait=True, timeout=5.0)

        logger.info("UnifiedThreadingManager shutdown complete")


# Global instance management
_global_threading_manager: UnifiedThreadingManager | None = None


def get_unified_threading_manager() -> UnifiedThreadingManager:
    """Get or create the global unified threading manager."""
    global _global_threading_manager
    if _global_threading_manager is None:
        _global_threading_manager = UnifiedThreadingManager()
    return _global_threading_manager


def shutdown_unified_threading():
    """Shutdown the global threading manager."""
    global _global_threading_manager
    if _global_threading_manager:
        _global_threading_manager.shutdown()
        _global_threading_manager = None
