"""
Threading Performance Optimization for XPCS Toolkit

This module optimizes the existing threading architecture to address performance
bottlenecks in GUI responsiveness, worker coordination, and resource management.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import PriorityQueue, Empty
from typing import Any, Callable, Dict, List, Optional, Set
from threading import Lock, Event

import psutil

from .logging_config import get_logger
from .memory_manager import get_memory_manager, MemoryPressure

logger = get_logger(__name__)


class WorkerPriority(Enum):
    """Priority levels for worker tasks."""
    CRITICAL = 1      # UI responsiveness critical
    HIGH = 2          # User-triggered operations
    NORMAL = 3        # Background processing
    LOW = 4           # Cleanup, maintenance


class ThreadPoolType(Enum):
    """Types of specialized thread pools."""
    GUI_CRITICAL = "gui_critical"      # High priority, minimal workers
    COMPUTATION = "computation"        # CPU-intensive work
    IO_BOUND = "io_bound"             # File I/O operations
    PLOTTING = "plotting"             # Visualization tasks


@dataclass
class WorkerMetrics:
    """Metrics for worker performance monitoring."""
    worker_id: str
    thread_id: int
    start_time: float
    end_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    success: bool = True
    error_message: str = ""
    priority: WorkerPriority = WorkerPriority.NORMAL


@dataclass
class ThreadPoolStats:
    """Statistics for thread pool performance."""
    pool_type: ThreadPoolType
    active_workers: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    memory_pressure_events: int = 0
    queue_length: int = 0


class AdaptiveThreadPoolManager:
    """
    Manages adaptive thread pools that adjust based on system resources and workload.
    """

    def __init__(self, max_workers_per_pool: Optional[int] = None):
        self.max_workers_per_pool = max_workers_per_pool or min(8, (psutil.cpu_count() or 1) + 4)

        # Specialized thread pools
        self.pools: Dict[ThreadPoolType, ThreadPoolExecutor] = {}
        self.pool_stats: Dict[ThreadPoolType, ThreadPoolStats] = {}
        self.worker_metrics: List[WorkerMetrics] = []

        # Resource monitoring
        self.memory_manager = get_memory_manager()
        self._resource_monitor_thread = None
        self._shutdown_event = Event()
        self._stats_lock = Lock()

        # Initialize pools
        self._initialize_pools()
        self._start_resource_monitoring()

        logger.info(f"AdaptiveThreadPoolManager initialized with {self.max_workers_per_pool} max workers per pool")

    def _initialize_pools(self):
        """Initialize specialized thread pools with appropriate sizes."""
        pool_configs = {
            ThreadPoolType.GUI_CRITICAL: 2,  # Minimal for UI responsiveness
            ThreadPoolType.COMPUTATION: self.max_workers_per_pool,  # Full CPU utilization
            ThreadPoolType.IO_BOUND: min(4, self.max_workers_per_pool),  # I/O doesn't need many threads
            ThreadPoolType.PLOTTING: min(3, self.max_workers_per_pool)   # Visualization work
        }

        for pool_type, max_workers in pool_configs.items():
            self.pools[pool_type] = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix=f"xpcs_{pool_type.value}"
            )
            self.pool_stats[pool_type] = ThreadPoolStats(pool_type=pool_type)

    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        self._resource_monitor_thread = threading.Thread(
            target=self._resource_monitor_worker,
            daemon=True,
            name="resource_monitor"
        )
        self._resource_monitor_thread.start()

    def _resource_monitor_worker(self):
        """Background worker that monitors system resources and adjusts pools."""
        while not self._shutdown_event.wait(timeout=5.0):
            try:
                # Check memory pressure
                memory_pressure = self.memory_manager.get_memory_pressure()

                if memory_pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
                    self._handle_memory_pressure(memory_pressure)

                # Update pool statistics
                self._update_pool_statistics()

                # Log periodic statistics
                if int(time.time()) % 60 == 0:  # Every minute
                    self._log_pool_statistics()

            except Exception as e:
                logger.warning(f"Resource monitor error: {e}")

    def _handle_memory_pressure(self, pressure: MemoryPressure):
        """Handle high memory pressure by reducing thread pool sizes."""
        if pressure == MemoryPressure.CRITICAL:
            # Drastically reduce pool sizes
            reduction_factor = 0.5
            logger.warning("Critical memory pressure: reducing thread pool sizes by 50%")
        else:
            # Moderate reduction
            reduction_factor = 0.75
            logger.info("High memory pressure: reducing thread pool sizes by 25%")

        with self._stats_lock:
            for pool_type, stats in self.pool_stats.items():
                stats.memory_pressure_events += 1

        # Note: ThreadPoolExecutor doesn't support dynamic resizing
        # In a production system, we would implement custom worker management

    def submit_task(self, pool_type: ThreadPoolType, func: Callable, *args,
                   priority: WorkerPriority = WorkerPriority.NORMAL,
                   worker_id: Optional[str] = None, **kwargs):
        """
        Submit a task to the appropriate thread pool.

        Parameters
        ----------
        pool_type : ThreadPoolType
            Type of thread pool to use
        func : Callable
            Function to execute
        priority : WorkerPriority
            Task priority level
        worker_id : str, optional
            Unique identifier for the worker
        *args, **kwargs
            Arguments for the function

        Returns
        -------
        concurrent.futures.Future
            Future object for the submitted task
        """
        if pool_type not in self.pools:
            raise ValueError(f"Unknown pool type: {pool_type}")

        pool = self.pools[pool_type]
        worker_id = worker_id or f"{pool_type.value}_{int(time.time()*1000)}"

        # Wrap the function to collect metrics
        wrapped_func = self._wrap_function_with_metrics(
            func, worker_id, priority, pool_type
        )

        # Submit to pool
        future = pool.submit(wrapped_func, *args, **kwargs)

        # Update statistics
        with self._stats_lock:
            self.pool_stats[pool_type].queue_length += 1

        logger.debug(f"Submitted task {worker_id} to {pool_type.value} pool")
        return future

    def _wrap_function_with_metrics(self, func: Callable, worker_id: str,
                                   priority: WorkerPriority, pool_type: ThreadPoolType):
        """Wrap function to collect performance metrics."""
        def wrapped_func(*args, **kwargs):
            start_time = time.time()
            thread_id = threading.get_ident()

            # Create metrics object
            metrics = WorkerMetrics(
                worker_id=worker_id,
                thread_id=thread_id,
                start_time=start_time,
                priority=priority
            )

            # Monitor memory usage
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

            try:
                # Update active workers count
                with self._stats_lock:
                    self.pool_stats[pool_type].active_workers += 1

                # Execute the function
                result = func(*args, **kwargs)

                # Success metrics
                metrics.success = True

                return result

            except Exception as e:
                # Error metrics
                metrics.success = False
                metrics.error_message = str(e)
                logger.warning(f"Worker {worker_id} failed: {e}")
                raise

            finally:
                # Completion metrics
                end_time = time.time()
                metrics.end_time = end_time

                memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
                metrics.memory_usage_mb = memory_after - memory_before

                # Update statistics
                with self._stats_lock:
                    stats = self.pool_stats[pool_type]
                    stats.active_workers -= 1
                    stats.queue_length -= 1

                    if metrics.success:
                        stats.completed_tasks += 1
                        execution_time = end_time - start_time
                        # Update average execution time
                        if stats.average_execution_time == 0:
                            stats.average_execution_time = execution_time
                        else:
                            stats.average_execution_time = (
                                stats.average_execution_time * 0.9 + execution_time * 0.1
                            )
                    else:
                        stats.failed_tasks += 1

                # Store metrics
                self.worker_metrics.append(metrics)

                # Limit metrics history
                if len(self.worker_metrics) > 1000:
                    self.worker_metrics = self.worker_metrics[-500:]

                logger.debug(
                    f"Worker {worker_id} completed: {end_time - start_time:.3f}s, "
                    f"memory: {metrics.memory_usage_mb:+.1f}MB"
                )

        return wrapped_func

    def _update_pool_statistics(self):
        """Update pool statistics."""
        # This method can be extended to collect additional metrics
        pass

    def _log_pool_statistics(self):
        """Log current pool statistics."""
        with self._stats_lock:
            for pool_type, stats in self.pool_stats.items():
                logger.info(
                    f"Pool {pool_type.value}: active={stats.active_workers}, "
                    f"completed={stats.completed_tasks}, failed={stats.failed_tasks}, "
                    f"avg_time={stats.average_execution_time:.3f}s, "
                    f"queue={stats.queue_length}"
                )

    def get_pool_statistics(self) -> Dict[ThreadPoolType, ThreadPoolStats]:
        """Get current pool statistics."""
        with self._stats_lock:
            return self.pool_stats.copy()

    def shutdown(self, wait: bool = True):
        """Shutdown all thread pools."""
        logger.info("Shutting down AdaptiveThreadPoolManager")

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for resource monitor
        if self._resource_monitor_thread:
            self._resource_monitor_thread.join(timeout=5.0)

        # Shutdown pools
        for pool_type, pool in self.pools.items():
            logger.debug(f"Shutting down {pool_type.value} pool")
            pool.shutdown(wait=wait)


class SmartTaskScheduler:
    """
    Intelligent task scheduler that optimizes task execution based on system state.
    """

    def __init__(self, thread_manager: AdaptiveThreadPoolManager):
        self.thread_manager = thread_manager
        self.memory_manager = get_memory_manager()
        self.pending_tasks: PriorityQueue = PriorityQueue()
        self.task_dependencies: Dict[str, Set[str]] = {}
        self._scheduler_thread = None
        self._scheduler_active = Event()
        self._shutdown_event = Event()

        self._start_scheduler()

    def _start_scheduler(self):
        """Start the task scheduler thread."""
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_worker,
            daemon=True,
            name="task_scheduler"
        )
        self._scheduler_active.set()
        self._scheduler_thread.start()

    def _scheduler_worker(self):
        """Background scheduler that manages task execution."""
        while not self._shutdown_event.wait(timeout=0.1):
            try:
                if not self._scheduler_active.is_set():
                    continue

                # Check if we should process tasks
                if self._should_process_tasks():
                    self._process_pending_tasks()

            except Exception as e:
                logger.warning(f"Task scheduler error: {e}")

    def _should_process_tasks(self) -> bool:
        """Determine if we should process pending tasks based on system state."""
        # Check memory pressure
        memory_pressure = self.memory_manager.get_memory_pressure()
        if memory_pressure == MemoryPressure.CRITICAL:
            return False

        # Check if any pools are overloaded
        pool_stats = self.thread_manager.get_pool_statistics()
        for stats in pool_stats.values():
            if stats.active_workers >= stats.queue_length * 2:  # Overloaded
                return False

        return True

    def _process_pending_tasks(self):
        """Process pending tasks from the priority queue."""
        try:
            # Get the highest priority task
            priority, task_id, task_info = self.pending_tasks.get_nowait()

            # Check dependencies
            if self._dependencies_satisfied(task_id):
                # Submit the task
                self._submit_task(task_info)
            else:
                # Re-queue the task
                self.pending_tasks.put((priority, task_id, task_info))

        except Empty:
            # No pending tasks
            pass

    def _dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied."""
        dependencies = self.task_dependencies.get(task_id, set())
        # Simplified dependency checking - in reality would track completed tasks
        return len(dependencies) == 0

    def _submit_task(self, task_info: Dict[str, Any]):
        """Submit a task to the appropriate thread pool."""
        pool_type = task_info['pool_type']
        func = task_info['func']
        args = task_info.get('args', ())
        kwargs = task_info.get('kwargs', {})
        priority = task_info.get('priority', WorkerPriority.NORMAL)
        worker_id = task_info.get('worker_id')

        future = self.thread_manager.submit_task(
            pool_type, func, *args, priority=priority, worker_id=worker_id, **kwargs
        )

        # Store future for dependency tracking
        task_info['future'] = future

    def schedule_task(self, task_id: str, pool_type: ThreadPoolType, func: Callable,
                     *args, priority: WorkerPriority = WorkerPriority.NORMAL,
                     dependencies: Optional[Set[str]] = None, **kwargs):
        """
        Schedule a task for execution.

        Parameters
        ----------
        task_id : str
            Unique identifier for the task
        pool_type : ThreadPoolType
            Type of thread pool to use
        func : Callable
            Function to execute
        priority : WorkerPriority
            Task priority
        dependencies : Set[str], optional
            Set of task IDs this task depends on
        """
        task_info = {
            'task_id': task_id,
            'pool_type': pool_type,
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'worker_id': task_id
        }

        # Store dependencies
        if dependencies:
            self.task_dependencies[task_id] = dependencies

        # Add to priority queue (lower number = higher priority)
        self.pending_tasks.put((priority.value, task_id, task_info))

        logger.debug(f"Scheduled task {task_id} with priority {priority.name}")

    def pause_scheduling(self):
        """Pause task scheduling."""
        self._scheduler_active.clear()
        logger.info("Task scheduling paused")

    def resume_scheduling(self):
        """Resume task scheduling."""
        self._scheduler_active.set()
        logger.info("Task scheduling resumed")

    def shutdown(self):
        """Shutdown the task scheduler."""
        logger.info("Shutting down SmartTaskScheduler")
        self._shutdown_event.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5.0)


class GUIResponsivenessOptimizer:
    """
    Optimizes GUI responsiveness by intelligently managing blocking operations.
    """

    def __init__(self, thread_manager: AdaptiveThreadPoolManager):
        self.thread_manager = thread_manager
        self.responsiveness_threshold_ms = 16  # 60 FPS target
        self.blocking_operations = []
        self._last_ui_update = time.time()

    def ensure_ui_responsiveness(self, operation_name: str):
        """
        Context manager that ensures UI remains responsive during operations.

        Parameters
        ----------
        operation_name : str
            Name of the operation for logging
        """
        return UIResponsivenessContext(self, operation_name)

    def check_ui_responsiveness(self):
        """Check if UI responsiveness is at risk."""
        current_time = time.time()
        time_since_update = (current_time - self._last_ui_update) * 1000  # ms

        if time_since_update > self.responsiveness_threshold_ms:
            logger.warning(f"UI responsiveness risk: {time_since_update:.1f}ms since last update")
            return False

        return True

    def yield_to_ui(self):
        """Yield execution to allow UI updates."""
        # This would be called periodically during long operations
        current_time = time.time()
        time_since_update = (current_time - self._last_ui_update) * 1000

        if time_since_update > self.responsiveness_threshold_ms:
            # Force UI processing
            from PySide6.QtWidgets import QApplication
            if QApplication.instance():
                QApplication.processEvents()
            self._last_ui_update = time.time()

    def record_ui_update(self):
        """Record that a UI update occurred."""
        self._last_ui_update = time.time()


class UIResponsivenessContext:
    """Context manager for ensuring UI responsiveness."""

    def __init__(self, optimizer: GUIResponsivenessOptimizer, operation_name: str):
        self.optimizer = optimizer
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if duration > 0.1:  # Log operations > 100ms
            logger.debug(f"UI operation '{self.operation_name}' took {duration*1000:.1f}ms")


# Global instances
_thread_manager = None
_task_scheduler = None
_gui_optimizer = None


def get_thread_manager() -> AdaptiveThreadPoolManager:
    """Get global thread manager instance."""
    global _thread_manager
    if _thread_manager is None:
        _thread_manager = AdaptiveThreadPoolManager()
    return _thread_manager


def get_task_scheduler() -> SmartTaskScheduler:
    """Get global task scheduler instance."""
    global _task_scheduler
    if _task_scheduler is None:
        _task_scheduler = SmartTaskScheduler(get_thread_manager())
    return _task_scheduler


def get_gui_optimizer() -> GUIResponsivenessOptimizer:
    """Get global GUI optimizer instance."""
    global _gui_optimizer
    if _gui_optimizer is None:
        _gui_optimizer = GUIResponsivenessOptimizer(get_thread_manager())
    return _gui_optimizer


# Convenience functions
def submit_computation_task(func: Callable, *args, priority: WorkerPriority = WorkerPriority.NORMAL, **kwargs):
    """Submit a computation-intensive task."""
    return get_thread_manager().submit_task(ThreadPoolType.COMPUTATION, func, *args, priority=priority, **kwargs)


def submit_io_task(func: Callable, *args, priority: WorkerPriority = WorkerPriority.NORMAL, **kwargs):
    """Submit an I/O-bound task."""
    return get_thread_manager().submit_task(ThreadPoolType.IO_BOUND, func, *args, priority=priority, **kwargs)


def submit_plot_task(func: Callable, *args, priority: WorkerPriority = WorkerPriority.HIGH, **kwargs):
    """Submit a plotting task."""
    return get_thread_manager().submit_task(ThreadPoolType.PLOTTING, func, *args, priority=priority, **kwargs)


def ensure_ui_responsive(operation_name: str):
    """Context manager for ensuring UI responsiveness."""
    return get_gui_optimizer().ensure_ui_responsiveness(operation_name)


def shutdown_threading_system():
    """Shutdown the entire threading optimization system."""
    global _thread_manager, _task_scheduler, _gui_optimizer

    if _task_scheduler:
        _task_scheduler.shutdown()
        _task_scheduler = None

    if _thread_manager:
        _thread_manager.shutdown()
        _thread_manager = None

    _gui_optimizer = None

    logger.info("Threading optimization system shutdown complete")