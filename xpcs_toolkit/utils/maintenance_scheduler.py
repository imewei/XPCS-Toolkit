"""
Automated Maintenance Scheduler for XPCS Toolkit CPU Optimizations.

This module provides automated maintenance and tuning capabilities for the optimization
systems including cache cleanup, thread pool optimization, memory management tuning,
and performance parameter adjustment based on system load and usage patterns.

Features:
- Automated cache cleanup and optimization
- Thread pool size adjustment based on workload
- Memory pressure-based cleanup scheduling
- Performance parameter auto-tuning
- Maintenance task scheduling with priorities
- System load-aware maintenance operations
- Integration with existing monitoring and logging systems
"""

from __future__ import annotations

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import heapq

from PySide6.QtCore import QObject, QTimer, Signal

# Import existing optimization systems
from ..threading.enhanced_thread_pool import get_thread_pool_manager
from ..threading.signal_optimization import get_signal_optimizer
from .advanced_cache import get_global_cache
from .adaptive_memory import get_adaptive_memory_manager
from .optimization_health_monitor import get_health_monitor, HealthStatus
from .memory_utils import SystemMemoryMonitor
from .logging_config import get_logger

logger = get_logger(__name__)


class MaintenanceType(Enum):
    """Types of maintenance operations."""

    CACHE_CLEANUP = "cache_cleanup"
    MEMORY_OPTIMIZATION = "memory_optimization"
    THREAD_POOL_TUNING = "thread_pool_tuning"
    SIGNAL_OPTIMIZATION = "signal_optimization"
    SYSTEM_CLEANUP = "system_cleanup"
    PERFORMANCE_TUNING = "performance_tuning"


class MaintenancePriority(Enum):
    """Priority levels for maintenance tasks."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MaintenanceTrigger(Enum):
    """Triggers for maintenance operations."""

    SCHEDULED = "scheduled"
    THRESHOLD_BASED = "threshold_based"
    HEALTH_CHECK = "health_check"
    USER_REQUESTED = "user_requested"
    SYSTEM_EVENT = "system_event"


@dataclass
class MaintenanceTask:
    """Definition of a maintenance task."""

    task_id: str
    task_type: MaintenanceType
    priority: MaintenancePriority
    trigger: MaintenanceTrigger
    description: str
    maintenance_function: Callable[[], bool]  # Returns success status

    # Scheduling
    interval_seconds: float = 0.0  # 0 means one-time task
    last_run: float = 0.0
    next_run: float = 0.0

    # Conditions
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_runtime_seconds: float = 300.0  # 5 minutes max

    # Results tracking
    success_count: int = 0
    failure_count: int = 0
    average_runtime: float = 0.0
    last_error: Optional[str] = None

    def __lt__(self, other) -> bool:
        """For priority queue ordering."""
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value  # Higher priority first
        return self.next_run < other.next_run


@dataclass
class MaintenanceResult:
    """Result of a maintenance operation."""

    task_id: str
    task_type: MaintenanceType
    success: bool
    runtime_seconds: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    actions_taken: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MaintenanceScheduler(QObject):
    """
    Automated maintenance scheduler for CPU optimization systems.
    """

    # Signals
    maintenance_started = Signal(str)  # task_id
    maintenance_completed = Signal(object)  # MaintenanceResult
    maintenance_failed = Signal(str, str)  # task_id, error_message
    critical_maintenance_needed = Signal(str, str)  # component, reason
    maintenance_report_ready = Signal(object)  # Dict[str, MaintenanceResult]

    def __init__(self, check_interval: float = 60.0, parent=None):
        """
        Initialize the maintenance scheduler.

        Args:
            check_interval: How often to check for maintenance tasks (seconds)
            parent: Qt parent object
        """
        super().__init__(parent)

        self.check_interval = check_interval
        self._running = False
        self._lock = threading.RLock()

        # Task management
        self._tasks: Dict[str, MaintenanceTask] = {}
        self._task_queue: List[MaintenanceTask] = []  # Priority queue
        self._running_tasks: Dict[str, threading.Thread] = {}

        # Results tracking
        self._results_history: deque[MaintenanceResult] = deque(maxlen=1000)
        self._last_maintenance_times: Dict[str, float] = {}

        # System monitoring
        self._health_monitor = get_health_monitor()
        self._memory_monitor = SystemMemoryMonitor()

        # Scheduler timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._check_maintenance_schedule)

        # Performance tracking
        self._system_load_history: deque = deque(maxlen=100)
        self._maintenance_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Initialize default maintenance tasks
        self._register_default_tasks()

        logger.info("MaintenanceScheduler initialized")

    def start_scheduler(self) -> None:
        """Start the maintenance scheduler."""
        with self._lock:
            if self._running:
                logger.warning("Maintenance scheduler already running")
                return

            self._running = True
            self._timer.start(int(self.check_interval * 1000))  # Convert to ms
            logger.info("Maintenance scheduler started")

    def stop_scheduler(self) -> None:
        """Stop the maintenance scheduler."""
        with self._lock:
            if not self._running:
                return

            self._running = False
            self._timer.stop()

            # Wait for running tasks to complete (with timeout)
            for task_id, thread in self._running_tasks.items():
                if thread.is_alive():
                    logger.info(
                        f"Waiting for maintenance task '{task_id}' to complete..."
                    )
                    thread.join(timeout=30.0)
                    if thread.is_alive():
                        logger.warning(
                            f"Maintenance task '{task_id}' did not complete in time"
                        )

            self._running_tasks.clear()
            logger.info("Maintenance scheduler stopped")

    def schedule_task(self, task: MaintenanceTask) -> None:
        """
        Schedule a maintenance task.

        Args:
            task: The maintenance task to schedule
        """
        with self._lock:
            # Set initial next_run time
            if task.next_run == 0.0:
                task.next_run = (
                    time.time() + task.interval_seconds
                    if task.interval_seconds > 0
                    else time.time()
                )

            self._tasks[task.task_id] = task
            heapq.heappush(self._task_queue, task)

            logger.debug(f"Scheduled maintenance task: {task.task_id}")

    def run_task_now(self, task_id: str) -> bool:
        """
        Run a specific maintenance task immediately.

        Args:
            task_id: ID of the task to run

        Returns:
            True if task was started, False otherwise
        """
        if task_id not in self._tasks:
            logger.error(f"Maintenance task '{task_id}' not found")
            return False

        if task_id in self._running_tasks:
            logger.warning(f"Maintenance task '{task_id}' is already running")
            return False

        task = self._tasks[task_id]
        return self._run_maintenance_task(task)

    def get_maintenance_report(self) -> Dict[str, Any]:
        """
        Get comprehensive maintenance report.

        Returns:
            Dictionary containing maintenance statistics and history
        """
        with self._lock:
            recent_results = list(self._results_history)[-50:]  # Last 50 results

            report = {
                "scheduler_status": {
                    "running": self._running,
                    "active_tasks": len(self._running_tasks),
                    "scheduled_tasks": len(self._tasks),
                    "check_interval": self.check_interval,
                },
                "recent_maintenance": [],
                "task_statistics": {},
                "system_impact": {
                    "average_runtime": 0.0,
                    "success_rate": 0.0,
                    "total_maintenance_count": len(self._results_history),
                },
            }

            # Add recent results
            for result in recent_results:
                report["recent_maintenance"].append(
                    {
                        "task_id": result.task_id,
                        "task_type": result.task_type.value,
                        "success": result.success,
                        "runtime": result.runtime_seconds,
                        "message": result.message,
                        "timestamp": result.timestamp,
                        "actions_taken": result.actions_taken,
                    }
                )

            # Calculate statistics
            if recent_results:
                successful_tasks = [r for r in recent_results if r.success]
                report["system_impact"]["success_rate"] = len(successful_tasks) / len(
                    recent_results
                )
                report["system_impact"]["average_runtime"] = statistics.mean(
                    [r.runtime_seconds for r in recent_results]
                )

            # Add per-task statistics
            for task_id, task in self._tasks.items():
                report["task_statistics"][task_id] = {
                    "type": task.task_type.value,
                    "priority": task.priority.value,
                    "success_count": task.success_count,
                    "failure_count": task.failure_count,
                    "average_runtime": task.average_runtime,
                    "last_run": task.last_run,
                    "next_run": task.next_run,
                }

            return report

    def _check_maintenance_schedule(self) -> None:
        """Check for maintenance tasks that need to be run."""
        if not self._running:
            return

        current_time = time.time()
        tasks_to_run = []

        with self._lock:
            # Check scheduled tasks
            while self._task_queue and self._task_queue[0].next_run <= current_time:
                task = heapq.heappop(self._task_queue)
                if task.task_id not in self._running_tasks:  # Not already running
                    tasks_to_run.append(task)

        # Check for trigger-based tasks
        health_triggered_tasks = self._check_health_triggers()
        tasks_to_run.extend(health_triggered_tasks)

        # Run eligible tasks
        for task in tasks_to_run:
            if len(self._running_tasks) < 3:  # Limit concurrent maintenance tasks
                self._run_maintenance_task(task)
            else:
                # Re-schedule for later
                task.next_run = current_time + 60.0  # Try again in 1 minute
                with self._lock:
                    heapq.heappush(self._task_queue, task)

    def _check_health_triggers(self) -> List[MaintenanceTask]:
        """Check for maintenance tasks triggered by health conditions."""
        triggered_tasks = []

        try:
            health_report = self._health_monitor.get_health_report()

            for component, result in health_report.items():
                if result.status == HealthStatus.CRITICAL:
                    # Trigger critical maintenance
                    self.critical_maintenance_needed.emit(component, result.message)

                    # Look for maintenance tasks that can address this issue
                    for task in self._tasks.values():
                        if (
                            task.trigger == MaintenanceTrigger.HEALTH_CHECK
                            and task.priority == MaintenancePriority.CRITICAL
                            and component.lower() in task.description.lower()
                        ):
                            triggered_tasks.append(task)

                elif result.status == HealthStatus.WARNING:
                    # Check for preventive maintenance tasks
                    for task in self._tasks.values():
                        if (
                            task.trigger == MaintenanceTrigger.THRESHOLD_BASED
                            and component.lower() in task.description.lower()
                            and time.time() - task.last_run > task.interval_seconds
                        ):
                            triggered_tasks.append(task)

        except Exception as e:
            logger.error(f"Error checking health triggers: {e}")

        return triggered_tasks

    def _run_maintenance_task(self, task: MaintenanceTask) -> bool:
        """
        Run a maintenance task in a separate thread.

        Args:
            task: The maintenance task to run

        Returns:
            True if task was started successfully
        """
        if task.task_id in self._running_tasks:
            return False

        def task_wrapper():
            start_time = time.perf_counter()
            result = None

            try:
                logger.info(f"Starting maintenance task: {task.task_id}")
                self.maintenance_started.emit(task.task_id)

                # Run the maintenance function
                success = task.maintenance_function()
                runtime = time.perf_counter() - start_time

                # Update task statistics
                task.last_run = time.time()
                if success:
                    task.success_count += 1
                else:
                    task.failure_count += 1

                # Update average runtime
                total_runs = task.success_count + task.failure_count
                task.average_runtime = (
                    (task.average_runtime * (total_runs - 1)) + runtime
                ) / total_runs

                # Create result
                result = MaintenanceResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    success=success,
                    runtime_seconds=runtime,
                    message=f"Maintenance task completed {'successfully' if success else 'with errors'}",
                )

                # Schedule next run if recurring
                if task.interval_seconds > 0:
                    task.next_run = time.time() + task.interval_seconds
                    with self._lock:
                        heapq.heappush(self._task_queue, task)

                logger.info(
                    f"Maintenance task '{task.task_id}' completed in {runtime:.2f}s: {'success' if success else 'failed'}"
                )

            except Exception as e:
                runtime = time.perf_counter() - start_time
                error_msg = str(e)

                task.failure_count += 1
                task.last_error = error_msg

                result = MaintenanceResult(
                    task_id=task.task_id,
                    task_type=task.task_type,
                    success=False,
                    runtime_seconds=runtime,
                    message=f"Maintenance task failed: {error_msg}",
                )

                logger.error(f"Maintenance task '{task.task_id}' failed: {error_msg}")
                self.maintenance_failed.emit(task.task_id, error_msg)

            finally:
                # Clean up
                with self._lock:
                    if task.task_id in self._running_tasks:
                        del self._running_tasks[task.task_id]

                if result:
                    self._results_history.append(result)
                    self.maintenance_completed.emit(result)

        # Start task thread
        thread = threading.Thread(
            target=task_wrapper, name=f"maintenance_{task.task_id}"
        )
        thread.daemon = True

        with self._lock:
            self._running_tasks[task.task_id] = thread

        thread.start()
        return True

    def _register_default_tasks(self) -> None:
        """Register default maintenance tasks for all optimization systems."""

        # Cache cleanup task
        self.schedule_task(
            MaintenanceTask(
                task_id="cache_cleanup",
                task_type=MaintenanceType.CACHE_CLEANUP,
                priority=MaintenancePriority.NORMAL,
                trigger=MaintenanceTrigger.SCHEDULED,
                description="Clean up expired cache entries and optimize memory usage",
                maintenance_function=self._cache_cleanup_task,
                interval_seconds=300.0,  # Every 5 minutes
            )
        )

        # Memory optimization task
        self.schedule_task(
            MaintenanceTask(
                task_id="memory_optimization",
                task_type=MaintenanceType.MEMORY_OPTIMIZATION,
                priority=MaintenancePriority.HIGH,
                trigger=MaintenanceTrigger.THRESHOLD_BASED,
                description="Optimize memory usage and clean up unused resources",
                maintenance_function=self._memory_optimization_task,
                interval_seconds=600.0,  # Every 10 minutes
            )
        )

        # Thread pool tuning task
        self.schedule_task(
            MaintenanceTask(
                task_id="thread_pool_tuning",
                task_type=MaintenanceType.THREAD_POOL_TUNING,
                priority=MaintenancePriority.NORMAL,
                trigger=MaintenanceTrigger.SCHEDULED,
                description="Adjust thread pool sizes based on workload patterns",
                maintenance_function=self._thread_pool_tuning_task,
                interval_seconds=1800.0,  # Every 30 minutes
            )
        )

        # Signal optimization task
        self.schedule_task(
            MaintenanceTask(
                task_id="signal_optimization",
                task_type=MaintenanceType.SIGNAL_OPTIMIZATION,
                priority=MaintenancePriority.LOW,
                trigger=MaintenanceTrigger.SCHEDULED,
                description="Optimize signal connections and clear unused connections",
                maintenance_function=self._signal_optimization_task,
                interval_seconds=900.0,  # Every 15 minutes
            )
        )

        # Critical memory cleanup task (triggered by health checks)
        self.schedule_task(
            MaintenanceTask(
                task_id="critical_memory_cleanup",
                task_type=MaintenanceType.SYSTEM_CLEANUP,
                priority=MaintenancePriority.CRITICAL,
                trigger=MaintenanceTrigger.HEALTH_CHECK,
                description="Emergency memory cleanup when system is under pressure",
                maintenance_function=self._critical_memory_cleanup_task,
                interval_seconds=0.0,  # One-time, triggered by health checks
            )
        )

    # Maintenance task implementations

    def _cache_cleanup_task(self) -> bool:
        """Perform cache cleanup and optimization."""
        try:
            actions_taken = []

            # Get global cache
            cache = get_global_cache()
            if cache:
                # Get cache statistics before cleanup
                stats_before = cache.get_statistics()

                # Perform cleanup
                cleanup_stats = cache.cleanup()

                if cleanup_stats.get("entries_removed", 0) > 0:
                    actions_taken.append(
                        f"Removed {cleanup_stats['entries_removed']} expired cache entries"
                    )

                if cleanup_stats.get("memory_freed_mb", 0) > 0:
                    actions_taken.append(
                        f"Freed {cleanup_stats['memory_freed_mb']:.1f} MB of cache memory"
                    )

                # Optimize cache structure if needed
                if stats_before.get("fragmentation", 0.0) > 0.3:
                    cache.optimize()
                    actions_taken.append(
                        "Optimized cache structure to reduce fragmentation"
                    )

            # Clean up computation cache
            comp_cache = None  # Would get computation cache instance
            if comp_cache:
                # Similar cleanup for computation cache
                pass

            logger.info(
                f"Cache cleanup completed: {'; '.join(actions_taken) if actions_taken else 'No cleanup needed'}"
            )
            return True

        except Exception as e:
            logger.error(f"Cache cleanup task failed: {e}")
            return False

    def _memory_optimization_task(self) -> bool:
        """Perform memory optimization and cleanup."""
        try:
            actions_taken = []

            # Get memory stats
            memory_stats = self._memory_monitor.get_memory_stats()
            memory_percent = memory_stats.get("memory_percent", 0.0)

            # Check if memory optimization is needed
            if memory_percent > 70.0:
                # Get adaptive memory manager
                memory_manager = get_adaptive_memory_manager()
                if memory_manager:
                    # Trigger memory optimization
                    optimization_result = memory_manager.optimize_memory_usage()

                    if optimization_result.get("memory_freed_mb", 0) > 0:
                        freed_mb = optimization_result["memory_freed_mb"]
                        actions_taken.append(
                            f"Freed {freed_mb:.1f} MB through adaptive memory optimization"
                        )

                    if optimization_result.get("caches_resized", 0) > 0:
                        resized = optimization_result["caches_resized"]
                        actions_taken.append(
                            f"Resized {resized} cache(s) to reduce memory pressure"
                        )

                # Force garbage collection if memory is very high
                if memory_percent > 85.0:
                    import gc

                    collected = gc.collect()
                    actions_taken.append(
                        f"Forced garbage collection, collected {collected} objects"
                    )

            else:
                actions_taken.append(
                    "Memory usage within normal range, no optimization needed"
                )

            logger.info(f"Memory optimization completed: {'; '.join(actions_taken)}")
            return True

        except Exception as e:
            logger.error(f"Memory optimization task failed: {e}")
            return False

    def _thread_pool_tuning_task(self) -> bool:
        """Tune thread pool sizes based on workload patterns."""
        try:
            actions_taken = []

            # Get thread pool manager
            thread_manager = get_thread_pool_manager()
            if not thread_manager:
                return False

            # Get current statistics
            stats = thread_manager.get_pool_statistics()
            current_threads = stats.get("max_threads", 0)
            stats.get("active_threads", 0)
            queued_tasks = stats.get("queued_tasks", 0)
            avg_utilization = stats.get("avg_utilization", 0.0)

            # Calculate optimal thread count based on patterns
            optimal_threads = current_threads

            # If consistently high utilization, increase thread count
            if avg_utilization > 0.8 and queued_tasks > 10:
                optimal_threads = min(current_threads + 2, 20)  # Cap at 20 threads
                actions_taken.append(
                    f"Increased thread pool size from {current_threads} to {optimal_threads}"
                )

            # If consistently low utilization, decrease thread count
            elif avg_utilization < 0.3 and queued_tasks == 0 and current_threads > 4:
                optimal_threads = max(current_threads - 1, 4)  # Minimum 4 threads
                actions_taken.append(
                    f"Decreased thread pool size from {current_threads} to {optimal_threads}"
                )

            # Apply changes if needed
            if optimal_threads != current_threads:
                thread_manager.set_pool_size(optimal_threads)
            else:
                actions_taken.append("Thread pool size optimal, no changes needed")

            logger.info(f"Thread pool tuning completed: {'; '.join(actions_taken)}")
            return True

        except Exception as e:
            logger.error(f"Thread pool tuning task failed: {e}")
            return False

    def _signal_optimization_task(self) -> bool:
        """Optimize signal connections and clear unused connections."""
        try:
            actions_taken = []

            # Get signal optimizer
            optimizer = get_signal_optimizer()
            if not optimizer:
                return False

            # Perform signal optimization
            optimization_result = optimizer.optimize_connections()

            if optimization_result.get("connections_cleaned", 0) > 0:
                cleaned = optimization_result["connections_cleaned"]
                actions_taken.append(f"Cleaned {cleaned} unused signal connections")

            if optimization_result.get("cache_optimized", False):
                actions_taken.append("Optimized signal connection cache")

            # Check batching efficiency and adjust if needed
            stats = optimizer.get_optimization_stats()
            batching_efficiency = stats.get("batching_efficiency", 0.0)

            if batching_efficiency < 0.5:
                optimizer.adjust_batching_parameters()
                actions_taken.append(
                    "Adjusted signal batching parameters to improve efficiency"
                )

            if not actions_taken:
                actions_taken.append("Signal optimization complete, no changes needed")

            logger.info(f"Signal optimization completed: {'; '.join(actions_taken)}")
            return True

        except Exception as e:
            logger.error(f"Signal optimization task failed: {e}")
            return False

    def _critical_memory_cleanup_task(self) -> bool:
        """Emergency memory cleanup when system is under severe memory pressure."""
        try:
            actions_taken = []

            # Get current memory state
            memory_stats = self._memory_monitor.get_memory_stats()
            memory_percent = memory_stats.get("memory_percent", 0.0)

            if memory_percent < 85.0:
                actions_taken.append(
                    "Memory pressure resolved, no emergency cleanup needed"
                )
                return True

            logger.warning(
                f"Performing critical memory cleanup - memory usage at {memory_percent:.1f}%"
            )

            # Aggressive cache clearing
            cache = get_global_cache()
            if cache:
                # Clear non-essential cached data
                cleared_mb = cache.emergency_cleanup()
                if cleared_mb > 0:
                    actions_taken.append(
                        f"Emergency cache cleanup freed {cleared_mb:.1f} MB"
                    )

            # Force adaptive memory manager to free up memory
            memory_manager = get_adaptive_memory_manager()
            if memory_manager:
                freed_mb = memory_manager.emergency_memory_release()
                if freed_mb > 0:
                    actions_taken.append(
                        f"Emergency memory release freed {freed_mb:.1f} MB"
                    )

            # Force garbage collection multiple times
            import gc

            total_collected = 0
            for _ in range(3):
                collected = gc.collect()
                total_collected += collected
                time.sleep(0.1)

            if total_collected > 0:
                actions_taken.append(
                    f"Aggressive garbage collection freed {total_collected} objects"
                )

            # Final memory check
            final_memory_stats = self._memory_monitor.get_memory_stats()
            final_memory_percent = final_memory_stats.get("memory_percent", 0.0)

            if final_memory_percent < memory_percent:
                reduction = memory_percent - final_memory_percent
                actions_taken.append(
                    f"Reduced memory usage by {reduction:.1f}% (now at {final_memory_percent:.1f}%)"
                )

            logger.warning(
                f"Critical memory cleanup completed: {'; '.join(actions_taken)}"
            )
            return True

        except Exception as e:
            logger.error(f"Critical memory cleanup task failed: {e}")
            return False


# Global instance
_maintenance_scheduler_instance: Optional[MaintenanceScheduler] = None
_scheduler_lock = threading.Lock()


def get_maintenance_scheduler() -> MaintenanceScheduler:
    """Get the global maintenance scheduler instance."""
    global _maintenance_scheduler_instance

    if _maintenance_scheduler_instance is None:
        with _scheduler_lock:
            if _maintenance_scheduler_instance is None:
                _maintenance_scheduler_instance = MaintenanceScheduler()

    return _maintenance_scheduler_instance


def start_maintenance_scheduler() -> None:
    """Start the global maintenance scheduler."""
    scheduler = get_maintenance_scheduler()
    scheduler.start_scheduler()


def stop_maintenance_scheduler() -> None:
    """Stop the global maintenance scheduler."""
    scheduler = get_maintenance_scheduler()
    scheduler.stop_scheduler()
