"""
Enhanced thread pool management system for XPCS Toolkit.

This module provides:
- Dynamic thread pool sizing based on system resources and workload
- Smart load balancing and worker distribution
- Thread health monitoring and automatic recovery
- Resource-aware scheduling with priority queues
- Performance analytics and optimization
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, PriorityQueue
from typing import Any

import psutil
from PySide6.QtCore import (
    QMutex,
    QMutexLocker,
    QObject,
    QRunnable,
    QThreadPool,
    QTimer,
    Signal,
)

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ThreadPoolHealth(Enum):
    """Thread pool health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    CRITICAL = "critical"


class LoadBalanceStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    PRIORITY_FIRST = "priority_first"
    RESOURCE_AWARE = "resource_aware"


@dataclass
class ThreadPoolMetrics:
    """Thread pool performance metrics."""

    pool_id: str
    active_threads: int = 0
    max_threads: int = 0
    queued_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    health_status: ThreadPoolHealth = ThreadPoolHealth.HEALTHY
    last_updated: float = field(default_factory=time.perf_counter)

    def calculate_load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0+)."""
        if self.max_threads <= 0:
            return 0.0
        return (self.active_threads + self.queued_tasks * 0.5) / self.max_threads


@dataclass
class TaskInfo:
    """Information about a queued task."""

    task: QRunnable
    priority: int = 5  # Lower numbers = higher priority
    submission_time: float = field(default_factory=time.perf_counter)
    estimated_duration: float = 1.0  # Estimated execution time in seconds
    resource_requirements: dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2

    def __lt__(self, other):
        """Priority queue comparison."""
        return self.priority < other.priority


class SmartThreadPool(QThreadPool):
    """
    Enhanced thread pool with dynamic sizing and intelligent scheduling.
    """

    # Signals for monitoring
    pool_metrics_updated = Signal(object)  # ThreadPoolMetrics
    health_status_changed = Signal(str, str)  # pool_id, new_health_status

    def __init__(self, pool_id: str = "default", parent=None):
        super().__init__(parent)
        self.pool_id = pool_id

        # Dynamic sizing parameters
        self._base_thread_count = psutil.cpu_count()
        self._max_thread_limit = min(
            32, self._base_thread_count * 3
        )  # Reasonable upper limit
        self._min_thread_limit = max(2, self._base_thread_count // 2)

        # Initialize with smart defaults
        initial_threads = self._calculate_optimal_thread_count()
        self.setMaxThreadCount(initial_threads)

        # Task queuing and priority management
        self._priority_queue = PriorityQueue()
        self._active_tasks: set[QRunnable] = set()
        self._task_history: deque = deque(maxlen=1000)  # Keep recent task history

        # Performance monitoring
        self._metrics = ThreadPoolMetrics(pool_id=pool_id, max_threads=initial_threads)
        self._resource_monitor_timer = QTimer()
        self._resource_monitor_timer.timeout.connect(self._update_metrics)
        self._resource_monitor_timer.start(2000)  # Update every 2 seconds

        # Load balancing
        self._load_balance_strategy = LoadBalanceStrategy.RESOURCE_AWARE
        self._last_rebalance_time = time.perf_counter()
        self._rebalance_interval = 10.0  # Rebalance every 10 seconds

        # Thread management
        self._thread_adjustment_timer = QTimer()
        self._thread_adjustment_timer.timeout.connect(self._adjust_thread_count)
        self._thread_adjustment_timer.start(5000)  # Adjust every 5 seconds

        # Statistics
        self._total_tasks_submitted = 0
        self._total_tasks_completed = 0
        self._total_execution_time = 0.0

        self._mutex = QMutex()

        logger.info(
            f"SmartThreadPool '{pool_id}' initialized with {initial_threads} threads "
            f"(range: {self._min_thread_limit}-{self._max_thread_limit})"
        )

    def _calculate_optimal_thread_count(self) -> int:
        """Calculate optimal thread count based on system resources."""
        try:
            # Get system resources
            cpu_count = psutil.cpu_count()
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_usage = psutil.cpu_percent(interval=0.1)

            # Base calculation on CPU count
            optimal_threads = cpu_count

            # Adjust based on available memory
            if memory_gb < 4:
                optimal_threads = max(2, cpu_count // 2)
            elif memory_gb < 8:
                optimal_threads = cpu_count
            elif memory_gb >= 16:
                optimal_threads = min(cpu_count + 4, self._max_thread_limit)

            # Adjust based on current CPU usage
            if cpu_usage > 80:
                optimal_threads = max(self._min_thread_limit, optimal_threads - 2)
            elif cpu_usage < 30:
                optimal_threads = min(self._max_thread_limit, optimal_threads + 2)

            return max(
                self._min_thread_limit, min(self._max_thread_limit, optimal_threads)
            )

        except Exception as e:
            logger.warning(f"Error calculating optimal thread count: {e}")
            return self._base_thread_count

    def start_with_priority(
        self,
        task: QRunnable,
        priority: int = 5,
        estimated_duration: float = 1.0,
        resource_requirements: dict[str, float] | None = None,
    ):
        """Start a task with priority and resource information."""
        with QMutexLocker(self._mutex):
            task_info = TaskInfo(
                task=task,
                priority=priority,
                estimated_duration=estimated_duration,
                resource_requirements=resource_requirements or {},
            )

            # Check if we can start immediately
            if self.activeThreadCount() < self.maxThreadCount():
                self._start_task_immediately(task_info)
            else:
                # Queue for later execution
                self._priority_queue.put(task_info)
                self._metrics.queued_tasks = self._priority_queue.qsize()

            self._total_tasks_submitted += 1

    def _start_task_immediately(self, task_info: TaskInfo):
        """Start a task immediately."""
        # Wrap the task to track execution
        wrapped_task = self._wrap_task_for_monitoring(task_info)

        # Start the task
        super().start(wrapped_task)
        self._active_tasks.add(task_info.task)
        self._metrics.active_threads = self.activeThreadCount()

    def _wrap_task_for_monitoring(self, task_info: TaskInfo) -> QRunnable:
        """Wrap a task to add monitoring and cleanup."""
        original_run = task_info.task.run
        start_time = time.perf_counter()

        def monitored_run():
            try:
                # Execute original task
                result = original_run()

                # Track successful completion
                execution_time = time.perf_counter() - start_time
                self._on_task_completed(task_info, execution_time, success=True)

                return result

            except Exception as e:
                # Track failed execution
                execution_time = time.perf_counter() - start_time
                self._on_task_completed(
                    task_info, execution_time, success=False, error=e
                )

                # Handle retry logic
                if task_info.retry_count < task_info.max_retries:
                    task_info.retry_count += 1
                    logger.info(
                        f"Retrying task (attempt {task_info.retry_count}/{task_info.max_retries})"
                    )
                    # Re-queue with lower priority
                    task_info.priority += 1
                    with QMutexLocker(self._mutex):
                        self._priority_queue.put(task_info)
                else:
                    logger.error(
                        f"Task failed after {task_info.max_retries} retries: {e}"
                    )

                raise

        task_info.task.run = monitored_run
        return task_info.task

    def _on_task_completed(
        self,
        task_info: TaskInfo,
        execution_time: float,
        success: bool,
        error: Exception | None = None,
    ):
        """Handle task completion."""
        with QMutexLocker(self._mutex):
            # Remove from active tasks
            self._active_tasks.discard(task_info.task)

            # Update metrics
            self._total_tasks_completed += 1
            self._total_execution_time += execution_time

            if success:
                self._metrics.completed_tasks += 1
            else:
                self._metrics.failed_tasks += 1

            # Update average execution time
            if self._total_tasks_completed > 0:
                self._metrics.avg_execution_time = (
                    self._total_execution_time / self._total_tasks_completed
                )

            # Add to task history
            self._task_history.append(
                {
                    "task_info": task_info,
                    "execution_time": execution_time,
                    "success": success,
                    "completion_time": time.perf_counter(),
                }
            )

            # Update active thread count
            self._metrics.active_threads = self.activeThreadCount()

            # Try to start queued tasks
            self._process_queued_tasks()

    def _process_queued_tasks(self):
        """Process queued tasks if threads are available."""
        while (
            self.activeThreadCount() < self.maxThreadCount()
            and not self._priority_queue.empty()
        ):
            try:
                task_info = self._priority_queue.get_nowait()
                self._start_task_immediately(task_info)
                self._metrics.queued_tasks = self._priority_queue.qsize()
            except Empty:
                break

    def _adjust_thread_count(self):
        """Dynamically adjust thread count based on load and performance."""
        current_time = time.perf_counter()

        # Only adjust if enough time has passed
        if current_time - self._last_rebalance_time < self._rebalance_interval:
            return

        optimal_count = self._calculate_optimal_thread_count()
        current_count = self.maxThreadCount()

        # Calculate load factor and queue backlog
        load_factor = self._metrics.calculate_load_factor()
        queue_size = self._priority_queue.qsize()

        # Decision logic for thread adjustment
        should_increase = (
            load_factor > 0.8  # High load
            or queue_size > current_count * 2  # Large backlog
            or (
                self._metrics.avg_execution_time > 3.0 and queue_size > 0
            )  # Slow tasks with backlog
        )

        should_decrease = (
            load_factor < 0.3  # Low load
            and queue_size == 0  # No backlog
            and self._metrics.avg_execution_time < 1.0  # Fast tasks
        )

        if should_increase and current_count < self._max_thread_limit:
            new_count = min(optimal_count, current_count + 2)
            self.setMaxThreadCount(new_count)
            self._metrics.max_threads = new_count
            logger.info(
                f"ThreadPool '{self.pool_id}' increased to {new_count} threads "
                f"(load: {load_factor:.2f}, queue: {queue_size})"
            )

        elif should_decrease and current_count > self._min_thread_limit:
            new_count = max(optimal_count, current_count - 1)
            self.setMaxThreadCount(new_count)
            self._metrics.max_threads = new_count
            logger.info(
                f"ThreadPool '{self.pool_id}' decreased to {new_count} threads "
                f"(load: {load_factor:.2f})"
            )

        # Try to process queued tasks after adjustment
        self._process_queued_tasks()
        self._last_rebalance_time = current_time

    def _update_metrics(self):
        """Update pool metrics and health status."""
        try:
            # Update basic metrics
            self._metrics.active_threads = self.activeThreadCount()
            self._metrics.queued_tasks = self._priority_queue.qsize()
            self._metrics.last_updated = time.perf_counter()

            # Update resource usage
            try:
                process = psutil.Process()
                self._metrics.cpu_usage = process.cpu_percent()
                self._metrics.memory_usage_mb = process.memory_info().rss / (
                    1024 * 1024
                )
            except Exception:
                pass  # Continue without process metrics if unavailable

            # Determine health status
            old_health = self._metrics.health_status
            load_factor = self._metrics.calculate_load_factor()

            if (
                load_factor <= 0.7
                and self._metrics.queued_tasks < self._metrics.max_threads
            ):
                self._metrics.health_status = ThreadPoolHealth.HEALTHY
            elif (
                load_factor <= 1.0
                and self._metrics.queued_tasks < self._metrics.max_threads * 2
            ):
                self._metrics.health_status = ThreadPoolHealth.DEGRADED
            elif load_factor <= 1.5:
                self._metrics.health_status = ThreadPoolHealth.OVERLOADED
            else:
                self._metrics.health_status = ThreadPoolHealth.CRITICAL

            # Emit signals if health changed
            if old_health != self._metrics.health_status:
                self.health_status_changed.emit(
                    self.pool_id, self._metrics.health_status.value
                )
                logger.warning(
                    f"ThreadPool '{self.pool_id}' health changed: {old_health.value} -> {self._metrics.health_status.value}"
                )

            # Emit metrics update
            self.pool_metrics_updated.emit(self._metrics)

        except Exception as e:
            logger.error(f"Error updating metrics for ThreadPool '{self.pool_id}': {e}")

    def get_metrics(self) -> ThreadPoolMetrics:
        """Get current pool metrics."""
        return self._metrics

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive pool statistics."""
        uptime = time.perf_counter() - (
            self._last_rebalance_time - self._rebalance_interval
        )

        return {
            "pool_id": self.pool_id,
            "uptime_seconds": uptime,
            "thread_limits": {
                "min": self._min_thread_limit,
                "max": self._max_thread_limit,
                "current": self.maxThreadCount(),
            },
            "tasks": {
                "total_submitted": self._total_tasks_submitted,
                "total_completed": self._total_tasks_completed,
                "success_rate": (
                    self._metrics.completed_tasks / max(self._total_tasks_completed, 1)
                )
                * 100,
                "currently_active": len(self._active_tasks),
                "queued": self._priority_queue.qsize(),
            },
            "performance": {
                "avg_execution_time": self._metrics.avg_execution_time,
                "load_factor": self._metrics.calculate_load_factor(),
                "throughput_per_second": (self._total_tasks_completed / max(uptime, 1)),
            },
            "resources": {
                "cpu_usage_percent": self._metrics.cpu_usage,
                "memory_usage_mb": self._metrics.memory_usage_mb,
            },
            "health": {
                "status": self._metrics.health_status.value,
                "last_updated": self._metrics.last_updated,
            },
        }

    def clear_queue(self):
        """Clear all queued tasks."""
        with QMutexLocker(self._mutex):
            while not self._priority_queue.empty():
                try:
                    self._priority_queue.get_nowait()
                except Empty:
                    break
            self._metrics.queued_tasks = 0
            logger.info(f"ThreadPool '{self.pool_id}' queue cleared")

    def shutdown_gracefully(self, timeout_ms: int = 30000):
        """Gracefully shutdown the thread pool."""
        logger.info(f"Shutting down ThreadPool '{self.pool_id}'...")

        # Stop timers
        self._resource_monitor_timer.stop()
        self._thread_adjustment_timer.stop()

        # Clear queue
        self.clear_queue()

        # Wait for active tasks to complete
        if not self.waitForDone(timeout_ms):
            logger.warning(
                f"ThreadPool '{self.pool_id}' shutdown timed out after {timeout_ms}ms"
            )

        logger.info(f"ThreadPool '{self.pool_id}' shutdown complete")


class ThreadPoolManager(QObject):
    """
    Manages multiple smart thread pools with load balancing.
    """

    # Signals
    manager_stats_updated = Signal(object)  # Manager statistics
    pool_created = Signal(str)  # pool_id
    pool_removed = Signal(str)  # pool_id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pools: dict[str, SmartThreadPool] = {}
        self._load_balance_strategy = LoadBalanceStrategy.RESOURCE_AWARE

        # Create default pool
        self.create_pool("default")

        # Global monitoring
        self._global_stats_timer = QTimer()
        self._global_stats_timer.timeout.connect(self._update_global_stats)
        self._global_stats_timer.start(5000)  # Update every 5 seconds

        logger.info("ThreadPoolManager initialized with default pool")

    def create_pool(self, pool_id: str, **kwargs) -> SmartThreadPool:
        """Create a new thread pool."""
        if pool_id in self._pools:
            logger.warning(f"Pool '{pool_id}' already exists")
            return self._pools[pool_id]

        pool = SmartThreadPool(pool_id=pool_id, parent=self, **kwargs)
        self._pools[pool_id] = pool

        # Connect pool signals
        pool.pool_metrics_updated.connect(self._on_pool_metrics_updated)
        pool.health_status_changed.connect(self._on_pool_health_changed)

        self.pool_created.emit(pool_id)
        logger.info(f"Created thread pool: {pool_id}")

        return pool

    def get_pool(self, pool_id: str) -> SmartThreadPool | None:
        """Get a thread pool by ID."""
        return self._pools.get(pool_id)

    def remove_pool(self, pool_id: str, shutdown_timeout_ms: int = 30000) -> bool:
        """Remove and shutdown a thread pool."""
        if pool_id not in self._pools:
            return False

        if pool_id == "default":
            logger.warning("Cannot remove default pool")
            return False

        pool = self._pools[pool_id]
        pool.shutdown_gracefully(shutdown_timeout_ms)

        del self._pools[pool_id]
        self.pool_removed.emit(pool_id)

        logger.info(f"Removed thread pool: {pool_id}")
        return True

    def submit_task(
        self,
        task: QRunnable,
        pool_id: str | None = None,
        priority: int = 5,
        **kwargs,
    ) -> bool:
        """Submit a task to the best available pool."""
        target_pool = None

        if pool_id:
            target_pool = self._pools.get(pool_id)
            if not target_pool:
                logger.error(f"Pool '{pool_id}' not found")
                return False
        else:
            # Select best pool based on strategy
            target_pool = self._select_optimal_pool()

        if target_pool:
            target_pool.start_with_priority(task, priority, **kwargs)
            return True

        return False

    def _select_optimal_pool(self) -> SmartThreadPool | None:
        """Select the optimal pool based on load balancing strategy."""
        if not self._pools:
            return None

        if self._load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            # Simple round-robin (not implemented for brevity)
            return next(iter(self._pools.values()))

        if self._load_balance_strategy == LoadBalanceStrategy.LEAST_LOADED:
            # Select pool with lowest load factor
            best_pool = min(
                self._pools.values(),
                key=lambda p: p.get_metrics().calculate_load_factor(),
            )
            return best_pool

        if self._load_balance_strategy == LoadBalanceStrategy.RESOURCE_AWARE:
            # Select pool with best resource availability
            best_score = float("inf")
            best_pool = None

            for pool in self._pools.values():
                metrics = pool.get_metrics()
                # Score based on load, queue size, and health
                score = (
                    metrics.calculate_load_factor() * 0.4
                    + (metrics.queued_tasks / max(metrics.max_threads, 1)) * 0.3
                    + (metrics.health_status.value == "critical") * 0.3
                )

                if score < best_score:
                    best_score = score
                    best_pool = pool

            return best_pool

        # Default to first available pool
        return next(iter(self._pools.values()))

    def set_load_balance_strategy(self, strategy: LoadBalanceStrategy):
        """Set the load balancing strategy."""
        self._load_balance_strategy = strategy
        logger.info(f"Load balancing strategy set to: {strategy.value}")

    def _on_pool_metrics_updated(self, metrics: ThreadPoolMetrics):
        """Handle pool metrics updates."""
        # Could implement global optimization logic here

    def _on_pool_health_changed(self, pool_id: str, new_health: str):
        """Handle pool health changes."""
        if new_health == "critical":
            logger.error(f"Pool '{pool_id}' is in critical state!")
            # Could implement automatic pool scaling or task migration here

    def _update_global_stats(self):
        """Update and emit global manager statistics."""
        stats = self.get_global_statistics()
        self.manager_stats_updated.emit(stats)

    def get_global_statistics(self) -> dict[str, Any]:
        """Get global thread pool manager statistics."""
        total_pools = len(self._pools)
        total_threads = sum(pool.maxThreadCount() for pool in self._pools.values())
        total_active = sum(pool.activeThreadCount() for pool in self._pools.values())
        total_queued = sum(
            pool.get_metrics().queued_tasks for pool in self._pools.values()
        )

        # Health distribution
        health_counts = defaultdict(int)
        for pool in self._pools.values():
            health_counts[pool.get_metrics().health_status.value] += 1

        return {
            "total_pools": total_pools,
            "threads": {
                "total_capacity": total_threads,
                "currently_active": total_active,
                "utilization_percent": (total_active / max(total_threads, 1)) * 100,
            },
            "tasks": {
                "total_queued": total_queued,
                "avg_queue_per_pool": total_queued / max(total_pools, 1),
            },
            "health_distribution": dict(health_counts),
            "load_balance_strategy": self._load_balance_strategy.value,
            "pool_list": list(self._pools.keys()),
        }

    def shutdown_all_pools(self, timeout_ms: int = 30000):
        """Shutdown all thread pools gracefully."""
        logger.info("Shutting down all thread pools...")

        self._global_stats_timer.stop()

        for pool_id, pool in list(self._pools.items()):
            if pool_id != "default":  # Shutdown non-default pools first
                pool.shutdown_gracefully(timeout_ms)
                del self._pools[pool_id]

        # Shutdown default pool last
        if "default" in self._pools:
            self._pools["default"].shutdown_gracefully(timeout_ms)
            del self._pools["default"]

        logger.info("All thread pools shut down")


# Global thread pool manager instance
_global_thread_pool_manager: ThreadPoolManager | None = None


def get_thread_pool_manager() -> ThreadPoolManager:
    """Get the global thread pool manager."""
    global _global_thread_pool_manager
    if _global_thread_pool_manager is None:
        _global_thread_pool_manager = ThreadPoolManager()
    return _global_thread_pool_manager


def initialize_enhanced_threading() -> ThreadPoolManager:
    """Initialize the enhanced threading system."""
    global _global_thread_pool_manager
    if _global_thread_pool_manager is None:
        _global_thread_pool_manager = ThreadPoolManager()
    return _global_thread_pool_manager


def shutdown_enhanced_threading():
    """Shutdown the enhanced threading system."""
    global _global_thread_pool_manager
    if _global_thread_pool_manager:
        _global_thread_pool_manager.shutdown_all_pools()
        _global_thread_pool_manager = None
