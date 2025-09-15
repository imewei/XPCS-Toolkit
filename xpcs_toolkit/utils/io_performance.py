"""
I/O Performance monitoring utilities for XPCS Toolkit.

This module provides comprehensive performance monitoring, optimization
strategies, and diagnostics for HDF5 file operations.
"""

import os
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import h5py
import psutil

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class IOOperation:
    """Information about a single I/O operation."""

    operation_type: str  # 'read', 'write', 'open', 'close'
    file_path: str
    dataset_path: str | None = None
    start_time: float = 0.0
    end_time: float = 0.0
    data_size_mb: float = 0.0
    success: bool = True
    error_message: str | None = None
    thread_id: int = 0

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def throughput_mbps(self) -> float:
        """Throughput in MB/s."""
        if self.duration_ms > 0 and self.data_size_mb > 0:
            return self.data_size_mb / (self.duration_ms / 1000)
        return 0.0


@dataclass
class IOStats:
    """Aggregated I/O statistics."""

    total_operations: int = 0
    total_duration_ms: float = 0.0
    total_data_mb: float = 0.0
    successful_operations: int = 0
    failed_operations: int = 0
    operations_by_type: dict[str, int] = field(default_factory=dict)

    @property
    def average_duration_ms(self) -> float:
        """Average operation duration in milliseconds."""
        return (
            self.total_duration_ms / self.total_operations
            if self.total_operations > 0
            else 0.0
        )

    @property
    def average_throughput_mbps(self) -> float:
        """Average throughput in MB/s."""
        if self.total_duration_ms > 0 and self.total_data_mb > 0:
            return self.total_data_mb / (self.total_duration_ms / 1000)
        return 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        return (
            (self.successful_operations / self.total_operations * 100)
            if self.total_operations > 0
            else 0.0
        )


class IOPerformanceMonitor:
    """
    Comprehensive I/O performance monitoring for HDF5 operations.

    Features:
    - Real-time operation tracking
    - Aggregated statistics by file and operation type
    - Performance bottleneck identification
    - Memory usage correlation
    - Thread-safe operation logging
    """

    def __init__(
        self, enable_detailed_logging: bool = True, max_operations_logged: int = 10000
    ):
        self.enable_detailed_logging = enable_detailed_logging
        self.max_operations_logged = max_operations_logged

        # Thread-safe containers
        self._lock = threading.RLock()
        self._operations: list[IOOperation] = []
        self._stats_by_file: dict[str, IOStats] = defaultdict(IOStats)
        self._stats_global = IOStats()

        # Performance thresholds (can be configured)
        self.slow_operation_threshold_ms = 1000  # 1 second
        self.low_throughput_threshold_mbps = 10  # 10 MB/s

        logger.info("IOPerformanceMonitor initialized")

    def start_operation(
        self,
        operation_type: str,
        file_path: str,
        dataset_path: str | None = None,
        data_size_mb: float = 0.0,
    ) -> IOOperation:
        """
        Start tracking an I/O operation.

        Parameters
        ----------
        operation_type : str
            Type of operation ('read', 'write', 'open', 'close')
        file_path : str
            Path to the file being operated on
        dataset_path : str, optional
            Path to dataset within HDF5 file
        data_size_mb : float
            Size of data being processed in MB

        Returns
        -------
        IOOperation
            Operation object to be completed later
        """
        operation = IOOperation(
            operation_type=operation_type,
            file_path=file_path,
            dataset_path=dataset_path,
            start_time=time.time(),
            data_size_mb=data_size_mb,
            thread_id=threading.get_ident(),
        )

        return operation

    def complete_operation(
        self,
        operation: IOOperation,
        success: bool = True,
        error_message: str | None = None,
    ):
        """
        Complete tracking of an I/O operation.

        Parameters
        ----------
        operation : IOOperation
            The operation object returned from start_operation
        success : bool
            Whether the operation succeeded
        error_message : str, optional
            Error message if operation failed
        """
        operation.end_time = time.time()
        operation.success = success
        operation.error_message = error_message

        with self._lock:
            # Add to detailed log if enabled
            if self.enable_detailed_logging:
                if len(self._operations) >= self.max_operations_logged:
                    # Remove oldest operations to prevent memory growth
                    self._operations = self._operations[
                        self.max_operations_logged // 2 :
                    ]

                self._operations.append(operation)

            # Update statistics
            self._update_stats(operation)

            # Log slow operations
            if operation.duration_ms > self.slow_operation_threshold_ms:
                logger.warning(
                    f"Slow I/O operation: {operation.operation_type} on {operation.file_path} "
                    f"took {operation.duration_ms:.1f}ms"
                )

            # Log low throughput operations
            if (
                operation.throughput_mbps > 0
                and operation.throughput_mbps < self.low_throughput_threshold_mbps
            ):
                logger.warning(
                    f"Low throughput I/O: {operation.operation_type} on {operation.file_path} "
                    f"achieved {operation.throughput_mbps:.2f} MB/s"
                )

    def _update_stats(self, operation: IOOperation):
        """Update internal statistics with completed operation."""
        # Update global stats
        self._stats_global.total_operations += 1
        self._stats_global.total_duration_ms += operation.duration_ms
        self._stats_global.total_data_mb += operation.data_size_mb

        if operation.success:
            self._stats_global.successful_operations += 1
        else:
            self._stats_global.failed_operations += 1

        # Update operation type count
        if operation.operation_type not in self._stats_global.operations_by_type:
            self._stats_global.operations_by_type[operation.operation_type] = 0
        self._stats_global.operations_by_type[operation.operation_type] += 1

        # Update per-file stats
        file_stats = self._stats_by_file[operation.file_path]
        file_stats.total_operations += 1
        file_stats.total_duration_ms += operation.duration_ms
        file_stats.total_data_mb += operation.data_size_mb

        if operation.success:
            file_stats.successful_operations += 1
        else:
            file_stats.failed_operations += 1

        if operation.operation_type not in file_stats.operations_by_type:
            file_stats.operations_by_type[operation.operation_type] = 0
        file_stats.operations_by_type[operation.operation_type] += 1

    def get_global_stats(self) -> dict[str, Any]:
        """Get comprehensive global I/O statistics."""
        with self._lock:
            stats_dict = {
                "total_operations": self._stats_global.total_operations,
                "total_duration_ms": self._stats_global.total_duration_ms,
                "total_data_mb": self._stats_global.total_data_mb,
                "average_duration_ms": self._stats_global.average_duration_ms,
                "average_throughput_mbps": self._stats_global.average_throughput_mbps,
                "success_rate_percent": self._stats_global.success_rate,
                "operations_by_type": dict(self._stats_global.operations_by_type),
                "unique_files_accessed": len(self._stats_by_file),
            }

            # Add system resource information
            memory = psutil.virtual_memory()
            stats_dict.update(
                {
                    "system_memory_used_percent": memory.percent,
                    "system_memory_available_mb": memory.available / (1024 * 1024),
                }
            )

            return stats_dict

    def get_file_stats(self, file_path: str) -> dict[str, Any] | None:
        """Get statistics for a specific file."""
        with self._lock:
            if file_path not in self._stats_by_file:
                return None

            file_stats = self._stats_by_file[file_path]
            return {
                "file_path": file_path,
                "total_operations": file_stats.total_operations,
                "total_duration_ms": file_stats.total_duration_ms,
                "total_data_mb": file_stats.total_data_mb,
                "average_duration_ms": file_stats.average_duration_ms,
                "average_throughput_mbps": file_stats.average_throughput_mbps,
                "success_rate_percent": file_stats.success_rate,
                "operations_by_type": dict(file_stats.operations_by_type),
                "file_size_mb": os.path.getsize(file_path) / (1024 * 1024)
                if os.path.exists(file_path)
                else 0,
            }

    def get_recent_operations(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent operations for detailed analysis."""
        with self._lock:
            recent_ops = self._operations[-limit:] if self._operations else []

            return [
                {
                    "operation_type": op.operation_type,
                    "file_path": op.file_path,
                    "dataset_path": op.dataset_path,
                    "duration_ms": op.duration_ms,
                    "data_size_mb": op.data_size_mb,
                    "throughput_mbps": op.throughput_mbps,
                    "success": op.success,
                    "error_message": op.error_message,
                    "timestamp": op.start_time,
                }
                for op in recent_ops
            ]

    def identify_bottlenecks(self) -> dict[str, Any]:
        """Identify potential I/O performance bottlenecks."""
        with self._lock:
            bottlenecks = {
                "slow_files": [],
                "low_throughput_files": [],
                "high_failure_rate_files": [],
                "recommendations": [],
            }

            for file_path, stats in self._stats_by_file.items():
                # Check for slow average operations
                if stats.average_duration_ms > self.slow_operation_threshold_ms:
                    bottlenecks["slow_files"].append(
                        {
                            "file_path": file_path,
                            "average_duration_ms": stats.average_duration_ms,
                            "total_operations": stats.total_operations,
                        }
                    )

                # Check for low throughput
                if (
                    stats.average_throughput_mbps > 0
                    and stats.average_throughput_mbps
                    < self.low_throughput_threshold_mbps
                ):
                    bottlenecks["low_throughput_files"].append(
                        {
                            "file_path": file_path,
                            "average_throughput_mbps": stats.average_throughput_mbps,
                            "total_data_mb": stats.total_data_mb,
                        }
                    )

                # Check for high failure rates
                if stats.success_rate < 95.0 and stats.total_operations > 5:
                    bottlenecks["high_failure_rate_files"].append(
                        {
                            "file_path": file_path,
                            "success_rate_percent": stats.success_rate,
                            "failed_operations": stats.failed_operations,
                        }
                    )

            # Generate recommendations
            if bottlenecks["slow_files"]:
                bottlenecks["recommendations"].append(
                    "Consider using connection pooling or batch operations for slow files"
                )

            if bottlenecks["low_throughput_files"]:
                bottlenecks["recommendations"].append(
                    "Consider chunked reading or memory mapping for large datasets"
                )

            if bottlenecks["high_failure_rate_files"]:
                bottlenecks["recommendations"].append(
                    "Check file accessibility and consider health monitoring"
                )

            return bottlenecks

    def clear_stats(self):
        """Clear all collected statistics and operations."""
        with self._lock:
            self._operations.clear()
            self._stats_by_file.clear()
            self._stats_global = IOStats()
            logger.info("IOPerformanceMonitor statistics cleared")

    def set_thresholds(
        self,
        slow_operation_ms: float | None = None,
        low_throughput_mbps: float | None = None,
    ):
        """Update performance thresholds for bottleneck detection."""
        if slow_operation_ms is not None:
            self.slow_operation_threshold_ms = slow_operation_ms

        if low_throughput_mbps is not None:
            self.low_throughput_threshold_mbps = low_throughput_mbps

        logger.info(
            f"Performance thresholds updated: slow_operation={self.slow_operation_threshold_ms}ms, "
            f"low_throughput={self.low_throughput_threshold_mbps}MB/s"
        )


# Global performance monitor instance
_performance_monitor = IOPerformanceMonitor()


def get_performance_monitor() -> IOPerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def io_operation_tracker(operation_type: str, data_size_mb: float = 0.0):
    """
    Decorator for tracking I/O operations.

    Parameters
    ----------
    operation_type : str
        Type of I/O operation
    data_size_mb : float
        Size of data being processed
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Try to extract file path from arguments
            file_path = "unknown"
            if args and hasattr(args[0], "fname"):
                file_path = args[0].fname
            elif "fname" in kwargs:
                file_path = kwargs["fname"]
            elif args and isinstance(args[0], str):
                file_path = args[0]

            # Start tracking
            operation = _performance_monitor.start_operation(
                operation_type, file_path, data_size_mb=data_size_mb
            )

            try:
                result = func(*args, **kwargs)
                _performance_monitor.complete_operation(operation, success=True)
                return result
            except Exception as e:
                _performance_monitor.complete_operation(
                    operation, success=False, error_message=str(e)
                )
                raise

        return wrapper

    return decorator


def estimate_hdf5_dataset_size(file_path: str, dataset_path: str) -> float:
    """
    Estimate the size of an HDF5 dataset in MB without fully loading it.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file
    dataset_path : str
        Path to dataset within file

    Returns
    -------
    float
        Estimated size in MB
    """
    try:
        with h5py.File(file_path, "r") as f:
            if dataset_path in f:
                dataset = f[dataset_path]
                return dataset.size * dataset.dtype.itemsize / (1024 * 1024)
        return 0.0
    except Exception as e:
        logger.warning(f"Could not estimate dataset size for {dataset_path}: {e}")
        return 0.0
