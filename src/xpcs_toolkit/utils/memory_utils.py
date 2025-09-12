"""
Memory optimization utilities for XPCS Toolkit.

This module provides tools for monitoring memory usage, optimizing array operations,
and implementing memory-efficient alternatives for common operations.
"""

from __future__ import annotations

import numpy as np
import psutil
import logging
import time
import threading
from typing import NamedTuple

logger = logging.getLogger(__name__)


class MemoryStatus(NamedTuple):
    """Memory status information."""

    used_mb: float
    available_mb: float
    percent_used: float
    timestamp: float


class CachedMemoryMonitor:
    """
    Cached memory monitoring system to reduce overhead from frequent psutil calls.

    Implements TTL-based caching with configurable thresholds and hysteresis
    to prevent memory monitoring overhead and cleanup thrashing.
    """

    def __init__(
        self,
        cache_ttl_seconds: float = 5.0,
        cleanup_threshold: float = 0.85,
        cleanup_stop_threshold: float = 0.75,
        background_update: bool = True,
    ):
        """
        Initialize cached memory monitor.

        Parameters
        ----------
        cache_ttl_seconds : float
            Cache time-to-live in seconds
        cleanup_threshold : float
            Memory threshold for triggering cleanup (0-1)
        cleanup_stop_threshold : float
            Memory threshold for stopping cleanup (0-1)
        background_update : bool
            Whether to update cache in background thread
        """
        self._cache_ttl = cache_ttl_seconds
        self._cleanup_threshold = cleanup_threshold
        self._cleanup_stop_threshold = cleanup_stop_threshold

        # Thread-safe cache
        self._lock = threading.Lock()
        self._cached_status: MemoryStatus | None = None
        self._cache_hits = 0
        self._cache_misses = 0

        # Background updating
        self._background_timer: threading.Timer | None = None
        self._background_update = background_update
        self._stop_background = False

        if background_update:
            self._start_background_monitoring()

    def _get_fresh_memory_status(self) -> MemoryStatus:
        """Get fresh memory status from psutil."""
        memory = psutil.virtual_memory()
        used_mb = (memory.total - memory.available) / (1024 * 1024)
        available_mb = memory.available / (1024 * 1024)
        percent_used = memory.percent / 100.0

        return MemoryStatus(used_mb, available_mb, percent_used, time.time())

    def _is_cache_valid(self, status: MemoryStatus) -> bool:
        """Check if cached status is still valid."""
        return (time.time() - status.timestamp) < self._cache_ttl

    def get_memory_status(self) -> MemoryStatus:
        """
        Get current memory status with caching.

        Returns
        -------
        MemoryStatus
            Current memory information
        """
        with self._lock:
            # Check cache validity
            if self._cached_status and self._is_cache_valid(self._cached_status):
                self._cache_hits += 1
                return self._cached_status

            # Cache miss - get fresh data
            self._cache_misses += 1
            self._cached_status = self._get_fresh_memory_status()
            return self._cached_status

    def is_memory_pressure_high(self, threshold: float | None = None) -> bool:
        """
        Check if memory pressure is high using cached data.

        Parameters
        ----------
        threshold : float, optional
            Custom threshold, defaults to cleanup_threshold

        Returns
        -------
        bool
            True if memory pressure is high
        """
        if threshold is None:
            threshold = self._cleanup_threshold

        status = self.get_memory_status()
        return status.percent_used > threshold

    def should_cleanup_memory(self) -> bool:
        """
        Check if memory cleanup should be triggered with hysteresis.

        Returns
        -------
        bool
            True if cleanup should be triggered
        """
        return self.is_memory_pressure_high(self._cleanup_threshold)

    def should_stop_cleanup(self) -> bool:
        """
        Check if memory cleanup should stop with hysteresis.

        Returns
        -------
        bool
            True if cleanup should stop
        """
        status = self.get_memory_status()
        return status.percent_used <= self._cleanup_stop_threshold

    def get_memory_info(self) -> tuple[float, float, float]:
        """
        Get memory info in legacy format.

        Returns
        -------
        tuple
            (used_memory_mb, available_memory_mb, percent_used)
        """
        status = self.get_memory_status()
        return status.used_mb, status.available_mb, status.percent_used

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "ttl_seconds": self._cache_ttl,
            "cleanup_threshold": self._cleanup_threshold,
            "cleanup_stop_threshold": self._cleanup_stop_threshold,
        }

    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        if not self._stop_background:
            self._update_cache_background()
            # Schedule next update
            self._background_timer = threading.Timer(
                self._cache_ttl * 0.8,  # Update slightly before cache expires
                self._start_background_monitoring,
            )
            self._background_timer.daemon = True
            self._background_timer.start()

    def _update_cache_background(self):
        """Update cache in background thread."""
        try:
            with self._lock:
                self._cached_status = self._get_fresh_memory_status()
        except Exception as e:
            logger.warning(f"Background memory monitoring update failed: {e}")

    def stop_background_monitoring(self):
        """Stop background monitoring."""
        self._stop_background = True
        if self._background_timer:
            self._background_timer.cancel()

    def configure_thresholds(
        self,
        cleanup_threshold: float | None = None,
        cleanup_stop_threshold: float | None = None,
    ):
        """
        Update memory pressure thresholds.

        Parameters
        ----------
        cleanup_threshold : float, optional
            New cleanup threshold
        cleanup_stop_threshold : float, optional
            New cleanup stop threshold
        """
        if cleanup_threshold is not None:
            self._cleanup_threshold = cleanup_threshold
        if cleanup_stop_threshold is not None:
            self._cleanup_stop_threshold = cleanup_stop_threshold

        logger.info(
            f"Updated memory thresholds: cleanup={self._cleanup_threshold:.1%}, "
            f"stop={self._cleanup_stop_threshold:.1%}"
        )


# Global cached memory monitor instance
_cached_memory_monitor: CachedMemoryMonitor | None = None
_monitor_lock = threading.Lock()


def get_cached_memory_monitor(**kwargs) -> CachedMemoryMonitor:
    """
    Get the global cached memory monitor instance.

    Parameters
    ----------
    **kwargs
        Configuration parameters for first-time initialization

    Returns
    -------
    CachedMemoryMonitor
        Global memory monitor instance
    """
    global _cached_memory_monitor

    with _monitor_lock:
        if _cached_memory_monitor is None:
            # Set defaults optimized for XPCS workloads
            defaults = {
                "cache_ttl_seconds": 5.0,
                "cleanup_threshold": 0.85,  # More conservative than original 0.80
                "cleanup_stop_threshold": 0.75,  # Hysteresis
                "background_update": True,
            }
            defaults.update(kwargs)
            _cached_memory_monitor = CachedMemoryMonitor(**defaults)
            logger.info("Initialized cached memory monitor with optimized settings")

        return _cached_memory_monitor


class MemoryTracker:
    """Track memory usage of arrays and operations."""

    @staticmethod
    def array_memory_mb(arr: np.ndarray) -> float:
        """
        Calculate memory usage of a numpy array in MB.

        Parameters
        ----------
        arr : np.ndarray
            Input array

        Returns
        -------
        float
            Memory usage in MB
        """
        return arr.nbytes / (1024 * 1024)

    @staticmethod
    def get_optimal_dtype(
        data: np.ndarray, preserve_precision: bool = True
    ) -> np.dtype:
        """
        Suggest optimal dtype for array to minimize memory usage.

        Parameters
        ----------
        data : np.ndarray
            Input array
        preserve_precision : bool
            Whether to preserve floating point precision

        Returns
        -------
        np.dtype
            Recommended dtype
        """
        if np.issubdtype(data.dtype, np.integer):
            # For integers, find the smallest type that can hold the range
            min_val, max_val = np.min(data), np.max(data)

            if min_val >= 0:  # Unsigned integers
                if max_val < 2**8:
                    return np.uint8
                elif max_val < 2**16:
                    return np.uint16
                elif max_val < 2**32:
                    return np.uint32
                else:
                    return np.uint64
            else:  # Signed integers
                if min_val >= -(2**7) and max_val < 2**7:
                    return np.int8
                elif min_val >= -(2**15) and max_val < 2**15:
                    return np.int16
                elif min_val >= -(2**31) and max_val < 2**31:
                    return np.int32
                else:
                    return np.int64

        elif np.issubdtype(data.dtype, np.floating):
            if preserve_precision:
                return np.float32 if data.dtype == np.float64 else data.dtype
            else:
                # Check if data can be represented as float32 without significant loss
                if data.dtype == np.float64:
                    data_f32 = data.astype(np.float32)
                    if np.allclose(data, data_f32, rtol=1e-6):
                        return np.float32
                return data.dtype

        return data.dtype


class MemoryOptimizer:
    """Provide memory-efficient alternatives for common operations."""

    @staticmethod
    def safe_copy(arr: np.ndarray, optimize_dtype: bool = False) -> np.ndarray:
        """
        Create a memory-efficient copy of an array.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        optimize_dtype : bool
            Whether to optimize the dtype

        Returns
        -------
        np.ndarray
            Optimized copy
        """
        if optimize_dtype:
            optimal_dtype = MemoryTracker.get_optimal_dtype(arr)
            if optimal_dtype != arr.dtype:
                logger.debug(f"Optimizing dtype from {arr.dtype} to {optimal_dtype}")
                return arr.astype(optimal_dtype)

        return arr.copy()

    @staticmethod
    def masked_operation(
        arr: np.ndarray, mask: np.ndarray, operation: str, value: float = 0.0
    ) -> np.ndarray:
        """
        Perform masked operations efficiently.

        Parameters
        ----------
        arr : np.ndarray
            Input array
        mask : np.ndarray
            Boolean mask
        operation : str
            Operation type: 'fill', 'replace', 'nan_fill'
        value : float
            Value for operation

        Returns
        -------
        np.ndarray
            Result array
        """
        if operation == "fill":
            return np.where(mask, arr, value)
        elif operation == "replace":
            return np.where(mask, value, arr)
        elif operation == "nan_fill":
            return np.where(mask, arr, np.nan)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    @staticmethod
    def efficient_bincount(
        indices: np.ndarray,
        weights: np.ndarray | None = None,
        minlength: int = 0,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Memory-efficient bincount with optional normalization.

        Parameters
        ----------
        indices : np.ndarray
            Bin indices
        weights : np.ndarray, optional
            Weights for each index
        minlength : int
            Minimum length of output
        normalize : bool
            Whether to normalize by bin counts

        Returns
        -------
        np.ndarray
            Binned data
        """
        if weights is not None:
            result = np.bincount(indices, weights, minlength=minlength)
        else:
            result = np.bincount(indices, minlength=minlength)

        if normalize:
            counts = np.bincount(indices, minlength=minlength)
            valid_mask = counts > 0
            result[valid_mask] /= counts[valid_mask]

        return result


class SystemMemoryMonitor:
    """Monitor system memory usage."""

    @staticmethod
    def get_memory_info() -> tuple[float, float, float]:
        """
        Get current memory usage information.

        Returns
        -------
        tuple
            (used_memory_mb, available_memory_mb, percent_used)
        """
        memory = psutil.virtual_memory()
        used_mb = (memory.total - memory.available) / (1024 * 1024)
        available_mb = memory.available / (1024 * 1024)
        percent_used = memory.percent

        return used_mb, available_mb, percent_used

    @staticmethod
    def check_memory_pressure(threshold: float = 80.0) -> bool:
        """
        Check if system memory pressure is high.

        Parameters
        ----------
        threshold : float
            Memory usage percentage threshold

        Returns
        -------
        bool
            True if memory pressure is high
        """
        _, _, percent_used = SystemMemoryMonitor.get_memory_info()
        return percent_used > threshold

    @staticmethod
    def suggest_dtype_optimization(arrays: list) -> dict:
        """
        Suggest dtype optimizations for a list of arrays.

        Parameters
        ----------
        arrays : list
            List of numpy arrays

        Returns
        -------
        dict
            Optimization suggestions
        """
        suggestions = {}
        total_current_mb = 0
        total_optimized_mb = 0

        for i, arr in enumerate(arrays):
            if not isinstance(arr, np.ndarray):
                continue

            current_mb = MemoryTracker.array_memory_mb(arr)
            optimal_dtype = MemoryTracker.get_optimal_dtype(arr)

            if optimal_dtype != arr.dtype:
                # Calculate potential savings
                optimized_mb = current_mb * (
                    np.dtype(optimal_dtype).itemsize / arr.dtype.itemsize
                )
                savings_mb = current_mb - optimized_mb

                suggestions[f"array_{i}"] = {
                    "current_dtype": str(arr.dtype),
                    "optimal_dtype": str(optimal_dtype),
                    "current_memory_mb": current_mb,
                    "optimized_memory_mb": optimized_mb,
                    "savings_mb": savings_mb,
                    "shape": arr.shape,
                }

                total_optimized_mb += optimized_mb
            else:
                total_optimized_mb += current_mb

            total_current_mb += current_mb

        summary = {
            "total_current_mb": total_current_mb,
            "total_optimized_mb": total_optimized_mb,
            "total_savings_mb": total_current_mb - total_optimized_mb,
            "arrays": suggestions,
        }

        return summary


def memory_profiler(func):
    """
    Decorator to profile memory usage of a function.

    Parameters
    ----------
    func : callable
        Function to profile

    Returns
    -------
    callable
        Decorated function
    """

    def wrapper(*args, **kwargs):
        # Get memory before execution
        used_before, _, _ = SystemMemoryMonitor.get_memory_info()

        # Execute function
        result = func(*args, **kwargs)

        # Get memory after execution
        used_after, _, _ = SystemMemoryMonitor.get_memory_info()

        memory_diff = used_after - used_before
        logger.debug(f"Function {func.__name__} memory usage: {memory_diff:.2f} MB")

        return result

    return wrapper
