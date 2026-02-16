"""Memory monitoring utilities for XpcsFile.

This module provides memory monitoring and status tracking for XPCS data operations.
"""

from __future__ import annotations

import time

import numpy as np
import psutil

# Cached psutil result with TTL to avoid repeated syscalls.
# Each psutil.virtual_memory() is a syscall (~0.1-0.5ms). With 30+ calls
# per operation cycle, caching saves 3-15ms of pure overhead.
_VMEM_CACHE_TTL = 2.0  # seconds
_vmem_cache_result: psutil._common.svmem | None = None
_vmem_cache_timestamp: float = 0.0


def _get_virtual_memory() -> psutil._common.svmem:
    """Return psutil.virtual_memory(), cached with a 2-second TTL."""
    global _vmem_cache_result, _vmem_cache_timestamp  # noqa: PLW0603
    now = time.monotonic()
    if _vmem_cache_result is None or (now - _vmem_cache_timestamp) >= _VMEM_CACHE_TTL:
        _vmem_cache_result = psutil.virtual_memory()
        _vmem_cache_timestamp = now
    return _vmem_cache_result


class MemoryStatus:
    """Container for memory status information."""

    def __init__(self, percent_used: float):
        self.percent_used = percent_used


class MemoryMonitor:
    """Simple memory monitoring utilities using psutil."""

    def get_memory_info(self) -> tuple[float, float, float]:
        """Get current memory usage information.

        Returns
        -------
        tuple[float, float, float]
            (used_mb, available_mb, pressure_ratio)
        """
        memory = _get_virtual_memory()
        used_mb = (memory.total - memory.available) / 1024 / 1024
        available_mb = memory.available / 1024 / 1024
        pressure_ratio = memory.percent / 100.0
        return used_mb, available_mb, pressure_ratio

    def get_memory_status(self) -> MemoryStatus:
        """Get memory status object.

        Returns
        -------
        MemoryStatus
            Object containing percent_used attribute
        """
        memory = _get_virtual_memory()
        return MemoryStatus(memory.percent / 100.0)

    @staticmethod
    def get_memory_usage() -> tuple[float, float]:
        """Get current memory usage in MB (static method for backward compatibility)."""
        monitor = get_cached_memory_monitor()
        used_mb, available_mb, _ = monitor.get_memory_info()
        return used_mb, available_mb

    @staticmethod
    def get_memory_pressure() -> float:
        """Calculate memory pressure as a percentage (0-1) (static method for backward compatibility)."""
        monitor = get_cached_memory_monitor()
        status = monitor.get_memory_status()
        return status.percent_used

    @staticmethod
    def is_memory_pressure_high(threshold: float = 0.85) -> bool:
        """Check if memory pressure is above threshold (static method for backward compatibility)."""
        memory = _get_virtual_memory()
        return (memory.percent / 100.0) > threshold

    @staticmethod
    def estimate_array_memory(shape: tuple, dtype: np.dtype) -> float:
        """Estimate memory usage of a numpy array in MB.

        Parameters
        ----------
        shape : tuple
            Array shape
        dtype : np.dtype
            Array data type

        Returns
        -------
        float
            Estimated memory usage in MB
        """
        elements = np.prod(shape)
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = elements * bytes_per_element
        return total_bytes / (1024 * 1024)


# Global memory monitor instance
_memory_monitor = None


def get_cached_memory_monitor() -> MemoryMonitor:
    """Get or create a cached memory monitor instance.

    Returns
    -------
    MemoryMonitor
        Singleton memory monitor instance
    """
    global _memory_monitor  # noqa: PLW0603 - intentional singleton pattern
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor
