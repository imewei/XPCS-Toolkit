"""
Unified Memory Management System for XPCS Toolkit

This module provides a comprehensive memory management solution that consolidates
all caching strategies and implements intelligent memory pressure handling.
"""

import gc
import threading
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from enum import Enum
from typing import Any

import numpy as np
import psutil

from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


class CacheType(Enum):
    """Types of cached data for different management strategies."""

    COMPUTATION = "computation"  # Fitting results, FFT data, etc.
    ARRAY_DATA = "array_data"  # SAXS 2D, log data, etc.
    METADATA = "metadata"  # File headers, qmaps, etc.
    PLOT_DATA = "plot_data"  # Plot configurations, curves, etc.


class MemoryPressure(Enum):
    """Memory pressure levels for adaptive management."""

    LOW = "low"  # < 60% memory usage
    MODERATE = "moderate"  # 60-75% memory usage
    HIGH = "high"  # 75-85% memory usage
    CRITICAL = "critical"  # > 85% memory usage


class CacheEntry:
    """Enhanced cache entry with comprehensive metadata."""

    def __init__(self, data: Any, cache_type: CacheType, size_mb: float | None = None):
        self.data = data
        self.cache_type = cache_type
        self.size_mb = size_mb or self._estimate_size_mb(data)
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.access_frequency = 0.0  # accesses per second
        self.is_pinned = False  # Prevent eviction if True
        self.generation = 0  # For generational cache management

    def _estimate_size_mb(self, data: Any) -> float:
        """Estimate memory usage of cached data."""
        if isinstance(data, np.ndarray):
            return data.nbytes / (1024 * 1024)
        if hasattr(data, "__sizeof__"):
            return data.__sizeof__() / (1024 * 1024)
        import sys

        return sys.getsizeof(data) / (1024 * 1024)

    def touch(self) -> None:
        """Update access metadata."""
        current_time = time.time()
        self.access_count += 1

        # Calculate access frequency (exponential moving average)
        time_delta = current_time - self.last_accessed
        if time_delta > 0:
            recent_frequency = 1.0 / time_delta
            alpha = 0.3  # Smoothing factor
            self.access_frequency = (
                alpha * recent_frequency + (1 - alpha) * self.access_frequency
            )

        self.last_accessed = current_time

    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    def time_since_access(self) -> float:
        """Get time since last access in seconds."""
        return time.time() - self.last_accessed


class UnifiedMemoryManager:
    """
    Unified memory management system that consolidates all caching strategies.

    Features:
    - Predictive memory pressure detection
    - Intelligent cache eviction with multiple strategies
    - Type-aware cache management
    - Memory-mapped file support for large arrays
    - Automatic memory optimization
    """

    def __init__(
        self,
        max_memory_mb: float = 2048,
        pressure_thresholds: dict[str, float] | None = None,
        enable_monitoring: bool = True,
    ):
        self.max_memory_mb = max_memory_mb
        self.pressure_thresholds = pressure_thresholds or {
            "low": 0.6,
            "moderate": 0.75,
            "high": 0.85,
            "critical": 0.95,
        }
        self.enable_monitoring = enable_monitoring

        # Cache storage with type separation
        self._caches: dict[CacheType, OrderedDict[str, CacheEntry]] = {
            cache_type: OrderedDict() for cache_type in CacheType
        }

        # Memory tracking
        self._current_memory_mb = 0.0
        self._peak_memory_mb = 0.0
        self._memory_history: list[tuple[float, float]] = []  # (timestamp, memory_mb)

        # Thread safety
        self._cache_locks = {cache_type: threading.RLock() for cache_type in CacheType}
        self._global_lock = threading.RLock()

        # Weak references for automatic cleanup
        self._object_registry: weakref.WeakSet[Any] = weakref.WeakSet()

        # Monitoring and statistics
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "evictions": 0,
            "memory_pressure_events": 0,
            "automatic_cleanups": 0,
            "preload_hits": 0,
            "preload_requests": 0,
            "compression_saves_mb": 0,
        }

        # Advanced caching features
        self._preload_queue: dict[
            str, tuple[CacheType, Any, float]
        ] = {}  # key -> (cache_type, loader_func, priority)
        self._compression_enabled = True
        self._hot_data_threshold = 5  # Access count to mark as hot
        self._cache_partitions = self._setup_partitions()

        # Background monitoring
        self._monitoring_thread = None
        self._shutdown_event = threading.Event()

        if self.enable_monitoring:
            self._start_monitoring()

        logger.info(f"UnifiedMemoryManager initialized with {max_memory_mb}MB limit")

    def _make_space_in_partition(
        self, cache_type: CacheType, required_mb: float
    ) -> None:
        """Make space within a specific cache partition using intelligent eviction"""
        cache = self._caches[cache_type]
        partition_limit_mb = self.max_memory_mb * self._cache_partitions[cache_type]
        current_mb = sum(entry.size_mb for entry in cache.values())

        if current_mb + required_mb <= partition_limit_mb:
            return  # No eviction needed

        target_mb = partition_limit_mb * 0.7  # Leave 30% buffer
        evicted_mb = 0

        # First pass: evict aged items
        aged_keys = []
        for key, entry in cache.items():
            if not entry.is_pinned and entry.time_since_access() > 900:  # 15 minutes
                aged_keys.append((key, entry.size_mb))

        # Sort by age (oldest first)
        aged_keys.sort(key=lambda x: cache[x[0]].time_since_access(), reverse=True)

        for key, size_mb in aged_keys:
            if current_mb - evicted_mb <= target_mb:
                break
            self._evict_entry(cache_type, key)
            evicted_mb += size_mb

        # Second pass: LRU eviction if still needed
        if current_mb - evicted_mb > target_mb:
            self._evict_lru(cache_type, target_mb - (current_mb - evicted_mb))

    def preload_data(
        self, key: str, loader_func: callable, cache_type: CacheType, priority: int = 5
    ):
        """
        Schedule data for intelligent preloading.

        Parameters
        ----------
        key : str
            Cache key for the data
        loader_func : callable
            Function to load the data
        cache_type : CacheType
            Type of cache to store in
        priority : int
            Priority level (1-10, higher = more important)
        """
        self._preload_queue[key] = (cache_type, loader_func, priority)
        self._stats["preload_requests"] += 1
        logger.debug(f"Queued preload for {key} with priority {priority}")

    def _process_preload_queue(self):
        """Process preload queue during low memory pressure periods"""
        if self.get_memory_pressure() != MemoryPressure.LOW:
            return

        # Sort by priority (highest first)
        sorted_items = sorted(
            self._preload_queue.items(), key=lambda x: x[1][2], reverse=True
        )

        for key, (cache_type, loader_func, _priority) in sorted_items[
            :3
        ]:  # Process top 3
            try:
                if key not in self._caches[cache_type]:  # Only if not already cached
                    data = loader_func()
                    if self.cache_put(key, data, cache_type):
                        logger.debug(f"Successfully preloaded {key}")
                        self._stats["preload_hits"] += 1
                # Remove from queue after processing
                del self._preload_queue[key]
            except Exception as e:
                logger.debug(f"Preload failed for {key}: {e}")
                # Don't remove failed items immediately - retry later

    def _setup_partitions(self) -> dict[CacheType, float]:
        """Setup cache memory partitions with intelligent allocation"""
        return {
            CacheType.ARRAY_DATA: 0.5,  # 50% for large array data
            CacheType.COMPUTATION: 0.25,  # 25% for computation results
            CacheType.PLOT_DATA: 0.15,  # 15% for plot data
            CacheType.METADATA: 0.10,  # 10% for metadata
        }

    def _start_monitoring(self):
        """Start background memory monitoring thread."""
        import os

        # Skip starting background threads in test mode to prevent threading issues
        if os.environ.get("XPCS_TEST_MODE") == "1":
            return

        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self._monitoring_thread.start()

    def _monitoring_loop(self):
        """Background monitoring loop for memory pressure and optimization."""
        while not self._shutdown_event.wait(timeout=10.0):  # Check every 10 seconds
            try:
                self._update_memory_history()
                self._check_memory_pressure()
                self._perform_maintenance()

                # Process preload queue during low pressure periods
                if len(self._preload_queue) > 0:
                    self._process_preload_queue()

            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")

    def _update_memory_history(self):
        """Update memory usage history for trend analysis."""
        system_memory = psutil.virtual_memory()
        current_time = time.time()

        with self._global_lock:
            self._memory_history.append((current_time, system_memory.percent))

            # Keep only last hour of history
            cutoff_time = current_time - 3600
            self._memory_history = [
                (t, m) for t, m in self._memory_history if t > cutoff_time
            ]

    def _check_memory_pressure(self):
        """Check for memory pressure and trigger appropriate responses."""
        pressure = self.get_memory_pressure()

        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            self._stats["memory_pressure_events"] += 1
            logger.warning(f"Memory pressure detected: {pressure.value}")

            # Trigger appropriate cleanup based on pressure level
            if pressure == MemoryPressure.CRITICAL:
                self._emergency_cleanup()
            else:
                self._aggressive_cleanup()

    def _perform_maintenance(self):
        """Perform routine cache maintenance."""
        with self._global_lock:
            # Update cache generation for aging-based eviction
            time.time()

            for cache_type in CacheType:
                with self._cache_locks[cache_type]:
                    cache = self._caches[cache_type]

                    # Age out old entries
                    aged_keys = []
                    for key, entry in cache.items():
                        if entry.time_since_access() > 1800:  # 30 minutes
                            aged_keys.append(key)

                    for key in aged_keys:
                        if not cache[key].is_pinned:
                            self._evict_entry(cache_type, key)

    def get_memory_pressure(self) -> MemoryPressure:
        """Get current memory pressure level."""
        system_memory = psutil.virtual_memory()
        pressure_ratio = system_memory.percent / 100.0

        if pressure_ratio >= self.pressure_thresholds["critical"]:
            return MemoryPressure.CRITICAL
        if pressure_ratio >= self.pressure_thresholds["high"]:
            return MemoryPressure.HIGH
        if pressure_ratio >= self.pressure_thresholds["moderate"]:
            return MemoryPressure.MODERATE
        return MemoryPressure.LOW

    def cache_put(
        self, key: str, data: Any, cache_type: CacheType, pin: bool = False
    ) -> bool:
        """
        Store data in the appropriate cache with intelligent eviction.

        Parameters
        ----------
        key : str
            Unique cache key
        data : Any
            Data to cache
        cache_type : CacheType
            Type of cache for management strategy
        pin : bool
            Prevent eviction if True

        Returns
        -------
        bool
            True if successfully cached
        """
        # Check partition limits before creating entry
        partition_limit_mb = self.max_memory_mb * self._cache_partitions[cache_type]

        entry = CacheEntry(data, cache_type)
        entry.is_pinned = pin

        # Check if data is too large for its partition
        if entry.size_mb > partition_limit_mb * 0.8:  # 80% of partition
            logger.warning(
                f"Data too large for {cache_type.value} partition: {entry.size_mb:.1f}MB > {partition_limit_mb * 0.8:.1f}MB"
            )
            return False

        with self._cache_locks[cache_type]:
            cache = self._caches[cache_type]

            # Calculate current partition usage
            current_partition_mb = sum(entry.size_mb for entry in cache.values())

            # Make space if needed within partition
            if current_partition_mb + entry.size_mb > partition_limit_mb:
                self._make_space_in_partition(cache_type, entry.size_mb)

            # Check if we need to make space
            if entry.size_mb > self.max_memory_mb * 0.5:
                logger.warning(f"Data too large to cache: {entry.size_mb:.1f}MB")
                return False

            # Evict if necessary
            required_space = entry.size_mb
            if self._current_memory_mb + required_space > self.max_memory_mb:
                self._make_space(cache_type, required_space)

            # Remove existing entry if key exists
            if key in cache:
                old_entry = cache[key]
                self._current_memory_mb -= old_entry.size_mb
                del cache[key]

            # Add new entry
            cache[key] = entry
            self._current_memory_mb += entry.size_mb

            # Update peak memory
            self._peak_memory_mb = max(self._peak_memory_mb, self._current_memory_mb)

            logger.debug(
                f"Cached {key} ({entry.size_mb:.1f}MB, type: {cache_type.value})"
            )
            return True

    def cache_get(self, key: str, cache_type: CacheType) -> Any | None:
        """
        Retrieve data from cache.

        Parameters
        ----------
        key : str
            Cache key
        cache_type : CacheType
            Type of cache to search

        Returns
        -------
        Any or None
            Cached data or None if not found
        """
        with self._cache_locks[cache_type]:
            cache = self._caches[cache_type]

            if key in cache:
                entry = cache[key]
                entry.touch()

                # Move to end (LRU)
                cache.move_to_end(key)

                self._stats["cache_hits"] += 1
                logger.debug(f"Cache hit for {key}")
                return entry.data

            self._stats["cache_misses"] += 1
            logger.debug(f"Cache miss for {key}")
            return None

    def _make_space(self, cache_type: CacheType, required_mb: float):
        """Make space in cache using intelligent eviction strategies."""
        freed_mb = 0.0

        # Strategy 1: Evict from same cache type first
        freed_mb += self._evict_lru(cache_type, required_mb * 0.7)

        # Strategy 2: Evict from less critical cache types
        if freed_mb < required_mb:
            eviction_order = [
                CacheType.PLOT_DATA,
                CacheType.COMPUTATION,
                CacheType.METADATA,
                CacheType.ARRAY_DATA,
            ]

            for other_type in eviction_order:
                if other_type != cache_type and freed_mb < required_mb:
                    freed_mb += self._evict_lru(
                        other_type, (required_mb - freed_mb) * 1.2
                    )

        # Strategy 3: Age-based eviction
        if freed_mb < required_mb:
            freed_mb += self._evict_by_age(1800)  # 30 minutes

        return freed_mb

    def _evict_lru(self, cache_type: CacheType, target_mb: float) -> float:
        """Evict least recently used entries from specified cache."""
        freed_mb = 0.0

        with self._cache_locks[cache_type]:
            cache = self._caches[cache_type]

            # Sort by last access time (oldest first)
            sorted_items = sorted(cache.items(), key=lambda x: x[1].last_accessed)

            keys_to_remove = []
            for key, entry in sorted_items:
                if freed_mb >= target_mb:
                    break

                if not entry.is_pinned:
                    keys_to_remove.append(key)
                    freed_mb += entry.size_mb

            # Remove entries
            for key in keys_to_remove:
                self._evict_entry(cache_type, key)

        return freed_mb

    def _evict_by_age(self, max_age_seconds: float) -> float:
        """Evict entries older than specified age."""
        freed_mb = 0.0
        current_time = time.time()

        for cache_type in CacheType:
            with self._cache_locks[cache_type]:
                cache = self._caches[cache_type]

                aged_keys = []
                for key, entry in cache.items():
                    if (
                        current_time - entry.created_at > max_age_seconds
                        and not entry.is_pinned
                    ):
                        aged_keys.append(key)

                for key in aged_keys:
                    entry = cache[key]
                    freed_mb += entry.size_mb
                    self._evict_entry(cache_type, key)

        return freed_mb

    def _evict_entry(self, cache_type: CacheType, key: str):
        """Remove entry from cache and update counters."""
        with self._cache_locks[cache_type]:
            cache = self._caches[cache_type]

            if key in cache:
                entry = cache[key]
                self._current_memory_mb -= entry.size_mb
                del cache[key]
                self._stats["evictions"] += 1

                logger.debug(f"Evicted {key} ({entry.size_mb:.1f}MB)")

    def _aggressive_cleanup(self):
        """Perform aggressive cleanup under memory pressure."""
        logger.info("Performing aggressive memory cleanup")

        # Clear non-essential caches
        self.clear_cache_type(CacheType.PLOT_DATA)

        # Evict old computation results
        self._evict_by_age(900)  # 15 minutes

        # Force garbage collection
        gc.collect()

        self._stats["automatic_cleanups"] += 1

    def _emergency_cleanup(self):
        """Emergency cleanup under critical memory pressure."""
        logger.warning("Performing emergency memory cleanup")

        # Clear all non-pinned caches
        for cache_type in [CacheType.PLOT_DATA, CacheType.COMPUTATION]:
            self.clear_cache_type(cache_type)

        # Aggressive eviction from remaining caches
        self._evict_by_age(300)  # 5 minutes

        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()

        self._stats["automatic_cleanups"] += 1

    def clear_cache_type(self, cache_type: CacheType):
        """Clear all entries from specified cache type."""
        with self._cache_locks[cache_type]:
            cache = self._caches[cache_type]

            # Calculate freed memory
            freed_mb = sum(entry.size_mb for entry in cache.values())

            cache.clear()
            self._current_memory_mb -= freed_mb

            logger.info(f"Cleared {cache_type.value} cache, freed {freed_mb:.1f}MB")

    def clear_all_caches(self):
        """Clear all caches."""
        for cache_type in CacheType:
            self.clear_cache_type(cache_type)

        logger.info("Cleared all caches")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._global_lock:
            stats = self._stats.copy()

            # Add current state
            stats.update(
                {
                    "current_memory_mb": self._current_memory_mb,
                    "peak_memory_mb": self._peak_memory_mb,
                    "max_memory_mb": self.max_memory_mb,
                    "memory_utilization": self._current_memory_mb / self.max_memory_mb,
                    "cache_efficiency": (
                        stats["cache_hits"]
                        / max(1, stats["cache_hits"] + stats["cache_misses"])
                    ),
                    "memory_pressure": self.get_memory_pressure().value,
                }
            )

            # Add per-type statistics
            for cache_type in CacheType:
                with self._cache_locks[cache_type]:
                    cache = self._caches[cache_type]
                    type_memory = sum(entry.size_mb for entry in cache.values())

                    stats[f"{cache_type.value}_entries"] = len(cache)
                    stats[f"{cache_type.value}_memory_mb"] = type_memory

            return stats

    def shutdown(self):
        """Shutdown memory manager and cleanup resources."""
        if self._monitoring_thread:
            self._shutdown_event.set()
            self._monitoring_thread.join(timeout=5.0)

        self.clear_all_caches()
        logger.info("UnifiedMemoryManager shutdown complete")

    def get_enhanced_stats(self) -> dict[str, Any]:
        """Get enhanced statistics including new caching features"""
        stats = self.get_cache_stats()

        # Add new feature statistics
        stats.update(
            {
                "preload_hit_ratio": (
                    stats["preload_hits"] / max(1, stats["preload_requests"])
                ),
                "preload_queue_size": len(self._preload_queue),
                "partition_usage": {
                    cache_type.value: {
                        "limit_mb": self.max_memory_mb
                        * self._cache_partitions[cache_type],
                        "used_mb": sum(
                            entry.size_mb for entry in self._caches[cache_type].values()
                        ),
                        "utilization": sum(
                            entry.size_mb for entry in self._caches[cache_type].values()
                        )
                        / (self.max_memory_mb * self._cache_partitions[cache_type]),
                    }
                    for cache_type in CacheType
                },
            }
        )

        return stats


# Global memory manager instance
_global_memory_manager: UnifiedMemoryManager | None = None


def get_memory_manager() -> UnifiedMemoryManager:
    """Get or create the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = UnifiedMemoryManager()
    return _global_memory_manager


def shutdown_memory_manager():
    """Shutdown the global memory manager."""
    global _global_memory_manager
    if _global_memory_manager:
        _global_memory_manager.shutdown()
        _global_memory_manager = None


# Convenience functions for common operations
def cache_computation(key: str, data: Any) -> bool:
    """Cache computation result."""
    return get_memory_manager().cache_put(key, data, CacheType.COMPUTATION)


def get_computation(key: str) -> Any | None:
    """Get cached computation result."""
    return get_memory_manager().cache_get(key, CacheType.COMPUTATION)


def cache_array(key: str, array: np.ndarray) -> bool:
    """Cache large array data."""
    return get_memory_manager().cache_put(key, array, CacheType.ARRAY_DATA)


def get_array(key: str) -> np.ndarray | None:
    """Get cached array data."""
    return get_memory_manager().cache_get(key, CacheType.ARRAY_DATA)


@contextmanager
def memory_pressure_monitor():
    """Context manager for monitoring memory pressure during operations."""
    manager = get_memory_manager()
    initial_pressure = manager.get_memory_pressure()
    initial_memory = manager._current_memory_mb

    try:
        yield manager
    finally:
        final_pressure = manager.get_memory_pressure()
        final_memory = manager._current_memory_mb

        if final_pressure != initial_pressure:
            logger.info(
                f"Memory pressure changed: {initial_pressure.value} -> {final_pressure.value}"
            )

        memory_delta = final_memory - initial_memory
        if abs(memory_delta) > 50:  # 50MB threshold
            logger.info(f"Significant memory change: {memory_delta:+.1f}MB")
