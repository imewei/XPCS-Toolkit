"""Data caching utilities for XpcsFile.

This module provides LRU caching with memory management for XPCS data.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np

from xpcsviewer.utils.logging_config import get_logger
from xpcsviewer.xpcs_file.memory import MemoryMonitor

logger = get_logger(__name__)


class CacheItem:
    """Individual cache item with metadata."""

    def __init__(self, data: Any, size_mb: float):
        self.data = data
        self.size_mb = size_mb
        self.access_count = 0
        self.last_accessed = 0
        self.created_at = 0
        self._update_access_time()

    def _update_access_time(self):
        """Update access timestamp and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1
        if self.created_at == 0:
            self.created_at = self.last_accessed

    def touch(self):
        """Mark item as accessed."""
        self._update_access_time()


class DataCache:
    """
    LRU cache with memory limit and automatic cleanup for XPCS data.

    Features:
    - LRU eviction policy
    - Memory limit enforcement (default 500MB)
    - Automatic cleanup on memory pressure
    - Memory usage tracking per item
    - Thread-safe operations
    """

    def __init__(
        self, max_memory_mb: float = 500.0, memory_pressure_threshold: float = 0.85
    ):
        self.max_memory_mb = max_memory_mb
        self.memory_pressure_threshold = memory_pressure_threshold
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._current_memory_mb = 0.0
        self._lock = threading.RLock()
        self._cleanup_in_progress = False

        logger.info(f"DataCache initialized with {max_memory_mb}MB limit")

    def _generate_key(self, file_path: str, data_type: str) -> str:
        """Generate cache key from file path and data type."""
        return f"{file_path}:{data_type}"

    def _evict_lru_items(self, required_memory_mb: float = 0) -> float:
        """
        Evict least recently used items to free memory.

        Parameters
        ----------
        required_memory_mb : float
            Minimum memory to free

        Returns
        -------
        float
            Amount of memory freed in MB
        """
        freed_memory = 0.0
        items_to_remove = []

        # Sort by last access time (oldest first)
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)

        for key, item in sorted_items:
            if (
                freed_memory >= required_memory_mb
                and self._current_memory_mb <= self.max_memory_mb
            ):
                break

            items_to_remove.append(key)
            freed_memory += item.size_mb
            self._current_memory_mb -= item.size_mb

            logger.debug(f"Evicting cache item {key}, size: {item.size_mb:.2f}MB")

        # Remove items from cache
        for key in items_to_remove:
            del self._cache[key]

        return freed_memory

    def _cleanup_on_memory_pressure(self):
        """Perform cleanup when system memory pressure is high."""
        if self._cleanup_in_progress:
            return

        self._cleanup_in_progress = True
        try:
            if MemoryMonitor.is_memory_pressure_high(self.memory_pressure_threshold):
                # Aggressive cleanup: remove 50% of cache
                target_memory = self.max_memory_mb * 0.5
                freed = self._evict_lru_items(self._current_memory_mb - target_memory)
                logger.info(f"Memory pressure cleanup: freed {freed:.2f}MB")
        finally:
            self._cleanup_in_progress = False

    def put(self, file_path: str, data_type: str, data: Any) -> bool:
        """
        Store data in cache.

        Parameters
        ----------
        file_path : str
            File path identifier
        data_type : str
            Type of data ('saxs_2d', 'saxs_2d_log', etc.)
        data : Any
            Data to cache

        Returns
        -------
        bool
            True if successfully cached
        """
        with self._lock:
            key = self._generate_key(file_path, data_type)

            # Estimate memory usage
            if isinstance(data, np.ndarray):
                size_mb = MemoryMonitor.estimate_array_memory(data.shape, data.dtype)
            else:
                # Rough estimate for other data types
                size_mb = sys.getsizeof(data) / (1024 * 1024)

            # Check if data is too large for cache
            if size_mb > self.max_memory_mb * 0.8:
                logger.warning(
                    f"Data too large for cache: {size_mb:.2f}MB > {self.max_memory_mb * 0.8:.2f}MB"
                )
                return False

            # Evict items if necessary
            required_memory = size_mb
            if self._current_memory_mb + required_memory > self.max_memory_mb:
                self._evict_lru_items(required_memory)

            # Remove existing item if it exists
            if key in self._cache:
                old_item = self._cache[key]
                self._current_memory_mb -= old_item.size_mb
                del self._cache[key]

            # Add new item
            cache_item = CacheItem(data, size_mb)
            self._cache[key] = cache_item
            self._current_memory_mb += size_mb

            # Check for memory pressure
            self._cleanup_on_memory_pressure()

            logger.debug(
                f"Cached {key}, size: {size_mb:.2f}MB, total: {self._current_memory_mb:.2f}MB"
            )
            return True

    def get(self, file_path: str, data_type: str) -> Any | None:
        """
        Retrieve data from cache.

        Parameters
        ----------
        file_path : str
            File path identifier
        data_type : str
            Type of data

        Returns
        -------
        Any or None
            Cached data or None if not found
        """
        with self._lock:
            key = self._generate_key(file_path, data_type)

            if key in self._cache:
                item = self._cache[key]
                item.touch()
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                logger.debug(f"Cache hit for {key}")
                return item.data

            logger.debug(f"Cache miss for {key}")
            return None

    def clear(self):
        """Clear all cached data."""
        with self._lock:
            self._cache.clear()
            self._current_memory_mb = 0.0
            logger.info("Cache cleared")

    def clear_file(self, file_path: str):
        """Clear all cached data for a specific file."""
        with self._lock:
            keys_to_remove = [
                key for key in self._cache if key.startswith(f"{file_path}:")
            ]
            freed_memory = 0.0

            for key in keys_to_remove:
                item = self._cache[key]
                freed_memory += item.size_mb
                self._current_memory_mb -= item.size_mb
                del self._cache[key]

            if freed_memory > 0:
                logger.debug(
                    f"Cleared {len(keys_to_remove)} items for {file_path}, freed {freed_memory:.2f}MB"
                )

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            used_mb, available_mb = MemoryMonitor.get_memory_usage()
            pressure = MemoryMonitor.get_memory_pressure()

            return {
                "cache_items": len(self._cache),
                "cache_memory_mb": self._current_memory_mb,
                "cache_limit_mb": self.max_memory_mb,
                "cache_utilization": self._current_memory_mb / self.max_memory_mb,
                "system_memory_used_mb": used_mb,
                "system_memory_available_mb": available_mb,
                "system_memory_pressure": pressure,
                "items_by_type": {},
            }

    def force_cleanup(self, target_memory_mb: float | None = None):
        """Force cleanup to target memory usage."""
        with self._lock:
            if target_memory_mb is None:
                target_memory_mb = self.max_memory_mb * 0.5

            if self._current_memory_mb > target_memory_mb:
                freed = self._evict_lru_items(
                    self._current_memory_mb - target_memory_mb
                )
                logger.info(f"Forced cleanup: freed {freed:.2f}MB")
