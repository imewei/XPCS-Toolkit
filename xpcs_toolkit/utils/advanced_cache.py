"""
Advanced Multi-Level Caching System for XPCS Toolkit.

This module implements a sophisticated caching architecture with L1/L2/L3 tiers,
intelligent eviction policies, compression, and adaptive memory management.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import pickle
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .logging_config import get_logger
from .memory_utils import SystemMemoryMonitor

logger = get_logger(__name__)


class CacheLevel(Enum):
    """Cache level enumeration."""

    L1 = "L1"  # Hot memory cache
    L2 = "L2"  # Compressed memory cache
    L3 = "L3"  # Persistent disk cache


class EvictionPolicy(Enum):
    """Cache eviction policy enumeration."""

    LRU = "LRU"  # Least Recently Used
    LFU = "LFU"  # Least Frequently Used
    TTL = "TTL"  # Time To Live
    MIXED = "MIXED"  # Mixed LRU + LFU + TTL


@dataclass
class CacheEntry:
    """Individual cache entry with comprehensive metadata."""

    key: str
    data: Any = None
    compressed_data: bytes = field(default=None)
    disk_path: Path | None = None

    # Timing information
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    last_modified: float = field(default_factory=time.time)
    ttl_expires_at: float | None = None

    # Usage statistics
    access_count: int = 0
    access_frequency: float = 0.0  # accesses per second

    # Size information
    memory_size_bytes: int = 0
    compressed_size_bytes: int = 0
    disk_size_bytes: int = 0

    # Metadata
    cache_level: CacheLevel = CacheLevel.L1
    data_type: str = "unknown"
    checksum: str | None = None
    compression_ratio: float = 1.0

    def touch(self):
        """Update access statistics."""
        current_time = time.time()
        self.access_count += 1
        self.last_accessed = current_time

        # Calculate frequency (accesses per second since creation)
        age_seconds = max(current_time - self.created_at, 1.0)
        self.access_frequency = self.access_count / age_seconds

    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl_expires_at is None:
            return False
        return time.time() > self.ttl_expires_at

    def calculate_priority_score(
        self, policy: EvictionPolicy = EvictionPolicy.MIXED
    ) -> float:
        """Calculate priority score for eviction (lower = more likely to evict)."""
        current_time = time.time()
        age_hours = (current_time - self.last_accessed) / 3600.0

        if policy == EvictionPolicy.LRU:
            return -self.last_accessed  # Older = lower score
        if policy == EvictionPolicy.LFU:
            return self.access_frequency
        if policy == EvictionPolicy.TTL:
            if self.ttl_expires_at:
                return self.ttl_expires_at - current_time
            return float("inf")
        # MIXED policy
        # Combined score: frequency (40%) + recency (40%) + size efficiency (20%)
        recency_score = 1.0 / (1.0 + age_hours)  # Higher for recent access
        frequency_score = (
            min(self.access_frequency, 10.0) / 10.0
        )  # Normalized frequency
        size_efficiency = self.compression_ratio if self.compression_ratio > 0 else 1.0

        return (
            0.4 * frequency_score
            + 0.4 * recency_score
            + 0.2 * min(size_efficiency, 2.0) / 2.0
        )


class CacheStatistics:
    """Comprehensive cache statistics tracking."""

    def __init__(self):
        self.start_time = time.time()
        self._lock = threading.RLock()

        # Hit/Miss statistics by level
        self.l1_hits = 0
        self.l1_misses = 0
        self.l2_hits = 0
        self.l2_misses = 0
        self.l3_hits = 0
        self.l3_misses = 0

        # Operation counts
        self.total_gets = 0
        self.total_puts = 0
        self.total_evictions = 0
        self.total_compressions = 0
        self.total_decompressions = 0

        # Memory usage
        self.current_l1_memory_mb = 0.0
        self.current_l2_memory_mb = 0.0
        self.peak_l1_memory_mb = 0.0
        self.peak_l2_memory_mb = 0.0

        # Disk usage
        self.current_l3_disk_mb = 0.0
        self.peak_l3_disk_mb = 0.0

        # Performance metrics
        self.avg_get_time_ms = 0.0
        self.avg_put_time_ms = 0.0
        self.avg_compression_time_ms = 0.0
        self.total_bytes_compressed = 0
        self.total_bytes_saved = 0

    def record_hit(self, level: CacheLevel, time_ms: float):
        """Record cache hit."""
        with self._lock:
            self.total_gets += 1
            if level == CacheLevel.L1:
                self.l1_hits += 1
            elif level == CacheLevel.L2:
                self.l2_hits += 1
            else:
                self.l3_hits += 1

            # Update average get time
            total_ops = self.l1_hits + self.l2_hits + self.l3_hits
            self.avg_get_time_ms = (
                (self.avg_get_time_ms * (total_ops - 1) + time_ms) / total_ops
                if total_ops > 0
                else time_ms
            )

    def record_miss(self, level: CacheLevel):
        """Record cache miss."""
        with self._lock:
            if level == CacheLevel.L1:
                self.l1_misses += 1
            elif level == CacheLevel.L2:
                self.l2_misses += 1
            else:
                self.l3_misses += 1

    def record_put(self, time_ms: float):
        """Record cache put operation."""
        with self._lock:
            self.total_puts += 1
            self.avg_put_time_ms = (
                self.avg_put_time_ms * (self.total_puts - 1) + time_ms
            ) / self.total_puts

    def record_eviction(self):
        """Record cache eviction."""
        with self._lock:
            self.total_evictions += 1

    def record_compression(
        self, original_bytes: int, compressed_bytes: int, time_ms: float
    ):
        """Record compression operation."""
        with self._lock:
            self.total_compressions += 1
            self.total_bytes_compressed += original_bytes
            self.total_bytes_saved += original_bytes - compressed_bytes

            self.avg_compression_time_ms = (
                self.avg_compression_time_ms * (self.total_compressions - 1) + time_ms
            ) / self.total_compressions

    def update_memory_usage(self, l1_mb: float, l2_mb: float):
        """Update memory usage statistics."""
        with self._lock:
            self.current_l1_memory_mb = l1_mb
            self.current_l2_memory_mb = l2_mb
            self.peak_l1_memory_mb = max(self.peak_l1_memory_mb, l1_mb)
            self.peak_l2_memory_mb = max(self.peak_l2_memory_mb, l2_mb)

    def update_disk_usage(self, l3_mb: float):
        """Update disk usage statistics."""
        with self._lock:
            self.current_l3_disk_mb = l3_mb
            self.peak_l3_disk_mb = max(self.peak_l3_disk_mb, l3_mb)

    def get_hit_rate(self) -> dict[str, float]:
        """Calculate hit rates for each level."""
        with self._lock:
            total_l1 = self.l1_hits + self.l1_misses
            total_l2 = self.l2_hits + self.l2_misses
            total_l3 = self.l3_hits + self.l3_misses
            total_overall = total_l1 + total_l2 + total_l3

            return {
                "l1_hit_rate": self.l1_hits / total_l1 if total_l1 > 0 else 0.0,
                "l2_hit_rate": self.l2_hits / total_l2 if total_l2 > 0 else 0.0,
                "l3_hit_rate": self.l3_hits / total_l3 if total_l3 > 0 else 0.0,
                "overall_hit_rate": (self.l1_hits + self.l2_hits + self.l3_hits)
                / total_overall
                if total_overall > 0
                else 0.0,
            }

    def get_compression_efficiency(self) -> dict[str, float]:
        """Calculate compression efficiency metrics."""
        with self._lock:
            if self.total_bytes_compressed == 0:
                return {"compression_ratio": 1.0, "bytes_saved_percentage": 0.0}

            compression_ratio = (
                self.total_bytes_compressed - self.total_bytes_saved
            ) / self.total_bytes_compressed
            bytes_saved_percentage = (
                self.total_bytes_saved / self.total_bytes_compressed
            ) * 100.0

            return {
                "compression_ratio": compression_ratio,
                "bytes_saved_percentage": bytes_saved_percentage,
            }

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive statistics summary."""
        with self._lock:
            runtime_hours = (time.time() - self.start_time) / 3600.0
            hit_rates = self.get_hit_rate()
            compression_stats = self.get_compression_efficiency()

            return {
                "runtime_hours": runtime_hours,
                "hit_rates": hit_rates,
                "operation_counts": {
                    "total_gets": self.total_gets,
                    "total_puts": self.total_puts,
                    "total_evictions": self.total_evictions,
                    "total_compressions": self.total_compressions,
                },
                "memory_usage_mb": {
                    "current_l1": self.current_l1_memory_mb,
                    "current_l2": self.current_l2_memory_mb,
                    "peak_l1": self.peak_l1_memory_mb,
                    "peak_l2": self.peak_l2_memory_mb,
                    "current_l3_disk": self.current_l3_disk_mb,
                    "peak_l3_disk": self.peak_l3_disk_mb,
                },
                "performance_ms": {
                    "avg_get_time": self.avg_get_time_ms,
                    "avg_put_time": self.avg_put_time_ms,
                    "avg_compression_time": self.avg_compression_time_ms,
                },
                "compression": compression_stats,
            }


class MultiLevelCache:
    """
    Advanced multi-level caching system with L1/L2/L3 tiers.

    L1: Hot memory cache for frequently accessed data
    L2: Compressed memory cache for intermediate storage
    L3: Persistent disk cache for long-term storage
    """

    def __init__(
        self,
        l1_max_memory_mb: float = 500.0,
        l2_max_memory_mb: float = 1000.0,
        l3_max_disk_mb: float = 5000.0,
        l3_cache_dir: Path | None = None,
        eviction_policy: EvictionPolicy = EvictionPolicy.MIXED,
        compression_level: int = 6,
        enable_checksums: bool = True,
        cleanup_interval_seconds: float = 300.0,
    ):  # 5 minutes
        # Configuration
        self.l1_max_memory_mb = l1_max_memory_mb
        self.l2_max_memory_mb = l2_max_memory_mb
        self.l3_max_disk_mb = l3_max_disk_mb
        self.eviction_policy = eviction_policy
        self.compression_level = compression_level
        self.enable_checksums = enable_checksums
        self.cleanup_interval_seconds = cleanup_interval_seconds

        # Cache storage
        self._l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l2_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._l3_index: OrderedDict[str, CacheEntry] = OrderedDict()

        # Thread safety
        self._lock = threading.RLock()
        self._cleanup_lock = threading.RLock()

        # Disk cache directory
        if l3_cache_dir is None:
            self.l3_cache_dir = Path(tempfile.gettempdir()) / "xpcs_cache"
        else:
            self.l3_cache_dir = Path(l3_cache_dir)

        self.l3_cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = CacheStatistics()

        # Background cleanup
        self._cleanup_thread = None
        self._cleanup_stop_event = threading.Event()
        self._start_cleanup_thread()

        # Memory pressure monitoring
        self._memory_pressure_threshold = 0.85

        logger.info(
            f"MultiLevelCache initialized: L1={l1_max_memory_mb}MB, L2={l2_max_memory_mb}MB, L3={l3_max_disk_mb}MB"
        )

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        # Skip starting background threads in test mode
        if os.environ.get("XPCS_TEST_MODE", "").lower() in ("1", "true"):
            logger.debug("Skipping cleanup thread start in test mode")
            return

        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker, daemon=True
        )
        self._cleanup_thread.start()

    def _cleanup_worker(self):
        """Background worker for cache maintenance."""
        while not self._cleanup_stop_event.wait(self.cleanup_interval_seconds):
            try:
                self._perform_maintenance()
            except Exception as e:
                logger.error(f"Error in cache cleanup worker: {e}")

    def _perform_maintenance(self):
        """Perform routine cache maintenance."""
        with self._cleanup_lock:
            # Remove expired entries
            self._remove_expired_entries()

            # Check memory pressure and cleanup if needed
            if SystemMemoryMonitor.check_memory_pressure(
                self._memory_pressure_threshold * 100
            ):
                logger.info(
                    "High memory pressure detected, performing aggressive cleanup"
                )
                self._cleanup_on_memory_pressure()

            # Update statistics
            self._update_usage_statistics()

            # Cleanup orphaned disk files
            self._cleanup_orphaned_disk_files()

    def _remove_expired_entries(self):
        """Remove expired entries from all cache levels."""
        expired_keys = []

        # Check L1
        for key, entry in list(self._l1_cache.items()):
            if entry.is_expired():
                expired_keys.append((key, CacheLevel.L1))

        # Check L2
        for key, entry in list(self._l2_cache.items()):
            if entry.is_expired():
                expired_keys.append((key, CacheLevel.L2))

        # Check L3
        for key, entry in list(self._l3_index.items()):
            if entry.is_expired():
                expired_keys.append((key, CacheLevel.L3))

        # Remove expired entries
        for key, level in expired_keys:
            self._remove_entry(key, level)
            logger.debug(f"Removed expired entry {key} from {level}")

    def _cleanup_on_memory_pressure(self):
        """Aggressive cleanup when memory pressure is high."""
        # Promote data to lower levels and free memory
        self._promote_l1_to_l2(force=True, target_freed_mb=self.l1_max_memory_mb * 0.5)
        self._promote_l2_to_l3(force=True, target_freed_mb=self.l2_max_memory_mb * 0.3)

    def _update_usage_statistics(self):
        """Update memory and disk usage statistics."""
        l1_memory = sum(
            entry.memory_size_bytes for entry in self._l1_cache.values()
        ) / (1024 * 1024)
        l2_memory = sum(
            entry.compressed_size_bytes for entry in self._l2_cache.values()
        ) / (1024 * 1024)
        l3_disk = sum(entry.disk_size_bytes for entry in self._l3_index.values()) / (
            1024 * 1024
        )

        self.stats.update_memory_usage(l1_memory, l2_memory)
        self.stats.update_disk_usage(l3_disk)

    def _cleanup_orphaned_disk_files(self):
        """Remove disk files that are no longer referenced."""
        if not self.l3_cache_dir.exists():
            return

        # Get all files in cache directory
        disk_files = set()
        try:
            for file_path in self.l3_cache_dir.rglob("*.cache"):
                disk_files.add(file_path)
        except Exception as e:
            logger.error(f"Error scanning cache directory: {e}")
            return

        # Get referenced files
        referenced_files = set()
        for entry in self._l3_index.values():
            if entry.disk_path and entry.disk_path.exists():
                referenced_files.add(entry.disk_path)

        # Remove orphaned files
        orphaned_files = disk_files - referenced_files
        for file_path in orphaned_files:
            try:
                file_path.unlink()
                logger.debug(f"Removed orphaned cache file: {file_path}")
            except Exception as e:
                logger.debug(f"Failed to remove orphaned file {file_path}: {e}")

    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate a unique cache key from arguments."""
        key_parts = []

        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, (list, tuple)):
                key_parts.append(str(sorted(arg) if isinstance(arg, list) else arg))
            elif isinstance(arg, dict):
                key_parts.append(str(sorted(arg.items())))
            elif isinstance(arg, np.ndarray):
                # Use shape, dtype, and a sample of values for arrays
                if arg.size > 1000:
                    sample = arg.flat[:100:10]  # Sample every 10th element of first 100
                else:
                    sample = arg.flatten()[:100]  # First 100 elements
                key_parts.append(
                    f"array_{arg.shape}_{arg.dtype}_{hash(sample.tobytes())}"
                )
            else:
                key_parts.append(str(hash(str(arg))))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # Create hash from concatenated parts
        cache_string = "|".join(key_parts)
        return hashlib.sha256(cache_string.encode()).hexdigest()[
            :16
        ]  # Use shorter hash

    def _calculate_data_size(self, data: Any) -> int:
        """Calculate memory size of data in bytes."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        if isinstance(data, (bytes, bytearray)):
            return len(data)
        # Use pickle to estimate size for other objects
        try:
            return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback to sys.getsizeof for unpickleable objects
            import sys

            return sys.getsizeof(data)

    def _calculate_checksum(self, data: Any) -> str:
        """Calculate checksum for data integrity verification."""
        if not self.enable_checksums:
            return None

        try:
            if isinstance(data, np.ndarray):
                return hashlib.md5(data.tobytes(), usedforsecurity=False).hexdigest()
            if isinstance(data, (bytes, bytearray)):
                return hashlib.md5(data, usedforsecurity=False).hexdigest()
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(serialized, usedforsecurity=False).hexdigest()
        except Exception as e:
            logger.debug(f"Failed to calculate checksum: {e}")
            return None

    def _compress_data(self, data: Any) -> tuple[bytes, float]:
        """Compress data and return (compressed_bytes, compression_time_ms)."""
        start_time = time.time()

        try:
            # Serialize data first
            if isinstance(data, np.ndarray):
                # Use numpy-specific serialization for better compression
                serialized = data.tobytes()
            else:
                serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

            # Compress using gzip
            compressed = gzip.compress(serialized, compresslevel=self.compression_level)

            compression_time_ms = (time.time() - start_time) * 1000.0
            return compressed, compression_time_ms

        except Exception as e:
            logger.error(f"Failed to compress data: {e}")
            # Return uncompressed data as fallback
            compression_time_ms = (time.time() - start_time) * 1000.0
            return pickle.dumps(
                data, protocol=pickle.HIGHEST_PROTOCOL
            ), compression_time_ms

    def _decompress_data(
        self, compressed_data: bytes, data_type: str = "unknown"
    ) -> Any:
        """Decompress data back to original form."""
        try:
            decompressed = gzip.decompress(compressed_data)

            if data_type.startswith("array_"):
                # Parse array metadata from data_type
                # Format: "array_shape_dtype"
                parts = data_type.split("_")
                if len(parts) >= 3:
                    shape_str = parts[1]
                    dtype_str = parts[2]

                    # Parse shape tuple safely
                    import ast

                    try:
                        shape = (
                            ast.literal_eval(shape_str)
                            if shape_str.startswith("(")
                            else (int(shape_str),)
                        )
                    except (ValueError, SyntaxError) as e:
                        logger.warning(
                            f"Failed to parse shape string '{shape_str}': {e}"
                        )
                        return None
                    dtype = np.dtype(dtype_str)

                    # Reconstruct array
                    return np.frombuffer(decompressed, dtype=dtype).reshape(shape)

            # Default: use pickle (safe for internal cache data)
            # Note: This pickle usage is safe as it only deserializes data
            # that was previously serialized by this same cache system
            return pickle.loads(decompressed)

        except Exception as e:
            logger.error(f"Failed to decompress data: {e}")
            # Try pickle as fallback (safe for internal cache data)
            # Note: This pickle usage is safe as it only deserializes data
            # that was previously serialized by this same cache system
            try:
                return pickle.loads(compressed_data)
            except Exception as e2:
                logger.error(f"Failed to unpickle fallback data: {e2}")
                return None

    def _get_data_type_string(self, data: Any) -> str:
        """Generate data type string for metadata."""
        if isinstance(data, np.ndarray):
            return f"array_{data.shape}_{data.dtype}"
        if isinstance(data, dict):
            return "dict"
        if isinstance(data, (list, tuple)):
            return f"{type(data).__name__}_{len(data)}"
        return type(data).__name__

    def get(self, key: str) -> tuple[Any, bool]:
        """
        Get data from cache with multi-level lookup.

        Returns
        -------
        tuple
            (data, found) where found indicates if data was found in cache
        """
        start_time = time.time()

        with self._lock:
            # Try L1 first (hot memory cache)
            if key in self._l1_cache:
                entry = self._l1_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self._l1_cache.move_to_end(key)  # LRU update

                    time_ms = (time.time() - start_time) * 1000.0
                    self.stats.record_hit(CacheLevel.L1, time_ms)
                    return entry.data, True
                self._remove_entry(key, CacheLevel.L1)

            # Try L2 (compressed memory cache)
            if key in self._l2_cache:
                entry = self._l2_cache[key]
                if not entry.is_expired():
                    # Decompress data
                    data = self._decompress_data(entry.compressed_data, entry.data_type)
                    if data is not None:
                        entry.touch()
                        self._l2_cache.move_to_end(key)  # LRU update

                        # Promote to L1 if frequently accessed
                        if (
                            entry.access_frequency > 0.1
                        ):  # More than 0.1 accesses per second
                            self._promote_to_l1(key, data, entry)

                        time_ms = (time.time() - start_time) * 1000.0
                        self.stats.record_hit(CacheLevel.L2, time_ms)
                        return data, True
                else:
                    self._remove_entry(key, CacheLevel.L2)

            # Try L3 (disk cache)
            if key in self._l3_index:
                entry = self._l3_index[key]
                if (
                    not entry.is_expired()
                    and entry.disk_path
                    and entry.disk_path.exists()
                ):
                    try:
                        with open(entry.disk_path, "rb") as f:
                            compressed_data = f.read()

                        # Verify checksum if enabled
                        if self.enable_checksums and entry.checksum:
                            actual_checksum = hashlib.md5(
                                compressed_data, usedforsecurity=False
                            ).hexdigest()
                            if actual_checksum != entry.checksum:
                                logger.warning(
                                    f"Checksum mismatch for {key}, removing from L3 cache"
                                )
                                self._remove_entry(key, CacheLevel.L3)
                                self.stats.record_miss(CacheLevel.L3)
                                return None, False

                        # Decompress data
                        data = self._decompress_data(compressed_data, entry.data_type)
                        if data is not None:
                            entry.touch()
                            self._l3_index.move_to_end(key)  # LRU update

                            # Promote to L2 for faster future access
                            self._promote_to_l2(key, data, entry)

                            time_ms = (time.time() - start_time) * 1000.0
                            self.stats.record_hit(CacheLevel.L3, time_ms)
                            return data, True

                    except Exception as e:
                        logger.error(f"Failed to read L3 cache entry {key}: {e}")
                        self._remove_entry(key, CacheLevel.L3)
                else:
                    self._remove_entry(key, CacheLevel.L3)

            # Cache miss
            self.stats.record_miss(CacheLevel.L1)
            return None, False

    def put(self, key: str, data: Any, ttl_seconds: float | None = None) -> bool:
        """
        Put data into cache with intelligent tier placement.

        Parameters
        ----------
        key : str
            Cache key
        data : Any
            Data to cache
        ttl_seconds : float, optional
            Time to live in seconds

        Returns
        -------
        bool
            True if successfully cached
        """
        start_time = time.time()

        with self._lock:
            try:
                # Calculate data properties
                data_size = self._calculate_data_size(data)
                data_size_mb = data_size / (1024 * 1024)
                data_type = self._get_data_type_string(data)
                checksum = self._calculate_checksum(data)

                # Set TTL expiration
                ttl_expires_at = None
                if ttl_seconds is not None:
                    ttl_expires_at = time.time() + ttl_seconds

                # Remove existing entries for this key
                self._remove_all_entries(key)

                # Decide placement strategy based on size and current memory usage
                current_l1_mb = sum(
                    e.memory_size_bytes for e in self._l1_cache.values()
                ) / (1024 * 1024)

                # Try L1 first for small, frequently accessed data
                if (
                    data_size_mb < 100.0  # Less than 100MB
                    and current_l1_mb + data_size_mb < self.l1_max_memory_mb * 0.9
                ):  # Within 90% of limit
                    entry = CacheEntry(
                        key=key,
                        data=data,
                        memory_size_bytes=data_size,
                        cache_level=CacheLevel.L1,
                        data_type=data_type,
                        checksum=checksum,
                        ttl_expires_at=ttl_expires_at,
                    )

                    self._l1_cache[key] = entry

                    # Ensure L1 memory limit
                    if current_l1_mb + data_size_mb > self.l1_max_memory_mb:
                        self._enforce_l1_memory_limit()

                    time_ms = (time.time() - start_time) * 1000.0
                    self.stats.record_put(time_ms)
                    return True

                # Try L2 for medium-sized data
                if data_size_mb < 500.0:  # Less than 500MB
                    compressed_data, compression_time_ms = self._compress_data(data)
                    compressed_size_mb = len(compressed_data) / (1024 * 1024)

                    current_l2_mb = sum(
                        e.compressed_size_bytes for e in self._l2_cache.values()
                    ) / (1024 * 1024)

                    if current_l2_mb + compressed_size_mb < self.l2_max_memory_mb * 0.9:
                        compression_ratio = (
                            compressed_size_mb / data_size_mb
                            if data_size_mb > 0
                            else 1.0
                        )

                        entry = CacheEntry(
                            key=key,
                            compressed_data=compressed_data,
                            memory_size_bytes=data_size,
                            compressed_size_bytes=len(compressed_data),
                            cache_level=CacheLevel.L2,
                            data_type=data_type,
                            checksum=checksum,
                            compression_ratio=compression_ratio,
                            ttl_expires_at=ttl_expires_at,
                        )

                        self._l2_cache[key] = entry

                        # Record compression statistics
                        self.stats.record_compression(
                            data_size, len(compressed_data), compression_time_ms
                        )

                        # Ensure L2 memory limit
                        if current_l2_mb + compressed_size_mb > self.l2_max_memory_mb:
                            self._enforce_l2_memory_limit()

                        time_ms = (time.time() - start_time) * 1000.0
                        self.stats.record_put(time_ms)
                        return True

                # Use L3 for large data
                compressed_data, compression_time_ms = self._compress_data(data)
                disk_path = self._get_disk_path(key)

                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(disk_path, "wb") as f:
                        f.write(compressed_data)

                    compression_ratio = (
                        len(compressed_data) / data_size if data_size > 0 else 1.0
                    )

                    entry = CacheEntry(
                        key=key,
                        disk_path=disk_path,
                        memory_size_bytes=data_size,
                        disk_size_bytes=len(compressed_data),
                        cache_level=CacheLevel.L3,
                        data_type=data_type,
                        checksum=hashlib.md5(
                            compressed_data, usedforsecurity=False
                        ).hexdigest()
                        if self.enable_checksums
                        else None,
                        compression_ratio=compression_ratio,
                        ttl_expires_at=ttl_expires_at,
                    )

                    self._l3_index[key] = entry

                    # Record compression statistics
                    self.stats.record_compression(
                        data_size, len(compressed_data), compression_time_ms
                    )

                    # Ensure L3 disk limit
                    current_l3_mb = sum(
                        e.disk_size_bytes for e in self._l3_index.values()
                    ) / (1024 * 1024)
                    if current_l3_mb > self.l3_max_disk_mb:
                        self._enforce_l3_disk_limit()

                    time_ms = (time.time() - start_time) * 1000.0
                    self.stats.record_put(time_ms)
                    return True

                except Exception as e:
                    logger.error(f"Failed to write L3 cache entry {key}: {e}")
                    if disk_path.exists():
                        disk_path.unlink()
                    return False

            except Exception as e:
                logger.error(f"Failed to cache data for key {key}: {e}")
                return False

    def _get_disk_path(self, key: str) -> Path:
        """Generate disk path for cache entry."""
        # Use first 2 chars of key for subdirectory to avoid too many files in one dir
        subdir = key[:2]
        return self.l3_cache_dir / subdir / f"{key}.cache"

    def _remove_entry(self, key: str, level: CacheLevel):
        """Remove entry from specific cache level."""
        if level == CacheLevel.L1 and key in self._l1_cache:
            del self._l1_cache[key]
        elif level == CacheLevel.L2 and key in self._l2_cache:
            del self._l2_cache[key]
        elif level == CacheLevel.L3 and key in self._l3_index:
            entry = self._l3_index[key]
            if entry.disk_path and entry.disk_path.exists():
                try:
                    entry.disk_path.unlink()
                except Exception as e:
                    logger.debug(f"Failed to remove disk file {entry.disk_path}: {e}")
            del self._l3_index[key]

    def _remove_all_entries(self, key: str):
        """Remove entry from all cache levels."""
        self._remove_entry(key, CacheLevel.L1)
        self._remove_entry(key, CacheLevel.L2)
        self._remove_entry(key, CacheLevel.L3)

    def _promote_to_l1(self, key: str, data: Any, source_entry: CacheEntry):
        """Promote data from L2/L3 to L1."""
        data_size_mb = source_entry.memory_size_bytes / (1024 * 1024)
        current_l1_mb = sum(e.memory_size_bytes for e in self._l1_cache.values()) / (
            1024 * 1024
        )

        # Check if there's space in L1
        if current_l1_mb + data_size_mb < self.l1_max_memory_mb * 0.9:
            entry = CacheEntry(
                key=key,
                data=data,
                memory_size_bytes=source_entry.memory_size_bytes,
                cache_level=CacheLevel.L1,
                data_type=source_entry.data_type,
                checksum=source_entry.checksum,
                ttl_expires_at=source_entry.ttl_expires_at,
            )

            # Preserve access statistics
            entry.access_count = source_entry.access_count
            entry.access_frequency = source_entry.access_frequency
            entry.created_at = source_entry.created_at
            entry.last_accessed = source_entry.last_accessed

            self._l1_cache[key] = entry

            # Ensure memory limits
            if current_l1_mb + data_size_mb > self.l1_max_memory_mb:
                self._enforce_l1_memory_limit()

    def _promote_to_l2(self, key: str, data: Any, source_entry: CacheEntry):
        """Promote data from L3 to L2."""
        try:
            compressed_data, compression_time_ms = self._compress_data(data)
            compressed_size_mb = len(compressed_data) / (1024 * 1024)
            current_l2_mb = sum(
                e.compressed_size_bytes for e in self._l2_cache.values()
            ) / (1024 * 1024)

            # Check if there's space in L2
            if current_l2_mb + compressed_size_mb < self.l2_max_memory_mb * 0.9:
                compression_ratio = compressed_size_mb / (
                    source_entry.memory_size_bytes / (1024 * 1024)
                )

                entry = CacheEntry(
                    key=key,
                    compressed_data=compressed_data,
                    memory_size_bytes=source_entry.memory_size_bytes,
                    compressed_size_bytes=len(compressed_data),
                    cache_level=CacheLevel.L2,
                    data_type=source_entry.data_type,
                    checksum=source_entry.checksum,
                    compression_ratio=compression_ratio,
                    ttl_expires_at=source_entry.ttl_expires_at,
                )

                # Preserve access statistics
                entry.access_count = source_entry.access_count
                entry.access_frequency = source_entry.access_frequency
                entry.created_at = source_entry.created_at
                entry.last_accessed = source_entry.last_accessed

                self._l2_cache[key] = entry

                # Record compression statistics
                self.stats.record_compression(
                    source_entry.memory_size_bytes,
                    len(compressed_data),
                    compression_time_ms,
                )

                # Ensure memory limits
                if current_l2_mb + compressed_size_mb > self.l2_max_memory_mb:
                    self._enforce_l2_memory_limit()

        except Exception as e:
            logger.error(f"Failed to promote {key} to L2: {e}")

    def _enforce_l1_memory_limit(self):
        """Enforce L1 memory limit by evicting entries."""
        current_mb = sum(e.memory_size_bytes for e in self._l1_cache.values()) / (
            1024 * 1024
        )
        target_mb = self.l1_max_memory_mb * 0.8  # Target 80% of limit

        if current_mb <= target_mb:
            return

        # Sort entries by priority score (lowest first)
        entries = [
            (k, v, v.calculate_priority_score(self.eviction_policy))
            for k, v in self._l1_cache.items()
        ]
        entries.sort(key=lambda x: x[2])

        # Evict entries until under limit
        for key, entry, _score in entries:
            if current_mb <= target_mb:
                break

            # Try to promote to L2 first
            data = entry.data
            if data is not None:
                try:
                    self._promote_to_l2(key, data, entry)
                except Exception as e:
                    logger.debug(f"Failed to promote {key} to L2: {e}")

            # Remove from L1
            current_mb -= entry.memory_size_bytes / (1024 * 1024)
            del self._l1_cache[key]
            self.stats.record_eviction()

    def _enforce_l2_memory_limit(self):
        """Enforce L2 memory limit by evicting entries."""
        current_mb = sum(e.compressed_size_bytes for e in self._l2_cache.values()) / (
            1024 * 1024
        )
        target_mb = self.l2_max_memory_mb * 0.8  # Target 80% of limit

        if current_mb <= target_mb:
            return

        # Sort entries by priority score (lowest first)
        entries = [
            (k, v, v.calculate_priority_score(self.eviction_policy))
            for k, v in self._l2_cache.items()
        ]
        entries.sort(key=lambda x: x[2])

        # Evict entries until under limit
        for key, entry, _score in entries:
            if current_mb <= target_mb:
                break

            # Try to promote to L3 first
            if entry.compressed_data is not None:
                try:
                    self._promote_to_l3(key, entry)
                except Exception as e:
                    logger.debug(f"Failed to promote {key} to L3: {e}")

            # Remove from L2
            current_mb -= entry.compressed_size_bytes / (1024 * 1024)
            del self._l2_cache[key]
            self.stats.record_eviction()

    def _enforce_l3_disk_limit(self):
        """Enforce L3 disk limit by evicting entries."""
        current_mb = sum(e.disk_size_bytes for e in self._l3_index.values()) / (
            1024 * 1024
        )
        target_mb = self.l3_max_disk_mb * 0.8  # Target 80% of limit

        if current_mb <= target_mb:
            return

        # Sort entries by priority score (lowest first)
        entries = [
            (k, v, v.calculate_priority_score(self.eviction_policy))
            for k, v in self._l3_index.items()
        ]
        entries.sort(key=lambda x: x[2])

        # Evict entries until under limit
        for key, entry, _score in entries:
            if current_mb <= target_mb:
                break

            # Remove from L3
            current_mb -= entry.disk_size_bytes / (1024 * 1024)
            self._remove_entry(key, CacheLevel.L3)
            self.stats.record_eviction()

    def _promote_to_l3(self, key: str, source_entry: CacheEntry):
        """Promote data from L2 to L3."""
        try:
            disk_path = self._get_disk_path(key)
            disk_path.parent.mkdir(parents=True, exist_ok=True)

            with open(disk_path, "wb") as f:
                f.write(source_entry.compressed_data)

            entry = CacheEntry(
                key=key,
                disk_path=disk_path,
                memory_size_bytes=source_entry.memory_size_bytes,
                disk_size_bytes=source_entry.compressed_size_bytes,
                cache_level=CacheLevel.L3,
                data_type=source_entry.data_type,
                checksum=hashlib.md5(
                    source_entry.compressed_data, usedforsecurity=False
                ).hexdigest()
                if self.enable_checksums
                else None,
                compression_ratio=source_entry.compression_ratio,
                ttl_expires_at=source_entry.ttl_expires_at,
            )

            # Preserve access statistics
            entry.access_count = source_entry.access_count
            entry.access_frequency = source_entry.access_frequency
            entry.created_at = source_entry.created_at
            entry.last_accessed = source_entry.last_accessed

            self._l3_index[key] = entry

        except Exception as e:
            logger.error(f"Failed to promote {key} to L3: {e}")

    def _promote_l1_to_l2(self, force: bool = False, target_freed_mb: float = 0.0):
        """Promote entries from L1 to L2."""
        if not force and len(self._l1_cache) < 10:  # Don't bother if few entries
            return

        freed_mb = 0.0
        entries_to_promote = []

        # Select entries for promotion (least frequently accessed)
        for key, entry in list(self._l1_cache.items()):
            if not force and entry.access_frequency > 0.5:  # Keep hot entries in L1
                continue

            entries_to_promote.append((key, entry))
            if not force and len(entries_to_promote) >= 5:  # Limit batch size
                break
            if target_freed_mb > 0 and freed_mb >= target_freed_mb:
                break

        # Promote selected entries
        for key, entry in entries_to_promote:
            data = entry.data
            if data is not None:
                try:
                    self._promote_to_l2(key, data, entry)
                    freed_mb += entry.memory_size_bytes / (1024 * 1024)
                    del self._l1_cache[key]
                except Exception as e:
                    logger.debug(f"Failed to promote {key} from L1 to L2: {e}")

    def _promote_l2_to_l3(self, force: bool = False, target_freed_mb: float = 0.0):
        """Promote entries from L2 to L3."""
        if not force and len(self._l2_cache) < 5:  # Don't bother if few entries
            return

        freed_mb = 0.0
        entries_to_promote = []

        # Select entries for promotion (least frequently accessed)
        for key, entry in list(self._l2_cache.items()):
            if (
                not force and entry.access_frequency > 0.1
            ):  # Keep moderately hot entries in L2
                continue

            entries_to_promote.append((key, entry))
            if not force and len(entries_to_promote) >= 3:  # Limit batch size
                break
            if target_freed_mb > 0 and freed_mb >= target_freed_mb:
                break

        # Promote selected entries
        for key, entry in entries_to_promote:
            try:
                self._promote_to_l3(key, entry)
                freed_mb += entry.compressed_size_bytes / (1024 * 1024)
                del self._l2_cache[key]
            except Exception as e:
                logger.debug(f"Failed to promote {key} from L2 to L3: {e}")

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""
        with self._lock:
            found = False
            if key in self._l1_cache:
                del self._l1_cache[key]
                found = True
            if key in self._l2_cache:
                del self._l2_cache[key]
                found = True
            if key in self._l3_index:
                self._remove_entry(key, CacheLevel.L3)
                found = True
            return found

    def clear(self, level: CacheLevel | None = None):
        """Clear cache entries."""
        with self._lock:
            if level is None or level == CacheLevel.L1:
                self._l1_cache.clear()
            if level is None or level == CacheLevel.L2:
                self._l2_cache.clear()
            if level is None or level == CacheLevel.L3:
                # Remove all disk files
                for entry in self._l3_index.values():
                    if entry.disk_path and entry.disk_path.exists():
                        try:
                            entry.disk_path.unlink()
                        except Exception as e:
                            logger.debug(f"Failed to remove {entry.disk_path}: {e}")
                self._l3_index.clear()

        logger.info(f"Cleared cache level: {level or 'ALL'}")

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            base_stats = self.stats.get_summary()

            # Add current cache counts
            base_stats["current_counts"] = {
                "l1_entries": len(self._l1_cache),
                "l2_entries": len(self._l2_cache),
                "l3_entries": len(self._l3_index),
            }

            # Add memory pressure info
            used_mb, available_mb, percent_used = SystemMemoryMonitor.get_memory_info()
            base_stats["system_memory"] = {
                "used_mb": used_mb,
                "available_mb": available_mb,
                "percent_used": percent_used,
                "pressure_threshold": self._memory_pressure_threshold * 100,
            }

            return base_stats

    def shutdown(self):
        """Shutdown cache system gracefully."""
        logger.info("Shutting down MultiLevelCache")

        # Stop cleanup thread
        self._cleanup_stop_event.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        # Final statistics
        stats = self.get_stats()
        logger.info(f"Cache shutdown stats: {stats['hit_rates']}")

        # Clear all caches
        self.clear()


# Global cache instance with reasonable defaults
_global_multi_cache: MultiLevelCache | None = None


def get_global_cache(
    l1_max_memory_mb: float = 500.0,
    l2_max_memory_mb: float = 1000.0,
    l3_max_disk_mb: float = 5000.0,
    reset: bool = False,
) -> MultiLevelCache:
    """
    Get or create global multi-level cache instance.

    Parameters
    ----------
    l1_max_memory_mb : float
        Maximum memory for L1 cache in MB
    l2_max_memory_mb : float
        Maximum memory for L2 cache in MB
    l3_max_disk_mb : float
        Maximum disk space for L3 cache in MB
    reset : bool
        Whether to reset existing cache

    Returns
    -------
    MultiLevelCache
        Global cache instance
    """
    global _global_multi_cache

    if _global_multi_cache is None or reset:
        if _global_multi_cache is not None:
            _global_multi_cache.shutdown()

        _global_multi_cache = MultiLevelCache(
            l1_max_memory_mb=l1_max_memory_mb,
            l2_max_memory_mb=l2_max_memory_mb,
            l3_max_disk_mb=l3_max_disk_mb,
        )

    return _global_multi_cache


def cache_computation(cache_key_fn=None, ttl_seconds=None):
    """
    Decorator for caching computation results.

    Parameters
    ----------
    cache_key_fn : callable, optional
        Function to generate cache key from arguments
    ttl_seconds : float, optional
        Time to live for cached results

    Returns
    -------
    callable
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_global_cache()

            # Generate cache key
            if cache_key_fn:
                key = cache_key_fn(*args, **kwargs)
            else:
                key = cache._generate_cache_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result, found = cache.get(key)
            if found:
                return result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache.put(key, result, ttl_seconds=ttl_seconds)

            return result

        return wrapper

    return decorator
