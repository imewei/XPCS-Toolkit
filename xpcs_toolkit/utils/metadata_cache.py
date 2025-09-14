"""
Specialized caching for metadata and HDF5 file information in XPCS Toolkit.

This module provides optimized caching for file metadata, HDF5 structure information,
Q-map calculations, and other frequently accessed file-based data.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from .advanced_cache import get_global_cache
from .logging_config import get_logger
from .memory_utils import SystemMemoryMonitor

logger = get_logger(__name__)


@dataclass
class FileMetadata:
    """Cached file metadata with integrity checking."""

    file_path: str
    file_size: int
    file_mtime: float
    analysis_type: str
    metadata_dict: Dict[str, Any]
    large_datasets: List[Dict[str, Any]]
    cached_at: float = field(default_factory=time.time)
    access_count: int = 0

    def is_valid(self) -> bool:
        """Check if cached metadata is still valid."""
        try:
            if not os.path.exists(self.file_path):
                return False

            current_mtime = os.path.getmtime(self.file_path)
            current_size = os.path.getsize(self.file_path)

            return current_mtime == self.file_mtime and current_size == self.file_size
        except (OSError, FileNotFoundError):
            return False

    def touch(self):
        """Update access statistics."""
        self.access_count += 1


@dataclass
class QMapData:
    """Cached Q-map data with detector geometry."""

    file_path: str
    qmap_params: Dict[str, Any]
    sqmap: np.ndarray
    dqmap: np.ndarray
    sqlist: np.ndarray
    dqlist: np.ndarray
    mask: np.ndarray
    geometry_params: Dict[str, float]
    cached_at: float = field(default_factory=time.time)
    checksum: str = ""

    def calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification."""
        data_parts = [
            self.sqmap.tobytes(),
            self.dqmap.tobytes(),
            self.sqlist.tobytes(),
            self.dqlist.tobytes(),
            self.mask.tobytes(),
        ]
        combined_data = b"".join(data_parts)
        return hashlib.md5(combined_data, usedforsecurity=False).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify data integrity using checksum."""
        if not self.checksum:
            return True  # No checksum to verify
        return self.calculate_checksum() == self.checksum


@dataclass
class PrefetchRequest:
    """Request for prefetching data."""

    file_path: str
    data_types: List[str]
    priority: int = 1  # Higher = more important
    requested_at: float = field(default_factory=time.time)


class MetadataCache:
    """
    Specialized cache for file metadata, HDF5 structure, and Q-map data.

    Features:
    - File modification time tracking for invalidation
    - Compressed storage for large metadata
    - Smart prefetching based on access patterns
    - Background prefetch queue processing
    """

    def __init__(
        self,
        max_metadata_entries: int = 1000,
        max_qmap_entries: int = 100,
        enable_prefetch: bool = True,
        prefetch_queue_size: int = 50,
    ):
        self._cache = get_global_cache()
        self._max_metadata_entries = max_metadata_entries
        self._max_qmap_entries = max_qmap_entries
        self._enable_prefetch = enable_prefetch

        # Thread safety
        self._lock = threading.RLock()

        # Tracking structures
        self._metadata_keys: OrderedDict[str, str] = (
            OrderedDict()
        )  # file_path -> cache_key
        self._qmap_keys: OrderedDict[str, str] = OrderedDict()  # file_path -> cache_key

        # Access pattern tracking
        self._access_patterns: Dict[str, List[float]] = {}  # file_path -> access_times
        self._file_relationships: Dict[str, Set[str]] = {}  # file_path -> related_files

        # Prefetch queue
        self._prefetch_queue: List[PrefetchRequest] = []
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_stop_event = threading.Event()

        if self._enable_prefetch:
            self._start_prefetch_worker()

        logger.info(
            f"MetadataCache initialized with prefetch={'enabled' if enable_prefetch else 'disabled'}"
        )

    def _start_prefetch_worker(self):
        """Start background prefetch worker thread."""
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for prefetching data."""
        while not self._prefetch_stop_event.wait(1.0):  # Check every second
            try:
                self._process_prefetch_queue()
            except Exception as e:
                logger.error(f"Error in metadata prefetch worker: {e}")

    def _process_prefetch_queue(self):
        """Process prefetch requests from queue."""
        if not self._prefetch_queue:
            return

        with self._lock:
            # Sort by priority and age
            self._prefetch_queue.sort(key=lambda req: (-req.priority, req.requested_at))

            # Process a few requests
            processed_count = 0
            while self._prefetch_queue and processed_count < 3:  # Limit batch size
                request = self._prefetch_queue.pop(0)

                try:
                    self._execute_prefetch(request)
                    processed_count += 1
                except Exception as e:
                    logger.debug(f"Failed to prefetch {request.file_path}: {e}")

    def _execute_prefetch(self, request: PrefetchRequest):
        """Execute a prefetch request."""
        # Check if we should skip this prefetch (file already cached, memory pressure, etc.)
        if SystemMemoryMonitor.check_memory_pressure(85.0):
            logger.debug(
                f"Skipping prefetch due to memory pressure: {request.file_path}"
            )
            return

        # Check if metadata is already cached
        if "metadata" in request.data_types:
            metadata_key = self._generate_metadata_key(request.file_path)
            cached_metadata, found = self._cache.get(metadata_key)
            if not found:
                logger.debug(f"Prefetching metadata for {request.file_path}")
                # This would trigger actual metadata loading - simplified here
                # In practice, this would call the actual metadata reading functions

    def _generate_metadata_key(self, file_path: str) -> str:
        """Generate cache key for file metadata."""
        return f"metadata:{hashlib.md5(file_path.encode(), usedforsecurity=False).hexdigest()}"

    def _generate_qmap_key(self, file_path: str, params: Dict[str, Any]) -> str:
        """Generate cache key for Q-map data."""
        # Include file path and relevant parameters
        key_data = f"{file_path}:{str(sorted(params.items()))}"
        return f"qmap:{hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()}"

    def _record_access(self, file_path: str):
        """Record file access for pattern analysis."""
        current_time = time.time()

        with self._lock:
            if file_path not in self._access_patterns:
                self._access_patterns[file_path] = []

            self._access_patterns[file_path].append(current_time)

            # Keep only recent access times (last 24 hours)
            cutoff_time = current_time - (24 * 3600)
            self._access_patterns[file_path] = [
                t for t in self._access_patterns[file_path] if t > cutoff_time
            ]

    def _predict_related_files(self, file_path: str) -> List[str]:
        """Predict related files based on naming patterns and directory structure."""
        related_files = []

        try:
            file_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)

            # Look for files with similar patterns
            if os.path.exists(file_dir):
                for filename in os.listdir(file_dir):
                    if filename.endswith(".hdf") or filename.endswith(".h5"):
                        # Simple pattern matching - could be more sophisticated
                        base_pattern = (
                            file_name.split("_")[0]
                            if "_" in file_name
                            else file_name.split(".")[0]
                        )
                        if base_pattern in filename and filename != file_name:
                            related_files.append(os.path.join(file_dir, filename))

                            # Limit to avoid excessive prefetching
                            if len(related_files) >= 5:
                                break

        except Exception as e:
            logger.debug(f"Error predicting related files for {file_path}: {e}")

        return related_files

    def get_file_metadata(
        self, file_path: str, force_refresh: bool = False
    ) -> Optional[FileMetadata]:
        """
        Get cached file metadata with automatic invalidation.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        force_refresh : bool
            Force refresh from disk

        Returns
        -------
        Optional[FileMetadata]
            Cached metadata or None if not available
        """
        self._record_access(file_path)

        cache_key = self._generate_metadata_key(file_path)

        if not force_refresh:
            cached_metadata, found = self._cache.get(cache_key)
            if found and isinstance(cached_metadata, FileMetadata):
                if cached_metadata.is_valid():
                    cached_metadata.touch()
                    logger.debug(f"Metadata cache hit: {file_path}")

                    # Schedule prefetch for related files
                    self._schedule_related_prefetch(file_path)

                    return cached_metadata
                else:
                    # Invalid metadata, remove from cache
                    self._cache.invalidate(cache_key)

        logger.debug(f"Metadata cache miss: {file_path}")
        return None

    def put_file_metadata(
        self, file_path: str, metadata: FileMetadata, ttl_seconds: float = 7200.0
    ):  # 2 hours default TTL
        """
        Cache file metadata.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        metadata : FileMetadata
            Metadata to cache
        ttl_seconds : float
            Time to live for cached data
        """
        cache_key = self._generate_metadata_key(file_path)

        # Cache with appropriate level (metadata is small, use L1 or L2)
        success = self._cache.put(cache_key, metadata, ttl_seconds=ttl_seconds)

        if success:
            with self._lock:
                self._metadata_keys[file_path] = cache_key

                # Enforce entry limits
                while len(self._metadata_keys) > self._max_metadata_entries:
                    old_path, old_key = self._metadata_keys.popitem(last=False)
                    self._cache.invalidate(old_key)

            logger.debug(f"Cached metadata for {file_path}")
        else:
            logger.warning(f"Failed to cache metadata for {file_path}")

    def get_qmap_data(
        self, file_path: str, params: Dict[str, Any], force_refresh: bool = False
    ) -> Optional[QMapData]:
        """
        Get cached Q-map data.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        params : dict
            Q-map calculation parameters
        force_refresh : bool
            Force refresh from calculation

        Returns
        -------
        Optional[QMapData]
            Cached Q-map data or None if not available
        """
        self._record_access(file_path)

        cache_key = self._generate_qmap_key(file_path, params)

        if not force_refresh:
            cached_qmap, found = self._cache.get(cache_key)
            if found and isinstance(cached_qmap, QMapData):
                if cached_qmap.verify_integrity():
                    logger.debug(f"Q-map cache hit: {file_path}")
                    return cached_qmap
                else:
                    # Data corruption detected, remove from cache
                    logger.warning(
                        f"Q-map data corruption detected, removing from cache: {file_path}"
                    )
                    self._cache.invalidate(cache_key)

        logger.debug(f"Q-map cache miss: {file_path}")
        return None

    def put_qmap_data(
        self,
        file_path: str,
        params: Dict[str, Any],
        qmap_data: QMapData,
        ttl_seconds: float = 14400.0,
    ):  # 4 hours default TTL
        """
        Cache Q-map data.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        params : dict
            Q-map calculation parameters
        qmap_data : QMapData
            Q-map data to cache
        ttl_seconds : float
            Time to live for cached data
        """
        cache_key = self._generate_qmap_key(file_path, params)

        # Calculate and store checksum for integrity verification
        qmap_data.checksum = qmap_data.calculate_checksum()

        # Q-map data can be large, so it might go to L2 or L3
        success = self._cache.put(cache_key, qmap_data, ttl_seconds=ttl_seconds)

        if success:
            with self._lock:
                self._qmap_keys[file_path] = cache_key

                # Enforce entry limits
                while len(self._qmap_keys) > self._max_qmap_entries:
                    old_path, old_key = self._qmap_keys.popitem(last=False)
                    self._cache.invalidate(old_key)

            # Estimate data size for logging
            data_size_mb = sum(
                arr.nbytes
                for arr in [
                    qmap_data.sqmap,
                    qmap_data.dqmap,
                    qmap_data.sqlist,
                    qmap_data.dqlist,
                    qmap_data.mask,
                ]
            ) / (1024 * 1024)
            logger.info(f"Cached Q-map data for {file_path} ({data_size_mb:.1f}MB)")
        else:
            logger.warning(f"Failed to cache Q-map data for {file_path}")

    def _schedule_related_prefetch(self, file_path: str):
        """Schedule prefetching for files related to the accessed file."""
        if not self._enable_prefetch:
            return

        # Predict related files
        related_files = self._predict_related_files(file_path)

        if related_files:
            with self._lock:
                # Check if we should prefetch (don't spam the queue)
                current_time = time.time()
                for related_file in related_files:
                    # Only prefetch if not already queued recently
                    already_queued = any(
                        req.file_path == related_file
                        and (current_time - req.requested_at) < 300  # 5 minutes
                        for req in self._prefetch_queue
                    )

                    if not already_queued:
                        request = PrefetchRequest(
                            file_path=related_file, data_types=["metadata"], priority=1
                        )
                        self._prefetch_queue.append(request)

                        # Limit queue size
                        if len(self._prefetch_queue) > 50:
                            break

            logger.debug(f"Scheduled prefetch for {len(related_files)} related files")

    def invalidate_file(self, file_path: str):
        """Invalidate all cached data for a specific file."""
        invalidated_count = 0

        with self._lock:
            # Invalidate metadata
            if file_path in self._metadata_keys:
                cache_key = self._metadata_keys[file_path]
                if self._cache.invalidate(cache_key):
                    invalidated_count += 1
                del self._metadata_keys[file_path]

            # Invalidate Q-map data
            if file_path in self._qmap_keys:
                cache_key = self._qmap_keys[file_path]
                if self._cache.invalidate(cache_key):
                    invalidated_count += 1
                del self._qmap_keys[file_path]

            # Clear access patterns
            if file_path in self._access_patterns:
                del self._access_patterns[file_path]

        if invalidated_count > 0:
            logger.info(
                f"Invalidated {invalidated_count} cache entries for {file_path}"
            )

        return invalidated_count

    def warm_cache(self, file_paths: List[str], data_types: List[str] = None):
        """
        Warm cache by prefetching data for multiple files.

        Parameters
        ----------
        file_paths : list
            List of file paths to prefetch
        data_types : list, optional
            Types of data to prefetch ('metadata', 'qmap')
        """
        if not self._enable_prefetch:
            logger.warning("Prefetch is disabled, cannot warm cache")
            return

        if data_types is None:
            data_types = ["metadata"]

        with self._lock:
            for file_path in file_paths:
                request = PrefetchRequest(
                    file_path=file_path,
                    data_types=data_types,
                    priority=2,  # Higher priority for explicit warming
                )
                self._prefetch_queue.append(request)

        logger.info(f"Scheduled cache warming for {len(file_paths)} files")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            base_stats = self._cache.get_stats()

            # Access pattern analysis
            total_accesses = sum(len(times) for times in self._access_patterns.values())
            active_files = len(
                [fp for fp, times in self._access_patterns.items() if times]
            )

            # Calculate average access frequency
            current_time = time.time()
            recent_cutoff = current_time - 3600  # Last hour
            recent_accesses = sum(
                len([t for t in times if t > recent_cutoff])
                for times in self._access_patterns.values()
            )

            metadata_stats = {
                "cached_metadata_files": len(self._metadata_keys),
                "cached_qmap_files": len(self._qmap_keys),
                "prefetch_queue_size": len(self._prefetch_queue),
                "total_file_accesses": total_accesses,
                "active_files": active_files,
                "recent_accesses_per_hour": recent_accesses,
                "access_patterns": {
                    fp: len(times)
                    for fp, times in list(self._access_patterns.items())[:10]  # Top 10
                },
            }

            base_stats["metadata_cache"] = metadata_stats
            return base_stats

    def cleanup_expired_metadata(self, max_age_hours: float = 48.0):
        """Clean up expired metadata entries."""
        current_time = time.time()
        cutoff_time = current_time - (max_age_hours * 3600)

        removed_count = 0

        with self._lock:
            # Check metadata entries
            files_to_remove = []
            for file_path, cache_key in list(self._metadata_keys.items()):
                cached_metadata, found = self._cache.get(cache_key)
                if found and isinstance(cached_metadata, FileMetadata):
                    if cached_metadata.cached_at < cutoff_time:
                        if self._cache.invalidate(cache_key):
                            removed_count += 1
                        files_to_remove.append(file_path)
                elif not found:
                    # Cache key not found, remove from tracking
                    files_to_remove.append(file_path)

            # Remove from tracking
            for file_path in files_to_remove:
                del self._metadata_keys[file_path]

            # Similar cleanup for Q-map entries
            files_to_remove = []
            for file_path, cache_key in list(self._qmap_keys.items()):
                cached_qmap, found = self._cache.get(cache_key)
                if found and isinstance(cached_qmap, QMapData):
                    if cached_qmap.cached_at < cutoff_time:
                        if self._cache.invalidate(cache_key):
                            removed_count += 1
                        files_to_remove.append(file_path)
                elif not found:
                    files_to_remove.append(file_path)

            for file_path in files_to_remove:
                del self._qmap_keys[file_path]

        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired metadata entries")

        return removed_count

    def shutdown(self):
        """Shutdown metadata cache gracefully."""
        logger.info("Shutting down MetadataCache")

        # Stop prefetch thread
        if self._prefetch_thread:
            self._prefetch_stop_event.set()
            self._prefetch_thread.join(timeout=3.0)

        # Clear tracking structures
        with self._lock:
            self._metadata_keys.clear()
            self._qmap_keys.clear()
            self._prefetch_queue.clear()


# Global metadata cache instance
_global_metadata_cache: Optional[MetadataCache] = None


def get_metadata_cache(enable_prefetch: bool = True) -> MetadataCache:
    """Get global metadata cache instance."""
    global _global_metadata_cache

    if _global_metadata_cache is None:
        _global_metadata_cache = MetadataCache(enable_prefetch=enable_prefetch)

    return _global_metadata_cache


def cache_file_metadata(file_path: str, ttl_seconds: float = 7200.0):
    """Decorator for caching file metadata operations."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = get_metadata_cache()

            # Check for cached metadata first
            cached_metadata = cache.get_file_metadata(file_path)
            if cached_metadata:
                return cached_metadata.metadata_dict

            # Execute function to get metadata
            result = func(*args, **kwargs)

            # Create metadata object and cache it
            try:
                stat = os.stat(file_path)
                metadata = FileMetadata(
                    file_path=file_path,
                    file_size=stat.st_size,
                    file_mtime=stat.st_mtime,
                    analysis_type=result.get("analysis_type", "unknown"),
                    metadata_dict=result,
                    large_datasets=result.get("large_datasets", []),
                )

                cache.put_file_metadata(file_path, metadata, ttl_seconds)

            except Exception as e:
                logger.debug(f"Failed to cache metadata for {file_path}: {e}")

            return result

        return wrapper

    return decorator
