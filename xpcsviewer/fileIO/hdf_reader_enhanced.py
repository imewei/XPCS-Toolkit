"""
Enhanced HDF5 Reader with Intelligent Chunking and Read-ahead Caching

This module provides optimized HDF5 I/O operations with intelligent caching,
chunking strategies, and read-ahead mechanisms for XPCS data analysis.
"""

import threading
import time
import weakref
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import h5py
import numpy as np

from ..utils.logging_config import get_logger
from ..utils.memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


class AccessPattern(Enum):
    """HDF5 data access patterns for optimization."""

    SEQUENTIAL = "sequential"
    RANDOM = "random"
    BLOCK = "block"
    SPARSE = "sparse"


@dataclass
class ReadRequest:
    """Request for HDF5 data reading."""

    file_path: str
    dataset_path: str
    slice_info: tuple[slice, ...] | None
    priority: float = 0.5
    requested_time: float = 0.0
    access_pattern: AccessPattern = AccessPattern.RANDOM


@dataclass
class CacheEntry:
    """Enhanced cache entry for HDF5 data."""

    data: np.ndarray
    file_path: str
    dataset_path: str
    slice_info: tuple[slice, ...] | None
    access_count: int = 0
    last_accessed: float = 0.0
    created_time: float = 0.0
    size_mb: float = 0.0
    access_pattern: AccessPattern = AccessPattern.RANDOM


class IntelligentChunker:
    """
    Intelligent chunking strategy for HDF5 datasets based on access patterns.
    """

    def __init__(self):
        self.access_history = defaultdict(list)
        self.chunk_cache = {}
        self.memory_manager = get_memory_manager()

    def analyze_access_pattern(
        self,
        file_path: str,
        dataset_path: str,
        recent_accesses: list[tuple[slice, ...]],
    ) -> AccessPattern:
        """
        Analyze access pattern from recent slice requests.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_path : str
            Path to dataset within file
        recent_accesses : list[tuple[slice, ...]]
            Recent access slice patterns

        Returns
        -------
        AccessPattern
            Detected access pattern
        """
        if len(recent_accesses) < 3:
            return AccessPattern.RANDOM

        # Analyze for sequential pattern
        if self._is_sequential_pattern(recent_accesses):
            return AccessPattern.SEQUENTIAL

        # Analyze for block pattern
        if self._is_block_pattern(recent_accesses):
            return AccessPattern.BLOCK

        # Check for sparse pattern
        if self._is_sparse_pattern(recent_accesses):
            return AccessPattern.SPARSE

        return AccessPattern.RANDOM

    def _is_sequential_pattern(self, accesses: list[tuple[slice, ...]]) -> bool:
        """Check if accesses follow sequential pattern."""
        if len(accesses) < 3:
            return False

        # Extract start positions for first dimension
        starts = []
        for access in accesses[-5:]:  # Check last 5 accesses
            if len(access) > 0 and isinstance(access[0], slice):
                starts.append(access[0].start or 0)

        if len(starts) < 3:
            return False

        # Check if positions are increasing with regular intervals
        diffs = np.diff(starts)
        if len(diffs) > 1:
            # Regular intervals suggest sequential access
            return np.std(diffs) < np.mean(diffs) * 0.3

        return False

    def _is_block_pattern(self, accesses: list[tuple[slice, ...]]) -> bool:
        """Check if accesses follow block pattern."""
        # Block pattern: accessing contiguous blocks of data
        block_sizes = []
        for access in accesses[-5:]:
            if len(access) > 0 and isinstance(access[0], slice):
                size = (access[0].stop or 1) - (access[0].start or 0)
                block_sizes.append(size)

        if len(block_sizes) < 3:
            return False

        # Consistent block sizes suggest block pattern
        return np.std(block_sizes) < np.mean(block_sizes) * 0.2

    def _is_sparse_pattern(self, accesses: list[tuple[slice, ...]]) -> bool:
        """Check if accesses follow sparse pattern."""
        # Sparse pattern: non-contiguous, irregular access
        if len(accesses) < 3:
            return False

        # Check for large gaps between accesses
        starts = []
        for access in accesses[-5:]:
            if len(access) > 0 and isinstance(access[0], slice):
                starts.append(access[0].start or 0)

        if len(starts) < 3:
            return False

        diffs = np.diff(sorted(starts))
        if len(diffs) > 1:
            # Large, irregular gaps suggest sparse access
            return np.std(diffs) > np.mean(diffs) * 2.0

        return False

    def get_optimal_chunk_shape(
        self,
        dataset_shape: tuple[int, ...],
        dtype: np.dtype,
        access_pattern: AccessPattern,
        target_chunk_mb: float = 10.0,
    ) -> tuple[int, ...]:
        """
        Calculate optimal chunk shape for dataset based on access pattern.

        Parameters
        ----------
        dataset_shape : tuple[int, ...]
            Shape of the dataset
        dtype : np.dtype
            Data type of the dataset
        access_pattern : AccessPattern
            Detected access pattern
        target_chunk_mb : float
            Target chunk size in MB

        Returns
        -------
        tuple[int, ...]
            Optimal chunk shape
        """
        itemsize = dtype.itemsize
        target_elements = int(target_chunk_mb * 1024 * 1024 / itemsize)

        if access_pattern == AccessPattern.SEQUENTIAL:
            # For sequential access, optimize for reading along first dimension
            chunk_shape = list(dataset_shape)
            if len(chunk_shape) > 1:
                # Keep other dimensions, adjust first dimension
                elements_per_row = np.prod(chunk_shape[1:])
                rows_per_chunk = max(1, target_elements // elements_per_row)
                chunk_shape[0] = min(rows_per_chunk, dataset_shape[0])
            else:
                chunk_shape[0] = min(target_elements, dataset_shape[0])

        elif access_pattern == AccessPattern.BLOCK:
            # For block access, create balanced chunks
            chunk_shape = []
            remaining_elements = target_elements
            for dim_size in dataset_shape:
                dim_chunk = min(
                    int(remaining_elements ** (1 / len(dataset_shape))), dim_size
                )
                chunk_shape.append(dim_chunk)
                remaining_elements = max(1, remaining_elements // dim_chunk)

        elif access_pattern == AccessPattern.SPARSE:
            # For sparse access, use smaller chunks
            target_elements = target_elements // 2  # Smaller chunks for sparse access
            chunk_shape = []
            for dim_size in dataset_shape:
                dim_chunk = min(
                    int((target_elements / len(dataset_shape)) ** 0.5), dim_size
                )
                chunk_shape.append(max(1, dim_chunk))

        else:  # RANDOM
            # For random access, use balanced moderate-sized chunks
            chunk_shape = []
            elements_per_dim = int(target_elements ** (1 / len(dataset_shape)))
            for dim_size in dataset_shape:
                chunk_shape.append(min(elements_per_dim, dim_size))

        return tuple(chunk_shape)


class ReadAheadCache:
    """
    Intelligent read-ahead cache for HDF5 data based on access patterns.
    """

    def __init__(self, max_cache_mb: float = 200.0):
        self.max_cache_mb = max_cache_mb
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_patterns = defaultdict(deque)
        self.prediction_history = defaultdict(list)
        self._lock = threading.RLock()
        self.memory_manager = get_memory_manager()

    def record_access(
        self, file_path: str, dataset_path: str, slice_info: tuple[slice, ...] | None
    ):
        """Record an access for pattern analysis."""
        key = f"{file_path}:{dataset_path}"
        with self._lock:
            self.access_patterns[key].append((time.time(), slice_info))
            # Keep only recent accesses
            if len(self.access_patterns[key]) > 20:
                self.access_patterns[key].popleft()

    def predict_next_access(
        self, file_path: str, dataset_path: str, current_slice: tuple[slice, ...] | None
    ) -> list[tuple[slice, ...]]:
        """
        Predict next likely access patterns for read-ahead.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_path : str
            Path to dataset
        current_slice : Optional[tuple[slice, ...]]
            Current slice being accessed

        Returns
        -------
        list[tuple[slice, ...]]
            Predicted next slice accesses
        """
        key = f"{file_path}:{dataset_path}"
        predictions = []

        with self._lock:
            if key not in self.access_patterns or len(self.access_patterns[key]) < 2:
                return predictions

            recent_accesses = list(self.access_patterns[key])[-5:]

            # Analyze pattern and predict
            if self._is_sequential_access(recent_accesses):
                predictions.extend(self._predict_sequential(current_slice))
            elif self._is_sliding_window(recent_accesses):
                predictions.extend(self._predict_sliding_window(current_slice))

        return predictions[:3]  # Limit to 3 predictions

    def _is_sequential_access(
        self, accesses: list[tuple[float, tuple[slice, ...]]]
    ) -> bool:
        """Check if recent accesses show sequential pattern."""
        if len(accesses) < 3:
            return False

        slice_starts = []
        for _, slice_info in accesses:
            if slice_info and len(slice_info) > 0 and isinstance(slice_info[0], slice):
                slice_starts.append(slice_info[0].start or 0)

        if len(slice_starts) < 3:
            return False

        # Check for increasing sequence
        diffs = np.diff(slice_starts)
        return all(d > 0 for d in diffs) and np.std(diffs) < np.mean(diffs) * 0.5

    def _is_sliding_window(
        self, accesses: list[tuple[float, tuple[slice, ...]]]
    ) -> bool:
        """Check if accesses show sliding window pattern."""
        if len(accesses) < 3:
            return False

        # Check for overlapping slices
        slice_ranges = []
        for _, slice_info in accesses:
            if slice_info and len(slice_info) > 0 and isinstance(slice_info[0], slice):
                start = slice_info[0].start or 0
                stop = slice_info[0].stop or start + 1
                slice_ranges.append((start, stop))

        if len(slice_ranges) < 3:
            return False

        # Check for overlapping ranges
        overlaps = 0
        for i in range(len(slice_ranges) - 1):
            if slice_ranges[i][1] > slice_ranges[i + 1][0]:
                overlaps += 1

        return overlaps >= len(slice_ranges) - 2

    def _predict_sequential(
        self, current_slice: tuple[slice, ...] | None
    ) -> list[tuple[slice, ...]]:
        """Predict next slices for sequential access."""
        predictions = []
        if (
            current_slice
            and len(current_slice) > 0
            and isinstance(current_slice[0], slice)
        ):
            start = current_slice[0].start or 0
            stop = current_slice[0].stop or start + 1
            step = stop - start

            # Predict next 2 sequential slices
            for i in range(1, 3):
                next_start = stop + (i - 1) * step
                next_stop = next_start + step
                next_slice = (slice(next_start, next_stop), *list(current_slice[1:]))
                predictions.append(next_slice)

        return predictions

    def _predict_sliding_window(
        self, current_slice: tuple[slice, ...] | None
    ) -> list[tuple[slice, ...]]:
        """Predict next slices for sliding window access."""
        predictions = []
        if (
            current_slice
            and len(current_slice) > 0
            and isinstance(current_slice[0], slice)
        ):
            start = current_slice[0].start or 0
            stop = current_slice[0].stop or start + 1
            window_size = stop - start

            # Predict shifted windows
            for shift in [window_size // 4, window_size // 2]:
                next_start = start + shift
                next_stop = stop + shift
                next_slice = (slice(next_start, next_stop), *list(current_slice[1:]))
                predictions.append(next_slice)

        return predictions

    def get_cached_data(
        self, file_path: str, dataset_path: str, slice_info: tuple[slice, ...] | None
    ) -> np.ndarray | None:
        """Get cached data if available."""
        cache_key = f"{file_path}:{dataset_path}:{slice_info}"
        with self._lock:
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                entry.access_count += 1
                entry.last_accessed = time.time()
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                logger.debug(f"Cache hit for {cache_key}")
                return entry.data
        return None

    def cache_data(
        self,
        file_path: str,
        dataset_path: str,
        slice_info: tuple[slice, ...] | None,
        data: np.ndarray,
    ):
        """Cache data with intelligent eviction."""
        if data is None:
            return

        cache_key = f"{file_path}:{dataset_path}:{slice_info}"
        data_size_mb = data.nbytes / (1024 * 1024)

        # Skip caching very large chunks
        if data_size_mb > self.max_cache_mb / 2:
            logger.debug(f"Skipping cache for large data: {data_size_mb:.1f}MB")
            return

        with self._lock:
            # Make space if needed
            self._make_space(data_size_mb)

            # Create cache entry
            entry = CacheEntry(
                data=data.copy(),
                file_path=file_path,
                dataset_path=dataset_path,
                slice_info=slice_info,
                access_count=1,
                last_accessed=time.time(),
                created_time=time.time(),
                size_mb=data_size_mb,
            )

            self.cache[cache_key] = entry
            logger.debug(f"Cached data: {cache_key} ({data_size_mb:.1f}MB)")

    def _make_space(self, required_mb: float):
        """Make space in cache using LRU eviction."""
        current_size = sum(entry.size_mb for entry in self.cache.values())

        # Evict if we exceed capacity or need space
        while current_size + required_mb > self.max_cache_mb and self.cache:
            # Remove least recently used item
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            current_size -= oldest_entry.size_mb
            logger.debug(
                f"Evicted cache entry: {oldest_key} ({oldest_entry.size_mb:.1f}MB)"
            )

    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            freed_mb = sum(entry.size_mb for entry in self.cache.values())
            self.cache.clear()
            logger.info(f"Cleared read-ahead cache, freed {freed_mb:.1f}MB")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_size = sum(entry.size_mb for entry in self.cache.values())
            return {
                "cache_entries": len(self.cache),
                "total_size_mb": total_size,
                "max_size_mb": self.max_cache_mb,
                "utilization": total_size / self.max_cache_mb
                if self.max_cache_mb > 0
                else 0,
            }


class EnhancedHDF5Reader:
    """
    Enhanced HDF5 reader with intelligent chunking and read-ahead caching.
    """

    def __init__(self, max_cache_mb: float = 200.0, enable_read_ahead: bool = True):
        self.max_cache_mb = max_cache_mb
        self.enable_read_ahead = enable_read_ahead

        # Core components
        self.chunker = IntelligentChunker()
        self.cache = ReadAheadCache(max_cache_mb)
        self.memory_manager = get_memory_manager()

        # Connection management
        self._connections = weakref.WeakValueDictionary()
        self._connection_lock = threading.Lock()

        # Statistics
        self.stats = {
            "reads": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_read": 0,
            "read_ahead_predictions": 0,
        }

        logger.info(f"EnhancedHDF5Reader initialized with {max_cache_mb}MB cache")

    @contextmanager
    def get_file_connection(self, file_path: str):
        """Get HDF5 file connection with connection pooling."""
        with self._connection_lock:
            if file_path in self._connections:
                file_obj = self._connections[file_path]
                if file_obj.id.valid:
                    yield file_obj
                    return

            # Create new connection
            try:
                file_obj = h5py.File(file_path, "r")
                self._connections[file_path] = file_obj
                yield file_obj
            except Exception as e:
                logger.error(f"Failed to open HDF5 file {file_path}: {e}")
                raise

    def read_dataset(
        self,
        file_path: str,
        dataset_path: str,
        slice_info: tuple[slice, ...] | None = None,
        enable_read_ahead: bool | None = None,
    ) -> np.ndarray:
        """
        Read dataset with intelligent caching and read-ahead.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_path : str
            Path to dataset within file
        slice_info : Optional[tuple[slice, ...]]
            Slice to read (None for full dataset)
        enable_read_ahead : Optional[bool]
            Override global read-ahead setting

        Returns
        -------
        np.ndarray
            Requested data
        """
        self.stats["reads"] += 1

        # Check cache first
        cached_data = self.cache.get_cached_data(file_path, dataset_path, slice_info)
        if cached_data is not None:
            self.stats["cache_hits"] += 1
            self.cache.record_access(file_path, dataset_path, slice_info)
            return cached_data

        self.stats["cache_misses"] += 1

        # Read from file
        with self.get_file_connection(file_path) as f:
            dataset = f[dataset_path]

            # Read requested data
            data = dataset[slice_info] if slice_info else dataset[:]

            data = np.array(data)  # Ensure we own the data
            self.stats["bytes_read"] += data.nbytes

            # Cache the data
            self.cache.cache_data(file_path, dataset_path, slice_info, data)

            # Record access for pattern analysis
            self.cache.record_access(file_path, dataset_path, slice_info)

            # Trigger read-ahead if enabled
            if (
                enable_read_ahead
                if enable_read_ahead is not None
                else self.enable_read_ahead
            ):
                self._trigger_read_ahead(
                    file_path, dataset_path, slice_info, dataset.shape, dataset.dtype
                )

            return data

    def _trigger_read_ahead(
        self,
        file_path: str,
        dataset_path: str,
        current_slice: tuple[slice, ...] | None,
        dataset_shape: tuple[int, ...],
        dtype: np.dtype,
    ):
        """Trigger read-ahead for predicted accesses."""
        # Check memory pressure before read-ahead
        pressure = self.memory_manager.get_memory_pressure()
        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.debug("Skipping read-ahead due to memory pressure")
            return

        # Get predictions
        predictions = self.cache.predict_next_access(
            file_path, dataset_path, current_slice
        )

        if predictions:
            self.stats["read_ahead_predictions"] += len(predictions)
            logger.debug(
                f"Read-ahead: {len(predictions)} predictions for {dataset_path}"
            )

            # Execute read-ahead in background (simplified - could use threading)
            for pred_slice in predictions:
                try:
                    # Validate slice bounds
                    if self._is_valid_slice(pred_slice, dataset_shape):
                        # Check if already cached
                        if (
                            self.cache.get_cached_data(
                                file_path, dataset_path, pred_slice
                            )
                            is None
                        ):
                            # Read and cache
                            with self.get_file_connection(file_path) as f:
                                dataset = f[dataset_path]
                                pred_data = dataset[pred_slice]
                                pred_data = np.array(pred_data)
                                self.cache.cache_data(
                                    file_path, dataset_path, pred_slice, pred_data
                                )
                                logger.debug(f"Read-ahead cached: {pred_slice}")
                except Exception as e:
                    logger.debug(f"Read-ahead failed for {pred_slice}: {e}")

    def _is_valid_slice(
        self, slice_info: tuple[slice, ...], shape: tuple[int, ...]
    ) -> bool:
        """Check if slice is valid for dataset shape."""
        if len(slice_info) > len(shape):
            return False

        for i, s in enumerate(slice_info):
            if isinstance(s, slice):
                start = s.start or 0
                stop = s.stop or shape[i]
                if start < 0 or stop > shape[i] or start >= stop:
                    return False

        return True

    def read_multiple_datasets(
        self,
        file_path: str,
        dataset_paths: list[str],
        slice_info: tuple[slice, ...] | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Efficiently read multiple datasets from the same file.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_paths : list[str]
            List of dataset paths to read
        slice_info : Optional[tuple[slice, ...]]
            Slice to apply to all datasets

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of dataset_path -> data
        """
        results = {}

        with self.get_file_connection(file_path) as f:
            for dataset_path in dataset_paths:
                try:
                    # Check cache first
                    cached_data = self.cache.get_cached_data(
                        file_path, dataset_path, slice_info
                    )
                    if cached_data is not None:
                        results[dataset_path] = cached_data
                        self.stats["cache_hits"] += 1
                        continue

                    # Read from file
                    dataset = f[dataset_path]
                    data = dataset[slice_info] if slice_info else dataset[:]

                    data = np.array(data)
                    results[dataset_path] = data
                    self.stats["cache_misses"] += 1
                    self.stats["bytes_read"] += data.nbytes

                    # Cache the data
                    self.cache.cache_data(file_path, dataset_path, slice_info, data)

                except Exception as e:
                    logger.warning(f"Failed to read dataset {dataset_path}: {e}")

        return results

    def get_dataset_info(self, file_path: str, dataset_path: str) -> dict[str, Any]:
        """
        Get dataset metadata without loading data.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_path : str
            Path to dataset

        Returns
        -------
        dict[str, Any]
            Dataset metadata
        """
        with self.get_file_connection(file_path) as f:
            dataset = f[dataset_path]
            return {
                "shape": dataset.shape,
                "dtype": dataset.dtype,
                "size_mb": np.prod(dataset.shape)
                * dataset.dtype.itemsize
                / (1024 * 1024),
                "chunks": dataset.chunks,
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
            }

    def optimize_chunking_for_dataset(
        self,
        file_path: str,
        dataset_path: str,
        access_pattern: AccessPattern | None = None,
    ) -> tuple[int, ...]:
        """
        Get optimal chunk shape for dataset based on access pattern.

        Parameters
        ----------
        file_path : str
            Path to HDF5 file
        dataset_path : str
            Path to dataset
        access_pattern : Optional[AccessPattern]
            Known access pattern (auto-detected if None)

        Returns
        -------
        tuple[int, ...]
            Optimal chunk shape
        """
        info = self.get_dataset_info(file_path, dataset_path)

        if access_pattern is None:
            # Auto-detect pattern from access history
            key = f"{file_path}:{dataset_path}"
            recent_accesses = [
                slice_info for _, slice_info in self.cache.access_patterns[key]
            ]
            access_pattern = self.chunker.analyze_access_pattern(
                file_path, dataset_path, recent_accesses
            )

        return self.chunker.get_optimal_chunk_shape(
            info["shape"], info["dtype"], access_pattern
        )

    def clear_caches(self):
        """Clear all caches."""
        self.cache.clear_cache()
        logger.info("Cleared all HDF5 caches")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = self.stats.copy()
        stats["cache_hit_rate"] = stats["cache_hits"] / max(
            1, stats["cache_hits"] + stats["cache_misses"]
        )
        stats["mb_read"] = stats["bytes_read"] / (1024 * 1024)
        stats.update(self.cache.get_cache_stats())
        return stats


# Global enhanced reader instance
_global_enhanced_reader: EnhancedHDF5Reader | None = None


def get_enhanced_hdf5_reader() -> EnhancedHDF5Reader:
    """Get or create the global enhanced HDF5 reader."""
    global _global_enhanced_reader
    if _global_enhanced_reader is None:
        _global_enhanced_reader = EnhancedHDF5Reader()
    return _global_enhanced_reader


def get_enhanced_reader() -> EnhancedHDF5Reader:
    """Alias for get_enhanced_hdf5_reader for backward compatibility."""
    return get_enhanced_hdf5_reader()


def read_hdf5_optimized(
    file_path: str, dataset_path: str, slice_info: tuple[slice, ...] | None = None
) -> np.ndarray:
    """Convenience function for optimized HDF5 reading."""
    return get_enhanced_hdf5_reader().read_dataset(file_path, dataset_path, slice_info)


def read_multiple_hdf5_optimized(
    file_path: str,
    dataset_paths: list[str],
    slice_info: tuple[slice, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Convenience function for reading multiple datasets."""
    return get_enhanced_hdf5_reader().read_multiple_datasets(
        file_path, dataset_paths, slice_info
    )
