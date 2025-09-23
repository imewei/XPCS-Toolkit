"""
Intelligent Lazy Loading System for XPCS Data

This module provides smart data loading that minimizes memory usage while
maintaining performance through intelligent prefetching and caching strategies.
"""

import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import h5py
import numpy as np

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


class AccessPattern(Enum):
    """Types of data access patterns for prediction."""

    SEQUENTIAL = "sequential"  # Access data in order
    RANDOM = "random"  # Random access pattern
    SLIDING_WINDOW = "sliding"  # Moving window access
    FULL_SCAN = "full_scan"  # Complete dataset access
    SPARSE = "sparse"  # Sparse, irregular access


@dataclass
class AccessEvent:
    """Record of a data access event."""

    timestamp: float
    data_key: str
    slice_info: tuple[slice, ...] | None
    access_size_mb: float
    cache_hit: bool


@dataclass
class PrefetchRequest:
    """Request for data prefetching."""

    data_key: str
    priority: float  # 0.0 to 1.0, higher is more urgent
    estimated_access_time: float
    slice_info: tuple[slice, ...] | None
    justification: str  # Why this prefetch is requested


class LazyDataProxy(ABC):
    """Abstract base class for lazy-loaded data proxies."""

    def __init__(
        self, data_key: str, loader_func: Callable[..., Any], estimated_size_mb: float
    ) -> None:
        self.data_key = data_key
        self.loader_func = loader_func
        self.estimated_size_mb = estimated_size_mb
        self._loaded_data = None
        self._last_access = 0.0
        self._access_count = 0
        self._loader = None
        self._lock = threading.RLock()

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Get data slice, loading if necessary."""

    @abstractmethod
    def __array__(self) -> np.ndarray[Any, Any]:
        """Convert to numpy array, loading if necessary."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Get data shape without loading."""
        if self._loaded_data is not None:
            return self._loaded_data.shape
        # Try to get shape from metadata without loading
        return self._get_shape_from_metadata()

    @abstractmethod
    def _get_shape_from_metadata(self) -> tuple[int, ...]:
        """Get shape from file metadata without loading data."""

    def _record_access(
        self, slice_info: tuple[slice, ...] | None = None, cache_hit: bool = True
    ) -> None:
        """Record access for pattern analysis."""
        if self._loader:
            access_event = AccessEvent(
                timestamp=time.time(),
                data_key=self.data_key,
                slice_info=slice_info,
                access_size_mb=self.estimated_size_mb,
                cache_hit=cache_hit,
            )
            self._loader._record_access(access_event)

        self._last_access = time.time()
        self._access_count += 1


class LazyHDF5Array(LazyDataProxy):
    """Lazy loader for HDF5 array data with intelligent slicing."""

    def __init__(
        self,
        data_key: str,
        hdf5_path: str,
        dataset_path: str,
        estimated_size_mb: float,
        chunk_size_mb: float = 100.0,
    ):
        super().__init__(data_key, self._load_hdf5_data, estimated_size_mb)
        self.hdf5_path = hdf5_path
        self.dataset_path = dataset_path
        self.chunk_size_mb = chunk_size_mb
        self._metadata_cache = {}

    def _load_hdf5_data(self, slice_info=None):
        """Load data from HDF5 file, optionally with slicing."""
        try:
            with h5py.File(self.hdf5_path, "r") as f:
                dataset = f[self.dataset_path]

                if slice_info:
                    # Load only requested slice
                    data = dataset[slice_info]
                    logger.debug(f"Loaded HDF5 slice {slice_info} from {self.data_key}")
                else:
                    # Load full dataset
                    data = dataset[:]
                    logger.debug(f"Loaded full HDF5 dataset {self.data_key}")

                return np.array(data)

        except Exception as e:
            logger.error(f"Failed to load HDF5 data {self.data_key}: {e}")
            raise

    def _get_shape_from_metadata(self):
        """Get shape from HDF5 metadata without loading data."""
        cache_key = f"{self.hdf5_path}:{self.dataset_path}:shape"
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        try:
            with h5py.File(self.hdf5_path, "r") as f:
                shape = f[self.dataset_path].shape
                self._metadata_cache[cache_key] = shape
                return shape
        except Exception as e:
            logger.warning(f"Could not get shape metadata for {self.data_key}: {e}")
            return None

    def __getitem__(self, key):
        """Get data slice, with intelligent loading."""
        with self._lock:
            # If full data is loaded, use it
            if self._loaded_data is not None:
                self._record_access(slice_info=key, cache_hit=True)
                return self._loaded_data[key]

            # For slice access, load only what's needed for large datasets
            if self.estimated_size_mb > self.chunk_size_mb:
                # Load only the requested slice
                data = self._load_hdf5_data(slice_info=key)
                self._record_access(slice_info=key, cache_hit=False)
                return data
            # For smaller datasets, load everything
            self._loaded_data = self._load_hdf5_data()
            self._record_access(cache_hit=False)
            return self._loaded_data[key]

    def __array__(self):
        """Convert to numpy array, loading if necessary."""
        with self._lock:
            if self._loaded_data is None:
                self._loaded_data = self._load_hdf5_data()
                self._record_access(cache_hit=False)
            else:
                self._record_access(cache_hit=True)

            return self._loaded_data

    @property
    def nbytes(self):
        """Get size in bytes."""
        shape = self.shape
        if shape:
            # Estimate 4 bytes per element for float32
            return np.prod(shape) * 4
        return 0


class IntelligentPrefetcher:
    """Intelligent data prefetching based on access patterns."""

    def __init__(self, max_prefetch_mb: float = 500.0):
        self.max_prefetch_mb = max_prefetch_mb
        self.access_history: deque = deque(maxlen=1000)
        self.pattern_cache: dict[str, AccessPattern] = {}
        self.active_prefetches: set[str] = set()
        self.prefetch_thread = None
        self.prefetch_queue: deque = deque()
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        # Start prefetch worker thread
        self._start_prefetch_worker()

    def _start_prefetch_worker(self):
        """Start background prefetch worker thread."""
        import os

        # Skip starting background threads in test mode to prevent threading issues
        if os.environ.get("XPCS_TEST_MODE") == "1":
            return

        self.prefetch_thread = threading.Thread(
            target=self._prefetch_worker, daemon=True
        )
        self.prefetch_thread.start()

    def _prefetch_worker(self):
        """Background worker for executing prefetch requests."""
        while not self._shutdown_event.wait(timeout=0.1):
            try:
                with self._lock:
                    if not self.prefetch_queue:
                        continue

                    # Get highest priority prefetch request
                    request = max(self.prefetch_queue, key=lambda x: x.priority)
                    self.prefetch_queue.remove(request)

                if request.data_key not in self.active_prefetches:
                    self._execute_prefetch(request)

            except Exception as e:
                logger.warning(f"Prefetch worker error: {e}")

    def _execute_prefetch(self, request: PrefetchRequest):
        """Execute a prefetch request."""
        try:
            self.active_prefetches.add(request.data_key)
            logger.debug(f"Prefetching {request.data_key}: {request.justification}")

            # This would trigger the lazy loader to load data
            # Implementation depends on how the loader is integrated

        except Exception as e:
            logger.warning(f"Prefetch failed for {request.data_key}: {e}")
        finally:
            self.active_prefetches.discard(request.data_key)

    def analyze_access_pattern(self, data_key: str) -> AccessPattern:
        """Analyze access pattern for a specific data key."""
        if data_key in self.pattern_cache:
            return self.pattern_cache[data_key]

        # Get recent accesses for this data key
        recent_accesses = [
            event
            for event in self.access_history
            if event.data_key == data_key
            and event.timestamp > time.time() - 300  # Last 5 minutes
        ]

        if len(recent_accesses) < 3:
            pattern = AccessPattern.RANDOM
        else:
            # Analyze temporal pattern
            timestamps = [event.timestamp for event in recent_accesses]
            intervals = np.diff(timestamps)

            # Check for regular intervals (sequential access)
            if np.std(intervals) < np.mean(intervals) * 0.3:  # Low variance
                pattern = AccessPattern.SEQUENTIAL
            else:
                # Check for sliding window pattern
                slice_info_list = [
                    event.slice_info for event in recent_accesses if event.slice_info
                ]
                if len(slice_info_list) >= 2:
                    # Analyze slice patterns
                    pattern = AccessPattern.SLIDING_WINDOW
                else:
                    pattern = AccessPattern.RANDOM

        self.pattern_cache[data_key] = pattern
        return pattern

    def predict_next_access(self, data_key: str) -> list[PrefetchRequest]:
        """Predict next data access and generate prefetch requests."""
        pattern = self.analyze_access_pattern(data_key)
        requests = []

        if pattern == AccessPattern.SEQUENTIAL:
            # Predict next sequential access
            requests.append(
                PrefetchRequest(
                    data_key=f"{data_key}_next_sequential",
                    priority=0.8,
                    estimated_access_time=time.time() + 10.0,
                    slice_info=None,
                    justification="Sequential access pattern detected",
                )
            )

        elif pattern == AccessPattern.SLIDING_WINDOW:
            # Predict window advancement
            requests.append(
                PrefetchRequest(
                    data_key=f"{data_key}_window_advance",
                    priority=0.7,
                    estimated_access_time=time.time() + 5.0,
                    slice_info=None,
                    justification="Sliding window pattern detected",
                )
            )

        return requests

    def record_access(self, event: AccessEvent):
        """Record an access event for pattern analysis."""
        self.access_history.append(event)

        # Generate predictions based on new access
        predictions = self.predict_next_access(event.data_key)
        for prediction in predictions:
            with self._lock:
                self.prefetch_queue.append(prediction)

    def shutdown(self):
        """Shutdown the prefetcher."""
        self._shutdown_event.set()
        if self.prefetch_thread:
            self.prefetch_thread.join(timeout=5.0)


class IntelligentLazyLoader:
    """
    Main lazy loading system with intelligent prefetching and memory management.
    """

    def __init__(
        self,
        max_memory_mb: float = 1024.0,
        prefetch_memory_mb: float = 256.0,
        enable_prefetching: bool = True,
    ):
        self.max_memory_mb = max_memory_mb
        self.prefetch_memory_mb = prefetch_memory_mb
        self.enable_prefetching = enable_prefetching

        # Storage for lazy data proxies
        self.data_proxies: dict[str, LazyDataProxy] = {}
        self.weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

        # Access pattern analysis
        self.access_history: deque = deque(maxlen=2000)
        self.pattern_analyzer = None

        # Memory management
        self.memory_manager = get_memory_manager()

        # Prefetching system
        if enable_prefetching:
            self.prefetcher = IntelligentPrefetcher(max_prefetch_mb=prefetch_memory_mb)
        else:
            self.prefetcher = None

        logger.info(f"IntelligentLazyLoader initialized with {max_memory_mb}MB limit")

    def register_hdf5_data(
        self, data_key: str, hdf5_path: str, dataset_path: str, estimated_size_mb: float
    ) -> LazyHDF5Array:
        """
        Register HDF5 dataset for lazy loading.

        Parameters
        ----------
        data_key : str
            Unique identifier for this data
        hdf5_path : str
            Path to HDF5 file
        dataset_path : str
            Path within HDF5 file to dataset
        estimated_size_mb : float
            Estimated size of dataset in MB

        Returns
        -------
        LazyHDF5Array
            Lazy data proxy
        """
        if data_key in self.data_proxies:
            logger.debug(f"Data key {data_key} already registered")
            return self.data_proxies[data_key]

        # Check memory pressure before registering large datasets
        if estimated_size_mb > 100:  # Large dataset
            pressure = self.memory_manager.get_memory_pressure()
            if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
                logger.warning(
                    f"High memory pressure, large dataset {data_key} may use chunked loading"
                )

        proxy = LazyHDF5Array(data_key, hdf5_path, dataset_path, estimated_size_mb)
        proxy._loader = self  # Set back-reference for access recording

        self.data_proxies[data_key] = proxy
        self.weak_refs[data_key] = proxy

        logger.debug(
            f"Registered lazy HDF5 data: {data_key} ({estimated_size_mb:.1f}MB)"
        )
        return proxy

    def get_data(self, data_key: str) -> LazyDataProxy | None:
        """Get lazy data proxy by key."""
        return self.data_proxies.get(data_key)

    def _record_access(self, event: AccessEvent):
        """Record access event for analysis."""
        self.access_history.append(event)

        # Forward to prefetcher if enabled
        if self.prefetcher:
            self.prefetcher.record_access(event)

        # Check if we should trigger cleanup
        self._check_memory_cleanup()

    def _check_memory_cleanup(self):
        """Check if memory cleanup is needed."""
        pressure = self.memory_manager.get_memory_pressure()

        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning("Memory pressure detected, cleaning up lazy-loaded data")
            self._cleanup_unused_data()

    def _cleanup_unused_data(self):
        """Clean up unused lazy-loaded data."""
        current_time = time.time()
        cleanup_threshold = 300  # 5 minutes

        keys_to_cleanup = []
        for key, proxy in self.data_proxies.items():
            if (
                proxy._loaded_data is not None
                and current_time - proxy._last_access > cleanup_threshold
            ):
                keys_to_cleanup.append(key)

        for key in keys_to_cleanup:
            proxy = self.data_proxies[key]
            if proxy._loaded_data is not None:
                memory_freed = proxy._loaded_data.nbytes / (1024 * 1024)
                proxy._loaded_data = None
                logger.debug(f"Cleaned up lazy data {key}, freed {memory_freed:.1f}MB")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics for lazy loader."""
        total_registered_mb = sum(
            proxy.estimated_size_mb for proxy in self.data_proxies.values()
        )
        total_loaded_mb = sum(
            proxy._loaded_data.nbytes / (1024 * 1024)
            for proxy in self.data_proxies.values()
            if proxy._loaded_data is not None
        )

        return {
            "total_registered_data_mb": total_registered_mb,
            "total_loaded_data_mb": total_loaded_mb,
            "num_registered_datasets": len(self.data_proxies),
            "num_loaded_datasets": sum(
                1
                for proxy in self.data_proxies.values()
                if proxy._loaded_data is not None
            ),
            "memory_efficiency": 1.0
            - (total_loaded_mb / max(1.0, total_registered_mb)),
            "access_events_recorded": len(self.access_history),
        }

    def shutdown(self):
        """Shutdown the lazy loader system."""
        if self.prefetcher:
            self.prefetcher.shutdown()

        # Clear all loaded data
        for proxy in self.data_proxies.values():
            proxy._loaded_data = None

        logger.info("IntelligentLazyLoader shutdown complete")


# Global lazy loader instance
_global_lazy_loader: IntelligentLazyLoader | None = None


def get_lazy_loader() -> IntelligentLazyLoader:
    """Get or create the global lazy loader instance."""
    global _global_lazy_loader
    if _global_lazy_loader is None:
        _global_lazy_loader = IntelligentLazyLoader()
    return _global_lazy_loader


def register_lazy_hdf5(
    data_key: str, hdf5_path: str, dataset_path: str, estimated_size_mb: float
) -> LazyHDF5Array:
    """Convenience function for registering HDF5 data for lazy loading."""
    return get_lazy_loader().register_hdf5_data(
        data_key, hdf5_path, dataset_path, estimated_size_mb
    )


def shutdown_lazy_loader():
    """Shutdown the global lazy loader."""
    global _global_lazy_loader
    if _global_lazy_loader:
        _global_lazy_loader.shutdown()
        _global_lazy_loader = None
