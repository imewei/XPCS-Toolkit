"""
Lazy Loading System for XPCS Data

This module provides smart data loading that minimizes memory usage by
deferring HDF5 reads until data is accessed and releasing loaded arrays
under memory pressure.
"""

import threading
import time
import weakref
from typing import Any

import h5py
import numpy as np

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


class LazyHDF5Array:
    """Lazy loader for HDF5 array data with intelligent slicing."""

    def __init__(
        self,
        data_key: str,
        hdf5_path: str,
        dataset_path: str,
        estimated_size_mb: float,
        chunk_size_mb: float = 100.0,
    ):
        self.data_key = data_key
        self.hdf5_path = hdf5_path
        self.dataset_path = dataset_path
        self.estimated_size_mb = estimated_size_mb
        self.chunk_size_mb = chunk_size_mb
        self._loaded_data: np.ndarray | None = None
        self._last_access = 0.0
        self._access_count = 0
        self._loader: IntelligentLazyLoader | None = None
        self._lock = threading.RLock()
        self._metadata_cache: dict[str, tuple[int, ...]] = {}

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

    @property
    def shape(self) -> tuple[int, ...]:
        """Get data shape, from loaded data or file metadata."""
        if self._loaded_data is not None:
            return self._loaded_data.shape
        return self._get_shape_from_metadata()

    def _record_access(self) -> None:
        """Record access for idle-cleanup tracking and notify the loader."""
        self._last_access = time.time()
        self._access_count += 1
        loader = self._loader
        if loader is not None:
            loader.notify_access()

    def __getitem__(self, key):
        """Get data slice, with intelligent loading."""
        with self._lock:
            # If full data is loaded, use it
            if self._loaded_data is not None:
                self._record_access()
                return self._loaded_data[key]

            # For slice access, load only what's needed for large datasets
            if self.estimated_size_mb > self.chunk_size_mb:
                # Load only the requested slice
                data = self._load_hdf5_data(slice_info=key)
                self._record_access()
                return data
            # For smaller datasets, load everything
            self._loaded_data = self._load_hdf5_data()
            self._record_access()
            return self._loaded_data[key]

    def __array__(self):
        """Convert to numpy array, loading if necessary."""
        with self._lock:
            if self._loaded_data is None:
                self._loaded_data = self._load_hdf5_data()
            self._record_access()
            return self._loaded_data

    @property
    def nbytes(self):
        """Get size in bytes."""
        shape = self.shape
        if shape:
            # Estimate 4 bytes per element for float32
            return np.prod(shape) * 4
        return 0


class IntelligentLazyLoader:
    """Lazy loading system with memory-pressure-aware cleanup."""

    def __init__(self, max_memory_mb: float = 1024.0):
        self.max_memory_mb = max_memory_mb

        # Storage for lazy data proxies
        self.data_proxies: dict[str, LazyHDF5Array] = {}
        self.weak_refs: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

        # Memory management
        self.memory_manager = get_memory_manager()

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
        with self._lock:
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
        proxy._loader = self  # Back-reference for cleanup-on-access

        with self._lock:
            self.data_proxies[data_key] = proxy
            self.weak_refs[data_key] = proxy

        logger.debug(
            f"Registered lazy HDF5 data: {data_key} ({estimated_size_mb:.1f}MB)"
        )
        return proxy

    def get_data(self, data_key: str) -> LazyHDF5Array | None:
        """Get lazy data proxy by key."""
        with self._lock:
            return self.data_proxies.get(data_key)

    def notify_access(self):
        """Called by a proxy on access; triggers cleanup under memory pressure."""
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
        with self._lock:
            proxies = list(self.data_proxies.items())
        for key, proxy in proxies:
            if (
                proxy._loaded_data is not None
                and current_time - proxy._last_access > cleanup_threshold
            ):
                keys_to_cleanup.append(key)

        for key in keys_to_cleanup:
            with self._lock:
                proxy = self.data_proxies.get(key)
            if proxy is None:
                continue
            if proxy._loaded_data is not None:
                memory_freed = proxy._loaded_data.nbytes / (1024 * 1024)
                proxy._loaded_data = None
                logger.debug(f"Cleaned up lazy data {key}, freed {memory_freed:.1f}MB")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get memory usage statistics for lazy loader."""
        with self._lock:
            proxies = list(self.data_proxies.values())
        total_registered_mb = sum(
            proxy.estimated_size_mb for proxy in proxies
        )
        total_loaded_mb = sum(
            proxy._loaded_data.nbytes / (1024 * 1024)
            for proxy in proxies
            if proxy._loaded_data is not None
        )

        return {
            "total_registered_data_mb": total_registered_mb,
            "total_loaded_data_mb": total_loaded_mb,
            "num_registered_datasets": len(proxies),
            "num_loaded_datasets": sum(
                1
                for proxy in proxies
                if proxy._loaded_data is not None
            ),
            "memory_efficiency": 1.0
            - (total_loaded_mb / max(1.0, total_registered_mb)),
        }

    def shutdown(self):
        """Shutdown the lazy loader system."""
        # Clear all loaded data
        with self._lock:
            proxies = list(self.data_proxies.values())
        for proxy in proxies:
            proxy._loaded_data = None

        logger.info("IntelligentLazyLoader shutdown complete")


# Global lazy loader instance
_global_lazy_loader: IntelligentLazyLoader | None = None
_global_lazy_loader_lock = threading.Lock()


def get_lazy_loader() -> IntelligentLazyLoader:
    """Get or create the global lazy loader instance.

    Uses double-checked locking to be thread-safe under concurrent first
    access from multiple threads without paying the lock cost on every
    subsequent call. (BUG-030)
    """
    global _global_lazy_loader  # noqa: PLW0603 - intentional singleton pattern
    if _global_lazy_loader is None:
        with _global_lazy_loader_lock:
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
    global _global_lazy_loader  # noqa: PLW0603 - intentional singleton pattern
    if _global_lazy_loader:
        _global_lazy_loader.shutdown()
        _global_lazy_loader = None
