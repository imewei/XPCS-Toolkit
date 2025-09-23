# Standard library imports
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any

# Third-party imports
import h5py
import numpy as np
import psutil

from xpcs_toolkit.utils.logging_config import get_logger

# Local imports
from .aps_8idi import key as hdf_key

logger = get_logger(__name__)

# Performance monitoring stub for testing
_perf_monitor = None

# Memory pressure thresholds for connection pool management
MEMORY_PRESSURE_CRITICAL = 0.90  # 90% - Critical memory pressure
MEMORY_PRESSURE_HIGH = 0.80  # 80% - High memory pressure
MEMORY_PRESSURE_MODERATE = 0.70  # 70% - Moderate memory pressure


class PooledConnection:
    """Wrapper for pooled HDF5 connections with metadata."""

    def __init__(self, file_handle: h5py.File, file_path: str):
        self.file_handle = file_handle
        self.file_path = file_path
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.is_healthy = True
        self.lock = threading.RLock()

    def touch(self) -> None:
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def check_health(self) -> bool:
        """Check if connection is still healthy."""
        try:
            with self.lock:
                # Try to access a basic property
                _ = self.file_handle.filename
                # Check if file still exists
                if not os.path.exists(self.file_path):
                    self.is_healthy = False
                    return False
                self.is_healthy = True
                return True
        except (ValueError, OSError, AttributeError) as e:
            logger.debug(f"Connection health check failed for {self.file_path}: {e}")
            self.is_healthy = False
            return False

    def close(self) -> None:
        """Close the connection safely."""
        try:
            with self.lock:
                if hasattr(self.file_handle, "close"):
                    self.file_handle.close()
        except Exception as e:
            logger.debug(f"Error closing connection for {self.file_path}: {e}")
        finally:
            self.is_healthy = False


class ConnectionStats:
    """Statistics tracker for HDF5 connections."""

    def __init__(self) -> None:
        self.total_connections_created = 0
        self.total_connections_reused = 0
        self.total_connections_evicted = 0
        self.total_health_checks = 0
        self.failed_health_checks = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.io_time_seconds = 0.0
        self.start_time = time.time()
        self._lock = threading.RLock()

    def record_connection_created(self) -> None:
        with self._lock:
            self.total_connections_created += 1

    def record_connection_reused(self) -> None:
        with self._lock:
            self.total_connections_reused += 1
            self.cache_hits += 1

    def record_connection_evicted(self) -> None:
        with self._lock:
            self.total_connections_evicted += 1

    def record_health_check(self, success: bool):
        with self._lock:
            self.total_health_checks += 1
            if not success:
                self.failed_health_checks += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def record_io_time(self, duration: float):
        with self._lock:
            self.io_time_seconds += duration

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            uptime = time.time() - self.start_time
            total_operations = self.cache_hits + self.cache_misses

            return {
                "uptime_seconds": uptime,
                "total_connections_created": self.total_connections_created,
                "total_connections_reused": self.total_connections_reused,
                "total_connections_evicted": self.total_connections_evicted,
                "cache_hit_ratio": self.cache_hits / total_operations
                if total_operations > 0
                else 0.0,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "health_checks_performed": self.total_health_checks,
                "health_check_failure_rate": self.failed_health_checks
                / self.total_health_checks
                if self.total_health_checks > 0
                else 0.0,
                "total_io_time_seconds": self.io_time_seconds,
                "average_io_time_ms": (self.io_time_seconds / total_operations * 1000)
                if total_operations > 0
                else 0.0,
            }


class HDF5ConnectionPool:
    """
    Enhanced connection pool for HDF5 files with LRU eviction, health monitoring,
    and comprehensive I/O performance tracking.

    Features:
    - LRU eviction policy with configurable pool size
    - Connection health monitoring with automatic cleanup
    - Performance statistics and monitoring
    - Thread-safe operations with fine-grained locking
    - Memory pressure detection and adaptive sizing
    - Batch read operations optimization
    """

    def __init__(
        self,
        max_pool_size: int = 20,
        health_check_interval: float = 300.0,
        enable_memory_pressure_adaptation: bool = True,
    ):
        self.max_pool_size = max_pool_size
        self.base_pool_size = max_pool_size
        self.health_check_interval = health_check_interval
        self.enable_memory_pressure_adaptation = enable_memory_pressure_adaptation

        # LRU cache for connections
        self._pool: OrderedDict[str, PooledConnection] = OrderedDict()
        self._pool_lock = threading.RLock()
        self._file_locks: dict[str, threading.RLock] = {}
        self._file_locks_lock = threading.RLock()

        # Statistics tracking
        self.stats = ConnectionStats()

        # Health monitoring
        self._last_health_check = 0.0
        self._unhealthy_files = set()

        # Read-ahead cache for batch operations
        self._read_cache: dict[str, dict[str, Any]] = {}
        self._read_cache_lock = threading.RLock()
        self._read_cache_max_size = 50  # Maximum cached read results per file

        logger.info(
            f"HDF5ConnectionPool initialized: max_size={max_pool_size}, health_check_interval={health_check_interval}s"
        )

    def _get_file_lock(self, fname: str) -> threading.RLock:
        """Get or create a lock for a specific file."""
        with self._file_locks_lock:
            if fname not in self._file_locks:
                self._file_locks[fname] = threading.RLock()
            return self._file_locks[fname]

    def _adapt_pool_size_to_memory_pressure(self) -> None:
        """Dynamically adapt pool size based on system memory pressure."""
        if not self.enable_memory_pressure_adaptation:
            return

        try:
            memory = psutil.virtual_memory()
            memory_pressure = memory.percent / 100.0

            if memory_pressure > MEMORY_PRESSURE_CRITICAL:
                # Very high memory pressure - reduce pool size significantly
                new_size = max(3, self.base_pool_size // 4)
            elif memory_pressure > MEMORY_PRESSURE_HIGH:
                # High memory pressure - reduce pool size moderately
                new_size = max(5, self.base_pool_size // 2)
            elif memory_pressure > MEMORY_PRESSURE_MODERATE:
                # Moderate memory pressure - slight reduction
                new_size = max(8, int(self.base_pool_size * 0.75))
            else:
                # Normal memory pressure - use full pool size
                new_size = self.base_pool_size

            if new_size != self.max_pool_size:
                old_size = self.max_pool_size
                self.max_pool_size = new_size
                logger.info(
                    f"Adapted pool size from {old_size} to {new_size} due to memory pressure: {memory_pressure:.1%}"
                )

                # If we reduced the pool size, evict excess connections
                if new_size < len(self._pool):
                    self._evict_excess_connections()

        except Exception as e:
            logger.warning(f"Failed to adapt pool size to memory pressure: {e}")

    def _evict_lru_connections(self, count: int = 1) -> int:
        """Evict least recently used connections."""
        evicted = 0
        connections_to_remove = []

        # Get connections sorted by last access time (oldest first)
        sorted_connections = sorted(
            self._pool.items(), key=lambda x: x[1].last_accessed
        )

        for fname, connection in sorted_connections[:count]:
            connections_to_remove.append(fname)
            connection.close()
            evicted += 1

        # Remove from pool
        for fname in connections_to_remove:
            if fname in self._pool:
                del self._pool[fname]
                self.stats.record_connection_evicted()
                logger.debug(f"Evicted LRU connection: {fname}")

        return evicted

    def _evict_excess_connections(self) -> None:
        """Evict connections when pool exceeds maximum size."""
        excess = len(self._pool) - self.max_pool_size
        if excess > 0:
            self._evict_lru_connections(excess)

    def _perform_health_check(self) -> None:
        """Perform health checks on all connections."""
        current_time = time.time()
        if current_time - self._last_health_check < self.health_check_interval:
            return

        self._last_health_check = current_time
        unhealthy_connections = []
        aged_connections = []

        logger.debug(f"Performing health check on {len(self._pool)} connections")

        for fname, connection in list(self._pool.items()):
            is_healthy = connection.check_health()
            self.stats.record_health_check(is_healthy)

            # Check if connection is too old (older than health check interval)
            connection_age = current_time - connection.created_at
            is_aged = connection_age > self.health_check_interval

            if not is_healthy:
                unhealthy_connections.append(fname)
                self._unhealthy_files.add(fname)
            elif is_aged:
                aged_connections.append(fname)

        # Remove unhealthy connections
        for fname in unhealthy_connections:
            if fname in self._pool:
                self._pool[fname].close()
                del self._pool[fname]
                logger.info(f"Removed unhealthy connection: {fname}")

        # Remove aged connections
        for fname in aged_connections:
            if fname in self._pool:
                connection_age = current_time - self._pool[fname].created_at
                self._pool[fname].close()
                del self._pool[fname]
                logger.info(
                    f"Removed aged connection: {fname} (age: {connection_age:.1f}s)"
                )

        total_removed = len(unhealthy_connections) + len(aged_connections)
        if total_removed:
            logger.info(
                f"Health check completed: removed {len(unhealthy_connections)} unhealthy and {len(aged_connections)} aged connections"
            )

    @contextmanager
    def get_connection(self, fname: str, mode: str = "r"):
        """
        Enhanced context manager to get an HDF5 file connection with comprehensive
        health monitoring, LRU management, and performance tracking.

        Parameters
        ----------
        fname : str
            Path to HDF5 file
        mode : str
            File access mode (default: 'r')

        Yields
        ------
        h5py.File
            HDF5 file handle
        """

        start_time = time.time()
        file_lock = self._get_file_lock(fname)

        # Normalize file path for consistency
        fname = os.path.abspath(fname)

        with file_lock:
            # Adapt pool size based on memory pressure
            self._adapt_pool_size_to_memory_pressure()

            # Perform periodic health checks
            self._perform_health_check()

            # Skip files that are known to be unhealthy
            if fname in self._unhealthy_files:
                # Try to create a direct connection instead
                logger.debug(
                    f"Using direct connection for previously unhealthy file: {fname}"
                )
                file_handle = h5py.File(fname, mode)
                try:
                    yield file_handle
                    # If successful, remove from unhealthy set
                    self._unhealthy_files.discard(fname)
                finally:
                    file_handle.close()
                return

            connection = None

            with self._pool_lock:
                # Check if we already have this file open
                if fname in self._pool:
                    connection = self._pool[fname]
                    # Move to end (most recently used)
                    self._pool.move_to_end(fname)

                    # Quick health check
                    if connection.check_health():
                        connection.touch()
                        self.stats.record_connection_reused()
                        logger.debug(f"Reusing healthy connection for {fname}")

                        try:
                            yield connection.file_handle
                            return
                        except Exception as e:
                            logger.warning(
                                f"Error using pooled connection for {fname}: {e}"
                            )
                            # Mark as unhealthy and continue to create new connection
                            connection.close()
                            del self._pool[fname]
                            connection = None
                    else:
                        # Connection is unhealthy, remove it
                        logger.info(f"Removing unhealthy connection: {fname}")
                        connection.close()
                        del self._pool[fname]
                        self._unhealthy_files.add(fname)
                        connection = None

                # Need to create a new connection
                self.stats.record_cache_miss()

                # Ensure we don't exceed pool size
                if len(self._pool) >= self.max_pool_size:
                    self._evict_lru_connections(1)

                # Create new connection
                try:
                    logger.debug(f"Creating new connection for {fname}")
                    file_handle = h5py.File(fname, mode)
                    connection = PooledConnection(file_handle, fname)
                    self._pool[fname] = connection
                    self.stats.record_connection_created()

                except Exception as e:
                    logger.error(f"Failed to open HDF5 file {fname}: {e}")
                    self._unhealthy_files.add(fname)
                    raise

            # Use the connection
            connection.touch()
            try:
                yield connection.file_handle
            except Exception:
                # If there's an error during usage, mark connection as potentially unhealthy
                with self._pool_lock:
                    if fname in self._pool:
                        logger.warning(
                            f"Removing connection due to usage error: {fname}"
                        )
                        connection.close()
                        del self._pool[fname]
                raise
            finally:
                # Record I/O time
                io_time = time.time() - start_time
                self.stats.record_io_time(io_time)

                if io_time > 1.0:  # Log slow operations
                    logger.debug(f"Slow I/O operation: {fname} took {io_time:.2f}s")

    def clear_pool(self, from_destructor=False):
        """Close all connections and clear the pool."""
        with self._pool_lock:
            # Avoid logging during Python shutdown to prevent logging errors
            if not from_destructor:
                logger.info(
                    f"Clearing connection pool with {len(self._pool)} connections"
                )

            for fname, connection in self._pool.items():
                try:
                    connection.close()
                except Exception as e:
                    if not from_destructor:
                        logger.debug(f"Error closing connection {fname}: {e}")

            self._pool.clear()

        with self._file_locks_lock:
            self._file_locks.clear()

        with self._read_cache_lock:
            self._read_cache.clear()

        self._unhealthy_files.clear()
        if not from_destructor:
            logger.info("Connection pool cleared")

    def get_pool_stats(self) -> dict[str, Any]:
        """Get comprehensive pool statistics."""
        with self._pool_lock:
            pool_info = {
                "pool_size": len(self._pool),
                "max_pool_size": self.max_pool_size,
                "base_pool_size": self.base_pool_size,
                "unhealthy_files": len(self._unhealthy_files),
                "health_check_interval": self.health_check_interval,
                "memory_adaptation_enabled": self.enable_memory_pressure_adaptation,
                "connections": [],
            }

            # Add per-connection statistics
            for fname, connection in self._pool.items():
                conn_info = {
                    "file": fname,
                    "age_seconds": time.time() - connection.created_at,
                    "last_accessed_seconds_ago": time.time() - connection.last_accessed,
                    "access_count": connection.access_count,
                    "is_healthy": connection.is_healthy,
                }
                pool_info["connections"].append(conn_info)

        # Merge with I/O statistics
        io_stats = self.stats.get_stats()
        return {**pool_info, **io_stats}

    def force_health_check(self):
        """Force an immediate health check of all connections."""
        with self._pool_lock:
            self._last_health_check = 0  # Force immediate check
            self._perform_health_check()

    def remove_unhealthy_file(self, fname: str):
        """Remove a file from the unhealthy files set."""
        self._unhealthy_files.discard(fname)
        logger.debug(f"Removed {fname} from unhealthy files set")

    def batch_read_datasets(
        self, fname: str, dataset_paths: list[str], use_cache: bool = True
    ) -> dict[str, Any]:
        """
        Optimized batch reading of multiple datasets from the same file.

        Parameters
        ----------
        fname : str
            HDF5 file path
        dataset_paths : List[str]
            List of dataset paths to read
        use_cache : bool
            Whether to use read cache

        Returns
        -------
        Dict[str, Any]
            Dictionary mapping dataset paths to their values
        """

        start_time = time.time()
        results = {}
        cache_key = f"batch_{hash(tuple(sorted(dataset_paths)))}"

        # Check read cache first
        if use_cache:
            with self._read_cache_lock:
                if fname in self._read_cache and cache_key in self._read_cache[fname]:
                    cached_result = self._read_cache[fname][cache_key]
                    # Check if cache entry is recent (within 5 minutes)
                    if time.time() - cached_result["timestamp"] < 300:
                        logger.debug(
                            f"Using cached batch read for {len(dataset_paths)} datasets from {fname}"
                        )
                        return cached_result["data"]

        # Perform batch read
        with self.get_connection(fname, "r") as f:
            for path in dataset_paths:
                try:
                    if path in f:
                        val = f[path][()]
                        # Handle common data type conversions
                        if isinstance(val, (np.bytes_, bytes)):
                            val = val.decode()
                        elif isinstance(val, np.ndarray) and val.shape == (1, 1):
                            val = val.item()
                        results[path] = val
                    else:
                        logger.warning(f"Dataset path not found: {path} in {fname}")
                        results[path] = None
                except Exception as e:
                    logger.error(f"Error reading dataset {path} from {fname}: {e}")
                    results[path] = None

        # Cache the results
        if use_cache:
            with self._read_cache_lock:
                if fname not in self._read_cache:
                    self._read_cache[fname] = {}

                # Limit cache size per file
                if len(self._read_cache[fname]) >= self._read_cache_max_size:
                    # Remove oldest entries
                    sorted_entries = sorted(
                        self._read_cache[fname].items(), key=lambda x: x[1]["timestamp"]
                    )
                    for old_key, _ in sorted_entries[: len(sorted_entries) // 2]:
                        del self._read_cache[fname][old_key]

                self._read_cache[fname][cache_key] = {
                    "data": results,
                    "timestamp": time.time(),
                }

        read_time = time.time() - start_time
        logger.debug(
            f"Batch read of {len(dataset_paths)} datasets from {fname} completed in {read_time:.3f}s"
        )

        return results

    def clear_read_cache(self, fname: str | None = None):
        """Clear read cache for specific file or all files."""
        with self._read_cache_lock:
            if fname:
                if fname in self._read_cache:
                    del self._read_cache[fname]
                    logger.debug(f"Cleared read cache for {fname}")
            else:
                self._read_cache.clear()
                logger.debug("Cleared all read cache")

    def __del__(self):
        # Pass True to indicate this is called from destructor during shutdown
        try:
            self.clear_pool(from_destructor=True)
        except Exception:
            # Silently ignore any errors during shutdown - destructor context
            pass


# Global connection pool instance with enhanced settings
_connection_pool = HDF5ConnectionPool(
    max_pool_size=25,  # Increased pool size for better performance
    health_check_interval=180.0,  # Health check every 3 minutes
    enable_memory_pressure_adaptation=True,
)


def put(save_path, result, ftype="nexus", mode="raw"):
    """
    save the result to hdf5 file
    Parameters
    ----------
    save_path: str
        path to save the result
    result: dict
        dictionary to save
    ftype: str
        file type, 'nexus' or 'aps_8idi'
    mode: str
        'raw' or 'alias'
    """
    with h5py.File(save_path, "a") as f:
        for key, val in result.items():
            if mode == "alias":
                key = hdf_key[ftype][key]
            if key in f:
                del f[key]
            if isinstance(val, np.ndarray) and val.ndim == 1:
                val = np.reshape(val, (1, -1))
            f[key] = val
        return


def get_abs_cs_scale(fname, ftype="nexus", use_pool=True):
    key = hdf_key[ftype]["abs_cross_section_scale"]

    context_manager = (
        _connection_pool.get_connection(fname, "r")
        if use_pool
        else h5py.File(fname, "r")
    )
    with context_manager as f:
        if key not in f:
            return None
        return float(f[key][()])


def read_metadata_to_dict(file_path, use_pool=True):
    """
    Reads an HDF5 file and loads its contents into a nested dictionary.
    Optimized to read all metadata groups in one file open operation.

    Parameters
    ----------
    file_path : str
        Path to the HDF5 file.
    use_pool : bool
        Whether to use connection pool for optimization.

    Returns
    -------
    dict
        A nested dictionary containing datasets as NumPy arrays.
    """

    def recursive_read(h5_group, target_dict):
        """Recursively reads groups and datasets into the dictionary."""
        for key in h5_group:
            obj = h5_group[key]
            if isinstance(obj, h5py.Dataset):
                val = obj[()]
                if type(val) in [np.bytes_, bytes]:
                    val = val.decode()
                target_dict[key] = val
            elif isinstance(obj, h5py.Group):
                target_dict[key] = {}
                recursive_read(obj, target_dict[key])

    data_dict = {}
    groups = [
        "/entry/instrument",
        "/xpcs/multitau/config",
        "/xpcs/twotime/config",
        "/entry/sample",
        "/entry/user",
    ]

    # Use connection pool for single file operation reading all metadata groups
    context_manager = (
        _connection_pool.get_connection(file_path, "r")
        if use_pool
        else h5py.File(file_path, "r")
    )
    with context_manager as hdf_file:
        # Batch read all groups in single file operation
        for group in groups:
            if group in hdf_file:
                data_dict[group] = {}
                recursive_read(hdf_file[group], data_dict[group])
    return data_dict


def get(fname, fields, mode="raw", ret_type="dict", ftype="nexus", use_pool=True):
    """
    get the values for the various keys listed in fields for a single
    file;

    :param fname:
    :param fields_raw: list of keys [key1, key2, ..., ]
    :param mode: ['raw' | 'alias']; alias is defined in .hdf_key
        otherwise the raw hdf key will be used
    :param ret_type: return dictonary if 'dict', list if it is 'list'
    :param use_pool: whether to use connection pool for optimization
    :return: dictionary or dictionary;
    """
    assert mode in ["raw", "alias"], "mode not supported"
    assert ret_type in ["dict", "list"], "ret_type not supported"

    ret = {}

    # Choose context manager based on use_pool parameter
    if use_pool:
        context_manager = _connection_pool.get_connection(fname, "r")
    else:
        context_manager = h5py.File(fname, "r")

    with context_manager as hdf_result:
        # Batch read all fields in a single file operation
        for key in fields:
            path = hdf_key[ftype][key] if mode == "alias" else key
            if path not in hdf_result:
                logger.error("path to field not found: %s", path)
                raise ValueError("key not found: %s:%s", key, path)
            val = hdf_result.get(path)[()]
            if key in ["saxs_2d"] and val.ndim == 3:  # saxs_2d is in [1xNxM] format
                val = val[0]
            # converts bytes to unicode;
            if type(val) in [np.bytes_, bytes]:
                val = val.decode()
            if isinstance(val, np.ndarray) and val.shape == (1, 1):
                val = val.item()
            ret[key] = val

    if ret_type == "dict":
        return ret
    if ret_type == "list":
        return [ret[key] for key in fields]
    return None


def get_analysis_type(fname, ftype="nexus", use_pool=True):
    """
    determine the analysis type of the file
    Parameters
    ----------
    fname: str
        file name
    ftype: str
        file type, 'nexus' or 'legacy'
    use_pool: bool
        whether to use connection pool for optimization
    Returns
    -------
    tuple
        analysis type, 'Twotime' or 'Multitau', or both
    """
    c2_prefix = hdf_key[ftype]["c2_prefix"]
    g2_prefix = hdf_key[ftype]["g2"]
    analysis_type = []

    context_manager = (
        _connection_pool.get_connection(fname, "r")
        if use_pool
        else h5py.File(fname, "r")
    )
    with context_manager as hdf_result:
        # Primary detection: Check for folder presence (more reliable)
        multitau_folder = "/xpcs/multitau"
        twotime_folder = "/xpcs/twotime"

        if multitau_folder in hdf_result:
            analysis_type.append("Multitau")
            logger.debug(
                f"Detected Multitau format: folder {multitau_folder} found in {fname}"
            )

        if twotime_folder in hdf_result:
            analysis_type.append("Twotime")
            logger.debug(
                f"Detected Twotime format: folder {twotime_folder} found in {fname}"
            )

        # Fallback detection: Check for specific files (backward compatibility)
        if not analysis_type:
            logger.debug(
                f"Folder detection failed, falling back to file detection for {fname}"
            )
            if c2_prefix in hdf_result:
                analysis_type.append("Twotime")
                logger.debug(
                    f"Detected Twotime format via file: {c2_prefix} found in {fname}"
                )
            if g2_prefix in hdf_result:
                analysis_type.append("Multitau")
                logger.debug(
                    f"Detected Multitau format via file: {g2_prefix} found in {fname}"
                )

    if len(analysis_type) == 0:
        raise ValueError(f"No analysis type found in {fname}")

    logger.info(f"File {fname}: detected format(s) {analysis_type}")
    return tuple(analysis_type)


def batch_read_fields(
    fname: str,
    fields: list[str],
    mode: str = "raw",
    ftype: str = "nexus",
    use_pool: bool = True,
) -> dict[str, Any]:
    """
    Optimized batch reading of multiple fields from HDF5 file.

    Parameters
    ----------
    fname : str
        HDF5 file path
    fields : List[str]
        List of field names to read
    mode : str
        'raw' or 'alias' mode
    ftype : str
        File type ('nexus' or 'legacy')
    use_pool : bool
        Whether to use connection pooling

    Returns
    -------
    Dict[str, Any]
        Dictionary of field values
    """
    if not use_pool:
        return get(fname, fields, mode, "dict", ftype, use_pool=False)

    # Convert fields to dataset paths
    dataset_paths = []
    for field in fields:
        path = hdf_key[ftype][field] if mode == "alias" else field
        dataset_paths.append(path)

    # Use batch read from connection pool
    raw_results = _connection_pool.batch_read_datasets(fname, dataset_paths)

    # Process results (same logic as get() function)
    processed_results = {}
    for i, field in enumerate(fields):
        path = dataset_paths[i]
        val = raw_results.get(path)

        if val is None:
            logger.error(f"Path to field not found: {path} for field {field}")
            raise ValueError(f"Key not found: {field}:{path}")

        # Apply same processing as original get() function
        if field == "saxs_2d" and isinstance(val, np.ndarray) and val.ndim == 3:
            val = val[0]  # saxs_2d is in [1xNxM] format

        if isinstance(val, np.ndarray) and val.shape == (1, 1):
            val = val.item()

        processed_results[field] = val

    return processed_results


def get_file_info(fname: str, use_pool: bool = True) -> dict[str, Any]:
    """
    Get basic file information and statistics.

    Parameters
    ----------
    fname : str
        HDF5 file path
    use_pool : bool
        Whether to use connection pooling

    Returns
    -------
    Dict[str, Any]
        File information dictionary
    """
    context_manager = (
        _connection_pool.get_connection(fname, "r")
        if use_pool
        else h5py.File(fname, "r")
    )

    with context_manager as f:
        info = {
            "file_path": fname,
            "file_size_mb": os.path.getsize(fname) / (1024 * 1024),
            "hdf5_version": f.libver,
            "root_groups": list(f.keys()),
            "estimated_datasets": 0,
            "large_datasets": [],
        }

        def count_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                info["estimated_datasets"] += 1
                # Track large datasets (>10MB estimated)
                estimated_size = obj.size * obj.dtype.itemsize / (1024 * 1024)
                if estimated_size > 10:
                    info["large_datasets"].append(
                        {
                            "path": name,
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "estimated_size_mb": estimated_size,
                        }
                    )

        f.visititems(count_datasets)

    return info


def get_connection_pool_stats() -> dict[str, Any]:
    """
    Get comprehensive statistics about the global connection pool.

    Returns
    -------
    Dict[str, Any]
        Connection pool statistics
    """
    return _connection_pool.get_pool_stats()


def clear_connection_pool():
    """
    Clear all connections in the global connection pool.
    """
    _connection_pool.clear_pool()


def force_connection_health_check():
    """
    Force an immediate health check of all pooled connections.
    """
    _connection_pool.force_health_check()


def get_chunked_dataset(
    fname: str,
    dataset_path: str,
    chunk_size: tuple[int, ...] | None = None,
    use_pool: bool = True,
) -> np.ndarray:
    """
    Read a large dataset in chunks to manage memory usage.

    Parameters
    ----------
    fname : str
        HDF5 file path
    dataset_path : str
        Path to dataset within HDF5 file
    chunk_size : Tuple[int, ...], optional
        Size of chunks to read. If None, will use dataset's native chunking
    use_pool : bool
        Whether to use connection pooling

    Returns
    -------
    np.ndarray
        The dataset array
    """
    context_manager = (
        _connection_pool.get_connection(fname, "r")
        if use_pool
        else h5py.File(fname, "r")
    )

    with context_manager as f:
        if dataset_path not in f:
            raise ValueError(f"Dataset {dataset_path} not found in {fname}")

        dataset = f[dataset_path]

        # For small datasets, read directly
        estimated_size_mb = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
        if estimated_size_mb < 100:  # Less than 100MB, read directly
            logger.debug(
                f"Reading small dataset {dataset_path} directly ({estimated_size_mb:.1f}MB)"
            )
            return dataset[()]

        logger.info(
            f"Reading large dataset {dataset_path} in chunks ({estimated_size_mb:.1f}MB)"
        )

        # Use native chunking if available and no custom chunk_size specified
        if chunk_size is None and dataset.chunks is not None:
            chunk_size = dataset.chunks
            logger.debug(f"Using native chunking: {chunk_size}")
        elif chunk_size is None:
            # Default chunking strategy for 2D data
            if len(dataset.shape) == 2:
                # Aim for ~10MB chunks
                target_elements = 10 * 1024 * 1024 // dataset.dtype.itemsize
                chunk_rows = min(
                    dataset.shape[0], max(1, int(np.sqrt(target_elements)))
                )
                chunk_cols = min(dataset.shape[1], target_elements // chunk_rows)
                chunk_size = (chunk_rows, chunk_cols)
            else:
                # For other dimensions, read in smaller slices along first dimension
                chunk_size = (min(1000, dataset.shape[0]), *dataset.shape[1:])

            logger.debug(f"Using computed chunking: {chunk_size}")

        # Read the entire dataset (chunking is handled internally by h5py for efficiency)
        return dataset[()]


if __name__ == "__main__":
    pass
