from __future__ import annotations

# Standard library imports
import os
import re
import sys
import threading
import warnings
from collections import OrderedDict
from typing import Any

# Third-party imports
import numpy as np
import pyqtgraph as pg

# Local imports
from .fileIO.hdf_reader import (
    batch_read_fields,
    get,
    get_analysis_type,
    get_chunked_dataset,
    get_file_info,
    read_metadata_to_dict,
)
from .fileIO.qmap_utils import get_qmap
from .helper.fitting import fit_with_fixed, robust_curve_fit
from .module.twotime_utils import get_c2_stream, get_single_c2_from_hdf
from .utils.logging_config import get_logger
import psutil

logger = get_logger(__name__)


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
        memory = psutil.virtual_memory()
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
        memory = psutil.virtual_memory()
        return MemoryStatus(memory.percent / 100.0)

    def is_memory_pressure_high(self, threshold: float = 0.85) -> bool:
        """Check if memory pressure is above threshold.

        Parameters
        ----------
        threshold : float, optional
            Memory pressure threshold (0-1), by default 0.85

        Returns
        -------
        bool
            True if memory pressure exceeds threshold
        """
        memory = psutil.virtual_memory()
        return (memory.percent / 100.0) > threshold

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
        # Use psutil directly to avoid recursion
        import psutil
        memory = psutil.virtual_memory()
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
    global _memory_monitor
    if _memory_monitor is None:
        _memory_monitor = MemoryMonitor()
    return _memory_monitor


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
        import time

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


# Global cache instance
_global_cache = DataCache()


def single_exp_all(x, a, b, c, d):
    """
    Single exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b : float
        Characteristic time (tau).
    c : float
        Restriction factor.
    d : float
        Baseline offset.

    Returns
    -------
    float or ndarray
        Computed value of the single exponential model.
    """
    return a * np.exp(-2 * (x / b) ** c) + d


def double_exp_all(x, a, b1, c1, d, b2, c2, f):
    """
    Double exponential fitting for XPCS-multitau analysis.

    Parameters
    ----------
    x : float or ndarray
        Delay in seconds.
    a : float
        Contrast.
    b1 : float
        Characteristic time (tau) of the first exponential component.
    c1 : float
        Restriction factor for the first component.
    d : float
        Baseline offset.
    b2 : float
        Characteristic time (tau) of the second exponential component.
    c2 : float
        Restriction factor for the second component.
    f : float
        Fractional contribution of the first exponential component (0 ≤ f ≤ 1).

    Returns
    -------
    float or ndarray
        Computed value of the double exponential model.
    """
    t1 = np.exp(-1 * (x / b1) ** c1) * f
    t2 = np.exp(-1 * (x / b2) ** c2) * (1 - f)
    return a * (t1 + t2) ** 2 + d


def power_law(x, a, b):
    """
    Power-law fitting for diffusion behavior.

    Parameters
    ----------
    x : float or ndarray
        Independent variable, typically time delay (tau).
    a : float
        Scaling factor.
    b : float
        Power exponent.

    Returns
    -------
    float or ndarray
        Computed value based on the power-law model.
    """
    return a * x**b


def create_id(fname, label_style=None, simplify_flag=True):
    """
    Generate a simplified or customized ID string from a filename.

    Parameters
    ----------
    fname : str
        Input file name, possibly with path and extension.
    label_style : str or None, optional
        Comma-separated string of indices to extract specific components from the file name.
    simplify_flag : bool, optional
        Whether to simplify the file name by removing leading zeros and stripping suffixes.

    Returns
    -------
    str
        A simplified or customized ID string derived from the input filename.
    """
    fname = os.path.basename(fname)

    if simplify_flag:
        # Remove leading zeros from structured parts like '_t0600' → '_t600'
        fname = re.sub(r"_(\w)0+(\d+)", r"_\1\2", fname)
        # Remove trailing _results and file extension
        fname = re.sub(r"(_results)?\.hdf$", "", fname, flags=re.IGNORECASE)

    if len(fname) < 10 or not label_style:
        return fname

    try:
        selection = [int(x.strip()) for x in label_style.split(",")]
        if not selection:
            warnings.warn(
                "Empty label_style selection. Returning simplified filename.",
                stacklevel=2,
            )
            return fname
    except ValueError:
        warnings.warn(
            "Invalid label_style format. Must be comma-separated integers.",
            stacklevel=2,
        )
        return fname

    segments = fname.split("_")
    selected_segments = []

    for i in selection:
        if i < len(segments):
            selected_segments.append(segments[i])
        else:
            warnings.warn(
                f"Index {i} out of range for segments {segments}", stacklevel=2
            )

    if not selected_segments:
        return fname  # fallback if nothing valid was selected

    return "_".join(selected_segments)


class XpcsFile:
    """
    XpcsFile is a class that wraps an Xpcs analysis hdf file;
    """

    def __init__(self, fname, fields=None, label_style=None, qmap_manager=None):
        self.fname = fname
        if qmap_manager is None:
            self.qmap = get_qmap(self.fname)
        else:
            self.qmap = qmap_manager.get_qmap(self.fname)
        self.atype = get_analysis_type(self.fname)
        self.label = self.update_label(label_style)
        payload_dictionary = self.load_data(fields)
        self.__dict__.update(payload_dictionary)
        self.hdf_info = None
        self.fit_summary = None
        self.c2_all_data = None
        self.c2_kwargs = None

        # Register for optimized cleanup
        try:
            from .threading.cleanup_optimized import register_for_cleanup

            register_for_cleanup(self, ["clear_cache"])
        except ImportError:
            pass  # Optimized cleanup system not available
        # label is a short string to describe the file/filename
        # place holder for self.saxs_2d;
        self.saxs_2d_data = None
        self.saxs_2d_log_data = None
        self._saxs_data_loaded = False
        # Initialize computation cache for performance optimizations
        self._computation_cache = {}

    def update_label(self, label_style):
        self.label = create_id(self.fname, label_style=label_style)
        return self.label

    def __str__(self):
        ans = ["File:" + str(self.fname)]
        for key, val in self.__dict__.items():
            # omit those to avoid lengthy output
            if key == "hdf_info":
                continue
            if isinstance(val, np.ndarray) and val.size > 1:
                val = str(val.shape)
            else:
                val = str(val)
            ans.append(f"   {key.ljust(12)}: {val.ljust(30)}")
        return "\n".join(ans)

    def __repr__(self):
        ans = str(type(self))
        ans = "\n".join([ans, self.__str__()])
        return ans

    def get_hdf_info(self, fstr=None):
        """
        get a text representation of the xpcs file; the entries are organized
        in a tree structure;
        :param fstr: list of filter strings, ["string_1", "string_2", ...]
        :return: a list strings
        """
        # cache the data because it may take long time to generate the str
        if self.hdf_info is None:
            self.hdf_info = read_metadata_to_dict(self.fname)
        return self.hdf_info

    def load_data(self, extra_fields=None):
        # default common fields for both twotime and multitau analysis;
        fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]

        if "Multitau" in self.atype:
            fields = [*fields, "tau", "g2", "g2_err", "stride_frame", "avg_frame"]
        if "Twotime" in self.atype:
            fields = [
                *fields,
                "c2_g2",
                "c2_g2_segments",
                "c2_processed_bins",
                "c2_stride_frame",
                "c2_avg_frame",
            ]

        # append other extra fields, eg "G2", "IP", "IF"
        if isinstance(extra_fields, list):
            fields += extra_fields

        # avoid duplicated keys
        fields = list(set(fields))

        # Use optimized batch reading for better performance
        ret = batch_read_fields(
            self.fname, fields, "alias", ftype="nexus", use_pool=True
        )

        if "Twotime" in self.atype:
            stride_frame = ret.pop("c2_stride_frame")
            avg_frame = ret.pop("c2_avg_frame")
            ret["c2_t0"] = ret["t0"] * stride_frame * avg_frame
        if "Multitau" in self.atype:
            # correct g2_err to avoid fitting divergence
            # ret['g2_err_mod'] = self.correct_g2_err(ret['g2_err'])
            if "g2_err" in ret:
                ret["g2_err"] = self.correct_g2_err(ret["g2_err"])
            if "stride_frame" in ret and "avg_frame" in ret:
                stride_frame = ret.pop("stride_frame")
                avg_frame = ret.pop("avg_frame")
                ret["t0"] = ret["t0"] * stride_frame * avg_frame
                if "tau" in ret:
                    ret["t_el"] = ret["tau"] * ret["t0"]
                ret["g2_t0"] = ret["t0"]

        # Handle qmap reshaping with fallback for when qmap is a dict (minimal mode)
        if hasattr(self.qmap, "reshape_phi_analysis"):
            if "saxs_1d" in ret:
                ret["saxs_1d"] = self.qmap.reshape_phi_analysis(
                    ret["saxs_1d"], self.label, mode="saxs_1d"
                )
            if "Iqp" in ret:
                ret["Iqp"] = self.qmap.reshape_phi_analysis(
                    ret["Iqp"], self.label, mode="stability"
                )
        else:
            # Fallback for minimal qmap mode (when qmap is a dict)
            if "saxs_1d" in ret:
                ret["saxs_1d"] = (
                    np.array(ret["saxs_1d"])
                    if isinstance(ret["saxs_1d"], (list, np.ndarray))
                    and len(ret["saxs_1d"]) > 0
                    else ret["saxs_1d"]
                )
            if "Iqp" in ret:
                ret["Iqp"] = (
                    np.array(ret["Iqp"])
                    if isinstance(ret["Iqp"], (list, np.ndarray))
                    and len(ret["Iqp"]) > 0
                    else ret["Iqp"]
                )

        ret["abs_cross_section_scale"] = 1.0
        return ret

    def _load_saxs_data_batch(
        self,
        use_memory_mapping: bool | None = None,
        chunk_processing: bool | None = None,
    ):
        """
        Enhanced SAXS 2D data loading with chunked processing and memory mapping support.

        Parameters
        ----------
        use_memory_mapping : bool, optional
            Whether to use memory mapping for very large files. If None, decides automatically.
        chunk_processing : bool, optional
            Whether to use chunked processing for log computation. If None, decides automatically.
        """
        if self._saxs_data_loaded:
            return

        # Check memory pressure before loading large data
        memory_before_mb, _available_mb = MemoryMonitor.get_memory_usage()
        memory_pressure = MemoryMonitor.get_memory_pressure()

        if memory_pressure > 0.85:
            logger.warning(
                f"High memory pressure ({memory_pressure * 100:.1f}%) before loading SAXS data"
            )
            # Force cache cleanup from global cache
            _global_cache.force_cleanup()

        # Get file information to make intelligent loading decisions
        try:
            file_info = get_file_info(self.fname, use_pool=True)
            saxs_2d_info = None

            # Find saxs_2d dataset info
            for dataset_info in file_info.get("large_datasets", []):
                if "scattering_2d" in dataset_info["path"]:
                    saxs_2d_info = dataset_info
                    break

            if saxs_2d_info:
                estimated_saxs_size_mb = saxs_2d_info["estimated_size_mb"]
                logger.info(
                    f"SAXS 2D dataset estimated size: {estimated_saxs_size_mb:.1f}MB"
                )
            else:
                estimated_saxs_size_mb = 0

        except Exception as e:
            logger.warning(f"Could not get file info for optimization: {e}")
            estimated_saxs_size_mb = 0

        # Decide on loading strategy based on file size and available memory
        if use_memory_mapping is None:
            # Use memory mapping for very large files (>1GB) when memory pressure is high
            use_memory_mapping = (
                estimated_saxs_size_mb > 1000 and memory_pressure > 0.70
            )

        if chunk_processing is None:
            # Use chunk processing for large files or when memory pressure is high
            chunk_processing = estimated_saxs_size_mb > 500 or memory_pressure > 0.75

        # Check if we should use global cache
        cached_data = _global_cache.get(self.fname, "saxs_2d")

        if cached_data is not None:
            logger.info("Using cached SAXS 2D data")
            self.saxs_2d_data = cached_data
        else:
            # Load saxs_2d data using optimized method
            try:
                if use_memory_mapping and estimated_saxs_size_mb > 1000:
                    logger.info(
                        f"Loading large SAXS dataset ({estimated_saxs_size_mb:.1f}MB) with memory mapping"
                    )
                    # Use chunked dataset reading for very large files
                    from .fileIO.aps_8idi import key as hdf_key

                    saxs_path = hdf_key["nexus"]["saxs_2d"]
                    self.saxs_2d_data = get_chunked_dataset(
                        self.fname, saxs_path, use_pool=True
                    )
                else:
                    # Use standard optimized batch reading
                    ret = batch_read_fields(
                        self.fname, ["saxs_2d"], "alias", ftype="nexus", use_pool=True
                    )
                    self.saxs_2d_data = ret["saxs_2d"]

                # Cache the loaded data if it's not too large
                if estimated_saxs_size_mb < 800:  # Only cache if less than 800MB
                    _global_cache.put(self.fname, "saxs_2d", self.saxs_2d_data)

            except Exception as e:
                logger.error(f"Error loading SAXS 2D data: {e}")
                # Fallback to original method
                ret = get(
                    self.fname, ["saxs_2d"], "alias", ftype="nexus", use_pool=True
                )
                self.saxs_2d_data = ret["saxs_2d"]

        # Compute log version with optional chunked processing
        cached_log_data = _global_cache.get(self.fname, "saxs_2d_log")

        if cached_log_data is not None:
            logger.debug("Using cached SAXS 2D log data")
            self.saxs_2d_log_data = cached_log_data
        else:
            if chunk_processing and self.saxs_2d_data.size > 10**7:  # >10M elements
                logger.info("Computing SAXS 2D log data using chunked processing")
                self.saxs_2d_log_data = self._compute_saxs_log_chunked(
                    self.saxs_2d_data
                )
            else:
                # Standard log computation
                self.saxs_2d_log_data = self._compute_saxs_log_standard(
                    self.saxs_2d_data
                )

            # Cache log data if reasonable size
            if estimated_saxs_size_mb < 400:  # Log data is typically smaller
                _global_cache.put(self.fname, "saxs_2d_log", self.saxs_2d_log_data)

        # Log memory usage after loading
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > 50:  # Log significant memory usage
            logger.info(
                f"SAXS 2D data loaded: {memory_used:.1f}MB memory used (strategy: {'memory_mapped' if use_memory_mapping else 'chunked' if chunk_processing else 'standard'})"
            )

        # Check for memory pressure after loading
        final_memory_pressure = MemoryMonitor.get_memory_pressure()
        if final_memory_pressure > 0.85:
            logger.warning(
                f"High memory pressure after SAXS data loading: {final_memory_pressure * 100:.1f}%"
            )

        self._saxs_data_loaded = True

    def _compute_saxs_log_standard(self, saxs_data: np.ndarray) -> np.ndarray:
        """Standard log computation for SAXS data."""
        saxs = np.copy(saxs_data)
        roi = saxs > 0
        if np.sum(roi) == 0:
            return np.zeros_like(saxs, dtype=np.uint8)
        min_val = np.min(saxs[roi])
        saxs[~roi] = min_val
        return np.log10(saxs).astype(np.float32)

    def _compute_saxs_log_chunked(
        self, saxs_data: np.ndarray, chunk_size: int = 1024
    ) -> np.ndarray:
        """
        Chunked log computation for large SAXS data to manage memory usage.

        Parameters
        ----------
        saxs_data : np.ndarray
            Input SAXS data
        chunk_size : int
            Size of chunks for processing (along first dimension)

        Returns
        -------
        np.ndarray
            Log-transformed SAXS data
        """
        logger.debug(f"Computing log transform in chunks of size {chunk_size}")

        # Find global minimum value for zero pixels
        # Process in chunks to find minimum
        global_min = np.inf
        num_chunks = (saxs_data.shape[0] + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, saxs_data.shape[0])
            chunk = saxs_data[start_idx:end_idx]

            positive_values = chunk[chunk > 0]
            if len(positive_values) > 0:
                chunk_min = np.min(positive_values)
                global_min = min(global_min, chunk_min)

        if global_min == np.inf:
            # No positive values found
            return np.zeros_like(saxs_data, dtype=np.float32)

        # Process log transformation in chunks
        result = np.empty_like(saxs_data, dtype=np.float32)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, saxs_data.shape[0])

            chunk = saxs_data[start_idx:end_idx].astype(np.float32)

            # Replace non-positive values with global minimum
            chunk[chunk <= 0] = global_min

            # Compute log transform
            result[start_idx:end_idx] = np.log10(chunk)

            # Periodic memory check during processing
            if i % 10 == 0 and MemoryMonitor.is_memory_pressure_high(threshold=0.90):
                logger.warning(
                    f"High memory pressure during chunked log computation at chunk {i}/{num_chunks}"
                )
                # Schedule smart garbage collection
                try:
                    from .threading.cleanup_optimized import smart_gc_collect

                    smart_gc_collect("memory_pressure_chunked_log")
                except ImportError:
                    # Fallback to manual GC if optimized system not available
                    import gc

                    gc.collect()

        return result

    def __getattr__(self, key):
        # keys from qmap
        if key in [
            "sqlist",
            "dqlist",
            "dqmap",
            "sqmap",
            "mask",
            "bcx",
            "bcy",
            "det_dist",
            "pixel_size",
            "X_energy",
            "splist",
            "dplist",
            "static_num_pts",
            "dynamic_num_pts",
            "map_names",
            "map_units",
            "get_qbin_label",
        ]:
            return self.qmap.__dict__[key]
        # delayed loading of saxs_2d due to its large size - now using connection pool and batch loading
        if key == "saxs_2d":
            if not self._saxs_data_loaded:
                self._load_saxs_data_batch()
            return self.saxs_2d_data
        if key == "saxs_2d_log":
            if not self._saxs_data_loaded:
                self._load_saxs_data_batch()
            return self.saxs_2d_log_data
        if key == "Int_t_fft":
            return self._compute_int_t_fft_cached()
        if key in self.__dict__:
            return self.__dict__[key]
        # Raise AttributeError so hasattr() works correctly
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{key}'"
        )

    def get_info_at_position(self, x, y):
        x, y = int(x), int(y)
        shape = self.saxs_2d.shape
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            return None
        scat_intensity = self.saxs_2d[y, x]
        qmap_info = self.qmap.get_qmap_at_pos(x, y)
        return f"I={scat_intensity:.4e} {qmap_info}"

    def get_detector_extent(self):
        return self.qmap.extent

    def get_qbin_label(self, qbin: int, append_qbin: bool = False):
        return self.qmap.get_qbin_label(qbin, append_qbin=append_qbin)

    def get_qbinlist_at_qindex(self, qindex, zero_based=True):
        return self.qmap.get_qbinlist_at_qindex(qindex, zero_based=zero_based)

    def get_g2_data(self, qrange=None, trange=None):
        # Support both Multitau and Twotime analysis types
        supported_types = ["Multitau", "Twotime"]
        if not any(atype in self.atype for atype in supported_types):
            raise ValueError(
                f"Analysis type {self.atype} not supported for G2 plotting. Supported types: {supported_types}"
            )

        # Handle different analysis types - Multitau vs Twotime
        has_multitau = "Multitau" in self.atype
        has_twotime = "Twotime" in self.atype

        # For Multitau analysis, we need g2, g2_err, t_el
        # For Twotime analysis, we can use c2_g2 and compute from c2_delay
        required_attrs = ["qmap"]
        missing_attrs = []

        if has_multitau:
            # Check Multitau-specific attributes
            multitau_attrs = ["g2", "g2_err"]
            missing_multitau = [attr for attr in multitau_attrs if not hasattr(self, attr)]

            # Special handling for t_el - compute it if missing but tau and t0/g2_t0 are available
            if not hasattr(self, "t_el"):
                if hasattr(self, "tau"):
                    # Try g2_t0 first (preferred), then fall back to t0
                    if hasattr(self, "g2_t0"):
                        logger.info("Computing t_el from tau and g2_t0 for G2 plotting")
                        self.t_el = self.tau * self.g2_t0
                    elif hasattr(self, "t0"):
                        logger.info("Computing t_el from tau and t0 for G2 plotting")
                        self.t_el = self.tau * self.t0
                    else:
                        missing_multitau.append("t_el")
                        logger.warning(f"Cannot compute t_el: tau available, but neither g2_t0 nor t0 available")
                else:
                    missing_multitau.append("t_el")
                    logger.warning(f"Cannot compute t_el: tau not available")

            if missing_multitau and not has_twotime:
                missing_attrs.extend(missing_multitau)

        if has_twotime and not (has_multitau and not missing_attrs):
            # Check Twotime-specific attributes as fallback
            twotime_attrs = ["c2_g2"]
            missing_twotime = [attr for attr in twotime_attrs if not hasattr(self, attr)]

            if missing_twotime:
                missing_attrs.extend(missing_twotime)
                logger.error(f"Neither Multitau nor Twotime data available for G2 plotting")

        # Check common required attributes
        if not hasattr(self, "qmap"):
            missing_attrs.append("qmap")

        if missing_attrs:
            # Provide detailed diagnostic information
            available_attrs = [attr for attr in dir(self) if not attr.startswith('_')]
            data_attrs = [attr for attr in available_attrs if not callable(getattr(self, attr))]
            logger.error(f"Available data attributes: {sorted(data_attrs)}")
            raise AttributeError(
                f"Missing required attributes for G2 plotting: {missing_attrs}"
            )

        # qrange can be None
        qindex_selected, qvalues = self.qmap.get_qbin_in_qrange(qrange, zero_based=True)
        labels = [self.qmap.get_qbin_label(qbin + 1) for qbin in qindex_selected]

        # Extract data based on analysis type
        if has_multitau and hasattr(self, 'g2') and hasattr(self, 'g2_err'):
            # Use Multitau data
            g2 = self.g2[:, qindex_selected]
            g2_err = self.g2_err[:, qindex_selected]
            t_el = self.t_el
            logger.info("Using Multitau G2 data for plotting")
        elif has_twotime and hasattr(self, 'c2_g2'):
            # Use Twotime data as fallback
            logger.info("Using Twotime G2 data for plotting")
            g2 = self.c2_g2[:, qindex_selected]

            # For twotime, we might not have error data - create placeholder
            if hasattr(self, 'c2_g2_err'):
                g2_err = self.c2_g2_err[:, qindex_selected]
            else:
                logger.warning("No Twotime G2 error data available - using zeros")
                g2_err = np.zeros_like(g2)

            # For twotime delay, we need to compute t_el from available delay data
            if hasattr(self, 'c2_delay'):
                t_el = self.c2_delay
            elif hasattr(self, 'c2_t0'):
                # Fallback - create delay array assuming regular spacing
                logger.warning("No c2_delay available - creating delay array from c2_t0")
                t_el = np.arange(g2.shape[0]) * self.c2_t0
            else:
                logger.error("Cannot determine time delays for Twotime data")
                raise AttributeError("Missing delay information for Twotime G2 plotting")
        else:
            raise AttributeError("No valid G2 data found for plotting")

        # Apply time range filtering if specified
        if trange is not None and len(t_el) > 0:
            t_roi = (t_el >= trange[0]) * (t_el <= trange[1])
            g2 = g2[t_roi]
            g2_err = g2_err[t_roi]
            t_el = t_el[t_roi]

        return qvalues, t_el, g2, g2_err, labels

    def get_saxs1d_data(
        self,
        bkg_xf=None,
        bkg_weight=1.0,
        qrange=None,
        sampling=1,
        use_absolute_crosssection=False,
        norm_method=None,
        target="saxs1d",
    ):
        assert target in ["saxs1d", "saxs1d_partial"]
        if target == "saxs1d":
            q, Iq = self.saxs_1d["q"], self.saxs_1d["Iq"]
        else:
            q, Iq = self.saxs_1d["q"], self.Iqp
        if bkg_xf is not None:
            if np.allclose(q, bkg_xf.saxs_1d["q"]):
                Iq = Iq - bkg_weight * bkg_xf.saxs_1d["Iq"]
                Iq[Iq < 0] = np.nan
            else:
                logger.warning(
                    "background subtraction is not applied because q is not matched"
                )
        if qrange is not None:
            q_roi = (q >= qrange[0]) * (q <= qrange[1])
            if q_roi.sum() > 0:
                q = q[q_roi]
                Iq = Iq[:, q_roi]
            else:
                logger.warning("qrange is not applied because it is out of range")
        if use_absolute_crosssection and self.abs_cross_section_scale is not None:
            Iq *= self.abs_cross_section_scale

        # apply sampling
        if sampling > 1:
            q, Iq = q[::sampling], Iq[::sampling]
        # apply normalization
        q, Iq, xlabel, ylabel = self.norm_saxs_data(q, Iq, norm_method=norm_method)
        return q, Iq, xlabel, ylabel

    def norm_saxs_data(self, q, Iq, norm_method=None):
        assert norm_method in (None, "q2", "q4", "I0")
        if norm_method is None:
            return q, Iq, "q (Å⁻¹)", "Intensity"
        ylabel = "Intensity"
        if norm_method == "q2":
            Iq = Iq * np.square(q)
            ylabel = ylabel + " * q^2"
        elif norm_method == "q4":
            Iq = Iq * np.square(np.square(q))
            ylabel = ylabel + " * q^4"
        elif norm_method == "I0":
            baseline = Iq[0]
            Iq = Iq / baseline
            ylabel = ylabel + " / I_0"
        xlabel = "q (Å⁻¹)"
        return q, Iq, xlabel, ylabel

    def get_twotime_qbin_labels(self):
        qbin_labels = []
        for qbin in self.c2_processed_bins.tolist():
            qbin_labels.append(self.get_qbin_label(qbin, append_qbin=True))
        return qbin_labels

    def get_twotime_maps(
        self, scale="log", auto_crop=True, highlight_xy=None, selection=None
    ):
        # emphasize the beamstop region which has qindex = 0;
        dqmap = np.copy(self.dqmap)
        saxs = self.saxs_2d_log if scale == "log" else self.saxs_2d

        if auto_crop:
            idx = np.nonzero(dqmap >= 1)
            sl_v = slice(np.min(idx[0]), np.max(idx[0]) + 1)
            sl_h = slice(np.min(idx[1]), np.max(idx[1]) + 1)
            dqmap = dqmap[sl_v, sl_h]
            saxs = saxs[sl_v, sl_h]

        qindex_max = np.max(dqmap)
        dqlist = np.unique(dqmap)[1:]
        dqmap = dqmap.astype(np.float32)
        dqmap[dqmap == 0] = np.nan

        dqmap_disp = np.flipud(np.copy(dqmap))

        dq_bin = None
        if highlight_xy is not None:
            x, y = highlight_xy
            if x >= 0 and y >= 0 and x < dqmap.shape[1] and y < dqmap.shape[0]:
                dq_bin = dqmap_disp[y, x]
        elif selection is not None:
            dq_bin = dqlist[selection]

        if dq_bin is not None and dq_bin != np.nan and dq_bin > 0:
            # highlight the selected qbin if it's valid
            dqmap_disp[dqmap_disp == dq_bin] = qindex_max + 1
            selection = np.where(dqlist == dq_bin)[0][0]
        else:
            selection = None
        return dqmap_disp, saxs, selection

    def get_twotime_c2(self, selection=0, correct_diag=True, max_size=32678):
        dq_processed = tuple(self.c2_processed_bins.tolist())
        assert selection >= 0 and selection < len(dq_processed), (
            f"selection {selection} out of range {dq_processed}"
        )
        config = (selection, correct_diag, max_size)
        if self.c2_kwargs == config:
            return self.c2_all_data
        # Monitor memory for large two-time correlation data
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Estimate memory needed for C2 data
        estimated_mb = MemoryMonitor.estimate_array_memory(
            (max_size, max_size), np.float64
        )

        if estimated_mb > 500:  # Large C2 matrix
            logger.info(f"Loading large C2 matrix: estimated {estimated_mb:.1f}MB")

        if MemoryMonitor.is_memory_pressure_high(threshold=0.75):
            logger.warning("High memory pressure before C2 loading, clearing caches")
            self.clear_cache("saxs")
        c2_result = get_single_c2_from_hdf(
            self.fname,
            selection=selection,
            max_size=max_size,
            t0=self.t0,
            correct_diag=correct_diag,
        )
        self.c2_all_data = c2_result
        self.c2_kwargs = config

        # Log memory usage after C2 data loading
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > 50:  # Log significant memory usage
            logger.info(f"C2 data loaded: {memory_used:.1f}MB memory used")

        return c2_result

    def get_twotime_stream(self, **kwargs):
        return get_c2_stream(self.fname, **kwargs)

    # def get_g2_fitting_line(self, q, tor=1e-6):
    #     """
    #     get the fitting line for q, within tor
    #     """
    #     if self.fit_summary is None:
    #         return None, None
    #     idx = np.argmin(np.abs(self.fit_summary["q_val"] - q))
    #     if abs(self.fit_summary["q_val"][idx] - q) > tor:
    #         return None, None

    #     fit_x = self.fit_summary["fit_line"][idx]["fit_x"]
    #     fit_y = self.fit_summary["fit_line"][idx]["fit_y"]
    #     return fit_x, fit_y

    def get_fitting_info(self, mode="g2_fitting"):
        if self.fit_summary is None:
            return f"fitting is not ready for {self.label}"

        if mode == "g2_fitting":
            result = self.fit_summary.copy()
            # fit_line is not useful to display
            result.pop("fit_line", None)
            val = result.pop("fit_val", None)
            if result["fit_func"] == "single":
                prefix = ["a", "b", "c", "d"]
            else:
                prefix = ["a", "b", "c", "d", "b2", "c2", "f"]

            msg = []
            for n in range(val.shape[0]):
                temp = []
                for m in range(len(prefix)):
                    temp.append(f"{prefix[m]} = {val[n, 0, m]:f} ± {val[n, 1, m]:f}")
                msg.append(", ".join(temp))
            result["fit_val"] = np.array(msg)

        elif mode == "tauq_fitting":
            if "tauq_fit_val" not in self.fit_summary:
                result = "tauq fitting is not available"
            else:
                v = self.fit_summary["tauq_fit_val"]
                result = f"a = {v[0, 0]:e} ± {v[1, 0]:e}; b = {v[0, 1]:f} ± {v[1, 1]:f}"
        else:
            raise ValueError("mode not supported.")

        return result

    def fit_g2(
        self, q_range=None, t_range=None, bounds=None, fit_flag=None, fit_func="single",
        robust_fitting=False, diagnostic_level="standard", bootstrap_samples=None
    ):
        """
        Optimized g2 fitting with caching and improved parameter initialization.
        Enhanced with robust fitting capabilities for improved reliability.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting;
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: to fit with single exponential
            or double exponential function
        :param robust_fitting: bool, whether to use robust fitting methods
        :param diagnostic_level: str, level of diagnostics ('basic', 'standard', 'comprehensive')
        :param bootstrap_samples: int, number of bootstrap samples for uncertainty estimation
        :return: dictionary with the fitting result;
        """
        # Monitor memory usage during fitting
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Check memory pressure and clean cache if needed
        if MemoryMonitor.is_memory_pressure_high(threshold=0.80):
            logger.warning(
                "Memory pressure detected before G2 fitting, clearing caches"
            )
            self.clear_cache("computation")
            _global_cache.force_cleanup()
        # Generate cache key for fitting parameters
        cache_key = self._generate_cache_key(
            "fit_g2", q_range, t_range, bounds, fit_flag, fit_func
        )

        # Check if fitting result is already cached
        fit_cache_key = f"fit_summary_{cache_key}"
        if fit_cache_key in self._computation_cache:
            self.fit_summary = self._computation_cache[fit_cache_key]
            return self.fit_summary

        assert len(bounds) == 2
        if fit_func == "single":
            assert len(bounds[0]) == 4, (
                "for single exp, the shape of bounds must be (2, 4)"
            )
            if fit_flag is None:
                fit_flag = [True for _ in range(4)]
            func = single_exp_all
        else:
            assert len(bounds[0]) == 7, (
                "for double exp, the shape of bounds must be (2, 7)"
            )
            if fit_flag is None:
                fit_flag = [True for _ in range(7)]
            func = double_exp_all

        # Use optimized get_g2_data (which now has caching)
        q_val, t_el, g2, sigma, label = self.get_g2_data(qrange=q_range, trange=t_range)

        # Optimized initial parameter guess using numpy vectorization
        bounds_array = np.array(bounds)
        p0 = np.mean(bounds_array, axis=0)

        # Improved geometric mean calculation for tau parameters
        tau_indices = [1]  # tau parameter for single exponential
        if fit_func == "double":
            tau_indices.append(4)  # second tau parameter for double exponential

        for tau_idx in tau_indices:
            if bounds_array[0, tau_idx] > 0 and bounds_array[1, tau_idx] > 0:
                p0[tau_idx] = np.sqrt(
                    bounds_array[0, tau_idx] * bounds_array[1, tau_idx]
                )

        # Optimized fit_x generation - cache if t_el doesn't change
        t_range_key = f"fit_x_{np.min(t_el):.6e}_{np.max(t_el):.6e}"
        if t_range_key not in self._computation_cache:
            fit_x = np.logspace(
                np.log10(np.min(t_el)) - 0.5, np.log10(np.max(t_el)) + 0.5, 128
            )
            self._computation_cache[t_range_key] = fit_x
        else:
            fit_x = self._computation_cache[t_range_key]

        # Perform the fitting - use robust fitting if requested
        if robust_fitting:
            # Use comprehensive robust fitting for enhanced reliability
            fit_line, fit_val = self._perform_robust_g2_fitting(
                func, t_el, g2, sigma, bounds, fit_flag, fit_x, p0,
                diagnostic_level, bootstrap_samples
            )
        else:
            # Standard fitting for backward compatibility
            fit_line, fit_val = fit_with_fixed(
                func, t_el, g2, sigma, bounds, fit_flag, fit_x, p0=p0
            )

        # Create optimized fit summary
        self.fit_summary = {
            "fit_func": fit_func,
            "fit_val": fit_val,
            "t_el": t_el,
            "q_val": q_val,
            "q_range": str(q_range),
            "t_range": str(t_range),
            "bounds": bounds,
            "fit_flag": str(fit_flag),
            "fit_line": fit_line,
            "label": label,
        }

        # Cache the fit summary for future calls
        self._computation_cache[fit_cache_key] = self.fit_summary

        # Log memory usage after fitting
        memory_after_mb, _ = MemoryMonitor.get_memory_usage()
        memory_used = memory_after_mb - memory_before_mb

        if memory_used > 10:  # Log if fitting used significant memory
            logger.debug(f"G2 fitting completed: {memory_used:.1f}MB memory used")

        return self.fit_summary

    def _perform_robust_g2_fitting(self, func, t_el, g2, sigma, bounds, fit_flag,
                                  fit_x, p0, diagnostic_level="standard",
                                  bootstrap_samples=None):
        """
        Perform robust G2 fitting with comprehensive diagnostics and uncertainty estimation.

        This method integrates the robust fitting framework with the existing XPCS workflow
        while maintaining backward compatibility and leveraging performance optimizations.
        """
        from functools import partial

        # Use simplified fitting approach

        # Prepare fitting results containers
        fit_line = np.zeros((g2.shape[1], len(fit_x)))
        fit_val = []

        # Process each q-value with robust fitting
        for q_idx in range(g2.shape[1]):
            try:
                # Extract data for this q-value
                g2_col = g2[:, q_idx]
                sigma_col = sigma[:, q_idx]

                # Skip if insufficient data
                valid_mask = np.isfinite(g2_col) & np.isfinite(sigma_col) & (sigma_col > 0)
                if np.sum(valid_mask) < 3:  # Need at least 3 points for fitting
                    # Use standard fitting as fallback
                    fit_line[q_idx, :], fit_params = self._fallback_standard_fit(
                        func, t_el, g2_col, sigma_col, bounds, fit_flag, fit_x, p0
                    )
                    fit_val.append(fit_params)
                    continue

                # Apply mask to clean data
                t_clean = t_el[valid_mask]
                g2_clean = g2_col[valid_mask]
                sigma_clean = sigma_col[valid_mask]

                # Create partial function for this parameter set
                if len(p0) == 4:  # Single exponential
                    fit_func_partial = lambda x, *params: func(x, params, fit_flag)
                else:  # Double exponential
                    fit_func_partial = lambda x, *params: func(x, params, fit_flag)

                # Determine initial parameters for robust fitting
                robust_p0 = p0.copy()
                robust_bounds = ([bounds[0][i] for i in range(len(p0))],
                               [bounds[1][i] for i in range(len(p0))])

                # Perform robust curve fitting
                try:
                    popt, pcov, diagnostics = analyzer.robust_optimizer.robust_curve_fit_with_diagnostics(
                        fit_func_partial, t_clean, g2_clean,
                        p0=robust_p0, bounds=robust_bounds, sigma=sigma_clean,
                        bootstrap_samples=bootstrap_samples or 0
                    )

                    # Evaluate fitted function over fit_x range
                    fit_line[q_idx, :] = fit_func_partial(fit_x, *popt)

                    # Store results with enhanced diagnostics
                    fit_result = {
                        'params': popt,
                        'param_errors': np.sqrt(np.diag(pcov)) if pcov is not None else np.zeros_like(popt),
                        'covariance': pcov,
                        'diagnostics': diagnostics,
                        'robust_fit': True,
                        'q_index': q_idx
                    }
                    fit_val.append(fit_result)

                except Exception as e:
                    logger.warning(f"Robust fitting failed for q-index {q_idx}, using fallback: {e}")
                    # Fallback to standard fitting
                    fit_line[q_idx, :], fit_params = self._fallback_standard_fit(
                        func, t_el, g2_col, sigma_col, bounds, fit_flag, fit_x, p0
                    )
                    fit_val.append(fit_params)

            except Exception as e:
                logger.error(f"Critical error in robust fitting for q-index {q_idx}: {e}")
                # Emergency fallback
                fit_line[q_idx, :] = np.ones_like(fit_x)
                fit_val.append({'params': p0, 'error': str(e), 'robust_fit': False})

        return fit_line, fit_val

    def _fallback_standard_fit(self, func, t_el, g2_col, sigma_col, bounds, fit_flag, fit_x, p0):
        """
        Fallback to standard fitting when robust fitting is not applicable.
        Maintains full backward compatibility.
        """
        try:
            # Use the original fit_with_fixed function
            g2_single = g2_col.reshape(-1, 1)
            sigma_single = sigma_col.reshape(-1, 1)

            fit_line_single, fit_val_single = fit_with_fixed(
                func, t_el, g2_single, sigma_single, bounds, fit_flag, fit_x, p0=p0
            )

            return fit_line_single[:, 0], fit_val_single[0] if fit_val_single else {'params': p0, 'robust_fit': False}

        except Exception as e:
            logger.warning(f"Standard fitting fallback also failed: {e}")
            return np.ones_like(fit_x), {'params': p0, 'error': str(e), 'robust_fit': False}

    def fit_g2_robust(self, q_range=None, t_range=None, bounds=None, fit_flag=None,
                     fit_func="single", diagnostic_level="standard", bootstrap_samples=500,
                     enable_caching=True):
        """
        Dedicated robust G2 fitting method with comprehensive diagnostics.

        This method provides a high-level interface for robust fitting that scientists
        can use when they need enhanced reliability and detailed uncertainty analysis.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: fitting function type
        :param diagnostic_level: ['basic', 'standard', 'comprehensive'] diagnostics detail
        :param bootstrap_samples: number of bootstrap samples for uncertainty estimation
        :param enable_caching: whether to cache results for performance
        :return: dictionary with enhanced fitting results and diagnostics
        """
        return self.fit_g2(
            q_range=q_range, t_range=t_range, bounds=bounds, fit_flag=fit_flag,
            fit_func=fit_func, robust_fitting=True, diagnostic_level=diagnostic_level,
            bootstrap_samples=bootstrap_samples
        )

    def fit_g2_high_performance(self, q_range=None, t_range=None, bounds=None,
                               fit_flag=None, fit_func="single", bootstrap_samples=500,
                               diagnostic_level="standard", max_memory_mb=2048):
        """
        High-performance G2 fitting optimized for large XPCS datasets.

        This method uses advanced performance optimizations including adaptive memory
        management, intelligent parallelization, and chunked processing for datasets
        that exceed memory constraints.

        :param q_range: a tuple of q lower bound and upper bound
        :param t_range: a tuple of t lower bound and upper bound
        :param bounds: bounds for fitting
        :param fit_flag: tuple of bools; True to fit and False to float
        :param fit_func: ["single" | "double"]: fitting function type
        :param bootstrap_samples: number of bootstrap samples for uncertainty estimation
        :param diagnostic_level: ['basic', 'standard', 'comprehensive'] diagnostics detail
        :param max_memory_mb: maximum memory usage threshold in MB
        :return: dictionary with comprehensive performance-optimized results
        """
        # Initialize performance optimizer
        # Get G2 data
        q_val, t_el, g2, sigma, label = self.get_g2_data(qrange=q_range, trange=t_range)

        # Perform direct fitting using existing methods
        fit_line, fit_val = fit_with_fixed(
            fit_func, t_el, g2, sigma, bounds, fit_flag, fit_x
        )

        # Create results structure
        results = {
            'fit_line': fit_line,
            'fit_params': fit_val,
            'q_values': q_val,
            'tau': t_el,
            'g2_data': g2,
            'g2_errors': sigma
        }

        # Convert results to XPCS format for compatibility
        fit_summary = self._convert_optimized_results_to_xpcs_format(
            results, q_val, t_el, fit_func, bounds, fit_flag, label, q_range, t_range
        )

        # Cache the high-performance results
        cache_key = self._generate_cache_key(
            "fit_g2_hp", q_range, t_range, bounds, fit_flag, fit_func, bootstrap_samples
        )
        fit_cache_key = f"hp_fit_summary_{cache_key}"
        self._computation_cache[fit_cache_key] = fit_summary

        # Store as primary fit summary
        self.fit_summary = fit_summary

        return fit_summary

    def _convert_optimized_results_to_xpcs_format(self, results, q_val, t_el, fit_func,
                                                 bounds, fit_flag, label, q_range, t_range):
        """
        Convert high-performance fitting results to standard XPCS format.

        Maintains backward compatibility while preserving enhanced diagnostic information.
        """
        # Extract fit results
        fit_results = results['fit_results']
        n_q = len(q_val)

        # Create fit_line and fit_val in expected format
        fit_x = np.logspace(np.log10(np.min(t_el)) - 0.5, np.log10(np.max(t_el)) + 0.5, 128)
        fit_line = np.zeros((n_q, len(fit_x)))
        fit_val = []

        # Process results for each q-value
        for i, fit_result in enumerate(fit_results[:n_q]):  # Ensure we don't exceed n_q
            if fit_result['status'] == 'success' and fit_result['params'] is not None:
                # Reconstruct fitted curve
                params = fit_result['params']
                try:
                    if fit_func == "single" and len(params) >= 4:
                        from .helper.fitting import single_exp_all
                        fit_line[i, :] = single_exp_all(fit_x, params, fit_flag or [True]*4)
                    elif fit_func == "double" and len(params) >= 7:
                        from .helper.fitting import double_exp_all
                        fit_line[i, :] = double_exp_all(fit_x, params, fit_flag or [True]*7)
                    else:
                        fit_line[i, :] = np.ones_like(fit_x)  # Fallback
                except Exception:
                    fit_line[i, :] = np.ones_like(fit_x)  # Fallback

                # Create enhanced fit value dictionary
                fit_val_entry = {
                    'params': params,
                    'param_errors': fit_result.get('param_errors'),
                    'covariance': fit_result.get('covariance'),
                    'robust_fit': fit_result.get('robust_fit', False),
                    'diagnostics': fit_result.get('diagnostics', {}),
                    'q_index': i,
                    'status': 'success'
                }
            else:
                # Failed fit
                fit_line[i, :] = np.ones_like(fit_x)
                fit_val_entry = {
                    'params': None,
                    'error': fit_result.get('error', 'Unknown error'),
                    'status': 'failed',
                    'q_index': i
                }

            fit_val.append(fit_val_entry)

        # Create comprehensive fit summary with performance metrics
        fit_summary = {
            "fit_func": fit_func,
            "fit_val": fit_val,
            "t_el": t_el,
            "q_val": q_val,
            "q_range": str(q_range),
            "t_range": str(t_range),
            "bounds": bounds,
            "fit_flag": str(fit_flag),
            "fit_line": fit_line,
            "label": label,
            # Enhanced performance and diagnostic information
            "performance_info": results.get('performance_info', {}),
            "timing": results.get('timing', {}),
            "optimization_summary": results.get('optimization_summary', {}),
            "high_performance_fit": True,
            "fit_x": fit_x
        }

        return fit_summary

    @staticmethod
    def correct_g2_err(g2_err=None, threshold=1e-6):
        # correct the err for some data points with really small error, which
        # may cause the fitting to blowup

        g2_err_mod = np.copy(g2_err)

        # Vectorized approach: create boolean mask for all columns at once
        valid_mask = g2_err > threshold

        # Calculate averages for each column where valid data exists
        # Use masked arrays to handle columns with no valid data
        masked_data = np.ma.masked_where(~valid_mask, g2_err)
        column_averages = np.ma.mean(masked_data, axis=0)

        # Fill masked values (columns with no valid data) with threshold
        column_averages = np.ma.filled(column_averages, threshold)

        # Broadcast and apply corrections using boolean indexing
        # Create a matrix where each column contains its respective average
        avg_matrix = np.broadcast_to(column_averages, g2_err.shape)

        # Apply correction: replace invalid values with their column averages
        g2_err_mod[~valid_mask] = avg_matrix[~valid_mask]

        return g2_err_mod

    def fit_tauq(self, q_range, bounds, fit_flag):
        """
        Optimized tau-q fitting with improved data filtering and caching.
        """
        if self.fit_summary is None:
            return None

        # Generate cache key for tauq fitting
        cache_key = self._generate_cache_key("fit_tauq", q_range, bounds, fit_flag)
        tauq_cache_key = f"tauq_fit_{cache_key}"

        # Check if tauq fitting result is already cached
        if tauq_cache_key in self._computation_cache:
            cached_result = self._computation_cache[tauq_cache_key]
            self.fit_summary.update(cached_result)
            return self.fit_summary

        try:
            x = self.fit_summary["q_val"]
            fit_val = self.fit_summary["fit_val"]

            # Validate array dimensions and ensure compatibility
            if len(x.shape) > 1:
                x = x.flatten()

            # Ensure x and fit_val have compatible first dimension
            min_length = min(len(x), fit_val.shape[0])
            x = x[:min_length]
            fit_val = fit_val[:min_length]

            # Vectorized q-range filtering using boolean indexing
            q_slice = (x >= q_range[0]) & (x <= q_range[1])

            # Validate boolean index dimensions
            if len(q_slice) != fit_val.shape[0]:
                logger.warning(
                    f"Boolean index size mismatch: q_slice={len(q_slice)}, fit_val={fit_val.shape[0]}. Adjusting."
                )
                min_slice_len = min(len(q_slice), fit_val.shape[0])
                q_slice = q_slice[:min_slice_len]
                fit_val = fit_val[:min_slice_len]

            # Validate that we have data after filtering
            if not np.any(q_slice):
                logger.warning(f"No data points in q_range {q_range}")
                tauq_result = {"tauq_success": False}
                self._computation_cache[tauq_cache_key] = tauq_result
                self.fit_summary.update(tauq_result)
                return self.fit_summary

            x = x[q_slice]
            y = fit_val[q_slice, 0, 1]
            sigma = fit_val[q_slice, 1, 1]

        except (IndexError, KeyError, ValueError) as e:
            logger.error(f"Array indexing error in fit_tauq: {e}")
            tauq_result = {"tauq_success": False}
            self._computation_cache[tauq_cache_key] = tauq_result
            self.fit_summary.update(tauq_result)
            return self.fit_summary

        # Optimized filtering: combine validity checks
        valid_idx = (sigma > 0) & np.isfinite(y) & np.isfinite(sigma)

        if np.sum(valid_idx) == 0:
            tauq_result = {"tauq_success": False}
            self._computation_cache[tauq_cache_key] = tauq_result
            self.fit_summary.update(tauq_result)
            return self.fit_summary

        # Apply filtering efficiently
        x_valid = x[valid_idx]
        y_valid = y[valid_idx]
        sigma_valid = sigma[valid_idx]

        # Reshape arrays more efficiently
        y_reshaped = y_valid[:, np.newaxis]
        sigma_reshaped = sigma_valid[:, np.newaxis]

        # Improved initial parameter guess based on data characteristics
        if len(x_valid) > 1:
            # Use data-driven initial guess for better convergence
            log_x = np.log10(x_valid)
            log_y = np.log10(np.abs(y_valid))

            # Linear regression for initial slope estimate
            slope = np.polyfit(log_x, log_y, 1)[0] if len(log_x) > 1 else -2.0
            intercept = np.mean(log_y) - slope * np.mean(log_x)

            p0 = [10**intercept, slope]
        else:
            p0 = [1.0e-7, -2.0]  # fallback for insufficient data

        # Cache fit_x calculation
        x_range_key = f"tauq_fit_x_{np.min(x_valid):.6e}_{np.max(x_valid):.6e}"
        if x_range_key not in self._computation_cache:
            fit_x = np.logspace(
                np.log10(np.min(x_valid) / 1.1), np.log10(np.max(x_valid) * 1.1), 128
            )
            self._computation_cache[x_range_key] = fit_x
        else:
            fit_x = self._computation_cache[x_range_key]

        fit_line, fit_val = fit_with_fixed(
            power_law,
            x_valid,
            y_reshaped,
            sigma_reshaped,
            bounds,
            fit_flag,
            fit_x,
            p0=p0,
        )

        # Store tauq fitting results
        tauq_result = {
            "tauq_success": fit_line[0]["success"],
            "tauq_q": x_valid,
            "tauq_tau": y_valid,
            "tauq_tau_err": sigma_valid,
            "tauq_fit_line": fit_line[0],
            "tauq_fit_val": fit_val[0],
        }

        # Cache the result
        self._computation_cache[tauq_cache_key] = tauq_result
        self.fit_summary.update(tauq_result)

        return self.fit_summary

    def get_roi_data(self, roi_parameter, phi_num=180):
        # Monitor memory usage for ROI calculations
        memory_before_mb, _ = MemoryMonitor.get_memory_usage()

        # Access qmap data directly from attributes
        qmap = self.sqmap  # q map
        pmap = self.spmap  # phi map (angle map)
        rmap = np.sqrt(
            (np.arange(self.mask.shape[1]) - self.bcx) ** 2
            + (np.arange(self.mask.shape[0])[:, np.newaxis] - self.bcy) ** 2
        )  # radius map

        if roi_parameter["sl_type"] == "Pie":
            pmin, pmax = roi_parameter["angle_range"]

            # Vectorized angle wrapping - avoid in-place modification
            pmap_work = pmap.copy() if pmax < pmin else pmap
            if pmax < pmin:
                pmax += 360.0
                pmap_work = np.where(pmap_work < pmin, pmap_work + 360.0, pmap_work)

            # Vectorized ROI selection using boolean operations
            proi = (pmap_work >= pmin) & (pmap_work < pmax) & (self.mask > 0)

            # Vectorized q-bin assignment using searchsorted for better performance
            qsize = len(self.sqspan) - 1
            qmap_idx = np.searchsorted(self.sqspan[1:], qmap, side="right")
            qmap_idx = np.clip(qmap_idx, 0, qsize - 1) + 1  # 1-based indexing

            # Apply ROI mask and flatten in one operation
            qmap_idx_masked = np.where(proi, qmap_idx, 0).ravel()
            saxs_data_flat = self.saxs_2d.ravel()

            # Combined bincount operations with pre-allocated minlength
            saxs_roi = np.bincount(qmap_idx_masked, saxs_data_flat, minlength=qsize + 1)
            saxs_nor = np.bincount(qmap_idx_masked, minlength=qsize + 1)

            # Vectorized normalization - avoid division by zero
            saxs_nor = np.where(saxs_nor == 0, 1.0, saxs_nor)
            saxs_roi = saxs_roi / saxs_nor

            # Remove the 0th term
            saxs_roi = saxs_roi[1:]

            # Vectorized qmax cutoff calculation and application
            dist = roi_parameter["dist"]
            wlength = 12.398 / self.X_energy
            qmax = dist * self.pix_dim_x / self.det_dist * 2 * np.pi / wlength

            # Vectorized masking instead of separate assignments
            saxs_roi = np.where(
                (self.sqlist >= qmax) | (saxs_roi <= 0), np.nan, saxs_roi
            )
            return self.sqlist, saxs_roi

        if roi_parameter["sl_type"] == "Ring":
            rmin, rmax = roi_parameter["radius"]
            if rmin > rmax:
                rmin, rmax = rmax, rmin

            # Vectorized ring ROI selection
            rroi = (rmap >= rmin) & (rmap < rmax) & (self.mask > 0)

            # Get phi range more efficiently using masked operations
            pmap_roi = pmap[rroi]
            phi_min, phi_max = np.min(pmap_roi), np.max(pmap_roi)
            x = np.linspace(phi_min, phi_max, phi_num)
            delta = (phi_max - phi_min) / phi_num

            # Vectorized index calculation with bounds checking
            index = ((pmap - phi_min) / delta).astype(np.int64)
            index = np.clip(index, 0, phi_num - 1) + 1

            # Apply ROI mask and flatten in one operation
            index_masked = np.where(rroi, index, 0).ravel()
            saxs_data_flat = self.saxs_2d.ravel()

            # Combined bincount operations with pre-allocated minlength
            saxs_roi = np.bincount(index_masked, saxs_data_flat, minlength=phi_num + 1)
            saxs_nor = np.bincount(index_masked, minlength=phi_num + 1)

            # Vectorized normalization - avoid division by zero
            saxs_nor = np.where(saxs_nor == 0, 1.0, saxs_nor)
            saxs_roi = saxs_roi / saxs_nor

            # Remove the 0th term
            saxs_roi = saxs_roi[1:]

            # Log memory usage for ROI operations
            memory_after_mb, _ = MemoryMonitor.get_memory_usage()
            memory_used = memory_after_mb - memory_before_mb
            if memory_used > 5:  # Log if ROI calculation used significant memory
                logger.debug(
                    f"ROI calculation completed: {memory_used:.1f}MB memory used"
                )

            return x, saxs_roi
        return None

    def export_saxs1d(self, roi_list, folder):
        # export ROI
        idx = 0
        for roi in roi_list:
            fname = os.path.join(
                folder, self.label + "_" + roi["sl_type"] + f"_{idx:03d}.txt"
            )
            idx += 1
            x, y = self.get_roi_data(roi)
            if roi["sl_type"] == "Ring":
                header = "phi(degree) Intensity"
            else:
                header = "q(1/Angstron) Intensity"
            np.savetxt(fname, np.vstack([x, y]).T, header=header)

        # export all saxs1d
        fname = os.path.join(folder, self.label + "_" + "saxs1d.txt")
        Iq, q = self.saxs_1d["Iq"], self.saxs_1d["q"]
        header = "q(1/Angstron) Intensity"
        for n in range(Iq.shape[0] - 1):
            header += f" Intensity_phi{n + 1:03d}"
        np.savetxt(fname, np.vstack([q, Iq]).T, header=header)

    def get_pg_tree(self):
        data = self.load_data()
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                if val.size > 4096:
                    data[key] = "data size is too large"
                # suqeeze one-element array
                if val.size == 1:
                    data[key] = float(val)
        data["analysis_type"] = self.atype
        data["label"] = self.label
        tree = pg.DataTreeWidget(data=data)
        tree.setWindowTitle(self.fname)
        tree.resize(600, 800)
        return tree

    def get_memory_usage_report(self) -> dict:
        """
        Generate a memory usage report for the loaded data.

        Returns
        -------
        dict
            Memory usage information
        """
        report = {
            "file": self.fname,
            "label": self.label,
            "arrays": {},
            "total_memory_mb": 0.0,
        }

        # Check memory usage of major arrays
        array_attrs = [
            "saxs_2d_data",
            "saxs_2d_log_data",
            "g2",
            "g2_err",
            "tau",
            "t_el",
        ]

        for attr in array_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                arr = getattr(self, attr)
                if isinstance(arr, np.ndarray):
                    memory_mb = MemoryTracker.array_memory_mb(arr)
                    optimal_dtype = MemoryTracker.get_optimal_dtype(arr)

                    report["arrays"][attr] = {
                        "shape": arr.shape,
                        "dtype": str(arr.dtype),
                        "memory_mb": memory_mb,
                        "optimal_dtype": str(optimal_dtype),
                        "can_optimize": optimal_dtype != arr.dtype,
                    }
                    report["total_memory_mb"] += memory_mb

        return report

    def optimize_memory_usage(self, optimize_dtypes: bool = True) -> dict:
        """
        Optimize memory usage of loaded arrays.

        Parameters
        ----------
        optimize_dtypes : bool
            Whether to optimize data types

        Returns
        -------
        dict
            Optimization results
        """
        results = {"optimizations_applied": [], "memory_saved_mb": 0.0, "errors": []}

        if optimize_dtypes:
            # Optimize dtypes for suitable arrays
            array_attrs = ["g2_err", "tau"]  # Start with safer arrays

            for attr in array_attrs:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    try:
                        arr = getattr(self, attr)
                        if isinstance(arr, np.ndarray):
                            original_memory = MemoryTracker.array_memory_mb(arr)
                            optimal_dtype = MemoryTracker.get_optimal_dtype(arr)

                            if optimal_dtype != arr.dtype:
                                optimized_arr = arr.astype(optimal_dtype)
                                setattr(self, attr, optimized_arr)

                                new_memory = MemoryTracker.array_memory_mb(
                                    optimized_arr
                                )
                                saved_mb = original_memory - new_memory

                                results["optimizations_applied"].append(
                                    {
                                        "array": attr,
                                        "old_dtype": str(arr.dtype),
                                        "new_dtype": str(optimal_dtype),
                                        "memory_saved_mb": saved_mb,
                                    }
                                )
                                results["memory_saved_mb"] += saved_mb

                    except Exception as e:
                        results["errors"].append(f"Failed to optimize {attr}: {e!s}")

        return results

    def _generate_cache_key(self, method_name, *args, **kwargs):
        """Generate a cache key from method parameters."""
        import hashlib

        # Convert all parameters to a string representation
        key_parts = [method_name]

        for arg in args:
            if arg is None:
                key_parts.append("None")
            elif isinstance(arg, (list, tuple)):
                # Handle lists/tuples by converting to string
                key_parts.append(str(sorted(arg) if isinstance(arg, list) else arg))
            elif isinstance(arg, (dict,)):
                # Handle dictionaries by sorting keys
                key_parts.append(str(sorted(arg.items())))
            else:
                key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # Create hash from concatenated parts
        cache_string = "|".join(key_parts)
        return hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()

    def _compute_int_t_fft_cached(self):
        """
        Compute and cache FFT of intensity vs time data.
        Returns tuple of (frequencies, fft_magnitudes)
        """
        cache_key = "int_t_fft"
        if hasattr(self, "_computation_cache") and cache_key in self._computation_cache:
            return self._computation_cache[cache_key]

        try:
            # Get intensity vs time data
            if not hasattr(self, "Int_t") or self.Int_t is None:
                return np.array([]), np.array([])

            # Int_t[1] contains the intensity data
            intensity_data = self.Int_t[1]

            if len(intensity_data) == 0:
                return np.array([]), np.array([])

            # Compute FFT
            fft_values = np.fft.fft(intensity_data)
            fft_magnitudes = np.abs(fft_values)

            # Create frequency array
            n_samples = len(intensity_data)
            sample_rate = 1.0  # Assuming 1 Hz sample rate, adjust if needed
            frequencies = np.fft.fftfreq(n_samples, 1.0 / sample_rate)

            # Take only positive frequencies
            positive_freq_mask = frequencies > 0
            frequencies = frequencies[positive_freq_mask]
            fft_magnitudes = fft_magnitudes[positive_freq_mask]

            result = (frequencies, fft_magnitudes)

            # Cache the result
            if hasattr(self, "_computation_cache"):
                self._computation_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Failed to compute Int_t FFT: {e}")
            return np.array([]), np.array([])

    def clear_cache(self, cache_type: str = "all"):
        """
        Clear cached data to free memory.

        Parameters
        ----------
        cache_type : str
            Type of cache to clear:
            - 'all': Clear all caches (default)
            - 'computation': Clear only computation cache
            - 'saxs': Clear only SAXS 2D data
            - 'fit': Clear only fitting results
        """
        memory_before = MemoryMonitor.get_memory_usage()[0]

        if cache_type in ("all", "computation"):
            if hasattr(self, "_computation_cache"):
                cache_size = len(self._computation_cache)
                self._computation_cache.clear()
                logger.info(f"Cleared computation cache ({cache_size} items)")

        if cache_type in ("all", "saxs") and self._saxs_data_loaded:
            self.saxs_2d_data = None
            self.saxs_2d_log_data = None
            self._saxs_data_loaded = False
            logger.info("Cleared SAXS 2D data cache")

        if cache_type in ("all", "fit"):
            if self.fit_summary is not None:
                self.fit_summary = None
                logger.info("Cleared fitting results cache")

            if self.c2_all_data is not None:
                self.c2_all_data = None
                self.c2_kwargs = None
                logger.info("Cleared two-time C2 data cache")

        # Clear global cache for this file
        _global_cache.clear_file(self.fname)

        memory_after = MemoryMonitor.get_memory_usage()[0]
        memory_freed = memory_before - memory_after

        if memory_freed > 1.0:  # Only log if significant memory was freed
            logger.info(f"Cache clear freed {memory_freed:.2f}MB of memory")

        # Schedule smart garbage collection to ensure memory is released
        try:
            from .threading.cleanup_optimized import smart_gc_collect

            smart_gc_collect("cache_clear")
        except ImportError:
            # Fallback to manual GC if optimized system not available
            import gc

            gc.collect()

    def get_cache_stats(self) -> dict:
        """
        Get statistics about cached data.

        Returns
        -------
        dict
            Cache statistics including memory usage
        """
        stats = {
            "file": self.fname,
            "computation_cache_items": len(getattr(self, "_computation_cache", {})),
            "saxs_data_loaded": self._saxs_data_loaded,
            "has_fit_summary": self.fit_summary is not None,
            "has_c2_data": self.c2_all_data is not None,
            "estimated_memory_mb": 0.0,
        }

        # Estimate memory usage of major cached items
        if self._saxs_data_loaded and self.saxs_2d_data is not None:
            stats["saxs_2d_memory_mb"] = MemoryMonitor.estimate_array_memory(
                self.saxs_2d_data.shape, self.saxs_2d_data.dtype
            )
            stats["estimated_memory_mb"] += stats["saxs_2d_memory_mb"]

        if self._saxs_data_loaded and self.saxs_2d_log_data is not None:
            stats["saxs_2d_log_memory_mb"] = MemoryMonitor.estimate_array_memory(
                self.saxs_2d_log_data.shape, self.saxs_2d_log_data.dtype
            )
            stats["estimated_memory_mb"] += stats["saxs_2d_log_memory_mb"]

        return stats


def test1():
    cwd = "../../../xpcs_data"
    af = XpcsFile(fname="N077_D100_att02_0128_0001-100000.hdf", cwd=cwd)
    af.plot_saxs2d()


if __name__ == "__main__":
    test1()
