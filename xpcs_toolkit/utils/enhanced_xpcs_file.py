"""
Enhanced XpcsFile integration with advanced caching system.

This module provides integration between the existing XpcsFile class and the new
multi-level caching infrastructure for optimal performance.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple
import numpy as np

from .advanced_cache import get_global_cache
from .computation_cache import get_computation_cache, G2FitResult
from .metadata_cache import get_metadata_cache, FileMetadata, QMapData
from .adaptive_memory import (
    get_adaptive_memory_manager,
    smart_cache_decorator,
    MemoryStrategy,
)
from .logging_config import get_logger

logger = get_logger(__name__)


class CachedXpcsFileMixin:
    """
    Mixin class to enhance XpcsFile with advanced caching capabilities.

    This mixin can be used to upgrade existing XpcsFile instances with
    intelligent caching without breaking existing functionality.
    """

    def __init__(self, *args, **kwargs):
        # Initialize caching components
        self._advanced_cache = get_global_cache()
        self._computation_cache = get_computation_cache()
        self._metadata_cache = get_metadata_cache()
        self._memory_manager = get_adaptive_memory_manager()

        # Enhanced caching flags
        self._cache_enabled = True
        self._smart_prefetch_enabled = True

        logger.debug(
            f"Enhanced caching initialized for {getattr(self, 'fname', 'unknown')}"
        )

    @smart_cache_decorator("saxs_2d", memory_cost_mb=100.0)
    def get_saxs_2d_cached(
        self, use_memory_mapping: bool = None, chunk_processing: bool = None
    ) -> np.ndarray:
        """
        Enhanced SAXS 2D data loading with intelligent caching.

        Parameters
        ----------
        use_memory_mapping : bool, optional
            Whether to use memory mapping
        chunk_processing : bool, optional
            Whether to use chunked processing

        Returns
        -------
        np.ndarray
            SAXS 2D data
        """
        if not self._cache_enabled:
            return self._load_saxs_data_original()

        # Generate cache key including processing parameters
        cache_key = f"saxs_2d:{self.fname}:{use_memory_mapping}:{chunk_processing}"

        # Try to get from cache first
        cached_data, found = self._advanced_cache.get(cache_key)
        if found:
            logger.debug(f"SAXS 2D cache hit for {self.fname}")
            return cached_data

        # Load data using enhanced method
        start_time = time.time()
        data = self._load_saxs_data_batch(use_memory_mapping, chunk_processing)
        load_time_ms = (time.time() - start_time) * 1000.0

        # Cache the data with intelligent TTL based on file size
        data_size_mb = data.nbytes / (1024 * 1024) if hasattr(data, "nbytes") else 0
        ttl_seconds = max(
            3600, min(14400, data_size_mb * 10)
        )  # 1-4 hours based on size

        self._advanced_cache.put(cache_key, data, ttl_seconds=ttl_seconds)

        logger.info(
            f"SAXS 2D data loaded and cached: {self.fname} "
            f"({data_size_mb:.1f}MB, {load_time_ms:.1f}ms)"
        )

        return data

    @smart_cache_decorator("saxs_2d_log", memory_cost_mb=50.0)
    def get_saxs_2d_log_cached(self) -> np.ndarray:
        """Get cached SAXS 2D log data with smart computation."""
        cache_key = f"saxs_2d_log:{self.fname}"

        # Try cache first
        cached_data, found = self._advanced_cache.get(cache_key)
        if found:
            return cached_data

        # Compute log data
        saxs_2d = self.get_saxs_2d_cached()
        log_data = self._compute_saxs_log_chunked(saxs_2d)

        # Cache with shorter TTL since it can be recomputed
        self._advanced_cache.put(cache_key, log_data, ttl_seconds=7200)

        return log_data

    def fit_g2_cached(
        self, q_range=None, t_range=None, bounds=None, fit_flag=None, fit_func="single"
    ) -> Dict[str, Any]:
        """
        Enhanced G2 fitting with result caching and performance tracking.

        Parameters
        ----------
        q_range : tuple, optional
            Q range for fitting
        t_range : tuple, optional
            Time range for fitting
        bounds : list, optional
            Fitting bounds
        fit_flag : list, optional
            Fitting flags
        fit_func : str
            Fitting function type

        Returns
        -------
        dict
            Fitting results
        """
        if not self._cache_enabled:
            return self.fit_g2(q_range, t_range, bounds, fit_flag, fit_func)

        # Check computation cache first
        cached_result = self._computation_cache.get_cached_g2_fitting(
            file_path=self.fname,
            q_range=q_range,
            t_range=t_range,
            bounds=bounds,
            fit_flag=fit_flag,
            fit_func=fit_func,
        )

        if cached_result:
            logger.debug(f"G2 fitting cache hit for {self.fname}")
            # Convert back to dictionary format expected by existing code
            return self._g2_result_to_dict(cached_result)

        # Use computation cache decorator
        @self._computation_cache.cache_g2_fitting(
            file_path=self.fname,
            q_range=q_range,
            t_range=t_range,
            bounds=bounds,
            fit_flag=fit_flag,
            fit_func=fit_func,
        )
        def _perform_g2_fitting():
            return self.fit_g2(q_range, t_range, bounds, fit_flag, fit_func)

        result = _perform_g2_fitting()
        return (
            self._g2_result_to_dict(result)
            if isinstance(result, G2FitResult)
            else result
        )

    def get_saxs1d_data_cached(
        self, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
        """
        Enhanced SAXS 1D data extraction with caching.

        Parameters
        ----------
        **kwargs
            SAXS 1D processing parameters

        Returns
        -------
        tuple
            (q, Iq, xlabel, ylabel)
        """

        # Use computation cache for SAXS analysis
        @self._computation_cache.cache_saxs_analysis(
            file_path=self.fname, processing_params=kwargs
        )
        def _get_saxs1d_data():
            return self.get_saxs1d_data(**kwargs)

        result = _get_saxs1d_data()

        # Handle cached result object
        if hasattr(result, "q"):  # SAXSResult object
            return result.q, result.intensity, result.xlabel, result.ylabel
        else:
            return result

    def get_twotime_c2_cached(self, selection=0, correct_diag=True, max_size=32678):
        """
        Enhanced two-time correlation with intelligent caching.

        Parameters
        ----------
        selection : int
            Q-bin selection
        correct_diag : bool
            Whether to correct diagonal
        max_size : int
            Maximum correlation matrix size

        Returns
        -------
        Any
            Two-time correlation result
        """

        # Use computation cache for two-time correlation
        @self._computation_cache.cache_twotime_correlation(
            file_path=self.fname,
            selection=selection,
            max_size=max_size,
            correct_diag=correct_diag,
        )
        def _get_twotime_c2():
            return self.get_twotime_c2(selection, correct_diag, max_size)

        result = _get_twotime_c2()

        # Handle cached result object
        if hasattr(result, "c2_data"):  # TwoTimeResult object
            return result.c2_data
        else:
            return result

    def get_file_metadata_cached(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get cached file metadata with automatic validation.

        Parameters
        ----------
        force_refresh : bool
            Force refresh from disk

        Returns
        -------
        dict
            File metadata dictionary
        """
        cached_metadata = self._metadata_cache.get_file_metadata(
            self.fname, force_refresh
        )

        if cached_metadata:
            return cached_metadata.metadata_dict

        # Load metadata using original method
        metadata = self.get_hdf_info()

        # Create metadata object for caching
        try:
            import os

            stat = os.stat(self.fname)

            metadata_obj = FileMetadata(
                file_path=self.fname,
                file_size=stat.st_size,
                file_mtime=stat.st_mtime,
                analysis_type=getattr(self, "atype", "unknown"),
                metadata_dict=metadata,
                large_datasets=[],  # Could be enhanced to detect large datasets
            )

            self._metadata_cache.put_file_metadata(self.fname, metadata_obj)

        except Exception as e:
            logger.debug(f"Failed to cache metadata for {self.fname}: {e}")

        return metadata

    def get_qmap_cached(self, force_refresh: bool = False) -> Any:
        """
        Get cached Q-map data with integrity verification.

        Parameters
        ----------
        force_refresh : bool
            Force recalculation

        Returns
        -------
        Any
            Q-map object
        """
        # Simple parameter dict for Q-map (could be enhanced)
        params = {"detector_geometry": "default"}

        cached_qmap = self._metadata_cache.get_qmap_data(
            self.fname, params, force_refresh
        )

        if cached_qmap:
            # Reconstruct qmap object from cached data
            # This is a simplified version - full implementation would
            # need to recreate the complete qmap object
            return self._reconstruct_qmap_from_cache(cached_qmap)

        # Generate Q-map using original method
        qmap_obj = self.qmap  # Assuming qmap is already calculated

        # Cache Q-map data
        try:
            qmap_data = QMapData(
                file_path=self.fname,
                qmap_params=params,
                sqmap=qmap_obj.sqmap,
                dqmap=qmap_obj.dqmap,
                sqlist=qmap_obj.sqlist,
                dqlist=qmap_obj.dqlist,
                mask=qmap_obj.mask,
                geometry_params={
                    "bcx": qmap_obj.bcx,
                    "bcy": qmap_obj.bcy,
                    "det_dist": qmap_obj.det_dist,
                    "pixel_size": qmap_obj.pixel_size,
                },
            )

            self._metadata_cache.put_qmap_data(self.fname, params, qmap_data)

        except Exception as e:
            logger.debug(f"Failed to cache Q-map for {self.fname}: {e}")

        return qmap_obj

    def _g2_result_to_dict(self, g2_result: G2FitResult) -> Dict[str, Any]:
        """Convert G2FitResult object back to dictionary format."""
        if isinstance(g2_result, G2FitResult):
            return {
                "fit_func": g2_result.fit_func,
                "fit_val": g2_result.fit_val,
                "t_el": g2_result.t_el,
                "q_val": g2_result.q_val,
                "q_range": g2_result.q_range,
                "t_range": g2_result.t_range,
                "bounds": g2_result.bounds,
                "fit_flag": g2_result.fit_flag,
                "fit_line": g2_result.fit_line,
                "label": g2_result.label,
            }
        return g2_result

    def _reconstruct_qmap_from_cache(self, cached_qmap: QMapData) -> Any:
        """Reconstruct qmap object from cached data."""

        # This is a simplified reconstruction - full implementation
        # would need to recreate the complete qmap object with all methods
        class CachedQMap:
            def __init__(self, data):
                self.sqmap = data.sqmap
                self.dqmap = data.dqmap
                self.sqlist = data.sqlist
                self.dqlist = data.dqlist
                self.mask = data.mask
                self.bcx = data.geometry_params["bcx"]
                self.bcy = data.geometry_params["bcy"]
                self.det_dist = data.geometry_params["det_dist"]
                self.pixel_size = data.geometry_params["pixel_size"]

        return CachedQMap(cached_qmap)

    def enable_smart_caching(self, strategy: MemoryStrategy = MemoryStrategy.BALANCED):
        """
        Enable smart caching with specified memory strategy.

        Parameters
        ----------
        strategy : MemoryStrategy
            Memory management strategy
        """
        self._cache_enabled = True
        self._memory_manager = get_adaptive_memory_manager(strategy=strategy)
        logger.info(
            f"Smart caching enabled with {strategy.value} strategy for {self.fname}"
        )

    def disable_caching(self):
        """Disable caching for this file."""
        self._cache_enabled = False
        logger.info(f"Caching disabled for {self.fname}")

    def invalidate_cache(self):
        """Invalidate all cached data for this file."""
        # Invalidate all cache types
        self._computation_cache.invalidate_file_cache(self.fname)
        self._metadata_cache.invalidate_file(self.fname)

        # Clear any file-specific entries in advanced cache
        # This is simplified - a full implementation would track all keys
        logger.info(f"Cache invalidated for {self.fname}")

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics for this file."""
        return {
            "file_path": self.fname,
            "cache_enabled": self._cache_enabled,
            "advanced_cache_stats": self._advanced_cache.get_stats(),
            "computation_cache_stats": self._computation_cache.get_computation_stats(),
            "metadata_cache_stats": self._metadata_cache.get_cache_statistics(),
            "memory_manager_stats": self._memory_manager.get_performance_stats(),
        }

    def warm_file_cache(self, data_types: list = None):
        """
        Warm cache for this file by prefetching common data.

        Parameters
        ----------
        data_types : list, optional
            List of data types to prefetch
        """
        if data_types is None:
            data_types = ["metadata", "qmap", "saxs_1d"]

        logger.info(f"Warming cache for {self.fname} with data types: {data_types}")

        # Prefetch metadata
        if "metadata" in data_types:
            try:
                self.get_file_metadata_cached()
            except Exception as e:
                logger.debug(f"Failed to prefetch metadata: {e}")

        # Prefetch Q-map
        if "qmap" in data_types:
            try:
                self.get_qmap_cached()
            except Exception as e:
                logger.debug(f"Failed to prefetch Q-map: {e}")

        # Prefetch common SAXS data
        if "saxs_1d" in data_types:
            try:
                self.get_saxs1d_data_cached()
            except Exception as e:
                logger.debug(f"Failed to prefetch SAXS 1D: {e}")


def enhance_xpcs_file_with_caching(xpcs_file_instance):
    """
    Enhance an existing XpcsFile instance with advanced caching capabilities.

    Parameters
    ----------
    xpcs_file_instance : XpcsFile
        Existing XpcsFile instance to enhance

    Returns
    -------
    XpcsFile
        Enhanced XpcsFile instance with caching capabilities
    """
    # Dynamically add caching methods to the instance
    mixin = CachedXpcsFileMixin()

    # Copy mixin methods to the instance
    for method_name in dir(mixin):
        if not method_name.startswith("_") or method_name == "__init__":
            continue
        method = getattr(mixin, method_name)
        if callable(method):
            setattr(xpcs_file_instance, method_name, method.__get__(xpcs_file_instance))

    # Initialize caching components
    mixin.__init__()
    for attr_name in [
        "_advanced_cache",
        "_computation_cache",
        "_metadata_cache",
        "_memory_manager",
        "_cache_enabled",
        "_smart_prefetch_enabled",
    ]:
        setattr(xpcs_file_instance, attr_name, getattr(mixin, attr_name))

    logger.info(
        f"Enhanced XpcsFile instance with advanced caching: {xpcs_file_instance.fname}"
    )

    return xpcs_file_instance


class SmartXpcsFileManager:
    """
    Manager class for handling multiple XpcsFile instances with intelligent caching.

    Features:
    - Automatic cache warming for related files
    - Memory pressure management
    - Performance optimization across files
    """

    def __init__(self, memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED):
        self._memory_manager = get_adaptive_memory_manager(strategy=memory_strategy)
        self._active_files: Dict[str, Any] = {}  # file_path -> XpcsFile instance
        self._file_relationships: Dict[str, set] = {}  # file_path -> related files

        logger.info(
            f"SmartXpcsFileManager initialized with {memory_strategy.value} strategy"
        )

    def add_file(self, xpcs_file_instance, enable_caching: bool = True):
        """
        Add XpcsFile instance to manager.

        Parameters
        ----------
        xpcs_file_instance : XpcsFile
            XpcsFile instance to manage
        enable_caching : bool
            Whether to enable caching for this file
        """
        file_path = xpcs_file_instance.fname

        if enable_caching:
            enhanced_file = enhance_xpcs_file_with_caching(xpcs_file_instance)
            self._active_files[file_path] = enhanced_file
        else:
            self._active_files[file_path] = xpcs_file_instance

        # Analyze file relationships for smart prefetching
        self._analyze_file_relationships(file_path)

        logger.debug(f"Added file to manager: {file_path}")

    def remove_file(self, file_path: str):
        """Remove file from manager and clean up cache."""
        if file_path in self._active_files:
            file_instance = self._active_files[file_path]

            # Clean up cache if enhanced
            if hasattr(file_instance, "invalidate_cache"):
                file_instance.invalidate_cache()

            del self._active_files[file_path]
            logger.debug(f"Removed file from manager: {file_path}")

    def _analyze_file_relationships(self, file_path: str):
        """Analyze relationships between files for smart prefetching."""
        import os

        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)

        related_files = set()

        try:
            # Look for files with similar patterns in the same directory
            if os.path.exists(file_dir):
                base_pattern = (
                    file_name.split("_")[0]
                    if "_" in file_name
                    else file_name.split(".")[0]
                )

                for filename in os.listdir(file_dir):
                    if (
                        filename.endswith((".hdf", ".h5"))
                        and base_pattern in filename
                        and filename != file_name
                    ):
                        related_path = os.path.join(file_dir, filename)
                        related_files.add(related_path)

                        # Limit to avoid excessive relationships
                        if len(related_files) >= 10:
                            break

        except Exception as e:
            logger.debug(f"Error analyzing file relationships: {e}")

        self._file_relationships[file_path] = related_files

    def warm_related_caches(self, file_path: str):
        """Warm caches for files related to the specified file."""
        related_files = self._file_relationships.get(file_path, set())

        if related_files:
            # Use metadata cache for warming related files
            metadata_cache = get_metadata_cache()
            metadata_cache.warm_cache(list(related_files))

            logger.debug(
                f"Warmed cache for {len(related_files)} files related to {file_path}"
            )

    def optimize_memory_usage(self):
        """Optimize memory usage across all managed files."""
        # Get memory pressure
        memory_info = self._memory_manager.get_memory_recommendations()
        memory_pressure = memory_info["current_memory_pressure"]

        if memory_pressure > 85.0:
            # High memory pressure - perform aggressive cleanup
            logger.warning(
                f"High memory pressure ({memory_pressure:.1f}%), performing cleanup"
            )

            # Clean least recently used files
            for file_path, file_instance in list(self._active_files.items()):
                if hasattr(file_instance, "clear_cache"):
                    file_instance.clear_cache("saxs")  # Clear large SAXS data

        elif memory_pressure > 75.0:
            # Moderate memory pressure - selective cleanup
            logger.info(
                f"Moderate memory pressure ({memory_pressure:.1f}%), optimizing caches"
            )

            # Promote data to lower cache levels
            advanced_cache = get_global_cache()
            advanced_cache._promote_l1_to_l2(force=True, target_freed_mb=100.0)

    def get_manager_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for the manager."""
        stats = {
            "active_files": len(self._active_files),
            "total_relationships": sum(
                len(related) for related in self._file_relationships.values()
            ),
            "memory_manager_stats": self._memory_manager.get_performance_stats(),
            "file_cache_stats": {},
        }

        # Get cache statistics for each file
        for file_path, file_instance in self._active_files.items():
            if hasattr(file_instance, "get_cache_statistics"):
                stats["file_cache_stats"][file_path] = (
                    file_instance.get_cache_statistics()
                )

        return stats

    def shutdown(self):
        """Shutdown manager and clean up resources."""
        logger.info("Shutting down SmartXpcsFileManager")

        # Clean up all files
        for file_path in list(self._active_files.keys()):
            self.remove_file(file_path)

        # Shutdown memory manager
        self._memory_manager.shutdown()
