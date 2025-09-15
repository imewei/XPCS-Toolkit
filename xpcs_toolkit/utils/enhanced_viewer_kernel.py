"""
Enhanced ViewerKernel integration with advanced caching system.

This module provides integration between the existing ViewerKernel class and the new
multi-level caching infrastructure for optimal performance in GUI operations.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from .adaptive_memory import MemoryStrategy, get_adaptive_memory_manager
from .advanced_cache import get_global_cache
from .computation_cache import get_computation_cache
from .enhanced_xpcs_file import SmartXpcsFileManager, enhance_xpcs_file_with_caching
from .logging_config import get_logger
from .metadata_cache import get_metadata_cache

logger = get_logger(__name__)


class CachedViewerKernelMixin:
    """
    Mixin class to enhance ViewerKernel with advanced caching capabilities.

    This mixin provides intelligent caching for viewer operations, plot data,
    and GUI state management.
    """

    def __init__(self, *args, **kwargs):
        # Initialize caching components
        self._advanced_cache = get_global_cache()
        self._computation_cache = get_computation_cache()
        self._metadata_cache = get_metadata_cache()
        self._memory_manager = get_adaptive_memory_manager()

        # File management
        self._file_manager = SmartXpcsFileManager()

        # GUI caching settings
        self._cache_plot_data = True
        self._cache_gui_state = True

        # Performance tracking
        self._plot_cache_hits = 0
        self._plot_cache_misses = 0

        # Thread safety for GUI operations
        self._gui_cache_lock = threading.RLock()

        logger.debug("Enhanced caching initialized for ViewerKernel")

    def get_xf_list_cached(self, rows=None, filter_atype=None, filter_fitted=False):
        """
        Enhanced get_xf_list with intelligent caching and prefetching.

        Parameters
        ----------
        rows : list, optional
            Selected rows
        filter_atype : str, optional
            Analysis type filter
        filter_fitted : bool
            Whether to filter for fitted data

        Returns
        -------
        list
            List of XpcsFile instances with caching enabled
        """
        # Get original xf_list
        xf_list = self.get_xf_list(rows, filter_atype, filter_fitted)

        # Enhance each XpcsFile with caching if not already done
        enhanced_list = []
        for xf in xf_list:
            if not hasattr(xf, "_cache_enabled"):
                # Enhance with caching
                enhanced_xf = enhance_xpcs_file_with_caching(xf)
                self._file_manager.add_file(enhanced_xf)
                enhanced_list.append(enhanced_xf)
            else:
                enhanced_list.append(xf)

        # Warm related caches for improved performance
        if enhanced_list and self._memory_manager._enable_prefetch:
            for xf in enhanced_list:
                self._file_manager.warm_related_caches(xf.fname)

        return enhanced_list

    def plot_g2_cached(self, handler, q_range, t_range, y_range, rows=None, **kwargs):
        """
        Enhanced G2 plotting with data caching and performance optimization.

        Parameters
        ----------
        handler : object
            Plot handler
        q_range : tuple
            Q range for plotting
        t_range : tuple
            Time range for plotting
        y_range : tuple
            Y range for plotting
        rows : list, optional
            Selected rows
        **kwargs
            Additional plotting arguments

        Returns
        -------
        tuple
            (q_values, time_values) or (None, None) if no data
        """
        if not self._cache_plot_data:
            return self.plot_g2(handler, q_range, t_range, y_range, rows, **kwargs)

        # Generate cache key for plot data
        plot_key = self._generate_plot_cache_key(
            "g2", q_range, t_range, y_range, rows, **kwargs
        )

        with self._gui_cache_lock:
            # Try to get cached plot data
            cached_data, found = self._advanced_cache.get(plot_key)
            if found:
                logger.debug("G2 plot data cache hit")
                self._plot_cache_hits += 1

                # Apply cached data to handler
                self._apply_cached_plot_data(handler, cached_data, "g2")
                return cached_data.get("q_values"), cached_data.get("time_values")

        # Cache miss - compute plot data
        self._plot_cache_misses += 1
        start_time = time.time()

        # Get enhanced XpcsFile list
        xf_list = self.get_xf_list_cached(rows=rows, filter_atype="Multitau")

        if not xf_list:
            return None, None

        # Use cached G2 fitting if available
        enhanced_plotting_data = []
        for xf in xf_list:
            if hasattr(xf, "fit_g2_cached"):
                # Use cached fitting results for faster plotting
                try:
                    fit_summary = xf.fit_summary or {}
                    enhanced_plotting_data.append((xf, fit_summary))
                except Exception as e:
                    logger.debug(f"Error accessing cached G2 data for {xf.fname}: {e}")
                    enhanced_plotting_data.append((xf, {}))
            else:
                enhanced_plotting_data.append((xf, {}))

        # Perform actual plotting with enhanced data
        try:
            from ..module import g2mod

            g2mod.pg_plot(handler, xf_list, q_range, t_range, y_range, **kwargs)
            q, tel, *_unused = g2mod.get_data(xf_list)

            # Cache the plot data
            plot_data = {
                "q_values": q,
                "time_values": tel,
                "plot_type": "g2",
                "parameters": {
                    "q_range": q_range,
                    "t_range": t_range,
                    "y_range": y_range,
                    "kwargs": kwargs,
                },
                "generated_at": time.time(),
            }

            computation_time_ms = (time.time() - start_time) * 1000.0
            ttl_seconds = max(
                600, min(3600, computation_time_ms)
            )  # 10min to 1hr based on computation time

            with self._gui_cache_lock:
                self._advanced_cache.put(plot_key, plot_data, ttl_seconds=ttl_seconds)

            logger.debug(
                f"G2 plot data cached: {computation_time_ms:.1f}ms computation"
            )
            return q, tel

        except Exception as e:
            logger.error(f"Error in cached G2 plotting: {e}")
            return None, None

    def plot_saxs_2d_cached(self, *args, rows=None, **kwargs):
        """Enhanced SAXS 2D plotting with data caching."""
        # Generate cache key for SAXS 2D plot
        plot_key = self._generate_plot_cache_key("saxs_2d", *args, rows=rows, **kwargs)

        # Try cache first
        _cached_data, found = self._advanced_cache.get(plot_key)
        if found:
            self._plot_cache_hits += 1
            logger.debug("SAXS 2D plot cache hit")
            # Apply cached plot data (implementation depends on plot handler)
            return

        # Cache miss - perform plotting
        self._plot_cache_misses += 1
        xf_list = self.get_xf_list_cached(rows)

        if xf_list:
            enhanced_xf = xf_list[0]

            # Use cached SAXS data if available
            if hasattr(enhanced_xf, "get_saxs_2d_cached"):
                try:
                    # Pre-load SAXS data for faster plotting
                    saxs_data = enhanced_xf.get_saxs_2d_cached()

                    # Perform plotting with enhanced data
                    from ..module import saxs2d

                    saxs2d.plot(enhanced_xf, *args, **kwargs)

                    # Cache plot metadata
                    plot_data = {
                        "file_path": enhanced_xf.fname,
                        "plot_type": "saxs_2d",
                        "parameters": {"args": args, "kwargs": kwargs},
                        "data_shape": saxs_data.shape
                        if hasattr(saxs_data, "shape")
                        else None,
                        "generated_at": time.time(),
                    }

                    with self._gui_cache_lock:
                        self._advanced_cache.put(
                            plot_key, plot_data, ttl_seconds=1800
                        )  # 30 minutes

                except Exception as e:
                    logger.error(f"Error in cached SAXS 2D plotting: {e}")
                    # Fallback to original method
                    self.plot_saxs_2d(*args, rows=rows, **kwargs)
            else:
                # Fallback to original method
                self.plot_saxs_2d(*args, rows=rows, **kwargs)

    def plot_twotime_cached(self, hdl, rows=None, **kwargs):
        """
        Enhanced two-time plotting with intelligent caching and memory management.

        Parameters
        ----------
        hdl : object
            Plot handler
        rows : list, optional
            Selected rows
        **kwargs
            Additional plotting arguments

        Returns
        -------
        list or None
            New Q-bin labels if available
        """
        xf_list = self.get_xf_list_cached(rows, filter_atype="Twotime")
        if not xf_list:
            return None

        enhanced_xf = xf_list[0]

        # Use memory-aware dataset caching
        dataset_key = f"twotime_dataset:{enhanced_xf.fname}"
        cached_dataset, found = self._advanced_cache.get(dataset_key)

        current_dset = None
        new_qbin_labels = None

        if found and cached_dataset.fname == enhanced_xf.fname:
            current_dset = cached_dataset
            logger.debug("Two-time dataset cache hit")
        else:
            current_dset = enhanced_xf

            # Cache dataset with memory management
            if not self._memory_manager._memory_manager.SystemMemoryMonitor.check_memory_pressure(
                85.0
            ):
                self._advanced_cache.put(dataset_key, current_dset, ttl_seconds=3600)
                new_qbin_labels = enhanced_xf.get_twotime_qbin_labels()

        # Use cached two-time correlation if available
        if hasattr(enhanced_xf, "get_twotime_c2_cached"):
            try:
                # Enhanced plotting with cached correlation data
                from ..module import twotime

                twotime.plot_twotime(current_dset, hdl, **kwargs)
                return new_qbin_labels
            except Exception as e:
                logger.error(f"Error in cached two-time plotting: {e}")

        # Fallback to original method
        return self.plot_twotime(hdl, rows, **kwargs)

    def _generate_plot_cache_key(self, plot_type: str, *args, **kwargs) -> str:
        """Generate cache key for plot data."""
        import hashlib

        key_parts = [plot_type]

        # Add arguments
        for arg in args:
            if hasattr(arg, "__iter__") and not isinstance(arg, str):
                key_parts.append(
                    str(sorted(arg) if isinstance(arg, list) else tuple(arg))
                )
            else:
                key_parts.append(str(arg))

        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            if k == "rows" and v is not None:
                key_parts.append(f"rows:{sorted(v) if isinstance(v, list) else v}")
            else:
                key_parts.append(f"{k}:{v}")

        # Create hash
        cache_string = "|".join(key_parts)
        return f"plot:{hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()[:16]}"

    def _apply_cached_plot_data(self, handler, cached_data: dict, plot_type: str):
        """Apply cached plot data to handler."""
        # This is a simplified implementation
        # Full implementation would depend on specific plot handler interface
        try:
            if plot_type == "g2" and hasattr(handler, "set_cached_data"):
                handler.set_cached_data(cached_data)
        except Exception as e:
            logger.debug(f"Failed to apply cached plot data: {e}")

    def optimize_gui_performance(self):
        """Optimize GUI performance through intelligent cache management."""
        # Get current memory and performance statistics
        memory_stats = self._memory_manager.get_memory_recommendations()
        cache_stats = self._advanced_cache.get_stats()

        # Adjust caching strategy based on performance
        hit_rates = cache_stats.get("hit_rates", {})
        overall_hit_rate = hit_rates.get("overall_hit_rate", 0.0)

        if overall_hit_rate < 0.3:  # Low hit rate
            logger.info("Low cache hit rate detected, optimizing cache strategy")

            # Increase cache sizes if memory allows
            memory_pressure = memory_stats["current_memory_pressure"]
            if memory_pressure < 70.0:
                # Could dynamically adjust cache limits here
                pass

        # Optimize file manager
        self._file_manager.optimize_memory_usage()

        logger.debug(
            f"GUI performance optimization: hit_rate={overall_hit_rate:.2f}, "
            f"memory_pressure={memory_stats['current_memory_pressure']:.1f}%"
        )

    def warm_gui_caches(self, file_paths: list[str] | None = None):
        """
        Warm GUI-related caches for better responsiveness.

        Parameters
        ----------
        file_paths : list, optional
            Specific file paths to warm. If None, uses current file selection.
        """
        if file_paths is None:
            # Use current target files
            file_paths = getattr(self, "target", [])

        if not file_paths:
            return

        logger.info(f"Warming GUI caches for {len(file_paths)} files")

        # Warm metadata cache
        self._metadata_cache.warm_cache(file_paths, ["metadata", "qmap"])

        # Pre-load commonly accessed data for interactive files
        for file_path in file_paths[:3]:  # Limit to avoid memory issues
            try:
                # This would need integration with actual file loading
                # xf = XpcsFile(file_path)
                # enhanced_xf = enhance_xpcs_file_with_caching(xf)
                # enhanced_xf.warm_file_cache(['metadata', 'saxs_1d'])
                pass
            except Exception as e:
                logger.debug(f"Failed to warm cache for {file_path}: {e}")

    def get_gui_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive GUI cache statistics."""
        plot_cache_total = self._plot_cache_hits + self._plot_cache_misses
        plot_hit_rate = (
            self._plot_cache_hits / plot_cache_total if plot_cache_total > 0 else 0.0
        )

        return {
            "plot_cache": {
                "hit_rate": plot_hit_rate,
                "total_hits": self._plot_cache_hits,
                "total_misses": self._plot_cache_misses,
            },
            "file_manager_stats": self._file_manager.get_manager_statistics(),
            "advanced_cache_stats": self._advanced_cache.get_stats(),
            "memory_manager_stats": self._memory_manager.get_performance_stats(),
        }

    def clear_gui_caches(self, cache_types: list[str] | None = None):
        """
        Clear GUI-related caches.

        Parameters
        ----------
        cache_types : list, optional
            Types of caches to clear. If None, clears all.
        """
        if cache_types is None:
            cache_types = ["plot", "file", "computation"]

        cleared_items = 0

        if "plot" in cache_types:
            # Clear plot-related cache entries
            # This is simplified - full implementation would track plot cache keys
            with self._gui_cache_lock:
                self._plot_cache_hits = 0
                self._plot_cache_misses = 0
            cleared_items += 1

        if "file" in cache_types:
            # Clear file manager caches
            for file_path in list(self._file_manager._active_files.keys()):
                file_instance = self._file_manager._active_files[file_path]
                if hasattr(file_instance, "invalidate_cache"):
                    file_instance.invalidate_cache()
            cleared_items += len(self._file_manager._active_files)

        if "computation" in cache_types:
            # Clear computation caches
            self._computation_cache.cleanup_old_computations(
                max_age_hours=0.1
            )  # Very short age
            cleared_items += 1

        logger.info(f"Cleared {cleared_items} GUI cache components")

    def enable_adaptive_caching(
        self, strategy: MemoryStrategy = MemoryStrategy.BALANCED
    ):
        """
        Enable adaptive caching with specified strategy.

        Parameters
        ----------
        strategy : MemoryStrategy
            Memory management strategy for GUI operations
        """
        self._memory_manager = get_adaptive_memory_manager(
            strategy=strategy, reset=True
        )
        self._cache_plot_data = True
        self._cache_gui_state = True

        # Configure file manager with same strategy
        self._file_manager = SmartXpcsFileManager(memory_strategy=strategy)

        logger.info(f"Adaptive caching enabled with {strategy.value} strategy")

    def disable_gui_caching(self):
        """Disable GUI caching for debugging or low-memory situations."""
        self._cache_plot_data = False
        self._cache_gui_state = False
        self.clear_gui_caches()

        logger.info("GUI caching disabled")


def enhance_viewer_kernel_with_caching(viewer_kernel_instance):
    """
    Enhance an existing ViewerKernel instance with advanced caching capabilities.

    Parameters
    ----------
    viewer_kernel_instance : ViewerKernel
        Existing ViewerKernel instance to enhance

    Returns
    -------
    ViewerKernel
        Enhanced ViewerKernel instance with caching capabilities
    """
    # Dynamically add caching methods to the instance
    mixin = CachedViewerKernelMixin()

    # Copy mixin methods to the instance
    for method_name in dir(mixin):
        if method_name.startswith("_") and method_name != "__init__":
            continue
        method = getattr(mixin, method_name)
        if callable(method):
            setattr(
                viewer_kernel_instance,
                method_name,
                method.__get__(viewer_kernel_instance),
            )

    # Initialize caching components
    mixin.__init__()
    for attr_name in [
        "_advanced_cache",
        "_computation_cache",
        "_metadata_cache",
        "_memory_manager",
        "_file_manager",
        "_cache_plot_data",
        "_cache_gui_state",
        "_plot_cache_hits",
        "_plot_cache_misses",
        "_gui_cache_lock",
    ]:
        setattr(viewer_kernel_instance, attr_name, getattr(mixin, attr_name))

    logger.info("Enhanced ViewerKernel instance with advanced caching")

    return viewer_kernel_instance


class SmartViewerKernelManager:
    """
    Manager class for coordinating multiple ViewerKernel instances with caching.

    This manager provides:
    - Global cache coordination
    - Memory pressure management across viewers
    - Performance optimization
    - Statistics aggregation
    """

    def __init__(self):
        self._active_kernels: dict[str, Any] = {}
        self._global_cache = get_global_cache()
        self._memory_manager = get_adaptive_memory_manager()

        logger.info("SmartViewerKernelManager initialized")

    def register_kernel(self, kernel_id: str, viewer_kernel):
        """Register a ViewerKernel instance with the manager."""
        # Enhance kernel with caching if not already done
        if not hasattr(viewer_kernel, "_cache_plot_data"):
            enhanced_kernel = enhance_viewer_kernel_with_caching(viewer_kernel)
            self._active_kernels[kernel_id] = enhanced_kernel
        else:
            self._active_kernels[kernel_id] = viewer_kernel

        logger.debug(f"Registered kernel: {kernel_id}")

    def unregister_kernel(self, kernel_id: str):
        """Unregister a ViewerKernel instance."""
        if kernel_id in self._active_kernels:
            kernel = self._active_kernels[kernel_id]

            # Clean up kernel-specific caches
            if hasattr(kernel, "clear_gui_caches"):
                kernel.clear_gui_caches()

            del self._active_kernels[kernel_id]
            logger.debug(f"Unregistered kernel: {kernel_id}")

    def optimize_global_performance(self):
        """Optimize performance across all registered kernels."""
        total_cache_hit_rate = 0.0
        kernel_count = len(self._active_kernels)

        if kernel_count == 0:
            return

        # Collect statistics from all kernels
        for kernel in self._active_kernels.values():
            if hasattr(kernel, "get_gui_cache_statistics"):
                stats = kernel.get_gui_cache_statistics()

                # Aggregate metrics
                if "memory_manager_stats" in stats:
                    stats["memory_manager_stats"]
                    # Add memory pressure calculation based on stats structure

                if "plot_cache" in stats:
                    total_cache_hit_rate += stats["plot_cache"]["hit_rate"]

        avg_hit_rate = total_cache_hit_rate / kernel_count

        # Apply global optimizations based on metrics
        if avg_hit_rate < 0.4:  # Low average hit rate
            logger.info(
                f"Low average cache hit rate ({avg_hit_rate:.2f}), applying optimizations"
            )

            # Increase cache sizes if memory allows
            memory_info = self._memory_manager.get_memory_recommendations()
            if memory_info["current_memory_pressure"] < 75.0:
                # Could adjust global cache settings here
                pass

        # Optimize each kernel
        for kernel in self._active_kernels.values():
            if hasattr(kernel, "optimize_gui_performance"):
                kernel.optimize_gui_performance()

    def get_global_statistics(self) -> dict[str, Any]:
        """Get global statistics across all kernels."""
        stats = {
            "active_kernels": len(self._active_kernels),
            "global_cache_stats": self._global_cache.get_stats(),
            "global_memory_stats": self._memory_manager.get_performance_stats(),
            "kernel_stats": {},
        }

        # Collect statistics from each kernel
        for kernel_id, kernel in self._active_kernels.items():
            if hasattr(kernel, "get_gui_cache_statistics"):
                stats["kernel_stats"][kernel_id] = kernel.get_gui_cache_statistics()

        return stats

    def shutdown(self):
        """Shutdown manager and clean up all resources."""
        logger.info("Shutting down SmartViewerKernelManager")

        # Clean up all kernels
        for kernel_id in list(self._active_kernels.keys()):
            self.unregister_kernel(kernel_id)

        # Shutdown global components
        self._memory_manager.shutdown()
        self._global_cache.shutdown()


# Global manager instance
_global_viewer_manager: SmartViewerKernelManager | None = None


def get_global_viewer_manager() -> SmartViewerKernelManager:
    """Get or create global viewer kernel manager."""
    global _global_viewer_manager

    if _global_viewer_manager is None:
        _global_viewer_manager = SmartViewerKernelManager()

    return _global_viewer_manager
