"""
Visualization Performance Optimization for XPCS Toolkit

This module addresses critical performance bottlenecks in the visualization layer
that were identified during comprehensive analysis, including PyQtGraph optimization,
matplotlib performance improvements, and intelligent plot handler selection.
"""

import importlib.util
import time
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any

import numpy as np

# Check PyQtGraph availability
PYQTGRAPH_AVAILABLE = importlib.util.find_spec("pyqtgraph") is not None
if PYQTGRAPH_AVAILABLE:
    import pyqtgraph as pg

# Check Matplotlib availability
MATPLOTLIB_AVAILABLE = importlib.util.find_spec("matplotlib") is not None

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager


# Lazy import threading to avoid circular dependencies
def _get_threading_manager():
    try:
        from ..threading.unified_threading import get_unified_threading_manager

        return get_unified_threading_manager()
    except ImportError:
        return None


logger = get_logger(__name__)


class PlotBackend(Enum):
    """Available plotting backends."""

    PYQTGRAPH = "pyqtgraph"
    MATPLOTLIB = "matplotlib"
    AUTO = "auto"


class PlotComplexity(Enum):
    """Plot complexity levels for optimization decisions."""

    SIMPLE = "simple"  # Single image/plot
    MODERATE = "moderate"  # Multiple series, ROIs
    COMPLEX = "complex"  # Many series, complex interactions
    VERY_COMPLEX = "very_complex"  # Real-time updates, large datasets


@contextmanager
def optimized_pyqtgraph_context():
    """Context manager for optimized PyQtGraph settings."""
    if not PYQTGRAPH_AVAILABLE:
        yield
        return

    # Store original settings
    original_settings = {
        "useOpenGL": pg.getConfigOption("useOpenGL"),
        "enableExperimental": pg.getConfigOption("enableExperimental"),
        "crashWarning": pg.getConfigOption("crashWarning"),
        "imageAxisOrder": pg.getConfigOption("imageAxisOrder"),
    }

    try:
        # Apply optimized settings
        pg.setConfigOptions(
            useOpenGL=True,  # Enable OpenGL acceleration
            enableExperimental=True,  # Enable performance features
            crashWarning=False,  # Reduce warning overhead
            imageAxisOrder="row-major",  # Optimize for NumPy
            antialias=False,  # Disable antialiasing for large datasets
        )

        logger.debug("Applied optimized PyQtGraph settings")
        yield

    finally:
        # Restore original settings
        pg.setConfigOptions(**original_settings)
        logger.debug("Restored original PyQtGraph settings")


class ImageDisplayOptimizer:
    """Optimizes large image display performance."""

    def __init__(self, max_display_size: tuple[int, int] = (2048, 2048)):
        self.max_display_size = max_display_size
        self.memory_manager = get_memory_manager()

    def optimize_image_for_display(
        self, image: np.ndarray, target_size: tuple[int, int] | None = None
    ) -> np.ndarray:
        """
        Optimize image for display performance.

        Parameters
        ----------
        image : np.ndarray
            Input image array
        target_size : tuple[int, int], optional
            Target display size (height, width)

        Returns
        -------
        np.ndarray
            Optimized image for display
        """
        if target_size is None:
            target_size = self.max_display_size

        # Check if downsampling is needed
        height, width = image.shape[:2]
        target_height, target_width = target_size

        if height <= target_height and width <= target_width:
            # No optimization needed
            return image

        # Calculate optimal downsampling factor
        height_factor = height / target_height
        width_factor = width / target_width
        downsample_factor = max(height_factor, width_factor)

        if downsample_factor <= 1.5:
            # Minor size difference, no downsampling
            return image

        # Apply intelligent downsampling
        logger.info(
            f"Downsampling image from {image.shape} by factor {downsample_factor:.1f}"
        )

        # Use area-based downsampling for better quality
        new_height = int(height / downsample_factor)
        new_width = int(width / downsample_factor)

        if len(image.shape) == 2:
            # 2D image
            downsampled = self._downsample_2d(image, (new_height, new_width))
        elif len(image.shape) == 3:
            # 3D image (e.g., time series)
            downsampled = np.zeros(
                (image.shape[0], new_height, new_width), dtype=image.dtype
            )
            for t in range(image.shape[0]):
                downsampled[t] = self._downsample_2d(image[t], (new_height, new_width))
        else:
            logger.warning(f"Unsupported image dimensionality: {len(image.shape)}")
            return image

        memory_saved = (image.nbytes - downsampled.nbytes) / (1024 * 1024)
        logger.debug(f"Image downsampling saved {memory_saved:.1f}MB memory")

        return downsampled

    def _downsample_2d(
        self, image: np.ndarray, target_shape: tuple[int, int]
    ) -> np.ndarray:
        """Downsample 2D image using area averaging."""
        from scipy import ndimage

        height, width = image.shape
        target_height, target_width = target_shape

        # Calculate zoom factors
        zoom_y = target_height / height
        zoom_x = target_width / width

        # Use scipy's zoom with area averaging
        downsampled = ndimage.zoom(image, (zoom_y, zoom_x), order=1, prefilter=True)

        return downsampled.astype(image.dtype)


class PlotPerformanceOptimizer:
    """Optimizes plot performance based on data characteristics and system resources."""

    def __init__(self):
        self.memory_manager = get_memory_manager()
        self.image_optimizer = ImageDisplayOptimizer()

    def select_optimal_backend(
        self, plot_type: str, data_size_mb: float, complexity: PlotComplexity
    ) -> PlotBackend:
        """
        Select optimal plotting backend based on data characteristics.

        Parameters
        ----------
        plot_type : str
            Type of plot ('image', 'line', 'scatter', etc.)
        data_size_mb : float
            Size of data to plot in MB
        complexity : PlotComplexity
            Complexity level of the plot

        Returns
        -------
        PlotBackend
            Recommended plotting backend
        """
        memory_pressure = self.memory_manager.get_memory_pressure()

        # Decision matrix based on plot characteristics
        if plot_type in ["image", "saxs_2d"]:
            if data_size_mb > 50 or memory_pressure in [
                MemoryPressure.HIGH,
                MemoryPressure.CRITICAL,
            ]:
                # Large images: use PyQtGraph with optimization
                return PlotBackend.PYQTGRAPH
            # Smaller images: either backend works
            return PlotBackend.PYQTGRAPH

        if plot_type in ["line", "scatter", "g2"]:
            if complexity in [PlotComplexity.COMPLEX, PlotComplexity.VERY_COMPLEX]:
                # Complex plots with many series: PyQtGraph for performance
                return PlotBackend.PYQTGRAPH
            if data_size_mb < 10:
                # Small datasets: matplotlib for quality
                return PlotBackend.MATPLOTLIB
            # Medium datasets: PyQtGraph for speed
            return PlotBackend.PYQTGRAPH

        # Default to PyQtGraph for unknown types
        return PlotBackend.PYQTGRAPH

    def optimize_pyqtgraph_plot(
        self, plot_widget, data: np.ndarray, plot_type: str
    ) -> dict[str, Any]:
        """
        Apply PyQtGraph-specific optimizations.

        Parameters
        ----------
        plot_widget : object
            PyQtGraph plot widget
        data : np.ndarray
            Data to be plotted
        plot_type : str
            Type of plot

        Returns
        -------
        dict[str, Any]
            Optimization settings applied
        """
        optimizations = {}

        if plot_type == "image":
            # Image-specific optimizations
            if hasattr(plot_widget, "getImageItem"):
                image_item = plot_widget.getImageItem()
                if image_item is not None:
                    # Enable OpenGL if available
                    if hasattr(image_item, "setOpts"):
                        image_item.setOpts(useOpenGL=True)
                        optimizations["opengl_enabled"] = True

            # Optimize image data
            if data.size > 1e6:  # > 1M pixels
                optimized_data = self.image_optimizer.optimize_image_for_display(data)
                optimizations["image_downsampled"] = data.shape != optimized_data.shape
                optimizations["original_size"] = data.shape
                optimizations["optimized_size"] = optimized_data.shape
                data = optimized_data

        elif plot_type in ["line", "scatter"]:
            # Line/scatter plot optimizations
            if hasattr(plot_widget, "setDownsampling"):
                # Enable automatic downsampling for large datasets
                if data.size > 1e5:  # > 100k points
                    plot_widget.setDownsampling(auto=True, mode="peak")
                    optimizations["downsampling_enabled"] = True

            # Disable antialiasing for large datasets
            if data.size > 5e4:  # > 50k points
                if hasattr(plot_widget, "setAntialiasing"):
                    plot_widget.setAntialiasing(False)
                    optimizations["antialiasing_disabled"] = True

        return optimizations

    def optimize_matplotlib_plot(
        self, figure, axes, data: np.ndarray, plot_type: str
    ) -> dict[str, Any]:
        """
        Apply matplotlib-specific optimizations.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Matplotlib figure
        axes : matplotlib.axes.Axes
            Matplotlib axes
        data : np.ndarray
            Data to be plotted
        plot_type : str
            Type of plot

        Returns
        -------
        dict[str, Any]
            Optimization settings applied
        """
        optimizations = {}

        # General matplotlib optimizations
        figure.set_facecolor("white")  # Faster rendering
        optimizations["background_optimized"] = True

        if plot_type == "image":
            # Image-specific optimizations
            if data.size > 1e6:  # > 1M pixels
                optimized_data = self.image_optimizer.optimize_image_for_display(data)
                optimizations["image_downsampled"] = data.shape != optimized_data.shape
                data = optimized_data

            # Use faster interpolation
            if hasattr(axes, "imshow"):
                optimizations["interpolation"] = "nearest"

        elif plot_type in ["line", "scatter"]:
            # Line/scatter optimizations
            if data.size > 1e5:  # > 100k points
                # Downsample data for matplotlib
                downsample_factor = int(np.ceil(data.size / 1e4))  # Target ~10k points
                if len(data.shape) == 1:
                    data = data[::downsample_factor]
                elif len(data.shape) == 2:
                    data = data[::downsample_factor, :]
                optimizations["data_downsampled"] = True
                optimizations["downsample_factor"] = downsample_factor

        return optimizations


class ProgressiveImageLoader:
    """Loads large images progressively to maintain responsiveness."""

    def __init__(self, chunk_size_mb: float = 50.0):
        self.chunk_size_mb = chunk_size_mb

    def load_image_progressive(
        self, image_loader_func, progress_callback: Callable | None = None
    ) -> np.ndarray:
        """
        Load image progressively with progress reporting.

        Parameters
        ----------
        image_loader_func : callable
            Function that loads the image data
        progress_callback : callable, optional
            Callback for progress updates (current, total, message)

        Returns
        -------
        np.ndarray
            Loaded image data
        """
        # This is a framework for progressive loading
        # Implementation would depend on the specific data source

        if progress_callback:
            progress_callback(0, 100, "Starting image load...")

        start_time = time.time()

        try:
            # Load image data
            image_data = image_loader_func()

            if progress_callback:
                progress_callback(50, 100, "Processing image data...")

            # Apply any necessary post-processing
            if hasattr(image_data, "shape") and len(image_data.shape) > 2:
                # Multi-dimensional data - may need chunked processing
                pass

            if progress_callback:
                progress_callback(100, 100, "Image load completed")

            load_time = time.time() - start_time
            logger.debug(f"Progressive image load completed in {load_time:.2f}s")

            return image_data

        except Exception as e:
            if progress_callback:
                progress_callback(0, 100, f"Image load failed: {e!s}")
            raise


class PlotCacheManager:
    """Manages caching of plot data and rendered plots."""

    def __init__(self, max_cache_size_mb: float = 200.0):
        self.max_cache_size_mb = max_cache_size_mb
        self.plot_cache = {}
        self.cache_access_times = {}
        self.current_cache_size_mb = 0.0

    def get_plot_cache_key(self, data_hash: str, plot_params: dict[str, Any]) -> str:
        """Generate cache key for plot data."""
        # Create a hash from plot parameters
        param_str = str(sorted(plot_params.items()))
        param_hash = hash(param_str)
        return f"{data_hash}_{param_hash}"

    def cache_plot_data(self, cache_key: str, plot_data: Any):
        """Cache processed plot data."""
        try:
            # Estimate size
            if hasattr(plot_data, "nbytes"):
                size_mb = plot_data.nbytes / (1024 * 1024)
            else:
                size_mb = 1.0  # Default estimate

            # Check if we need to evict old entries
            while (
                self.current_cache_size_mb + size_mb > self.max_cache_size_mb
                and self.plot_cache
            ):
                self._evict_oldest_entry()

            # Cache the data
            self.plot_cache[cache_key] = plot_data
            self.cache_access_times[cache_key] = time.time()
            self.current_cache_size_mb += size_mb

            logger.debug(f"Cached plot data: {cache_key} ({size_mb:.1f}MB)")

        except Exception as e:
            logger.warning(f"Failed to cache plot data: {e}")

    def get_cached_plot_data(self, cache_key: str) -> Any | None:
        """Retrieve cached plot data."""
        if cache_key in self.plot_cache:
            self.cache_access_times[cache_key] = time.time()
            logger.debug(f"Retrieved cached plot data: {cache_key}")
            return self.plot_cache[cache_key]
        return None

    def _evict_oldest_entry(self):
        """Evict the oldest cache entry."""
        if not self.cache_access_times:
            return

        oldest_key = min(
            self.cache_access_times.keys(), key=lambda k: self.cache_access_times[k]
        )

        # Remove from cache
        plot_data = self.plot_cache.pop(oldest_key, None)
        self.cache_access_times.pop(oldest_key, None)

        if plot_data is not None and hasattr(plot_data, "nbytes"):
            size_mb = plot_data.nbytes / (1024 * 1024)
            self.current_cache_size_mb -= size_mb

        logger.debug(f"Evicted oldest plot cache entry: {oldest_key}")


# Global instances
_plot_optimizer = None
_plot_cache_manager = None


def get_plot_optimizer() -> PlotPerformanceOptimizer:
    """Get global plot optimizer instance."""
    global _plot_optimizer
    if _plot_optimizer is None:
        _plot_optimizer = PlotPerformanceOptimizer()
    return _plot_optimizer


def get_plot_cache_manager() -> PlotCacheManager:
    """Get global plot cache manager instance."""
    global _plot_cache_manager
    if _plot_cache_manager is None:
        _plot_cache_manager = PlotCacheManager()
    return _plot_cache_manager


# Convenience functions
def optimize_plot_performance(
    plot_widget,
    data: np.ndarray,
    plot_type: str,
    backend: PlotBackend = PlotBackend.AUTO,
) -> dict[str, Any]:
    """
    Convenience function to optimize plot performance.

    Parameters
    ----------
    plot_widget : object
        Plot widget (PyQtGraph or matplotlib)
    data : np.ndarray
        Data to plot
    plot_type : str
        Type of plot
    backend : PlotBackend
        Plotting backend to optimize for

    Returns
    -------
    Dict[str, Any]
        Applied optimizations
    """
    optimizer = get_plot_optimizer()

    if backend == PlotBackend.AUTO:
        data_size_mb = data.nbytes / (1024 * 1024) if hasattr(data, "nbytes") else 1.0
        complexity = PlotComplexity.MODERATE  # Default assumption
        backend = optimizer.select_optimal_backend(plot_type, data_size_mb, complexity)

    if backend == PlotBackend.PYQTGRAPH:
        return optimizer.optimize_pyqtgraph_plot(plot_widget, data, plot_type)
    if backend == PlotBackend.MATPLOTLIB:
        # Extract figure and axes from widget
        figure = getattr(plot_widget, "figure", None)
        axes = getattr(plot_widget, "axes", None)
        if figure is not None and axes is not None:
            return optimizer.optimize_matplotlib_plot(figure, axes, data, plot_type)

    return {}


class AdvancedGUIRenderer:
    """
    Advanced GUI rendering optimization with frame rate control and adaptive quality.

    This class provides sophisticated rendering optimizations including:
    - Adaptive frame rate limiting
    - Dynamic quality scaling based on performance
    - Intelligent update batching
    - GPU acceleration when available
    """

    def __init__(self, target_fps: int = 60, quality_auto_adjust: bool = True):
        self.target_fps = target_fps
        self.frame_time_target = 1.0 / target_fps
        self.quality_auto_adjust = quality_auto_adjust

        # Performance tracking
        self._frame_times = []
        self._last_frame_time = time.time()
        self._performance_samples = 50

        # Quality settings
        self._current_quality = 1.0  # 1.0 = full quality, 0.5 = half quality, etc.
        self._min_quality = 0.3
        self._max_quality = 1.0

        # Update batching
        self._pending_updates = {}
        self._batch_timer = None
        self._batch_interval_ms = 16  # ~60 FPS

        self.memory_manager = get_memory_manager()
        self.threading_manager = _get_threading_manager()

        logger.info(f"AdvancedGUIRenderer initialized: {target_fps}FPS target")

    @contextmanager
    def optimized_rendering_context(self):
        """Context manager for optimized rendering operations."""
        start_time = time.time()

        try:
            # Apply pre-rendering optimizations
            original_settings = self._apply_rendering_optimizations()
            yield

        finally:
            # Restore settings and update performance metrics
            self._restore_rendering_settings(original_settings)
            self._update_performance_metrics(start_time)

    def _apply_rendering_optimizations(self) -> dict[str, Any]:
        """Apply rendering optimizations and return original settings."""
        original_settings = {}

        if PYQTGRAPH_AVAILABLE:
            # Store original PyQtGraph settings
            original_settings["pyqtgraph"] = {
                "antialias": pg.getConfigOption("antialias"),
                "useOpenGL": pg.getConfigOption("useOpenGL"),
                "enableExperimental": pg.getConfigOption("enableExperimental"),
            }

            # Apply performance optimizations based on current quality
            quality_factor = self._current_quality

            pg.setConfigOptions(
                antialias=quality_factor > 0.7,  # Disable AA below 70% quality
                useOpenGL=True,  # Always use OpenGL for better performance
                enableExperimental=True,  # Enable performance features
            )

            # Reduce update frequency for complex scenes
            if quality_factor < 0.6:
                pg.setConfigOption("crashWarning", False)

        return original_settings

    def _restore_rendering_settings(self, original_settings: dict[str, Any]):
        """Restore original rendering settings."""
        if PYQTGRAPH_AVAILABLE and "pyqtgraph" in original_settings:
            pg.setConfigOptions(**original_settings["pyqtgraph"])

    def _update_performance_metrics(self, start_time: float):
        """Update frame time metrics for adaptive quality."""
        frame_time = time.time() - start_time
        self._frame_times.append(frame_time)

        # Keep only recent samples
        if len(self._frame_times) > self._performance_samples:
            self._frame_times.pop(0)

        # Adaptive quality adjustment
        if self.quality_auto_adjust and len(self._frame_times) >= 10:
            avg_frame_time = sum(self._frame_times[-10:]) / 10
            self._adjust_quality_based_on_performance(avg_frame_time)

    def _adjust_quality_based_on_performance(self, avg_frame_time: float):
        """Adjust rendering quality based on performance metrics."""
        if avg_frame_time > self.frame_time_target * 1.5:
            # Performance is poor, reduce quality
            new_quality = max(self._min_quality, self._current_quality * 0.9)
            if new_quality != self._current_quality:
                self._current_quality = new_quality
                logger.debug(
                    f"Reduced rendering quality to {new_quality:.2f} due to performance"
                )

        elif avg_frame_time < self.frame_time_target * 0.8:
            # Performance is good, can increase quality
            new_quality = min(self._max_quality, self._current_quality * 1.05)
            if new_quality != self._current_quality:
                self._current_quality = new_quality
                logger.debug(f"Increased rendering quality to {new_quality:.2f}")

    def batch_widget_update(
        self, widget_id: str, update_func: callable, *args, **kwargs
    ):
        """
        Batch widget updates to reduce rendering overhead.

        Parameters
        ----------
        widget_id : str
            Unique identifier for the widget
        update_func : callable
            Function to perform the update
        *args, **kwargs
            Arguments for the update function
        """
        self._pending_updates[widget_id] = (update_func, args, kwargs)

        # Start batch timer if not already running
        if self._batch_timer is None and self.threading_manager:
            from ..threading.unified_threading import TaskPriority, TaskType

            def process_batched_updates():
                if self._pending_updates:
                    # Process all pending updates
                    updates_to_process = dict(self._pending_updates)
                    self._pending_updates.clear()

                    for widget_id, (func, args, kwargs) in updates_to_process.items():
                        try:
                            func(*args, **kwargs)
                        except Exception as e:
                            logger.warning(
                                f"Batched update failed for {widget_id}: {e}"
                            )

                self._batch_timer = None

            # Schedule batched update processing
            self.threading_manager.submit_task(
                f"batch_update_{time.time()}",
                process_batched_updates,
                TaskPriority.HIGH,
                TaskType.GUI_UPDATE,
            )

    def optimize_large_dataset_display(
        self, data: np.ndarray, max_points: int = 10000
    ) -> np.ndarray:
        """
        Optimize display of large datasets through intelligent downsampling.

        Parameters
        ----------
        data : np.ndarray
            Input data array
        max_points : int
            Maximum number of points to display

        Returns
        -------
        np.ndarray
            Optimized data for display
        """
        if data.size <= max_points:
            return data

        # Apply different strategies based on data dimensionality
        if len(data.shape) == 1:
            # 1D data: use peak-preserving downsampling
            downsample_factor = max(1, data.size // max_points)
            return self._peak_preserving_downsample(data, downsample_factor)

        if len(data.shape) == 2:
            # 2D data: use area-based downsampling
            total_pixels = data.shape[0] * data.shape[1]
            if total_pixels > max_points:
                reduction_factor = np.sqrt(total_pixels / max_points)
                new_height = int(data.shape[0] / reduction_factor)
                new_width = int(data.shape[1] / reduction_factor)

                try:
                    from scipy import ndimage

                    zoom_y = new_height / data.shape[0]
                    zoom_x = new_width / data.shape[1]
                    return ndimage.zoom(data, (zoom_y, zoom_x), order=1)
                except ImportError:
                    # Fallback to simple slicing
                    step_y = max(1, data.shape[0] // new_height)
                    step_x = max(1, data.shape[1] // new_width)
                    return data[::step_y, ::step_x]

        return data

    def _peak_preserving_downsample(self, data: np.ndarray, factor: int) -> np.ndarray:
        """Downsample 1D data while preserving peaks."""
        if factor <= 1:
            return data

        # Reshape data into chunks and find min/max for each
        n_complete_chunks = len(data) // factor
        reshaped = data[: n_complete_chunks * factor].reshape(-1, factor)

        # For each chunk, keep both min and max to preserve features
        mins = np.min(reshaped, axis=1)
        maxs = np.max(reshaped, axis=1)

        # Interleave mins and maxs, then flatten
        result = np.column_stack((mins, maxs)).flatten()

        # Add remaining data points if any
        remaining = data[n_complete_chunks * factor :]
        if len(remaining) > 0:
            result = np.concatenate([result, remaining])

        return result

    def get_rendering_stats(self) -> dict[str, Any]:
        """Get comprehensive rendering performance statistics."""
        stats = {
            "target_fps": self.target_fps,
            "current_quality": self._current_quality,
            "frame_count": len(self._frame_times),
            "pending_updates": len(self._pending_updates),
        }

        if self._frame_times:
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            current_fps = 1.0 / max(0.001, avg_frame_time)  # Avoid division by zero

            stats.update(
                {
                    "avg_frame_time_ms": avg_frame_time * 1000,
                    "current_fps": current_fps,
                    "frame_time_target_ms": self.frame_time_target * 1000,
                    "performance_ratio": self.frame_time_target / avg_frame_time,
                }
            )

        return stats


# Global instance
_advanced_gui_renderer = None


def get_advanced_gui_renderer() -> AdvancedGUIRenderer:
    """Get global advanced GUI renderer instance."""
    global _advanced_gui_renderer
    if _advanced_gui_renderer is None:
        _advanced_gui_renderer = AdvancedGUIRenderer()
    return _advanced_gui_renderer
