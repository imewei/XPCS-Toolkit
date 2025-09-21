"""
Visualization Performance Optimization for XPCS Toolkit

This module addresses critical performance bottlenecks in the visualization layer
that were identified during comprehensive analysis, including PyQtGraph optimization,
matplotlib performance improvements, and intelligent plot handler selection.
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import psutil

try:
    import pyqtgraph as pg
    from pyqtgraph import ImageView, PlotWidget
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from .logging_config import get_logger
from .memory_manager import get_memory_manager, MemoryPressure
from .memory_predictor import predict_operation_memory

logger = get_logger(__name__)


class PlotBackend(Enum):
    """Available plotting backends."""
    PYQTGRAPH = "pyqtgraph"
    MATPLOTLIB = "matplotlib"
    AUTO = "auto"


class PlotComplexity(Enum):
    """Plot complexity levels for optimization decisions."""
    SIMPLE = "simple"          # Single image/plot
    MODERATE = "moderate"      # Multiple series, ROIs
    COMPLEX = "complex"        # Many series, complex interactions
    VERY_COMPLEX = "very_complex"  # Real-time updates, large datasets


@contextmanager
def optimized_pyqtgraph_context():
    """Context manager for optimized PyQtGraph settings."""
    if not PYQTGRAPH_AVAILABLE:
        yield
        return

    # Store original settings
    original_settings = {
        'useOpenGL': pg.getConfigOption('useOpenGL'),
        'enableExperimental': pg.getConfigOption('enableExperimental'),
        'crashWarning': pg.getConfigOption('crashWarning'),
        'imageAxisOrder': pg.getConfigOption('imageAxisOrder')
    }

    try:
        # Apply optimized settings
        pg.setConfigOptions(
            useOpenGL=True,  # Enable OpenGL acceleration
            enableExperimental=True,  # Enable performance features
            crashWarning=False,  # Reduce warning overhead
            imageAxisOrder='row-major',  # Optimize for NumPy
            antialias=False  # Disable antialiasing for large datasets
        )

        logger.debug("Applied optimized PyQtGraph settings")
        yield

    finally:
        # Restore original settings
        pg.setConfigOptions(**original_settings)
        logger.debug("Restored original PyQtGraph settings")


class ImageDisplayOptimizer:
    """Optimizes large image display performance."""

    def __init__(self, max_display_size: Tuple[int, int] = (2048, 2048)):
        self.max_display_size = max_display_size
        self.memory_manager = get_memory_manager()

    def optimize_image_for_display(self, image: np.ndarray,
                                  target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Optimize image for display performance.

        Parameters
        ----------
        image : np.ndarray
            Input image array
        target_size : Tuple[int, int], optional
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
        logger.info(f"Downsampling image from {image.shape} by factor {downsample_factor:.1f}")

        # Use area-based downsampling for better quality
        new_height = int(height / downsample_factor)
        new_width = int(width / downsample_factor)

        if len(image.shape) == 2:
            # 2D image
            downsampled = self._downsample_2d(image, (new_height, new_width))
        elif len(image.shape) == 3:
            # 3D image (e.g., time series)
            downsampled = np.zeros((image.shape[0], new_height, new_width), dtype=image.dtype)
            for t in range(image.shape[0]):
                downsampled[t] = self._downsample_2d(image[t], (new_height, new_width))
        else:
            logger.warning(f"Unsupported image dimensionality: {len(image.shape)}")
            return image

        memory_saved = (image.nbytes - downsampled.nbytes) / (1024 * 1024)
        logger.debug(f"Image downsampling saved {memory_saved:.1f}MB memory")

        return downsampled

    def _downsample_2d(self, image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
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

    def select_optimal_backend(self, plot_type: str, data_size_mb: float,
                              complexity: PlotComplexity) -> PlotBackend:
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
        if plot_type in ['image', 'saxs_2d']:
            if data_size_mb > 50 or memory_pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
                # Large images: use PyQtGraph with optimization
                return PlotBackend.PYQTGRAPH
            else:
                # Smaller images: either backend works
                return PlotBackend.PYQTGRAPH

        elif plot_type in ['line', 'scatter', 'g2']:
            if complexity in [PlotComplexity.COMPLEX, PlotComplexity.VERY_COMPLEX]:
                # Complex plots with many series: PyQtGraph for performance
                return PlotBackend.PYQTGRAPH
            elif data_size_mb < 10:
                # Small datasets: matplotlib for quality
                return PlotBackend.MATPLOTLIB
            else:
                # Medium datasets: PyQtGraph for speed
                return PlotBackend.PYQTGRAPH

        else:
            # Default to PyQtGraph for unknown types
            return PlotBackend.PYQTGRAPH

    def optimize_pyqtgraph_plot(self, plot_widget, data: np.ndarray,
                               plot_type: str) -> Dict[str, Any]:
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
        Dict[str, Any]
            Optimization settings applied
        """
        optimizations = {}

        if plot_type == 'image':
            # Image-specific optimizations
            if hasattr(plot_widget, 'getImageItem'):
                image_item = plot_widget.getImageItem()
                if image_item is not None:
                    # Enable OpenGL if available
                    if hasattr(image_item, 'setOpts'):
                        image_item.setOpts(useOpenGL=True)
                        optimizations['opengl_enabled'] = True

            # Optimize image data
            if data.size > 1e6:  # > 1M pixels
                optimized_data = self.image_optimizer.optimize_image_for_display(data)
                optimizations['image_downsampled'] = data.shape != optimized_data.shape
                optimizations['original_size'] = data.shape
                optimizations['optimized_size'] = optimized_data.shape
                data = optimized_data

        elif plot_type in ['line', 'scatter']:
            # Line/scatter plot optimizations
            if hasattr(plot_widget, 'setDownsampling'):
                # Enable automatic downsampling for large datasets
                if data.size > 1e5:  # > 100k points
                    plot_widget.setDownsampling(auto=True, mode='peak')
                    optimizations['downsampling_enabled'] = True

            # Disable antialiasing for large datasets
            if data.size > 5e4:  # > 50k points
                if hasattr(plot_widget, 'setAntialiasing'):
                    plot_widget.setAntialiasing(False)
                    optimizations['antialiasing_disabled'] = True

        return optimizations

    def optimize_matplotlib_plot(self, figure, axes, data: np.ndarray,
                                plot_type: str) -> Dict[str, Any]:
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
        Dict[str, Any]
            Optimization settings applied
        """
        optimizations = {}

        # General matplotlib optimizations
        figure.set_facecolor('white')  # Faster rendering
        optimizations['background_optimized'] = True

        if plot_type == 'image':
            # Image-specific optimizations
            if data.size > 1e6:  # > 1M pixels
                optimized_data = self.image_optimizer.optimize_image_for_display(data)
                optimizations['image_downsampled'] = data.shape != optimized_data.shape
                data = optimized_data

            # Use faster interpolation
            if hasattr(axes, 'imshow'):
                optimizations['interpolation'] = 'nearest'

        elif plot_type in ['line', 'scatter']:
            # Line/scatter optimizations
            if data.size > 1e5:  # > 100k points
                # Downsample data for matplotlib
                downsample_factor = int(np.ceil(data.size / 1e4))  # Target ~10k points
                if len(data.shape) == 1:
                    data = data[::downsample_factor]
                elif len(data.shape) == 2:
                    data = data[::downsample_factor, :]
                optimizations['data_downsampled'] = True
                optimizations['downsample_factor'] = downsample_factor

        return optimizations


class ProgressiveImageLoader:
    """Loads large images progressively to maintain responsiveness."""

    def __init__(self, chunk_size_mb: float = 50.0):
        self.chunk_size_mb = chunk_size_mb

    def load_image_progressive(self, image_loader_func,
                              progress_callback: Optional[callable] = None) -> np.ndarray:
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
            if hasattr(image_data, 'shape') and len(image_data.shape) > 2:
                # Multi-dimensional data - may need chunked processing
                pass

            if progress_callback:
                progress_callback(100, 100, "Image load completed")

            load_time = time.time() - start_time
            logger.debug(f"Progressive image load completed in {load_time:.2f}s")

            return image_data

        except Exception as e:
            if progress_callback:
                progress_callback(0, 100, f"Image load failed: {str(e)}")
            raise


class PlotCacheManager:
    """Manages caching of plot data and rendered plots."""

    def __init__(self, max_cache_size_mb: float = 200.0):
        self.max_cache_size_mb = max_cache_size_mb
        self.plot_cache = {}
        self.cache_access_times = {}
        self.current_cache_size_mb = 0.0

    def get_plot_cache_key(self, data_hash: str, plot_params: Dict[str, Any]) -> str:
        """Generate cache key for plot data."""
        # Create a hash from plot parameters
        param_str = str(sorted(plot_params.items()))
        param_hash = hash(param_str)
        return f"{data_hash}_{param_hash}"

    def cache_plot_data(self, cache_key: str, plot_data: Any):
        """Cache processed plot data."""
        try:
            # Estimate size
            if hasattr(plot_data, 'nbytes'):
                size_mb = plot_data.nbytes / (1024 * 1024)
            else:
                size_mb = 1.0  # Default estimate

            # Check if we need to evict old entries
            while self.current_cache_size_mb + size_mb > self.max_cache_size_mb and self.plot_cache:
                self._evict_oldest_entry()

            # Cache the data
            self.plot_cache[cache_key] = plot_data
            self.cache_access_times[cache_key] = time.time()
            self.current_cache_size_mb += size_mb

            logger.debug(f"Cached plot data: {cache_key} ({size_mb:.1f}MB)")

        except Exception as e:
            logger.warning(f"Failed to cache plot data: {e}")

    def get_cached_plot_data(self, cache_key: str) -> Optional[Any]:
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

        oldest_key = min(self.cache_access_times.keys(),
                        key=lambda k: self.cache_access_times[k])

        # Remove from cache
        plot_data = self.plot_cache.pop(oldest_key, None)
        self.cache_access_times.pop(oldest_key, None)

        if plot_data is not None and hasattr(plot_data, 'nbytes'):
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
def optimize_plot_performance(plot_widget, data: np.ndarray, plot_type: str,
                             backend: PlotBackend = PlotBackend.AUTO) -> Dict[str, Any]:
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
        data_size_mb = data.nbytes / (1024 * 1024) if hasattr(data, 'nbytes') else 1.0
        complexity = PlotComplexity.MODERATE  # Default assumption
        backend = optimizer.select_optimal_backend(plot_type, data_size_mb, complexity)

    if backend == PlotBackend.PYQTGRAPH:
        return optimizer.optimize_pyqtgraph_plot(plot_widget, data, plot_type)
    elif backend == PlotBackend.MATPLOTLIB:
        # Extract figure and axes from widget
        figure = getattr(plot_widget, 'figure', None)
        axes = getattr(plot_widget, 'axes', None)
        if figure is not None and axes is not None:
            return optimizer.optimize_matplotlib_plot(figure, axes, data, plot_type)

    return {}