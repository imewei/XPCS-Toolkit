"""I/O boundary adapters for backend array conversions.

This module provides adapters for converting backend arrays at I/O boundaries
where NumPy arrays are required (HDF5, PyQtGraph, Matplotlib).

The adapters centralize conversion logic for:
- Performance monitoring
- Caching optimization opportunities
- Consistent error handling
- Easy debugging and logging

Public API:
    PyQtGraphAdapter: Convert arrays for PyQtGraph visualization
    HDF5Adapter: Convert arrays for HDF5 file I/O
    MatplotlibAdapter: Convert arrays for Matplotlib plotting
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsviewer.utils.logging_config import get_logger

from ._conversions import ensure_numpy

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from ._base import BackendProtocol

logger = get_logger(__name__)


def _log_array_info(prefix: str, array: Any) -> None:
    """Log array shape and dtype at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        shape = getattr(array, "shape", "N/A")
        dtype = getattr(array, "dtype", "N/A")
        logger.debug(f"{prefix}: shape={shape}, dtype={dtype}")


class PyQtGraphAdapter:
    """Adapter for PyQtGraph I/O boundary conversions.

    This adapter ensures backend arrays are properly converted to NumPy
    arrays for PyQtGraph visualization, with optional performance monitoring.

    Attributes
    ----------
    backend : BackendProtocol
        The backend to use for array operations
    enable_monitoring : bool
        Whether to log performance metrics
    """

    def __init__(self, backend: BackendProtocol, enable_monitoring: bool = False):
        """Initialize PyQtGraph adapter.

        Parameters
        ----------
        backend : BackendProtocol
            Backend instance
        enable_monitoring : bool, optional
            Enable performance monitoring, by default False
        """
        self.backend = backend
        self.enable_monitoring = enable_monitoring
        self._conversion_count = 0
        self._total_conversion_time = 0.0

    def to_pyqtgraph(self, array: ArrayLike) -> np.ndarray:
        """Convert backend array to PyQtGraph-compatible NumPy array.

        PyQtGraph requires NumPy arrays for all visualization operations.
        This method ensures the array is converted with proper monitoring.

        Parameters
        ----------
        array : array-like
            Backend array to convert

        Returns
        -------
        np.ndarray
            NumPy array suitable for PyQtGraph
        """
        _log_array_info("to_pyqtgraph input", array)

        if self.enable_monitoring:
            import time

            start_time = time.perf_counter()

        result: np.ndarray = ensure_numpy(array)

        if self.enable_monitoring:
            elapsed = time.perf_counter() - start_time
            self._conversion_count += 1
            self._total_conversion_time += elapsed

            if elapsed > 0.01:  # Log slow conversions (>10ms)
                logger.debug(
                    f"PyQtGraph conversion took {elapsed * 1000:.2f}ms for "
                    f"array shape {result.shape}"
                )

        _log_array_info("to_pyqtgraph output", result)
        return result

    def from_pyqtgraph(self, array: np.ndarray) -> Any:
        """Convert NumPy array from PyQtGraph to backend array.

        Use this when receiving data from PyQtGraph that needs to be
        used in backend computations.

        Parameters
        ----------
        array : np.ndarray
            NumPy array from PyQtGraph

        Returns
        -------
        BackendArray
            Array in backend's native format
        """
        return self.backend.from_numpy(array)

    def get_stats(self) -> dict[str, Any]:
        """Get conversion statistics.

        Returns
        -------
        dict
            Statistics including conversion count and average time
        """
        avg_time = (
            self._total_conversion_time / self._conversion_count
            if self._conversion_count > 0
            else 0.0
        )
        return {
            "conversion_count": self._conversion_count,
            "total_conversion_time_seconds": self._total_conversion_time,
            "average_conversion_time_ms": avg_time * 1000,
        }

    def reset_stats(self) -> None:
        """Reset conversion statistics."""
        self._conversion_count = 0
        self._total_conversion_time = 0.0


class HDF5Adapter:
    """Adapter for HDF5 I/O boundary conversions.

    This adapter ensures backend arrays are properly converted to NumPy
    arrays for HDF5 file operations (h5py requires NumPy arrays).

    Attributes
    ----------
    backend : BackendProtocol
        The backend to use for array operations
    enable_monitoring : bool
        Whether to log performance metrics
    """

    def __init__(self, backend: BackendProtocol, enable_monitoring: bool = False):
        """Initialize HDF5 adapter.

        Parameters
        ----------
        backend : BackendProtocol
            Backend instance
        enable_monitoring : bool, optional
            Enable performance monitoring, by default False
        """
        self.backend = backend
        self.enable_monitoring = enable_monitoring
        self._write_count = 0
        self._read_count = 0
        self._total_write_time = 0.0
        self._total_read_time = 0.0

    def to_hdf5(self, array: ArrayLike) -> np.ndarray:
        """Convert backend array to HDF5-compatible NumPy array.

        h5py requires NumPy arrays for dataset creation and writing.
        This method ensures the array is converted with proper monitoring.

        Parameters
        ----------
        array : array-like
            Backend array to convert

        Returns
        -------
        np.ndarray
            NumPy array suitable for HDF5 writing
        """
        if self.enable_monitoring:
            import time

            start_time = time.perf_counter()

        result: np.ndarray = ensure_numpy(array)

        if self.enable_monitoring:
            elapsed = time.perf_counter() - start_time
            self._write_count += 1
            self._total_write_time += elapsed

            if elapsed > 0.01:  # Log slow conversions (>10ms)
                logger.debug(
                    f"HDF5 write conversion took {elapsed * 1000:.2f}ms for "
                    f"array shape {result.shape}"
                )

        return result

    def from_hdf5(self, array: np.ndarray) -> Any:
        """Convert NumPy array from HDF5 to backend array.

        Use this when reading data from HDF5 files that needs to be
        used in backend computations.

        Parameters
        ----------
        array : np.ndarray
            NumPy array from HDF5

        Returns
        -------
        BackendArray
            Array in backend's native format
        """
        if self.enable_monitoring:
            import time

            start_time = time.perf_counter()

        result: Any = self.backend.from_numpy(array)

        if self.enable_monitoring:
            elapsed = time.perf_counter() - start_time
            self._read_count += 1
            self._total_read_time += elapsed

        return result

    def get_stats(self) -> dict[str, Any]:
        """Get conversion statistics.

        Returns
        -------
        dict
            Statistics including read/write counts and average times
        """
        avg_write_time = (
            self._total_write_time / self._write_count if self._write_count > 0 else 0.0
        )
        avg_read_time = (
            self._total_read_time / self._read_count if self._read_count > 0 else 0.0
        )
        return {
            "write_count": self._write_count,
            "read_count": self._read_count,
            "total_write_time_seconds": self._total_write_time,
            "total_read_time_seconds": self._total_read_time,
            "average_write_time_ms": avg_write_time * 1000,
            "average_read_time_ms": avg_read_time * 1000,
        }

    def reset_stats(self) -> None:
        """Reset conversion statistics."""
        self._write_count = 0
        self._read_count = 0
        self._total_write_time = 0.0
        self._total_read_time = 0.0


class MatplotlibAdapter:
    """Adapter for Matplotlib I/O boundary conversions.

    This adapter ensures backend arrays are properly converted to NumPy
    arrays for Matplotlib plotting operations.

    Attributes
    ----------
    backend : BackendProtocol
        The backend to use for array operations
    enable_monitoring : bool
        Whether to log performance metrics
    """

    def __init__(self, backend: BackendProtocol, enable_monitoring: bool = False):
        """Initialize Matplotlib adapter.

        Parameters
        ----------
        backend : BackendProtocol
            Backend instance
        enable_monitoring : bool, optional
            Enable performance monitoring, by default False
        """
        self.backend = backend
        self.enable_monitoring = enable_monitoring
        self._conversion_count = 0
        self._total_conversion_time = 0.0

    def to_matplotlib(self, *arrays: ArrayLike) -> tuple[np.ndarray, ...] | np.ndarray:
        """Convert backend arrays to Matplotlib-compatible NumPy arrays.

        Matplotlib requires NumPy arrays for all plotting operations.
        This method can convert multiple arrays at once for convenience.

        Parameters
        ----------
        *arrays : array-like
            One or more backend arrays to convert

        Returns
        -------
        np.ndarray or tuple of np.ndarray
            If single array: NumPy array
            If multiple arrays: Tuple of NumPy arrays

        Examples
        --------
        >>> adapter = MatplotlibAdapter(backend)
        >>> x_np, y_np = adapter.to_matplotlib(x, y)
        >>> plt.plot(x_np, y_np)
        """
        if self.enable_monitoring:
            import time

            start_time = time.perf_counter()

        results = tuple(ensure_numpy(arr) for arr in arrays)

        if self.enable_monitoring:
            elapsed = time.perf_counter() - start_time
            self._conversion_count += len(arrays)
            self._total_conversion_time += elapsed

            if elapsed > 0.01:  # Log slow conversions (>10ms)
                shapes = [arr.shape for arr in results]
                logger.debug(
                    f"Matplotlib conversion took {elapsed * 1000:.2f}ms for "
                    f"{len(arrays)} arrays with shapes {shapes}"
                )

        # Return single array if only one was provided (convenience)
        return results[0] if len(results) == 1 else results

    def get_stats(self) -> dict[str, Any]:
        """Get conversion statistics.

        Returns
        -------
        dict
            Statistics including conversion count and average time
        """
        avg_time = (
            self._total_conversion_time / self._conversion_count
            if self._conversion_count > 0
            else 0.0
        )
        return {
            "conversion_count": self._conversion_count,
            "total_conversion_time_seconds": self._total_conversion_time,
            "average_conversion_time_ms": avg_time * 1000,
        }

    def reset_stats(self) -> None:
        """Reset conversion statistics."""
        self._conversion_count = 0
        self._total_conversion_time = 0.0


def create_adapters(
    backend: BackendProtocol | None = None, enable_monitoring: bool = False
) -> tuple[PyQtGraphAdapter, HDF5Adapter, MatplotlibAdapter]:
    """Create all I/O adapters for a given backend.

    Convenience function to create all adapters at once.

    Parameters
    ----------
    backend : BackendProtocol, optional
        Backend instance. If None, uses current backend.
    enable_monitoring : bool, optional
        Enable performance monitoring for all adapters, by default False

    Returns
    -------
    tuple
        (PyQtGraphAdapter, HDF5Adapter, MatplotlibAdapter)

    Examples
    --------
    >>> from xpcsviewer.backends import get_backend
    >>> from xpcsviewer.backends.io_adapter import create_adapters
    >>> backend = get_backend()
    >>> pyqt_adapter, hdf5_adapter, mpl_adapter = create_adapters(backend)
    """
    if backend is None:
        from . import get_backend

        backend = get_backend()

    return (
        PyQtGraphAdapter(backend, enable_monitoring),
        HDF5Adapter(backend, enable_monitoring),
        MatplotlibAdapter(backend, enable_monitoring),
    )
