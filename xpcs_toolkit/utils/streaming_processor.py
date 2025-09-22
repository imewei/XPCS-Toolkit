"""
Streaming Data Processing for XPCS Toolkit

This module provides memory-efficient streaming processing for large XPCS datasets,
particularly for operations like logarithmic transformations of SAXS data.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


@dataclass
class ChunkInfo:
    """Information about a data chunk."""

    index: int
    slice_obj: tuple[slice, ...]
    shape: tuple[int, ...]
    estimated_size_mb: float
    total_chunks: int


class StreamingProcessor(ABC):
    """Abstract base class for streaming data processors."""

    def __init__(self, chunk_size_mb: float = 50.0):
        self.chunk_size_mb = chunk_size_mb
        self.memory_manager = get_memory_manager()

    @abstractmethod
    def process_chunk(self, chunk: np.ndarray, chunk_info: ChunkInfo) -> np.ndarray:
        """Process a single data chunk."""

    @abstractmethod
    def combine_chunks(
        self, processed_chunks: list, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Combine processed chunks into final result."""

    def process_array_streaming(
        self, data: np.ndarray, output_dtype: np.dtype | None = None
    ) -> np.ndarray:
        """
        Process array data using streaming to reduce memory footprint.

        Parameters
        ----------
        data : np.ndarray
            Input array to process
        output_dtype : np.dtype, optional
            Output data type for memory optimization

        Returns
        -------
        np.ndarray
            Processed array
        """
        if output_dtype is None:
            output_dtype = data.dtype

        # Calculate optimal chunk size based on memory constraints
        chunk_slices = self._calculate_chunk_slices(data.shape, data.dtype)
        total_chunks = len(chunk_slices)

        logger.info(
            f"Processing array with streaming: {total_chunks} chunks of ~{self.chunk_size_mb:.1f}MB each"
        )

        processed_chunks = []
        memory_before = self.memory_manager.get_cache_stats()["current_memory_mb"]

        for i, slice_obj in enumerate(chunk_slices):
            # Monitor memory pressure before each chunk
            pressure = self.memory_manager.get_memory_pressure()
            if pressure == MemoryPressure.CRITICAL:
                logger.warning(
                    "Critical memory pressure during streaming, triggering cleanup"
                )
                self.memory_manager._emergency_cleanup()

            # Create chunk info
            chunk_shape = tuple(
                slice_obj[dim].stop - slice_obj[dim].start
                for dim in range(len(slice_obj))
            )
            chunk_info = ChunkInfo(
                index=i,
                slice_obj=slice_obj,
                shape=chunk_shape,
                estimated_size_mb=np.prod(chunk_shape)
                * np.dtype(data.dtype).itemsize
                / (1024 * 1024),
                total_chunks=total_chunks,
            )

            # Extract and process chunk
            chunk = data[slice_obj]
            processed_chunk = self.process_chunk(chunk, chunk_info)
            processed_chunks.append((slice_obj, processed_chunk))

            # Progress reporting
            if i % max(1, total_chunks // 10) == 0:
                progress = (i + 1) / total_chunks * 100
                logger.debug(
                    f"Streaming progress: {progress:.1f}% ({i + 1}/{total_chunks})"
                )

        # Combine chunks into final result
        result = self.combine_chunks(processed_chunks, data.shape)

        memory_after = self.memory_manager.get_cache_stats()["current_memory_mb"]
        memory_used = memory_after - memory_before
        logger.info(
            f"Streaming processing completed, memory used: {memory_used:+.1f}MB"
        )

        return result

    def _calculate_chunk_slices(self, shape: tuple[int, ...], dtype: np.dtype) -> list:
        """Calculate optimal chunk slices for streaming processing."""
        itemsize = dtype.itemsize
        total_size_mb = np.prod(shape) * itemsize / (1024 * 1024)

        # If data is small enough, process as single chunk
        if total_size_mb <= self.chunk_size_mb:
            return [tuple(slice(0, dim_size) for dim_size in shape)]

        # For large data, calculate chunks along the first dimension
        elements_per_chunk = int(self.chunk_size_mb * 1024 * 1024 / itemsize)

        # Calculate elements per row for multi-dimensional arrays
        if len(shape) > 1:
            elements_per_row = np.prod(shape[1:])
            rows_per_chunk = max(1, elements_per_chunk // elements_per_row)
        else:
            rows_per_chunk = elements_per_chunk

        chunk_slices = []
        for start_row in range(0, shape[0], rows_per_chunk):
            end_row = min(start_row + rows_per_chunk, shape[0])

            # Create slice tuple
            slice_obj = tuple(
                [
                    slice(start_row, end_row) if dim == 0 else slice(0, shape[dim])
                    for dim in range(len(shape))
                ]
            )
            chunk_slices.append(slice_obj)

        return chunk_slices


class SAXSLogProcessor(StreamingProcessor):
    """Streaming processor for SAXS logarithmic transformations."""

    def __init__(self, chunk_size_mb: float = 50.0, epsilon: float = 1e-10):
        super().__init__(chunk_size_mb)
        self.epsilon = epsilon

    def process_chunk(self, chunk: np.ndarray, chunk_info: ChunkInfo) -> np.ndarray:
        """
        Process SAXS data chunk with logarithmic transformation.

        Parameters
        ----------
        chunk : np.ndarray
            Input SAXS data chunk
        chunk_info : ChunkInfo
            Information about the chunk

        Returns
        -------
        np.ndarray
            Log-transformed chunk
        """
        # Create a copy to avoid modifying input
        saxs_chunk = chunk.copy()

        # Apply log transformation with safe handling of non-positive values
        positive_mask = saxs_chunk > 0

        if np.any(positive_mask):
            # Find minimum positive value for replacement
            min_positive = np.min(saxs_chunk[positive_mask])
            replacement_value = max(min_positive, self.epsilon)
        else:
            replacement_value = self.epsilon

        # Replace non-positive values
        saxs_chunk[~positive_mask] = replacement_value

        # Apply log10 transformation
        log_chunk = np.log10(saxs_chunk).astype(np.float32)

        logger.debug(
            f"Processed chunk {chunk_info.index + 1}/{chunk_info.total_chunks}: "
            f"{chunk_info.shape} -> log transform"
        )

        return log_chunk

    def combine_chunks(
        self, processed_chunks: list, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Combine processed log chunks into final result array.

        Parameters
        ----------
        processed_chunks : list
            List of (slice_obj, processed_chunk) tuples
        original_shape : tuple[int, ...]
            Shape of original array

        Returns
        -------
        np.ndarray
            Combined log-transformed array
        """
        # Create output array
        result = np.zeros(original_shape, dtype=np.float32)

        # Fill in processed chunks
        for slice_obj, processed_chunk in processed_chunks:
            result[slice_obj] = processed_chunk

        return result


class ROIProcessor(StreamingProcessor):
    """Streaming processor for ROI (Region of Interest) calculations."""

    def __init__(self, roi_mask: np.ndarray, chunk_size_mb: float = 50.0):
        super().__init__(chunk_size_mb)
        self.roi_mask = roi_mask

    def process_chunk(self, chunk: np.ndarray, chunk_info: ChunkInfo) -> dict:
        """
        Process chunk for ROI calculations.

        Parameters
        ----------
        chunk : np.ndarray
            Input data chunk
        chunk_info : ChunkInfo
            Information about the chunk

        Returns
        -------
        dict
            ROI statistics for the chunk
        """
        # Apply ROI mask to chunk
        roi_slice = self.roi_mask[chunk_info.slice_obj[1:]]  # Skip time dimension
        roi_data = chunk * roi_slice[np.newaxis, ...]  # Broadcast for time dimension

        # Calculate ROI statistics
        roi_sum = np.sum(roi_data, axis=(1, 2))  # Sum over spatial dimensions
        roi_mean = np.mean(roi_data, axis=(1, 2))
        roi_std = np.std(roi_data, axis=(1, 2))

        return {
            "sum": roi_sum,
            "mean": roi_mean,
            "std": roi_std,
            "slice_obj": chunk_info.slice_obj,
        }

    def combine_chunks(
        self, processed_chunks: list, original_shape: tuple[int, ...]
    ) -> dict:
        """
        Combine ROI chunk results into final statistics.

        Parameters
        ----------
        processed_chunks : list
            List of (slice_obj, roi_stats) tuples
        original_shape : tuple[int, ...]
            Shape of original array

        Returns
        -------
        dict
            Combined ROI statistics
        """
        # Initialize result arrays
        total_time_points = original_shape[0]
        roi_sum = np.zeros(total_time_points)
        roi_mean = np.zeros(total_time_points)
        roi_std = np.zeros(total_time_points)

        # Combine chunk results
        for slice_obj, roi_stats in processed_chunks:
            time_slice = slice_obj[0]
            roi_sum[time_slice] = roi_stats["sum"]
            roi_mean[time_slice] = roi_stats["mean"]
            roi_std[time_slice] = roi_stats["std"]

        return {"roi_sum": roi_sum, "roi_mean": roi_mean, "roi_std": roi_std}


class AdaptiveChunkSizer:
    """
    Adaptive chunk sizing based on memory pressure and system performance.
    """

    def __init__(self, base_chunk_size_mb: float = 50.0):
        self.base_chunk_size_mb = base_chunk_size_mb
        self.memory_manager = get_memory_manager()
        self.performance_history = []

    def get_optimal_chunk_size(
        self, data_shape: tuple[int, ...], data_dtype: np.dtype
    ) -> float:
        """
        Calculate optimal chunk size based on current conditions.

        Parameters
        ----------
        data_shape : tuple[int, ...]
            Shape of data to be processed
        data_dtype : np.dtype
            Data type of array

        Returns
        -------
        float
            Optimal chunk size in MB
        """
        # Base chunk size
        chunk_size = self.base_chunk_size_mb

        # Adjust based on memory pressure
        pressure = self.memory_manager.get_memory_pressure()

        if pressure == MemoryPressure.CRITICAL:
            chunk_size *= 0.25  # Very small chunks
        elif pressure == MemoryPressure.HIGH:
            chunk_size *= 0.5  # Small chunks
        elif pressure == MemoryPressure.MODERATE:
            chunk_size *= 0.75  # Moderately small chunks
        # LOW pressure uses base size

        # Adjust based on total data size
        total_size_mb = np.prod(data_shape) * data_dtype.itemsize / (1024 * 1024)

        if total_size_mb < 100:  # Small dataset
            chunk_size = min(chunk_size, total_size_mb)  # Don't over-chunk
        elif total_size_mb > 1000:  # Very large dataset
            chunk_size = min(chunk_size, 100.0)  # Cap chunk size

        # Ensure minimum chunk size
        chunk_size = max(chunk_size, 10.0)

        logger.debug(
            f"Adaptive chunk size: {chunk_size:.1f}MB "
            f"(pressure: {pressure.value}, data: {total_size_mb:.1f}MB)"
        )

        return chunk_size


def process_saxs_log_streaming(
    data: np.ndarray, chunk_size_mb: float | None = None, epsilon: float = 1e-10
) -> np.ndarray:
    """
    Convenience function for streaming SAXS log processing.

    Parameters
    ----------
    data : np.ndarray
        Input SAXS data
    chunk_size_mb : float, optional
        Chunk size in MB (auto-calculated if None)
    epsilon : float
        Small value for handling non-positive data

    Returns
    -------
    np.ndarray
        Log-transformed SAXS data
    """
    if chunk_size_mb is None:
        # Use adaptive chunk sizing
        sizer = AdaptiveChunkSizer()
        chunk_size_mb = sizer.get_optimal_chunk_size(data.shape, data.dtype)

    processor = SAXSLogProcessor(chunk_size_mb=chunk_size_mb, epsilon=epsilon)
    return processor.process_array_streaming(data)


def process_roi_streaming(
    data: np.ndarray, roi_mask: np.ndarray, chunk_size_mb: float | None = None
) -> dict:
    """
    Convenience function for streaming ROI processing.

    Parameters
    ----------
    data : np.ndarray
        Input data array
    roi_mask : np.ndarray
        ROI mask array
    chunk_size_mb : float, optional
        Chunk size in MB (auto-calculated if None)

    Returns
    -------
    dict
        ROI statistics
    """
    if chunk_size_mb is None:
        # Use adaptive chunk sizing
        sizer = AdaptiveChunkSizer()
        chunk_size_mb = sizer.get_optimal_chunk_size(data.shape, data.dtype)

    processor = ROIProcessor(roi_mask=roi_mask, chunk_size_mb=chunk_size_mb)

    # Calculate optimal chunk slices
    chunk_slices = processor._calculate_chunk_slices(data.shape, data.dtype)

    processed_chunks = []
    for i, slice_obj in enumerate(chunk_slices):
        chunk_shape = tuple(
            slice_obj[dim].stop - slice_obj[dim].start for dim in range(len(slice_obj))
        )
        chunk_info = ChunkInfo(
            index=i,
            slice_obj=slice_obj,
            shape=chunk_shape,
            estimated_size_mb=np.prod(chunk_shape)
            * data.dtype.itemsize
            / (1024 * 1024),
            total_chunks=len(chunk_slices),
        )

        chunk = data[slice_obj]
        roi_stats = processor.process_chunk(chunk, chunk_info)
        processed_chunks.append((slice_obj, roi_stats))

    return processor.combine_chunks(processed_chunks, data.shape)


class MemoryEfficientIterator:
    """
    Memory-efficient iterator for large arrays with automatic cleanup.
    """

    def __init__(self, data: np.ndarray, chunk_size_mb: float = 50.0):
        self.data = data
        self.chunk_size_mb = chunk_size_mb
        self.memory_manager = get_memory_manager()

        # Calculate chunk slices
        processor = StreamingProcessor(chunk_size_mb)
        self.chunk_slices = processor._calculate_chunk_slices(data.shape, data.dtype)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= len(self.chunk_slices):
            raise StopIteration

        # Check memory pressure before yielding chunk
        pressure = self.memory_manager.get_memory_pressure()
        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning("Memory pressure during iteration, triggering cleanup")
            self.memory_manager._aggressive_cleanup()

        slice_obj = self.chunk_slices[self.current_index]
        chunk = self.data[slice_obj]

        chunk_info = ChunkInfo(
            index=self.current_index,
            slice_obj=slice_obj,
            shape=chunk.shape,
            estimated_size_mb=chunk.nbytes / (1024 * 1024),
            total_chunks=len(self.chunk_slices),
        )

        self.current_index += 1
        return chunk, chunk_info

    def __len__(self):
        return len(self.chunk_slices)
