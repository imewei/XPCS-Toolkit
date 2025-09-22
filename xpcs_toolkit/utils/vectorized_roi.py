"""
Vectorized ROI (Region of Interest) Calculations for XPCS Toolkit

This module provides highly optimized vectorized ROI calculations with advanced
memory management and parallel processing capabilities for XPCS data analysis.
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .logging_config import get_logger
from .memory_manager import MemoryPressure, get_memory_manager

logger = get_logger(__name__)


class ROIType(Enum):
    """Types of ROI calculations."""

    PIE = "pie"
    RING = "ring"
    PHI_RING = "phi_ring"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"


@dataclass
class ROIParameters:
    """Parameters for ROI calculation."""

    roi_type: ROIType
    parameters: dict[str, Any]
    label: str = ""


@dataclass
class ROIResult:
    """Result of ROI calculation."""

    x_values: np.ndarray
    roi_data: np.ndarray
    roi_type: ROIType
    parameters: dict[str, Any]
    metadata: dict[str, Any]
    processing_time: float
    memory_used_mb: float


class VectorizedROICalculator(ABC):
    """Abstract base class for vectorized ROI calculators."""

    def __init__(self, chunk_size_mb: float = 100.0):
        self.chunk_size_mb = chunk_size_mb
        self.memory_manager = get_memory_manager()

    @abstractmethod
    def calculate_roi_mask(
        self, geometry_data: dict[str, np.ndarray], roi_params: ROIParameters
    ) -> np.ndarray:
        """Calculate the ROI mask for the given parameters."""

    @abstractmethod
    def process_roi_data(
        self, saxs_data: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> ROIResult:
        """Process SAXS data with the ROI mask."""

    def calculate_roi(
        self,
        saxs_data: np.ndarray,
        geometry_data: dict[str, np.ndarray],
        roi_params: ROIParameters,
        use_streaming: bool | None = None,
    ) -> ROIResult:
        """
        Main ROI calculation method with automatic streaming detection.

        Parameters
        ----------
        saxs_data : np.ndarray
            SAXS data array
        geometry_data : Dict[str, np.ndarray]
            Geometry arrays (qmap, rmap, pmap, mask, etc.)
        roi_params : ROIParameters
            ROI calculation parameters
        use_streaming : bool, optional
            Whether to use streaming processing (auto-detected if None)

        Returns
        -------
        ROIResult
            ROI calculation result
        """
        start_time = time.time()
        memory_before_mb = self.memory_manager.get_cache_stats()["current_memory_mb"]

        # Calculate ROI mask
        roi_mask = self.calculate_roi_mask(geometry_data, roi_params)

        # Determine if streaming is needed
        if use_streaming is None:
            data_size_mb = saxs_data.nbytes / (1024 * 1024)
            use_streaming = data_size_mb > self.chunk_size_mb * 2

        if use_streaming:
            logger.info(
                f"Using streaming ROI calculation for {roi_params.roi_type.value}"
            )
            result = self._process_roi_streaming(saxs_data, roi_mask, roi_params)
        else:
            logger.debug(
                f"Using vectorized ROI calculation for {roi_params.roi_type.value}"
            )
            result = self.process_roi_data(saxs_data, roi_mask, roi_params)

        # Add timing and memory usage
        processing_time = time.time() - start_time
        memory_after_mb = self.memory_manager.get_cache_stats()["current_memory_mb"]
        result.processing_time = processing_time
        result.memory_used_mb = memory_after_mb - memory_before_mb

        logger.debug(
            f"ROI calculation completed: {processing_time:.3f}s, "
            f"{result.memory_used_mb:+.1f}MB memory"
        )

        return result

    def _process_roi_streaming(
        self, saxs_data: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> ROIResult:
        """Process ROI calculation using streaming for memory efficiency."""
        from .streaming_processor import MemoryEfficientIterator

        # Create streaming iterator for SAXS data
        iterator = MemoryEfficientIterator(saxs_data, chunk_size_mb=self.chunk_size_mb)

        # Initialize accumulation arrays
        if roi_params.roi_type == ROIType.PIE:
            # For pie ROI, we accumulate binned q-values
            qsize = roi_params.parameters.get("qsize", 1000)
            accumulated_data = np.zeros(qsize)
            normalization = np.zeros(qsize)

        elif roi_params.roi_type == ROIType.RING:
            # For ring ROI, we accumulate angular values
            phi_num = roi_params.parameters.get("phi_num", 180)
            accumulated_data = np.zeros(phi_num)
            normalization = np.zeros(phi_num)

        else:
            # Generic accumulation
            accumulated_data = 0.0
            normalization = 0

        # Process chunks
        for chunk, chunk_info in iterator:
            # Extract roi mask for this chunk
            chunk_mask_slice = tuple(slice(0, dim_size) for dim_size in chunk.shape[1:])
            chunk_roi_mask = roi_mask[chunk_mask_slice]

            # Process this chunk
            chunk_result = self._process_chunk_vectorized(
                chunk, chunk_roi_mask, roi_params, chunk_info
            )

            # Accumulate results
            if isinstance(accumulated_data, np.ndarray):
                accumulated_data += chunk_result["data"]
                normalization += chunk_result["norm"]
            else:
                accumulated_data += chunk_result["data"]
                normalization += chunk_result["norm"]

        # Finalize results
        if isinstance(accumulated_data, np.ndarray) and normalization.sum() > 0:
            final_data = accumulated_data / np.maximum(normalization, 1e-10)
        else:
            final_data = accumulated_data / max(normalization, 1e-10)

        # Generate x-values based on ROI type
        x_values = self._generate_x_values(roi_params)

        return ROIResult(
            x_values=x_values,
            roi_data=final_data,
            roi_type=roi_params.roi_type,
            parameters=roi_params.parameters,
            metadata={"streaming_used": True, "chunks_processed": len(iterator)},
            processing_time=0.0,  # Will be set by caller
            memory_used_mb=0.0,  # Will be set by caller
        )

    def _process_chunk_vectorized(
        self,
        chunk: np.ndarray,
        roi_mask: np.ndarray,
        roi_params: ROIParameters,
        chunk_info,
    ) -> dict[str, np.ndarray]:
        """Process a single chunk with vectorized operations."""
        # Apply ROI mask using broadcasting
        masked_chunk = chunk * roi_mask[np.newaxis, ...]

        if roi_params.roi_type == ROIType.PIE:
            return self._process_pie_chunk(masked_chunk, roi_mask, roi_params)
        if roi_params.roi_type == ROIType.RING:
            return self._process_ring_chunk(masked_chunk, roi_mask, roi_params)
        # Generic processing
        data = np.sum(masked_chunk)
        norm = np.sum(roi_mask)
        return {"data": data, "norm": norm}

    def _process_pie_chunk(
        self, masked_chunk: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> dict[str, np.ndarray]:
        """Process pie ROI chunk with q-value binning."""
        qmap_idx = roi_params.parameters.get("qmap_idx")
        qsize = roi_params.parameters.get("qsize", 1000)

        if qmap_idx is None:
            # Fallback to simple accumulation
            return {"data": np.sum(masked_chunk), "norm": np.sum(roi_mask)}

        # Vectorized binning using np.bincount
        flat_qmap = np.where(roi_mask, qmap_idx, 0).ravel()
        masked_chunk.ravel()

        # Process each time frame
        chunk_data = np.zeros(qsize)
        chunk_norm = np.zeros(qsize)

        for t in range(masked_chunk.shape[0]):
            frame_data = masked_chunk[t].ravel()
            binned_data = np.bincount(flat_qmap, frame_data, minlength=qsize + 1)[1:]
            binned_norm = np.bincount(
                flat_qmap, roi_mask.ravel().astype(float), minlength=qsize + 1
            )[1:]

            chunk_data += binned_data
            chunk_norm += binned_norm

        return {"data": chunk_data, "norm": chunk_norm}

    def _process_ring_chunk(
        self, masked_chunk: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> dict[str, np.ndarray]:
        """Process ring ROI chunk with angular binning."""
        phi_idx = roi_params.parameters.get("phi_idx")
        phi_num = roi_params.parameters.get("phi_num", 180)

        if phi_idx is None:
            return {"data": np.sum(masked_chunk), "norm": np.sum(roi_mask)}

        # Vectorized angular binning
        flat_phi = np.where(roi_mask, phi_idx, 0).ravel()

        chunk_data = np.zeros(phi_num)
        chunk_norm = np.zeros(phi_num)

        for t in range(masked_chunk.shape[0]):
            frame_data = masked_chunk[t].ravel()
            binned_data = np.bincount(flat_phi, frame_data, minlength=phi_num + 1)[1:]
            binned_norm = np.bincount(
                flat_phi, roi_mask.ravel().astype(float), minlength=phi_num + 1
            )[1:]

            chunk_data += binned_data
            chunk_norm += binned_norm

        return {"data": chunk_data, "norm": chunk_norm}

    def _generate_x_values(self, roi_params: ROIParameters) -> np.ndarray:
        """Generate x-axis values based on ROI type."""
        if roi_params.roi_type == ROIType.PIE:
            # Q-values for pie ROI
            qmin = roi_params.parameters.get("qmin", 0.0)
            qmax = roi_params.parameters.get("qmax", 1.0)
            qsize = roi_params.parameters.get("qsize", 1000)
            return np.linspace(qmin, qmax, qsize)

        if roi_params.roi_type == ROIType.RING:
            # Angular values for ring ROI
            phi_min = roi_params.parameters.get("phi_min", 0.0)
            phi_max = roi_params.parameters.get("phi_max", 360.0)
            phi_num = roi_params.parameters.get("phi_num", 180)
            return np.linspace(phi_min, phi_max, phi_num)

        # Default x-values
        return np.arange(len(roi_params.parameters.get("default_size", 100)))


class PieROICalculator(VectorizedROICalculator):
    """Optimized calculator for pie-shaped ROI."""

    def calculate_roi_mask(
        self, geometry_data: dict[str, np.ndarray], roi_params: ROIParameters
    ) -> np.ndarray:
        """Calculate pie ROI mask using vectorized operations."""
        pmap = geometry_data["pmap"]
        mask = geometry_data.get("mask", np.ones_like(pmap))

        pmin, pmax = roi_params.parameters["angle_range"]

        # Handle angle wraparound for pie ROI
        if pmin < pmax:
            angle_mask = (pmap >= pmin) & (pmap < pmax)
        else:
            # Wraparound case (e.g., pmin=350, pmax=10)
            angle_mask = (pmap >= pmin) | (pmap < pmax)

        # Combine with detector mask
        roi_mask = angle_mask & (mask > 0)

        logger.debug(
            f"Pie ROI mask: {np.sum(roi_mask)} pixels selected "
            f"({np.sum(roi_mask) / roi_mask.size * 100:.1f}%)"
        )

        return roi_mask

    def process_roi_data(
        self, saxs_data: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> ROIResult:
        """Process pie ROI with optimized q-value binning."""
        qmap_idx = roi_params.parameters["qmap_idx"]
        qsize = roi_params.parameters["qsize"]
        qspan = roi_params.parameters.get("qspan", np.arange(qsize + 1))

        # Vectorized binning for all time frames at once
        if saxs_data.ndim == 3:
            # Multiple time frames
            roi_data = np.zeros((saxs_data.shape[0], qsize))

            # Prepare masked indices
            flat_qmap = np.where(roi_mask, qmap_idx, 0).ravel()

            for t in range(saxs_data.shape[0]):
                frame_data = saxs_data[t].ravel()
                binned = np.bincount(flat_qmap, frame_data, minlength=qsize + 1)
                roi_data[t] = binned[1:]  # Remove 0th bin

            # Average over time if multiple frames
            final_roi_data = np.mean(roi_data, axis=0)

        else:
            # Single frame
            flat_qmap = np.where(roi_mask, qmap_idx, 0).ravel()
            flat_data = saxs_data.ravel()
            binned = np.bincount(flat_qmap, flat_data, minlength=qsize + 1)
            final_roi_data = binned[1:]

        # Generate q-values
        x_values = qspan[:-1] if len(qspan) == qsize + 1 else np.arange(qsize)

        # Apply distance cutoff if specified
        if "dist" in roi_params.parameters:
            roi_params.parameters["dist"]
            qmax = roi_params.parameters.get("qmax")
            if qmax is not None:
                qmax_idx = np.searchsorted(x_values, qmax)
                final_roi_data[qmax_idx:] = np.nan

        return ROIResult(
            x_values=x_values,
            roi_data=final_roi_data,
            roi_type=roi_params.roi_type,
            parameters=roi_params.parameters,
            metadata={"vectorized": True, "qsize": qsize},
            processing_time=0.0,
            memory_used_mb=0.0,
        )


class RingROICalculator(VectorizedROICalculator):
    """Optimized calculator for ring-shaped ROI."""

    def calculate_roi_mask(
        self, geometry_data: dict[str, np.ndarray], roi_params: ROIParameters
    ) -> np.ndarray:
        """Calculate ring ROI mask using vectorized operations."""
        rmap = geometry_data["rmap"]
        mask = geometry_data.get("mask", np.ones_like(rmap))

        rmin, rmax = roi_params.parameters["radius_range"]

        # Vectorized ring mask calculation
        ring_mask = (rmap >= rmin) & (rmap < rmax) & (mask > 0)

        logger.debug(
            f"Ring ROI mask: {np.sum(ring_mask)} pixels selected "
            f"({np.sum(ring_mask) / ring_mask.size * 100:.1f}%)"
        )

        return ring_mask

    def process_roi_data(
        self, saxs_data: np.ndarray, roi_mask: np.ndarray, roi_params: ROIParameters
    ) -> ROIResult:
        """Process ring ROI with optimized angular binning."""
        pmap = roi_params.parameters["pmap"]
        phi_num = roi_params.parameters.get("phi_num", 180)

        # Calculate angular indices for binning
        pmap_roi = pmap[roi_mask]
        phi_min, phi_max = np.min(pmap_roi), np.max(pmap_roi)
        phi_indices = np.floor((pmap - phi_min) / (phi_max - phi_min) * phi_num).astype(
            int
        )
        phi_indices = np.clip(phi_indices, 0, phi_num - 1)

        # Create angular binning mask
        angular_idx = np.where(roi_mask, phi_indices, 0).ravel()

        # Process SAXS data
        if saxs_data.ndim == 3:
            roi_data = np.zeros((saxs_data.shape[0], phi_num))
            for t in range(saxs_data.shape[0]):
                frame_data = saxs_data[t].ravel()
                binned = np.bincount(angular_idx, frame_data, minlength=phi_num + 1)
                roi_data[t] = binned[1:]
            final_roi_data = np.mean(roi_data, axis=0)
        else:
            flat_data = saxs_data.ravel()
            binned = np.bincount(angular_idx, flat_data, minlength=phi_num + 1)
            final_roi_data = binned[1:]

        # Generate angular x-values
        x_values = np.linspace(phi_min, phi_max, phi_num)

        return ROIResult(
            x_values=x_values,
            roi_data=final_roi_data,
            roi_type=roi_params.roi_type,
            parameters=roi_params.parameters,
            metadata={
                "vectorized": True,
                "phi_num": phi_num,
                "phi_range": (phi_min, phi_max),
            },
            processing_time=0.0,
            memory_used_mb=0.0,
        )


class ParallelROIProcessor:
    """Process multiple ROIs in parallel for enhanced performance."""

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers
        self.memory_manager = get_memory_manager()

    def calculate_multiple_rois(
        self,
        saxs_data: np.ndarray,
        geometry_data: dict[str, np.ndarray],
        roi_list: list[ROIParameters],
        use_parallel: bool = True,
    ) -> list[ROIResult]:
        """
        Calculate multiple ROIs with optional parallel processing.

        Parameters
        ----------
        saxs_data : np.ndarray
            SAXS data array
        geometry_data : Dict[str, np.ndarray]
            Geometry arrays
        roi_list : List[ROIParameters]
            List of ROI parameters to calculate
        use_parallel : bool
            Whether to use parallel processing

        Returns
        -------
        List[ROIResult]
            List of ROI calculation results
        """
        if not roi_list:
            return []

        # Check memory pressure
        pressure = self.memory_manager.get_memory_pressure()
        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning(
                "High memory pressure detected, using sequential ROI processing"
            )
            use_parallel = False

        if not use_parallel or len(roi_list) < 2:
            # Sequential processing
            return self._calculate_sequential(saxs_data, geometry_data, roi_list)
        # Parallel processing
        return self._calculate_parallel(saxs_data, geometry_data, roi_list)

    def _calculate_sequential(
        self,
        saxs_data: np.ndarray,
        geometry_data: dict[str, np.ndarray],
        roi_list: list[ROIParameters],
    ) -> list[ROIResult]:
        """Calculate ROIs sequentially."""
        results = []
        for roi_params in roi_list:
            calculator = self._get_calculator(roi_params.roi_type)
            result = calculator.calculate_roi(saxs_data, geometry_data, roi_params)
            results.append(result)
        return results

    def _calculate_parallel(
        self,
        saxs_data: np.ndarray,
        geometry_data: dict[str, np.ndarray],
        roi_list: list[ROIParameters],
    ) -> list[ROIResult]:
        """Calculate ROIs in parallel."""
        import os

        max_workers = self.max_workers or min(len(roi_list), os.cpu_count() or 1)

        logger.info(
            f"Processing {len(roi_list)} ROIs in parallel with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all ROI calculations
            future_to_roi = {}
            for i, roi_params in enumerate(roi_list):
                calculator = self._get_calculator(roi_params.roi_type)
                future = executor.submit(
                    calculator.calculate_roi, saxs_data, geometry_data, roi_params
                )
                future_to_roi[future] = i

            # Collect results in order
            results = [None] * len(roi_list)
            for future in as_completed(future_to_roi):
                index = future_to_roi[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"ROI calculation {index} failed: {e}")
                    # Create empty result for failed calculation
                    results[index] = ROIResult(
                        x_values=np.array([]),
                        roi_data=np.array([]),
                        roi_type=roi_list[index].roi_type,
                        parameters=roi_list[index].parameters,
                        metadata={"error": str(e)},
                        processing_time=0.0,
                        memory_used_mb=0.0,
                    )

        return results

    def _get_calculator(self, roi_type: ROIType) -> VectorizedROICalculator:
        """Get appropriate calculator for ROI type."""
        if roi_type == ROIType.PIE:
            return PieROICalculator()
        if roi_type == ROIType.RING:
            return RingROICalculator()
        # Default to pie calculator for unknown types
        logger.warning(f"Unknown ROI type {roi_type}, using pie calculator")
        return PieROICalculator()


# Convenience functions for easy usage
def calculate_pie_roi(
    saxs_data: np.ndarray,
    geometry_data: dict[str, np.ndarray],
    angle_range: tuple[float, float],
    **kwargs,
) -> ROIResult:
    """Convenience function for pie ROI calculation."""
    roi_params = ROIParameters(
        roi_type=ROIType.PIE, parameters={"angle_range": angle_range, **kwargs}
    )
    calculator = PieROICalculator()
    return calculator.calculate_roi(saxs_data, geometry_data, roi_params)


def calculate_ring_roi(
    saxs_data: np.ndarray,
    geometry_data: dict[str, np.ndarray],
    radius_range: tuple[float, float],
    **kwargs,
) -> ROIResult:
    """Convenience function for ring ROI calculation."""
    roi_params = ROIParameters(
        roi_type=ROIType.RING, parameters={"radius_range": radius_range, **kwargs}
    )
    calculator = RingROICalculator()
    return calculator.calculate_roi(saxs_data, geometry_data, roi_params)


def calculate_multiple_rois_parallel(
    saxs_data: np.ndarray,
    geometry_data: dict[str, np.ndarray],
    roi_list: list[ROIParameters],
    max_workers: int | None = None,
) -> list[ROIResult]:
    """Convenience function for parallel multiple ROI calculation."""
    processor = ParallelROIProcessor(max_workers=max_workers)
    return processor.calculate_multiple_rois(
        saxs_data, geometry_data, roi_list, use_parallel=True
    )
