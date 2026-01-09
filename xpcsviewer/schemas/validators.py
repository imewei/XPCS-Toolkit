"""Data structure validators using dataclasses with validation.

This module provides type-safe, validated data structures for all shared
data contracts in the XPCS Viewer codebase.

Each schema includes:
- Type annotations for all fields
- Validation in __post_init__
- Factory methods for converting from legacy dicts
- Docstrings documenting units and ranges
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np

from xpcsviewer.utils.logging_config import get_logger

logger = get_logger(__name__)


def _log_validation_result(
    schema_name: str, shape: tuple | None, success: bool
) -> None:
    """Log schema validation result at DEBUG level."""
    if logger.isEnabledFor(logging.DEBUG):
        status = "passed" if success else "failed"
        logger.debug(f"Schema validation {status}: {schema_name} shape={shape}")


@dataclass
class QMapSchema:
    """Q-map data structure with validation.

    This schema represents the momentum transfer (Q) maps used for XPCS analysis.

    Attributes
    ----------
    sqmap : np.ndarray
        Static Q-map, shape=(H, W), dtype=float64, units specified in sqmap_unit
    dqmap : np.ndarray
        Dynamic Q-map, shape=(H, W), dtype=float64, units specified in dqmap_unit
    phis : np.ndarray
        Azimuthal angle map, shape=(H, W), dtype=float64, units specified in phis_unit
    sqmap_unit : str
        Units for static Q-map, must be 'nm^-1' or 'A^-1'
    dqmap_unit : str
        Units for dynamic Q-map, must be 'nm^-1' or 'A^-1'
    phis_unit : str
        Units for azimuthal angle, must be 'rad' or 'deg'
    mask : np.ndarray, optional
        Mask array, shape=(H, W), dtype=int32, values 0 (masked) or 1 (valid)
    partition_map : np.ndarray, optional
        Q-bin partition map, shape=(H, W), dtype=int32
    """

    sqmap: np.ndarray
    dqmap: np.ndarray
    phis: np.ndarray
    sqmap_unit: Literal["nm^-1", "A^-1"]
    dqmap_unit: Literal["nm^-1", "A^-1"]
    phis_unit: Literal["rad", "deg"]
    mask: np.ndarray | None = None
    partition_map: np.ndarray | None = None

    def __post_init__(self):
        """Validate Q-map schema after initialization."""
        # Shape validation
        if not (self.sqmap.shape == self.dqmap.shape == self.phis.shape):
            raise ValueError(
                f"Q-maps must have identical shapes. Got sqmap={self.sqmap.shape}, "
                f"dqmap={self.dqmap.shape}, phis={self.phis.shape}"
            )

        # Dimension validation
        if len(self.sqmap.shape) != 2:
            raise ValueError(f"Q-maps must be 2D arrays. Got shape={self.sqmap.shape}")

        # Units validation
        if self.sqmap_unit not in ("nm^-1", "A^-1"):
            raise ValueError(
                f"sqmap_unit must be 'nm^-1' or 'A^-1', got '{self.sqmap_unit}'"
            )
        if self.dqmap_unit not in ("nm^-1", "A^-1"):
            raise ValueError(
                f"dqmap_unit must be 'nm^-1' or 'A^-1', got '{self.dqmap_unit}'"
            )
        if self.phis_unit not in ("rad", "deg"):
            raise ValueError(
                f"phis_unit must be 'rad' or 'deg', got '{self.phis_unit}'"
            )

        # Mask validation (if provided)
        if self.mask is not None:
            if self.mask.shape != self.sqmap.shape:
                raise ValueError(
                    f"Mask shape {self.mask.shape} must match "
                    f"Q-map shape {self.sqmap.shape}"
                )
            if not np.all((self.mask == 0) | (self.mask == 1)):
                raise ValueError("Mask values must be 0 or 1")

        # Partition map validation (if provided)
        if self.partition_map is not None:
            if self.partition_map.shape != self.sqmap.shape:
                raise ValueError(
                    f"Partition map shape {self.partition_map.shape} must match "
                    f"Q-map shape {self.sqmap.shape}"
                )

    @classmethod
    def from_dict(cls, data: dict) -> QMapSchema:
        """Create QMapSchema from legacy dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys: sqmap, dqmap, phis, sqmap_unit, dqmap_unit, phis_unit,
            and optionally mask, partition_map

        Returns
        -------
        QMapSchema
            Validated Q-map schema instance
        """
        return cls(
            sqmap=data["sqmap"],
            dqmap=data["dqmap"],
            phis=data["phis"],
            sqmap_unit=data["sqmap_unit"],
            dqmap_unit=data["dqmap_unit"],
            phis_unit=data["phis_unit"],
            mask=data.get("mask"),
            partition_map=data.get("partition_map"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary representation of Q-map
        """
        result = {
            "sqmap": self.sqmap,
            "dqmap": self.dqmap,
            "phis": self.phis,
            "sqmap_unit": self.sqmap_unit,
            "dqmap_unit": self.dqmap_unit,
            "phis_unit": self.phis_unit,
        }
        if self.mask is not None:
            result["mask"] = self.mask
        if self.partition_map is not None:
            result["partition_map"] = self.partition_map
        return result


@dataclass
class GeometryMetadata:
    """Detector geometry metadata with validation.

    Attributes
    ----------
    bcx : float
        Beam center X coordinate (column), in pixels, 0-indexed
    bcy : float
        Beam center Y coordinate (row), in pixels, 0-indexed
    det_dist : float
        Detector-to-sample distance in millimeters, must be positive
    lambda_ : float
        X-ray wavelength in Angstroms, must be positive
    pix_dim : float
        Pixel size in millimeters, must be positive
    shape : tuple[int, int]
        Detector shape as (height, width) in pixels
    det_rotation : float, optional
        Detector rotation angle in degrees
    incident_angle : float, optional
        Grazing incidence angle in degrees
    """

    bcx: float
    bcy: float
    det_dist: float
    lambda_: float
    pix_dim: float
    shape: tuple[int, int]
    det_rotation: float | None = None
    incident_angle: float | None = None

    def __post_init__(self):
        """Validate geometry metadata after initialization."""
        # Positive value validation
        if self.det_dist <= 0:
            raise ValueError(f"Detector distance must be positive, got {self.det_dist}")
        if self.lambda_ <= 0:
            raise ValueError(f"Wavelength must be positive, got {self.lambda_}")
        if self.pix_dim <= 0:
            raise ValueError(f"Pixel size must be positive, got {self.pix_dim}")

        # Shape validation
        if len(self.shape) != 2:
            raise ValueError(f"Shape must be (height, width), got {self.shape}")
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError(f"Shape dimensions must be positive, got {self.shape}")

        # Beam center validation - should be within detector bounds
        if not (0 <= self.bcx <= self.shape[1]):
            logger.warning(
                f"Beam center X ({self.bcx}) is outside "
                f"detector width ({self.shape[1]})"
            )
        if not (0 <= self.bcy <= self.shape[0]):
            logger.warning(
                f"Beam center Y ({self.bcy}) is outside "
                f"detector height ({self.shape[0]})"
            )

    @classmethod
    def from_dict(cls, data: dict) -> GeometryMetadata:
        """Create GeometryMetadata from legacy dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys: bcx, bcy, det_dist, lambda_, pix_dim, shape,
            and optionally det_rotation, incident_angle

        Returns
        -------
        GeometryMetadata
            Validated geometry metadata instance
        """
        return cls(
            bcx=float(data["bcx"]),
            bcy=float(data["bcy"]),
            det_dist=float(data["det_dist"]),
            lambda_=float(data["lambda_"]),
            pix_dim=float(data["pix_dim"]),
            shape=tuple(data["shape"]),
            det_rotation=data.get("det_rotation"),
            incident_angle=data.get("incident_angle"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary representation of geometry metadata
        """
        result = {
            "bcx": self.bcx,
            "bcy": self.bcy,
            "det_dist": self.det_dist,
            "lambda_": self.lambda_,
            "pix_dim": self.pix_dim,
            "shape": self.shape,
        }
        if self.det_rotation is not None:
            result["det_rotation"] = self.det_rotation
        if self.incident_angle is not None:
            result["incident_angle"] = self.incident_angle
        return result


@dataclass
class G2Data:
    """G2 correlation data structure with validation.

    Attributes
    ----------
    g2 : np.ndarray
        G2 correlation values, shape=(n_q, n_delay), dtype=float64
    g2_err : np.ndarray
        G2 correlation errors, shape=(n_q, n_delay), dtype=float64
    delay_times : np.ndarray
        Delay times in seconds, shape=(n_delay,), dtype=float64
    q_values : list[float]
        Q values for each bin, length=n_q
    """

    g2: np.ndarray
    g2_err: np.ndarray
    delay_times: np.ndarray
    q_values: list[float]

    def __post_init__(self):
        """Validate G2 data after initialization."""
        # Shape consistency validation
        if self.g2.shape != self.g2_err.shape:
            raise ValueError(
                f"g2 and g2_err must have same shape. "
                f"Got g2={self.g2.shape}, g2_err={self.g2_err.shape}"
            )

        # Dimension validation
        if len(self.g2.shape) != 2:
            raise ValueError(
                f"g2 must be 2D array (n_q, n_delay). Got shape={self.g2.shape}"
            )

        n_q, n_delay = self.g2.shape

        # Delay times validation
        if len(self.delay_times) != n_delay:
            raise ValueError(
                f"delay_times length ({len(self.delay_times)}) must match "
                f"g2 second dimension ({n_delay})"
            )

        # Q values validation
        if len(self.q_values) != n_q:
            raise ValueError(
                f"q_values length ({len(self.q_values)}) must match "
                f"g2 first dimension ({n_q})"
            )

        # Physical constraints
        if np.any(self.delay_times < 0):
            raise ValueError("Delay times must be non-negative")

        if np.any(self.g2_err < 0):
            raise ValueError("G2 errors must be non-negative")

    @classmethod
    def from_dict(cls, data: dict) -> G2Data:
        """Create G2Data from legacy dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys: g2, g2_err, delay_times, q_values

        Returns
        -------
        G2Data
            Validated G2 data instance
        """
        return cls(
            g2=data["g2"],
            g2_err=data["g2_err"],
            delay_times=data["delay_times"],
            q_values=data["q_values"],
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary representation of G2 data
        """
        return {
            "g2": self.g2,
            "g2_err": self.g2_err,
            "delay_times": self.delay_times,
            "q_values": self.q_values,
        }


@dataclass
class PartitionSchema:
    """Q-bin partition data structure with validation.

    Attributes
    ----------
    partition_map : np.ndarray
        Partition map with Q-bin indices, shape=(H, W), dtype=int32
    num_pts : int
        Number of Q-bins, must be positive
    val_list : list[float]
        Q-bin center values in nm^-1 or A^-1, length=num_pts
    num_list : list[int]
        Number of pixels per Q-bin, length=num_pts
    metadata : GeometryMetadata
        Geometry metadata used for partition computation
    version : str
        Schema version string (e.g., "1.0.0")
    mask : np.ndarray, optional
        Mask used during partitioning, shape=(H, W), dtype=int32
    method : str, optional
        Binning method, must be 'linear' or 'log'
    """

    partition_map: np.ndarray
    num_pts: int
    val_list: list[float]
    num_list: list[int]
    metadata: GeometryMetadata
    version: str = "1.0.0"
    mask: np.ndarray | None = None
    method: Literal["linear", "log"] | None = None

    def __post_init__(self):
        """Validate partition schema after initialization."""
        # Positive value validation
        if self.num_pts <= 0:
            raise ValueError(f"num_pts must be positive, got {self.num_pts}")

        # Dimension validation
        if len(self.partition_map.shape) != 2:
            raise ValueError(
                f"partition_map must be 2D array. Got shape={self.partition_map.shape}"
            )

        # List length validation
        if len(self.val_list) != self.num_pts:
            raise ValueError(
                f"val_list length ({len(self.val_list)}) must equal "
                f"num_pts ({self.num_pts})"
            )
        if len(self.num_list) != self.num_pts:
            raise ValueError(
                f"num_list length ({len(self.num_list)}) must equal "
                f"num_pts ({self.num_pts})"
            )

        # Mask validation (if provided)
        if self.mask is not None:
            if self.mask.shape != self.partition_map.shape:
                raise ValueError(
                    f"mask shape {self.mask.shape} must match partition_map "
                    f"shape {self.partition_map.shape}"
                )

        # Method validation (if provided)
        if self.method is not None and self.method not in ("linear", "log"):
            raise ValueError(f"method must be 'linear' or 'log', got '{self.method}'")

        # Physical constraints
        if any(n < 0 for n in self.num_list):
            raise ValueError("num_list values must be non-negative")

        # Consistency check: sum of num_list should roughly match
        # non-zero partition_map pixels
        total_pixels = sum(self.num_list)
        nonzero_pixels = np.count_nonzero(self.partition_map)
        if abs(total_pixels - nonzero_pixels) > self.num_pts:
            logger.warning(
                f"Partition consistency warning: sum(num_list)={total_pixels}, "
                f"nonzero pixels={nonzero_pixels}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> PartitionSchema:
        """Create PartitionSchema from legacy dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys: partition_map, num_pts, val_list, num_list, metadata,
            and optionally version, mask, method

        Returns
        -------
        PartitionSchema
            Validated partition schema instance
        """
        # Handle metadata - can be dict or GeometryMetadata
        metadata = data["metadata"]
        if isinstance(metadata, dict):
            metadata = GeometryMetadata.from_dict(metadata)

        return cls(
            partition_map=data["partition_map"],
            num_pts=int(data["num_pts"]),
            val_list=list(data["val_list"]),
            num_list=list(data["num_list"]),
            metadata=metadata,
            version=data.get("version", "1.0.0"),
            mask=data.get("mask"),
            method=data.get("method"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary representation of partition schema
        """
        result = {
            "partition_map": self.partition_map,
            "num_pts": self.num_pts,
            "val_list": self.val_list,
            "num_list": self.num_list,
            "metadata": self.metadata.to_dict(),
            "version": self.version,
        }
        if self.mask is not None:
            result["mask"] = self.mask
        if self.method is not None:
            result["method"] = self.method
        return result


@dataclass
class MaskSchema:
    """Mask data structure with validation.

    Attributes
    ----------
    mask : np.ndarray
        Mask array, shape=(H, W), dtype=int32, values 0 (masked) or 1 (valid)
    metadata : GeometryMetadata
        Geometry metadata associated with the mask
    version : str
        Schema version string (e.g., "1.0.0")
    description : str, optional
        Human-readable description of the mask
    """

    mask: np.ndarray
    metadata: GeometryMetadata
    version: str = "1.0.0"
    description: str | None = None

    def __post_init__(self):
        """Validate mask schema after initialization."""
        # Dimension validation
        if len(self.mask.shape) != 2:
            raise ValueError(f"mask must be 2D array. Got shape={self.mask.shape}")

        # Value validation
        if not np.all((self.mask == 0) | (self.mask == 1)):
            raise ValueError("Mask values must be 0 or 1")

        # Shape consistency with metadata
        if self.mask.shape != self.metadata.shape:
            logger.warning(
                f"Mask shape {self.mask.shape} does not match metadata shape "
                f"{self.metadata.shape}"
            )

    @classmethod
    def from_dict(cls, data: dict) -> MaskSchema:
        """Create MaskSchema from legacy dictionary.

        Parameters
        ----------
        data : dict
            Dictionary with keys: mask, metadata, and optionally version, description

        Returns
        -------
        MaskSchema
            Validated mask schema instance
        """
        # Handle metadata - can be dict or GeometryMetadata
        metadata = data["metadata"]
        if isinstance(metadata, dict):
            metadata = GeometryMetadata.from_dict(metadata)

        return cls(
            mask=data["mask"],
            metadata=metadata,
            version=data.get("version", "1.0.0"),
            description=data.get("description"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for backward compatibility.

        Returns
        -------
        dict
            Dictionary representation of mask schema
        """
        result = {
            "mask": self.mask,
            "metadata": self.metadata.to_dict(),
            "version": self.version,
        }
        if self.description is not None:
            result["description"] = self.description
        return result
