"""Mask assembly classes for SimpleMask.

This module provides classes for creating, combining, and managing detector masks.
Supports various mask types including file-based, threshold-based, parameter-based,
and drawing-based masks.

Ported from pySimpleMask with modifications:
- Removed TIFF support for initial release (HDF5 only)
- Removed skimage.io dependency
"""

import logging
import os
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# Angular periodicity constants for degree-based coordinates
ANGLE_MIN_DEG = -180
ANGLE_MAX_DEG = 180


def create_qring(
    qmin: float,
    qmax: float,
    pmin: float,
    pmax: float,
    qnum: int = 1,
    flag_const_width: bool = True,
) -> list[tuple[float, float, float, float]]:
    """Create Q-ring definitions for mask parameter constraints.

    Args:
        qmin: Minimum Q value
        qmax: Maximum Q value
        pmin: Minimum phi value
        pmax: Maximum phi value
        qnum: Number of Q rings to generate
        flag_const_width: If True, rings have constant width; if False, width scales

    Returns:
        List of (qlow, qhigh, pmin, pmax) tuples defining each ring
    """
    qrings = []
    if qmin > qmax:
        qmin, qmax = qmax, qmin
    qcen = (qmin + qmax) / 2.0
    qhalf = (qmax - qmin) / 2.0

    for n in range(1, qnum + 1):
        if flag_const_width:
            low = qcen * n - qhalf
            high = qcen * n + qhalf
        else:
            low = (qcen - qhalf) * n
            high = (qcen + qhalf) * n
        qrings.append((low, high, pmin, pmax))
    return qrings


class MaskBase:
    """Base class for all mask types."""

    def __init__(self, shape: tuple[int, int] = (512, 1024)) -> None:
        """Initialize mask with given detector shape.

        Args:
            shape: Detector dimensions as (height, width)
        """
        self.shape = shape
        self.zero_loc: np.ndarray | None = None
        self.mtype = "base"
        self.qrings: list = []

    def describe(self) -> str:
        """Return description of mask statistics."""
        if self.zero_loc is None:
            return "Mask is not initialized"
        bad_num = len(self.zero_loc[0])
        total_num = self.shape[0] * self.shape[1]
        ratio = bad_num / total_num * 100.0
        return f"{self.mtype}: bad_pixel: {bad_num}/{ratio:0.3f}%"

    def get_mask(self) -> np.ndarray:
        """Return the mask as a 2D boolean array.

        Returns:
            Boolean array where True = valid pixel, False = masked pixel
        """
        mask = np.ones(self.shape, dtype=bool)
        if self.zero_loc is not None:
            mask[tuple(self.zero_loc)] = 0
        return mask

    def combine_mask(self, mask: np.ndarray | None) -> np.ndarray:
        """Combine this mask with another mask.

        Args:
            mask: Existing mask to combine with, or None

        Returns:
            Combined mask array
        """
        if self.zero_loc is not None:
            if mask is None:
                mask = self.get_mask()
            else:
                mask[tuple(self.zero_loc)] = 0
        return mask


class MaskList(MaskBase):
    """Mask defined by a list of pixel coordinates."""

    def __init__(self, shape: tuple[int, int] = (512, 1024)) -> None:
        super().__init__(shape=shape)
        self.mtype = "list"
        self.xylist: np.ndarray | None = None

    def append_zero_pt(self, row: int, col: int) -> None:
        """Append a single pixel to the mask.

        Args:
            row: Row index
            col: Column index
        """
        self.zero_loc = np.append(
            self.zero_loc, np.array([row, col]).reshape(2, 1), axis=1
        )

    def evaluate(self, zero_loc: np.ndarray | None = None) -> None:
        """Set the mask from a coordinate array.

        Args:
            zero_loc: Array of shape (2, N) with [rows, cols] of masked pixels
        """
        self.zero_loc = zero_loc


class MaskFile(MaskBase):
    """Mask loaded from an HDF5 file."""

    def __init__(
        self,
        shape: tuple[int, int] = (512, 1024),
        fname: str | None = None,  # noqa: ARG002 - kept for API compatibility
        **kwargs,  # noqa: ARG002 - kept for API compatibility
    ) -> None:
        super().__init__(shape=shape)
        self.mtype = "file"

    def evaluate(self, fname: str | None = None, key: str | None = None) -> None:
        """Load mask from an HDF5 file.

        Args:
            fname: Path to HDF5 file
            key: Dataset key within the HDF5 file containing the mask
        """
        if fname is None or not os.path.isfile(fname):
            self.zero_loc = None
            return

        _, ext = os.path.splitext(fname)
        mask = None

        if ext in [".hdf", ".h5", ".hdf5"]:
            try:
                with h5py.File(fname, "r") as f:
                    mask = f[key][()]
            except Exception:
                logger.error(f"Cannot read HDF file: {fname}, key: {key}")
        else:
            logger.error(f"MaskFile only supports HDF5 files. Found: {fname}")

        if mask is None:
            self.zero_loc = None
            return

        # Handle transposed masks
        if mask.shape != self.shape:
            mask = np.swapaxes(mask, 0, 1)

        if mask.shape != self.shape:
            logger.error(
                f"Mask shape {mask.shape} doesn't match detector shape {self.shape}"
            )
            self.zero_loc = None
            return

        # Masked pixels have value <= 0
        mask = mask <= 0
        self.zero_loc = np.array(np.nonzero(mask))


class MaskThreshold(MaskBase):
    """Mask based on intensity thresholds."""

    def __init__(self, shape: tuple[int, int] = (512, 1024)) -> None:
        super().__init__(shape=shape)
        self.mtype = "threshold"

    def evaluate(
        self,
        saxs_lin: np.ndarray | None = None,
        low: float = 0,
        high: float = 1e8,
        low_enable: bool = True,
        high_enable: bool = True,
    ) -> None:
        """Create mask based on intensity thresholds.

        Args:
            saxs_lin: 2D intensity array
            low: Lower threshold value
            high: Upper threshold value
            low_enable: Whether to apply lower threshold
            high_enable: Whether to apply upper threshold
        """
        if saxs_lin is None:
            self.zero_loc = None
            return

        mask = np.ones_like(saxs_lin, dtype=bool)
        if low_enable:
            mask = mask * (saxs_lin >= low)
        if high_enable:
            mask = mask * (saxs_lin < high)
        mask = np.logical_not(mask)
        self.zero_loc = np.array(np.nonzero(mask))


class MaskParameter(MaskBase):
    """Mask based on Q-map parameter constraints (e.g., Q-rings)."""

    def __init__(self, shape: tuple[int, int] = (512, 1024)) -> None:
        super().__init__(shape=shape)
        self.mtype = "parameter"
        self.constraints: list = []

    def evaluate(
        self,
        qmap: dict[str, np.ndarray] | None = None,
        constraints: list[tuple[str, str, str, float, float]] | None = None,
    ) -> None:
        """Create mask based on Q-map constraints.

        Args:
            qmap: Dictionary of Q-map arrays
            constraints: List of (map_name, logic, unit, vbeg, vend) tuples
                - map_name: Key in qmap dict (e.g., "q", "phi")
                - logic: "AND" or "OR"
                - unit: Unit string (e.g., "deg")
                - vbeg, vend: Value range bounds
        """
        if qmap is None or constraints is None:
            self.zero_loc = None
            return

        mask = np.ones(self.shape, dtype=bool)
        for xmap_name, logic, unit, vbeg, vend in constraints:
            xmap = qmap[xmap_name]
            # Handle periodicity of angular coordinates
            if xmap_name in ["phi", "chi", "alpha"] and unit == "deg":
                xmap = np.copy(xmap)
                if vbeg <= ANGLE_MIN_DEG <= vend <= ANGLE_MAX_DEG:
                    xmap[xmap > vend] -= 360.0
                if vend > ANGLE_MAX_DEG and ANGLE_MIN_DEG <= vbeg <= ANGLE_MAX_DEG:
                    xmap[xmap < vbeg] += 360.0
            mask_t = (xmap >= vbeg) * (xmap <= vend)
            if logic == "AND":
                mask = np.logical_and(mask, mask_t)
            elif logic == "OR":
                mask = np.logical_or(mask, mask_t)
        self.zero_loc = np.array(np.nonzero(~mask))


class MaskArray(MaskBase):
    """Mask from a direct boolean/integer array."""

    def __init__(self, shape: tuple[int, int] = (512, 1024)) -> None:
        super().__init__(shape=shape)
        self.mtype = "array"

    def evaluate(self, arr: np.ndarray | None = None) -> None:
        """Set mask from an array.

        Args:
            arr: Boolean or integer array where nonzero = masked
        """
        if arr is not None:
            self.zero_loc = np.array(np.nonzero(arr))


class MaskAssemble:
    """Manager for combining multiple mask types with undo/redo support."""

    def __init__(
        self,
        shape: tuple[int, int] = (128, 128),
        saxs_lin: np.ndarray | None = None,
        qmap: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Initialize mask assembly with detector shape and optional data.

        Args:
            shape: Detector dimensions as (height, width)
            saxs_lin: 2D intensity array for threshold masking
            qmap: Q-map dictionary for parameter masking
        """
        self.workers = {
            "mask_blemish": MaskFile(shape),
            "mask_file": MaskFile(shape),
            "mask_threshold": MaskThreshold(shape),
            "mask_list": MaskList(shape),
            "mask_draw": MaskArray(shape),
            "mask_outlier": MaskList(shape),
            "mask_parameter": MaskParameter(shape),
        }
        self.shape = shape
        self.saxs_lin = saxs_lin
        self.qmap = qmap

        # Initialize mask history for undo/redo
        initial_mask = (
            np.ones_like(saxs_lin, dtype=bool)
            if saxs_lin is not None
            else np.ones(shape, dtype=bool)
        )
        self.mask_record: list[np.ndarray] = [initial_mask]
        self.mask_ptr = 0
        # 0: no mask; 1: apply the default mask
        self.mask_ptr_min = 0

    def update_qmap(self, qmap_all: dict[str, np.ndarray]) -> None:
        """Update the Q-map used for parameter masking.

        Args:
            qmap_all: Dictionary of Q-map arrays
        """
        self.qmap = qmap_all

    def apply(self, target: str | None) -> np.ndarray:
        """Apply a mask type and add to history.

        Args:
            target: Mask type key (e.g., "mask_draw", "mask_threshold")
                   or None to return current mask

        Returns:
            Current combined mask array
        """
        if target is None:
            return self.get_mask()

        mask = self.get_one_mask(target)
        mask = np.logical_and(self.get_mask(), mask)

        # Only add to history if mask changed
        if not np.allclose(self.mask_record[-1], mask):
            # Remove any redo states
            while len(self.mask_record) > self.mask_ptr + 1:
                self.mask_record.pop()
            self.mask_record.append(mask)
            self.mask_ptr += 1
        return mask

    def evaluate(self, target: str, **kwargs: Any) -> str:
        """Evaluate a mask type with given parameters.

        Args:
            target: Mask type key
            **kwargs: Parameters for the mask evaluation

        Returns:
            Description string from the mask worker
        """
        if target == "mask_threshold":
            self.workers[target].evaluate(self.saxs_lin, **kwargs)
        elif target == "mask_parameter":
            self.workers[target].evaluate(qmap=self.qmap, **kwargs)
        else:
            self.workers[target].evaluate(**kwargs)

        return self.workers[target].describe()

    def redo_undo(self, action: str = "redo") -> None:
        """Navigate mask history.

        Args:
            action: One of "undo", "redo", or "reset"
        """
        if action == "undo":
            if self.mask_ptr > self.mask_ptr_min:
                self.mask_ptr -= 1
        elif action == "redo":
            if self.mask_ptr < len(self.mask_record) - 1:
                self.mask_ptr += 1
        elif action == "reset":
            # Keep the default mask if one exists
            while len(self.mask_record) > 1 + self.mask_ptr_min:
                self.mask_record.pop()
            self.mask_ptr = self.mask_ptr_min

    def get_one_mask(self, target: str) -> np.ndarray:
        """Get mask from a single worker.

        Args:
            target: Mask type key

        Returns:
            Mask array from the specified worker
        """
        return self.workers[target].get_mask()

    def get_mask(self) -> np.ndarray:
        """Get the current combined mask.

        Returns:
            Current mask from history
        """
        return self.mask_record[self.mask_ptr]
