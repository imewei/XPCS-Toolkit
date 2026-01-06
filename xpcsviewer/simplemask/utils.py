"""Utility functions for SimpleMask partitioning.

This module provides functions for generating Q-space partitions
and combining partition maps for XPCS analysis.

Ported from pySimpleMask with minimal modifications.
"""

import hashlib
import json
from typing import Any

import numpy as np


def hash_numpy_dict(input_dictionary: dict[str, Any]) -> str:
    """Compute a stable SHA256 hash for a dictionary containing NumPy arrays.

    Args:
        input_dictionary: Dictionary with NumPy arrays and other values

    Returns:
        SHA256 hash string of the dictionary contents
    """
    hasher = hashlib.sha256()

    for key in sorted(input_dictionary.keys()):
        hasher.update(str(key).encode())
        value = input_dictionary[key]

        if isinstance(value, np.ndarray):
            # Ensure consistent dtype & memory layout
            value = np.ascontiguousarray(value)
            hasher.update(value.astype(value.dtype.newbyteorder("=")).tobytes())
        elif isinstance(value, list):
            hasher.update(json.dumps(value, sort_keys=True).encode())
        else:
            hasher.update(json.dumps(value, sort_keys=True).encode())

    return hasher.hexdigest()


def optimize_integer_array(arr: np.ndarray) -> np.ndarray:
    """Optimize the data type of an integer array to minimize memory.

    Args:
        arr: NumPy array of integers

    Returns:
        Array with optimized integer dtype, or original if not applicable
    """
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return arr

    if not np.issubdtype(arr.dtype, np.integer):
        return arr

    min_val, max_val = arr.min(), arr.max()

    # Choose smallest dtype based on min/max
    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            new_dtype = np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            new_dtype = np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            new_dtype = np.uint32
        else:
            new_dtype = np.uint64
    elif min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
        new_dtype = np.int8
    elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
        new_dtype = np.int16
    elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
        new_dtype = np.int32
    else:
        new_dtype = np.int64

    return arr.astype(new_dtype) if new_dtype != arr.dtype else arr


def generate_partition(
    map_name: str,
    mask: np.ndarray,
    xmap: np.ndarray,
    num_pts: int,
    style: str = "linear",
    phi_offset: float | None = None,
    symmetry_fold: int = 1,
) -> dict[str, str | int | np.ndarray]:
    """Generate a partition map for X-ray scattering analysis.

    Args:
        map_name: Name of the map ("q", "phi", "x", "y")
        mask: 2D boolean mask array (True = valid)
        xmap: 2D array of values to partition
        num_pts: Number of partition bins
        style: Binning style - "linear" or "logarithmic"
        phi_offset: Offset for phi angle (only for phi map)
        symmetry_fold: Symmetry fold for phi partitioning

    Returns:
        Dictionary with keys:
            - map_name: Name of the partition
            - num_pts: Number of bins
            - partition: 2D array of bin labels (1-indexed, 0=masked)
            - v_list: Array of bin center values
    """
    xmap_phi = None
    unit_xmap = None

    if map_name == "phi":
        xmap_phi = xmap.copy()
        if phi_offset is not None:
            xmap = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(xmap + phi_offset))))
        if symmetry_fold > 1:
            unit_xmap = (xmap < (360 / symmetry_fold)) * (xmap >= 0)
            xmap = xmap + 180.0
            xmap = np.mod(xmap, 360.0 / symmetry_fold)

    roi = mask > 0
    v_min = np.nanmin(xmap[roi])
    v_max = np.nanmax(xmap[roi])

    if map_name == "q" and style == "logarithmic":
        mask = mask * (xmap > 0)
        valid_xmap = xmap[mask > 0]
        if valid_xmap.size == 0 or np.all(np.isnan(valid_xmap)):
            raise ValueError(
                "Invalid xmap values for logarithmic binning. All values are non-positive."
            )
        v_min = np.nanmin(valid_xmap)
        xmap = np.where(xmap > 0, xmap, np.nan)
        v_span = np.logspace(np.log10(v_min), np.log10(v_max), num_pts + 1, base=10)
        v_list = np.sqrt(v_span[1:] * v_span[:-1])
    else:
        v_span = np.linspace(v_min, v_max, num_pts + 1)
        v_list = (v_span[1:] + v_span[:-1]) / 2.0

    partition = np.digitize(xmap, v_span).astype(np.uint32) * mask
    partition[partition > num_pts] = 0
    partition[(xmap == v_max) * mask] = num_pts

    if map_name == "phi" and symmetry_fold > 1 and unit_xmap is not None:
        idx_map = unit_xmap * partition
        sum_value = np.bincount(idx_map.flatten(), weights=xmap_phi.flatten())
        norm_factor = np.bincount(idx_map.flatten())
        v_list = sum_value / np.clip(norm_factor, 1, None)
        v_list = v_list[1:]

    return {
        "map_name": map_name,
        "num_pts": num_pts,
        "partition": partition,
        "v_list": v_list,
    }


def combine_partitions(
    pack1: dict[str, str | int | np.ndarray],
    pack2: dict[str, str | int | np.ndarray],
    prefix: str = "dynamic",
) -> dict[str, list | np.ndarray]:
    """Combine two partition maps into a single partition space.

    Args:
        pack1: First partition dictionary (e.g., Q partition)
        pack2: Second partition dictionary (e.g., phi partition)
        prefix: Prefix for output keys ("dynamic" or "static")

    Returns:
        Dictionary with combined partition:
            - {prefix}_num_pts: [num_pts1, num_pts2]
            - {prefix}_roi_map: Combined 2D partition array
            - {prefix}_v_list_dim0: Bin centers for first dimension
            - {prefix}_v_list_dim1: Bin centers for second dimension
            - {prefix}_index_mapping: Unique partition indices
    """
    # Convert to zero-based indexing, merge, convert back
    partition = (
        (pack1["partition"].astype(np.int64) - 1) * pack2["num_pts"]
        + (pack2["partition"].astype(np.int64) - 1)
        + 1
    )

    partition = np.clip(partition, a_min=0, a_max=None).astype(np.uint32)

    start_index = np.min(partition)
    unique_idx, inverse = np.unique(partition, return_inverse=True)
    partition_natural_order = inverse.reshape(partition.shape).astype(np.uint32)

    # Shift if needed to preserve masked pixel indicator (0)
    if start_index > 0:
        partition_natural_order += 1

    return {
        f"{prefix}_num_pts": [pack1["num_pts"], pack2["num_pts"]],
        f"{prefix}_roi_map": partition_natural_order,
        f"{prefix}_v_list_dim0": pack1["v_list"],
        f"{prefix}_v_list_dim1": pack2["v_list"],
        f"{prefix}_index_mapping": unique_idx[unique_idx >= 1] - 1,
    }


def check_consistency(dqmap: np.ndarray, sqmap: np.ndarray, mask: np.ndarray) -> bool:
    """Check consistency between dynamic and static Q-maps.

    Ensures each unique value in sqmap corresponds to only one unique value in dqmap.

    Args:
        dqmap: Dynamic Q-map (coarse bins)
        sqmap: Static Q-map (fine bins)
        mask: Boolean mask array

    Returns:
        True if maps are consistent, False otherwise

    Raises:
        ValueError: If array shapes don't match
    """
    if dqmap.shape != sqmap.shape:
        raise ValueError("dqmap and sqmap must have the same shape")
    if dqmap.shape != mask.shape:
        raise ValueError("dqmap and mask must have the same shape")

    if not np.all((mask > 0) == (dqmap > 0)):
        return False
    if not np.all((mask > 0) == (sqmap > 0)):
        return False

    sq_flat = sqmap.ravel()
    dq_flat = dqmap.ravel()

    sq_to_dq: dict[int, int] = {}

    for sq_value, dq_value in zip(sq_flat, dq_flat, strict=False):
        if sq_value in sq_to_dq:
            if sq_to_dq[sq_value] != dq_value:
                return False
        else:
            sq_to_dq[sq_value] = dq_value

    return True
