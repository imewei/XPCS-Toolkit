"""Utility functions for SimpleMask partitioning.

This module provides functions for generating Q-space partitions
and combining partition maps for XPCS analysis.

Uses backend abstraction for GPU acceleration when available.
Ported from pySimpleMask with backend abstraction for JAX support.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# JIT cache for compiled functions (JAX arrays are not hashable for lru_cache)
_PARTITION_JIT_CACHE: dict[str, callable] = {}


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


def _get_partition_linear_jit():
    """Get or create JIT-compiled linear partition function.

    Returns a JIT-compiled function if JAX backend is active,
    otherwise returns None.

    Note: num_pts must be passed as a Python int (static argument) because
    JAX's linspace requires a concrete value for the number of points.
    """
    global _PARTITION_JIT_CACHE

    backend = get_backend()
    if backend.name != "jax":
        return None

    cache_key = "partition_linear_jit"
    if cache_key not in _PARTITION_JIT_CACHE:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _partition_linear_core(mask_b, xmap_b, v_min, v_max, v_span):
            """JIT-compiled linear partition core computation.

            Note: v_span is pre-computed outside JIT since linspace needs
            a concrete value for num_pts.
            """
            num_pts = v_span.shape[0] - 1
            v_list = (v_span[1:] + v_span[:-1]) / 2.0

            # Digitize: find bin indices
            partition = jnp.digitize(xmap_b, v_span) * mask_b
            partition = jnp.where(partition > num_pts, 0, partition)
            partition = jnp.where((xmap_b == v_max) * mask_b, num_pts, partition)

            return partition, v_list

        _PARTITION_JIT_CACHE[cache_key] = _partition_linear_core
        logger.debug("Created JIT-compiled linear partition function")

    return _PARTITION_JIT_CACHE[cache_key]


def _get_partition_log_jit():
    """Get or create JIT-compiled logarithmic partition function.

    Returns a JIT-compiled function if JAX backend is active,
    otherwise returns None.

    Note: num_pts must be passed as a Python int (static argument) because
    JAX's logspace requires a concrete value for the number of points.
    """
    global _PARTITION_JIT_CACHE

    backend = get_backend()
    if backend.name != "jax":
        return None

    cache_key = "partition_log_jit"
    if cache_key not in _PARTITION_JIT_CACHE:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _partition_log_core(mask_b, xmap_b, v_min, v_max, v_span):
            """JIT-compiled logarithmic partition core computation.

            Note: v_span is pre-computed outside JIT since logspace needs
            a concrete value for num_pts.
            """
            num_pts = v_span.shape[0] - 1
            v_list = jnp.sqrt(v_span[1:] * v_span[:-1])

            # Digitize: find bin indices
            partition = jnp.digitize(xmap_b, v_span) * mask_b
            partition = jnp.where(partition > num_pts, 0, partition)
            partition = jnp.where((xmap_b == v_max) * mask_b, num_pts, partition)

            return partition, v_list

        _PARTITION_JIT_CACHE[cache_key] = _partition_log_core
        logger.debug("Created JIT-compiled logarithmic partition function")

    return _PARTITION_JIT_CACHE[cache_key]


def _get_phi_transform_jit():
    """Get or create JIT-compiled phi angle transformation function.

    Returns a JIT-compiled function if JAX backend is active,
    otherwise returns None.
    """
    global _PARTITION_JIT_CACHE

    backend = get_backend()
    if backend.name != "jax":
        return None

    cache_key = "phi_transform_jit"
    if cache_key not in _PARTITION_JIT_CACHE:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _phi_transform_core(xmap_b, phi_offset, symmetry_fold):
            """JIT-compiled phi angle transformation."""
            # Apply phi offset
            angle_rad = jnp.deg2rad(xmap_b + phi_offset)
            xmap_transformed = jnp.rad2deg(
                jnp.arctan2(jnp.sin(angle_rad), jnp.cos(angle_rad))
            )

            # Apply symmetry folding
            unit_xmap = (xmap_transformed < (360.0 / symmetry_fold)) * (
                xmap_transformed >= 0
            )
            xmap_folded = xmap_transformed + 180.0
            xmap_folded = jnp.mod(xmap_folded, 360.0 / symmetry_fold)

            return xmap_folded, unit_xmap

        _PARTITION_JIT_CACHE[cache_key] = _phi_transform_core
        logger.debug("Created JIT-compiled phi transform function")

    return _PARTITION_JIT_CACHE[cache_key]


def _generate_partition_backend(
    map_name: str,
    mask: np.ndarray,
    xmap: np.ndarray,
    num_pts: int,
    style: str = "linear",
    phi_offset: float | None = None,
    symmetry_fold: int = 1,
) -> dict[str, str | int | np.ndarray]:
    """Backend-accelerated partition generation.

    Uses the backend abstraction layer for GPU acceleration when available.
    JIT compilation is applied for repeated calls with JAX backend.
    """
    backend = get_backend()

    # Convert inputs to backend arrays
    mask_b = backend.array(mask)
    xmap_b = backend.array(xmap)
    xmap_phi = None
    unit_xmap = None

    if map_name == "phi":
        xmap_phi = xmap_b
        if phi_offset is not None:
            # xmap = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(xmap + phi_offset))))
            angle_rad = backend.deg2rad(xmap_b + phi_offset)
            # Complex exponential angle extraction
            xmap_b = backend.rad2deg(
                backend.arctan2(backend.sin(angle_rad), backend.cos(angle_rad))
            )
        if symmetry_fold > 1:
            unit_xmap = (xmap_b < (360 / symmetry_fold)) * (xmap_b >= 0)
            xmap_b = xmap_b + 180.0
            xmap_b = backend.mod(xmap_b, 360.0 / symmetry_fold)

    roi = mask_b > 0

    # Use where to extract valid values for min/max computation
    valid_values = backend.where(roi, xmap_b, backend.array(float("nan")))
    v_min = float(backend.nanmin(valid_values))
    v_max = float(backend.nanmax(valid_values))

    # Try to use JIT-compiled versions for JAX backend
    if map_name == "q" and style == "logarithmic":
        mask_b = mask_b * (xmap_b > 0)
        valid_xmap = backend.where(mask_b > 0, xmap_b, backend.array(float("nan")))
        v_min_check = backend.nanmin(valid_xmap)
        if backend.isnan(v_min_check) or float(v_min_check) <= 0:
            raise ValueError(
                "Invalid xmap values for logarithmic binning. All values are non-positive."
            )
        v_min = float(backend.nanmin(valid_xmap))
        xmap_b = backend.where(xmap_b > 0, xmap_b, backend.array(float("nan")))

        # Pre-compute v_span (needs concrete num_pts value)
        v_span = backend.logspace(
            backend.log10(backend.array(v_min)),
            backend.log10(backend.array(v_max)),
            num_pts + 1,
        )

        # Try JIT-compiled logarithmic partition
        jit_fn = _get_partition_log_jit()
        if jit_fn is not None:
            import jax.numpy as jnp

            mask_jax = jnp.asarray(mask_b)
            xmap_jax = jnp.asarray(xmap_b)
            v_span_jax = jnp.asarray(v_span)
            partition, v_list = jit_fn(
                mask_jax, xmap_jax, jnp.asarray(v_min), jnp.asarray(v_max), v_span_jax
            )
        else:
            # Non-JIT path
            v_list = backend.sqrt(v_span[1:] * v_span[:-1])
            partition = backend.digitize(xmap_b, v_span) * mask_b
            partition = backend.where(partition > num_pts, backend.array(0), partition)
            partition = backend.where(
                (xmap_b == v_max) * mask_b, backend.array(num_pts), partition
            )
    else:
        # Pre-compute v_span (needs concrete num_pts value)
        v_span = backend.linspace(v_min, v_max, num_pts + 1)

        # Try JIT-compiled linear partition
        jit_fn = _get_partition_linear_jit()
        if jit_fn is not None:
            import jax.numpy as jnp

            mask_jax = jnp.asarray(mask_b)
            xmap_jax = jnp.asarray(xmap_b)
            v_span_jax = jnp.asarray(v_span)
            partition, v_list = jit_fn(
                mask_jax, xmap_jax, jnp.asarray(v_min), jnp.asarray(v_max), v_span_jax
            )
        else:
            # Non-JIT path
            v_list = (v_span[1:] + v_span[:-1]) / 2.0
            partition = backend.digitize(xmap_b, v_span) * mask_b
            partition = backend.where(partition > num_pts, backend.array(0), partition)
            partition = backend.where(
                (xmap_b == v_max) * mask_b, backend.array(num_pts), partition
            )

    # Convert to NumPy for output (I/O boundary)
    partition_np = ensure_numpy(partition).astype(np.uint32)
    v_list_np = ensure_numpy(v_list)

    if map_name == "phi" and symmetry_fold > 1 and unit_xmap is not None:
        # Use NumPy for bincount (complex operation)
        unit_xmap_np = ensure_numpy(unit_xmap)
        partition_np_i64 = partition_np.astype(np.int64)
        xmap_phi_np = ensure_numpy(xmap_phi)
        idx_map = (unit_xmap_np * partition_np_i64).astype(np.int64)
        sum_value = np.bincount(idx_map.flatten(), weights=xmap_phi_np.flatten())
        norm_factor = np.bincount(idx_map.flatten())
        v_list_np = sum_value / np.clip(norm_factor, 1, None)
        v_list_np = v_list_np[1:]

    return {
        "map_name": map_name,
        "num_pts": num_pts,
        "partition": partition_np,
        "v_list": v_list_np,
    }


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
    return _generate_partition_backend(
        map_name, mask, xmap, num_pts, style, phi_offset, symmetry_fold
    )


def _combine_partitions_backend(
    pack1: dict[str, str | int | np.ndarray],
    pack2: dict[str, str | int | np.ndarray],
    prefix: str = "dynamic",
) -> dict[str, list | np.ndarray]:
    """Backend-accelerated partition combination.

    Uses the backend abstraction layer for GPU acceleration when available.
    """
    backend = get_backend()

    p1 = backend.array(pack1["partition"].astype(np.int64))
    p2 = backend.array(pack2["partition"].astype(np.int64))
    num_pts2 = int(pack2["num_pts"])

    # Convert to zero-based indexing, merge, convert back
    partition = (p1 - 1) * num_pts2 + (p2 - 1) + 1

    # Clip to ensure non-negative
    partition = backend.clip(partition, 0, None)

    # Convert to NumPy for unique operation (complex)
    partition_np = ensure_numpy(partition).astype(np.int64)

    start_index = np.min(partition_np)
    unique_idx, inverse = np.unique(partition_np, return_inverse=True)
    partition_natural_order = inverse.reshape(partition_np.shape).astype(np.uint32)

    # Shift if needed to preserve masked pixel indicator (0)
    if start_index > 0:
        partition_natural_order += 1

    return {
        f"{prefix}_num_pts": [pack1["num_pts"], pack2["num_pts"]],
        f"{prefix}_roi_map": partition_natural_order,
        f"{prefix}_v_list_dim0": ensure_numpy(pack1["v_list"]),
        f"{prefix}_v_list_dim1": ensure_numpy(pack2["v_list"]),
        f"{prefix}_index_mapping": unique_idx[unique_idx >= 1] - 1,
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
    return _combine_partitions_backend(pack1, pack2, prefix)


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


def create_partition(
    qmap: np.ndarray,
    mask: np.ndarray,
    n_bins: int = 36,
    spacing: str = "linear",
) -> dict[str, str | int | np.ndarray]:
    """Create a Q-space partition from a Q-map.

    Convenience wrapper around generate_partition for simple Q-binning.

    Args:
        qmap: 2D array of Q-values (momentum transfer)
        mask: 2D boolean mask array (True = valid)
        n_bins: Number of partition bins (default 36)
        spacing: Binning style - "linear" or "log" (default "linear")

    Returns:
        Dictionary with keys:
            - map_name: "q"
            - num_pts: Number of bins
            - partition: 2D array of bin labels (1-indexed, 0=masked)
            - v_list: Array of bin center Q-values
    """
    # Map "log" to "logarithmic" for internal function
    style = "logarithmic" if spacing == "log" else spacing

    return generate_partition(
        map_name="q",
        mask=mask,
        xmap=qmap,
        num_pts=n_bins,
        style=style,
    )
