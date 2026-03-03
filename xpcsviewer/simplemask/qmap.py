"""Q-map computation for SimpleMask.

This module provides functions to compute momentum transfer (Q) maps
for transmission and reflection geometries based on detector geometry parameters.

Uses backend abstraction for GPU acceleration when available.
Ported from pySimpleMask with backend abstraction for JAX support.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy
from xpcsviewer.fileIO.qmap_utils import Q_UNIT_DISPLAY

logger = logging.getLogger(__name__)

# Energy to wavevector constant: lambda (Angstrom) = 12.39841984 / E (keV)
E2KCONST = 12.39841984

# JIT cache for compiled functions (JAX arrays are not hashable for lru_cache).
# This is a bounded cache: entries are evicted in FIFO order when the cache
# exceeds _JIT_CACHE_MAXSIZE.  External code should NOT clear this dict
# directly; the eviction logic is handled internally.
_JIT_CACHE: dict[str, Callable] = {}
_JIT_CACHE_MAXSIZE: int = 32


def _validate_geometry_metadata(metadata: dict, required_keys: tuple[str, ...]) -> None:
    """Validate that required geometry parameters are present and valid.

    Args:
        metadata: Dictionary containing geometry parameters
        required_keys: Tuple of required parameter names

    Raises:
        ValueError: If any required parameter is missing (None) or invalid
    """
    missing = []
    for key in required_keys:
        value = metadata.get(key)
        if value is None:
            missing.append(key)

    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            f"Cannot compute Q-map: missing required geometry parameter(s): {missing_str}. "
            f"Please ensure HDF file contains detector geometry metadata or set values "
            f"manually in the Mask Editor geometry panel."
        )


def compute_qmap(
    stype: str, metadata: dict
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute Q-map based on scattering geometry type.

    Args:
        stype: Scattering type - "Transmission" or "Reflection"
        metadata: Dictionary containing geometry parameters:
            - energy: X-ray energy in keV
            - bcx: Beam center X (column) in pixels
            - bcy: Beam center Y (row) in pixels
            - shape: Detector shape (height, width)
            - pix_dim: Pixel size in mm
            - det_dist: Sample-to-detector distance in mm
            - alpha_i_deg: Incident angle in degrees (reflection only)
            - orientation: Detector orientation (reflection only)

    Returns:
        Tuple of (qmap_dict, units_dict) where qmap_dict contains arrays
        for various Q-space coordinates and units_dict contains their units.

    Raises:
        ValueError: If required geometry parameters are missing (None)
    """
    logger.debug(f"compute_qmap: entry stype={stype}, shape={metadata.get('shape')}")
    # Validate required parameters before attempting computation
    required = ("energy", "bcx", "bcy", "shape", "pix_dim", "det_dist")
    _validate_geometry_metadata(metadata, required)

    if stype == "Transmission":
        result = compute_transmission_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
        )
        if logger.isEnabledFor(logging.DEBUG):
            sqmap = result[0].get("sqmap")
            if sqmap is not None:
                logger.debug(f"compute_qmap: exit sqmap shape={sqmap.shape}")
        return result
    if stype == "Reflection":
        result = compute_reflection_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            alpha_i_deg=metadata.get("alpha_i_deg", 0.14),
            orientation=metadata.get("orientation", "north"),
        )
        if logger.isEnabledFor(logging.DEBUG):
            sqmap = result[0].get("sqmap")
            if sqmap is not None:
                logger.debug(f"compute_qmap: exit sqmap shape={sqmap.shape}")
        return result
    raise ValueError(f"Unknown scattering type: {stype}")


def _get_transmission_qmap_jit():
    """Get or create JIT-compiled transmission Q-map function.

    Returns a JIT-compiled function if JAX backend is active,
    otherwise returns None.
    """
    global _JIT_CACHE

    backend = get_backend()
    if backend.name != "jax":
        return None

    cache_key = "transmission_qmap_jit"
    if cache_key not in _JIT_CACHE:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _transmission_qmap_core(k0, v, h, pix_dim, det_dist):
            """JIT-compiled core Q-map computation."""
            vg, hg = jnp.meshgrid(v, h, indexing="ij")

            # Radial distance in real space (mm)
            r = jnp.hypot(vg, hg) * pix_dim

            # Azimuthal angle (negated for convention)
            phi = jnp.arctan2(vg, hg) * (-1)

            # Scattering angle
            alpha = jnp.arctan(r / det_dist)
            alpha_deg = jnp.rad2deg(alpha)

            # Q components
            qr = jnp.sin(alpha) * k0
            qx = qr * jnp.cos(phi)
            qy = qr * jnp.sin(phi)
            phi_deg = jnp.rad2deg(phi)

            return phi_deg, alpha_deg, qr, qx, qy, hg, vg

        if len(_JIT_CACHE) >= _JIT_CACHE_MAXSIZE:
            oldest_key = next(iter(_JIT_CACHE))
            del _JIT_CACHE[oldest_key]
            logger.debug("Evicted oldest JIT cache entry: %s", oldest_key)
        _JIT_CACHE[cache_key] = _transmission_qmap_core
        logger.debug("Created JIT-compiled transmission Q-map function")

    return _JIT_CACHE[cache_key]


def compute_q_at_pixel(
    center_x: float,
    center_y: float,
    pixel_x: float,
    pixel_y: float,
    energy: float,
    pix_dim: float,
    det_dist: float,
) -> float | Any:
    """Compute Q value at a single pixel (differentiable).

    This function is differentiable with respect to center_x, center_y,
    and det_dist when using the JAX backend, enabling gradient-based
    calibration optimization.

    Args:
        center_x: Beam center X position (column) in pixels
        center_y: Beam center Y position (row) in pixels
        pixel_x: Pixel X position (column)
        pixel_y: Pixel Y position (row)
        energy: X-ray energy in keV
        pix_dim: Pixel dimension in mm
        det_dist: Sample-to-detector distance in mm

    Returns:
        Q value at the pixel position in Å⁻¹

    Example:
        >>> import jax
        >>> from xpcsviewer.simplemask.qmap import compute_q_at_pixel
        >>> # Compute Q
        >>> q = compute_q_at_pixel(128.0, 128.0, 200.0, 200.0, 10.0, 0.075, 5000.0)
        >>> # Compute gradient with respect to beam center
        >>> grad_fn = jax.grad(compute_q_at_pixel, argnums=(0, 1))
        >>> dq_dcx, dq_dcy = grad_fn(128.0, 128.0, 200.0, 200.0, 10.0, 0.075, 5000.0)
    """
    backend = get_backend()

    # Guard against zero or negative det_dist to prevent division by zero (BUG-051).
    det_dist = max(det_dist, 1e-12)

    # Wavevector magnitude: k0 = 2*pi/lambda, lambda = 12.39841984/E
    k0 = 2 * backend.pi / (E2KCONST / energy)

    # Distance from beam center
    dx = pixel_x - center_x
    dy = pixel_y - center_y

    # Radial distance in real space (mm)
    r = backend.sqrt(dx**2 + dy**2) * pix_dim

    # Scattering angle
    alpha = backend.arctan(r / det_dist)

    # Q magnitude
    q = backend.sin(alpha) * k0

    # Always return a Python float for consistent display behavior.
    # For gradient use cases, use compute_q_sum_squared or inline JAX computation.
    return float(q)


def compute_q_sum_squared(
    center_x: float,
    center_y: float,
    pixel_positions: list[tuple[float, float]],
    energy: float,
    pix_dim: float,
    det_dist: float,
) -> float | Any:
    """Compute sum of squared Q values at given pixels (differentiable).

    This function is useful for gradient-based calibration objectives.
    It is differentiable with respect to center_x, center_y, and det_dist
    when using the JAX backend.

    Args:
        center_x: Beam center X position (column) in pixels
        center_y: Beam center Y position (row) in pixels
        pixel_positions: List of (x, y) pixel positions
        energy: X-ray energy in keV
        pix_dim: Pixel dimension in mm
        det_dist: Sample-to-detector distance in mm

    Returns:
        Sum of Q² values at all pixel positions
    """
    backend = get_backend()

    if backend.name == "jax":
        import jax.numpy as jnp

        # Wavevector magnitude
        k0 = 2 * jnp.pi / (E2KCONST / energy)

        # Vectorized computation instead of Python for-loop so that
        # jax.grad / jax.jit can trace through without unrolling per-element
        # Python loops (BUG-057).
        positions_arr = jnp.array(pixel_positions)  # shape (N, 2)
        dx = positions_arr[:, 0] - center_x
        dy = positions_arr[:, 1] - center_y
        r = jnp.sqrt(dx**2 + dy**2) * pix_dim
        alpha = jnp.arctan(r / det_dist)
        q = jnp.sin(alpha) * k0
        return jnp.sum(q**2)
    # NumPy fallback
    positions_arr = np.array(pixel_positions)  # type: ignore[assignment]  # shape (N, 2)
    q_vals: Any = np.array(
        [
            compute_q_at_pixel(
                center_x, center_y, float(px), float(py), energy, pix_dim, det_dist
            )
            for px, py in positions_arr
        ]
    )
    return float(np.sum(q_vals**2))


def create_q_objective(
    target_q_values: np.ndarray,
    pixel_positions: list[tuple[float, float]],
    energy: float,
    pix_dim: float,
) -> Callable:
    """Create a differentiable objective function for Q-map calibration.

    Creates an objective function that measures the squared difference
    between predicted and target Q values. The objective is differentiable
    with respect to beam center and detector distance.

    Args:
        target_q_values: Array of target Q values at each position
        pixel_positions: List of (x, y) pixel coordinates
        energy: X-ray energy in keV
        pix_dim: Pixel dimension in mm

    Returns:
        Callable objective function: f(center_x, center_y, det_dist) -> loss

    Example:
        >>> import jax
        >>> from xpcsviewer.simplemask.qmap import create_q_objective
        >>> objective = create_q_objective(target_q, positions, 10.0, 0.075)
        >>> # Compute loss
        >>> loss = objective(128.0, 128.0, 5000.0)
        >>> # Compute gradient
        >>> grad_fn = jax.grad(objective, argnums=(0, 1, 2))
        >>> dcx, dcy, ddist = grad_fn(128.0, 128.0, 5000.0)

    Raises:
        RuntimeError: If JAX backend is not available.
    """
    backend = get_backend()
    if backend.name != "jax":
        raise RuntimeError(
            "Q-map calibration objective requires JAX backend. "
            "Set XPCS_USE_JAX=1 to enable."
        )

    import jax.numpy as jnp

    target_q = jnp.array(target_q_values)
    k0 = 2 * jnp.pi / (E2KCONST / energy)
    # Pre-convert pixel positions to a JAX array so the objective is fully
    # vectorised and traceable without a Python for-loop (BUG-057).
    positions_arr = jnp.array(pixel_positions)  # shape (N, 2)

    def objective(center_x, center_y, det_dist):
        """Compute sum of squared Q differences (vectorised, JIT-traceable)."""
        dx = positions_arr[:, 0] - center_x
        dy = positions_arr[:, 1] - center_y
        r = jnp.sqrt(dx**2 + dy**2) * pix_dim
        alpha = jnp.arctan(r / det_dist)
        q_pred = jnp.sin(alpha) * k0
        return jnp.sum((q_pred - target_q) ** 2)

    return objective


def _compute_transmission_qmap_backend(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Backend-accelerated transmission Q-map computation.

    Uses the backend abstraction layer for GPU acceleration when available.
    JIT compilation is applied for repeated calls with JAX backend.
    """
    # Guard against zero or negative det_dist to prevent division by zero (BUG-051).
    det_dist = max(det_dist, 1e-12)

    backend = get_backend()

    # Wavevector magnitude: k0 = 2*pi/lambda
    k0 = 2 * backend.pi / (E2KCONST / energy)

    # Create pixel coordinate arrays
    v = backend.arange(shape[0], dtype=np.float64) - center[0]
    h = backend.arange(shape[1], dtype=np.float64) - center[1]

    # Try to use JIT-compiled version for JAX backend
    jit_fn = _get_transmission_qmap_jit()
    if jit_fn is not None:
        import jax.numpy as jnp

        # Convert to JAX arrays for JIT function
        k0_jax = jnp.asarray(k0)
        v_jax = jnp.asarray(v)
        h_jax = jnp.asarray(h)
        pix_dim_jax = jnp.asarray(pix_dim)
        det_dist_jax = jnp.asarray(det_dist)

        phi_deg, alpha, qr, qx, qy, hg, vg = jit_fn(
            k0_jax, v_jax, h_jax, pix_dim_jax, det_dist_jax
        )
    else:
        # Non-JIT path for NumPy backend
        vg, hg = backend.meshgrid(v, h, indexing="ij")

        # Radial distance in real space (mm)
        r = backend.hypot(vg, hg) * pix_dim

        # Azimuthal angle (negated for convention)
        phi = backend.arctan2(vg, hg) * (-1)

        # Scattering angle
        alpha = backend.arctan(r / det_dist)

        # Q components
        qr = backend.sin(alpha) * k0
        qx = qr * backend.cos(phi)
        qy = qr * backend.sin(phi)
        phi_deg = backend.rad2deg(phi)

    # Create absolute pixel-index meshgrids (0..N-1) for status bar display.
    # These are independent of beam center, unlike hg/vg which are offsets.
    pix_y, pix_x = np.meshgrid(
        np.arange(shape[0], dtype=np.int32),
        np.arange(shape[1], dtype=np.int32),
        indexing="ij",
    )

    # Convert alpha to degrees (JIT path already returns degrees,
    # non-JIT path returns radians)
    if jit_fn is None:
        alpha = backend.rad2deg(alpha)

    # Convert all arrays to NumPy for output (I/O boundary)
    qmap = {
        "phi": ensure_numpy(phi_deg),
        "TTH": ensure_numpy(alpha),
        "q": ensure_numpy(qr),
        "qx": ensure_numpy(qx),
        "qy": ensure_numpy(qy),
        "x": pix_x,
        "y": pix_y,
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "q": Q_UNIT_DISPLAY,
        "qx": Q_UNIT_DISPLAY,
        "qy": Q_UNIT_DISPLAY,
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=4)  # Reduced from 128 to limit memory usage
def _compute_transmission_qmap_cached(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Cached inner computation for transmission Q-map."""
    return _compute_transmission_qmap_backend(energy, center, shape, pix_dim, det_dist)


def compute_transmission_qmap(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute Q-map for transmission geometry.

    Args:
        energy: X-ray energy in keV
        center: Beam center as (row, column) in pixels
        shape: Detector shape as (height, width)
        pix_dim: Pixel dimension in mm
        det_dist: Sample-to-detector distance in mm

    Returns:
        Tuple of (qmap, qmap_unit) dictionaries.

        qmap contains:

        - phi: Azimuthal angle (degrees)
        - TTH: Two-theta angle (degrees)
        - q: Momentum transfer magnitude (Angstrom^-1)
        - qx, qy: Q components (Angstrom^-1)
        - x, y: Pixel coordinates

        qmap_unit contains unit strings for each map.
    """
    # Return the cached dict directly -- no deep copy needed (BUG-055).
    # The arrays inside the dict are NumPy arrays produced once and never mutated
    # in place, so sharing them across callers is safe.  Callers that need their
    # own writable copy must call np.copy() themselves.
    qmap, qmap_unit = _compute_transmission_qmap_cached(
        energy, center, shape, pix_dim, det_dist
    )
    return qmap, qmap_unit


# Expose lru_cache methods on the public wrapper for tests/callers
compute_transmission_qmap.cache_clear = _compute_transmission_qmap_cached.cache_clear  # type: ignore[attr-defined]
compute_transmission_qmap.cache_info = _compute_transmission_qmap_cached.cache_info  # type: ignore[attr-defined]


def _get_reflection_qmap_jit(orientation: str):
    """Get or create JIT-compiled reflection Q-map function.

    Returns a JIT-compiled function if JAX backend is active,
    otherwise returns None. Orientation is used as a cache key
    since it affects control flow.
    """
    global _JIT_CACHE

    backend = get_backend()
    if backend.name != "jax":
        return None

    cache_key = f"reflection_qmap_jit_{orientation}"
    if cache_key not in _JIT_CACHE:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _reflection_qmap_core(k0, v, h, pix_dim, det_dist, alpha_i_deg):
            """JIT-compiled core reflection Q-map computation."""
            vg, hg = jnp.meshgrid(v, h, indexing="ij")
            vg = vg * (-1)

            # Orientation transformation (baked into this compiled variant)
            if orientation == "west":
                vg, hg = -hg, vg
            elif orientation == "south":
                vg, hg = -vg, -hg
            elif orientation == "east":
                vg, hg = hg, -vg
            # "north" is identity (no transform)

            r = jnp.hypot(vg, hg) * pix_dim
            phi = jnp.arctan2(vg, hg)
            tth_full = jnp.arctan(r / det_dist)

            alpha_i = jnp.deg2rad(alpha_i_deg)
            alpha_f = jnp.arctan(vg * pix_dim / det_dist) - alpha_i
            tth = jnp.arctan(hg * pix_dim / det_dist)

            # Q components for reflection geometry
            qx = k0 * (jnp.cos(alpha_f) * jnp.cos(tth) - jnp.cos(alpha_i))
            qy = k0 * (jnp.cos(alpha_f) * jnp.sin(tth))
            qz = k0 * (jnp.sin(alpha_i) + jnp.sin(alpha_f))
            qr = jnp.hypot(qx, qy)
            q = jnp.hypot(qr, qz)

            # Convert angular outputs to degrees
            phi_deg = jnp.rad2deg(phi)
            tth_full_deg = jnp.rad2deg(tth_full)
            tth_deg = jnp.rad2deg(tth)
            alpha_f_deg = jnp.rad2deg(alpha_f)

            return (
                phi_deg,
                tth_full_deg,
                tth_deg,
                alpha_f_deg,
                qx,
                qy,
                qz,
                qr,
                q,
                hg,
                vg,
            )

        if len(_JIT_CACHE) >= _JIT_CACHE_MAXSIZE:
            oldest_key = next(iter(_JIT_CACHE))
            del _JIT_CACHE[oldest_key]
            logger.debug("Evicted oldest JIT cache entry: %s", oldest_key)
        _JIT_CACHE[cache_key] = _reflection_qmap_core
        logger.debug(
            "Created JIT-compiled reflection Q-map function "
            f"(orientation={orientation})"
        )

    return _JIT_CACHE[cache_key]


def _compute_reflection_qmap_backend(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
    alpha_i_deg: float = 0.14,
    orientation: str = "north",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Backend-accelerated reflection Q-map computation.

    Uses the backend abstraction layer for GPU acceleration when available.
    JIT compilation is applied for repeated calls with JAX backend.
    """
    # Guard against zero or negative det_dist to prevent division by zero (BUG-051).
    det_dist = max(det_dist, 1e-12)

    backend = get_backend()

    # Wavevector magnitude
    k0 = 2 * backend.pi / (E2KCONST / energy)

    # Create pixel coordinate arrays
    v = backend.arange(shape[0], dtype=np.float64) - center[0]
    h = backend.arange(shape[1], dtype=np.float64) - center[1]

    # Try JIT-compiled path for JAX backend
    jit_fn = _get_reflection_qmap_jit(orientation)
    if jit_fn is not None:
        import jax.numpy as jnp

        k0_jax = jnp.asarray(k0)
        v_jax = jnp.asarray(v)
        h_jax = jnp.asarray(h)
        pix_dim_jax = jnp.asarray(pix_dim)
        det_dist_jax = jnp.asarray(det_dist)
        alpha_i_jax = jnp.asarray(alpha_i_deg)

        (
            phi,
            tth_full,
            tth,
            alpha_f,
            qx,
            qy,
            qz,
            qr,
            q,
            hg,
            vg,
        ) = jit_fn(k0_jax, v_jax, h_jax, pix_dim_jax, det_dist_jax, alpha_i_jax)
    else:
        # Non-JIT path for NumPy backend
        vg, hg = backend.meshgrid(v, h, indexing="ij")
        vg = vg * (-1)

        # Apply orientation transformation
        if orientation == "north":
            pass
        elif orientation == "west":
            vg, hg = -hg, vg
        elif orientation == "south":
            vg, hg = -vg, -hg
        elif orientation == "east":
            vg, hg = hg, -vg
        else:
            logger.warning(f"Unknown orientation: {orientation}. Using default north")

        r = backend.hypot(vg, hg) * pix_dim
        phi = backend.arctan2(vg, hg)
        tth_full = backend.arctan(r / det_dist)

        alpha_i: Any = backend.deg2rad(backend.array(alpha_i_deg))
        alpha_f = backend.arctan(vg * pix_dim / det_dist) - alpha_i
        tth = backend.arctan(hg * pix_dim / det_dist)

        # Q components for reflection geometry
        qx = k0 * (backend.cos(alpha_f) * backend.cos(tth) - backend.cos(alpha_i))
        qy = k0 * (backend.cos(alpha_f) * backend.sin(tth))
        qz = k0 * (backend.sin(alpha_i) + backend.sin(alpha_f))
        qr = backend.hypot(qx, qy)
        q = backend.hypot(qr, qz)

    # Convert to NumPy for output (I/O boundary)
    # When using JIT path, angular values are already in degrees.
    # When using non-JIT path, convert from radians to degrees.
    if jit_fn is not None:
        phi_deg = phi
        tth_full_deg = tth_full
        tth_deg = tth
        alpha_f_deg = alpha_f
    else:
        phi_deg = backend.rad2deg(phi)
        tth_full_deg = backend.rad2deg(tth_full)
        tth_deg = backend.rad2deg(tth)
        alpha_f_deg = backend.rad2deg(alpha_f)

    # Create absolute pixel-index meshgrids (0..N-1) for status bar display.
    # These are independent of beam center, unlike hg/vg which are offsets.
    pix_y, pix_x = np.meshgrid(
        np.arange(shape[0], dtype=np.int32),
        np.arange(shape[1], dtype=np.int32),
        indexing="ij",
    )

    qmap = {
        "phi": ensure_numpy(phi_deg),
        "TTH": ensure_numpy(tth_full_deg),
        "tth": ensure_numpy(tth_deg),
        "alpha_f": ensure_numpy(alpha_f_deg),
        "qx": ensure_numpy(qx),
        "qy": ensure_numpy(qy),
        "qz": ensure_numpy(qz),
        "qr": ensure_numpy(qr),
        "q": ensure_numpy(q),
        "x": pix_x,
        "y": pix_y,
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "tth": "deg",
        "alpha_f": "deg",
        "qx": Q_UNIT_DISPLAY,
        "qy": Q_UNIT_DISPLAY,
        "qz": Q_UNIT_DISPLAY,
        "qr": Q_UNIT_DISPLAY,
        "q": Q_UNIT_DISPLAY,
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=4)  # Reduced from 128 to limit memory usage
def _compute_reflection_qmap_cached(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
    alpha_i_deg: float = 0.14,
    orientation: str = "north",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Cached inner computation for reflection Q-map."""
    return _compute_reflection_qmap_backend(
        energy, center, shape, pix_dim, det_dist, alpha_i_deg, orientation
    )


def compute_reflection_qmap(
    energy: float,
    center: tuple[float, float],
    shape: tuple[int, int],
    pix_dim: float,
    det_dist: float,
    alpha_i_deg: float = 0.14,
    orientation: str = "north",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Compute Q-map for reflection (grazing incidence) geometry.

    Args:
        energy: X-ray energy in keV
        center: Beam center as (row, column) in pixels
        shape: Detector shape as (height, width)
        pix_dim: Pixel dimension in mm
        det_dist: Sample-to-detector distance in mm
        alpha_i_deg: Incident angle in degrees (default 0.14)
        orientation: Detector orientation - "north", "south", "east", "west"

    Returns:
        Tuple of (qmap, qmap_unit) dictionaries.

        qmap contains additional reflection-specific arrays:

        - qz, qr: Vertical and radial Q components
        - alpha_f: Exit angle
        - tth: In-plane two-theta
    """
    # Return the cached dict directly -- no deep copy needed (BUG-055).
    # Arrays are NumPy arrays produced once and never mutated in place.
    qmap, qmap_unit = _compute_reflection_qmap_cached(
        energy, center, shape, pix_dim, det_dist, alpha_i_deg, orientation
    )
    return qmap, qmap_unit


# Expose lru_cache methods on the public wrapper for tests/callers
compute_reflection_qmap.cache_clear = _compute_reflection_qmap_cached.cache_clear  # type: ignore[attr-defined]
compute_reflection_qmap.cache_info = _compute_reflection_qmap_cached.cache_info  # type: ignore[attr-defined]
