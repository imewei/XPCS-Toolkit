"""Q-map computation for SimpleMask.

This module provides functions to compute momentum transfer (Q) maps
for transmission and reflection geometries based on detector geometry parameters.

Uses backend abstraction for GPU acceleration when available.
Ported from pySimpleMask with backend abstraction for JAX support.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from xpcsviewer.backends import get_backend
from xpcsviewer.backends._conversions import ensure_numpy

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Energy to wavevector constant: lambda (Angstrom) = 12.39841984 / E (keV)
E2KCONST = 12.39841984

# JIT cache for compiled functions (JAX arrays are not hashable for lru_cache)
_JIT_CACHE: dict[str, callable] = {}


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
    # Validate required parameters before attempting computation
    required = ("energy", "bcx", "bcy", "shape", "pix_dim", "det_dist")
    _validate_geometry_metadata(metadata, required)

    if stype == "Transmission":
        return compute_transmission_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
        )
    if stype == "Reflection":
        return compute_reflection_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            alpha_i_deg=metadata.get("alpha_i_deg", 0.14),
            orientation=metadata.get("orientation", "north"),
        )
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

            # Q components
            qr = jnp.sin(alpha) * k0
            qx = qr * jnp.cos(phi)
            qy = qr * jnp.sin(phi)
            phi_deg = jnp.rad2deg(phi)

            return phi_deg, alpha, qr, qx, qy, hg, vg

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
) -> float:
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

    return float(q) if backend.name != "jax" else q


def compute_q_sum_squared(
    center_x: float,
    center_y: float,
    pixel_positions: list[tuple[float, float]],
    energy: float,
    pix_dim: float,
    det_dist: float,
) -> float:
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

        q_sum_sq = 0.0
        for px, py in pixel_positions:
            dx = px - center_x
            dy = py - center_y
            r = jnp.sqrt(dx**2 + dy**2) * pix_dim
            alpha = jnp.arctan(r / det_dist)
            q = jnp.sin(alpha) * k0
            q_sum_sq = q_sum_sq + q**2

        return q_sum_sq
    else:
        # NumPy fallback
        total = 0.0
        for px, py in pixel_positions:
            q = compute_q_at_pixel(
                center_x, center_y, px, py, energy, pix_dim, det_dist
            )
            total += q**2
        return total


def create_q_objective(
    target_q_values: np.ndarray,
    pixel_positions: list[tuple[float, float]],
    energy: float,
    pix_dim: float,
) -> callable:
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

    def objective(center_x, center_y, det_dist):
        """Compute sum of squared Q differences."""
        loss = 0.0
        for i, (px, py) in enumerate(pixel_positions):
            dx = px - center_x
            dy = py - center_y
            r = jnp.sqrt(dx**2 + dy**2) * pix_dim
            alpha = jnp.arctan(r / det_dist)
            q_pred = jnp.sin(alpha) * k0
            loss = loss + (q_pred - target_q[i]) ** 2
        return loss

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

    # Convert all arrays to NumPy for output (I/O boundary)
    qmap = {
        "phi": ensure_numpy(phi_deg),
        "TTH": ensure_numpy(alpha).astype(np.float32),
        "q": ensure_numpy(qr),
        "qx": ensure_numpy(qx).astype(np.float32),
        "qy": ensure_numpy(qy).astype(np.float32),
        "x": ensure_numpy(hg).astype(np.int32),
        "y": ensure_numpy(vg).astype(np.int32),
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "q": "Å⁻¹",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=128)
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
        - TTH: Two-theta angle (radians stored as float32)
        - q: Momentum transfer magnitude (Angstrom^-1)
        - qx, qy: Q components (Angstrom^-1)
        - x, y: Pixel coordinates

        qmap_unit contains unit strings for each map.
    """
    return _compute_transmission_qmap_backend(energy, center, shape, pix_dim, det_dist)


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
    """
    backend = get_backend()

    k0 = 2 * backend.pi / (E2KCONST / energy)

    # Create coordinate arrays
    v = backend.arange(shape[0], dtype=np.float64) - center[0]
    h = backend.arange(shape[1], dtype=np.float64) - center[1]
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

    alpha_i = backend.deg2rad(backend.array(alpha_i_deg))
    alpha_f = backend.arctan(vg * pix_dim / det_dist) - alpha_i
    tth = backend.arctan(hg * pix_dim / det_dist)

    # Q components for reflection geometry
    qx = k0 * (backend.cos(alpha_f) * backend.cos(tth) - backend.cos(alpha_i))
    qy = k0 * (backend.cos(alpha_f) * backend.sin(tth))
    qz = k0 * (backend.sin(alpha_i) + backend.sin(alpha_f))
    qr = backend.hypot(qx, qy)
    q = backend.hypot(qr, qz)

    # Convert to NumPy for output (I/O boundary)
    qmap = {
        "phi": ensure_numpy(backend.rad2deg(phi)),
        "TTH": ensure_numpy(backend.rad2deg(tth_full)),
        "tth": ensure_numpy(backend.rad2deg(tth)),
        "alpha_f": ensure_numpy(backend.rad2deg(alpha_f)),
        "qx": ensure_numpy(qx),
        "qy": ensure_numpy(qy),
        "qz": ensure_numpy(qz),
        "qr": ensure_numpy(qr),
        "q": ensure_numpy(q),
        "x": ensure_numpy(hg).astype(np.int32),
        "y": ensure_numpy(-vg).astype(np.int32),
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "tth": "deg",
        "alpha_f": "deg",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "qz": "Å⁻¹",
        "qr": "Å⁻¹",
        "q": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=128)
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
    return _compute_reflection_qmap_backend(
        energy, center, shape, pix_dim, det_dist, alpha_i_deg, orientation
    )
