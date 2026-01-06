"""Q-map computation for SimpleMask.

This module provides functions to compute momentum transfer (Q) maps
for transmission and reflection geometries based on detector geometry parameters.

Ported from pySimpleMask with minimal modifications.
"""

import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# Energy to wavevector constant: lambda (Angstrom) = 12.39841984 / E (keV)
E2KCONST = 12.39841984


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
    """
    if stype == "Transmission":
        return compute_transmission_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
        )
    elif stype == "Reflection":
        return compute_reflection_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            alpha_i_deg=metadata.get("alpha_i_deg", 0.14),
            orientation=metadata.get("orientation", "north"),
        )
    else:
        raise ValueError(f"Unknown scattering type: {stype}")


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
    # Wavevector magnitude: k0 = 2*pi/lambda
    k0 = 2 * np.pi / (E2KCONST / energy)

    # Create pixel coordinate grids relative to beam center
    # Use signed int to handle center positions properly
    v = np.arange(shape[0], dtype=np.int32) - int(center[0])
    h = np.arange(shape[1], dtype=np.int32) - int(center[1])
    vg, hg = np.meshgrid(v, h, indexing="ij")

    # Radial distance in real space (mm)
    r = np.hypot(vg, hg) * pix_dim

    # Azimuthal angle (negated for convention)
    phi = np.arctan2(vg, hg) * (-1)

    # Scattering angle
    alpha = np.arctan(r / det_dist)

    # Q components
    qr = np.sin(alpha) * k0
    qx = qr * np.cos(phi)
    qy = qr * np.sin(phi)
    phi = np.rad2deg(phi)

    # Keep phi and q as np.float64 to preserve precision
    qmap = {
        "phi": phi,
        "TTH": alpha.astype(np.float32),
        "q": qr,
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "x": hg,
        "y": vg,
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
    k0 = 2 * np.pi / (E2KCONST / energy)

    # Use signed int for reflection mode since we need negative values
    v = np.arange(shape[0], dtype=np.int32) - int(center[0])
    h = np.arange(shape[1], dtype=np.int32) - int(center[1])
    vg, hg = np.meshgrid(v, h, indexing="ij")
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

    r = np.hypot(vg, hg) * pix_dim
    phi = np.arctan2(vg, hg)
    tth_full = np.arctan(r / det_dist)

    alpha_i = np.deg2rad(alpha_i_deg)
    alpha_f = np.arctan(vg * pix_dim / det_dist) - alpha_i
    tth = np.arctan(hg * pix_dim / det_dist)

    # Q components for reflection geometry
    qx = k0 * (np.cos(alpha_f) * np.cos(tth) - np.cos(alpha_i))
    qy = k0 * (np.cos(alpha_f) * np.sin(tth))
    qz = k0 * (np.sin(alpha_i) + np.sin(alpha_f))
    qr = np.hypot(qx, qy)
    q = np.hypot(qr, qz)

    qmap = {
        "phi": phi,
        "TTH": tth_full,
        "tth": tth,
        "alpha_f": alpha_f,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qr": qr,
        "q": q,
        "x": hg,
        "y": vg * (-1),
    }

    # Convert angles to degrees
    for key in ["phi", "TTH", "tth", "alpha_f"]:
        qmap[key] = np.rad2deg(qmap[key])

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
