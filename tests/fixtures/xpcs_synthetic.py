"""XPCS-specific synthetic data generators for testing.

This module wraps and extends the base synthetic_data module with
XPCS-specific convenience functions for test fixtures.

Functions:
    generate_synthetic_g2_data: G2 correlation with known tau/amplitude
    generate_synthetic_c2_matrix: Two-time C2 matrix (symmetric)
    generate_synthetic_saxs_2d: Ring pattern SAXS 2D detector image
    generate_roi_parameters: ROI parameter sets for testing
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tests.fixtures.synthetic_data import (
    SyntheticXPCSGenerator,
    SyntheticXPCSParameters,
)
from tests.fixtures.synthetic_data import create_synthetic_g2_data as _create_g2


def generate_synthetic_g2_data(
    tau: float = 10e-3,
    amplitude: float = 0.5,
    baseline: float = 1.0,
    n_points: int = 50,
    noise_level: float = 0.01,
    double_exp: bool = False,
    tau2: float | None = None,
    amplitude2: float | None = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate synthetic G2 data with known parameters for fitting validation.

    Args:
        tau: Primary relaxation time in seconds (default: 10ms)
        amplitude: Primary amplitude/contrast (default: 0.5)
        baseline: Baseline value (default: 1.0)
        n_points: Number of delay time points (default: 50)
        noise_level: Relative noise level (default: 0.01)
        double_exp: If True, generate double exponential (default: False)
        tau2: Secondary relaxation time for double exponential
        tau2: Secondary amplitude for double exponential
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with keys:
            - delay_times: 1D array of tau values
            - g2_values: 1D array of g2 values
            - g2_errors: 1D array of error estimates
            - known_tau: Ground truth relaxation time(s)
            - known_amplitude: Ground truth amplitude(s)
            - baseline: Constant baseline value
    """
    np.random.seed(seed)

    if double_exp:
        # Double exponential
        n_components = 2
        tau2_val = tau2 if tau2 is not None else tau * 5.0
        amp2_val = amplitude2 if amplitude2 is not None else amplitude * 0.5
        params = SyntheticXPCSParameters(
            tau_min=1e-6,
            tau_max=1e2,
            n_tau=n_points,
            beta=amplitude,
            tau_c=tau,
            baseline=baseline,
            noise_level=noise_level,
            n_components=n_components,
            amplitudes=[amplitude, amp2_val],
            time_constants=[tau, tau2_val],
        )
        known_tau = [tau, tau2_val]
        known_amplitude = [amplitude, amp2_val]
    else:
        # Single exponential
        params = SyntheticXPCSParameters(
            tau_min=1e-6,
            tau_max=1e2,
            n_tau=n_points,
            beta=amplitude,
            tau_c=tau,
            baseline=baseline,
            noise_level=noise_level,
        )
        known_tau = tau
        known_amplitude = amplitude

    generator = SyntheticXPCSGenerator(params)
    generator._setup_random_state(seed)
    data = generator.generate_correlation_function()

    return {
        "delay_times": data["tau"],
        "g2_values": data["g2"],
        "g2_errors": data["g2_err"],
        "known_tau": known_tau,
        "known_amplitude": known_amplitude,
        "baseline": baseline,
        "g2_clean": data["g2_clean"],
    }


def generate_synthetic_c2_matrix(
    size: int = 100,
    tau_c: float = 10e-3,
    beta: float = 0.5,
    baseline: float = 1.0,
    noise_level: float = 0.05,
    correct_diagonal: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic two-time correlation (C2) matrix.

    Args:
        size: Matrix size (size x size) (default: 100)
        tau_c: Correlation time in seconds (default: 10ms)
        beta: Contrast parameter (default: 0.5)
        baseline: Baseline value (default: 1.0)
        noise_level: Relative noise level (default: 0.05)
        correct_diagonal: If True, apply diagonal correction (default: True)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with keys:
            - c2: 2D symmetric array (t1 x t2)
            - c2_uncorrected: C2 before diagonal correction (if applicable)
            - delay_times: 1D array of delay values
            - diagonal_corrected: Boolean indicating correction status
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    # Create time grid
    t_max = 10 * tau_c
    time_points = np.linspace(0, t_max, size)

    # Generate symmetric C2 matrix
    c2 = np.ones((size, size), dtype=np.float64)

    for i in range(size):
        for j in range(i, size):
            tau = abs(time_points[i] - time_points[j])
            value = baseline + beta * np.exp(-tau / tau_c)
            c2[i, j] = value
            c2[j, i] = value  # Symmetric

    # Add noise (symmetric)
    noise = rng.normal(0, noise_level * (c2 - baseline), c2.shape)
    noise = (noise + noise.T) / 2  # Make symmetric
    c2_noisy = c2 + noise
    c2_noisy = np.maximum(c2_noisy, 1.0)  # Ensure g2 >= 1

    c2_uncorrected = c2_noisy.copy()

    if correct_diagonal:
        # Diagonal correction: reduce diagonal values slightly
        # (simulates real TwoTime diagonal correction)
        diag_correction = 0.05 * beta
        np.fill_diagonal(c2_noisy, np.diag(c2_noisy) - diag_correction)

    return {
        "c2": c2_noisy,
        "c2_uncorrected": c2_uncorrected,
        "delay_times": time_points,
        "diagonal_corrected": correct_diagonal,
        "tau_c": tau_c,
        "beta": beta,
        "baseline": baseline,
    }


def generate_synthetic_saxs_2d(
    image_size: tuple[int, int] = (512, 512),
    center: tuple[int, int] | None = None,
    ring_q_values: list[float] | None = None,
    ring_widths: list[float] | None = None,
    background: float = 100.0,
    peak_intensity: float = 10000.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic SAXS 2D detector image with ring patterns.

    Args:
        image_size: Detector size (height, width) in pixels (default: 512x512)
        center: Beam center (y, x) in pixels (default: center of image)
        ring_q_values: Q values for rings in Å⁻¹ (default: [0.01, 0.05, 0.1])
        ring_widths: Ring widths in Å⁻¹ (default: [0.005, 0.01, 0.02])
        background: Background intensity (default: 100.0)
        peak_intensity: Peak ring intensity (default: 10000.0)
        noise_level: Relative Poisson noise factor (default: 0.1)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Dictionary with keys:
            - image: 2D array of pixel intensities
            - center: Tuple (y, x) of beam center
            - q_values: List of Q values for rings
            - pixel_size: Assumed pixel size in meters
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    height, width = image_size
    if center is None:
        center = (height // 2, width // 2)
    if ring_q_values is None:
        ring_q_values = [0.01, 0.05, 0.1]
    if ring_widths is None:
        ring_widths = [0.005, 0.01, 0.02]

    # Create coordinate grid
    y, x = np.ogrid[:height, :width]
    cy, cx = center
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Assume pixel size and detector distance to convert Q to pixels
    pixel_size = 75e-6  # 75 µm
    det_dist = 5.0  # 5 meters
    wavelength = 1.54e-10  # Cu Kα wavelength

    # Create image with background
    image = np.full((height, width), background, dtype=np.float64)

    # Add ring patterns
    for q_val, width_val in zip(ring_q_values, ring_widths, strict=False):
        # Convert Q to pixel radius: r = det_dist * tan(2*theta)
        # where sin(theta) = q * wavelength / (4 * pi)
        theta = np.arcsin(q_val * wavelength / (4 * np.pi))
        r_center = det_dist * np.tan(2 * theta) / pixel_size

        # Ring width in pixels
        theta_width = np.arcsin(width_val * wavelength / (4 * np.pi))
        r_width = det_dist * np.tan(2 * theta_width) / pixel_size

        # Gaussian ring profile
        ring = peak_intensity * np.exp(-((r - r_center) ** 2) / (2 * r_width**2))
        image += ring

    # Add Poisson noise
    image_noisy = rng.poisson(image * noise_level) / noise_level
    image_noisy = np.maximum(image_noisy, 0)  # Ensure non-negative

    return {
        "image": image_noisy.astype(np.float32),
        "center": center,
        "q_values": ring_q_values,
        "ring_widths": ring_widths,
        "pixel_size": pixel_size,
        "det_dist": det_dist,
        "image_clean": image.astype(np.float32),
    }


def generate_roi_parameters(
    n_rois: int = 5,
    roi_type: str = "ring",
    center: tuple[int, int] = (256, 256),
    r_inner_range: tuple[float, float] = (50, 100),
    r_outer_offset: float = 20,
    angle_range: tuple[float, float] = (0, 360),
    phi_num: int = 180,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate ROI parameter sets for testing.

    Args:
        n_rois: Number of ROI parameter sets to generate (default: 5)
        roi_type: Type of ROI ("ring", "sector", "wedge") (default: "ring")
        center: ROI center (y, x) in pixels (default: (256, 256))
        r_inner_range: Range for inner radius (min, max) (default: (50, 100))
        r_outer_offset: Offset from inner to outer radius (default: 20)
        angle_range: Angle range (start, end) in degrees (default: (0, 360))
        phi_num: Number of azimuthal bins (default: 180)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of dictionaries, each containing:
            - roi_type: Type of ROI
            - center: Tuple (y, x)
            - r_inner: Inner radius
            - r_outer: Outer radius
            - angle_start: Start angle in degrees (for sector/wedge)
            - angle_end: End angle in degrees (for sector/wedge)
            - phi_num: Number of azimuthal bins
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)

    roi_list = []
    for i in range(n_rois):
        r_inner = rng.uniform(r_inner_range[0], r_inner_range[1])
        r_outer = r_inner + r_outer_offset + rng.uniform(0, 10)

        roi = {
            "roi_type": roi_type,
            "center": center,
            "r_inner": r_inner,
            "r_outer": r_outer,
            "phi_num": phi_num,
            "index": i,
        }

        if roi_type in ("sector", "wedge"):
            # Add angle parameters for sector/wedge
            angle_span = rng.uniform(30, 90)  # Random angle span
            angle_start = rng.uniform(angle_range[0], angle_range[1] - angle_span)
            roi["angle_start"] = angle_start
            roi["angle_end"] = angle_start + angle_span
        else:
            roi["angle_start"] = angle_range[0]
            roi["angle_end"] = angle_range[1]

        roi_list.append(roi)

    return roi_list


# Re-export for convenience
__all__ = [
    "generate_synthetic_g2_data",
    "generate_synthetic_c2_matrix",
    "generate_synthetic_saxs_2d",
    "generate_roi_parameters",
]
