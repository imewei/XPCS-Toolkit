"""Synthetic XPCS dataset generators for testing.

This module provides comprehensive generators for creating realistic
synthetic XPCS data including correlation functions, scattering patterns,
detector geometries, and Q-space mappings.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SyntheticXPCSParameters:
    """Parameters for synthetic XPCS data generation."""

    # Correlation function parameters
    tau_min: float = 1e-6  # Minimum delay time (seconds)
    tau_max: float = 1e2  # Maximum delay time (seconds)
    n_tau: int = 50  # Number of tau points
    tau_spacing: str = "log"  # "log" or "linear"

    # Physical parameters
    beta: float = 0.8  # Contrast parameter
    tau_c: float = 1e-3  # Correlation time (seconds)
    baseline: float = 1.0  # Baseline value

    # Noise parameters
    noise_type: str = "gaussian"  # "gaussian", "poisson", or "mixed"
    noise_level: float = 0.02  # Relative noise level
    statistics: int = 1000  # Photon statistics (for Poisson noise)

    # Multi-exponential parameters (for complex systems)
    n_components: int = 1  # Number of exponential components
    amplitudes: list | None = None  # Component amplitudes
    time_constants: list | None = None  # Component time constants

    # Detector geometry
    detector_size: tuple[int, int] = (1024, 1024)  # pixels
    pixel_size: float = 75e-6  # meters per pixel
    det_dist: float = 5.0  # sample-detector distance (meters)
    beam_center: tuple[float, float] = (512.0, 512.0)  # beam center (pixels)
    X_energy: float = 8.0  # X-ray energy (keV)

    # Q-space parameters
    q_min: float = 0.001  # Minimum Q (Å⁻¹)
    q_max: float = 0.1  # Maximum Q (Å⁻¹)
    n_q_bins: int = 50  # Number of Q bins
    n_phi_bins: int = 36  # Number of azimuthal bins

    # SAXS parameters
    scattering_law: str = "power_law"  # "power_law", "guinier", "porod"
    power_law_exponent: float = -2.5  # For power law scattering
    guinier_rg: float = 50e-10  # Guinier radius of gyration (meters)
    background: float = 100.0  # Background intensity


class SyntheticXPCSGenerator:
    """Comprehensive generator for synthetic XPCS datasets."""

    def __init__(self, params: SyntheticXPCSParameters | None = None):
        """Initialize generator with parameters."""
        self.params = params or SyntheticXPCSParameters()
        self._setup_random_state()

    def _setup_random_state(self, seed: int = 42):
        """Setup reproducible random state."""
        np.random.seed(seed)
        self._rng = np.random.RandomState(seed)

    def generate_correlation_function(self) -> dict[str, np.ndarray]:
        """Generate synthetic G2 correlation function."""
        # Create time points
        if self.params.tau_spacing == "log":
            tau = np.logspace(
                np.log10(self.params.tau_min),
                np.log10(self.params.tau_max),
                self.params.n_tau,
            )
        else:
            tau = np.linspace(
                self.params.tau_min, self.params.tau_max, self.params.n_tau
            )

        # Generate clean correlation function
        if self.params.n_components == 1:
            # Single exponential
            g2_clean = self.params.baseline + self.params.beta * np.exp(
                -tau / self.params.tau_c
            )
        else:
            # Multi-exponential
            g2_clean = np.full_like(tau, self.params.baseline)
            amplitudes = (
                self.params.amplitudes
                or [self.params.beta / self.params.n_components]
                * self.params.n_components
            )
            time_constants = self.params.time_constants or [
                self.params.tau_c * (10**i) for i in range(self.params.n_components)
            ]

            for amp, tau_comp in zip(amplitudes, time_constants, strict=False):
                g2_clean += amp * np.exp(-tau / tau_comp)

        # Add noise
        g2_noisy, g2_err = self._add_noise(g2_clean, tau)

        return {
            "tau": tau,
            "g2": g2_noisy,
            "g2_err": g2_err,
            "g2_clean": g2_clean,
            "beta": self.params.beta,
            "tau_c": self.params.tau_c,
            "baseline": self.params.baseline,
        }

    def _add_noise(
        self, g2_clean: np.ndarray, tau: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add realistic noise to correlation function."""
        if self.params.noise_type == "gaussian":
            # Gaussian noise with intensity-dependent variance
            noise_std = self.params.noise_level * (g2_clean - self.params.baseline)
            noise = self._rng.normal(0, noise_std)
            g2_noisy = g2_clean + noise
            g2_err = noise_std

        elif self.params.noise_type == "poisson":
            # Poisson noise based on photon statistics
            # Convert correlation to count statistics
            counts = self.params.statistics * (g2_clean - self.params.baseline)
            noisy_counts = self._rng.poisson(counts)
            g2_noisy = self.params.baseline + noisy_counts / self.params.statistics
            g2_err = np.sqrt(counts) / self.params.statistics

        elif self.params.noise_type == "mixed":
            # Combination of Poisson and Gaussian noise
            # Poisson component
            counts = self.params.statistics * (g2_clean - self.params.baseline)
            noisy_counts = self._rng.poisson(counts)
            g2_poisson = self.params.baseline + noisy_counts / self.params.statistics

            # Gaussian component
            noise_std = (
                self.params.noise_level * 0.5 * (g2_clean - self.params.baseline)
            )
            noise = self._rng.normal(0, noise_std)
            g2_noisy = g2_poisson + noise

            # Combined error
            poisson_err = np.sqrt(counts) / self.params.statistics
            gaussian_err = noise_std
            g2_err = np.sqrt(poisson_err**2 + gaussian_err**2)

        else:
            raise ValueError(f"Unknown noise type: {self.params.noise_type}")

        # Ensure correlation inequality (g2 >= 1)
        g2_noisy = np.maximum(g2_noisy, 1.0)

        return g2_noisy, g2_err

    def generate_scattering_data(self) -> dict[str, np.ndarray]:
        """Generate synthetic SAXS scattering data."""
        q = np.linspace(self.params.q_min, self.params.q_max, 100)

        if self.params.scattering_law == "power_law":
            # Power law scattering: I(q) = A * q^(-alpha)
            intensity = q**self.params.power_law_exponent
            intensity = intensity / intensity[0] * 1e6  # Normalize

        elif self.params.scattering_law == "guinier":
            # Guinier approximation: I(q) = I0 * exp(-(q*Rg)^2/3)
            intensity = np.exp(-((q * self.params.guinier_rg) ** 2) / 3) * 1e6

        elif self.params.scattering_law == "porod":
            # Porod law: I(q) = A * q^(-4) + background
            intensity = q ** (-4) + self.params.background
            intensity = intensity / intensity[0] * 1e6

        else:
            raise ValueError(f"Unknown scattering law: {self.params.scattering_law}")

        # Add background
        intensity += self.params.background

        # Add Poisson noise based on intensity
        intensity_noisy = self._rng.poisson(intensity).astype(float)
        intensity_err = np.sqrt(intensity)

        return {
            "q": q,
            "intensity": intensity_noisy,
            "intensity_err": intensity_err,
            "intensity_clean": intensity,
        }

    def generate_detector_geometry(self) -> dict[str, Any]:
        """Generate detector geometry parameters."""
        # Calculate wavelength from energy
        h = 4.135667696e-15  # Planck constant (eV⋅s)
        c = 299792458  # Speed of light (m/s)
        wavelength = h * c / (self.params.X_energy * 1000)  # meters

        return {
            "detector_size": self.params.detector_size,
            "pixel_size": self.params.pixel_size,
            "det_dist": self.params.det_dist,
            "beam_center_x": self.params.beam_center[0],
            "beam_center_y": self.params.beam_center[1],
            "X_energy": self.params.X_energy,
            "wavelength": wavelength,
            "bcx": self.params.beam_center[0],
            "bcy": self.params.beam_center[1],
        }

    def generate_qmap_data(self) -> dict[str, np.ndarray]:
        """Generate Q-space mapping data."""
        geom = self.generate_detector_geometry()
        ny, nx = geom["detector_size"]

        # Create coordinate arrays
        x = np.arange(nx) - geom["beam_center_x"]
        y = np.arange(ny) - geom["beam_center_y"]
        X, Y = np.meshgrid(x, y)

        # Calculate scattering geometry
        pixel_size = geom["pixel_size"]
        det_dist = geom["det_dist"]
        wavelength = geom["wavelength"]

        # Scattering angles
        theta = 0.5 * np.arctan(np.sqrt(X**2 + Y**2) * pixel_size / det_dist)

        # Q magnitude
        q_magnitude = 4 * np.pi * np.sin(theta) / wavelength

        # Azimuthal angle
        phi = np.arctan2(Y, X) * 180 / np.pi

        # Create Q and phi bins
        q_bins = np.linspace(self.params.q_min, self.params.q_max, self.params.n_q_bins)
        phi_bins = np.linspace(-180, 180, self.params.n_phi_bins + 1)

        # Create mapping arrays
        dqmap = np.digitize(q_magnitude.flatten(), q_bins).reshape(q_magnitude.shape)
        sqmap = dqmap.copy()  # For simplicity, use same mapping

        # Create mask (circular aperture)
        _center_x, _center_y = geom["beam_center_x"], geom["beam_center_y"]
        radius = min(nx, ny) // 3  # Conservative mask
        mask_distance = np.sqrt((X) ** 2 + (Y) ** 2)
        mask = (mask_distance <= radius).astype(np.int32)

        return {
            "dqmap": dqmap.astype(np.int32),
            "sqmap": sqmap.astype(np.int32),
            "mask": mask,
            "q_magnitude": q_magnitude,
            "phi": phi,
            "dqlist": q_bins[:-1],
            "sqlist": q_bins[:-1],
            "dplist": phi_bins[:-1],
            "splist": phi_bins[:-1],
            "bcx": geom["bcx"],
            "bcy": geom["bcy"],
            "pixel_size": geom["pixel_size"],
            "det_dist": geom["det_dist"],
            "X_energy": geom["X_energy"],
            "dynamic_num_pts": len(q_bins) - 1,
            "static_num_pts": len(q_bins) - 1,
            "dynamic_index_mapping": np.arange(len(q_bins) - 1),
            "static_index_mapping": np.arange(len(q_bins) - 1),
            "map_names": [b"q", b"phi"],
            "map_units": [b"1/A", b"degree"],
        }

    def generate_multitau_data(self, n_q_points: int = 5) -> dict[str, np.ndarray]:
        """Generate multi-tau correlation data for multiple Q points."""
        base_corr = self.generate_correlation_function()

        # Create variations for different Q points
        g2_matrix = []
        g2_err_matrix = []

        for _i in range(n_q_points):
            # Vary correlation time and contrast slightly
            tau_c_var = base_corr["tau_c"] * (1 + 0.1 * self._rng.normal())
            beta_var = base_corr["beta"] * (1 + 0.05 * self._rng.normal())

            # Regenerate with variations
            g2_clean = base_corr["baseline"] + beta_var * np.exp(
                -base_corr["tau"] / tau_c_var
            )
            g2_noisy, g2_err = self._add_noise(g2_clean, base_corr["tau"])

            g2_matrix.append(g2_noisy)
            g2_err_matrix.append(g2_err)

        return {
            "normalized_g2": np.array(g2_matrix),
            "normalized_g2_err": np.array(g2_err_matrix),
            "delay_list": base_corr["tau"],
            "stride_frame": 1,
            "avg_frame": 1,
            "t0": 0.001,
            "t1": 0.001,
            "start_time": 1000000000,
        }

    def generate_twotime_data(self, n_time_points: int = 100) -> dict[str, np.ndarray]:
        """Generate two-time correlation data."""
        # Create time grid
        t_max = 10 * self.params.tau_c  # 10 correlation times
        time_points = np.linspace(0, t_max, n_time_points)

        # Generate two-time correlation matrix
        # For simplicity, use exponential decay along diagonals
        g2_twotime = np.ones((n_time_points, n_time_points))

        for i in range(n_time_points):
            for j in range(n_time_points):
                tau = abs(time_points[i] - time_points[j])
                g2_twotime[i, j] = self.params.baseline + self.params.beta * np.exp(
                    -tau / self.params.tau_c
                )

        # Add noise
        noise = self._rng.normal(
            0,
            self.params.noise_level * (g2_twotime - self.params.baseline),
            g2_twotime.shape,
        )
        g2_twotime += noise
        g2_twotime = np.maximum(g2_twotime, 1.0)  # Ensure correlation inequality

        return {
            "correlation_map": g2_twotime,
            "normalized_g2": g2_twotime,
            "elapsed_time": time_points,
            "time_spacing": time_points[1] - time_points[0],
            "delay_list": time_points,
        }


# Convenience functions for direct use
def create_synthetic_g2_data(
    tau_min: float = 1e-6,
    tau_max: float = 1e2,
    n_tau: int = 50,
    beta: float = 0.8,
    tau_c: float = 1e-3,
    noise_level: float = 0.02,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Create synthetic G2 correlation function data."""
    params = SyntheticXPCSParameters(
        tau_min=tau_min,
        tau_max=tau_max,
        n_tau=n_tau,
        beta=beta,
        tau_c=tau_c,
        noise_level=noise_level,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)
    return generator.generate_correlation_function()


def create_synthetic_saxs_data(
    q_min: float = 0.001,
    q_max: float = 0.1,
    scattering_law: str = "power_law",
    power_law_exponent: float = -2.5,
    background: float = 100.0,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Create synthetic SAXS scattering data."""
    params = SyntheticXPCSParameters(
        q_min=q_min,
        q_max=q_max,
        scattering_law=scattering_law,
        power_law_exponent=power_law_exponent,
        background=background,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)
    return generator.generate_scattering_data()


def create_detector_geometry(
    detector_size: tuple[int, int] = (1024, 1024),
    pixel_size: float = 75e-6,
    det_dist: float = 5.0,
    beam_center: tuple[float, float] = (512.0, 512.0),
    X_energy: float = 8.0,
    **kwargs,
) -> dict[str, Any]:
    """Create detector geometry parameters."""
    params = SyntheticXPCSParameters(
        detector_size=detector_size,
        pixel_size=pixel_size,
        det_dist=det_dist,
        beam_center=beam_center,
        X_energy=X_energy,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)
    return generator.generate_detector_geometry()


def create_qmap_data(
    detector_size: tuple[int, int] = (1024, 1024),
    q_min: float = 0.001,
    q_max: float = 0.1,
    n_q_bins: int = 50,
    n_phi_bins: int = 36,
    **kwargs,
) -> dict[str, np.ndarray]:
    """Create Q-space mapping data."""
    params = SyntheticXPCSParameters(
        detector_size=detector_size,
        q_min=q_min,
        q_max=q_max,
        n_q_bins=n_q_bins,
        n_phi_bins=n_phi_bins,
        **kwargs,
    )
    generator = SyntheticXPCSGenerator(params)
    return generator.generate_qmap_data()
