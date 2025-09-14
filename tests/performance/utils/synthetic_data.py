"""
Synthetic data generation utilities for performance testing.

Provides functions to generate realistic synthetic XPCS data for benchmarking
without requiring actual experimental data files.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import Mock

import h5py
import numpy as np


def generate_synthetic_g2_data(
    num_tau: int, num_q: int, decay_type: str = "single_exp"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic G2 correlation data.

    Parameters:
    -----------
    num_tau : int
        Number of lag times
    num_q : int
        Number of q-points
    decay_type : str
        Type of decay ("single_exp", "double_exp", "stretched_exp")

    Returns:
    --------
    tau : np.ndarray
        Lag times
    g2_data : np.ndarray
        G2 correlation functions (num_q, num_tau)
    g2_err : np.ndarray
        G2 errors (num_q, num_tau)
    """
    # Generate log-spaced tau values
    tau = np.logspace(-6, 2, num_tau)

    g2_data = np.zeros((num_q, num_tau))
    g2_err = np.zeros((num_q, num_tau))

    for i in range(num_q):
        # Vary parameters across q-points
        if decay_type == "single_exp":
            beta = 0.3 + 0.5 * np.random.random()
            tau_c = 0.01 * (1 + i * 0.2)
            g2_theory = 1 + beta * np.exp(-tau / tau_c)

        elif decay_type == "double_exp":
            beta1 = 0.2 + 0.3 * np.random.random()
            beta2 = 0.1 + 0.2 * np.random.random()
            tau_c1 = 0.005 * (1 + i * 0.1)
            tau_c2 = 0.1 * (1 + i * 0.1)
            g2_theory = (
                1 + beta1 * np.exp(-tau / tau_c1) + beta2 * np.exp(-tau / tau_c2)
            )

        else:  # stretched_exp
            beta = 0.4 + 0.4 * np.random.random()
            tau_c = 0.02 * (1 + i * 0.15)
            alpha = 0.6 + 0.3 * np.random.random()
            g2_theory = 1 + beta * np.exp(-((tau / tau_c) ** alpha))

        # Add realistic noise
        noise_level = 0.02 + 0.03 * np.random.random()
        noise = np.random.normal(0, noise_level * np.sqrt(g2_theory), num_tau)
        g2_data[i] = g2_theory + noise
        g2_err[i] = noise_level * np.sqrt(g2_data[i])

    return tau, g2_data, g2_err


def generate_synthetic_saxs_data(
    image_size: Tuple[int, int], pattern_type: str = "rings"
) -> np.ndarray:
    """
    Generate synthetic 2D SAXS scattering data.

    Parameters:
    -----------
    image_size : tuple
        (height, width) of the image
    pattern_type : str
        Type of scattering pattern ("rings", "powder", "single_crystal")

    Returns:
    --------
    saxs_image : np.ndarray
        2D scattering pattern
    """
    height, width = image_size
    center_y, center_x = height // 2, width // 2

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    theta = np.arctan2(y - center_y, x - center_x)

    if pattern_type == "rings":
        # Concentric rings pattern
        intensity = np.zeros_like(r)
        for ring_r in [20, 40, 60, 80, 100]:
            if ring_r < min(height, width) // 2:
                ring_intensity = 1000 * np.exp(-((r - ring_r) ** 2) / (2 * 3**2))
                intensity += ring_intensity

    elif pattern_type == "powder":
        # Powder diffraction pattern
        intensity = 500 * np.exp(-r / 30)  # Radial decay
        # Add some texture modulation
        intensity *= 1 + 0.3 * np.cos(4 * theta)

    else:  # single_crystal
        # Single crystal diffraction spots
        intensity = 100 * np.ones_like(r)
        # Add Bragg peaks
        for spot_x, spot_y in [(60, 80), (120, 60), (80, 120), (40, 40)]:
            if spot_x < width and spot_y < height:
                spot_intensity = 2000 * np.exp(
                    -((x - spot_x) ** 2 + (y - spot_y) ** 2) / (2 * 5**2)
                )
                intensity += spot_intensity

    # Add background and noise
    background = 50
    intensity += background

    # Add Poisson noise
    intensity = np.random.poisson(intensity).astype(np.uint32)

    return intensity


def generate_synthetic_twotime_data(
    num_frames: int, roi_size: Tuple[int, int]
) -> np.ndarray:
    """
    Generate synthetic intensity time series for two-time correlation.

    Parameters:
    -----------
    num_frames : int
        Number of time frames
    roi_size : tuple
        (height, width) of the region of interest

    Returns:
    --------
    intensity_data : np.ndarray
        Intensity time series (num_frames, height, width)
    """
    height, width = roi_size

    # Create base intensity pattern
    base_intensity = 100 * np.ones((height, width))

    # Add spatial structure
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height // 2, width // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    base_intensity *= 1 + 0.5 * np.exp(-(r**2) / (2 * 5**2))

    # Generate time series with correlations
    intensity_series = np.zeros((num_frames, height, width))

    # Temporal correlation parameters
    correlation_time = 100  # frames
    fluctuation_amplitude = 0.2

    for t in range(num_frames):
        # Add temporal fluctuations with exponential correlation
        fluctuation = np.zeros((height, width))
        for tau in range(min(t + 1, 5 * correlation_time)):
            weight = np.exp(-tau / correlation_time)
            if t - tau >= 0:
                random_field = np.random.normal(
                    0, fluctuation_amplitude, (height, width)
                )
                fluctuation += weight * random_field

        frame_intensity = base_intensity * (1 + fluctuation)
        # Add Poisson noise
        intensity_series[t] = np.random.poisson(np.maximum(frame_intensity, 1))

    return intensity_series.astype(np.uint16)


def generate_synthetic_xpcs_file_list(num_files: int) -> List[Mock]:
    """
    Generate a list of mock XPCS file objects for testing.

    Parameters:
    -----------
    num_files : int
        Number of mock files to create

    Returns:
    --------
    mock_files : list
        List of mock XpcsFile objects
    """
    mock_files = []

    for i in range(num_files):
        # Create mock file
        mock_file = Mock()
        mock_file.name = f"test_file_{i:03d}.h5"
        mock_file.atype = ["Multitau"]

        # Generate synthetic G2 data
        num_tau = 64
        num_q = 20
        tau, g2_data, g2_err = generate_synthetic_g2_data(num_tau, num_q, "single_exp")

        mock_file.taus = tau
        mock_file.g2 = g2_data
        mock_file.g2_err = g2_err
        mock_file.ql_dyn = np.linspace(0.001, 0.1, num_q)

        # Generate synthetic SAXS data
        saxs_2d = generate_synthetic_saxs_data((128, 128), "rings")
        mock_file.saxs_2d = saxs_2d
        mock_file.saxs_1d = np.mean(saxs_2d, axis=0)

        # Add metadata
        mock_file.detector_distance = 5.0
        mock_file.wavelength = 1.24e-10
        mock_file.pixel_size = 75e-6

        mock_files.append(mock_file)

    return mock_files


def create_synthetic_hdf5_file(filepath: Path, data_config: Dict[str, Any]) -> None:
    """
    Create a synthetic HDF5 file with XPCS data structure.

    Parameters:
    -----------
    filepath : Path
        Path where to save the file
    data_config : dict
        Configuration for data generation
    """
    with h5py.File(filepath, "w") as f:
        # File attributes
        f.attrs["format_version"] = "2.0"
        f.attrs["analysis_type"] = data_config.get("analysis_type", "Multitau")

        # Create exchange group
        exchange = f.create_group("exchange")

        # Raw data
        if "raw_data_shape" in data_config:
            shape = data_config["raw_data_shape"]
            raw_data = generate_synthetic_twotime_data(shape[0], shape[1:])
            exchange.create_dataset("data", data=raw_data, compression="gzip")

        # G2 correlation data
        if "g2_config" in data_config:
            config = data_config["g2_config"]
            tau, g2_data, g2_err = generate_synthetic_g2_data(
                config["num_tau"],
                config["num_q"],
                config.get("decay_type", "single_exp"),
            )
            exchange.create_dataset("C2T_all", data=g2_data)
            exchange.create_dataset("C2T_tau", data=tau)
            exchange.create_dataset("C2T_err", data=g2_err)

        # SAXS data
        if "saxs_config" in data_config:
            config = data_config["saxs_config"]
            saxs_2d = generate_synthetic_saxs_data(
                config["image_size"], config.get("pattern_type", "rings")
            )
            exchange.create_dataset("norm2_C2T_all", data=saxs_2d)

        # Q-mapping
        if "q_config" in data_config:
            config = data_config["q_config"]
            q_values = np.linspace(config["q_min"], config["q_max"], config["num_q"])
            exchange.create_dataset("sqmap", data=q_values)

        # Detector geometry
        detector_group = exchange.create_group("detector")
        detector_group.create_dataset(
            "distance", data=[data_config.get("detector_distance", 5.0)]
        )
        detector_group.create_dataset(
            "beam_center_x", data=[data_config.get("beam_center_x", 256)]
        )
        detector_group.create_dataset(
            "beam_center_y", data=[data_config.get("beam_center_y", 256)]
        )

        # X-ray parameters
        beam_group = exchange.create_group("beam")
        beam_group.create_dataset(
            "wavelength", data=[data_config.get("wavelength", 1.24e-10)]
        )


def generate_performance_test_dataset(size_category: str) -> Dict[str, Any]:
    """
    Generate data configuration for performance testing.

    Parameters:
    -----------
    size_category : str
        Size category ("tiny", "small", "medium", "large", "xlarge")

    Returns:
    --------
    config : dict
        Data configuration dictionary
    """
    size_configs = {
        "tiny": {
            "raw_data_shape": (50, 32, 32),
            "g2_config": {"num_tau": 32, "num_q": 10},
            "saxs_config": {"image_size": (32, 32)},
            "q_config": {"q_min": 0.001, "q_max": 0.05, "num_q": 10},
        },
        "small": {
            "raw_data_shape": (100, 64, 64),
            "g2_config": {"num_tau": 48, "num_q": 15},
            "saxs_config": {"image_size": (64, 64)},
            "q_config": {"q_min": 0.001, "q_max": 0.08, "num_q": 15},
        },
        "medium": {
            "raw_data_shape": (200, 128, 128),
            "g2_config": {"num_tau": 64, "num_q": 25},
            "saxs_config": {"image_size": (128, 128)},
            "q_config": {"q_min": 0.001, "q_max": 0.1, "num_q": 25},
        },
        "large": {
            "raw_data_shape": (500, 256, 256),
            "g2_config": {"num_tau": 96, "num_q": 40},
            "saxs_config": {"image_size": (256, 256)},
            "q_config": {"q_min": 0.001, "q_max": 0.15, "num_q": 40},
        },
        "xlarge": {
            "raw_data_shape": (1000, 512, 512),
            "g2_config": {"num_tau": 128, "num_q": 60},
            "saxs_config": {"image_size": (512, 512)},
            "q_config": {"q_min": 0.001, "q_max": 0.2, "num_q": 60},
        },
    }

    return size_configs.get(size_category, size_configs["medium"])


def create_memory_stress_dataset(memory_target_mb: float) -> Dict[str, Any]:
    """
    Create dataset configuration targeting specific memory usage.

    Parameters:
    -----------
    memory_target_mb : float
        Target memory usage in MB

    Returns:
    --------
    config : dict
        Data configuration dictionary
    """
    # Estimate memory usage for different components
    # Assuming float64 (8 bytes per element)

    bytes_per_mb = 1024 * 1024
    target_bytes = memory_target_mb * bytes_per_mb

    # Allocate 70% to raw data, 20% to G2, 10% to SAXS
    raw_data_bytes = target_bytes * 0.7
    g2_bytes = target_bytes * 0.2
    saxs_bytes = target_bytes * 0.1

    # Calculate dimensions
    # Raw data: frames × height × width (uint16 = 2 bytes)
    raw_elements = raw_data_bytes / 2
    # Assume square images and reasonable frame count
    frames = min(2000, int(np.sqrt(raw_elements / 100)))  # Assume 100 frames baseline
    pixels_per_frame = raw_elements / frames
    side_length = int(np.sqrt(pixels_per_frame))

    # G2 data: num_q × num_tau (float64 = 8 bytes)
    g2_elements = g2_bytes / 8
    num_q = min(100, int(np.sqrt(g2_elements)))
    num_tau = int(g2_elements / num_q)

    # SAXS data: height × width (uint32 = 4 bytes)
    saxs_elements = saxs_bytes / 4
    saxs_side = int(np.sqrt(saxs_elements))

    config = {
        "raw_data_shape": (frames, side_length, side_length),
        "g2_config": {"num_tau": num_tau, "num_q": num_q},
        "saxs_config": {"image_size": (saxs_side, saxs_side)},
        "q_config": {"q_min": 0.001, "q_max": 0.2, "num_q": num_q},
        "analysis_type": "Multitau",
        "detector_distance": 5.0,
        "wavelength": 1.24e-10,
    }

    return config


def estimate_dataset_memory_usage(config: Dict[str, Any]) -> float:
    """
    Estimate memory usage of dataset configuration in MB.

    Parameters:
    -----------
    config : dict
        Dataset configuration

    Returns:
    --------
    memory_mb : float
        Estimated memory usage in MB
    """
    total_bytes = 0

    # Raw data (uint16 = 2 bytes)
    if "raw_data_shape" in config:
        shape = config["raw_data_shape"]
        total_bytes += np.prod(shape) * 2

    # G2 data (float64 = 8 bytes each)
    if "g2_config" in config:
        g2_config = config["g2_config"]
        total_bytes += g2_config["num_q"] * g2_config["num_tau"] * 8 * 3  # g2, tau, err

    # SAXS data (uint32 = 4 bytes)
    if "saxs_config" in config:
        saxs_config = config["saxs_config"]
        total_bytes += np.prod(saxs_config["image_size"]) * 4

    # Q-mapping and metadata (float64 = 8 bytes)
    if "q_config" in config:
        q_config = config["q_config"]
        total_bytes += q_config["num_q"] * 8

    return total_bytes / (1024 * 1024)  # Convert to MB


def create_correlation_test_data(
    correlation_type: str, num_frames: int, roi_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create test data with specific correlation properties.

    Parameters:
    -----------
    correlation_type : str
        Type of correlation ("exponential", "stretched", "power_law", "oscillatory")
    num_frames : int
        Number of time frames
    roi_size : tuple
        (height, width) of ROI

    Returns:
    --------
    intensity_data : np.ndarray
        Time series with desired correlation properties
    """
    height, width = roi_size
    intensity_data = np.zeros((num_frames, height, width))

    # Base intensity level
    base_level = 100

    if correlation_type == "exponential":
        # Exponential decay correlation
        tau_c = num_frames / 10
        for t in range(num_frames):
            correlation = np.exp(-t / tau_c)
            fluctuation = correlation * np.random.normal(0, 0.3, (height, width))
            intensity_data[t] = base_level * (1 + fluctuation)

    elif correlation_type == "stretched":
        # Stretched exponential correlation
        tau_c = num_frames / 8
        alpha = 0.7
        for t in range(num_frames):
            correlation = np.exp(-((t / tau_c) ** alpha))
            fluctuation = correlation * np.random.normal(0, 0.3, (height, width))
            intensity_data[t] = base_level * (1 + fluctuation)

    elif correlation_type == "power_law":
        # Power law correlation
        for t in range(num_frames):
            if t > 0:
                correlation = t ** (-0.5)
            else:
                correlation = 1.0
            fluctuation = correlation * np.random.normal(0, 0.3, (height, width))
            intensity_data[t] = base_level * (1 + fluctuation)

    else:  # oscillatory
        # Oscillatory correlation
        period = num_frames / 5
        for t in range(num_frames):
            correlation = np.cos(2 * np.pi * t / period) * np.exp(-t / (num_frames / 3))
            fluctuation = correlation * np.random.normal(0, 0.2, (height, width))
            intensity_data[t] = base_level * (1 + fluctuation)

    # Add Poisson noise and ensure positive values
    intensity_data = np.random.poisson(np.maximum(intensity_data, 1))

    return intensity_data.astype(np.uint16)


# Validation functions
def validate_synthetic_data(
    data: np.ndarray, expected_properties: Dict[str, Any]
) -> bool:
    """
    Validate that synthetic data has expected properties.

    Parameters:
    -----------
    data : np.ndarray
        Synthetic data to validate
    expected_properties : dict
        Expected properties (shape, dtype, value_range, etc.)

    Returns:
    --------
    is_valid : bool
        Whether data meets expectations
    """
    try:
        # Check shape
        if "shape" in expected_properties:
            if data.shape != expected_properties["shape"]:
                return False

        # Check dtype
        if "dtype" in expected_properties:
            if data.dtype != expected_properties["dtype"]:
                return False

        # Check value range
        if "min_value" in expected_properties:
            if np.min(data) < expected_properties["min_value"]:
                return False

        if "max_value" in expected_properties:
            if np.max(data) > expected_properties["max_value"]:
                return False

        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            return False

        return True

    except Exception:
        return False
