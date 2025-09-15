"""
Reference Data Management for Scientific Validation

This module provides utilities for loading, validating, and managing reference
datasets used in scientific algorithm validation tests.
"""

import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Base path for reference data
REFERENCE_DATA_PATH = Path(__file__).parent


def load_reference_data(category: str, dataset_name: str) -> dict[str, Any]:
    """
    Load reference data from the specified category and dataset

    Args:
        category: Data category ('g2_analysis', 'saxs_analysis', etc.)
        dataset_name: Name of the dataset (without .npz extension)

    Returns:
        Dictionary containing data arrays and metadata

    Raises:
        FileNotFoundError: If the specified dataset doesn't exist
        ValueError: If the dataset is corrupted or invalid
    """
    data_path = REFERENCE_DATA_PATH / category / f"{dataset_name}.npz"

    if not data_path.exists():
        available = list_available_datasets(category)
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found in '{category}'. "
            f"Available: {available}"
        )

    try:
        data = dict(np.load(data_path, allow_pickle=True))

        # Validate required fields
        required_fields = ["x_values", "y_values", "metadata"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Convert metadata if it's stored as numpy array
        if isinstance(data["metadata"], np.ndarray):
            data["metadata"] = data["metadata"].item()

        return data

    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_name}': {e}")


def save_reference_data(
    category: str,
    dataset_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_errors: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """
    Save reference data to the specified category

    Args:
        category: Data category
        dataset_name: Name for the dataset
        x_values: Independent variable values
        y_values: Dependent variable values
        y_errors: Error estimates (optional)
        metadata: Dataset metadata (optional)
    """
    # Ensure category directory exists
    category_path = REFERENCE_DATA_PATH / category
    category_path.mkdir(exist_ok=True)

    # Prepare data dictionary
    data_dict = {"x_values": x_values, "y_values": y_values}

    if y_errors is not None:
        data_dict["y_errors"] = y_errors

    # Add metadata with validation info
    if metadata is None:
        metadata = {}

    metadata.update(
        {
            "creation_date": datetime.now().isoformat(),
            "data_shape": y_values.shape,
            "x_range": (float(np.min(x_values)), float(np.max(x_values))),
            "y_range": (float(np.min(y_values)), float(np.max(y_values))),
        }
    )

    data_dict["metadata"] = metadata

    # Save data
    output_path = category_path / f"{dataset_name}.npz"
    np.savez_compressed(output_path, **data_dict)

    print(f"Saved reference data: {output_path}")


def list_available_datasets(category: str) -> list[str]:
    """
    List available datasets in a category

    Args:
        category: Data category to list

    Returns:
        List of available dataset names (without .npz extension)
    """
    category_path = REFERENCE_DATA_PATH / category

    if not category_path.exists():
        return []

    datasets = []
    for file_path in category_path.glob("*.npz"):
        datasets.append(file_path.stem)

    return sorted(datasets)


def validate_reference_data(category: str, dataset_name: str) -> tuple[bool, list[str]]:
    """
    Validate reference data integrity and completeness

    Args:
        category: Data category
        dataset_name: Dataset name

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    try:
        data = load_reference_data(category, dataset_name)

        # Check required fields
        required_fields = ["x_values", "y_values", "metadata"]
        for field in required_fields:
            if field not in data:
                issues.append(f"Missing required field: {field}")

        if issues:
            return False, issues

        x_vals = data["x_values"]
        y_vals = data["y_values"]
        metadata = data["metadata"]

        # Validate array properties
        if len(x_vals) != len(y_vals):
            issues.append("x_values and y_values have different lengths")

        if len(x_vals) == 0:
            issues.append("Empty data arrays")

        # Check for NaN or inf values
        if np.any(~np.isfinite(x_vals)):
            issues.append("x_values contains NaN or inf")

        if np.any(~np.isfinite(y_vals)):
            issues.append("y_values contains NaN or inf")

        # Validate errors if present
        if "y_errors" in data:
            y_errs = data["y_errors"]
            if len(y_errs) != len(y_vals):
                issues.append("y_errors length doesn't match y_values")

            if np.any(y_errs < 0):
                issues.append("Negative error values found")

        # Validate metadata
        if not isinstance(metadata, dict):
            issues.append("Metadata is not a dictionary")
        else:
            # Check for recommended metadata fields
            recommended_fields = [
                "creation_date",
                "validation_method",
                "theoretical_params",
            ]
            missing_recommended = [
                field for field in recommended_fields if field not in metadata
            ]

            if missing_recommended:
                issues.append(f"Missing recommended metadata: {missing_recommended}")

        return len(issues) == 0, issues

    except Exception as e:
        issues.append(f"Failed to load or parse data: {e}")
        return False, issues


def create_analytical_g2_data():
    """Create analytical G2 reference data"""

    # Single exponential decay
    tau = np.logspace(-6, 1, 100)

    # Standard parameters
    baseline = 1.0
    beta = 0.8
    gamma = 1000.0

    g2_single = baseline + beta * np.exp(-gamma * tau)

    metadata_single = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "baseline": baseline,
            "beta": beta,
            "gamma": gamma,
            "model": "single_exponential",
        },
        "reference_source": "G2(τ) = baseline + β*exp(-γ*τ)",
        "numerical_precision": 1e-15,
        "notes": "Standard single exponential with typical XPCS parameters",
    }

    save_reference_data(
        "g2_analysis",
        "single_exponential_standard",
        tau,
        g2_single,
        metadata=metadata_single,
    )

    # Double exponential decay
    beta1, gamma1 = 0.5, 5000.0  # Fast component
    beta2, gamma2 = 0.3, 500.0  # Slow component

    g2_double = baseline + beta1 * np.exp(-gamma1 * tau) + beta2 * np.exp(-gamma2 * tau)

    metadata_double = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "baseline": baseline,
            "beta1": beta1,
            "gamma1": gamma1,
            "beta2": beta2,
            "gamma2": gamma2,
            "model": "double_exponential",
        },
        "reference_source": "G2(τ) = baseline + β₁*exp(-γ₁*τ) + β₂*exp(-γ₂*τ)",
        "numerical_precision": 1e-15,
        "notes": "Two-component system with fast and slow dynamics",
    }

    save_reference_data(
        "g2_analysis",
        "double_exponential_standard",
        tau,
        g2_double,
        metadata=metadata_double,
    )

    # Stretched exponential
    beta_kww = 0.7
    tau_char = 0.001
    A = 0.8

    g2_stretched = baseline + A * np.exp(-((tau / tau_char) ** beta_kww))

    metadata_stretched = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "baseline": baseline,
            "amplitude": A,
            "tau_char": tau_char,
            "beta_kww": beta_kww,
            "model": "stretched_exponential",
        },
        "reference_source": "G2(τ) = baseline + A*exp(-(τ/τ₀)^β)",
        "numerical_precision": 1e-15,
        "notes": "Kohlrausch-Williams-Watts (KWW) stretched exponential",
    }

    save_reference_data(
        "g2_analysis",
        "stretched_exponential_standard",
        tau,
        g2_stretched,
        metadata=metadata_stretched,
    )


def create_analytical_saxs_data():
    """Create analytical SAXS reference data"""

    q_values = np.logspace(-3, 0, 200)  # Å⁻¹

    # Sphere form factor
    radius = 25.0  # Å
    intensity_0 = 1000.0

    qR = q_values * radius
    qR = np.where(qR == 0, 1e-10, qR)

    form_factor = 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR**3)
    intensity_sphere = intensity_0 * form_factor**2

    metadata_sphere = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "radius": radius,
            "intensity_0": intensity_0,
            "model": "sphere_form_factor",
        },
        "reference_source": "Rayleigh sphere form factor",
        "numerical_precision": 1e-12,
        "notes": "Monodisperse sphere form factor",
    }

    save_reference_data(
        "saxs_analysis",
        "sphere_form_factor_25A",
        q_values,
        intensity_sphere,
        metadata=metadata_sphere,
    )

    # Power law scattering
    amplitude = 100.0
    exponent = 2.5
    background = 1.0

    intensity_power_law = amplitude * q_values ** (-exponent) + background

    metadata_power_law = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "amplitude": amplitude,
            "exponent": exponent,
            "background": background,
            "model": "power_law",
        },
        "reference_source": "I(q) = A*q^(-α) + B",
        "numerical_precision": 1e-15,
        "notes": "Power law scattering with constant background",
    }

    save_reference_data(
        "saxs_analysis",
        "power_law_2p5_exponent",
        q_values,
        intensity_power_law,
        metadata=metadata_power_law,
    )


def create_analytical_diffusion_data():
    """Create analytical diffusion reference data"""

    q_values = np.logspace(-3, -1, 50)  # Å⁻¹

    # Brownian diffusion
    D_brownian = 1e-12  # m²/s
    tau_brownian = 1.0 / (D_brownian * q_values**2)

    metadata_brownian = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "diffusion_coefficient": D_brownian,
            "model": "brownian_diffusion",
        },
        "reference_source": "τ(q) = 1/(D*q²)",
        "numerical_precision": 1e-15,
        "notes": "Simple Brownian diffusion",
    }

    save_reference_data(
        "diffusion_analysis",
        "brownian_standard",
        q_values,
        tau_brownian,
        metadata=metadata_brownian,
    )

    # Subdiffusion
    alpha_subdiff = 1.6
    A_subdiff = 1e-4
    tau_subdiff = A_subdiff * q_values ** (-alpha_subdiff)

    metadata_subdiff = {
        "validation_method": "analytical_solution",
        "theoretical_params": {
            "amplitude": A_subdiff,
            "exponent": alpha_subdiff,
            "model": "subdiffusion",
        },
        "reference_source": "τ(q) = A*q^(-α), α < 2",
        "numerical_precision": 1e-15,
        "notes": "Subdiffusive motion with α = 1.6",
    }

    save_reference_data(
        "diffusion_analysis",
        "subdiffusion_alpha_1p6",
        q_values,
        tau_subdiff,
        metadata=metadata_subdiff,
    )


def initialize_reference_data():
    """Initialize all reference datasets"""
    print("Creating reference data...")

    create_analytical_g2_data()
    create_analytical_saxs_data()
    create_analytical_diffusion_data()

    print("Reference data initialization complete.")

    # Validate all created datasets
    categories = ["g2_analysis", "saxs_analysis", "diffusion_analysis"]

    print("\nValidating reference data...")
    all_valid = True

    for category in categories:
        datasets = list_available_datasets(category)
        print(f"\n{category}:")

        for dataset in datasets:
            is_valid, issues = validate_reference_data(category, dataset)
            if is_valid:
                print(f"  ✓ {dataset}")
            else:
                print(f"  ✗ {dataset}: {'; '.join(issues)}")
                all_valid = False

    if all_valid:
        print("\n✓ All reference data validated successfully!")
    else:
        print("\n✗ Some reference data validation failed!")


if __name__ == "__main__":
    initialize_reference_data()
