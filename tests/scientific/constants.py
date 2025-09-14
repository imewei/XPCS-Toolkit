"""
Scientific Constants and Configuration for XPCS Validation

This module contains scientific constants and validation configuration
used throughout the scientific validation framework.
"""

# Define scientific constants used throughout validation
SCIENTIFIC_CONSTANTS = {
    # Physical constants
    "h_planck": 6.62607015e-34,  # Planck constant (J⋅Hz⁻¹)
    "c_light": 299792458,  # Speed of light (m/s)
    "k_boltzmann": 1.380649e-23,  # Boltzmann constant (J/K)
    # XPCS-specific constants
    "xray_wavelength_8idi": 7.35e-11,  # Typical wavelength at 8-ID-I (m)
    "detector_pixel_size": 172e-6,  # Typical detector pixel size (m)
    # Numerical tolerance constants
    "rtol_default": 1e-12,  # Relative tolerance for numerical comparisons
    "atol_default": 1e-15,  # Absolute tolerance for numerical comparisons
    "rtol_statistical": 1e-6,  # Tolerance for statistical tests
    "atol_statistical": 1e-9,  # Absolute tolerance for statistical tests
}

# Validation configuration
VALIDATION_CONFIG = {
    "hypothesis_max_examples": 100,  # Max examples for property-based tests
    "hypothesis_deadline": 5000,  # Deadline in milliseconds
    "statistical_significance": 0.01,  # P-value threshold for statistical tests
    "monte_carlo_samples": 10000,  # Samples for Monte Carlo validation
    "cross_validation_folds": 5,  # Folds for cross-validation
}
