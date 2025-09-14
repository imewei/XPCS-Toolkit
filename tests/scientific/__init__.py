"""
Scientific Algorithm Validation Framework for XPCS Toolkit

This module provides comprehensive scientific validation of all analysis algorithms
used in the XPCS Toolkit. It ensures mathematical correctness, physical validity,
and numerical stability of scientific computations.

Key Components:
- Property-based testing using Hypothesis
- Mathematical invariant validation
- Physical constraint verification
- Cross-validation with reference implementations
- Statistical validation of fitting algorithms

The framework is designed to catch regressions in scientific accuracy and
ensure all algorithms meet rigorous mathematical and physical standards.
"""

__version__ = "1.0.0"
__author__ = "XPCS Toolkit Scientific Validation Team"

# Import constants first to avoid circular imports
from .constants import SCIENTIFIC_CONSTANTS, VALIDATION_CONFIG

# Import core validation modules
# Note: Import algorithms, properties, and reference_validation only when needed
# to avoid circular imports during initialization
