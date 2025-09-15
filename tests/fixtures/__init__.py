"""Test fixtures package for XPCS Toolkit.

This package provides synthetic datasets, test data generators, and
reusable test fixtures for comprehensive testing of XPCS analysis functionality.
"""

from .hdf5_fixtures import (
    HDF5TestGenerator,
    XPCSTestFile,
    comprehensive_xpcs_file,
    create_comprehensive_hdf5,
    create_minimal_hdf5,
    minimal_xpcs_file,
    realistic_xpcs_file,
)
from .synthetic_data import (
    SyntheticXPCSGenerator,
    create_detector_geometry,
    create_qmap_data,
    create_synthetic_g2_data,
    create_synthetic_saxs_data,
)

__all__ = [
    "HDF5TestGenerator",
    "SyntheticXPCSGenerator",
    "XPCSTestFile",
    "comprehensive_xpcs_file",
    "create_comprehensive_hdf5",
    "create_detector_geometry",
    "create_minimal_hdf5",
    "create_qmap_data",
    "create_synthetic_g2_data",
    "create_synthetic_saxs_data",
    "minimal_xpcs_file",
    "realistic_xpcs_file",
]
