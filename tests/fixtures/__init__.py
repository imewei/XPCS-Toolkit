"""Test fixtures package for XPCS Toolkit.

This package provides synthetic datasets, test data generators, and
reusable test fixtures for comprehensive testing of XPCS analysis functionality.
"""

# Import focused fixture modules
from .core_fixtures import *
from .scientific_fixtures import *
from .qt_fixtures import *

__all__ = [
    # Core fixtures
    "temp_dir",
    "temp_file",
    "fixtures_dir",
    "test_logger",
    "capture_logs",
    "performance_timer",

    # Scientific fixtures
    "random_seed",
    "synthetic_correlation_data",
    "synthetic_scattering_data",
    "detector_geometry",
    "qmap_data",
    "minimal_xpcs_hdf5",
    "comprehensive_xpcs_hdf5",
    "assert_arrays_close",
    "correlation_function_validator",
    "create_test_dataset",

    # Qt fixtures
    "qt_application",
    "qt_widget",
    "qt_main_window",
    "qt_wait",
    "qt_click_helper",
    "qt_key_helper",
    "gui_test_helper",
]
