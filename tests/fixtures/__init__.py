"""Test fixtures package for XPCS Toolkit.

This package provides synthetic datasets, test data generators, and
reusable test fixtures for comprehensive testing of XPCS analysis functionality.
"""

# Import focused fixture modules
from .core_fixtures import *
from .qt_fixtures import *
from .scientific_fixtures import *

__all__ = [
    "assert_arrays_close",
    "capture_logs",
    "comprehensive_xpcs_hdf5",
    "correlation_function_validator",
    "create_test_dataset",
    "detector_geometry",
    "fixtures_dir",
    "gui_test_helper",
    "minimal_xpcs_hdf5",
    "performance_timer",
    "qmap_data",
    # Qt fixtures
    "qt_application",
    "qt_click_helper",
    "qt_key_helper",
    "qt_main_window",
    "qt_wait",
    "qt_widget",
    # Scientific fixtures
    "random_seed",
    "synthetic_correlation_data",
    "synthetic_scattering_data",
    # Core fixtures
    "temp_dir",
    "temp_file",
    "test_logger",
]
