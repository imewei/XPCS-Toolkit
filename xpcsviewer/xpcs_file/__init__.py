"""XpcsFile package - modular components for XPCS data handling.

This package provides the XpcsFile class and related utilities for loading
and manipulating XPCS data from HDF5/NeXus files.

For backward compatibility, all public symbols are re-exported from this package.

IMPORTANT: The xpcs_file package coexists with the original xpcs_file.py module.
When importing `from xpcsviewer.xpcs_file import XpcsFile`, Python will find
the package first. We re-export XpcsFile here for backward compatibility.
"""

from __future__ import annotations

# Cache utilities
from xpcsviewer.xpcs_file.cache import CacheItem, DataCache

# Fitting functions
from xpcsviewer.xpcs_file.fitting import (
    create_id,
    double_exp_all,
    power_law,
    single_exp_all,
)

# Memory monitoring utilities
from xpcsviewer.xpcs_file.memory import (
    MemoryMonitor,
    MemoryStatus,
    get_cached_memory_monitor,
)


# Re-export XpcsFile from original module for backward compatibility
# This import is done last to avoid circular import issues
def _lazy_import_xpcs_file():
    """Lazily import XpcsFile to avoid circular imports."""
    # Import from the actual xpcs_file module (not this package)
    import importlib.util
    import os
    import sys

    # Get the path to the xpcs_file.py module (sibling file)
    package_dir = os.path.dirname(os.path.dirname(__file__))
    module_path = os.path.join(package_dir, "xpcs_file.py")

    # Check if we need to import from file directly
    if "xpcsviewer._xpcs_file_module" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "xpcsviewer._xpcs_file_module", module_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["xpcsviewer._xpcs_file_module"] = module
            spec.loader.exec_module(module)

    return sys.modules["xpcsviewer._xpcs_file_module"]


# Import XpcsFile class
try:
    _xpcs_module = _lazy_import_xpcs_file()
    XpcsFile = _xpcs_module.XpcsFile
except (ImportError, AttributeError):
    # Fallback: module not yet available (during initial package setup)
    XpcsFile = None  # type: ignore


__all__ = [
    # Main class
    "XpcsFile",
    # Memory
    "MemoryStatus",
    "MemoryMonitor",
    "get_cached_memory_monitor",
    # Cache
    "CacheItem",
    "DataCache",
    # Fitting
    "single_exp_all",
    "double_exp_all",
    "power_law",
    "create_id",
]
