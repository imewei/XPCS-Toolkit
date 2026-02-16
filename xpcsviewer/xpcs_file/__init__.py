"""XpcsFile package - modular components for XPCS data handling.

This package provides the XpcsFile class and related utilities for loading
and manipulating XPCS data from HDF5/NeXus files.

For backward compatibility, all public symbols are re-exported from this package.

IMPORTANT: The xpcs_file package coexists with the original xpcs_file.py module.
When importing `from xpcsviewer.xpcs_file import XpcsFile`, Python will find
the package first. We use __getattr__ to lazily resolve XpcsFile from the
sibling module, avoiding circular imports during __init__.py execution.
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

# Lazy re-export of XpcsFile from the sibling xpcs_file.py module.
# Using __getattr__ ensures the package is FULLY initialized before
# attempting to load the module, avoiding circular import issues.
_xpcs_file_class = None


def __getattr__(name):
    global _xpcs_file_class  # noqa: PLW0603
    if name == "XpcsFile":
        if _xpcs_file_class is None:
            import importlib.util
            import os
            import sys

            module_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "xpcs_file.py"
            )
            if "xpcsviewer._xpcs_file_module" not in sys.modules:
                spec = importlib.util.spec_from_file_location(
                    "xpcsviewer._xpcs_file_module", module_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules["xpcsviewer._xpcs_file_module"] = module
                    spec.loader.exec_module(module)
            _xpcs_file_class = sys.modules["xpcsviewer._xpcs_file_module"].XpcsFile
        return _xpcs_file_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CacheItem",
    "DataCache",
    "MemoryMonitor",
    "MemoryStatus",
    "XpcsFile",
    "create_id",
    "double_exp_all",
    "get_cached_memory_monitor",
    "power_law",
    "single_exp_all",
]
