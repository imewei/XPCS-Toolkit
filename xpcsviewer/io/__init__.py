"""I/O utilities for XPCS Viewer.

This module provides high-level I/O operations with schema validation,
connection pooling, and consistent error handling.

Public API:
    HDF5Facade: Unified HDF5 I/O with schema validation
    HDF5ValidationError: Exception for validation failures
"""

from __future__ import annotations

from .hdf5_facade import HDF5Facade, HDF5ValidationError

__all__ = [
    "HDF5Facade",
    "HDF5ValidationError",
]
