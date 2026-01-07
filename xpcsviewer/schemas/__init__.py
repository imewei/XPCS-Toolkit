"""Schema definitions and validators for XPCS data structures.

This module provides dataclass-based schemas with built-in validation for
all shared data structures in the XPCS Viewer codebase.

Public API:
    QMapSchema: Q-map data structure
    GeometryMetadata: Detector geometry configuration
    G2Data: G2 correlation data
    PartitionSchema: Q-bin partition data
    MaskSchema: Mask data with metadata
"""

from __future__ import annotations

from .validators import (
    G2Data,
    GeometryMetadata,
    MaskSchema,
    PartitionSchema,
    QMapSchema,
)

__all__ = [
    "QMapSchema",
    "GeometryMetadata",
    "G2Data",
    "PartitionSchema",
    "MaskSchema",
]
