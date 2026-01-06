"""SimpleMask integration module for XPCS Viewer.

This module provides mask creation and editing tools integrated with XPCS Viewer,
ported from pySimpleMask. It includes drawing tools, Q-map computation, and
partition generation for XPCS analysis.

Components:
- SimpleMaskWindow: Main QMainWindow for mask editing
- SimpleMaskKernel: Core mask logic and operations
- Drawing tools: Rectangle, Circle, Polygon, Line ROIs
- Q-map computation: Transmission and reflection modes
"""

from xpcsviewer.simplemask.area_mask import (
    MaskArray,
    MaskAssemble,
    MaskBase,
    MaskFile,
    MaskList,
    MaskParameter,
    MaskThreshold,
)
from xpcsviewer.simplemask.pyqtgraph_mod import ImageViewROI, LineROI
from xpcsviewer.simplemask.qmap import (
    compute_qmap,
    compute_reflection_qmap,
    compute_transmission_qmap,
)
from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel
from xpcsviewer.simplemask.simplemask_window import SimpleMaskWindow
from xpcsviewer.simplemask.utils import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)

__all__ = [
    # PyQtGraph components
    "ImageViewROI",
    "LineROI",
    "MaskArray",
    # Mask classes
    "MaskAssemble",
    "MaskBase",
    "MaskFile",
    "MaskList",
    "MaskParameter",
    "MaskThreshold",
    # Window and Kernel
    "SimpleMaskKernel",
    "SimpleMaskWindow",
    "check_consistency",
    "combine_partitions",
    # Q-map computation
    "compute_qmap",
    "compute_reflection_qmap",
    "compute_transmission_qmap",
    # Utilities
    "generate_partition",
    "hash_numpy_dict",
    "optimize_integer_array",
]
