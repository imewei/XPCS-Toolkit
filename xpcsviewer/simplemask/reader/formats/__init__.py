"""Scattering-format loaders and extension-based dispatch.

Only HDF5 is wired up in this PR. IMM and Rigaku 500k/3M land in PR3 (see
docs/superpowers/specs/2026-08-10-multi-format-reader-stack-design.md) —
adding them means adding an ``elif`` branch here, not restructuring this
dispatch function.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import ScatteringDataset
from .hdf import HdfDataset

logger = logging.getLogger(__name__)

__all__ = ["ScatteringDataset", "HdfDataset", "get_format_loader"]


def get_format_loader(fname: str, **kwargs: Any) -> ScatteringDataset:
    """Return a scattering-format loader appropriate for ``fname``'s extension.

    Recognized extensions in this PR: ``.h5``/``.hdf5``/``.hdf``.

    Raises:
        ValueError: If the extension is not recognized.
    """
    if fname.endswith((".h5", ".hdf5", ".hdf")):
        logger.info("APS HDF dataset")
        return HdfDataset(fname, **kwargs)
    raise ValueError(f"unsupported dataset file: {fname}")
