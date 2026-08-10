"""Qt-free raw detector file reader stack (beamline + format dispatch)."""

from __future__ import annotations

import logging
from typing import Any

from .base_reader import FileReader

logger = logging.getLogger(__name__)

__all__ = ["FileReader", "get_reader"]


def get_reader(beamline: str, fname: str, **kwargs: Any) -> FileReader:
    """Construct the reader for a beamline.

    Args:
        beamline: Beamline identifier. Only ``"APS_8IDI"`` is supported in
            this PR; ``"APS_9IDD"`` and ``"NativeFiles"`` land in PR2/PR3.
        fname: Path to the data file.
        **kwargs: Additional arguments forwarded to the reader constructor
            (e.g., custom data_path for the format loader).

    Returns:
        A FileReader instance for the specified beamline.

    Raises:
        ValueError: If the beamline is not supported.
    """
    if beamline == "APS_8IDI":
        from .beamlines.aps_8idi import APS8IDIReader

        return APS8IDIReader(fname, **kwargs)
    raise ValueError(f"unsupported beamline: {beamline}")
