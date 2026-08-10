"""App-facing reader base class shared by all beamline readers.

Trimmed relative to upstream pySimpleMask's ``FileReader`` — see Task 4's
rationale in the implementation plan for why the qmap/display/coordinate
helper methods are intentionally not ported here.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def get_fake_metadata() -> dict[str, Any]:
    """Return placeholder metadata used when real metadata cannot be read.

    Includes every field the adapter (Task 6) requires so a failed NeXus
    read still produces a fully-formed (if fabricated) metadata dict rather
    than a partial one.
    """
    return {
        "energy": 12.3,  # keV
        "detector_distance": 12.3456,  # meter
        "pixel_size": 75e-6,  # meter
        "beam_center_x": 512.0,  # pixel
        "beam_center_y": 256.0,  # pixel
    }


def _coerce_float(value: object) -> object:
    """Cast numeric scalars (incl. NumPy types) to float, leaving others as-is."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return value


class FileReader:
    """Produces a scattering image + metadata dict for one raw detector file.

    Subclasses implement :meth:`get_scattering` and :meth:`_get_metadata`.
    ``self.metadata`` uses upstream pySimpleMask's field names/units
    (``beam_center_x``/``beam_center_y`` in pixels, ``pixel_size``/
    ``detector_distance`` in meters) -- translated to this project's own
    schema by :mod:`xpcsviewer.simplemask.reader.adapter` (Task 6), not
    here.
    """

    def __init__(self, fname: str) -> None:
        self.fname = fname
        self.ftype = "Base Class"
        self.stype = "Transmission"
        self.metadata: dict[str, Any] | None = None
        self.metadata_is_placeholder: bool = False
        self.shape: tuple[int, int] | None = None
        self.scat: np.ndarray | None = None

    def prepare_data(self, *args: Any, metadata_fname: str | None = None, **kwargs: Any) -> None:
        """Load metadata and the scattering image, deriving shape from the image."""
        self.metadata = self.get_metadata(metadata_fname=metadata_fname)
        self.scat = self.get_scattering(*args, **kwargs).astype(np.float32)
        self.shape = self.scat.shape
        # Derive detector shape from the actual image, not from (possibly
        # stale or placeholder) metadata.
        self.metadata["detector_shape_x"] = self.shape[1]
        self.metadata["detector_shape_y"] = self.shape[0]

    def get_scattering(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise NotImplementedError

    def get_metadata(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Read real metadata, falling back to a placeholder on any failure.

        Sets :attr:`metadata_is_placeholder` so callers can warn the user
        instead of silently proceeding with fabricated geometry.
        """
        try:
            metadata = self._get_metadata(*args, **kwargs)
            self.metadata_is_placeholder = False
        except Exception:
            logger.warning(
                "failed to get the real metadata, using default values instead",
                exc_info=True,
            )
            metadata = get_fake_metadata()
            self.metadata_is_placeholder = True
        return {key: _coerce_float(value) for key, value in metadata.items()}

    def _get_metadata(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError
