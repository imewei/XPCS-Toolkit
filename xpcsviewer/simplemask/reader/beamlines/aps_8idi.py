"""APS 8-ID-I reader (transmission SAXS/XPCS)."""

from __future__ import annotations

from typing import Any

from ..base_reader import FileReader
from ..formats import get_format_loader
from ..metadata import read_nexus_metadata

# Metadata key -> NeXus HDF5 path.
METADATA_KEYMAPS: dict[str, str] = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "x_pixel_size": "/entry/instrument/detector_1/x_pixel_size",
    "y_pixel_size": "/entry/instrument/detector_1/y_pixel_size",
    "ccdx": "/entry/instrument/detector_1/position_x",
    "ccdy": "/entry/instrument/detector_1/position_y",
    "ccdx0": "/entry/instrument/detector_1/beam_center_position_x",
    "ccdy0": "/entry/instrument/detector_1/beam_center_position_y",
    "bcx0": "/entry/instrument/detector_1/beam_center_x",
    "bcy0": "/entry/instrument/detector_1/beam_center_y",
}

OPTIONAL_FIELDS: list[str] = []


def get_nexus_metadata(fname: str, metadata_fname: str | None = None) -> dict[str, Any]:
    """Read 8-ID-I NeXus metadata and derive the beam center.

    Args:
        fname: Path to the NeXus HDF5 file.
        metadata_fname: Optional path to an external metadata file.

    Returns:
        Dictionary with derived metadata including energy, detector_distance,
        beam_center_x, beam_center_y, and pixel_size.

    Raises:
        KeyError: A required field is missing from the NeXus file.
        FileNotFoundError: No valid metadata source could be located.
    """
    meta, _meta_fname = read_nexus_metadata(
        fname, METADATA_KEYMAPS, OPTIONAL_FIELDS, metadata_fname=metadata_fname
    )

    # Beam center = recorded center + detector translation in pixels.
    # Type: ignore needed because dict values are Any, but arithmetic is safe.
    meta["beam_center_x"] = (
        meta["bcx0"] + (meta["ccdx"] - meta["ccdx0"]) / meta["x_pixel_size"]  # type: ignore[operator]
    )
    meta["beam_center_y"] = (
        meta["bcy0"] + (meta["ccdy"] - meta["ccdy0"]) / meta["y_pixel_size"]  # type: ignore[operator]
    )
    meta["pixel_size"] = meta["x_pixel_size"]

    for key in (
        "bcx0",
        "bcy0",
        "ccdx",
        "ccdy",
        "ccdx0",
        "ccdy0",
        "x_pixel_size",
        "y_pixel_size",
    ):
        meta.pop(key, None)

    return meta


class APS8IDIReader(FileReader):
    """Reader for APS 8-ID-I beamline transmission SAXS/XPCS data."""

    def __init__(self, fname: str, **kwargs: Any) -> None:
        """Initialize the APS 8-ID-I reader.

        Args:
            fname: Path to the HDF5 data file.
            **kwargs: Additional arguments forwarded to the format loader
                (e.g., custom data_path).
        """
        super().__init__(fname)
        self.ftype: str = "APS_8IDI"
        self.stype: str = "Transmission"
        # **kwargs forwarded to the format loader (e.g. a future custom
        # data_path override) -- get_reader() passes **kwargs through here,
        # so this constructor must accept and forward them, not swallow them.
        self.loader = get_format_loader(fname, **kwargs)
        # Detector shape from the format loader's inspected dataset shape
        # (not NeXus metadata). Provisional: FileReader.prepare_data()
        # overwrites this with the actual decoded scattering-image shape
        # once get_scattering() runs, so this value only matters if
        # something reads .shape before prepare_data() is called.
        self.shape = tuple(self.loader.det_size)  # type: ignore[assignment]

    def get_scattering(self, **kwargs: Any) -> Any:
        """Get scattering data from the loader.

        Args:
            **kwargs: Additional arguments passed to the loader.

        Returns:
            Scattering data array.
        """
        return self.loader.get_scattering(**kwargs)

    def _get_metadata(self, metadata_fname: str | None = None) -> dict[str, Any]:
        """Get metadata for this file.

        Args:
            metadata_fname: Optional path to an external metadata file.

        Returns:
            Dictionary of metadata.

        Raises:
            KeyError: A required metadata field is missing.
            FileNotFoundError: Metadata could not be located.

        Note:
            This calls get_nexus_metadata() directly rather than through a
            catch-and-default wrapper. That's deliberate: FileReader.
            get_metadata() (base_reader.py) is the ONE place that catches a
            failure here, falls back to placeholder metadata, and sets
            metadata_is_placeholder. Adding a local try/except in this
            method (or a module-level get_metadata() wrapper, as upstream
            pySimpleMask has) would swallow the exception before the base
            class ever sees it, silently making metadata_is_placeholder
            dead code. Let failures propagate.
        """
        return get_nexus_metadata(self.fname, metadata_fname=metadata_fname)
