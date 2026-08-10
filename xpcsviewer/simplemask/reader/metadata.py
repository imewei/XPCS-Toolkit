"""Shared NeXus/HDF5 metadata helpers used by the beamline readers."""

from __future__ import annotations

import glob
import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _normalize(value: object) -> object:
    """Turn a raw HDF5 value into a plain Python/NumPy scalar where possible."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0]
        return value
    return value


def has_nexus_fields(fname: str, keymap: dict[str, str], optional_fields: list[str] | None = None) -> bool:
    """Return True if ``fname`` is an HDF5 file containing every required field."""
    if not h5py.is_hdf5(fname):
        return False

    optional_set: set[str] = set(optional_fields or ())
    with h5py.File(fname, "r") as f:
        for key, hdf_path in keymap.items():
            if key in optional_set:
                continue
            if hdf_path not in f:
                return False
    return True


def read_keymap(fname: str, keymap: dict[str, str], optional_fields: list[str] | None = None) -> dict[str, object | None]:
    """Read metadata values from an HDF5 file using a key -> path mapping.

    Optional fields that are missing are returned as ``None``. Required
    fields that are missing raise ``KeyError``.
    """
    optional_set: set[str] = set(optional_fields or ())
    metadata: dict[str, object | None] = {}
    with h5py.File(fname, "r") as f:
        for key, hdf_path in keymap.items():
            if hdf_path not in f:
                if key in optional_set:
                    metadata[key] = None
                    continue
                raise KeyError(f"required field {hdf_path!r} missing in {fname}")
            metadata[key] = _normalize(f[hdf_path][()])
    return metadata


def find_metadata_file(fname: str) -> str:
    """Find a ``*_metadata.hdf`` file in the same folder as ``fname``.

    Raises:
        FileNotFoundError: If no metadata file is present.
    """
    pattern = os.path.join(os.path.dirname(fname), "*_metadata.hdf")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no *_metadata.hdf found in the folder of {fname}")
    if len(matches) > 1:
        logger.warning(
            "multiple *_metadata.hdf found in the folder of %s; using %s",
            fname,
            matches[0],
        )
    return matches[0]


def read_nexus_metadata(fname: str, keymap: dict[str, str], optional_fields: list[str] | None = None, metadata_fname: str | None = None) -> tuple[dict[str, object | None], str]:
    """Locate and read NeXus metadata for ``fname``.

    Discovery order: an explicit valid ``metadata_fname`` override, then
    ``fname`` itself, then a sibling ``*_metadata.hdf`` file.

    Returns:
        tuple: ``(metadata_dict, meta_fname)``.

    Raises:
        FileNotFoundError: If no source has all required fields.
    """
    if metadata_fname and has_nexus_fields(metadata_fname, keymap, optional_fields):
        meta_fname = metadata_fname
    elif has_nexus_fields(fname, keymap, optional_fields):
        meta_fname = fname
    else:
        if metadata_fname:
            logger.warning(
                "metadata_fname %s is missing required fields; "
                "falling back to automatic discovery",
                metadata_fname,
            )
        meta_fname = find_metadata_file(fname)
        if not has_nexus_fields(meta_fname, keymap, optional_fields):
            raise FileNotFoundError(f"No valid metadata found in {meta_fname}")

    logger.info("using metadata file: %s", meta_fname)
    metadata = read_keymap(meta_fname, keymap, optional_fields)
    metadata["meta_fname"] = meta_fname
    return metadata, meta_fname
