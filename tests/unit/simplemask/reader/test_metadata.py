"""Unit tests for reader/metadata.py NeXus keymap helpers."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader.metadata import (
    find_metadata_file,
    has_nexus_fields,
    read_keymap,
    read_nexus_metadata,
)

KEYMAP = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
}
OPTIONAL = ["detector_distance"]


def _write_nexus(path, energy=12.0, distance=5.0, include_distance=True):
    with h5py.File(path, "w") as f:
        f.create_dataset("/entry/instrument/incident_beam/incident_energy", data=energy)
        if include_distance:
            f.create_dataset("/entry/instrument/detector_1/distance", data=distance)


class TestHasNexusFields:
    def test_all_required_fields_present(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname)
        assert has_nexus_fields(fname, KEYMAP) is True

    def test_missing_required_field(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/some/other/path", data=1.0)
        assert has_nexus_fields(fname, KEYMAP) is False

    def test_missing_optional_field_still_true(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, include_distance=False)
        assert has_nexus_fields(fname, KEYMAP, optional_fields=OPTIONAL) is True

    def test_non_hdf5_file_returns_false(self, tmp_path):
        fname = str(tmp_path / "not_hdf5.txt")
        with open(fname, "w") as f:
            f.write("not an hdf5 file")
        assert has_nexus_fields(fname, KEYMAP) is False


class TestReadKeymap:
    def test_reads_scalar_values(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, energy=13.5, distance=4.2)
        result = read_keymap(fname, KEYMAP)
        assert result["energy"] == pytest.approx(13.5)
        assert result["detector_distance"] == pytest.approx(4.2)

    def test_missing_required_field_raises(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/some/other/path", data=1.0)
        with pytest.raises(KeyError):
            read_keymap(fname, KEYMAP)

    def test_missing_optional_field_returns_none(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, include_distance=False)
        result = read_keymap(fname, KEYMAP, optional_fields=OPTIONAL)
        assert result["detector_distance"] is None

    def test_decodes_byte_strings(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/entry/instrument/incident_beam/incident_energy", data=12.0
            )
            f.create_dataset(
                "/entry/instrument/detector_1/distance", data=np.bytes_(b"5.0m")
            )
        result = read_keymap(fname, KEYMAP)
        assert result["detector_distance"] == "5.0m"


class TestFindMetadataFile:
    def test_finds_sibling_metadata_file(self, tmp_path):
        (tmp_path / "scan_metadata.hdf").write_bytes(b"")
        found = find_metadata_file(str(tmp_path / "scan.h5"))
        assert found == str(tmp_path / "scan_metadata.hdf")

    def test_raises_when_none_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_metadata_file(str(tmp_path / "scan.h5"))


class TestReadNexusMetadata:
    def test_reads_from_primary_file(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, energy=11.0, distance=3.0)
        metadata, meta_fname = read_nexus_metadata(fname, KEYMAP)
        assert metadata["energy"] == pytest.approx(11.0)
        assert meta_fname == fname

    def test_falls_back_to_sibling_metadata_file(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        meta_path = str(tmp_path / "data_metadata.hdf")
        _write_nexus(meta_path, energy=9.0, distance=2.0)
        metadata, meta_fname = read_nexus_metadata(fname, KEYMAP)
        assert metadata["energy"] == pytest.approx(9.0)
        assert meta_fname == meta_path

    def test_explicit_metadata_fname_override_takes_priority(self, tmp_path):
        # fname itself has valid fields too -- an explicit, valid override
        # must still win over it per the documented discovery order.
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, energy=11.0, distance=3.0)
        override_path = str(tmp_path / "override.h5")
        _write_nexus(override_path, energy=99.0, distance=7.0)
        metadata, meta_fname = read_nexus_metadata(
            fname, KEYMAP, metadata_fname=override_path
        )
        assert metadata["energy"] == pytest.approx(99.0)
        assert meta_fname == override_path

    def test_invalid_metadata_fname_falls_back_to_discovery(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        _write_nexus(fname, energy=11.0, distance=3.0)
        bad_override = str(tmp_path / "missing_fields.h5")
        with h5py.File(bad_override, "w") as f:
            f.create_dataset("/some/other/path", data=1.0)
        metadata, meta_fname = read_nexus_metadata(
            fname, KEYMAP, metadata_fname=bad_override
        )
        assert metadata["energy"] == pytest.approx(11.0)
        assert meta_fname == fname

    def test_raises_when_no_valid_metadata_anywhere(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        with pytest.raises(FileNotFoundError):
            read_nexus_metadata(fname, KEYMAP)
