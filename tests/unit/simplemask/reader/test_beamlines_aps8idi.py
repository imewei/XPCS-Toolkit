"""Unit tests for reader/beamlines/aps_8idi.py and reader/__init__.get_reader."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader import get_reader
from xpcsviewer.simplemask.reader.beamlines.aps_8idi import (
    APS8IDIReader,
    get_nexus_metadata,
)


def _write_8idi_file(path, n_frames=3, shape=(4, 5), energy=11.0, distance=6.0):
    with h5py.File(path, "w") as f:
        data = np.full((n_frames, *shape), 8, dtype=np.uint16)
        f.create_dataset("/entry/data/data", data=data)
        f.create_dataset("/entry/instrument/incident_beam/incident_energy", data=energy)
        f.create_dataset("/entry/instrument/detector_1/distance", data=distance)
        f.create_dataset("/entry/instrument/detector_1/x_pixel_size", data=0.000075)
        f.create_dataset("/entry/instrument/detector_1/y_pixel_size", data=0.000075)
        f.create_dataset("/entry/instrument/detector_1/position_x", data=0.0)
        f.create_dataset("/entry/instrument/detector_1/position_y", data=0.0)
        f.create_dataset("/entry/instrument/detector_1/beam_center_position_x", data=0.0)
        f.create_dataset("/entry/instrument/detector_1/beam_center_position_y", data=0.0)
        f.create_dataset("/entry/instrument/detector_1/beam_center_x", data=100.0)
        f.create_dataset("/entry/instrument/detector_1/beam_center_y", data=50.0)


class TestGetNexusMetadata:
    def test_computes_beam_center_from_translation(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        meta = get_nexus_metadata(fname)
        # ccdx == ccdx0 and ccdy == ccdy0 in this fixture, so beam center
        # equals bcx0/bcy0 directly (no translation offset).
        assert meta["energy"] == pytest.approx(11.0)
        assert meta["detector_distance"] == pytest.approx(6.0)
        assert meta["pixel_size"] == pytest.approx(0.000075)
        assert "bcx0" not in meta
        assert "x_pixel_size" not in meta

    def test_missing_required_field_raises(self, tmp_path):
        fname = str(tmp_path / "incomplete.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2, 2)))
        with pytest.raises((KeyError, FileNotFoundError)):
            get_nexus_metadata(fname)


class TestAPS8IDIReader:
    def test_prepare_data_success(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname, n_frames=3, shape=(4, 5))
        reader = APS8IDIReader(fname)
        assert reader.ftype == "APS_8IDI"
        assert reader.stype == "Transmission"
        reader.prepare_data()
        assert reader.scat.shape == (4, 5)
        assert reader.metadata_is_placeholder is False
        assert reader.metadata["energy"] == pytest.approx(11.0)

    def test_forwards_kwargs_to_format_loader(self, tmp_path):
        # get_reader(beamline, fname, **kwargs) forwards kwargs all the way
        # to the format loader; a custom data_path must actually work.
        fname = str(tmp_path / "custom_path.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset(
                "/my/custom/path", data=np.full((2, 3, 4), 9, dtype=np.uint16)
            )
        reader = APS8IDIReader(fname, data_path="/my/custom/path")
        assert reader.shape == (3, 4)

    def test_prepare_data_falls_back_on_bad_metadata(self, tmp_path):
        fname = str(tmp_path / "no_metadata.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.full((2, 3, 4), 1, dtype=np.uint16))
        reader = APS8IDIReader(fname)
        reader.prepare_data()
        assert reader.metadata_is_placeholder is True
        # Placeholder metadata still lets prepare_data() complete.
        assert reader.scat.shape == (3, 4)


class TestGetReader:
    def test_aps_8idi_returns_correct_reader_type(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        reader = get_reader("APS_8IDI", fname)
        assert isinstance(reader, APS8IDIReader)

    def test_unsupported_beamline_raises(self, tmp_path):
        fname = str(tmp_path / "scan.h5")
        _write_8idi_file(fname)
        with pytest.raises(ValueError, match="unsupported beamline"):
            get_reader("APS_9IDD", fname)
