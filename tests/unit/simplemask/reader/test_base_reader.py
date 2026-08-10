"""Unit tests for reader/base_reader.py FileReader base class."""

import numpy as np
import pytest

from xpcsviewer.simplemask.reader.base_reader import FileReader, get_fake_metadata


class _FakeGoodReader(FileReader):
    """A reader whose metadata read always succeeds."""

    def get_scattering(self, *args, **kwargs):
        return np.full((4, 6), 5, dtype=np.uint16)

    def _get_metadata(self, *args, **kwargs):
        return {
            "energy": 11.0,
            "detector_distance": 5.0,
            "pixel_size": 0.000075,
            "beam_center_x": 3.0,
            "beam_center_y": 2.0,
        }


class _FakeFailingReader(FileReader):
    """A reader whose metadata read always fails."""

    def get_scattering(self, *args, **kwargs):
        return np.zeros((4, 6), dtype=np.uint16)

    def _get_metadata(self, *args, **kwargs):
        raise RuntimeError("simulated NeXus read failure")


def test_get_fake_metadata_has_every_metadata_read_field():
    meta = get_fake_metadata()
    for key in ("energy", "detector_distance", "pixel_size", "beam_center_x", "beam_center_y"):
        assert key in meta
    # detector_shape_x/y are NOT part of get_fake_metadata()'s own contract --
    # prepare_data() adds them afterward from the actual image shape. Pin
    # this down so the docstring's claim can't silently drift out of sync
    # with the code again (see test_prepare_data_falls_back_to_placeholder_
    # on_metadata_failure below for confirmation prepare_data() adds them).
    assert "detector_shape_x" not in meta
    assert "detector_shape_y" not in meta


def test_prepare_data_success_sets_scat_shape_and_metadata():
    reader = _FakeGoodReader("dummy.h5")
    reader.prepare_data()
    assert reader.scat.shape == (4, 6)
    assert reader.scat.dtype == np.float32
    assert reader.shape == (4, 6)
    assert reader.metadata["detector_shape_x"] == 6
    assert reader.metadata["detector_shape_y"] == 4
    assert reader.metadata["energy"] == pytest.approx(11.0)
    assert reader.metadata_is_placeholder is False


def test_prepare_data_falls_back_to_placeholder_on_metadata_failure():
    reader = _FakeFailingReader("dummy.h5")
    reader.prepare_data()
    assert reader.metadata_is_placeholder is True
    # Placeholder metadata still has every field prepare_data() needs.
    assert reader.metadata["detector_shape_x"] == 6
    assert reader.metadata["detector_shape_y"] == 4


def test_get_metadata_coerces_numpy_scalars_to_float():
    class _NumpyReader(FileReader):
        def get_scattering(self, *a, **k):
            return np.zeros((2, 2))

        def _get_metadata(self, *a, **k):
            return {"energy": np.float64(9.5)}

    reader = _NumpyReader("dummy.h5")
    meta = reader.get_metadata()
    assert type(meta["energy"]) is float
    assert meta["energy"] == pytest.approx(9.5)


def test_base_file_reader_cannot_be_instantiated_directly():
    # FileReader is abc.ABC with get_scattering/_get_metadata as
    # @abstractmethod (matching the sibling ScatteringDataset ABC in
    # reader/formats/base.py) -- a subclass missing either override can't
    # be constructed at all. This catches a missing override at
    # construction time rather than only once someone calls the
    # unimplemented method.
    with pytest.raises(TypeError):
        FileReader("dummy.h5")


def test_subclass_missing_get_scattering_cannot_be_instantiated():
    class _MissingGetScattering(FileReader):
        def _get_metadata(self, *a, **k):
            return {}

    with pytest.raises(TypeError):
        _MissingGetScattering("dummy.h5")


def test_subclass_missing_get_metadata_cannot_be_instantiated():
    class _MissingGetMetadata(FileReader):
        def get_scattering(self, *a, **k):
            return np.zeros((2, 2))

    with pytest.raises(TypeError):
        _MissingGetMetadata("dummy.h5")
