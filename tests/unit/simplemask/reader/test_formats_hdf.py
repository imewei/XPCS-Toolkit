"""Unit tests for reader/formats/hdf.py and format dispatch."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader.formats import get_format_loader
from xpcsviewer.simplemask.reader.formats.hdf import HdfDataset


class TestHdfDataset:
    def test_2d_dataset_shape_and_dtype(self, tmp_path):
        fname = str(tmp_path / "img.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.full((5, 7), 3, dtype=np.uint16))
        loader = HdfDataset(fname)
        assert loader.det_size == (5, 7)
        img = loader.get_scattering()
        assert img.shape == (5, 7)
        assert img.dtype == np.float32
        np.testing.assert_allclose(img, 3.0)

    def test_3d_dataset_shape_is_frame_shape(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((10, 5, 7), dtype=np.uint16))
        loader = HdfDataset(fname)
        assert loader.det_size == (5, 7)

    def test_3d_dataset_averages_frames(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            data = np.zeros((4, 2, 2), dtype=np.uint16)
            data[:] = 20
            f.create_dataset("/entry/data/data", data=data)
        loader = HdfDataset(fname)
        img = loader.get_scattering(num_frames=0)
        np.testing.assert_allclose(img, 20.0)

    def test_missing_dataset_path_raises(self, tmp_path):
        fname = str(tmp_path / "empty.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/wrong/path", data=np.zeros((2, 2)))
        with pytest.raises(KeyError):
            HdfDataset(fname)

    def test_custom_data_path(self, tmp_path):
        fname = str(tmp_path / "custom.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/my/custom/path", data=np.full((3, 3), 7, dtype=np.uint16))
        loader = HdfDataset(fname, data_path="/my/custom/path")
        img = loader.get_scattering()
        np.testing.assert_allclose(img, 7.0)

    def test_file_size_mb_positive(self, tmp_path):
        fname = str(tmp_path / "img.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((100, 100), dtype=np.uint16))
        loader = HdfDataset(fname)
        assert loader.file_size_mb > 0


class TestGetFormatLoader:
    def test_h5_extension_returns_hdf_dataset(self, tmp_path):
        fname = str(tmp_path / "data.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_hdf_extension_returns_hdf_dataset(self, tmp_path):
        fname = str(tmp_path / "data.hdf")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_hdf5_extension_returns_hdf_dataset(self, tmp_path):
        # .hdf5 is one of the extensions offered by the "Open Raw File"
        # dialog filter (Task 9) -- it must actually be accepted here.
        fname = str(tmp_path / "data.hdf5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((2, 2)))
        loader = get_format_loader(fname)
        assert isinstance(loader, HdfDataset)

    def test_unrecognized_extension_raises(self, tmp_path):
        fname = str(tmp_path / "data.xyz")
        with pytest.raises(ValueError, match="unsupported dataset file"):
            get_format_loader(fname)
