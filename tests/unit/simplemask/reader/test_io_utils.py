"""Unit tests for reader/io_utils.py frame-range resolution and averaging."""

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.reader.io_utils import (
    average_frames_parallel,
    resolve_frame_range,
)


class TestResolveFrameRange:
    def test_positive_num_frames_clamped_to_end(self):
        assert resolve_frame_range(total_frames=100, start_frame=90, num_frames=50) == 10

    def test_positive_num_frames_within_range(self):
        assert resolve_frame_range(total_frames=100, start_frame=0, num_frames=20) == 20

    def test_zero_means_all_remaining(self):
        assert resolve_frame_range(total_frames=100, start_frame=30, num_frames=0) == 70

    def test_none_means_all_remaining(self):
        assert resolve_frame_range(total_frames=100, start_frame=30, num_frames=None) == 70

    def test_negative_means_representative_subset_floor(self):
        # total_frames // 5 = 200, max(1000, 200) = 1000
        assert resolve_frame_range(total_frames=1000, start_frame=0, num_frames=-1) == 1000

    def test_negative_means_representative_subset_fraction(self):
        # total_frames // 5 = 2000, max(1000, 2000) = 2000
        assert resolve_frame_range(total_frames=10000, start_frame=0, num_frames=-1) == 2000

    def test_negative_subset_clamped_to_remaining(self):
        # representative subset would be 1000, but only 40 frames remain
        assert resolve_frame_range(total_frames=100, start_frame=60, num_frames=-1) == 40

    def test_start_frame_out_of_range_raises(self):
        with pytest.raises(ValueError):
            resolve_frame_range(total_frames=100, start_frame=100, num_frames=10)

    def test_negative_start_frame_raises(self):
        with pytest.raises(ValueError):
            resolve_frame_range(total_frames=100, start_frame=-1, num_frames=10)


class TestAverageFramesParallel:
    def _write_stack(self, path, n_frames=5, shape=(4, 6), fill_value=None):
        with h5py.File(path, "w") as f:
            if fill_value is None:
                data = np.arange(n_frames * shape[0] * shape[1], dtype=np.uint16).reshape(
                    (n_frames, *shape)
                )
            else:
                data = np.full((n_frames, *shape), fill_value, dtype=np.uint16)
            f.create_dataset("/entry/data/data", data=data)

    def test_averages_all_frames_by_default(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        self._write_stack(fname, n_frames=4, shape=(2, 2), fill_value=10)
        result = average_frames_parallel(fname, num_frames=0)
        assert result.shape == (2, 2)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 10.0)

    def test_averages_frame_subset(self, tmp_path):
        fname = str(tmp_path / "stack.h5")
        with h5py.File(fname, "w") as f:
            # frame 0 is all zeros, frame 1 is all tens; average of just frame 1 is 10
            data = np.zeros((2, 3, 3), dtype=np.uint16)
            data[1] = 10
            f.create_dataset("/entry/data/data", data=data)
        result = average_frames_parallel(fname, start_frame=1, num_frames=1)
        np.testing.assert_allclose(result, 10.0)

    def test_rejects_non_3d_dataset(self, tmp_path):
        fname = str(tmp_path / "flat.h5")
        with h5py.File(fname, "w") as f:
            f.create_dataset("/entry/data/data", data=np.zeros((4, 4)))
        with pytest.raises(ValueError):
            average_frames_parallel(fname)
