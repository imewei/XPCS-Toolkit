"""Unit tests for SimpleMask save/load functionality.

Tests mask persistence to HDF5 files, dimension validation,
and round-trip accuracy.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from xpcsviewer.simplemask.simplemask_kernel import SimpleMaskKernel


class TestMaskSave:
    """Tests for save_mask functionality."""

    def test_save_mask_creates_file(self):
        """save_mask should create HDF5 file."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)
            assert Path(save_path).exists()
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_mask_contains_mask_dataset(self):
        """Saved file should contain 'mask' dataset."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                assert "mask" in hf
                assert hf["mask"].shape == (100, 200)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_mask_preserves_shape(self):
        """Saved mask should preserve original shape."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((512, 1024), dtype=np.float64)
        metadata = {
            "bcx": 512,
            "bcy": 256,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                assert hf["mask"].shape == (512, 1024)
                assert tuple(hf.attrs["shape"]) == (512, 1024)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_mask_stores_version(self):
        """Saved file should contain version attribute."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                assert "version" in hf.attrs
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_save_mask_without_data(self):
        """save_mask should handle no mask gracefully."""
        kernel = SimpleMaskKernel()
        # No data loaded

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)
            # Should not create file or should create empty file
            # Implementation allows empty save
        finally:
            Path(save_path).unlink(missing_ok=True)


class TestMaskLoad:
    """Tests for load_mask functionality."""

    def test_load_mask_from_file(self):
        """load_mask should load mask from HDF5 file."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        # Create test mask file
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            # Create a mask with some False values
            test_mask = np.ones((100, 200), dtype=np.uint8)
            test_mask[20:40, 50:100] = 0  # Masked region

            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("mask", data=test_mask)

            success = kernel.load_mask(save_path)
            assert success
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_load_mask_without_kernel_init(self):
        """load_mask should return False if kernel not initialized."""
        kernel = SimpleMaskKernel()
        # No read_data called

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            with h5py.File(save_path, "w") as hf:
                hf.create_dataset("mask", data=np.ones((100, 200), dtype=np.uint8))

            success = kernel.load_mask(save_path)
            assert not success
        finally:
            Path(save_path).unlink(missing_ok=True)


class TestMaskRoundTrip:
    """Tests for save/load round-trip accuracy."""

    def test_round_trip_preserves_all_ones(self):
        """Round-trip should preserve all-ones mask."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        original_mask = kernel.mask.copy()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            # Create new kernel and load
            kernel2 = SimpleMaskKernel()
            kernel2.read_data(detector_image, metadata)
            kernel2.load_mask(save_path)

            # Compare masks
            np.testing.assert_array_equal(kernel2.mask, original_mask)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_round_trip_preserves_mask_pattern(self):
        """Round-trip should preserve mask pattern."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        # Modify mask with pattern
        kernel.mask[10:30, 20:50] = False
        kernel.mask[60:80, 100:150] = False

        original_mask = kernel.mask.copy()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            # Load and compare
            with h5py.File(save_path, "r") as hf:
                loaded_mask = hf["mask"][:]

            # Saved as uint8, convert back to bool for comparison
            loaded_mask_bool = loaded_mask.astype(bool)
            np.testing.assert_array_equal(loaded_mask_bool, original_mask)
        finally:
            Path(save_path).unlink(missing_ok=True)


class TestMaskFileFormat:
    """Tests for HDF5 file format compliance."""

    def test_mask_uses_compression(self):
        """Large masks should use compression."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((1024, 2048), dtype=np.float64)
        metadata = {
            "bcx": 1024,
            "bcy": 512,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                # Check compression is applied
                assert hf["mask"].compression == "lzf"
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_mask_stored_as_uint8(self):
        """Mask should be stored as uint8 to save space."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((100, 200), dtype=np.float64)
        metadata = {
            "bcx": 100,
            "bcy": 50,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                assert hf["mask"].dtype == np.uint8
        finally:
            Path(save_path).unlink(missing_ok=True)


class TestMaskDimensionValidation:
    """Tests for mask dimension validation (indirect via kernel)."""

    def test_mask_shape_attribute_stored(self):
        """Shape should be stored as file attribute."""
        kernel = SimpleMaskKernel()
        detector_image = np.ones((256, 512), dtype=np.float64)
        metadata = {
            "bcx": 256,
            "bcy": 128,
            "det_dist": 5000.0,
            "pix_dim": 0.075,
            "energy": 10.0,
        }
        kernel.read_data(detector_image, metadata)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            save_path = f.name

        try:
            kernel.save_mask(save_path)

            with h5py.File(save_path, "r") as hf:
                assert "shape" in hf.attrs
                stored_shape = tuple(hf.attrs["shape"])
                assert stored_shape == (256, 512)
        finally:
            Path(save_path).unlink(missing_ok=True)
