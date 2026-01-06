"""Tests for HDF5/JAX I/O (T075).

Tests that HDF5 I/O works correctly with JAX backend (US5).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestHDF5ReadWithJAX:
    """Tests for reading HDF5 files with JAX backend."""

    def test_read_hdf5_data_to_jax(self, monkeypatch, tmp_path) -> None:
        """Test reading HDF5 data and converting to JAX arrays."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Create test HDF5 file
        test_file = tmp_path / "test_data.h5"
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=np.random.random((100, 100)))
            f.create_dataset("mask", data=np.ones((100, 100), dtype=bool))

        # Read and convert
        with h5py.File(test_file, "r") as f:
            numpy_data = f["data"][:]
            numpy_mask = f["mask"][:]

        # Convert to backend array
        backend_data = backend.from_numpy(numpy_data)
        backend_mask = backend.from_numpy(numpy_mask)

        # Perform computation
        result = backend.sum(backend.where(backend_mask, backend_data, 0))

        assert not np.isnan(float(result))

    def test_qmap_computation_from_hdf5(self, monkeypatch, tmp_path) -> None:
        """Test Q-map computation from HDF5 geometry data."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Create test HDF5 file with geometry
        test_file = tmp_path / "geometry.h5"
        with h5py.File(test_file, "w") as f:
            f.create_dataset("energy", data=10.0)
            f.create_dataset("beam_center_x", data=64.0)
            f.create_dataset("beam_center_y", data=64.0)
            f.create_dataset("pixel_size", data=0.075)
            f.create_dataset("detector_distance", data=5000.0)

        # Read geometry and compute Q-map
        with h5py.File(test_file, "r") as f:
            energy = float(f["energy"][()])
            bcx = float(f["beam_center_x"][()])
            bcy = float(f["beam_center_y"][()])
            pix_dim = float(f["pixel_size"][()])
            det_dist = float(f["detector_distance"][()])

        qmap, units = compute_transmission_qmap(
            energy=energy,
            center=(bcy, bcx),
            shape=(128, 128),
            pix_dim=pix_dim,
            det_dist=det_dist,
        )

        assert "q" in qmap
        assert isinstance(qmap["q"], np.ndarray)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestHDF5WriteWithJAX:
    """Tests for writing results to HDF5 from JAX backend."""

    def test_write_qmap_to_hdf5(self, monkeypatch, tmp_path) -> None:
        """Test writing Q-map results to HDF5."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Compute Q-map
        qmap, units = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Write to HDF5
        output_file = tmp_path / "qmap_output.h5"
        with h5py.File(output_file, "w") as f:
            qmap_group = f.create_group("qmap")
            for key, data in qmap.items():
                qmap_group.create_dataset(key, data=data)

            units_group = f.create_group("units")
            for key, unit in units.items():
                units_group.attrs[key] = unit

        # Verify written data
        with h5py.File(output_file, "r") as f:
            assert "qmap/q" in f
            loaded_q = f["qmap/q"][:]
            np.testing.assert_array_equal(loaded_q, qmap["q"])

    def test_write_partition_to_hdf5(self, monkeypatch, tmp_path) -> None:
        """Test writing partition results to HDF5."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.utils import generate_partition

        # Generate partition
        qmap = np.random.random((128, 128)).astype(np.float64)
        mask = np.ones((128, 128), dtype=bool)

        result = generate_partition(
            map_name="q", mask=mask, xmap=qmap, num_pts=36, style="linear"
        )
        # generate_partition returns a dict with 'partition' key
        partition = result["partition"]

        # Write to HDF5
        output_file = tmp_path / "partition.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset("partition", data=partition)

        # Verify
        with h5py.File(output_file, "r") as f:
            loaded = f["partition"][:]
            np.testing.assert_array_equal(loaded, partition)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestHDF5MaskHandling:
    """Tests for mask handling with HDF5 and JAX."""

    def test_read_mask_from_hdf5(self, monkeypatch, tmp_path) -> None:
        """Test reading boolean mask from HDF5."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Create test mask file
        mask_data = np.random.random((100, 100)) > 0.5
        test_file = tmp_path / "mask.h5"
        with h5py.File(test_file, "w") as f:
            f.create_dataset("mask", data=mask_data, dtype=bool)

        # Read mask
        with h5py.File(test_file, "r") as f:
            loaded_mask = f["mask"][:]

        # Convert to backend and use
        backend_mask = backend.from_numpy(loaded_mask.astype(np.bool_))

        # Apply mask to data
        data = backend.ones((100, 100))
        masked_sum = backend.sum(backend.where(backend_mask, data, 0))

        expected_sum = np.sum(mask_data)
        assert abs(float(masked_sum) - expected_sum) < 1e-10

    def test_write_mask_to_hdf5(self, monkeypatch, tmp_path) -> None:
        """Test writing mask to HDF5 after JAX processing."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Create mask using backend
        x = backend.linspace(-1, 1, 100)
        y = backend.linspace(-1, 1, 100)
        xx, yy = backend.meshgrid(x, y, indexing="ij")
        mask = (xx**2 + yy**2) < 0.5

        # Convert to NumPy for HDF5
        numpy_mask = ensure_numpy(mask).astype(np.bool_)

        # Write to HDF5
        output_file = tmp_path / "mask_output.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset("mask", data=numpy_mask, dtype=bool)

        # Verify
        with h5py.File(output_file, "r") as f:
            loaded = f["mask"][:]
            np.testing.assert_array_equal(loaded, numpy_mask)


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestHDF5CompressionWithJAX:
    """Tests for HDF5 compression with JAX data."""

    def test_compressed_qmap_storage(self, monkeypatch, tmp_path) -> None:
        """Test compressed storage of Q-map data."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Compute large Q-map
        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(256.0, 256.0),
            shape=(512, 512),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Write with compression
        output_file = tmp_path / "compressed_qmap.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset(
                "q", data=qmap["q"], compression="gzip", compression_opts=4
            )

        # Verify data integrity after compression
        with h5py.File(output_file, "r") as f:
            loaded_q = f["q"][:]
            np.testing.assert_allclose(loaded_q, qmap["q"], rtol=1e-10)

    def test_chunked_storage_for_large_data(self, monkeypatch, tmp_path) -> None:
        """Test chunked storage for large datasets."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend
        from xpcsviewer.backends._conversions import ensure_numpy

        _reset_backend()

        backend = get_backend()

        # Create large array
        large_data = backend.linspace(0, 1, 1000000).reshape(1000, 1000)
        numpy_data = ensure_numpy(large_data)

        # Write with chunking
        output_file = tmp_path / "chunked_data.h5"
        with h5py.File(output_file, "w") as f:
            f.create_dataset(
                "data",
                data=numpy_data,
                chunks=(100, 100),
                compression="gzip",
            )

        # Verify
        with h5py.File(output_file, "r") as f:
            # Read partial chunk
            partial = f["data"][0:100, 0:100]
            np.testing.assert_allclose(partial, numpy_data[0:100, 0:100])
