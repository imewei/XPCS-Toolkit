"""Tests for Q-map tolerance (T084).

Tests that Q-map values match within 1e-6 relative tolerance per SC-004.
"""

from __future__ import annotations

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
class TestQMapTolerance:
    """Tests for Q-map tolerance requirements (SC-004)."""

    def test_qmap_numpy_jax_equivalence(self, monkeypatch) -> None:
        """Test Q-map values match between NumPy and JAX within 1e-6 tolerance."""
        # Compute with NumPy backend
        monkeypatch.setenv("XPCS_USE_JAX", "0")

        from xpcsviewer.backends import _reset_backend, set_backend

        _reset_backend()
        set_backend("numpy")

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap_numpy, units_numpy = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Compute with JAX backend
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        _reset_backend()

        qmap_jax, units_jax = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Compare all Q-map arrays
        for key in qmap_numpy.keys():
            numpy_vals = qmap_numpy[key]
            jax_vals = qmap_jax[key]

            # Calculate relative tolerance
            # Use absolute tolerance for values near zero
            np.testing.assert_allclose(
                jax_vals,
                numpy_vals,
                rtol=1e-6,
                atol=1e-12,
                err_msg=f"Q-map '{key}' differs between NumPy and JAX",
            )

    def test_qmap_reproducibility(self, monkeypatch) -> None:
        """Test Q-map computation is reproducible."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Compute twice
        qmap1, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        qmap2, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Should be exactly equal
        for key in qmap1.keys():
            np.testing.assert_array_equal(
                qmap1[key], qmap2[key], err_msg=f"Q-map '{key}' not reproducible"
            )

    def test_qmap_different_parameters(self, monkeypatch) -> None:
        """Test Q-map produces different results for different parameters."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        # Different energies should produce different Q values
        qmap_10kev, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        qmap_20kev, _ = compute_transmission_qmap(
            energy=20.0,  # Different energy
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Q is proportional to wavelength (inversely proportional to energy)
        # So at higher energy, Q should be larger for the same pixel
        q_ratio = qmap_20kev["q"][100, 100] / qmap_10kev["q"][100, 100]
        assert abs(q_ratio - 2.0) < 0.01  # Should be approximately 2x


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestPartitionTolerance:
    """Tests for partition tolerance requirements."""

    def test_partition_numpy_jax_equivalence(self, monkeypatch) -> None:
        """Test partition matches between NumPy and JAX."""
        # Compute with NumPy backend
        monkeypatch.setenv("XPCS_USE_JAX", "0")

        from xpcsviewer.backends import _reset_backend, set_backend

        _reset_backend()
        set_backend("numpy")

        from xpcsviewer.simplemask.utils import generate_partition

        qmap = np.random.random((128, 128)).astype(np.float64)
        mask = np.ones((128, 128), dtype=bool)

        result_numpy = generate_partition(
            map_name="q", mask=mask, xmap=qmap, num_pts=36, style="linear"
        )
        partition_numpy = result_numpy["partition"]

        # Compute with JAX backend
        monkeypatch.setenv("XPCS_USE_JAX", "1")
        _reset_backend()

        result_jax = generate_partition(
            map_name="q", mask=mask, xmap=qmap, num_pts=36, style="linear"
        )
        partition_jax = result_jax["partition"]

        # Partitions should be identical (integer labels)
        np.testing.assert_array_equal(
            partition_jax,
            partition_numpy,
            err_msg="Partition differs between NumPy and JAX",
        )


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestWavelengthPrecision:
    """Tests for wavelength/energy conversion precision."""

    def test_energy_to_wavelength_precision(self, monkeypatch) -> None:
        """Test energy to wavelength conversion precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import E2KCONST

        # E2KCONST = hc = 12.398 keV·Å
        # wavelength (Å) = E2KCONST / energy (keV)
        # 10 keV X-rays
        energy = 10.0
        wavelength = E2KCONST / energy

        # Expected: 1.2398 Å
        expected = 1.2398
        assert abs(wavelength - expected) < 1e-4

    def test_qmap_wavelength_consistency(self, monkeypatch) -> None:
        """Test Q-map uses correct wavelength."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, units = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        # Check Q values are in reasonable range for 10 keV X-rays
        # Q = 4π/λ * sin(θ/2), λ ≈ 1.24 Å for 10 keV
        # For typical SAXS/XPCS, Q should be in range 0 to ~0.5 Å⁻¹

        q_max = np.nanmax(qmap["q"])
        q_min = np.nanmin(qmap["q"])

        # Minimum Q at center should be near 0
        assert q_min < 0.01

        # Maximum Q for this geometry should be reasonable
        # At edge (64 pixels * 0.075 mm = 4.8 mm from center)
        # θ = arctan(4.8/5000) ≈ 0.001 rad
        # Q ≈ 4π/1.24 * sin(0.0005) ≈ 0.005 Å⁻¹
        assert 0.001 < q_max < 0.1


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestPixelPositionPrecision:
    """Tests for pixel position precision."""

    def test_pixel_grid_precision(self, monkeypatch) -> None:
        """Test pixel grid computation precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Create pixel grid
        shape = (128, 128)
        center = (64.0, 64.0)

        # Use backend array creation (float64 is default)
        y_coords = backend.arange(shape[0])
        x_coords = backend.arange(shape[1])
        yy, xx = backend.meshgrid(y_coords, x_coords, indexing="ij")

        dy = yy - center[0]
        dx = xx - center[1]

        # At center, displacement should be exactly 0
        center_dy = float(dy[64, 64])
        center_dx = float(dx[64, 64])

        assert abs(center_dy) < 1e-14
        assert abs(center_dx) < 1e-14

        # One pixel away should be exactly 1
        assert abs(float(dy[65, 64]) - 1.0) < 1e-14
        assert abs(float(dx[64, 65]) - 1.0) < 1e-14
