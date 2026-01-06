"""Unit tests for Q-map computation.

Tests numerical accuracy and shape consistency of Q-map generation
for transmission and reflection geometries.
"""

import numpy as np
import pytest

from xpcsviewer.simplemask.qmap import (
    E2KCONST,
    compute_qmap,
    compute_reflection_qmap,
    compute_transmission_qmap,
)


class TestComputeTransmissionQmap:
    """Tests for transmission geometry Q-map computation."""

    def test_qmap_shape_matches_detector(self):
        """Q-map arrays should match detector shape."""
        energy = 10.0  # keV
        center = (256, 512)  # (row, col)
        shape = (512, 1024)
        pix_dim = 0.075  # mm
        det_dist = 5000.0  # mm

        qmap, units = compute_transmission_qmap(
            energy, center, shape, pix_dim, det_dist
        )

        assert qmap["q"].shape == shape
        assert qmap["phi"].shape == shape
        assert qmap["qx"].shape == shape
        assert qmap["qy"].shape == shape
        assert qmap["x"].shape == shape
        assert qmap["y"].shape == shape
        assert qmap["TTH"].shape == shape

    def test_q_units_correct(self):
        """Q-map units should be correct."""
        qmap, units = compute_transmission_qmap(
            10.0, (256, 512), (512, 1024), 0.075, 5000.0
        )

        assert units["q"] == "Å⁻¹"
        assert units["qx"] == "Å⁻¹"
        assert units["qy"] == "Å⁻¹"
        assert units["phi"] == "deg"
        assert units["x"] == "pixel"
        assert units["y"] == "pixel"

    def test_q_at_center_is_zero(self):
        """Q should be zero at beam center."""
        energy = 10.0
        center = (256, 512)
        shape = (512, 1024)
        pix_dim = 0.075
        det_dist = 5000.0

        qmap, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Q at beam center should be approximately zero
        q_center = qmap["q"][center[0], center[1]]
        assert np.abs(q_center) < 1e-10

    def test_q_increases_radially(self):
        """Q should increase with distance from center."""
        energy = 10.0
        center = (256, 512)
        shape = (512, 1024)
        pix_dim = 0.075
        det_dist = 5000.0

        qmap, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Q at center
        q_center = qmap["q"][center[0], center[1]]

        # Q at offset positions
        q_offset_10 = qmap["q"][center[0] + 10, center[1]]
        q_offset_100 = qmap["q"][center[0] + 100, center[1]]

        assert q_offset_10 > q_center
        assert q_offset_100 > q_offset_10

    def test_wavelength_calculation(self):
        """Verify wavelength calculation from energy."""
        energy = 10.0  # keV
        expected_wavelength = E2KCONST / energy  # Angstrom
        expected_k0 = 2 * np.pi / expected_wavelength

        center = (256, 512)
        shape = (512, 1024)
        pix_dim = 0.075
        det_dist = 5000.0

        qmap, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # At small angles, q ~ k0 * sin(theta) ~ k0 * r / det_dist
        # for a pixel at (center[0], center[1] + 100)
        r = 100 * pix_dim  # mm
        theta = np.arctan(r / det_dist)
        expected_q = expected_k0 * np.sin(theta)

        actual_q = qmap["q"][center[0], center[1] + 100]
        assert np.abs(actual_q - expected_q) < 1e-6

    def test_phi_range(self):
        """Phi should be in degrees and cover full range."""
        qmap, _ = compute_transmission_qmap(
            10.0, (256, 512), (512, 1024), 0.075, 5000.0
        )

        phi = qmap["phi"]
        assert phi.min() >= -180.0
        assert phi.max() <= 180.0

    def test_caching_works(self):
        """Cached results should be identical."""
        args = (10.0, (256, 512), (512, 1024), 0.075, 5000.0)

        qmap1, _ = compute_transmission_qmap(*args)
        qmap2, _ = compute_transmission_qmap(*args)

        # Results should be identical (same object due to caching)
        assert qmap1["q"] is qmap2["q"]


class TestComputeReflectionQmap:
    """Tests for reflection geometry Q-map computation."""

    def test_qmap_shape_matches_detector(self):
        """Q-map arrays should match detector shape."""
        energy = 10.0
        center = (256, 512)
        shape = (512, 1024)
        pix_dim = 0.075
        det_dist = 5000.0

        qmap, _ = compute_reflection_qmap(
            energy, center, shape, pix_dim, det_dist, alpha_i_deg=0.14
        )

        assert qmap["q"].shape == shape
        assert qmap["qz"].shape == shape
        assert qmap["qr"].shape == shape

    def test_reflection_has_qz(self):
        """Reflection mode should have qz component."""
        qmap, units = compute_reflection_qmap(
            10.0, (256, 512), (512, 1024), 0.075, 5000.0, alpha_i_deg=0.14
        )

        assert "qz" in qmap
        assert "qr" in qmap
        assert units["qz"] == "Å⁻¹"
        assert units["qr"] == "Å⁻¹"

    def test_orientation_north(self):
        """North orientation should be default."""
        qmap_north, _ = compute_reflection_qmap(
            10.0, (256, 512), (512, 1024), 0.075, 5000.0, orientation="north"
        )
        qmap_default, _ = compute_reflection_qmap(
            10.0, (256, 512), (512, 1024), 0.075, 5000.0
        )

        np.testing.assert_array_equal(qmap_north["q"], qmap_default["q"])


class TestComputeQmap:
    """Tests for the unified compute_qmap function."""

    def test_transmission_mode(self):
        """compute_qmap should work with Transmission mode."""
        metadata = {
            "energy": 10.0,
            "bcx": 512,
            "bcy": 256,
            "shape": (512, 1024),
            "pix_dim": 0.075,
            "det_dist": 5000.0,
        }

        qmap, units = compute_qmap("Transmission", metadata)

        assert "q" in qmap
        assert "phi" in qmap
        assert qmap["q"].shape == (512, 1024)

    def test_reflection_mode(self):
        """compute_qmap should work with Reflection mode."""
        metadata = {
            "energy": 10.0,
            "bcx": 512,
            "bcy": 256,
            "shape": (512, 1024),
            "pix_dim": 0.075,
            "det_dist": 5000.0,
            "alpha_i_deg": 0.14,
            "orientation": "north",
        }

        qmap, units = compute_qmap("Reflection", metadata)

        assert "q" in qmap
        assert "qz" in qmap
        assert qmap["q"].shape == (512, 1024)

    def test_invalid_mode_raises(self):
        """Invalid scattering type should raise ValueError."""
        metadata = {
            "energy": 10.0,
            "bcx": 512,
            "bcy": 256,
            "shape": (512, 1024),
            "pix_dim": 0.075,
            "det_dist": 5000.0,
        }

        with pytest.raises(ValueError, match="Unknown scattering type"):
            compute_qmap("InvalidMode", metadata)


class TestNumericalAccuracy:
    """Tests for numerical accuracy of Q-map computation."""

    def test_small_angle_approximation(self):
        """At small angles, q ~ 4*pi*sin(theta/2)/lambda should hold."""
        energy = 10.0
        wavelength = E2KCONST / energy
        center = (256, 512)
        shape = (512, 1024)
        pix_dim = 0.075
        det_dist = 10000.0  # Large distance for small angles

        qmap, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # At pixel (center[0], center[1] + 10)
        r = 10 * pix_dim
        theta = np.arctan(r / det_dist)

        # Small angle: sin(theta) ~ theta
        expected_q = (2 * np.pi / wavelength) * np.sin(theta)
        actual_q = qmap["q"][center[0], center[1] + 10]

        # Should match to high precision at small angles
        assert np.abs(actual_q - expected_q) / expected_q < 1e-6

    def test_symmetry_about_center(self):
        """Q should be symmetric about beam center."""
        energy = 10.0
        # Use center in the middle of the detector
        center = (250, 500)
        shape = (500, 1000)
        pix_dim = 0.075
        det_dist = 5000.0

        qmap, _ = compute_transmission_qmap(energy, center, shape, pix_dim, det_dist)

        # Q at symmetric points should be equal (within floating point precision)
        # Test vertical symmetry (same column, symmetric rows)
        q_plus_v = qmap["q"][center[0] + 50, center[1]]
        q_minus_v = qmap["q"][center[0] - 50, center[1]]
        np.testing.assert_almost_equal(q_plus_v, q_minus_v, decimal=8)

        # Test horizontal symmetry (same row, symmetric columns)
        q_plus_h = qmap["q"][center[0], center[1] + 50]
        q_minus_h = qmap["q"][center[0], center[1] - 50]
        np.testing.assert_almost_equal(q_plus_h, q_minus_h, decimal=8)
