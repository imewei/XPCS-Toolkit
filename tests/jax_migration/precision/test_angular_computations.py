"""Tests for angular computation precision (T083).

Tests that angular computations maintain required precision.
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
class TestArctan2Precision:
    """Tests for arctan2 precision."""

    def test_arctan2_quadrant_boundaries(self, monkeypatch) -> None:
        """Test arctan2 precision at quadrant boundaries."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Test all four quadrant boundaries
        test_cases = [
            ((1.0, 0.0), 0.0),  # Positive x-axis
            ((0.0, 1.0), np.pi / 2),  # Positive y-axis
            ((-1.0, 0.0), np.pi),  # Negative x-axis
            ((0.0, -1.0), -np.pi / 2),  # Negative y-axis
        ]

        for (x, y), expected in test_cases:
            result = backend.arctan2(backend.array(y), backend.array(x))
            assert abs(float(result) - expected) < 1e-14, f"Failed for ({x}, {y})"

    def test_arctan2_small_angles(self, monkeypatch) -> None:
        """Test arctan2 precision for small angles."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Small angle: arctan2(1e-8, 1) ≈ 1e-8
        small_y = backend.array(1e-8)
        x = backend.array(1.0)

        result = backend.arctan2(small_y, x)

        # For small angles, arctan(y/x) ≈ y/x
        expected = 1e-8
        assert abs(float(result) - expected) < 1e-14


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestThetaPhiPrecision:
    """Tests for theta/phi (polar angle) precision."""

    def test_phi_map_full_circle(self, monkeypatch) -> None:
        """Test phi map covers full circle with correct values."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,
            det_dist=5000.0,
        )

        phi = qmap["phi"]

        # phi should range from -180 to 180 degrees
        assert np.nanmin(phi) >= -180.0
        assert np.nanmax(phi) <= 180.0

        # Check cardinal directions
        # Right of center (positive x): phi ≈ 0
        phi_right = phi[64, 80]  # To the right of center
        assert abs(phi_right) < 5.0  # Should be near 0

        # Above center: phi ≈ ±90 (sign depends on coordinate convention)
        phi_up = phi[80, 64]  # Above center
        assert abs(abs(phi_up) - 90.0) < 5.0  # Should be near ±90

    def test_q_map_increases_from_center(self, monkeypatch) -> None:
        """Test Q increases with distance from center."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend

        _reset_backend()

        from xpcsviewer.simplemask.qmap import compute_transmission_qmap

        qmap, _ = compute_transmission_qmap(
            energy=10.0,
            center=(64.0, 64.0),
            shape=(128, 128),
            pix_dim=0.075,  # 75 micron pixels
            det_dist=5000.0,  # 5 meter distance
        )

        q = qmap["q"]

        # At center, Q should be 0
        center_q = q[64, 64]
        assert abs(center_q) < 1e-10

        # Check Q increases with distance from center
        near_q = q[65, 64]  # 1 pixel from center
        far_q = q[100, 64]  # 36 pixels from center

        assert far_q > near_q > 0


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestSinCosPrecision:
    """Tests for sin/cos precision in angular computations."""

    def test_sin_cos_identity(self, monkeypatch) -> None:
        """Test sin²x + cos²x = 1 precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Test various angles
        angles = backend.linspace(0, 2 * backend.pi, 1000)

        sin_vals = backend.sin(angles)
        cos_vals = backend.cos(angles)

        identity = sin_vals**2 + cos_vals**2

        # All values should be very close to 1
        max_error = float(backend.max(backend.abs(identity - 1.0)))
        assert max_error < 1e-14

    def test_angle_to_q_precision(self, monkeypatch) -> None:
        """Test angle to Q conversion precision."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Q = 4π/λ * sin(θ/2)
        # For small angles, Q ≈ 2π/λ * θ
        wavelength = 1.24e-10  # 10 keV
        small_theta = backend.array(1e-6)  # Small angle in radians

        q_exact = 4 * backend.pi / wavelength * backend.sin(small_theta / 2)
        q_approx = 2 * backend.pi / wavelength * small_theta

        # For small angles, these should be very close
        relative_error = abs(float(q_exact - q_approx)) / float(q_exact)
        assert relative_error < 1e-10


@pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
class TestHypotPrecision:
    """Tests for hypot (Euclidean distance) precision."""

    def test_hypot_pythagorean(self, monkeypatch) -> None:
        """Test hypot produces correct Pythagorean results."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Classic 3-4-5 triangle
        x = backend.array(3.0)
        y = backend.array(4.0)
        result = backend.hypot(x, y)

        assert abs(float(result) - 5.0) < 1e-14

    def test_hypot_small_values(self, monkeypatch) -> None:
        """Test hypot precision for small values."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Very small values (sub-pixel distances)
        x = backend.array(1e-10)
        y = backend.array(1e-10)
        result = backend.hypot(x, y)

        expected = np.sqrt(2) * 1e-10
        relative_error = abs(float(result) - expected) / expected
        assert relative_error < 1e-10

    def test_hypot_large_values(self, monkeypatch) -> None:
        """Test hypot doesn't overflow for large values."""
        monkeypatch.setenv("XPCS_USE_JAX", "1")

        from xpcsviewer.backends import _reset_backend, get_backend

        _reset_backend()

        backend = get_backend()

        # Large values that would overflow if squared naively
        x = backend.array(1e150)
        y = backend.array(1e150)
        result = backend.hypot(x, y)

        # Should not overflow or produce inf
        assert np.isfinite(float(result))

        expected = np.sqrt(2) * 1e150
        relative_error = abs(float(result) - expected) / expected
        assert relative_error < 1e-10
