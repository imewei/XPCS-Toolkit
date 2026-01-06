"""Tests for Q-map numerical equivalence between NumPy and JAX backends.

This module verifies that Q-map computations produce identical results
regardless of which backend is used, ensuring numerical stability and
correctness of the backend abstraction layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from xpcsviewer.backends import get_backend, set_backend
from xpcsviewer.backends._numpy_backend import NumPyBackend
from xpcsviewer.simplemask.qmap import compute_qmap

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Check if JAX is available
try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def create_test_metadata(
    shape: tuple[int, int] = (256, 256),
    bcx: float = 128.0,
    bcy: float = 128.0,
    det_dist: float = 5000.0,
    pix_dim: float = 0.075,
    energy: float = 10.0,
) -> dict:
    """Create test metadata for Q-map computation."""
    return {
        "shape": shape,
        "bcx": bcx,
        "bcy": bcy,
        "det_dist": det_dist,
        "pix_dim": pix_dim,
        "energy": energy,
    }


class TestQmapNumericalEquivalence:
    """Test Q-map numerical equivalence between backends."""

    def test_transmission_qmap_numpy_baseline(self) -> None:
        """Test that transmission Q-map computation works with NumPy backend."""
        set_backend("numpy")
        metadata = create_test_metadata()

        qmap, units = compute_qmap("Transmission", metadata)

        assert "q" in qmap
        assert "phi" in qmap
        assert "x" in qmap
        assert "y" in qmap
        assert qmap["q"].shape == metadata["shape"]
        assert np.all(qmap["q"] >= 0)  # Q values are non-negative
        assert np.all(np.abs(qmap["phi"]) <= 180)  # Phi in [-180, 180]
        assert "Å" in units["q"]  # Q units should contain Angstrom

    def test_reflection_qmap_numpy_baseline(self) -> None:
        """Test that reflection Q-map computation works with NumPy backend."""
        set_backend("numpy")
        metadata = create_test_metadata()
        metadata["incidence_angle_deg"] = 0.1  # Add incidence angle for reflection

        qmap, units = compute_qmap("Reflection", metadata)

        assert "q" in qmap
        assert "phi" in qmap
        assert qmap["q"].shape == metadata["shape"]
        assert "Å" in units["q"]  # Q units should contain Angstrom

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_transmission_qmap_jax_equivalence(self) -> None:
        """Test that JAX produces identical Q-map to NumPy."""
        metadata = create_test_metadata()

        # Compute with NumPy
        set_backend("numpy")
        qmap_np, units_np = compute_qmap("Transmission", metadata)

        # Compute with JAX
        set_backend("jax")
        qmap_jax, units_jax = compute_qmap("Transmission", metadata)

        # Reset to default
        set_backend("numpy")

        # Verify keys match
        assert set(qmap_np.keys()) == set(qmap_jax.keys())
        assert set(units_np.keys()) == set(units_jax.keys())

        # Verify values are equivalent within tolerance (1e-6 per SC-004)
        for key in qmap_np:
            np.testing.assert_allclose(
                qmap_np[key],
                qmap_jax[key],
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Q-map '{key}' values differ between NumPy and JAX",
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_reflection_qmap_jax_equivalence(self) -> None:
        """Test that JAX produces identical reflection Q-map to NumPy."""
        metadata = create_test_metadata()
        metadata["incidence_angle_deg"] = 0.1

        # Compute with NumPy
        set_backend("numpy")
        qmap_np, _ = compute_qmap("Reflection", metadata)

        # Compute with JAX
        set_backend("jax")
        qmap_jax, _ = compute_qmap("Reflection", metadata)

        # Reset to default
        set_backend("numpy")

        for key in qmap_np:
            np.testing.assert_allclose(
                qmap_np[key],
                qmap_jax[key],
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Reflection Q-map '{key}' values differ between backends",
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_qmap_beam_center_variations(self) -> None:
        """Test Q-map equivalence with various beam center positions."""
        beam_centers = [
            (128, 128),  # Center
            (64, 64),  # Off-center
            (192, 64),  # Asymmetric
            (128.5, 128.7),  # Fractional
        ]

        for bcx, bcy in beam_centers:
            metadata = create_test_metadata(bcx=bcx, bcy=bcy)

            set_backend("numpy")
            qmap_np, _ = compute_qmap("Transmission", metadata)

            set_backend("jax")
            qmap_jax, _ = compute_qmap("Transmission", metadata)

            set_backend("numpy")

            for key in ["q", "phi"]:
                np.testing.assert_allclose(
                    qmap_np[key],
                    qmap_jax[key],
                    rtol=1e-6,
                    atol=1e-10,
                    err_msg=f"Q-map '{key}' differs at beam center ({bcx}, {bcy})",
                )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_qmap_detector_distance_variations(self) -> None:
        """Test Q-map equivalence with various detector distances."""
        distances = [1000.0, 5000.0, 10000.0, 20000.0]  # mm

        for det_dist in distances:
            metadata = create_test_metadata(det_dist=det_dist)

            set_backend("numpy")
            qmap_np, _ = compute_qmap("Transmission", metadata)

            set_backend("jax")
            qmap_jax, _ = compute_qmap("Transmission", metadata)

            set_backend("numpy")

            np.testing.assert_allclose(
                qmap_np["q"],
                qmap_jax["q"],
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Q values differ at detector distance {det_dist}mm",
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_qmap_energy_variations(self) -> None:
        """Test Q-map equivalence with various X-ray energies."""
        energies = [8.0, 10.0, 12.0, 15.0]  # keV

        for energy in energies:
            metadata = create_test_metadata(energy=energy)

            set_backend("numpy")
            qmap_np, _ = compute_qmap("Transmission", metadata)

            set_backend("jax")
            qmap_jax, _ = compute_qmap("Transmission", metadata)

            set_backend("numpy")

            np.testing.assert_allclose(
                qmap_np["q"],
                qmap_jax["q"],
                rtol=1e-6,
                atol=1e-10,
                err_msg=f"Q values differ at energy {energy}keV",
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_qmap_large_array(self) -> None:
        """Test Q-map equivalence with large detector arrays."""
        metadata = create_test_metadata(shape=(1024, 1024), bcx=512.0, bcy=512.0)

        set_backend("numpy")
        qmap_np, _ = compute_qmap("Transmission", metadata)

        set_backend("jax")
        qmap_jax, _ = compute_qmap("Transmission", metadata)

        set_backend("numpy")

        np.testing.assert_allclose(
            qmap_np["q"],
            qmap_jax["q"],
            rtol=1e-6,
            atol=1e-10,
            err_msg="Q values differ for large array (1024x1024)",
        )

    def test_qmap_output_is_numpy(self) -> None:
        """Verify Q-map output is always NumPy array at I/O boundary."""
        set_backend("numpy")
        metadata = create_test_metadata()
        qmap, _ = compute_qmap("Transmission", metadata)

        for key, arr in qmap.items():
            assert isinstance(arr, np.ndarray), (
                f"Q-map '{key}' should be NumPy array, got {type(arr)}"
            )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_qmap_output_is_numpy_with_jax(self) -> None:
        """Verify Q-map output is NumPy even when using JAX backend."""
        set_backend("jax")
        metadata = create_test_metadata()
        qmap, _ = compute_qmap("Transmission", metadata)
        set_backend("numpy")

        for key, arr in qmap.items():
            assert isinstance(arr, np.ndarray), (
                f"Q-map '{key}' should be NumPy array even with JAX backend, got {type(arr)}"
            )
