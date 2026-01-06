"""Tests for partition numerical equivalence between NumPy and JAX backends.

This module verifies that partition/Q-binning computations produce identical
results regardless of which backend is used, ensuring numerical stability
and correctness of the backend abstraction layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from xpcsviewer.backends import set_backend
from xpcsviewer.simplemask.utils import combine_partitions, generate_partition

if TYPE_CHECKING:
    from numpy.typing import NDArray


# Check if JAX is available
try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def create_test_mask(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a test mask with a circular ROI."""
    mask = np.zeros(shape, dtype=bool)
    center = (shape[0] // 2, shape[1] // 2)
    y, x = np.ogrid[: shape[0], : shape[1]]
    radius = min(shape) // 3
    dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    mask[dist_from_center <= radius] = True
    return mask


def create_test_qmap(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a test Q-map with radial values."""
    center = (shape[0] // 2, shape[1] // 2)
    y, x = np.ogrid[: shape[0], : shape[1]]
    qmap = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2) * 0.001
    return qmap.astype(np.float64)


def create_test_phi_map(shape: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Create a test phi map with angular values."""
    center = (shape[0] // 2, shape[1] // 2)
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    phi = np.rad2deg(np.arctan2(y - center[0], x - center[1]))
    return phi.astype(np.float64)


class TestPartitionNumericalEquivalence:
    """Test partition numerical equivalence between backends."""

    def test_linear_partition_numpy_baseline(self) -> None:
        """Test linear partition works with NumPy backend."""
        set_backend("numpy")
        mask = create_test_mask()
        qmap = create_test_qmap()

        result = generate_partition("q", mask, qmap, num_pts=10, style="linear")

        assert "partition" in result
        assert "v_list" in result
        assert result["partition"].shape == mask.shape
        assert len(result["v_list"]) == 10
        # Partition values should be 0 (masked) or 1-10 (bins)
        assert result["partition"].min() >= 0
        assert result["partition"].max() <= 10

    def test_logarithmic_partition_numpy_baseline(self) -> None:
        """Test logarithmic partition works with NumPy backend."""
        set_backend("numpy")
        mask = create_test_mask()
        qmap = create_test_qmap()
        # Ensure positive values for log scale
        qmap = np.clip(qmap, 0.001, None)

        result = generate_partition("q", mask, qmap, num_pts=10, style="logarithmic")

        assert "partition" in result
        assert "v_list" in result
        assert result["partition"].shape == mask.shape
        assert len(result["v_list"]) == 10

    def test_phi_partition_numpy_baseline(self) -> None:
        """Test phi partition works with NumPy backend."""
        set_backend("numpy")
        mask = create_test_mask()
        phi_map = create_test_phi_map()

        result = generate_partition("phi", mask, phi_map, num_pts=36, style="linear")

        assert "partition" in result
        assert "v_list" in result
        assert len(result["v_list"]) == 36

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_linear_partition_jax_equivalence(self) -> None:
        """Test that JAX produces identical linear partition to NumPy."""
        mask = create_test_mask()
        qmap = create_test_qmap()

        # Compute with NumPy
        set_backend("numpy")
        result_np = generate_partition("q", mask, qmap, num_pts=10, style="linear")

        # Compute with JAX
        set_backend("jax")
        result_jax = generate_partition("q", mask, qmap, num_pts=10, style="linear")

        # Reset to default
        set_backend("numpy")

        # Verify partition arrays are identical (integer values)
        np.testing.assert_array_equal(
            result_np["partition"],
            result_jax["partition"],
            err_msg="Linear partition arrays differ between NumPy and JAX",
        )

        # Verify bin centers are equivalent
        np.testing.assert_allclose(
            result_np["v_list"],
            result_jax["v_list"],
            rtol=1e-6,
            atol=1e-10,
            err_msg="Linear partition v_list differs between NumPy and JAX",
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_logarithmic_partition_jax_equivalence(self) -> None:
        """Test that JAX produces identical logarithmic partition to NumPy."""
        mask = create_test_mask()
        qmap = create_test_qmap()
        qmap = np.clip(qmap, 0.001, None)

        set_backend("numpy")
        result_np = generate_partition("q", mask, qmap, num_pts=10, style="logarithmic")

        set_backend("jax")
        result_jax = generate_partition(
            "q", mask, qmap, num_pts=10, style="logarithmic"
        )

        set_backend("numpy")

        np.testing.assert_array_equal(
            result_np["partition"],
            result_jax["partition"],
            err_msg="Logarithmic partition arrays differ between backends",
        )

        np.testing.assert_allclose(
            result_np["v_list"],
            result_jax["v_list"],
            rtol=1e-6,
            atol=1e-10,
            err_msg="Logarithmic partition v_list differs between backends",
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_phi_partition_jax_equivalence(self) -> None:
        """Test that JAX produces identical phi partition to NumPy."""
        mask = create_test_mask()
        phi_map = create_test_phi_map()

        set_backend("numpy")
        result_np = generate_partition("phi", mask, phi_map, num_pts=36, style="linear")

        set_backend("jax")
        result_jax = generate_partition(
            "phi", mask, phi_map, num_pts=36, style="linear"
        )

        set_backend("numpy")

        np.testing.assert_array_equal(
            result_np["partition"],
            result_jax["partition"],
            err_msg="Phi partition arrays differ between backends",
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_phi_partition_with_offset_jax_equivalence(self) -> None:
        """Test phi partition with offset produces identical results."""
        mask = create_test_mask()
        phi_map = create_test_phi_map()

        set_backend("numpy")
        result_np = generate_partition(
            "phi", mask, phi_map, num_pts=36, style="linear", phi_offset=45.0
        )

        set_backend("jax")
        result_jax = generate_partition(
            "phi", mask, phi_map, num_pts=36, style="linear", phi_offset=45.0
        )

        set_backend("numpy")

        np.testing.assert_array_equal(
            result_np["partition"],
            result_jax["partition"],
            err_msg="Phi partition with offset differs between backends",
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_phi_partition_with_symmetry_jax_equivalence(self) -> None:
        """Test phi partition with symmetry fold produces identical results."""
        mask = create_test_mask()
        phi_map = create_test_phi_map()

        set_backend("numpy")
        result_np = generate_partition(
            "phi", mask, phi_map, num_pts=18, style="linear", symmetry_fold=2
        )

        set_backend("jax")
        result_jax = generate_partition(
            "phi", mask, phi_map, num_pts=18, style="linear", symmetry_fold=2
        )

        set_backend("numpy")

        np.testing.assert_array_equal(
            result_np["partition"],
            result_jax["partition"],
            err_msg="Phi partition with symmetry fold differs between backends",
        )


class TestCombinePartitionsEquivalence:
    """Test combine_partitions numerical equivalence."""

    def test_combine_partitions_numpy_baseline(self) -> None:
        """Test partition combination works with NumPy backend."""
        set_backend("numpy")
        mask = create_test_mask()
        qmap = create_test_qmap()
        phi_map = create_test_phi_map()

        pack_q = generate_partition("q", mask, qmap, num_pts=10, style="linear")
        pack_phi = generate_partition("phi", mask, phi_map, num_pts=36, style="linear")

        combined = combine_partitions(pack_q, pack_phi, prefix="dynamic")

        assert "dynamic_roi_map" in combined
        assert "dynamic_num_pts" in combined
        assert "dynamic_v_list_dim0" in combined
        assert "dynamic_v_list_dim1" in combined
        assert combined["dynamic_roi_map"].shape == mask.shape
        assert combined["dynamic_num_pts"] == [10, 36]

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_combine_partitions_jax_equivalence(self) -> None:
        """Test that JAX produces identical combined partition to NumPy."""
        mask = create_test_mask()
        qmap = create_test_qmap()
        phi_map = create_test_phi_map()

        # Compute with NumPy
        set_backend("numpy")
        pack_q_np = generate_partition("q", mask, qmap, num_pts=10, style="linear")
        pack_phi_np = generate_partition(
            "phi", mask, phi_map, num_pts=36, style="linear"
        )
        combined_np = combine_partitions(pack_q_np, pack_phi_np, prefix="dynamic")

        # Compute with JAX
        set_backend("jax")
        pack_q_jax = generate_partition("q", mask, qmap, num_pts=10, style="linear")
        pack_phi_jax = generate_partition(
            "phi", mask, phi_map, num_pts=36, style="linear"
        )
        combined_jax = combine_partitions(pack_q_jax, pack_phi_jax, prefix="dynamic")

        set_backend("numpy")

        # Verify ROI map is identical
        np.testing.assert_array_equal(
            combined_np["dynamic_roi_map"],
            combined_jax["dynamic_roi_map"],
            err_msg="Combined ROI map differs between backends",
        )

        # Verify v_lists are equivalent
        np.testing.assert_allclose(
            combined_np["dynamic_v_list_dim0"],
            combined_jax["dynamic_v_list_dim0"],
            rtol=1e-6,
            atol=1e-10,
            err_msg="Combined v_list_dim0 differs between backends",
        )

        np.testing.assert_allclose(
            combined_np["dynamic_v_list_dim1"],
            combined_jax["dynamic_v_list_dim1"],
            rtol=1e-6,
            atol=1e-10,
            err_msg="Combined v_list_dim1 differs between backends",
        )

    def test_partition_output_is_numpy(self) -> None:
        """Verify partition output is always NumPy array at I/O boundary."""
        set_backend("numpy")
        mask = create_test_mask()
        qmap = create_test_qmap()

        result = generate_partition("q", mask, qmap, num_pts=10, style="linear")

        assert isinstance(result["partition"], np.ndarray), (
            f"Partition should be NumPy array, got {type(result['partition'])}"
        )
        assert isinstance(result["v_list"], np.ndarray), (
            f"v_list should be NumPy array, got {type(result['v_list'])}"
        )

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_partition_output_is_numpy_with_jax(self) -> None:
        """Verify partition output is NumPy even when using JAX backend."""
        set_backend("jax")
        mask = create_test_mask()
        qmap = create_test_qmap()

        result = generate_partition("q", mask, qmap, num_pts=10, style="linear")
        set_backend("numpy")

        assert isinstance(result["partition"], np.ndarray), (
            f"Partition should be NumPy array even with JAX backend"
        )
        assert isinstance(result["v_list"], np.ndarray), (
            f"v_list should be NumPy array even with JAX backend"
        )


class TestPartitionNumericStability:
    """Test partition numerical stability under edge cases."""

    def test_partition_with_sparse_mask(self) -> None:
        """Test partition with very sparse mask."""
        set_backend("numpy")
        shape = (256, 256)
        mask = np.zeros(shape, dtype=bool)
        # Only a few scattered pixels
        mask[100:110, 100:110] = True
        qmap = create_test_qmap(shape)

        result = generate_partition("q", mask, qmap, num_pts=5, style="linear")

        assert result["partition"].shape == shape
        assert result["partition"].max() <= 5

    def test_partition_with_extreme_values(self) -> None:
        """Test partition with extreme Q values."""
        set_backend("numpy")
        mask = create_test_mask()
        qmap = create_test_qmap() * 1000  # Large Q values

        result = generate_partition("q", mask, qmap, num_pts=10, style="linear")

        assert result["partition"].shape == mask.shape
        assert not np.any(np.isnan(result["v_list"]))

    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not installed")
    def test_partition_consistency_across_sizes(self) -> None:
        """Test partition consistency across different array sizes."""
        sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

        for shape in sizes:
            mask = create_test_mask(shape)
            qmap = create_test_qmap(shape)

            set_backend("numpy")
            result_np = generate_partition("q", mask, qmap, num_pts=10, style="linear")

            set_backend("jax")
            result_jax = generate_partition("q", mask, qmap, num_pts=10, style="linear")

            set_backend("numpy")

            np.testing.assert_array_equal(
                result_np["partition"],
                result_jax["partition"],
                err_msg=f"Partition differs at size {shape}",
            )
