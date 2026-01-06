"""Unit tests for area_mask module.

Tests mask classes and MaskAssemble functionality.
"""

import numpy as np
import pytest

from xpcsviewer.simplemask.area_mask import (
    MaskArray,
    MaskAssemble,
    MaskBase,
    MaskList,
    MaskParameter,
    MaskThreshold,
    create_qring,
)


class TestMaskBase:
    """Tests for MaskBase class."""

    def test_initialization(self):
        """MaskBase should initialize with correct shape."""
        mask = MaskBase(shape=(512, 1024))
        assert mask.shape == (512, 1024)
        assert mask.zero_loc is None
        assert mask.mtype == "base"

    def test_get_mask_no_zero_loc(self):
        """get_mask should return all ones when no pixels masked."""
        mask = MaskBase(shape=(100, 100))
        result = mask.get_mask()

        assert result.shape == (100, 100)
        assert result.dtype == bool
        assert np.all(result)

    def test_describe_not_initialized(self):
        """describe should indicate when mask is not initialized."""
        mask = MaskBase()
        assert "not initialized" in mask.describe()


class TestMaskList:
    """Tests for MaskList class."""

    def test_evaluate_sets_zero_loc(self):
        """evaluate should set zero_loc from coordinate array."""
        mask = MaskList(shape=(100, 100))
        coords = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        mask.evaluate(zero_loc=coords)

        np.testing.assert_array_equal(mask.zero_loc, coords)

    def test_get_mask_with_zero_loc(self):
        """get_mask should mask specified pixels."""
        mask = MaskList(shape=(10, 10))
        mask.evaluate(zero_loc=np.array([[1, 2], [3, 4]], dtype=np.int64))

        result = mask.get_mask()
        assert result[1, 3] == 0
        assert result[2, 4] == 0
        assert result[0, 0] == 1

    def test_append_zero_pt(self):
        """append_zero_pt should add single pixel to mask."""
        mask = MaskList(shape=(10, 10))
        mask.zero_loc = np.array([[1], [2]], dtype=np.int64)
        mask.append_zero_pt(5, 6)

        assert mask.zero_loc.shape == (2, 2)
        assert 5 in mask.zero_loc[0]
        assert 6 in mask.zero_loc[1]


class TestMaskThreshold:
    """Tests for MaskThreshold class."""

    def test_low_threshold(self):
        """Pixels below low threshold should be masked."""
        mask = MaskThreshold(shape=(10, 10))
        data = np.ones((10, 10)) * 100
        data[3, 3] = 5  # Below threshold

        mask.evaluate(
            saxs_lin=data, low=10, high=1000, low_enable=True, high_enable=True
        )

        result = mask.get_mask()
        assert result[3, 3] == 0
        assert result[5, 5] == 1

    def test_high_threshold(self):
        """Pixels above high threshold should be masked."""
        mask = MaskThreshold(shape=(10, 10))
        data = np.ones((10, 10)) * 100
        data[3, 3] = 10000  # Above threshold

        mask.evaluate(
            saxs_lin=data, low=10, high=1000, low_enable=True, high_enable=True
        )

        result = mask.get_mask()
        assert result[3, 3] == 0

    def test_disabled_thresholds(self):
        """Disabled thresholds should not affect mask."""
        mask = MaskThreshold(shape=(10, 10))
        data = np.ones((10, 10)) * 100
        data[3, 3] = 5

        mask.evaluate(saxs_lin=data, low=10, low_enable=False, high_enable=False)

        result = mask.get_mask()
        assert np.all(result)


class TestMaskParameter:
    """Tests for MaskParameter class."""

    def test_single_constraint(self):
        """Single AND constraint should mask outside range."""
        mask = MaskParameter(shape=(10, 10))
        qmap = {"q": np.arange(100).reshape(10, 10).astype(float)}
        constraints = [("q", "AND", "Å⁻¹", 20.0, 80.0)]

        mask.evaluate(qmap=qmap, constraints=constraints)

        result = mask.get_mask()
        # Pixels with q < 20 or q > 80 should be masked
        assert result[0, 0] == 0  # q=0
        assert result[2, 5] == 1  # q=25
        assert result[9, 9] == 0  # q=99

    def test_or_constraint(self):
        """OR constraint should include additional regions."""
        mask = MaskParameter(shape=(10, 10))
        qmap = {"q": np.arange(100).reshape(10, 10).astype(float)}
        constraints = [
            ("q", "AND", "Å⁻¹", 10.0, 20.0),
            ("q", "OR", "Å⁻¹", 80.0, 90.0),
        ]

        mask.evaluate(qmap=qmap, constraints=constraints)

        result = mask.get_mask()
        # Should include 10-20 OR 80-90
        assert result[1, 5] == 1  # q=15
        assert result[8, 5] == 1  # q=85
        assert result[5, 0] == 0  # q=50


class TestMaskArray:
    """Tests for MaskArray class."""

    def test_evaluate_from_array(self):
        """evaluate should create mask from boolean array."""
        mask = MaskArray(shape=(10, 10))
        arr = np.zeros((10, 10), dtype=bool)
        arr[3, 3] = True
        arr[5, 5] = True

        mask.evaluate(arr=arr)

        result = mask.get_mask()
        assert result[3, 3] == 0
        assert result[5, 5] == 0
        assert result[0, 0] == 1


class TestMaskAssemble:
    """Tests for MaskAssemble class."""

    def test_initialization(self):
        """MaskAssemble should initialize with workers."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)

        assert "mask_threshold" in ma.workers
        assert "mask_draw" in ma.workers
        assert "mask_file" in ma.workers
        assert ma.mask_ptr == 0

    def test_apply_adds_to_history(self):
        """apply should add new mask state to history when mask changes."""
        # Create data with some values outside threshold
        saxs = np.ones((10, 10))
        saxs[3, 3] = 5  # Outside threshold low=0.5, high=2

        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)

        # Set up a threshold mask that will mask pixel (3,3)
        ma.evaluate(
            "mask_threshold", low=0.5, high=2, low_enable=True, high_enable=True
        )
        initial_len = len(ma.mask_record)

        ma.apply("mask_threshold")

        # History should grow since mask changed
        assert len(ma.mask_record) == initial_len + 1
        assert ma.mask_ptr == initial_len

    def test_undo_decrements_pointer(self):
        """undo should move to previous mask state."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)

        # Create some history
        ma.mask_record.append(np.ones((10, 10), dtype=bool))
        ma.mask_ptr = 1

        ma.redo_undo("undo")

        assert ma.mask_ptr == 0

    def test_undo_respects_min(self):
        """undo should not go below mask_ptr_min."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)
        ma.mask_ptr_min = 0
        ma.mask_ptr = 0

        ma.redo_undo("undo")

        assert ma.mask_ptr == 0

    def test_redo_increments_pointer(self):
        """redo should move to next mask state."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)
        ma.mask_record.append(np.ones((10, 10), dtype=bool))
        ma.mask_ptr = 0

        ma.redo_undo("redo")

        assert ma.mask_ptr == 1

    def test_reset_clears_history(self):
        """reset should clear history back to initial state."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)

        # Add some history
        ma.mask_record.append(np.ones((10, 10), dtype=bool))
        ma.mask_record.append(np.ones((10, 10), dtype=bool))
        ma.mask_ptr = 2

        ma.redo_undo("reset")

        assert len(ma.mask_record) == 1
        assert ma.mask_ptr == 0

    def test_get_mask_returns_current(self):
        """get_mask should return mask at current pointer."""
        saxs = np.ones((10, 10))
        ma = MaskAssemble(shape=(10, 10), saxs_lin=saxs)

        mask1 = np.ones((10, 10), dtype=bool)
        mask2 = np.zeros((10, 10), dtype=bool)
        ma.mask_record = [mask1, mask2]
        ma.mask_ptr = 1

        result = ma.get_mask()
        np.testing.assert_array_equal(result, mask2)

    def test_update_qmap(self):
        """update_qmap should update qmap reference."""
        ma = MaskAssemble(shape=(10, 10))
        qmap = {"q": np.ones((10, 10))}

        ma.update_qmap(qmap)

        assert ma.qmap is qmap


class TestCreateQring:
    """Tests for create_qring function."""

    def test_single_ring(self):
        """Single ring should have correct bounds."""
        rings = create_qring(0.01, 0.02, -180, 180, qnum=1)

        assert len(rings) == 1
        np.testing.assert_almost_equal(rings[0][0], 0.01)
        np.testing.assert_almost_equal(rings[0][1], 0.02)
        assert rings[0][2] == -180
        assert rings[0][3] == 180

    def test_multiple_rings_const_width(self):
        """Multiple rings with constant width should scale center."""
        rings = create_qring(0.01, 0.02, -180, 180, qnum=3, flag_const_width=True)

        assert len(rings) == 3
        # Width should be constant
        width1 = rings[0][1] - rings[0][0]
        width2 = rings[1][1] - rings[1][0]
        np.testing.assert_almost_equal(width1, width2)

    def test_qmin_qmax_swap(self):
        """Function should handle qmin > qmax."""
        rings = create_qring(0.02, 0.01, -180, 180, qnum=1)

        np.testing.assert_almost_equal(rings[0][0], 0.01)  # Should be swapped
        np.testing.assert_almost_equal(rings[0][1], 0.02)
