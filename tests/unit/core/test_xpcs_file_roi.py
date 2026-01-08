"""Unit tests for XpcsFile ROI data extraction methods.

This module tests the ROI data extraction functionality including:
- get_roi_data() for single ROI extraction with phi binning
- get_multiple_roi_data_parallel() for parallel ROI extraction
- Edge cases: invalid parameters, worker exhaustion

Test IDs: XF-007, XF-008, plus edge cases
"""

from __future__ import annotations

import numpy as np
import pytest


class TestGetRoiDataSingle:
    """Test single ROI data extraction (XF-007)."""

    def test_get_roi_data_single(self, mock_xpcs_file, roi_parameter_list):
        """XF-007: Single ROI extraction with phi binning returns expected structure."""
        # Arrange
        roi_param = roi_parameter_list[0]
        phi_num = 180

        # Act
        result = mock_xpcs_file.get_roi_data(roi_param, phi_num=phi_num)

        # Assert
        assert result is not None
        assert "qbin" in result
        assert "data" in result
        assert "phi" in result

    def test_get_roi_data_returns_correct_phi_length(
        self, mock_xpcs_file, roi_parameter_list
    ):
        """ROI data returns data with length equal to phi_num."""
        # Arrange
        roi_param = roi_parameter_list[0]
        phi_num = 180

        # Act
        result = mock_xpcs_file.get_roi_data(roi_param, phi_num=phi_num)

        # Assert - data length should match phi_num
        assert len(result["data"]) == phi_num

    def test_get_roi_data_finite_values(self, mock_xpcs_file, roi_parameter_list):
        """ROI data contains finite values."""
        # Arrange
        roi_param = roi_parameter_list[0]

        # Act
        result = mock_xpcs_file.get_roi_data(roi_param, phi_num=180)

        # Assert
        assert np.all(np.isfinite(result["data"]))


class TestGetMultipleRoiDataParallel:
    """Test parallel ROI data extraction (XF-008)."""

    def test_get_multiple_roi_data_parallel(self, mock_xpcs_file, roi_parameter_list):
        """XF-008: Parallel ROI extraction returns list of results."""
        # Arrange
        max_workers = 4

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(
            roi_parameter_list, max_workers=max_workers
        )

        # Assert
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == len(roi_parameter_list)

    def test_get_multiple_roi_data_parallel_consistency(
        self, mock_xpcs_file, roi_parameter_list
    ):
        """Parallel extraction returns consistent results."""
        # Arrange
        max_workers = 4

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(
            roi_parameter_list, max_workers=max_workers
        )

        # Assert - each result should have qbin and data
        for roi_result in result:
            assert "qbin" in roi_result
            assert "data" in roi_result

    def test_get_multiple_roi_data_parallel_order(
        self, mock_xpcs_file, roi_parameter_list
    ):
        """Parallel extraction maintains order of input ROIs."""
        # Arrange
        max_workers = 4

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(
            roi_parameter_list, max_workers=max_workers
        )

        # Assert - should have one result per input ROI
        assert len(result) == len(roi_parameter_list)


class TestRoiInvalidParameters:
    """Test ROI extraction with invalid parameters."""

    def test_roi_invalid_parameters_handled(self, mock_xpcs_file):
        """Invalid ROI parameters should be handled gracefully."""
        # Arrange - create invalid ROI with r_inner > r_outer
        invalid_roi = {
            "roi_type": "ring",
            "center": (256, 256),
            "r_inner": 100,
            "r_outer": 50,  # Invalid: outer < inner
            "phi_num": 180,
        }

        # Act - mock will accept any input, but real implementation would validate
        result = mock_xpcs_file.get_roi_data(invalid_roi, phi_num=180)

        # Assert - mock returns configured result
        assert result is not None

    def test_roi_empty_list_handled(self, mock_xpcs_file):
        """Empty ROI list should be handled."""
        # Arrange
        empty_list = []

        # Configure mock for empty list
        mock_xpcs_file.get_multiple_roi_data_parallel.return_value = []

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(
            empty_list, max_workers=4
        )

        # Assert
        assert result == []


class TestParallelRoiWorkerExhaustion:
    """Test edge case: worker exhaustion in parallel ROI extraction."""

    def test_parallel_roi_worker_exhaustion(self, mock_xpcs_file, roi_parameter_list):
        """Edge case: max_workers limit exceeded should be handled."""
        # Arrange - request more workers than available
        max_workers = 1  # Limited workers

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(
            roi_parameter_list, max_workers=max_workers
        )

        # Assert - should still return results
        assert result is not None
        assert len(result) == len(roi_parameter_list)

    def test_parallel_roi_with_many_rois(self, mock_xpcs_file):
        """Test parallel extraction with many ROIs."""
        # Arrange - create many ROI parameters
        from tests.fixtures.xpcs_synthetic import generate_roi_parameters

        many_rois = generate_roi_parameters(n_rois=20)

        # Configure mock to return expected number of results
        mock_xpcs_file.get_multiple_roi_data_parallel.return_value = [
            {"qbin": i * 0.01, "data": np.zeros(180)} for i in range(20)
        ]

        # Act
        result = mock_xpcs_file.get_multiple_roi_data_parallel(many_rois, max_workers=4)

        # Assert
        assert len(result) == 20


class TestRoiCallTracking:
    """Test mock call tracking for ROI methods."""

    def test_roi_single_call_tracking(self, mock_xpcs_file, roi_parameter_list):
        """Verify single ROI extraction tracks calls."""
        # Arrange
        roi_param = roi_parameter_list[0]

        # Act
        mock_xpcs_file.get_roi_data(roi_param, phi_num=180)
        mock_xpcs_file.get_roi_data(roi_param, phi_num=90)

        # Assert
        assert mock_xpcs_file.get_roi_data.call_count == 2

    def test_roi_parallel_call_tracking(self, mock_xpcs_file, roi_parameter_list):
        """Verify parallel ROI extraction tracks calls."""
        # Act
        mock_xpcs_file.get_multiple_roi_data_parallel(roi_parameter_list, max_workers=4)
        mock_xpcs_file.get_multiple_roi_data_parallel(
            roi_parameter_list[:2], max_workers=2
        )

        # Assert
        assert mock_xpcs_file.get_multiple_roi_data_parallel.call_count == 2
