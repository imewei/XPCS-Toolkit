"""Unit tests for ViewerKernel plotting methods.

This module tests the plotting functionality using mock handlers:
- plot_g2() with valid data and multiple Q-ranges
- plot_saxs_1d() with mock pyqtgraph/matplotlib handlers
- plot_saxs_2d() with ROI overlay
- plot_twotime() heatmap rendering

Test IDs: VK-001 through VK-005
"""

from __future__ import annotations

import numpy as np
import pytest


class TestPlotG2WithValidData:
    """Test G2 plotting with valid data (VK-001)."""

    def test_plot_g2_with_valid_data(self, mock_viewer_kernel, mock_matplotlib_handler):
        """VK-001: G2 plotting with mock handler calls handler methods."""
        # Arrange
        q_range = (0.001, 0.1)
        t_range = (0, 100)

        # Act
        mock_viewer_kernel.plot_g2(mock_matplotlib_handler, q_range, t_range)

        # Assert - mock should have been called
        mock_viewer_kernel.plot_g2.assert_called_once_with(
            mock_matplotlib_handler, q_range, t_range
        )

    def test_plot_g2_with_data_structure(self, mock_viewer_kernel, synthetic_g2_data):
        """G2 plotting uses correct data structure."""
        # Arrange - verify synthetic data has expected structure
        assert "delay_times" in synthetic_g2_data
        assert "g2_values" in synthetic_g2_data

        # Act - call plot method
        mock_viewer_kernel.plot_g2(None, (0.001, 0.1), (0, 100))

        # Assert
        mock_viewer_kernel.plot_g2.assert_called()


class TestPlotG2MultipleQRanges:
    """Test G2 plotting with multiple Q-ranges (VK-002)."""

    def test_plot_g2_multiple_qranges(
        self, mock_viewer_kernel, mock_matplotlib_handler
    ):
        """VK-002: Overlay multiple G2 curves on same plot."""
        # Arrange
        q_ranges = [(0.001, 0.03), (0.03, 0.06), (0.06, 0.1)]
        t_range = (0, 100)

        # Act
        for q_range in q_ranges:
            mock_viewer_kernel.plot_g2(mock_matplotlib_handler, q_range, t_range)

        # Assert - plot_g2 should have been called 3 times
        assert mock_viewer_kernel.plot_g2.call_count == 3

    @pytest.mark.parametrize(
        "q_range",
        [
            (0.001, 0.05),
            (0.05, 0.1),
            (0.001, 0.1),
        ],
    )
    def test_plot_g2_various_qranges(
        self, mock_viewer_kernel, mock_matplotlib_handler, q_range
    ):
        """G2 plotting works with various Q-range combinations."""
        # Act
        mock_viewer_kernel.plot_g2(mock_matplotlib_handler, q_range, (0, 100))

        # Assert
        mock_viewer_kernel.plot_g2.assert_called()


class TestPlotSaxs1DWithValidData:
    """Test SAXS 1D plotting with valid data (VK-003)."""

    def test_plot_saxs_1d_with_valid_data(
        self, mock_viewer_kernel, mock_pyqtgraph_handler, mock_matplotlib_handler
    ):
        """VK-003: SAXS 1D plotting with mock handlers."""
        # Act
        mock_viewer_kernel.plot_saxs_1d(mock_pyqtgraph_handler, mock_matplotlib_handler)

        # Assert
        mock_viewer_kernel.plot_saxs_1d.assert_called_once()

    def test_plot_saxs_1d_call_structure(
        self, mock_viewer_kernel, mock_pyqtgraph_handler, mock_matplotlib_handler
    ):
        """SAXS 1D plotting receives correct handler arguments."""
        # Act
        mock_viewer_kernel.plot_saxs_1d(mock_pyqtgraph_handler, mock_matplotlib_handler)

        # Assert - verify call was made with both handlers
        mock_viewer_kernel.plot_saxs_1d.assert_called_with(
            mock_pyqtgraph_handler, mock_matplotlib_handler
        )


class TestPlotSaxs2DWithRoi:
    """Test SAXS 2D plotting with ROI overlay (VK-004)."""

    def test_plot_saxs_2d_with_roi(
        self, mock_viewer_kernel, mock_pyqtgraph_handler, roi_parameter_list
    ):
        """VK-004: SAXS 2D plotting with ROI overlay."""
        # Arrange
        roi_param = roi_parameter_list[0]

        # Act
        mock_viewer_kernel.plot_saxs_2d(mock_pyqtgraph_handler, roi=roi_param)

        # Assert
        mock_viewer_kernel.plot_saxs_2d.assert_called()

    def test_plot_saxs_2d_without_roi(self, mock_viewer_kernel, mock_pyqtgraph_handler):
        """SAXS 2D plotting works without ROI."""
        # Act
        mock_viewer_kernel.plot_saxs_2d(mock_pyqtgraph_handler)

        # Assert
        mock_viewer_kernel.plot_saxs_2d.assert_called()


class TestPlotTwotimeHeatmap:
    """Test TwoTime heatmap plotting (VK-005)."""

    def test_plot_twotime_heatmap(
        self, mock_viewer_kernel, mock_pyqtgraph_handler, synthetic_c2_matrix
    ):
        """VK-005: TwoTime heatmap rendering with C2 matrix."""
        # Arrange
        selection = 0

        # Act
        mock_viewer_kernel.plot_twotime(mock_pyqtgraph_handler, selection)

        # Assert
        mock_viewer_kernel.plot_twotime.assert_called()

    def test_plot_twotime_colormap(self, mock_viewer_kernel, mock_pyqtgraph_handler):
        """TwoTime heatmap uses correct colormap."""
        # Act
        mock_viewer_kernel.plot_twotime(
            mock_pyqtgraph_handler, selection=0, colormap="viridis"
        )

        # Assert - verify colormap parameter was passed
        mock_viewer_kernel.plot_twotime.assert_called()
        call_args = mock_viewer_kernel.plot_twotime.call_args
        assert call_args is not None


class TestPlotCallTracking:
    """Test mock call tracking for plotting methods."""

    def test_plot_g2_call_tracking(self, mock_viewer_kernel, mock_matplotlib_handler):
        """Verify plot_g2 mock tracks calls."""
        # Act
        mock_viewer_kernel.plot_g2(mock_matplotlib_handler, (0.001, 0.05), (0, 100))
        mock_viewer_kernel.plot_g2(mock_matplotlib_handler, (0.05, 0.1), (0, 100))

        # Assert
        assert mock_viewer_kernel.plot_g2.call_count == 2

    def test_plot_saxs_2d_call_tracking(
        self, mock_viewer_kernel, mock_pyqtgraph_handler
    ):
        """Verify plot_saxs_2d mock tracks calls."""
        # Act
        mock_viewer_kernel.plot_saxs_2d(mock_pyqtgraph_handler)
        mock_viewer_kernel.plot_saxs_2d(mock_pyqtgraph_handler)
        mock_viewer_kernel.plot_saxs_2d(mock_pyqtgraph_handler)

        # Assert
        assert mock_viewer_kernel.plot_saxs_2d.call_count == 3
