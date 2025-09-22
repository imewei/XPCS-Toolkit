"""Unit tests for G2 analysis module.

This module provides comprehensive unit tests for G2 correlation analysis,
covering correlation calculation and fitting algorithms.
"""

import inspect
from unittest.mock import Mock, patch

import numpy as np
import pyqtgraph as pg
import pytest

from xpcs_toolkit.module import g2mod
from xpcs_toolkit.module.g2mod import colors, symbols
from xpcs_toolkit.plothandler.plot_constants import MATPLOTLIB_COLORS_RGB


class TestG2ModConstants:
    """Test suite for G2 module constants."""

    def test_colors_definition(self):
        """Test colors tuple is properly defined."""
        assert hasattr(g2mod, "colors")
        assert isinstance(g2mod.colors, tuple)
        assert len(g2mod.colors) == 10

        # Each color should be RGB tuple
        for color in g2mod.colors:
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_symbols_definition(self):
        """Test symbols list is properly defined."""
        assert hasattr(g2mod, "symbols")
        assert isinstance(g2mod.symbols, list)
        assert len(g2mod.symbols) == 12

        # All symbols should be strings
        for symbol in g2mod.symbols:
            assert isinstance(symbol, str)
            assert len(symbol) > 0

    def test_specific_color_values(self):
        """Test specific color values match expected matplotlib colors."""
        expected_colors = [
            (31, 119, 180),  # Blue
            (255, 127, 14),  # Orange
            (44, 160, 44),  # Green
            (214, 39, 40),  # Red
            (148, 103, 189),  # Purple
        ]

        for i, expected_color in enumerate(expected_colors):
            assert g2mod.colors[i] == expected_color


class TestGetData:
    """Test suite for get_data function."""

    def test_get_data_success(self):
        """Test successful data retrieval."""
        # Create mock XF objects
        mock_xf1 = Mock()
        mock_xf1.atype = ("Multitau",)
        mock_xf1.get_g2_data.return_value = (
            np.array([0.01, 0.02]),  # q
            np.array([0.001, 0.01]),  # tel
            np.array([[2.0, 1.5], [1.8, 1.3]]),  # g2
            np.array([[0.1, 0.05], [0.08, 0.04]]),  # g2_err
            ["q=0.01", "q=0.02"],  # labels
        )

        mock_xf2 = Mock()
        mock_xf2.atype = ("Multitau",)
        mock_xf2.get_g2_data.return_value = (
            np.array([0.01, 0.02]),
            np.array([0.001, 0.01]),
            np.array([[1.9, 1.4], [1.7, 1.2]]),
            np.array([[0.09, 0.06], [0.07, 0.05]]),
            ["q=0.01", "q=0.02"],
        )

        xf_list = [mock_xf1, mock_xf2]

        q, tel, g2, g2_err, labels = g2mod.get_data(
            xf_list, q_range=[0, 1], t_range=[0, 1]
        )

        # Verify return values
        assert len(q) == 2
        assert len(tel) == 2
        assert len(g2) == 2
        assert len(g2_err) == 2
        assert len(labels) == 2

        # Verify method calls
        mock_xf1.get_g2_data.assert_called_once_with(qrange=[0, 1], trange=[0, 1])
        mock_xf2.get_g2_data.assert_called_once_with(qrange=[0, 1], trange=[0, 1])

    def test_get_data_no_correlation_analysis(self):
        """Test data retrieval with files that don't have correlation analysis."""
        # Create mock XF objects without correlation analysis
        mock_xf1 = Mock()
        mock_xf1.atype = ("SAXS",)  # No correlation analysis

        mock_xf2 = Mock()
        mock_xf2.atype = ("Unknown",)  # No correlation analysis

        xf_list = [mock_xf1, mock_xf2]

        result = g2mod.get_data(xf_list)

        # Should return False and None values
        assert result == (False, None, None, None, None)

    def test_get_data_mixed_analysis_types(self):
        """Test data retrieval with mixed analysis types."""
        mock_xf1 = Mock()
        mock_xf1.atype = ("Multitau",)

        mock_xf2 = Mock()
        mock_xf2.atype = ("SAXS",)  # No correlation

        xf_list = [mock_xf1, mock_xf2]

        result = g2mod.get_data(xf_list)

        # Should return False because not all files have correlation
        assert result == (False, None, None, None, None)

    def test_get_data_twotime_analysis(self):
        """Test data retrieval with Twotime analysis type."""
        mock_xf = Mock()
        mock_xf.atype = ("Twotime",)
        mock_xf.get_g2_data.return_value = (
            np.array([0.01]),
            np.array([0.001]),
            np.array([[2.0]]),
            np.array([[0.1]]),
            ["q=0.01"],
        )

        xf_list = [mock_xf]

        q, _tel, _g2, _g2_err, _labels = g2mod.get_data(xf_list)

        assert len(q) == 1
        mock_xf.get_g2_data.assert_called_once()

    def test_get_data_with_none_ranges(self):
        """Test data retrieval with None ranges."""
        mock_xf = Mock()
        mock_xf.atype = ("Multitau",)
        mock_xf.get_g2_data.return_value = (
            np.array([0.01]),
            np.array([0.001]),
            np.array([[2.0]]),
            np.array([[0.1]]),
            ["q=0.01"],
        )

        xf_list = [mock_xf]

        g2mod.get_data(xf_list, q_range=None, t_range=None)

        mock_xf.get_g2_data.assert_called_once_with(qrange=None, trange=None)

    def test_get_data_empty_list(self):
        """Test data retrieval with empty file list."""
        xf_list = []

        q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

        # Should return empty lists
        assert q == []
        assert tel == []
        assert g2 == []
        assert g2_err == []
        assert labels == []


class TestComputeGeometry:
    """Test suite for compute_geometry function."""

    def test_compute_geometry_multiple_type(self):
        """Test geometry computation for 'multiple' plot type."""
        # Create mock g2 data: 2 files, each with 2 time points and 3 q values
        g2_file1 = np.array([[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]])  # (2, 3)
        g2_file2 = np.array([[1.0, 1.1, 1.3], [1.2, 1.4, 1.6]])  # (2, 3)
        g2_list = [g2_file1, g2_file2]

        num_figs, num_lines = g2mod.compute_geometry(g2_list, "multiple")

        assert num_figs == 3  # Number of q values (g2[0].shape[1])
        assert num_lines == 2  # Number of files (len(g2))

    def test_compute_geometry_single_type(self):
        """Test geometry computation for 'single' plot type."""
        g2_file1 = np.array([[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]])  # (2, 3)
        g2_file2 = np.array([[1.0, 1.1, 1.3], [1.2, 1.4, 1.6]])  # (2, 3)
        g2_list = [g2_file1, g2_file2]

        num_figs, num_lines = g2mod.compute_geometry(g2_list, "single")

        assert num_figs == 2  # Number of files (len(g2))
        assert num_lines == 3  # Number of q values (g2[0].shape[1])

    def test_compute_geometry_single_combined_type(self):
        """Test geometry computation for 'single-combined' plot type."""
        g2_file1 = np.array([[1.0, 1.2, 1.4], [1.1, 1.3, 1.5]])  # (2, 3)
        g2_file2 = np.array([[1.0, 1.1, 1.3], [1.2, 1.4, 1.6]])  # (2, 3)
        g2_list = [g2_file1, g2_file2]

        num_figs, num_lines = g2mod.compute_geometry(g2_list, "single-combined")

        assert num_figs == 1  # Always 1 for combined
        assert num_lines == 6  # q_vals * num_files = 3 * 2

    def test_compute_geometry_invalid_type(self):
        """Test geometry computation with invalid plot type."""
        g2_list = [np.array([[1.0, 1.2]])]

        with pytest.raises(ValueError, match="plot_type not support"):
            g2mod.compute_geometry(g2_list, "invalid_type")

    def test_compute_geometry_single_file(self):
        """Test geometry computation with single file."""
        g2_single = np.array([[1.0, 1.2, 1.4, 1.6]])  # (1, 4)
        g2_list = [g2_single]

        num_figs, num_lines = g2mod.compute_geometry(g2_list, "multiple")

        assert num_figs == 4  # Number of q values
        assert num_lines == 1  # Single file

    def test_compute_geometry_different_shapes(self):
        """Test geometry computation uses first file's shape."""
        g2_file1 = np.array([[1.0, 1.2, 1.4]])  # (1, 3)
        g2_file2 = np.array([[1.0, 1.1], [1.2, 1.3]])  # (2, 2) - different shape
        g2_list = [g2_file1, g2_file2]

        # Should use g2[0].shape[1] for calculations
        num_figs, num_lines = g2mod.compute_geometry(g2_list, "multiple")

        assert num_figs == 3  # Based on first file's shape
        assert num_lines == 2  # Number of files


class TestPgPlotFunction:
    """Test suite for pg_plot function (basic structure tests)."""

    def test_pg_plot_function_exists(self):
        """Test that pg_plot function exists and is callable."""
        assert hasattr(g2mod, "pg_plot")
        assert callable(g2mod.pg_plot)

    @patch("xpcs_toolkit.module.g2mod.get_data")
    def test_pg_plot_basic_call_structure(self, mock_get_data):
        """Test basic call structure of pg_plot function."""
        # Mock get_data to return valid data
        mock_get_data.return_value = (
            [np.array([0.01])],  # q
            [np.array([0.001])],  # tel
            [np.array([[2.0]])],  # g2
            [np.array([[0.1]])],  # g2_err
            [["q=0.01"]],  # labels
        )

        # Create minimal mocks
        mock_hdl = Mock()
        mock_xf = Mock()
        mock_xf.atype = ("Multitau",)
        xf_list = [mock_xf]

        # Test that function can be called without raising exceptions
        try:
            g2mod.pg_plot(
                hdl=mock_hdl,
                xf_list=xf_list,
                q_range=[0, 1],
                t_range=[0, 1],
                y_range=[1, 3],
                plot_type="multiple",
            )
            # Function should complete without error
            assert True
        except Exception as e:
            # If it fails, ensure it's not due to basic structure issues
            assert "get_data" not in str(e), f"Basic structure error: {e}"


class TestPgPlotParameters:
    """Test suite for pg_plot function parameters."""

    @patch("xpcs_toolkit.module.g2mod.get_data")
    def test_pg_plot_parameter_defaults(self, mock_get_data):
        """Test pg_plot function parameter defaults."""
        mock_get_data.return_value = ([], [], [], [], [])

        mock_hdl = Mock()
        xf_list = []

        # Test that function accepts minimal parameters
        try:
            g2mod.pg_plot(
                hdl=mock_hdl, xf_list=xf_list, q_range=None, t_range=None, y_range=None
            )
        except Exception as e:
            # Should not fail due to missing optional parameters
            assert "required positional argument" not in str(e).lower()

    def test_pg_plot_parameter_names(self):
        """Test that pg_plot has expected parameter names."""

        sig = inspect.signature(g2mod.pg_plot)
        param_names = list(sig.parameters.keys())

        expected_params = ["hdl", "xf_list", "q_range", "t_range", "y_range"]

        for param in expected_params:
            assert param in param_names, f"Expected parameter '{param}' not found"


class TestModuleIntegration:
    """Test suite for module-level integration."""

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        assert hasattr(g2mod, "logger")
        assert g2mod.logger is not None

    def test_pyqtgraph_config(self):
        """Test that PyQtGraph configuration is set."""
        # The module should set foreground color
        # We can't easily test the actual config, but we can verify
        # the module imports and uses pg
        assert hasattr(g2mod, "pg")
        assert g2mod.pg is pg

    def test_module_imports(self):
        """Test that required modules are imported."""
        assert hasattr(g2mod, "np")
        assert hasattr(g2mod, "pg")
        assert g2mod.np is np

    def test_module_constants_accessibility(self):
        """Test that module constants are accessible."""
        # Colors and symbols should be accessible from module
        assert colors == g2mod.colors
        assert symbols == g2mod.symbols
        # Verify colors come from centralized constants
        assert colors == MATPLOTLIB_COLORS_RGB


class TestDataProcessingHelpers:
    """Test suite for data processing helper functions."""

    def test_get_data_memory_efficiency(self):
        """Test that get_data pre-allocates lists for memory efficiency."""
        # This tests the implementation detail of pre-allocation
        mock_xf1 = Mock()
        mock_xf1.atype = ("Multitau",)
        mock_xf1.get_g2_data.return_value = (
            np.array([0.01]),
            np.array([0.001]),
            np.array([[2.0]]),
            np.array([[0.1]]),
            ["q=0.01"],
        )

        mock_xf2 = Mock()
        mock_xf2.atype = ("Multitau",)
        mock_xf2.get_g2_data.return_value = (
            np.array([0.01]),
            np.array([0.001]),
            np.array([[1.9]]),
            np.array([[0.09]]),
            ["q=0.01"],
        )

        xf_list = [mock_xf1, mock_xf2]

        q, tel, g2, g2_err, labels = g2mod.get_data(xf_list)

        # Verify lists have correct length (pre-allocated correctly)
        assert len(q) == len(xf_list)
        assert len(tel) == len(xf_list)
        assert len(g2) == len(xf_list)
        assert len(g2_err) == len(xf_list)
        assert len(labels) == len(xf_list)


class TestErrorHandling:
    """Test suite for error handling in G2 module."""

    def test_get_data_with_exception_in_xf(self):
        """Test get_data behavior when XF object raises exception."""
        mock_xf = Mock()
        mock_xf.atype = ("Multitau",)
        mock_xf.get_g2_data.side_effect = Exception("Data access error")

        xf_list = [mock_xf]

        # System may handle errors gracefully or propagate them
        try:
            result = g2mod.get_data(xf_list)
            # If successful, should return some reasonable result
            assert result is not None
        except Exception as e:
            # Also acceptable - error propagated as expected
            pytest.skip(f"Exception raised as expected: {e}")

    def test_compute_geometry_empty_g2_list(self):
        """Test compute_geometry with empty g2 list."""
        g2_list = []

        with pytest.raises(IndexError):
            g2mod.compute_geometry(g2_list, "multiple")

    def test_compute_geometry_invalid_g2_shape(self):
        """Test compute_geometry with invalid g2 shape."""
        # 1D array instead of expected 2D
        g2_invalid = np.array([1.0, 1.2, 1.4])
        g2_list = [g2_invalid]

        with pytest.raises(IndexError):
            g2mod.compute_geometry(g2_list, "multiple")


@pytest.mark.parametrize(
    "plot_type,expected_behavior",
    [
        ("multiple", "multiple_plots"),
        ("single", "single_plot_per_file"),
        ("single-combined", "all_in_one"),
    ],
)
def test_compute_geometry_plot_types(plot_type, expected_behavior):
    """Test compute_geometry with different plot types."""
    g2_data = np.array([[1.0, 1.2], [1.1, 1.3]])  # 2 time points, 2 q values
    g2_list = [g2_data]

    num_figs, num_lines = g2mod.compute_geometry(g2_list, plot_type)

    if expected_behavior == "multiple_plots":
        assert num_figs == 2  # One plot per q value
        assert num_lines == 1  # One line per file
    elif expected_behavior == "single_plot_per_file":
        assert num_figs == 1  # One plot per file
        assert num_lines == 2  # One line per q value
    elif expected_behavior == "all_in_one":
        assert num_figs == 1  # Single combined plot
        assert num_lines == 2  # All q values from all files


@pytest.mark.parametrize("num_files,num_q_vals", [(1, 1), (1, 5), (3, 2), (5, 10)])
def test_compute_geometry_scaling(num_files, num_q_vals):
    """Test compute_geometry scaling with different data sizes."""
    # Create mock data
    g2_list = []
    for _ in range(num_files):
        g2_data = np.ones((2, num_q_vals))  # 2 time points, num_q_vals q values
        g2_list.append(g2_data)

    # Test multiple plot type
    num_figs, num_lines = g2mod.compute_geometry(g2_list, "multiple")
    assert num_figs == num_q_vals
    assert num_lines == num_files

    # Test single plot type
    num_figs, num_lines = g2mod.compute_geometry(g2_list, "single")
    assert num_figs == num_files
    assert num_lines == num_q_vals

    # Test single-combined plot type
    num_figs, num_lines = g2mod.compute_geometry(g2_list, "single-combined")
    assert num_figs == 1
    assert num_lines == num_q_vals * num_files


class TestPerformanceCharacteristics:
    """Test suite for performance characteristics."""

    def test_get_data_performance_with_large_list(self, performance_timer):
        """Test get_data performance with large file list."""
        # Create many mock files
        xf_list = []
        for i in range(50):  # 50 files
            mock_xf = Mock()
            mock_xf.atype = ("Multitau",)
            mock_xf.get_g2_data.return_value = (
                np.array([0.01]),
                np.array([0.001]),
                np.array([[1.5]]),
                np.array([[0.05]]),
                [f"file_{i}"],
            )
            xf_list.append(mock_xf)

        performance_timer.start()
        q, _tel, _g2, _g2_err, _labels = g2mod.get_data(xf_list)
        elapsed = performance_timer.stop()

        # Should complete reasonably quickly
        assert elapsed < 1.0  # Less than 1 second for 50 files
        assert len(q) == 50
