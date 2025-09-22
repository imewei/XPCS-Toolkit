"""Unit tests for XpcsFile class.

This module provides comprehensive unit tests for the XpcsFile class,
covering file loading, data access, fitting operations, and memory management.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile


class TestMemoryMonitor:
    """Test suite for MemoryMonitor class."""

    @patch("xpcs_toolkit.xpcs_file.get_cached_memory_monitor")
    def test_get_memory_usage(self, mock_get_monitor):
        """Test memory usage retrieval."""
        mock_monitor = Mock()
        mock_monitor.get_memory_info.return_value = (1024.0, 2048.0, 0.33)
        mock_get_monitor.return_value = mock_monitor

        used, available = MemoryMonitor.get_memory_usage()

        assert used == 1024.0
        assert available == 2048.0
        mock_monitor.get_memory_info.assert_called_once()

    @patch("xpcs_toolkit.xpcs_file.get_cached_memory_monitor")
    def test_get_memory_pressure(self, mock_get_monitor):
        """Test memory pressure calculation."""
        mock_monitor = Mock()
        mock_status = Mock()
        mock_status.percent_used = 0.75
        mock_monitor.get_memory_status.return_value = mock_status
        mock_get_monitor.return_value = mock_monitor

        pressure = MemoryMonitor.get_memory_pressure()

        assert pressure == 0.75
        mock_monitor.get_memory_status.assert_called_once()

    @patch("psutil.virtual_memory")
    def test_is_memory_pressure_high(self, mock_virtual_memory):
        """Test memory pressure threshold check."""
        # Mock psutil.virtual_memory to return high memory usage
        mock_memory = Mock()
        mock_memory.percent = 90.0  # 90% memory usage
        mock_virtual_memory.return_value = mock_memory

        is_high = MemoryMonitor.is_memory_pressure_high(0.85)

        assert is_high is True
        mock_virtual_memory.assert_called_once()

        # Test with low memory usage
        mock_memory.percent = 70.0  # 70% memory usage
        is_high = MemoryMonitor.is_memory_pressure_high(0.85)
        assert is_high is False

    def test_estimate_array_memory(self):
        """Test array memory estimation."""
        # Test float64 array
        shape = (1024, 1024)
        dtype = np.dtype(np.float64)
        memory_mb = MemoryMonitor.estimate_array_memory(shape, dtype)

        expected_mb = (1024 * 1024 * 8) / (1024**2)  # 8MB
        assert memory_mb == expected_mb

        # Test int32 array
        shape = (512, 512)
        dtype = np.dtype(np.int32)
        memory_mb = MemoryMonitor.estimate_array_memory(shape, dtype)

        expected_mb = (512 * 512 * 4) / (1024**2)  # 1MB
        assert memory_mb == expected_mb


class TestXpcsFileInit:
    """Test suite for XpcsFile initialization."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_init_minimal(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test minimal XpcsFile initialization."""
        # Setup mocks
        mock_get_qmap.return_value = {"dqmap": np.zeros((10, 10))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([1, 2, 3]),
            "Iqp": np.array([[1, 2], [3, 4]]),
            "Int_t": np.array([10, 20]),
            "t0": 0.001,
            "t1": 0.002,
            "start_time": 1000000000,
            "tau": np.array([0.001, 0.01, 0.1]),
            "g2": np.array([2.0, 1.5, 1.1]),
            "g2_err": np.array([0.1, 0.05, 0.02]),
            "stride_frame": 1,
            "avg_frame": 1,
        }

        # Create XpcsFile instance
        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test_label"):
            xfile = XpcsFile("test_file.hdf")

        # Verify initialization
        assert xfile.fname == "test_file.hdf"
        assert xfile.atype == "Multitau"
        assert xfile.label == "test_label"
        assert hasattr(xfile, "qmap")
        assert hasattr(xfile, "saxs_1d")
        assert hasattr(xfile, "g2")
        assert xfile.hdf_info is None
        assert xfile.fit_summary is None
        assert xfile.c2_all_data is None
        assert xfile.saxs_2d_data is None
        assert xfile._saxs_data_loaded is False

        # Verify method calls
        mock_get_qmap.assert_called_once_with("test_file.hdf")
        mock_get_atype.assert_called_once_with("test_file.hdf")
        mock_batch_read.assert_called_once()

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_init_with_qmap_manager(
        self, mock_batch_read, mock_get_atype, mock_get_qmap
    ):
        """Test XpcsFile initialization with custom qmap manager."""
        mock_qmap_manager = Mock()
        mock_qmap_manager.get_qmap.return_value = {"dqmap": np.ones((5, 5))}
        mock_get_atype.return_value = "Twotime"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([1, 2]),
            "Iqp": np.array([[1, 2]]),
            "Int_t": np.array([10]),
            "t0": 0.001,
            "t1": 0.002,
            "start_time": 1000000000,
            "c2_g2": np.array([[1.5]]),
            "c2_g2_segments": np.array([1]),
            "c2_processed_bins": np.array([10]),
            "c2_stride_frame": 1,
            "c2_avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test_label"):
            xfile = XpcsFile("test_file.hdf", qmap_manager=mock_qmap_manager)

        # Verify qmap manager was used instead of get_qmap
        mock_get_qmap.assert_not_called()
        mock_qmap_manager.get_qmap.assert_called_once_with("test_file.hdf")
        assert xfile.atype == "Twotime"

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_init_with_extra_fields(
        self, mock_batch_read, mock_get_atype, mock_get_qmap
    ):
        """Test XpcsFile initialization with extra fields."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((10, 10))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([1, 2, 3]),
            "G2": np.array([1.8, 1.4, 1.1]),  # Extra field
            "IP": np.array([100, 200, 300]),  # Extra field
            "t0": 0.001,
            "t1": 0.002,
            "start_time": 1000000000,
            "tau": np.array([0.001, 0.01, 0.1]),
            "g2": np.array([2.0, 1.5, 1.1]),
            "g2_err": np.array([0.1, 0.05, 0.02]),
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test_label"):
            xfile = XpcsFile("test_file.hdf", fields=["G2", "IP"])

        # Verify extra fields are loaded
        assert hasattr(xfile, "G2")
        assert hasattr(xfile, "IP")
        np.testing.assert_array_equal(xfile.G2, [1.8, 1.4, 1.1])
        np.testing.assert_array_equal(xfile.IP, [100, 200, 300])


class TestXpcsFileStringRepresentation:
    """Test suite for XpcsFile string representations."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_str_representation(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test __str__ method."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((10, 10))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([1.0, 2.0]),
            "t0": 0.001,
            "start_time": 1000000000,
            "tau": np.array([0.001, 0.01]),
            "g2": np.array([2.0, 1.5]),
            "g2_err": np.array([0.1, 0.05]),
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test_file"):
            xfile = XpcsFile("test_file.hdf")

        str_repr = str(xfile)

        assert "File:test_file.hdf" in str_repr
        assert "atype" in str_repr
        assert "Multitau" in str_repr
        assert "label" in str_repr
        assert "test_file" in str_repr
        assert "saxs_1d" in str_repr
        assert "(2,)" in str_repr  # Array shape representation

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_repr_representation(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test __repr__ method."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([1.0]),
            "t0": 0.001,
            "tau": np.array([0.001]),
            "g2": np.array([2.0]),
            "g2_err": np.array([0.1]),
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            xfile = XpcsFile("test.hdf")

        repr_str = repr(xfile)

        assert "xpcs_file.XpcsFile" in repr_str
        assert "File:test.hdf" in repr_str


class TestXpcsFileDataAccess:
    """Test suite for XpcsFile data access methods."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    @patch("xpcs_toolkit.xpcs_file.read_metadata_to_dict")
    def test_get_hdf_info(
        self, mock_read_metadata, mock_batch_read, mock_get_atype, mock_get_qmap
    ):
        """Test get_hdf_info method."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "t0": 0.001,
            "tau": np.array([0.001]),
            "g2": np.array([2.0]),
            "saxs_1d": np.array([1.0, 2.0]),
            "Iqp": np.array([[1.0, 2.0]]),
            "Int_t": np.array([10, 20]),
            "stride_frame": 1,
            "avg_frame": 1,
        }
        mock_read_metadata.return_value = {"entry": {"instrument": "APS-8IDI"}}

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            xfile = XpcsFile("test.hdf")

        # First call should read metadata
        info = xfile.get_hdf_info()
        assert info == {"entry": {"instrument": "APS-8IDI"}}
        mock_read_metadata.assert_called_once_with("test.hdf")

        # Second call should use cached data
        info2 = xfile.get_hdf_info()
        assert info2 == info
        assert mock_read_metadata.call_count == 1  # Still only called once

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_update_label(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test update_label method."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "t0": 0.001,
            "tau": np.array([0.001]),
            "saxs_1d": np.array([1.0, 2.0]),
            "Iqp": np.array([[1.0, 2.0]]),
            "Int_t": np.array([10, 20]),
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id") as mock_create_id:
            mock_create_id.side_effect = ["initial_label", "updated_label"]
            xfile = XpcsFile("test.hdf", label_style="short")

            # Update label with different style
            new_label = xfile.update_label("long")

            assert new_label == "updated_label"
            assert xfile.label == "updated_label"
            assert mock_create_id.call_count == 2
            mock_create_id.assert_any_call("test.hdf", label_style="short")
            mock_create_id.assert_any_call("test.hdf", label_style="long")


class TestXpcsFileLoadData:
    """Test suite for XpcsFile data loading methods."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_load_data_multitau(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test load_data method for Multitau analysis."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"

        expected_fields = [
            "saxs_1d",
            "Iqp",
            "Int_t",
            "t0",
            "t1",
            "start_time",
            "tau",
            "g2",
            "g2_err",
            "stride_frame",
            "avg_frame",
        ]

        mock_batch_read.return_value = {
            field: np.array([1.0]) for field in expected_fields
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            XpcsFile("test.hdf")

        # Verify batch_read_fields called with correct fields
        call_args = mock_batch_read.call_args[0]
        assert call_args[0] == "test.hdf"
        assert set(call_args[1]) == set(expected_fields)
        assert call_args[2] == "alias"

        call_kwargs = mock_batch_read.call_args[1]
        assert call_kwargs["ftype"] == "nexus"
        assert call_kwargs["use_pool"] is True

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_load_data_twotime(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test load_data method for Twotime analysis."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Twotime"

        base_fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]
        twotime_fields = [
            "c2_g2",
            "c2_g2_segments",
            "c2_processed_bins",
            "c2_stride_frame",
            "c2_avg_frame",
        ]
        expected_fields = base_fields + twotime_fields

        return_data = {field: np.array([1.0]) for field in expected_fields}
        return_data["c2_stride_frame"] = 2
        return_data["c2_avg_frame"] = 3
        return_data["t0"] = 0.001

        mock_batch_read.return_value = return_data

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            xfile = XpcsFile("test.hdf")

        # Verify twotime-specific processing
        assert hasattr(xfile, "c2_t0")
        assert xfile.c2_t0 == 0.001 * 2 * 3  # t0 * stride_frame * avg_frame

        # Verify stride and avg frames are removed from main attributes
        assert not hasattr(xfile, "c2_stride_frame")
        assert not hasattr(xfile, "c2_avg_frame")

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_load_data_with_extra_fields(
        self, mock_batch_read, mock_get_atype, mock_get_qmap
    ):
        """Test load_data method with extra fields."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"

        base_fields = [
            "saxs_1d",
            "Iqp",
            "Int_t",
            "t0",
            "t1",
            "start_time",
            "tau",
            "g2",
            "g2_err",
            "stride_frame",
            "avg_frame",
        ]
        extra_fields = ["G2", "IP", "IF"]

        return_data = {field: np.array([1.0]) for field in base_fields + extra_fields}
        mock_batch_read.return_value = return_data

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            XpcsFile("test.hdf", fields=extra_fields)

        # Verify all fields are loaded (no duplicates)
        call_args = mock_batch_read.call_args[0]
        loaded_fields = set(call_args[1])
        expected_all_fields = set(base_fields + extra_fields)
        assert loaded_fields == expected_all_fields

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_load_data_removes_duplicates(
        self, mock_batch_read, mock_get_atype, mock_get_qmap
    ):
        """Test load_data method removes duplicate field names."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"

        # Include duplicate field names in extra_fields
        extra_fields = [
            "saxs_1d",
            "tau",
            "custom_field",
        ]  # saxs_1d and tau are duplicates

        mock_batch_read.return_value = {
            "saxs_1d": np.array([1.0]),
            "tau": np.array([0.001]),
            "g2": np.array([2.0]),
            "custom_field": np.array([10.0]),
            "t0": 0.001,
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            XpcsFile("test.hdf", fields=extra_fields)

        # Verify no duplicate fields in the call
        call_args = mock_batch_read.call_args[0]
        loaded_fields = call_args[1]
        assert len(loaded_fields) == len(set(loaded_fields))  # No duplicates
        assert "custom_field" in loaded_fields


@pytest.mark.parametrize(
    "analysis_type,expected_fields",
    [
        ("Multitau", ["tau", "g2", "g2_err", "stride_frame", "avg_frame"]),
        (
            "Twotime",
            [
                "c2_g2",
                "c2_g2_segments",
                "c2_processed_bins",
                "c2_stride_frame",
                "c2_avg_frame",
            ],
        ),
        ("Mixed", []),  # Unknown type should not add extra fields
    ],
)
def test_analysis_type_fields(analysis_type, expected_fields):
    """Test that correct fields are loaded based on analysis type."""
    base_fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]

    with (
        patch("xpcs_toolkit.xpcs_file.get_qmap") as mock_qmap,
        patch("xpcs_toolkit.xpcs_file.get_analysis_type") as mock_atype,
        patch("xpcs_toolkit.xpcs_file.batch_read_fields") as mock_batch,
        patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"),
    ):
        mock_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_atype.return_value = analysis_type
        mock_batch.return_value = {
            field: np.array([1.0]) for field in base_fields + expected_fields
        }

        XpcsFile("test.hdf")

        call_args = mock_batch.call_args[0]
        loaded_fields = set(call_args[1])
        expected_all_fields = set(base_fields + expected_fields)
        assert loaded_fields == expected_all_fields


class TestCreateId:
    """Test suite for create_id function."""

    @patch("xpcs_toolkit.xpcs_file.create_id")
    def test_create_id_called_correctly(self, mock_create_id):
        """Test create_id function is called with correct parameters."""
        mock_create_id.return_value = "test_label"

        with (
            patch(
                "xpcs_toolkit.xpcs_file.get_qmap",
                return_value={"dqmap": np.zeros((5, 5))},
            ),
            patch("xpcs_toolkit.xpcs_file.get_analysis_type", return_value="Multitau"),
            patch(
                "xpcs_toolkit.xpcs_file.batch_read_fields",
                return_value={
                    "t0": 0.001,
                    "tau": np.array([0.001]),
                    "saxs_1d": np.array([1.0, 2.0]),
                    "Iqp": np.array([[1.0, 2.0]]),
                    "Int_t": np.array([10, 20]),
                    "stride_frame": 1,
                    "avg_frame": 1,
                },
            ),
        ):
            xfile = XpcsFile("test_file.hdf", label_style="custom")

            mock_create_id.assert_called_with("test_file.hdf", label_style="custom")
            assert xfile.label == "test_label"


# Performance and edge case tests
class TestXpcsFilePerformance:
    """Test suite for XpcsFile performance characteristics."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_initialization_time(
        self, mock_batch_read, mock_get_atype, mock_get_qmap, performance_timer
    ):
        """Test XpcsFile initialization completes within reasonable time."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((100, 100))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.random.rand(1000),
            "g2": np.random.rand(50),
            "tau": np.random.rand(50),
            "t0": 0.001,
            "stride_frame": 1,
            "avg_frame": 1,
            "Iqp": np.random.rand(10, 100),
            "Int_t": np.random.rand(1000),
        }

        performance_timer.start()

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            xfile = XpcsFile("large_file.hdf")

        elapsed = performance_timer.stop()

        # Initialization should complete within 1 second
        assert elapsed < 1.0
        assert xfile is not None


class TestXpcsFileEdgeCases:
    """Test suite for XpcsFile edge cases and error conditions."""

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_empty_data_handling(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test handling of empty data arrays."""
        mock_get_qmap.return_value = {"dqmap": np.array([])}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "saxs_1d": np.array([]),
            "g2": np.array([]),
            "t0": 0.001,
            "stride_frame": 1,
            "avg_frame": 1,
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            xfile = XpcsFile("empty_file.hdf")

        assert len(xfile.saxs_1d) == 0
        assert len(xfile.g2) == 0
        assert xfile.t0 == 0.001

    @patch("xpcs_toolkit.xpcs_file.get_qmap")
    @patch("xpcs_toolkit.xpcs_file.get_analysis_type")
    @patch("xpcs_toolkit.xpcs_file.batch_read_fields")
    def test_none_extra_fields(self, mock_batch_read, mock_get_atype, mock_get_qmap):
        """Test handling when extra_fields is None."""
        mock_get_qmap.return_value = {"dqmap": np.zeros((5, 5))}
        mock_get_atype.return_value = "Multitau"
        mock_batch_read.return_value = {
            "t0": 0.001,
            "stride_frame": 1,
            "avg_frame": 1,
            "tau": np.array([0.001]),
            "saxs_1d": np.array([1.0, 2.0]),
            "Iqp": np.array([[1.0, 2.0]]),
            "Int_t": np.array([10, 20]),
        }

        with patch("xpcs_toolkit.xpcs_file.create_id", return_value="test"):
            # Should not raise exception with fields=None
            xfile = XpcsFile("test.hdf", fields=None)

        assert xfile is not None

        # Verify only standard fields were requested (no extra fields added)
        call_args = mock_batch_read.call_args[0]
        loaded_fields = call_args[1]
        base_fields = ["saxs_1d", "Iqp", "Int_t", "t0", "t1", "start_time"]
        multitau_fields = ["tau", "g2", "g2_err", "stride_frame", "avg_frame"]
        expected_fields = base_fields + multitau_fields
        assert set(loaded_fields) == set(expected_fields)
