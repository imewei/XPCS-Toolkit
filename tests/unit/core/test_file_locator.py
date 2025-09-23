"""Unit tests for FileLocator class and related functions.

This module provides comprehensive unit tests for the FileLocator class,
covering file discovery, dataset creation, and path management.
"""

import os
from unittest.mock import Mock, patch

import pytest

from xpcs_toolkit.file_locator import FileLocator, create_xpcs_dataset
from xpcs_toolkit.fileIO.qmap_utils import QMapManager
from xpcs_toolkit.helper.listmodel import ListDataModel


class TestCreateXpcsDataset:
    """Test suite for create_xpcs_dataset function."""

    @patch("xpcs_toolkit.file_locator.XF")
    def test_create_xpcs_dataset_success(self, mock_xf_class):
        """Test successful creation of XPCS dataset."""
        mock_xf = Mock()
        mock_xf_class.return_value = mock_xf

        result = create_xpcs_dataset("test_file.hdf", fields=["custom_field"])

        assert result is mock_xf
        mock_xf_class.assert_called_once_with("test_file.hdf", fields=["custom_field"])

    @patch("xpcs_toolkit.file_locator.XF")
    @patch("xpcs_toolkit.file_locator.logger")
    def test_create_xpcs_dataset_key_error(self, mock_logger, mock_xf_class):
        """Test handling of KeyError during dataset creation."""
        mock_xf_class.side_effect = KeyError("missing_dataset")

        result = create_xpcs_dataset("test_file.hdf")

        assert result is None
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Missing required HDF5 dataset" in warning_msg
        assert "missing_dataset" in warning_msg

    @patch("xpcs_toolkit.file_locator.XF")
    @patch("xpcs_toolkit.file_locator.logger")
    def test_create_xpcs_dataset_io_error(self, mock_logger, mock_xf_class):
        """Test handling of IOError during dataset creation."""
        mock_xf_class.side_effect = OSError("File not found")

        result = create_xpcs_dataset("nonexistent_file.hdf")

        assert result is None
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "File I/O error" in warning_msg
        assert "File not found" in warning_msg

    @patch("xpcs_toolkit.file_locator.XF")
    @patch("xpcs_toolkit.file_locator.logger")
    def test_create_xpcs_dataset_generic_exception(self, mock_logger, mock_xf_class):
        """Test handling of generic exception during dataset creation."""
        mock_xf_class.side_effect = ValueError("Invalid data format")

        result = create_xpcs_dataset("invalid_file.hdf")

        assert result is None
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "ValueError" in warning_msg
        assert "Invalid data format" in warning_msg

        # Should also log debug traceback
        mock_logger.debug.assert_called_once()
        debug_msg = mock_logger.debug.call_args[0][0]
        assert "Traceback" in debug_msg

    @patch("xpcs_toolkit.file_locator.XF")
    def test_create_xpcs_dataset_with_kwargs(self, mock_xf_class):
        """Test create_xpcs_dataset passes kwargs correctly."""
        mock_xf = Mock()
        mock_xf_class.return_value = mock_xf

        kwargs = {
            "fields": ["custom_field"],
            "label_style": "long",
            "qmap_manager": Mock(),
        }

        result = create_xpcs_dataset("test_file.hdf", **kwargs)

        assert result is mock_xf
        mock_xf_class.assert_called_once_with("test_file.hdf", **kwargs)


class TestFileLocatorInit:
    """Test suite for FileLocator initialization."""

    def test_init_basic(self):
        """Test basic FileLocator initialization."""
        path = "/test/path"
        locator = FileLocator(path)

        assert locator.path == path
        assert isinstance(locator.source, ListDataModel)
        assert isinstance(locator.source_search, ListDataModel)
        assert isinstance(locator.target, ListDataModel)
        assert isinstance(locator.qmap_manager, QMapManager)
        assert isinstance(locator.cache, dict)
        assert len(locator.cache) == 0
        assert locator.timestamp is None

    @patch("xpcs_toolkit.file_locator.ListDataModel")
    @patch("xpcs_toolkit.file_locator.QMapManager")
    def test_init_creates_components(
        self, mock_qmap_manager_class, mock_list_model_class
    ):
        """Test that initialization creates required components."""
        mock_list_model = Mock()
        mock_qmap_manager = Mock()
        mock_list_model_class.return_value = mock_list_model
        mock_qmap_manager_class.return_value = mock_qmap_manager

        locator = FileLocator("/test/path")

        # Should create three ListDataModel instances
        assert mock_list_model_class.call_count == 3
        assert mock_qmap_manager_class.call_count == 1

        assert locator.qmap_manager is mock_qmap_manager


class TestFileLocatorPathManagement:
    """Test suite for FileLocator path management."""

    def test_set_path(self):
        """Test set_path method."""
        locator = FileLocator("/initial/path")
        assert locator.path == "/initial/path"

        locator.set_path("/new/path")
        assert locator.path == "/new/path"

    def test_path_property_access(self):
        """Test direct path property access."""
        locator = FileLocator("/test/path")

        # Path should be readable
        assert locator.path == "/test/path"

        # Path should be writable
        locator.path = "/modified/path"
        assert locator.path == "/modified/path"


class TestFileLocatorClearMethod:
    """Test suite for FileLocator clear method."""

    def test_clear_method(self):
        """Test clear method clears source models."""
        locator = FileLocator("/test/path")

        # Mock the clear methods
        locator.source.clear = Mock()
        locator.source_search.clear = Mock()

        locator.clear()

        locator.source.clear.assert_called_once()
        locator.source_search.clear.assert_called_once()

    def test_clear_does_not_affect_other_attributes(self):
        """Test that clear method only affects source models."""
        locator = FileLocator("/test/path")

        # Add some data to cache and target
        locator.cache["test_key"] = "test_value"
        locator.target = Mock()
        original_target = locator.target

        locator.clear()

        # Cache and target should remain unchanged
        assert locator.cache["test_key"] == "test_value"
        assert locator.target is original_target


class TestFileLocatorGetXfList:
    """Test suite for FileLocator get_xf_list method."""

    def test_get_xf_list_no_rows(self):
        """Test get_xf_list with no rows specified."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf", "file2.hdf"]

        # Mock the cache with XF objects
        mock_xf1 = Mock()
        mock_xf1.fit_summary = None
        mock_xf1.atype = "Multitau"
        mock_xf2 = Mock()
        mock_xf2.fit_summary = None
        mock_xf2.atype = "Twotime"

        # Use normalized paths for cross-platform compatibility
        path1 = os.path.normpath("/test/path/file1.hdf")
        path2 = os.path.normpath("/test/path/file2.hdf")
        locator.cache[path1] = mock_xf1
        locator.cache[path2] = mock_xf2

        result = locator.get_xf_list()

        assert len(result) == 2
        assert mock_xf1 in result
        assert mock_xf2 in result

    def test_get_xf_list_with_rows(self):
        """Test get_xf_list with specific rows."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf", "file2.hdf", "file3.hdf"]

        mock_xf1 = Mock()
        mock_xf1.fit_summary = None
        mock_xf1.atype = "Multitau"
        mock_xf2 = Mock()
        mock_xf2.fit_summary = None
        mock_xf2.atype = "Twotime"

        # Use normalized paths for cross-platform compatibility
        path1 = os.path.normpath("/test/path/file1.hdf")
        path2 = os.path.normpath("/test/path/file2.hdf")
        locator.cache[path1] = mock_xf1
        locator.cache[path2] = mock_xf2

        result = locator.get_xf_list(rows=[0, 1])

        assert len(result) == 2
        assert mock_xf1 in result
        assert mock_xf2 in result

    @patch("xpcs_toolkit.file_locator.create_xpcs_dataset")
    def test_get_xf_list_creates_missing_cache(self, mock_create_dataset):
        """Test get_xf_list creates cache entries for missing files."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf"]

        mock_xf = Mock()
        mock_xf.fit_summary = None
        mock_xf.atype = "Multitau"
        mock_create_dataset.return_value = mock_xf

        result = locator.get_xf_list()

        assert len(result) == 1
        assert result[0] is mock_xf
        # Use normalized path for cross-platform compatibility
        expected_path = os.path.normpath("/test/path/file1.hdf")
        assert expected_path in locator.cache
        assert locator.cache[expected_path] is mock_xf

        mock_create_dataset.assert_called_once_with(
            expected_path, qmap_manager=locator.qmap_manager
        )

    @patch("xpcs_toolkit.file_locator.create_xpcs_dataset")
    @patch("xpcs_toolkit.file_locator.logger")
    def test_get_xf_list_skips_none_objects(self, mock_logger, mock_create_dataset):
        """Test get_xf_list skips files that failed to load."""
        locator = FileLocator("/test/path")
        locator.target = ["good_file.hdf", "bad_file.hdf"]

        # First call returns valid object, second returns None
        mock_xf = Mock()
        mock_xf.fit_summary = None
        mock_xf.atype = "Multitau"
        mock_create_dataset.side_effect = [mock_xf, None]

        result = locator.get_xf_list()

        assert len(result) == 1
        assert result[0] is mock_xf

        # Should log warning for failed file
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "Skipping invalid file" in warning_msg
        assert "bad_file.hdf" in warning_msg

    def test_get_xf_list_filter_fitted(self):
        """Test get_xf_list with filter_fitted=True."""
        locator = FileLocator("/test/path")
        locator.target = ["fitted_file.hdf", "unfitted_file.hdf"]

        # Create mocks with different fit_summary states
        mock_fitted = Mock()
        mock_fitted.fit_summary = {"beta": 0.8}  # Has fit results
        mock_fitted.atype = "Multitau"

        mock_unfitted = Mock()
        mock_unfitted.fit_summary = None  # No fit results
        mock_unfitted.atype = "Multitau"

        # Use normalized paths for cross-platform compatibility
        fitted_path = os.path.normpath("/test/path/fitted_file.hdf")
        unfitted_path = os.path.normpath("/test/path/unfitted_file.hdf")
        locator.cache[fitted_path] = mock_fitted
        locator.cache[unfitted_path] = mock_unfitted

        result = locator.get_xf_list(filter_fitted=True)

        assert len(result) == 1
        assert result[0] is mock_fitted

    def test_get_xf_list_filter_atype(self):
        """Test get_xf_list with analysis type filter."""
        locator = FileLocator("/test/path")
        locator.target = ["multitau_file.hdf", "twotime_file.hdf"]

        mock_multitau = Mock()
        mock_multitau.fit_summary = None
        mock_multitau.atype = "Multitau"

        mock_twotime = Mock()
        mock_twotime.fit_summary = None
        mock_twotime.atype = "Twotime"

        # Use normalized paths for cross-platform compatibility
        multitau_path = os.path.normpath("/test/path/multitau_file.hdf")
        twotime_path = os.path.normpath("/test/path/twotime_file.hdf")
        locator.cache[multitau_path] = mock_multitau
        locator.cache[twotime_path] = mock_twotime

        result = locator.get_xf_list(filter_atype="Multitau")

        assert len(result) == 1
        assert result[0] is mock_multitau

    def test_get_xf_list_invalid_row_indices(self):
        """Test get_xf_list handles invalid row indices gracefully."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf", "file2.hdf"]

        mock_xf1 = Mock()
        mock_xf1.fit_summary = None
        mock_xf1.atype = "Multitau"
        # Use normalized paths for cross-platform compatibility
        path1 = os.path.normpath("/test/path/file1.hdf")
        locator.cache[path1] = mock_xf1

        # Include invalid indices
        result = locator.get_xf_list(rows=[-1, 0, 5, 10])

        # Should only include valid file
        assert len(result) == 1
        assert result[0] is mock_xf1


class TestFileLocatorGetHdfInfo:
    """Test suite for FileLocator get_hdf_info method."""

    @patch("xpcs_toolkit.file_locator.create_xpcs_dataset")
    def test_get_hdf_info_success(self, mock_create_dataset):
        """Test successful HDF info retrieval."""
        locator = FileLocator("/test/path")

        mock_xf = Mock()
        mock_hdf_info = {"entry": {"instrument": "APS-8IDI"}}
        mock_xf.get_hdf_info.return_value = mock_hdf_info
        mock_create_dataset.return_value = mock_xf

        result = locator.get_hdf_info("test_file.hdf", filter_str=["entry"])

        assert result == mock_hdf_info
        mock_create_dataset.assert_called_once_with(
            os.path.normpath("/test/path/test_file.hdf"),
            qmap_manager=locator.qmap_manager,
        )
        mock_xf.get_hdf_info.assert_called_once_with(["entry"])

    @patch("xpcs_toolkit.file_locator.create_xpcs_dataset")
    def test_get_hdf_info_without_filter(self, mock_create_dataset):
        """Test HDF info retrieval without filter string."""
        locator = FileLocator("/test/path")

        mock_xf = Mock()
        mock_hdf_info = {"full": "info"}
        mock_xf.get_hdf_info.return_value = mock_hdf_info
        mock_create_dataset.return_value = mock_xf

        result = locator.get_hdf_info("test_file.hdf")

        assert result == mock_hdf_info
        mock_xf.get_hdf_info.assert_called_once_with(None)


class TestFileLocatorCaching:
    """Test suite for FileLocator caching behavior."""

    def test_cache_persistence(self):
        """Test that cache persists between calls."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf"]

        mock_xf = Mock()
        mock_xf.fit_summary = None
        mock_xf.atype = "Multitau"

        # Manually add to cache with normalized path
        path1 = os.path.normpath("/test/path/file1.hdf")
        locator.cache[path1] = mock_xf

        # First call should use cache
        result1 = locator.get_xf_list()
        assert len(result1) == 1
        assert result1[0] is mock_xf

        # Second call should also use cache
        result2 = locator.get_xf_list()
        assert len(result2) == 1
        assert result2[0] is mock_xf
        assert result1[0] is result2[0]  # Same object reference

    def test_cache_key_format(self):
        """Test that cache keys use full path format."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf"]

        with patch("xpcs_toolkit.file_locator.create_xpcs_dataset") as mock_create:
            mock_xf = Mock()
            mock_xf.fit_summary = None
            mock_xf.atype = "Multitau"
            mock_create.return_value = mock_xf

            locator.get_xf_list()

            # Cache key should be full path (normalized for cross-platform compatibility)
            expected_path = os.path.normpath("/test/path/file1.hdf")
            assert expected_path in locator.cache
            assert "file1.hdf" not in locator.cache


class TestFileLocatorEdgeCases:
    """Test suite for FileLocator edge cases."""

    def test_empty_target_list(self):
        """Test behavior with empty target list."""
        locator = FileLocator("/test/path")
        locator.target = []

        result = locator.get_xf_list()
        assert len(result) == 0
        assert isinstance(result, list)

    def test_empty_rows_list(self):
        """Test behavior with empty rows list."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf", "file2.hdf"]

        result = locator.get_xf_list(rows=[])
        assert len(result) == 0
        assert isinstance(result, list)

    def test_target_as_list_model(self):
        """Test that target works as ListDataModel."""
        locator = FileLocator("/test/path")

        # target should support list-like operations
        assert hasattr(locator.target, "__len__")
        assert hasattr(locator.target, "__getitem__")

        # Should work with ListDataModel methods
        if hasattr(locator.target, "append"):
            locator.target.append("test_file.hdf")


class TestFileLocatorPerformance:
    """Test suite for FileLocator performance characteristics."""

    def test_initialization_performance(self, performance_timer):
        """Test FileLocator initialization performance."""
        performance_timer.start()

        locator = FileLocator("/test/path")

        elapsed = performance_timer.stop()

        # Initialization should be very fast (< 50ms)
        assert elapsed < 0.05
        assert locator is not None

    @patch("xpcs_toolkit.file_locator.create_xpcs_dataset")
    def test_cache_access_performance(self, mock_create_dataset, performance_timer):
        """Test that cached access is faster than creation."""
        locator = FileLocator("/test/path")
        locator.target = ["file1.hdf"] * 100  # Large list

        mock_xf = Mock()
        mock_xf.fit_summary = None
        mock_xf.atype = "Multitau"
        mock_create_dataset.return_value = mock_xf

        # First call - should create cache entries
        performance_timer.start()
        result1 = locator.get_xf_list(rows=[0])
        first_call_time = performance_timer.stop()

        # Second call - should use cache
        performance_timer.start()
        result2 = locator.get_xf_list(rows=[0])
        second_call_time = performance_timer.stop()

        # Cached access should be faster
        assert second_call_time < first_call_time
        assert len(result1) == len(result2) == 1


@pytest.mark.parametrize(
    "path_input",
    [
        "/absolute/path",
        "relative/path",
        "",
        ".",
        "/path/with/spaces in name",
        "/path/with/unicode/файл",
    ],
)
def test_file_locator_path_variations(path_input):
    """Test FileLocator with various path formats."""
    locator = FileLocator(path_input)
    assert locator.path == path_input


@pytest.mark.parametrize(
    "rows_input,expected_behavior",
    [
        (None, "use_all"),
        ([], "empty_result"),
        ([0, 1, 2], "specific_rows"),
        ([-1, 0, 1], "mixed_valid_invalid"),
        ([100, 200], "all_invalid"),
    ],
)
def test_get_xf_list_row_variations(rows_input, expected_behavior):
    """Test get_xf_list with various row specifications."""
    locator = FileLocator("/test/path")
    locator.target = ["file1.hdf", "file2.hdf"]  # 2 files available

    # Mock cache with valid objects
    for _i, filename in enumerate(locator.target):
        mock_xf = Mock()
        mock_xf.fit_summary = None
        mock_xf.atype = "Multitau"
        # Use normpath to ensure cross-platform compatibility
        cache_key = os.path.normpath(os.path.join("/test/path", filename))
        locator.cache[cache_key] = mock_xf

    result = locator.get_xf_list(rows=rows_input)

    if expected_behavior == "use_all":
        assert len(result) == 2
    elif expected_behavior == "empty_result":
        assert len(result) == 0
    elif expected_behavior == "specific_rows":
        assert len(result) == 2  # Both files exist
    elif expected_behavior == "mixed_valid_invalid":
        assert len(result) == 2  # Only valid indices processed
    elif expected_behavior == "all_invalid":
        assert len(result) == 0  # No valid indices
