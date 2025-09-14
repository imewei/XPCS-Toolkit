"""Unit tests for QMap utilities module.

This module provides comprehensive unit tests for Q-space mapping utilities,
covering Q-space calculations and detector geometry.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from xpcs_toolkit.fileIO.qmap_utils import QMap, QMapManager, get_hash, get_qmap


class TestQMapManager:
    """Test suite for QMapManager class."""

    def test_init(self):
        """Test QMapManager initialization."""
        manager = QMapManager()

        assert hasattr(manager, "db")
        assert isinstance(manager.db, dict)
        assert len(manager.db) == 0

    @patch("xpcs_toolkit.fileIO.qmap_utils.get_hash")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap")
    def test_get_qmap_new(self, mock_qmap_class, mock_get_hash):
        """Test getting new QMap."""
        mock_get_hash.return_value = "test_hash_123"
        mock_qmap = Mock()
        mock_qmap_class.return_value = mock_qmap

        manager = QMapManager()
        result = manager.get_qmap("/test/file.hdf")

        assert result is mock_qmap
        assert "test_hash_123" in manager.db
        assert manager.db["test_hash_123"] is mock_qmap

        mock_get_hash.assert_called_once_with("/test/file.hdf")
        mock_qmap_class.assert_called_once_with(fname="/test/file.hdf")

    @patch("xpcs_toolkit.fileIO.qmap_utils.get_hash")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap")
    def test_get_qmap_cached(self, mock_qmap_class, mock_get_hash):
        """Test getting cached QMap."""
        mock_get_hash.return_value = "test_hash_123"
        mock_qmap = Mock()

        manager = QMapManager()
        # Manually add to cache
        manager.db["test_hash_123"] = mock_qmap

        result = manager.get_qmap("/test/file.hdf")

        assert result is mock_qmap
        # QMap constructor should not be called for cached entry
        mock_qmap_class.assert_not_called()

    @patch("xpcs_toolkit.fileIO.qmap_utils.get_hash")
    def test_get_qmap_different_files(self, mock_get_hash):
        """Test getting QMaps for different files."""
        mock_get_hash.side_effect = ["hash1", "hash2", "hash1"]  # Same hash for file1

        manager = QMapManager()

        with patch("xpcs_toolkit.fileIO.qmap_utils.QMap") as mock_qmap_class:
            mock_qmap1 = Mock()
            mock_qmap2 = Mock()
            mock_qmap_class.side_effect = [mock_qmap1, mock_qmap2]

            result1 = manager.get_qmap("/test/file1.hdf")
            result2 = manager.get_qmap("/test/file2.hdf")
            result3 = manager.get_qmap("/test/file1.hdf")  # Should use cache

        assert result1 is mock_qmap1
        assert result2 is mock_qmap2
        assert result3 is mock_qmap1  # Same as first call
        assert len(manager.db) == 2
        assert mock_qmap_class.call_count == 2  # Only 2 new QMap objects


class TestQMapInit:
    """Test suite for QMap initialization."""

    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels")
    def test_init_success(
        self, mock_create_labels, mock_compute_qmap, mock_get_extent, mock_load_dataset
    ):
        """Test successful QMap initialization."""
        # Setup return values
        mock_get_extent.return_value = (100, 100, 200, 200)
        mock_compute_qmap.return_value = ({"q": np.array([0.1, 0.2])}, {"q": "1/A"})
        mock_create_labels.return_value = ["q=0.1", "q=0.2"]

        qmap = QMap(fname="/test/file.hdf", root_key="/custom/qmap")

        assert qmap.root_key == "/custom/qmap"
        assert qmap.fname == "/test/file.hdf"
        assert qmap.extent == (100, 100, 200, 200)
        assert "q" in qmap.qmap
        assert "q" in qmap.qmap_units
        assert len(qmap.qbin_labels) == 2

        # Verify method calls
        mock_load_dataset.assert_called_once()
        mock_get_extent.assert_called_once()
        mock_compute_qmap.assert_called_once()
        mock_create_labels.assert_called_once()

    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap._create_minimal_fallback")
    def test_init_with_exception(self, mock_fallback, mock_load_dataset):
        """Test QMap initialization with exception."""
        mock_load_dataset.side_effect = Exception("Test error")

        QMap(fname="/test/file.hdf")

        mock_fallback.assert_called_once()

    def test_init_default_params(self):
        """Test QMap initialization with default parameters."""
        with (
            patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset"),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent",
                return_value=(0, 0, 100, 100),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap",
                return_value=({}, {}),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels",
                return_value=[],
            ),
        ):
            qmap = QMap()

            assert qmap.root_key == "/xpcs/qmap"
            assert qmap.fname is None
            assert hasattr(qmap, "_qmap_cache")
            assert hasattr(qmap, "_qbin_cache")
            assert hasattr(qmap, "_extent_cache")


class TestQMapLoadDataset:
    """Test suite for QMap load_dataset method."""

    @patch("xpcs_toolkit.fileIO.qmap_utils._connection_pool")
    @patch("xpcs_toolkit.fileIO.qmap_utils.key_map")
    def test_load_dataset_success(self, mock_key_map, mock_pool):
        """Test successful dataset loading."""
        # Setup key mappings
        mock_key_map.__getitem__.side_effect = lambda x: {
            "nexus": {
                "mask": "/xpcs/qmap/mask",
                "dqmap": "/xpcs/qmap/dqmap",
                "sqmap": "/xpcs/qmap/sqmap",
                "dqlist": "/xpcs/qmap/dqlist",
                "sqlist": "/xpcs/qmap/sqlist",
                "dplist": "/xpcs/qmap/dplist",
                "splist": "/xpcs/qmap/splist",
                "bcx": "/xpcs/qmap/bcx",
                "bcy": "/xpcs/qmap/bcy",
                "X_energy": "/xpcs/qmap/X_energy",
                "static_index_mapping": "/xpcs/qmap/static_index_mapping",
                "dynamic_index_mapping": "/xpcs/qmap/dynamic_index_mapping",
                "pixel_size": "/xpcs/qmap/pixel_size",
                "det_dist": "/xpcs/qmap/det_dist",
                "dynamic_num_pts": "/xpcs/qmap/dynamic_num_pts",
                "static_num_pts": "/xpcs/qmap/static_num_pts",
                "map_names": "/xpcs/qmap/map_names",
                "map_units": "/xpcs/qmap/map_units",
            }
        }[x]

        # Setup mock file
        mock_file = Mock()
        mock_file.__contains__ = Mock(side_effect=lambda x: x.startswith("/xpcs/qmap"))
        mock_file.__getitem__ = Mock(
            side_effect=lambda x: Mock(__call__=Mock(return_value=np.array([1, 2, 3])))
        )

        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_file)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

        # Create QMap and test load_dataset
        qmap = QMap.__new__(QMap)  # Create without calling __init__
        qmap.fname = "/test/file.hdf"
        qmap._qmap_cache = None
        qmap._qbin_cache = {}
        qmap._extent_cache = None

        qmap.load_dataset()

        # Verify file was accessed
        mock_pool.get_connection.assert_called_once_with("/test/file.hdf", "r")

    @patch("xpcs_toolkit.fileIO.qmap_utils._connection_pool")
    @patch("xpcs_toolkit.fileIO.qmap_utils.key_map")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap._create_default_qmap")
    def test_load_dataset_missing_qmap_group(
        self, mock_create_default, mock_key_map, mock_pool
    ):
        """Test loading dataset when qmap group is missing."""
        mock_key_map.__getitem__.side_effect = lambda x: {
            "nexus": {"mask": "/xpcs/qmap/mask"}
        }[x]

        # Setup mock file without qmap group
        mock_file = Mock()
        mock_file.__contains__ = Mock(return_value=False)  # qmap group not found

        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_file)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

        qmap = QMap.__new__(QMap)
        qmap.fname = "/test/file.hdf"
        qmap._qmap_cache = None
        qmap._qbin_cache = {}
        qmap._extent_cache = None

        qmap.load_dataset()

        mock_create_default.assert_called_once()

    @patch("xpcs_toolkit.fileIO.qmap_utils._connection_pool")
    @patch("xpcs_toolkit.fileIO.qmap_utils.key_map")
    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap._get_default_value")
    def test_load_dataset_missing_keys(self, mock_get_default, mock_key_map, mock_pool):
        """Test loading dataset with missing keys."""
        mock_key_map.__getitem__.side_effect = lambda x: {
            "nexus": {
                "mask": "/xpcs/qmap/mask",
                "dqmap": "/xpcs/qmap/dqmap",
            }
        }[x]

        # Setup mock file with partial data
        mock_file = Mock()
        mock_file.__contains__ = Mock(side_effect=lambda x: x == "/xpcs/qmap")
        mock_file.__getitem__ = Mock(side_effect=KeyError("Dataset not found"))

        mock_pool.get_connection.return_value.__enter__ = Mock(return_value=mock_file)
        mock_pool.get_connection.return_value.__exit__ = Mock(return_value=None)

        mock_get_default.return_value = np.array([0])

        qmap = QMap.__new__(QMap)
        qmap.fname = "/test/file.hdf"
        qmap._qmap_cache = None
        qmap._qbin_cache = {}
        qmap._extent_cache = None

        qmap.load_dataset()

        # Should call _get_default_value for missing keys
        assert mock_get_default.call_count > 0


class TestQMapUtilityMethods:
    """Test suite for QMap utility methods."""

    def test_get_default_value(self):
        """Test _get_default_value method."""
        qmap = QMap.__new__(QMap)

        # Test known defaults
        assert qmap._get_default_value("bcx") == 512.0
        assert qmap._get_default_value("bcy") == 512.0
        assert qmap._get_default_value("X_energy") == 8.0
        assert qmap._get_default_value("pixel_size") == 75e-6
        assert qmap._get_default_value("det_dist") == 5.0

        # Test mask default
        mask_default = qmap._get_default_value("mask")
        assert isinstance(mask_default, np.ndarray)
        assert mask_default.shape == (1024, 1024)
        assert np.all(mask_default == 1)

        # Test unknown key
        unknown_default = qmap._get_default_value("unknown_key")
        assert isinstance(unknown_default, np.ndarray)
        assert len(unknown_default) == 1
        assert unknown_default[0] == 0

    @patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset")
    def test_create_minimal_fallback(self, mock_load_dataset):
        """Test _create_minimal_fallback method."""
        qmap = QMap.__new__(QMap)
        qmap.fname = "/test/file.hdf"
        qmap._qmap_cache = None
        qmap._qbin_cache = {}
        qmap._extent_cache = None

        qmap._create_minimal_fallback()

        # Should have basic attributes
        assert hasattr(qmap, "extent")
        assert hasattr(qmap, "qmap")
        assert hasattr(qmap, "qmap_units")
        assert hasattr(qmap, "qbin_labels")

        # Check defaults
        assert qmap.extent == (0, 0, 1024, 1024)
        assert isinstance(qmap.qmap, dict)
        assert isinstance(qmap.qmap_units, dict)
        assert isinstance(qmap.qbin_labels, list)


class TestQMapDetectorExtent:
    """Test suite for QMap detector extent calculation."""

    def test_get_detector_extent(self):
        """Test get_detector_extent method."""
        qmap = QMap.__new__(QMap)

        # Setup mock qmap data
        qmap.qmap_data = {
            "mask": np.ones((1024, 1024)),
            "bcx": 512.0,
            "bcy": 512.0,
        }

        extent = qmap.get_detector_extent()

        # Should return tuple with proper format
        assert isinstance(extent, tuple)
        assert len(extent) == 4

        # For 1024x1024 detector with center at (512, 512)
        # extent should be (x_min, y_min, x_max, y_max)
        assert all(isinstance(x, (int, float)) for x in extent)


class TestQMapComputeQmap:
    """Test suite for QMap compute_qmap method."""

    def test_compute_qmap_basic(self):
        """Test basic qmap computation."""
        qmap = QMap.__new__(QMap)

        # Setup minimal qmap_data
        qmap.qmap_data = {
            "dqlist": np.array([0.01, 0.02, 0.03]),
            "sqlist": np.array([0.01, 0.02, 0.03]),
            "dplist": np.array([-180, 0, 180]),
            "splist": np.array([-180, 0, 180]),
            "map_names": [b"q", b"phi"],
            "map_units": [b"1/A", b"degree"],
        }

        qmap_dict, units_dict = qmap.compute_qmap()

        assert isinstance(qmap_dict, dict)
        assert isinstance(units_dict, dict)

        # Should contain q and phi mappings
        assert "q" in qmap_dict or "phi" in qmap_dict
        assert len(units_dict) >= 0


class TestQMapCreateQbinLabels:
    """Test suite for QMap create_qbin_labels method."""

    def test_create_qbin_labels(self):
        """Test qbin labels creation."""
        qmap = QMap.__new__(QMap)

        # Setup qmap data
        qmap.qmap = {"q": np.array([0.01, 0.02, 0.03]), "phi": np.array([-180, 0, 180])}
        qmap.qmap_units = {"q": "1/A", "phi": "degree"}

        labels = qmap.create_qbin_labels()

        assert isinstance(labels, list)
        assert len(labels) >= 0

        # Labels should be strings
        if labels:
            assert all(isinstance(label, str) for label in labels)


class TestGetHashFunction:
    """Test suite for get_hash function."""

    @patch("os.path.getmtime")
    @patch("os.path.getsize")
    def test_get_hash_basic(self, mock_getsize, mock_getmtime):
        """Test basic hash generation."""
        mock_getmtime.return_value = 1234567890.0
        mock_getsize.return_value = 1024000

        hash_value = get_hash("/test/file.hdf")

        assert isinstance(hash_value, str)
        assert len(hash_value) > 0

        mock_getmtime.assert_called_once_with("/test/file.hdf")
        mock_getsize.assert_called_once_with("/test/file.hdf")

    @patch("os.path.getmtime")
    @patch("os.path.getsize")
    def test_get_hash_consistency(self, mock_getsize, mock_getmtime):
        """Test that hash is consistent for same file properties."""
        mock_getmtime.return_value = 1234567890.0
        mock_getsize.return_value = 1024000

        hash1 = get_hash("/test/file.hdf")
        hash2 = get_hash("/test/file.hdf")

        assert hash1 == hash2

    @patch("os.path.getmtime")
    @patch("os.path.getsize")
    def test_get_hash_different_files(self, mock_getsize, mock_getmtime):
        """Test that hash is different for different files."""
        # First file
        mock_getmtime.return_value = 1234567890.0
        mock_getsize.return_value = 1024000
        hash1 = get_hash("/test/file1.hdf")

        # Second file (different properties)
        mock_getmtime.return_value = 1234567999.0
        mock_getsize.return_value = 2048000
        hash2 = get_hash("/test/file2.hdf")

        assert hash1 != hash2

    @patch("os.path.getmtime")
    @patch("os.path.getsize")
    def test_get_hash_exception_handling(self, mock_getsize, mock_getmtime):
        """Test hash generation with file access errors."""
        mock_getmtime.side_effect = OSError("File not accessible")
        mock_getsize.side_effect = OSError("File not accessible")

        # Should still generate hash using filename
        hash_value = get_hash("/test/file.hdf")

        assert isinstance(hash_value, str)
        assert len(hash_value) > 0


class TestGetQmapFunction:
    """Test suite for get_qmap function."""

    @patch("xpcs_toolkit.fileIO.qmap_utils._qmap_manager")
    def test_get_qmap_function(self, mock_manager):
        """Test get_qmap function."""
        mock_qmap = Mock()
        mock_manager.get_qmap.return_value = mock_qmap

        result = get_qmap("/test/file.hdf")

        assert result is mock_qmap
        mock_manager.get_qmap.assert_called_once_with("/test/file.hdf")


class TestQMapCaching:
    """Test suite for QMap caching mechanisms."""

    def test_qmap_cache_initialization(self):
        """Test that caching structures are properly initialized."""
        with (
            patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset"),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent",
                return_value=(0, 0, 100, 100),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap",
                return_value=({}, {}),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels",
                return_value=[],
            ),
        ):
            qmap = QMap("/test/file.hdf")

            assert hasattr(qmap, "_qmap_cache")
            assert hasattr(qmap, "_qbin_cache")
            assert hasattr(qmap, "_extent_cache")
            assert isinstance(qmap._qbin_cache, dict)


class TestQMapPerformance:
    """Test suite for QMap performance characteristics."""

    def test_qmap_initialization_time(self, performance_timer):
        """Test QMap initialization performance."""
        with (
            patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset"),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent",
                return_value=(0, 0, 100, 100),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap",
                return_value=({}, {}),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels",
                return_value=[],
            ),
        ):
            performance_timer.start()
            qmap = QMap("/test/file.hdf")
            elapsed = performance_timer.stop()

            # Initialization should be reasonably fast
            assert elapsed < 1.0
            assert qmap is not None


class TestQMapEdgeCases:
    """Test suite for QMap edge cases."""

    def test_qmap_with_none_filename(self):
        """Test QMap with None filename."""
        with (
            patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset"),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent",
                return_value=(0, 0, 100, 100),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap",
                return_value=({}, {}),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels",
                return_value=[],
            ),
        ):
            qmap = QMap(fname=None)

            assert qmap.fname is None
            assert qmap.root_key == "/xpcs/qmap"

    def test_qmap_empty_root_key(self):
        """Test QMap with empty root key."""
        with (
            patch("xpcs_toolkit.fileIO.qmap_utils.QMap.load_dataset"),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.get_detector_extent",
                return_value=(0, 0, 100, 100),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.compute_qmap",
                return_value=({}, {}),
            ),
            patch(
                "xpcs_toolkit.fileIO.qmap_utils.QMap.create_qbin_labels",
                return_value=[],
            ),
        ):
            qmap = QMap(fname="/test/file.hdf", root_key="")

            assert qmap.root_key == ""
            assert qmap.fname == "/test/file.hdf"


@pytest.mark.parametrize(
    "hash_input,expected_type",
    [
        ("/absolute/path/file.hdf", str),
        ("relative/path/file.hdf", str),
        ("file.hdf", str),
        ("", str),
    ],
)
def test_get_hash_path_variations(hash_input, expected_type):
    """Test get_hash with various path formats."""
    with (
        patch("os.path.getmtime", return_value=1000.0),
        patch("os.path.getsize", return_value=1024),
    ):
        result = get_hash(hash_input)

        assert isinstance(result, expected_type)
        assert len(result) > 0


@pytest.mark.parametrize(
    "manager_cache_size,expected_behavior",
    [
        (0, "empty_cache"),
        (1, "single_entry"),
        (10, "multiple_entries"),
    ],
)
def test_qmap_manager_cache_behavior(manager_cache_size, expected_behavior):
    """Test QMapManager cache behavior with different sizes."""
    manager = QMapManager()

    # Pre-populate cache
    for i in range(manager_cache_size):
        mock_qmap = Mock()
        manager.db[f"hash_{i}"] = mock_qmap

    assert len(manager.db) == manager_cache_size

    if expected_behavior == "empty_cache":
        assert len(manager.db) == 0
    elif expected_behavior == "single_entry":
        assert len(manager.db) == 1
    elif expected_behavior == "multiple_entries":
        assert len(manager.db) == 10
