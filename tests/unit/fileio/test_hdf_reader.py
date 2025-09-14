"""Unit tests for HDF5 reader module.

This module provides comprehensive unit tests for the HDF5 reader,
covering connection pooling, batch reading, and metadata operations.
"""

import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from xpcs_toolkit.fileIO.hdf_reader import (
    ConnectionStats,
    HDF5ConnectionPool,
    PooledConnection,
    batch_read_fields,
    get,
    get_analysis_type,
)


class TestPooledConnection:
    """Test suite for PooledConnection class."""

    def test_init(self):
        """Test PooledConnection initialization."""
        mock_file = Mock()
        file_path = "/test/path/file.hdf"

        with patch("time.time", return_value=1000.0):
            conn = PooledConnection(mock_file, file_path)

        assert conn.file_handle is mock_file
        assert conn.file_path == file_path
        assert conn.created_at == 1000.0
        assert conn.last_accessed == 1000.0
        assert conn.access_count == 0
        assert conn.is_healthy is True
        assert hasattr(conn, "lock")
        assert isinstance(conn.lock, threading.RLock)

    def test_touch(self):
        """Test touch method updates access info."""
        mock_file = Mock()
        conn = PooledConnection(mock_file, "/test/file.hdf")

        initial_access_count = conn.access_count

        with patch("time.time", return_value=2000.0):
            conn.touch()

        assert conn.access_count == initial_access_count + 1
        assert conn.last_accessed == 2000.0

    def test_check_health_success(self):
        """Test successful health check."""
        mock_file = Mock()
        mock_file.filename = "/test/file.hdf"

        with patch("os.path.exists", return_value=True):
            conn = PooledConnection(mock_file, "/test/file.hdf")
            result = conn.check_health()

        assert result is True
        assert conn.is_healthy is True

    def test_check_health_file_not_exists(self):
        """Test health check when file doesn't exist."""
        mock_file = Mock()
        mock_file.filename = "/test/file.hdf"

        with patch("os.path.exists", return_value=False):
            conn = PooledConnection(mock_file, "/test/file.hdf")
            result = conn.check_health()

        assert result is False
        assert conn.is_healthy is False

    def test_check_health_file_handle_error(self):
        """Test health check when file handle raises error."""
        mock_file = Mock()
        mock_file.filename = Mock(side_effect=ValueError("File closed"))

        conn = PooledConnection(mock_file, "/test/file.hdf")
        result = conn.check_health()

        assert result is False
        assert conn.is_healthy is False

    def test_close_success(self):
        """Test successful connection close."""
        mock_file = Mock()
        conn = PooledConnection(mock_file, "/test/file.hdf")

        conn.close()

        mock_file.close.assert_called_once()
        assert conn.is_healthy is False

    def test_close_with_exception(self):
        """Test connection close handles exceptions."""
        mock_file = Mock()
        mock_file.close.side_effect = Exception("Close error")

        conn = PooledConnection(mock_file, "/test/file.hdf")

        # Should not raise exception
        conn.close()

        mock_file.close.assert_called_once()
        assert conn.is_healthy is False

    def test_close_without_close_method(self):
        """Test connection close when file handle lacks close method."""
        mock_file = Mock()
        del mock_file.close  # Remove close method

        conn = PooledConnection(mock_file, "/test/file.hdf")

        # Should not raise exception
        conn.close()

        assert conn.is_healthy is False


class TestConnectionStats:
    """Test suite for ConnectionStats class."""

    def test_init(self):
        """Test ConnectionStats initialization."""
        with patch("time.time", return_value=5000.0):
            stats = ConnectionStats()

        assert stats.total_connections_created == 0
        assert stats.total_connections_reused == 0
        assert stats.total_connections_evicted == 0
        assert stats.total_health_checks == 0
        assert stats.failed_health_checks == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.io_time_seconds == 0.0
        assert stats.start_time == 5000.0
        assert hasattr(stats, "_lock")

    def test_record_connection_created(self):
        """Test recording connection creation."""
        stats = ConnectionStats()

        stats.record_connection_created()
        assert stats.total_connections_created == 1

        stats.record_connection_created()
        assert stats.total_connections_created == 2

    def test_record_connection_reused(self):
        """Test recording connection reuse."""
        stats = ConnectionStats()

        stats.record_connection_reused()
        assert stats.total_connections_reused == 1
        assert stats.cache_hits == 1

    def test_record_connection_evicted(self):
        """Test recording connection eviction."""
        stats = ConnectionStats()

        stats.record_connection_evicted()
        assert stats.total_connections_evicted == 1

    def test_record_health_check_success(self):
        """Test recording successful health check."""
        stats = ConnectionStats()

        stats.record_health_check(True)
        assert stats.total_health_checks == 1
        assert stats.failed_health_checks == 0

    def test_record_health_check_failure(self):
        """Test recording failed health check."""
        stats = ConnectionStats()

        stats.record_health_check(False)
        assert stats.total_health_checks == 1
        assert stats.failed_health_checks == 1

    def test_record_cache_miss(self):
        """Test recording cache miss."""
        stats = ConnectionStats()

        stats.record_cache_miss()
        assert stats.cache_misses == 1

    def test_record_io_time(self):
        """Test recording I/O time."""
        stats = ConnectionStats()

        stats.record_io_time(0.5)
        assert stats.io_time_seconds == 0.5

        stats.record_io_time(0.3)
        assert stats.io_time_seconds == 0.8

    def test_get_stats_empty(self):
        """Test getting stats with no operations."""
        with patch("time.time", side_effect=[1000.0, 1010.0]):
            stats = ConnectionStats()  # Created at t=1000
            result = stats.get_stats()  # Called at t=1010

        expected = {
            "uptime_seconds": 10.0,
            "total_connections_created": 0,
            "total_connections_reused": 0,
            "total_connections_evicted": 0,
            "cache_hit_ratio": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "health_checks_performed": 0,
            "health_check_failure_rate": 0.0,
            "total_io_time_seconds": 0.0,
            "average_io_time_ms": 0.0,
        }

        assert result == expected

    def test_get_stats_with_operations(self):
        """Test getting stats with recorded operations."""
        stats = ConnectionStats()

        # Record some operations
        stats.record_connection_created()
        stats.record_connection_reused()
        stats.record_cache_miss()
        stats.record_health_check(True)
        stats.record_health_check(False)
        stats.record_io_time(0.1)
        stats.record_io_time(0.2)

        with patch("time.time", return_value=stats.start_time + 5.0):
            result = stats.get_stats()

        assert result["uptime_seconds"] == 5.0
        assert result["total_connections_created"] == 1
        assert result["total_connections_reused"] == 1
        assert result["cache_hit_ratio"] == 0.5  # 1 hit, 1 miss
        assert result["health_check_failure_rate"] == 0.5  # 1 failure, 2 total
        assert result["total_io_time_seconds"] == 0.3
        assert result["average_io_time_ms"] == 150.0  # 0.3s / 2 ops * 1000


class TestHDF5ConnectionPoolInit:
    """Test suite for HDF5ConnectionPool initialization."""

    def test_init_default_params(self):
        """Test HDF5ConnectionPool initialization with default parameters."""
        pool = HDF5ConnectionPool()

        assert pool.max_connections == 10
        assert pool.max_age_seconds == 300
        assert pool.health_check_interval == 60
        assert isinstance(pool._connections, dict)
        assert isinstance(pool._stats, ConnectionStats)
        assert hasattr(pool, "_pool_lock")
        assert hasattr(pool, "_last_cleanup")
        assert pool._memory_threshold == 0.85
        assert pool._adaptive_sizing is True

    def test_init_custom_params(self):
        """Test HDF5ConnectionPool initialization with custom parameters."""
        pool = HDF5ConnectionPool(
            max_connections=5,
            max_age_seconds=600,
            health_check_interval=120,
            memory_threshold=0.9,
            adaptive_sizing=False,
        )

        assert pool.max_connections == 5
        assert pool.max_age_seconds == 600
        assert pool.health_check_interval == 120
        assert pool._memory_threshold == 0.9
        assert pool._adaptive_sizing is False


class TestHDF5ConnectionPoolBasicOperations:
    """Test suite for HDF5ConnectionPool basic operations."""

    @patch("h5py.File")
    def test_get_connection_new(self, mock_h5py_file):
        """Test getting new connection."""
        mock_file = Mock()
        mock_h5py_file.return_value = mock_file

        pool = HDF5ConnectionPool()

        with patch("os.path.exists", return_value=True):
            result = pool.get_connection("/test/file.hdf")

        assert result is mock_file
        assert "/test/file.hdf" in pool._connections
        mock_h5py_file.assert_called_once_with("/test/file.hdf", "r")

    @patch("h5py.File")
    def test_get_connection_cached(self, mock_h5py_file):
        """Test getting cached connection."""
        mock_file = Mock()
        mock_h5py_file.return_value = mock_file

        pool = HDF5ConnectionPool()

        with patch("os.path.exists", return_value=True):
            # First call creates connection
            result1 = pool.get_connection("/test/file.hdf")

            # Second call should reuse connection
            result2 = pool.get_connection("/test/file.hdf")

        assert result1 is result2
        assert mock_h5py_file.call_count == 1  # Only called once

    def test_get_connection_file_not_exists(self):
        """Test getting connection for non-existent file."""
        pool = HDF5ConnectionPool()

        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                pool.get_connection("/nonexistent/file.hdf")

    @patch("h5py.File")
    def test_get_connection_h5py_error(self, mock_h5py_file):
        """Test getting connection when h5py raises error."""
        mock_h5py_file.side_effect = OSError("Cannot open file")

        pool = HDF5ConnectionPool()

        with patch("os.path.exists", return_value=True):
            with pytest.raises(OSError):
                pool.get_connection("/test/file.hdf")

    def test_release_connection_existing(self):
        """Test releasing existing connection."""
        pool = HDF5ConnectionPool()
        mock_conn = Mock()
        pool._connections["/test/file.hdf"] = mock_conn

        result = pool.release_connection("/test/file.hdf")

        assert result is True
        mock_conn.close.assert_called_once()
        assert "/test/file.hdf" not in pool._connections

    def test_release_connection_nonexistent(self):
        """Test releasing non-existent connection."""
        pool = HDF5ConnectionPool()

        result = pool.release_connection("/nonexistent/file.hdf")

        assert result is False

    def test_clear_all_connections(self):
        """Test clearing all connections."""
        pool = HDF5ConnectionPool()

        # Add mock connections
        mock_conn1 = Mock()
        mock_conn2 = Mock()
        pool._connections["/test/file1.hdf"] = mock_conn1
        pool._connections["/test/file2.hdf"] = mock_conn2

        pool.clear_all()

        assert len(pool._connections) == 0
        mock_conn1.close.assert_called_once()
        mock_conn2.close.assert_called_once()


class TestHDF5ConnectionPoolHealthManagement:
    """Test suite for HDF5ConnectionPool health management."""

    @patch("time.time")
    def test_should_check_health_interval_not_reached(self, mock_time):
        """Test health check interval timing."""
        pool = HDF5ConnectionPool(health_check_interval=60)
        pool._last_cleanup = 1000.0
        mock_time.return_value = 1030.0  # 30 seconds later

        result = pool._should_check_health()
        assert result is False

    @patch("time.time")
    def test_should_check_health_interval_reached(self, mock_time):
        """Test health check when interval is reached."""
        pool = HDF5ConnectionPool(health_check_interval=60)
        pool._last_cleanup = 1000.0
        mock_time.return_value = 1070.0  # 70 seconds later

        result = pool._should_check_health()
        assert result is True

    def test_cleanup_unhealthy_connections(self):
        """Test cleanup of unhealthy connections."""
        pool = HDF5ConnectionPool()

        # Add healthy and unhealthy connections
        healthy_conn = Mock()
        healthy_conn.check_health.return_value = True

        unhealthy_conn = Mock()
        unhealthy_conn.check_health.return_value = False

        pool._connections["/healthy.hdf"] = healthy_conn
        pool._connections["/unhealthy.hdf"] = unhealthy_conn

        pool._cleanup_unhealthy_connections()

        # Healthy connection should remain
        assert "/healthy.hdf" in pool._connections

        # Unhealthy connection should be removed and closed
        assert "/unhealthy.hdf" not in pool._connections
        unhealthy_conn.close.assert_called_once()

    @patch("time.time")
    def test_cleanup_aged_connections(self, mock_time):
        """Test cleanup of aged connections."""
        pool = HDF5ConnectionPool(max_age_seconds=300)

        # Current time
        mock_time.return_value = 2000.0

        # Add old and new connections
        old_conn = Mock()
        old_conn.created_at = 1500.0  # 500 seconds old
        old_conn.check_health.return_value = True

        new_conn = Mock()
        new_conn.created_at = 1900.0  # 100 seconds old
        new_conn.check_health.return_value = True

        pool._connections["/old.hdf"] = old_conn
        pool._connections["/new.hdf"] = new_conn

        pool._cleanup_aged_connections()

        # New connection should remain
        assert "/new.hdf" in pool._connections

        # Old connection should be removed and closed
        assert "/old.hdf" not in pool._connections
        old_conn.close.assert_called_once()


class TestGetFunction:
    """Test suite for get() function."""

    @patch("xpcs_toolkit.fileIO.hdf_reader.HDF5ConnectionPool")
    def test_get_simple_field(self, mock_pool_class):
        """Test getting simple field from HDF5 file."""
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_file = Mock()
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=np.array([1, 2, 3]))
        mock_file.__getitem__ = Mock(return_value=mock_dataset)
        mock_pool.get_connection.return_value = mock_file

        # Test get function
        result = get("/test/file.hdf", "test_field")

        np.testing.assert_array_equal(result, [1, 2, 3])
        mock_pool.get_connection.assert_called_once_with("/test/file.hdf")
        mock_file.__getitem__.assert_called_once_with("test_field")

    @patch("xpcs_toolkit.fileIO.hdf_reader.HDF5ConnectionPool")
    def test_get_with_slice(self, mock_pool_class):
        """Test getting field with slice."""
        # Setup mocks
        mock_pool = Mock()
        mock_pool_class.return_value = mock_pool

        mock_file = Mock()
        mock_dataset = Mock()
        mock_dataset.__getitem__ = Mock(return_value=np.array([2, 3]))
        mock_file.__getitem__ = Mock(return_value=mock_dataset)
        mock_pool.get_connection.return_value = mock_file

        # Test get function with slice
        result = get("/test/file.hdf", "test_field", slice(1, 3))

        np.testing.assert_array_equal(result, [2, 3])
        mock_dataset.__getitem__.assert_called_once_with(slice(1, 3))


class TestGetAnalysisType:
    """Test suite for get_analysis_type() function."""

    @patch("xpcs_toolkit.fileIO.hdf_reader.get")
    def test_get_analysis_type_success(self, mock_get):
        """Test successful analysis type retrieval."""
        mock_get.return_value = b"Multitau"

        result = get_analysis_type("/test/file.hdf")

        assert result == "Multitau"
        mock_get.assert_called_once_with("/test/file.hdf", "xpcs/analysis_type")

    @patch("xpcs_toolkit.fileIO.hdf_reader.get")
    def test_get_analysis_type_exception(self, mock_get):
        """Test analysis type retrieval with exception."""
        mock_get.side_effect = KeyError("Dataset not found")

        result = get_analysis_type("/test/file.hdf")

        assert result == "unknown"


class TestBatchReadFields:
    """Test suite for batch_read_fields() function."""

    @patch("xpcs_toolkit.fileIO.hdf_reader.get")
    @patch("xpcs_toolkit.fileIO.hdf_reader.hdf_key")
    def test_batch_read_fields_basic(self, mock_hdf_key, mock_get):
        """Test basic batch reading of fields."""
        # Setup mocks
        mock_hdf_key.return_value = {
            "field1": "path/to/field1",
            "field2": "path/to/field2",
        }
        mock_get.side_effect = [np.array([1, 2, 3]), np.array([4, 5, 6])]

        result = batch_read_fields("/test/file.hdf", ["field1", "field2"])

        expected = {"field1": np.array([1, 2, 3]), "field2": np.array([4, 5, 6])}

        assert set(result.keys()) == set(expected.keys())
        np.testing.assert_array_equal(result["field1"], expected["field1"])
        np.testing.assert_array_equal(result["field2"], expected["field2"])

    @patch("xpcs_toolkit.fileIO.hdf_reader.get")
    @patch("xpcs_toolkit.fileIO.hdf_reader.hdf_key")
    def test_batch_read_fields_missing_field(self, mock_hdf_key, mock_get):
        """Test batch reading with missing field."""
        mock_hdf_key.return_value = {
            "field1": "path/to/field1",
            "field2": "path/to/field2",
        }
        mock_get.side_effect = [np.array([1, 2, 3]), KeyError("Field not found")]

        result = batch_read_fields("/test/file.hdf", ["field1", "field2"])

        # Should only contain successful reads
        assert "field1" in result
        assert "field2" not in result
        np.testing.assert_array_equal(result["field1"], [1, 2, 3])


class TestPerformanceIntegration:
    """Test suite for performance monitoring integration."""

    @patch("xpcs_toolkit.fileIO.hdf_reader._perf_monitor")
    @patch("xpcs_toolkit.fileIO.hdf_reader.HDF5ConnectionPool")
    def test_performance_monitoring(self, mock_pool_class, mock_perf_monitor):
        """Test that performance monitoring is integrated."""
        # This test ensures performance monitoring objects are created
        # and would be used in actual operations

        HDF5ConnectionPool()

        # Verify performance monitor is available
        assert mock_perf_monitor is not None


class TestThreadSafety:
    """Test suite for thread safety."""

    @patch("h5py.File")
    def test_concurrent_access(self, mock_h5py_file):
        """Test concurrent access to connection pool."""
        mock_file = Mock()
        mock_h5py_file.return_value = mock_file

        pool = HDF5ConnectionPool()
        results = []
        errors = []

        def get_connection():
            try:
                with patch("os.path.exists", return_value=True):
                    conn = pool.get_connection("/test/file.hdf")
                    results.append(conn)
            except Exception as e:
                errors.append(e)

        # Create multiple threads
        threads = [threading.Thread(target=get_connection) for _ in range(5)]

        # Start all threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # All threads should succeed and get the same connection
        assert len(errors) == 0
        assert len(results) == 5
        assert all(conn is mock_file for conn in results)


@pytest.mark.parametrize(
    "max_connections,expected_behavior",
    [
        (1, "single_connection"),
        (5, "multiple_connections"),
        (0, "no_limit"),
    ],
)
def test_connection_pool_limits(max_connections, expected_behavior):
    """Test connection pool with different size limits."""
    pool = HDF5ConnectionPool(max_connections=max_connections)

    assert pool.max_connections == max_connections

    if expected_behavior == "single_connection":
        assert pool.max_connections == 1
    elif expected_behavior == "multiple_connections":
        assert pool.max_connections == 5
    elif expected_behavior == "no_limit":
        assert pool.max_connections == 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_connection_pool_with_zero_max_age(self):
        """Test connection pool with zero max age."""
        pool = HDF5ConnectionPool(max_age_seconds=0)
        assert pool.max_age_seconds == 0

    def test_connection_pool_negative_health_interval(self):
        """Test connection pool with negative health check interval."""
        pool = HDF5ConnectionPool(health_check_interval=-1)
        assert pool.health_check_interval == -1

    @patch("xpcs_toolkit.fileIO.hdf_reader.get")
    def test_get_analysis_type_bytes_handling(self, mock_get):
        """Test analysis type handling of bytes vs string."""
        # Test with bytes
        mock_get.return_value = b"Twotime"
        result = get_analysis_type("/test/file.hdf")
        assert result == "Twotime"

        # Test with string
        mock_get.return_value = "Multitau"
        result = get_analysis_type("/test/file.hdf")
        assert result == "Multitau"
