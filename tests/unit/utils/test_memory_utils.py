"""Unit tests for memory utilities module.

This module provides comprehensive unit tests for memory monitoring,
optimization utilities, and memory-efficient array operations.
"""

import threading
import time
from unittest.mock import Mock, patch

import psutil
import pytest

from xpcs_toolkit.utils.memory_utils import (
    CachedMemoryMonitor,
    MemoryStatus,
    MemoryTracker,
    get_cached_memory_monitor,
)


class TestMemoryStatus:
    """Test suite for MemoryStatus NamedTuple."""

    def test_memory_status_creation(self):
        """Test MemoryStatus creation and field access."""
        status = MemoryStatus(
            used_mb=1024.0,
            available_mb=2048.0,
            percent_used=0.33,
            timestamp=1234567890.0,
        )

        assert status.used_mb == 1024.0
        assert status.available_mb == 2048.0
        assert status.percent_used == 0.33
        assert status.timestamp == 1234567890.0

    def test_memory_status_immutable(self):
        """Test that MemoryStatus is immutable (NamedTuple property)."""
        status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)

        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            status.used_mb = 150.0

    def test_memory_status_equality(self):
        """Test MemoryStatus equality comparison."""
        status1 = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
        status2 = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
        status3 = MemoryStatus(150.0, 200.0, 0.43, 1000.0)

        assert status1 == status2
        assert status1 != status3

    def test_memory_status_tuple_behavior(self):
        """Test that MemoryStatus behaves like a tuple."""
        status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)

        # Can be unpacked
        used, available, percent, timestamp = status
        assert used == 100.0
        assert available == 200.0
        assert percent == 0.33
        assert timestamp == 1000.0

        # Can be indexed
        assert status[0] == 100.0
        assert status[1] == 200.0
        assert status[2] == 0.33
        assert status[3] == 1000.0

        # Has length
        assert len(status) == 4


class TestCachedMemoryMonitorInit:
    """Test suite for CachedMemoryMonitor initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor()

            assert monitor._cache_ttl == 5.0
            assert monitor._cleanup_threshold == 0.85
            assert monitor._cleanup_stop_threshold == 0.75
            assert monitor._background_update is True
            assert monitor._cached_status is None
            assert monitor._cache_hits == 0
            assert monitor._cache_misses == 0
            assert hasattr(monitor, "_lock")
            assert isinstance(monitor._lock, threading.Lock)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(
                cache_ttl_seconds=10.0,
                cleanup_threshold=0.9,
                cleanup_stop_threshold=0.8,
                background_update=False,
            )

            assert monitor._cache_ttl == 10.0
            assert monitor._cleanup_threshold == 0.9
            assert monitor._cleanup_stop_threshold == 0.8
            assert monitor._background_update is False

    @patch(
        "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
    )
    def test_background_monitoring_control(self, mock_start_monitoring):
        """Test background monitoring start control."""
        # With background_update=True
        CachedMemoryMonitor(background_update=True)
        mock_start_monitoring.assert_called_once()

        mock_start_monitoring.reset_mock()

        # With background_update=False
        CachedMemoryMonitor(background_update=False)
        mock_start_monitoring.assert_not_called()


class TestCachedMemoryMonitorCore:
    """Test suite for CachedMemoryMonitor core functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable background monitoring for tests
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            self.monitor = CachedMemoryMonitor(background_update=False)

    @patch("psutil.virtual_memory")
    @patch("time.time")
    def test_get_fresh_memory_status(self, mock_time, mock_virtual_memory):
        """Test getting fresh memory status from psutil."""
        # Setup mocks
        mock_time.return_value = 1234567890.0
        mock_memory = Mock()
        mock_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB available
        mock_memory.percent = 50.0  # 50% used
        mock_virtual_memory.return_value = mock_memory

        status = self.monitor._get_fresh_memory_status()

        assert isinstance(status, MemoryStatus)
        assert status.used_mb == 4096.0  # 4GB used in MB
        assert status.available_mb == 4096.0  # 4GB available in MB
        assert status.percent_used == 0.5  # 50% as fraction
        assert status.timestamp == 1234567890.0

    @patch("time.time")
    def test_is_cache_valid(self, mock_time):
        """Test cache validity checking."""
        mock_time.return_value = 1000.0

        # Valid cache (within TTL)
        valid_status = MemoryStatus(100.0, 200.0, 0.33, 997.0)  # 3 seconds old
        assert self.monitor._is_cache_valid(valid_status) is True

        # Invalid cache (beyond TTL)
        invalid_status = MemoryStatus(100.0, 200.0, 0.33, 990.0)  # 10 seconds old
        assert self.monitor._is_cache_valid(invalid_status) is False

    @patch(
        "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._get_fresh_memory_status"
    )
    def test_get_memory_status_cache_miss(self, mock_get_fresh):
        """Test get_memory_status with cache miss."""
        mock_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
        mock_get_fresh.return_value = mock_status

        result = self.monitor.get_memory_status()

        assert result == mock_status
        assert self.monitor._cached_status == mock_status
        assert self.monitor._cache_misses == 1
        assert self.monitor._cache_hits == 0
        mock_get_fresh.assert_called_once()

    @patch("xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._is_cache_valid")
    def test_get_memory_status_cache_hit(self, mock_is_valid):
        """Test get_memory_status with cache hit."""
        # Setup cached status
        cached_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
        self.monitor._cached_status = cached_status
        mock_is_valid.return_value = True

        result = self.monitor.get_memory_status()

        assert result == cached_status
        assert self.monitor._cache_hits == 1
        assert self.monitor._cache_misses == 0
        mock_is_valid.assert_called_once_with(cached_status)

    @patch(
        "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._get_fresh_memory_status"
    )
    @patch("xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._is_cache_valid")
    def test_get_memory_status_cache_invalid(self, mock_is_valid, mock_get_fresh):
        """Test get_memory_status with invalid cache."""
        # Setup invalid cached status
        old_status = MemoryStatus(100.0, 200.0, 0.33, 500.0)
        new_status = MemoryStatus(150.0, 250.0, 0.4, 1000.0)
        self.monitor._cached_status = old_status
        mock_is_valid.return_value = False
        mock_get_fresh.return_value = new_status

        result = self.monitor.get_memory_status()

        assert result == new_status
        assert self.monitor._cached_status == new_status
        assert self.monitor._cache_misses == 1
        mock_get_fresh.assert_called_once()

    def test_get_memory_info(self):
        """Test get_memory_info method."""
        with patch.object(self.monitor, "get_memory_status") as mock_get_status:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
            mock_get_status.return_value = mock_status

            used, available, percent = self.monitor.get_memory_info()

            assert used == 100.0
            assert available == 200.0
            assert percent == 0.33

    def test_is_memory_pressure_high_true(self):
        """Test memory pressure detection when pressure is high."""
        with patch.object(self.monitor, "get_memory_status") as mock_get_status:
            high_pressure_status = MemoryStatus(900.0, 100.0, 0.9, 1000.0)
            mock_get_status.return_value = high_pressure_status

            assert self.monitor.is_memory_pressure_high(threshold=0.85) is True

    def test_is_memory_pressure_high_false(self):
        """Test memory pressure detection when pressure is low."""
        with patch.object(self.monitor, "get_memory_status") as mock_get_status:
            low_pressure_status = MemoryStatus(200.0, 800.0, 0.2, 1000.0)
            mock_get_status.return_value = low_pressure_status

            assert self.monitor.is_memory_pressure_high(threshold=0.85) is False

    def test_is_memory_pressure_high_default_threshold(self):
        """Test memory pressure with default threshold."""
        with patch.object(self.monitor, "get_memory_status") as mock_get_status:
            # Use monitor's configured threshold
            status = MemoryStatus(900.0, 100.0, 0.9, 1000.0)
            mock_get_status.return_value = status

            # Should use monitor's default threshold (0.85)
            assert self.monitor.is_memory_pressure_high() is True


class TestCachedMemoryMonitorStats:
    """Test suite for CachedMemoryMonitor statistics tracking."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            self.monitor = CachedMemoryMonitor(background_update=False)

    def test_cache_stats_tracking(self):
        """Test cache hit/miss statistics tracking."""
        with (
            patch.object(self.monitor, "_get_fresh_memory_status") as mock_get_fresh,
            patch.object(self.monitor, "_is_cache_valid") as mock_is_valid,
        ):
            mock_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
            mock_get_fresh.return_value = mock_status

            # First call - cache miss
            mock_is_valid.return_value = False
            self.monitor.get_memory_status()
            assert self.monitor._cache_misses == 1
            assert self.monitor._cache_hits == 0

            # Second call - cache hit
            mock_is_valid.return_value = True
            self.monitor.get_memory_status()
            assert self.monitor._cache_misses == 1
            assert self.monitor._cache_hits == 1

    def test_get_cache_stats(self):
        """Test get_cache_stats method."""
        # Manually set stats
        self.monitor._cache_hits = 10
        self.monitor._cache_misses = 3

        stats = self.monitor.get_cache_stats()

        expected_stats = {
            "cache_hits": 10,
            "cache_misses": 3,
            "hit_rate": 10 / 13,  # 10 hits out of 13 total
            "ttl_seconds": 5.0,
            "cleanup_threshold": 0.85,
            "cleanup_stop_threshold": 0.75,
        }

        assert stats == expected_stats

    def test_get_cache_stats_no_requests(self):
        """Test get_cache_stats with no requests."""
        stats = self.monitor.get_cache_stats()

        expected_stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "hit_rate": 0.0,
            "ttl_seconds": 5.0,
            "cleanup_threshold": 0.85,
            "cleanup_stop_threshold": 0.75,
        }

        assert stats == expected_stats


class TestCachedMemoryMonitorThreadSafety:
    """Test suite for CachedMemoryMonitor thread safety."""

    def setup_method(self):
        """Set up test fixtures."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            self.monitor = CachedMemoryMonitor(background_update=False)

    def test_concurrent_access(self):
        """Test concurrent access to memory monitor."""
        results = []

        def get_memory_multiple_times():
            for _ in range(10):
                status = self.monitor.get_memory_status()
                results.append(status)
                time.sleep(0.001)  # Small delay

        with patch.object(self.monitor, "_get_fresh_memory_status") as mock_get_fresh:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
            mock_get_fresh.return_value = mock_status

            threads = [
                threading.Thread(target=get_memory_multiple_times) for _ in range(5)
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

        # Should have results from all threads
        assert len(results) == 50  # 5 threads * 10 calls each

        # All results should be MemoryStatus instances
        assert all(isinstance(result, MemoryStatus) for result in results)

    def test_lock_acquisition(self):
        """Test that lock is properly acquired and released."""
        with patch.object(self.monitor, "_get_fresh_memory_status") as mock_get_fresh:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, 1000.0)
            mock_get_fresh.return_value = mock_status

            # Test that lock is acquired during operation
            original_acquire = self.monitor._lock.acquire
            original_release = self.monitor._lock.release

            acquire_calls = []
            release_calls = []

            def track_acquire(*args, **kwargs):
                acquire_calls.append(time.time())
                return original_acquire(*args, **kwargs)

            def track_release(*args, **kwargs):
                release_calls.append(time.time())
                return original_release(*args, **kwargs)

            self.monitor._lock.acquire = track_acquire
            self.monitor._lock.release = track_release

            self.monitor.get_memory_status()

            assert len(acquire_calls) == 1
            assert len(release_calls) == 1


class TestMemoryTracker:
    """Test suite for MemoryTracker class."""

    def test_memory_tracker_exists(self):
        """Test that MemoryTracker class exists and can be imported."""
        assert MemoryTracker is not None
        assert hasattr(MemoryTracker, "__init__")

    def test_memory_tracker_basic_functionality(self):
        """Test basic MemoryTracker functionality."""
        tracker = MemoryTracker()

        # Should have basic memory tracking methods
        assert hasattr(tracker, "start_tracking")
        assert hasattr(tracker, "stop_tracking")
        assert hasattr(tracker, "get_peak_memory")


class TestGetCachedMemoryMonitor:
    """Test suite for get_cached_memory_monitor function."""

    def test_get_cached_memory_monitor_singleton(self):
        """Test that get_cached_memory_monitor returns singleton."""
        monitor1 = get_cached_memory_monitor()
        monitor2 = get_cached_memory_monitor()

        assert monitor1 is monitor2
        assert isinstance(monitor1, CachedMemoryMonitor)

    def test_get_cached_memory_monitor_thread_safe(self):
        """Test thread-safe singleton creation."""
        monitors = []

        def create_monitor():
            monitors.append(get_cached_memory_monitor())

        threads = [threading.Thread(target=create_monitor) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All should be the same instance
        assert all(monitor is monitors[0] for monitor in monitors)
        assert len({id(monitor) for monitor in monitors}) == 1


class TestCachedMemoryMonitorPerformance:
    """Test suite for CachedMemoryMonitor performance characteristics."""

    def test_cache_performance_benefit(self, performance_timer):
        """Test that caching provides performance benefit."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(
                cache_ttl_seconds=10.0, background_update=False
            )

        with patch.object(monitor, "_get_fresh_memory_status") as mock_get_fresh:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, time.time())
            mock_get_fresh.return_value = mock_status

            # First call - cache miss (should call _get_fresh_memory_status)
            performance_timer.start()
            status1 = monitor.get_memory_status()
            first_call_time = performance_timer.stop()

            # Subsequent calls - cache hits (should not call _get_fresh_memory_status)
            performance_timer.start()
            for _ in range(100):
                status = monitor.get_memory_status()
                assert status == status1
            cached_calls_time = performance_timer.stop()

            # Cached calls should be much faster
            avg_cached_time = cached_calls_time / 100
            assert avg_cached_time < first_call_time / 10  # At least 10x faster

            # Should only call _get_fresh_memory_status once
            assert mock_get_fresh.call_count == 1

    def test_memory_monitor_overhead(self, performance_timer):
        """Test memory monitoring overhead is minimal."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(background_update=False)

        performance_timer.start()

        for _ in range(100):
            monitor.get_memory_status()

        elapsed = performance_timer.stop()

        # Should complete quickly
        assert elapsed < 1.0  # Less than 1 second for 100 calls
        avg_time_per_call = elapsed / 100
        assert avg_time_per_call < 0.01  # Less than 10ms per call


class TestCachedMemoryMonitorEdgeCases:
    """Test suite for CachedMemoryMonitor edge cases."""

    def test_zero_cache_ttl(self):
        """Test behavior with zero cache TTL."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(
                cache_ttl_seconds=0.0, background_update=False
            )

        with patch.object(monitor, "_get_fresh_memory_status") as mock_get_fresh:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, time.time())
            mock_get_fresh.return_value = mock_status

            # Every call should be a cache miss with zero TTL
            monitor.get_memory_status()
            monitor.get_memory_status()

            assert mock_get_fresh.call_count == 2
            assert monitor._cache_misses == 2
            assert monitor._cache_hits == 0

    def test_negative_cache_ttl(self):
        """Test behavior with negative cache TTL."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(
                cache_ttl_seconds=-1.0, background_update=False
            )

        # Should handle gracefully (cache always invalid)
        with patch.object(monitor, "_get_fresh_memory_status") as mock_get_fresh:
            mock_status = MemoryStatus(100.0, 200.0, 0.33, time.time())
            mock_get_fresh.return_value = mock_status

            monitor.get_memory_status()
            monitor.get_memory_status()

            # Should always refresh with negative TTL
            assert mock_get_fresh.call_count == 2

    def test_extreme_memory_values(self):
        """Test handling of extreme memory values."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(background_update=False)

        with patch("psutil.virtual_memory") as mock_virtual_memory:
            # Test very large memory values
            mock_memory = Mock()
            mock_memory.total = 1024 * 1024 * 1024 * 1024  # 1TB
            mock_memory.available = 512 * 1024 * 1024 * 1024  # 512GB
            mock_memory.percent = 50.0
            mock_virtual_memory.return_value = mock_memory

            status = monitor._get_fresh_memory_status()

            assert status.used_mb > 500000  # Should be in hundreds of thousands MB
            assert status.available_mb > 500000
            assert 0 <= status.percent_used <= 1.0

    @patch("psutil.virtual_memory")
    def test_psutil_error_handling(self, mock_virtual_memory):
        """Test handling of psutil errors."""
        with patch(
            "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
        ):
            monitor = CachedMemoryMonitor(background_update=False)

        mock_virtual_memory.side_effect = psutil.Error("System error")

        # Should propagate psutil errors
        with pytest.raises(psutil.Error):
            monitor._get_fresh_memory_status()


@pytest.mark.parametrize(
    "cache_ttl,expected_behavior",
    [
        (0.1, "frequent_refresh"),
        (5.0, "normal_caching"),
        (60.0, "long_caching"),
        (0.0, "no_caching"),
    ],
)
def test_cache_ttl_variations(cache_ttl, expected_behavior):
    """Test CachedMemoryMonitor with different cache TTL values."""
    with patch(
        "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
    ):
        monitor = CachedMemoryMonitor(
            cache_ttl_seconds=cache_ttl, background_update=False
        )

    assert monitor._cache_ttl == cache_ttl

    if expected_behavior == "no_caching":
        assert monitor._cache_ttl == 0.0
    elif expected_behavior == "frequent_refresh":
        assert monitor._cache_ttl < 1.0
    elif expected_behavior == "normal_caching":
        assert 1.0 <= monitor._cache_ttl <= 10.0
    elif expected_behavior == "long_caching":
        assert monitor._cache_ttl > 10.0


@pytest.mark.parametrize(
    "cleanup_threshold,cleanup_stop,expected_valid",
    [
        (0.8, 0.7, True),  # Stop threshold < cleanup threshold (valid)
        (0.9, 0.85, True),  # Stop threshold < cleanup threshold (valid)
        (0.7, 0.8, False),  # Stop threshold > cleanup threshold (invalid)
        (0.85, 0.85, False),  # Stop threshold = cleanup threshold (edge case)
    ],
)
def test_cleanup_thresholds(cleanup_threshold, cleanup_stop, expected_valid):
    """Test cleanup threshold validation."""
    with patch(
        "xpcs_toolkit.utils.memory_utils.CachedMemoryMonitor._start_background_monitoring"
    ):
        monitor = CachedMemoryMonitor(
            cleanup_threshold=cleanup_threshold,
            cleanup_stop_threshold=cleanup_stop,
            background_update=False,
        )

    assert monitor._cleanup_threshold == cleanup_threshold
    assert monitor._cleanup_stop_threshold == cleanup_stop

    # Hysteresis should be valid (stop < cleanup)
    if expected_valid:
        assert monitor._cleanup_stop_threshold < monitor._cleanup_threshold
    else:
        assert monitor._cleanup_stop_threshold >= monitor._cleanup_threshold
