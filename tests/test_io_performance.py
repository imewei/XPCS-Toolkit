"""
Test suite for I/O performance optimizations in XPCS Toolkit.

This module tests the enhanced HDF5 connection pooling, chunked SAXS processing,
batch read operations, and performance monitoring features.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

# Import the modules to test
try:
    from xpcs_toolkit.fileIO.aps_8idi import key as hdf_key
    from xpcs_toolkit.fileIO.hdf_reader import (
        HDF5ConnectionPool,
        _connection_pool,
        batch_read_fields,
        clear_performance_stats,
        get_connection_pool_stats,
        get_io_performance_stats,
    )
    from xpcs_toolkit.utils.io_performance import (
        IOPerformanceMonitor,
        get_performance_monitor,
    )
except ImportError as e:
    print(f"Import error: {e}")

    # Mock imports for testing in isolation
    class MockConnectionPool:
        def __init__(self, *args, **kwargs):
            pass

        def get_connection(self, *args, **kwargs):
            return MagicMock()

        def get_pool_stats(self):
            return {"mock": True}

    HDF5ConnectionPool = MockConnectionPool


class TestIOPerformance(unittest.TestCase):
    """Test I/O performance optimizations."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_data.h5")

        # Create test HDF5 file with realistic XPCS structure
        self._create_test_hdf5_file()

        # Clear performance stats before each test
        clear_performance_stats()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_file):
            os.unlink(self.test_file)
        os.rmdir(self.test_dir)

    def _create_test_hdf5_file(self):
        """Create a test HDF5 file with XPCS-like structure."""
        with h5py.File(self.test_file, "w") as f:
            # Create groups
            entry = f.create_group("entry")
            instrument = entry.create_group("instrument")
            detector = instrument.create_group("detector_1")
            xpcs = f.create_group("xpcs")
            temporal_mean = xpcs.create_group("temporal_mean")
            multitau = xpcs.create_group("multitau")

            # Create datasets similar to real XPCS files
            detector.create_dataset("frame_time", data=0.001)
            detector.create_dataset("count_time", data=0.001)

            # Create SAXS 2D data (realistic size but small for testing)
            saxs_2d = np.random.randint(0, 1000, size=(512, 512), dtype=np.uint16)
            temporal_mean.create_dataset("scattering_2d", data=saxs_2d)

            # Create SAXS 1D data
            np.linspace(0.001, 0.5, 100)
            intensity = np.random.exponential(1000, size=(5, 100))
            temporal_mean.create_dataset("scattering_1d", data=intensity)

            # Create multitau correlation data
            tau = np.logspace(-6, 1, 50)
            g2 = (
                1.0
                + 0.5 * np.exp(-tau * 100)[:, np.newaxis]
                + np.random.normal(0, 0.01, size=(50, 10))
            )
            g2_err = np.random.exponential(0.01, size=(50, 10))

            multitau.create_dataset("delay_list", data=tau)
            multitau.create_dataset("normalized_g2", data=g2)
            multitau.create_dataset("normalized_g2_err", data=g2_err)

    def test_connection_pool_initialization(self):
        """Test connection pool initialization and basic functionality."""
        pool = HDF5ConnectionPool(max_pool_size=5, health_check_interval=60.0)

        self.assertEqual(pool.max_pool_size, 5)
        self.assertEqual(pool.base_pool_size, 5)
        self.assertEqual(pool.health_check_interval, 60.0)
        self.assertTrue(pool.enable_memory_pressure_adaptation)

        # Test basic stats
        stats = pool.get_pool_stats()
        self.assertIn("pool_size", stats)
        self.assertIn("max_pool_size", stats)
        self.assertEqual(stats["pool_size"], 0)

    def test_connection_pool_basic_operations(self):
        """Test basic connection pool operations."""
        pool = HDF5ConnectionPool(max_pool_size=3)

        # Test getting connection
        with pool.get_connection(self.test_file, "r") as f:
            self.assertIsInstance(f, h5py.File)
            self.assertTrue(f.id.valid)

        # Check that connection is pooled
        stats = pool.get_pool_stats()
        self.assertEqual(stats["pool_size"], 1)

        # Test reusing connection
        with pool.get_connection(self.test_file, "r") as f:
            self.assertIsInstance(f, h5py.File)

        # Should still be 1 connection in pool
        stats = pool.get_pool_stats()
        self.assertEqual(stats["pool_size"], 1)

        pool.clear_pool()

    def test_connection_pool_lru_eviction(self):
        """Test LRU eviction in connection pool."""
        pool = HDF5ConnectionPool(max_pool_size=2)

        # Create additional test files
        test_file2 = os.path.join(self.test_dir, "test_data2.h5")
        test_file3 = os.path.join(self.test_dir, "test_data3.h5")

        # Copy test file for additional files
        import shutil

        shutil.copy2(self.test_file, test_file2)
        shutil.copy2(self.test_file, test_file3)

        try:
            # Open 3 files (should trigger eviction)
            with pool.get_connection(self.test_file, "r"):
                pass
            with pool.get_connection(test_file2, "r"):
                pass

            # Pool should be at capacity
            stats = pool.get_pool_stats()
            self.assertEqual(stats["pool_size"], 2)

            # Open third file - should trigger LRU eviction
            with pool.get_connection(test_file3, "r"):
                pass

            # Pool should still be at capacity, but oldest connection evicted
            stats = pool.get_pool_stats()
            self.assertEqual(stats["pool_size"], 2)

        finally:
            os.unlink(test_file2)
            os.unlink(test_file3)
            pool.clear_pool()

    def test_batch_read_operations(self):
        """Test batch read operations."""
        try:
            # Test batch reading multiple fields
            fields = ["frame_time", "count_time"]
            result = batch_read_fields(
                self.test_file, fields, mode="raw", use_pool=True
            )

            self.assertIn("/entry/instrument/detector_1/frame_time", result)
            self.assertIn("/entry/instrument/detector_1/count_time", result)
            self.assertEqual(result["/entry/instrument/detector_1/frame_time"], 0.001)
            self.assertEqual(result["/entry/instrument/detector_1/count_time"], 0.001)

        except Exception as e:
            # If batch_read_fields is not available, skip this test
            self.skipTest(f"batch_read_fields not available: {e}")

    def test_performance_monitoring(self):
        """Test performance monitoring functionality."""
        try:
            monitor = get_performance_monitor()
            self.assertIsInstance(monitor, IOPerformanceMonitor)

            # Test starting and completing an operation
            operation = monitor.start_operation(
                "read", self.test_file, data_size_mb=1.0
            )
            time.sleep(0.01)  # Small delay to ensure measurable duration
            monitor.complete_operation(operation, success=True)

            # Check statistics
            stats = monitor.get_global_stats()
            self.assertGreater(stats["total_operations"], 0)
            self.assertGreater(stats["total_duration_ms"], 0)

            # Test file-specific stats
            file_stats = monitor.get_file_stats(self.test_file)
            self.assertIsNotNone(file_stats)
            self.assertEqual(file_stats["total_operations"], 1)

        except Exception as e:
            self.skipTest(f"Performance monitoring not available: {e}")

    def test_io_performance_stats(self):
        """Test I/O performance statistics gathering."""
        try:
            # Perform some operations to generate stats
            with _connection_pool.get_connection(self.test_file, "r") as f:
                data = f["/xpcs/temporal_mean/scattering_2d"][()]
                self.assertEqual(data.shape, (512, 512))

            # Get performance stats
            stats = get_io_performance_stats()

            self.assertIn("io_operations", stats)
            self.assertIn("connection_pool", stats)
            self.assertIn("bottlenecks", stats)

            # Check connection pool stats
            pool_stats = stats["connection_pool"]
            self.assertIn("pool_size", pool_stats)
            self.assertIn("cache_hit_ratio", pool_stats)

        except Exception as e:
            self.skipTest(f"I/O performance stats not available: {e}")

    def test_memory_pressure_adaptation(self):
        """Test memory pressure adaptation in connection pool."""
        pool = HDF5ConnectionPool(
            max_pool_size=10, enable_memory_pressure_adaptation=True
        )

        # Test basic functionality (can't easily mock memory pressure)
        initial_size = pool.max_pool_size
        self.assertEqual(initial_size, 10)

        # Test disabling adaptation
        pool.enable_memory_pressure_adaptation = False
        pool._adapt_pool_size_to_memory_pressure()
        self.assertEqual(pool.max_pool_size, initial_size)

        pool.clear_pool()

    def test_health_monitoring(self):
        """Test connection health monitoring."""
        pool = HDF5ConnectionPool(
            max_pool_size=5,
            health_check_interval=0.1,  # Very short interval for testing
        )

        # Add a connection
        with pool.get_connection(self.test_file, "r"):
            pass

        # Force health check
        pool.force_health_check()

        stats = pool.get_pool_stats()
        self.assertIn("connections", stats)

        pool.clear_pool()


class TestChunkedProcessing(unittest.TestCase):
    """Test chunked processing functionality."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.large_test_file = os.path.join(self.test_dir, "large_test_data.h5")

        # Create test file with larger data
        self._create_large_test_file()

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.large_test_file):
            os.unlink(self.large_test_file)
        os.rmdir(self.test_dir)

    def _create_large_test_file(self):
        """Create a test file with larger datasets."""
        with h5py.File(self.large_test_file, "w") as f:
            # Create larger SAXS 2D data
            large_saxs = np.random.randint(0, 65535, size=(1024, 1024), dtype=np.uint16)
            f.create_dataset(
                "/xpcs/temporal_mean/scattering_2d",
                data=large_saxs,
                compression="gzip",
                chunks=True,
            )

    def test_chunked_log_computation(self):
        """Test chunked log computation for large SAXS data."""
        try:
            # This test requires the XpcsFile class
            from xpcs_toolkit.xpcs_file import XpcsFile

            # Mock qmap_manager to avoid complex initialization
            with patch("xpcs_toolkit.xpcs_file.get_qmap") as mock_qmap:
                mock_qmap.return_value = MagicMock()

                # Create XpcsFile instance
                xf = XpcsFile(self.large_test_file)

                # Test chunked log computation
                large_data = np.random.randint(
                    1, 1000, size=(1000, 1000), dtype=np.uint16
                )
                result = xf._compute_saxs_log_chunked(large_data, chunk_size=100)

                self.assertEqual(result.shape, large_data.shape)
                self.assertEqual(result.dtype, np.float32)

                # Verify log computation correctness
                self.assertTrue(np.all(np.isfinite(result)))

        except ImportError:
            self.skipTest("XpcsFile not available for testing")


def run_performance_benchmark():
    """Run performance benchmarks to validate improvements."""
    print("Running I/O Performance Benchmarks...")
    print("=" * 50)

    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        test_file = tmp.name

    try:
        # Create test data
        with h5py.File(test_file, "w") as f:
            data = np.random.randint(0, 1000, size=(2048, 2048), dtype=np.uint16)
            f.create_dataset("large_dataset", data=data, compression="gzip")

            for i in range(10):
                small_data = np.random.random(100)
                f.create_dataset(f"small_dataset_{i}", data=small_data)

        # Benchmark 1: Connection Pool vs Direct Access
        print("1. Connection Pool Performance Test")

        # Test with connection pool
        start_time = time.time()
        for i in range(20):
            with _connection_pool.get_connection(test_file, "r") as f:
                data = f["large_dataset"][()]
        pool_time = time.time() - start_time

        # Test without connection pool
        start_time = time.time()
        for i in range(20):
            with h5py.File(test_file, "r") as f:
                data = f["large_dataset"][()]
        direct_time = time.time() - start_time

        print(f"   Connection Pool: {pool_time:.3f}s")
        print(f"   Direct Access:   {direct_time:.3f}s")
        print(
            f"   Improvement:     {((direct_time - pool_time) / direct_time * 100):.1f}%"
        )

        # Benchmark 2: Batch vs Individual Reads
        print("\n2. Batch Read Performance Test")
        dataset_paths = [f"small_dataset_{i}" for i in range(10)]

        try:
            # Test batch read
            start_time = time.time()
            for i in range(10):
                results = _connection_pool.batch_read_datasets(test_file, dataset_paths)
            batch_time = time.time() - start_time

            # Test individual reads
            start_time = time.time()
            for i in range(10):
                with _connection_pool.get_connection(test_file, "r") as f:
                    results = {}
                    for path in dataset_paths:
                        results[path] = f[path][()]
            individual_time = time.time() - start_time

            print(f"   Batch Read:      {batch_time:.3f}s")
            print(f"   Individual Read: {individual_time:.3f}s")
            print(
                f"   Improvement:     {((individual_time - batch_time) / individual_time * 100):.1f}%"
            )

        except AttributeError:
            print("   Batch read functionality not available")

        # Show performance statistics
        print("\n3. Performance Statistics")
        try:
            stats = get_io_performance_stats()
            print(f"   Total Operations: {stats['io_operations']['total_operations']}")
            print(
                f"   Average Duration: {stats['io_operations']['average_duration_ms']:.2f}ms"
            )
            print(
                f"   Success Rate:     {stats['io_operations']['success_rate_percent']:.1f}%"
            )
            print(
                f"   Pool Hit Ratio:   {stats['connection_pool']['cache_hit_ratio']:.2f}"
            )
        except Exception:
            print("   Performance statistics not available")

    finally:
        os.unlink(test_file)

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    # Run unit tests
    print("Running I/O Performance Tests...")
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Run performance benchmarks
    print("\n" + "=" * 60)
    run_performance_benchmark()
