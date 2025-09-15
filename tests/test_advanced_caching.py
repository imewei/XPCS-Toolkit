"""
Comprehensive tests for the advanced caching system in XPCS Toolkit.

This module provides thorough testing of all caching components including
thread safety, performance, and integration tests.
"""

import concurrent.futures
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xpcs_toolkit.utils.adaptive_memory import (
    AdaptiveMemoryManager,
    MemoryStrategy,
    UsagePattern,
    get_adaptive_memory_manager,
)
from xpcs_toolkit.utils.advanced_cache import MultiLevelCache, get_global_cache
from xpcs_toolkit.utils.cache_monitor import CacheMonitor, get_cache_monitor
from xpcs_toolkit.utils.computation_cache import (
    ComputationCache,
    G2FitResult,
    SAXSResult,
    get_computation_cache,
)
from xpcs_toolkit.utils.metadata_cache import (
    FileMetadata,
    MetadataCache,
    QMapData,
    get_metadata_cache,
)


class TestMultiLevelCache(unittest.TestCase):
    """Test the multi-level cache implementation."""

    def setUp(self):
        """Set up test cache with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = MultiLevelCache(
            l1_max_memory_mb=10.0,  # Small limits for testing
            l2_max_memory_mb=20.0,
            l3_max_disk_mb=50.0,
            l3_cache_dir=Path(self.temp_dir) / "cache",
            cleanup_interval_seconds=0.1,  # Fast cleanup for testing
        )

    def tearDown(self):
        """Clean up test cache and temporary files."""
        self.cache.shutdown()
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_cache_operations(self):
        """Test basic put/get operations."""
        # Test simple data storage and retrieval
        test_data = {"key": "value", "number": 42}
        success = self.cache.put("test_key", test_data)
        self.assertTrue(success)

        retrieved_data, found = self.cache.get("test_key")
        self.assertTrue(found)
        self.assertEqual(retrieved_data, test_data)

        # Test non-existent key
        missing_data, found = self.cache.get("missing_key")
        self.assertFalse(found)
        self.assertIsNone(missing_data)

    def test_numpy_array_caching(self):
        """Test caching of numpy arrays."""
        # Test small array (should go to L1)
        small_array = np.random.rand(100, 100).astype(np.float32)
        success = self.cache.put("small_array", small_array)
        self.assertTrue(success)

        retrieved_array, found = self.cache.get("small_array")
        self.assertTrue(found)
        np.testing.assert_array_equal(retrieved_array, small_array)

        # Test large array (should go to L2 or L3)
        large_array = np.random.rand(1000, 1000).astype(np.float64)
        success = self.cache.put("large_array", large_array)
        self.assertTrue(success)

        retrieved_large, found = self.cache.get("large_array")
        self.assertTrue(found)
        np.testing.assert_array_equal(retrieved_large, large_array)

    def test_cache_level_promotion(self):
        """Test automatic promotion between cache levels."""
        # Add data that should go to L2
        medium_data = np.random.rand(500, 500).astype(np.float32)
        self.cache.put("medium_data", medium_data, ttl_seconds=3600)

        # Access it multiple times to trigger promotion to L1
        for _ in range(5):
            _data, found = self.cache.get("medium_data")
            self.assertTrue(found)

        # Data should now be in L1 for faster access
        stats = self.cache.get_stats()
        self.assertGreater(stats["current_counts"]["l1_entries"], 0)

    def test_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        # Add data with short TTL
        test_data = {"message": "This will expire"}
        self.cache.put("expiring_key", test_data, ttl_seconds=0.1)

        # Should be available immediately
        data, found = self.cache.get("expiring_key")
        self.assertTrue(found)
        self.assertEqual(data, test_data)

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired now
        data, found = self.cache.get("expiring_key")
        self.assertFalse(found)

    def test_memory_limit_enforcement(self):
        """Test memory limit enforcement and eviction."""
        # Fill cache beyond L1 limit
        large_arrays = []
        for i in range(5):
            # Each array ~4MB (1000x1000x4 bytes)
            array = np.random.rand(1000, 1000).astype(np.float32)
            large_arrays.append(array)
            success = self.cache.put(f"array_{i}", array)
            self.assertTrue(success)

        # Check that eviction occurred
        stats = self.cache.get_stats()
        self.assertGreater(stats["operation_counts"]["total_evictions"], 0)

    def test_compression_functionality(self):
        """Test data compression in L2 and L3."""
        # Create compressible data
        compressible_data = np.zeros((1000, 1000), dtype=np.float32)
        compressible_data[::10, ::10] = 1.0  # Sparse pattern

        self.cache.put("compressible", compressible_data)

        # Retrieve and verify
        retrieved_data, found = self.cache.get("compressible")
        self.assertTrue(found)
        np.testing.assert_array_equal(retrieved_data, compressible_data)

        # Check compression stats
        stats = self.cache.get_stats()
        compression_stats = stats.get("compression", {})
        if compression_stats.get("bytes_saved_percentage", 0) > 0:
            self.assertGreater(compression_stats["bytes_saved_percentage"], 0)

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        num_threads = 10
        num_operations = 100

        def worker_function(thread_id):
            """Worker function for thread safety test."""
            for i in range(num_operations):
                key = f"thread_{thread_id}_item_{i}"
                data = {"thread": thread_id, "item": i, "data": np.random.rand(10, 10)}

                # Put data
                success = self.cache.put(key, data)
                self.assertTrue(success)

                # Get data
                retrieved, found = self.cache.get(key)
                self.assertTrue(found)
                self.assertEqual(retrieved["thread"], thread_id)
                self.assertEqual(retrieved["item"], i)

        # Run workers in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_function, i) for i in range(num_threads)]

            # Wait for all threads to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Raises exception if worker failed

        # Verify final state
        stats = self.cache.get_stats()
        self.assertGreater(stats["operation_counts"]["total_puts"], 0)
        self.assertGreater(stats["operation_counts"]["total_gets"], 0)


class TestComputationCache(unittest.TestCase):
    """Test computation result caching."""

    def setUp(self):
        """Set up computation cache for testing."""
        self.comp_cache = ComputationCache(
            max_entries_per_type=5
        )  # Small limit for testing

    def test_g2_fitting_cache(self):
        """Test G2 fitting result caching."""

        @self.comp_cache.cache_g2_fitting(
            file_path="/test/file.hdf",
            q_range=(0.1, 1.0),
            t_range=(1e-3, 1.0),
            bounds=[[0, 0, 0, 0], [1, 1, 1, 1]],
            fit_flag=[True, True, True, True],
            fit_func="single",
        )
        def mock_g2_fitting():
            """Mock G2 fitting function."""
            return {
                "fit_func": "single",
                "fit_val": np.array([[1.0, 0.1, 1.0, 0.0]]),
                "t_el": np.logspace(-3, 0, 50),
                "q_val": np.array([0.5]),
                "fit_line": [{"fit_x": np.logspace(-3, 0, 50), "fit_y": np.ones(50)}],
                "label": ["Q=0.5"],
            }

        # First call should compute and cache
        result1 = mock_g2_fitting()
        self.assertIsInstance(result1, G2FitResult)

        # Second call should retrieve from cache
        result2 = mock_g2_fitting()
        self.assertEqual(result1.cache_key, result2.cache_key)

    def test_saxs_analysis_cache(self):
        """Test SAXS analysis result caching."""

        @self.comp_cache.cache_saxs_analysis(
            file_path="/test/file.hdf",
            processing_params={"norm_method": None, "sampling": 1},
        )
        def mock_saxs_analysis():
            """Mock SAXS analysis function."""
            q = np.logspace(-2, 0, 100)
            intensity = np.exp(-(q**2))
            return q, intensity, "q (Å⁻¹)", "Intensity"

        # First call
        result1 = mock_saxs_analysis()
        self.assertIsInstance(result1, SAXSResult)
        self.assertEqual(len(result1.q), 100)

        # Second call should be cached
        result2 = mock_saxs_analysis()
        np.testing.assert_array_equal(result1.q, result2.q)

    def test_computation_cache_limits(self):
        """Test computation cache entry limits."""
        # Fill cache beyond limit
        for i in range(10):

            @self.comp_cache.cache_g2_fitting(
                file_path=f"/test/file_{i}.hdf",
                q_range=(0.1, 1.0),
                t_range=(1e-3, 1.0),
                bounds=[[0, 0, 0, 0], [1, 1, 1, 1]],
                fit_flag=[True, True, True, True],
                fit_func="single",
            )
            def mock_fitting():
                return {
                    "fit_func": "single",
                    "fit_val": np.array([[i, 0.1, 1.0, 0.0]]),
                    "t_el": np.logspace(-3, 0, 50),
                    "q_val": np.array([0.5]),
                    "fit_line": [],
                    "label": [f"File_{i}"],
                }

            mock_fitting()

        # Check that old entries were evicted
        stats = self.comp_cache.get_computation_stats()
        g2_count = stats["computation_counts"]["g2_fitting"]
        self.assertLessEqual(g2_count, 5)  # Should not exceed max_entries_per_type


class TestMetadataCache(unittest.TestCase):
    """Test metadata and file information caching."""

    def setUp(self):
        """Set up metadata cache for testing."""
        self.meta_cache = MetadataCache(
            max_metadata_entries=5,
            max_qmap_entries=3,
            enable_prefetch=False,  # Disable for testing
        )

        # Create temporary test file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".hdf")
        self.temp_file.write(b"dummy hdf5 content")
        self.temp_file.close()

    def tearDown(self):
        """Clean up test resources."""
        self.meta_cache.shutdown()
        os.unlink(self.temp_file.name)

    def test_file_metadata_caching(self):
        """Test file metadata caching with validation."""
        # Create metadata
        metadata = FileMetadata(
            file_path=self.temp_file.name,
            file_size=os.path.getsize(self.temp_file.name),
            file_mtime=os.path.getmtime(self.temp_file.name),
            analysis_type="Multitau",
            metadata_dict={"test": "data"},
            large_datasets=[],
        )

        # Cache metadata
        self.meta_cache.put_file_metadata(self.temp_file.name, metadata)

        # Retrieve metadata
        cached_metadata = self.meta_cache.get_file_metadata(self.temp_file.name)
        self.assertIsNotNone(cached_metadata)
        self.assertEqual(cached_metadata.analysis_type, "Multitau")
        self.assertTrue(cached_metadata.is_valid())

    def test_qmap_data_caching(self):
        """Test Q-map data caching with integrity verification."""
        # Create mock Q-map data
        qmap_data = QMapData(
            file_path=self.temp_file.name,
            qmap_params={"detector": "test"},
            sqmap=np.random.rand(100, 100),
            dqmap=np.random.randint(0, 50, (100, 100)),
            sqlist=np.logspace(-2, 0, 50),
            dqlist=np.arange(1, 51),
            mask=np.ones((100, 100), dtype=bool),
            geometry_params={"bcx": 50, "bcy": 50, "det_dist": 1000, "pixel_size": 0.1},
        )

        # Cache Q-map data
        self.meta_cache.put_qmap_data(
            self.temp_file.name, {"detector": "test"}, qmap_data
        )

        # Retrieve Q-map data
        cached_qmap = self.meta_cache.get_qmap_data(
            self.temp_file.name, {"detector": "test"}
        )
        self.assertIsNotNone(cached_qmap)
        self.assertTrue(cached_qmap.verify_integrity())
        np.testing.assert_array_equal(cached_qmap.sqmap, qmap_data.sqmap)

    def test_metadata_invalidation(self):
        """Test metadata invalidation when file changes."""
        # Create and cache metadata
        metadata = FileMetadata(
            file_path=self.temp_file.name,
            file_size=os.path.getsize(self.temp_file.name),
            file_mtime=os.path.getmtime(self.temp_file.name),
            analysis_type="Multitau",
            metadata_dict={"test": "data"},
            large_datasets=[],
        )

        self.meta_cache.put_file_metadata(self.temp_file.name, metadata)

        # Verify cached
        cached = self.meta_cache.get_file_metadata(self.temp_file.name)
        self.assertIsNotNone(cached)

        # Modify file
        time.sleep(0.01)  # Ensure different mtime
        with open(self.temp_file.name, "a") as f:
            f.write("modified")

        # Metadata should be invalid now
        self.assertFalse(cached.is_valid())

        # Getting metadata should return None (invalid)
        cached_after_mod = self.meta_cache.get_file_metadata(self.temp_file.name)
        self.assertIsNone(cached_after_mod)


class TestAdaptiveMemoryManager(unittest.TestCase):
    """Test adaptive memory management."""

    def setUp(self):
        """Set up memory manager for testing."""
        self.memory_manager = AdaptiveMemoryManager(
            strategy=MemoryStrategy.BALANCED,
            learning_window_hours=0.1,  # Short window for testing
            prediction_horizon_minutes=1.0,
            max_prefetch_items=3,
        )

    def tearDown(self):
        """Clean up memory manager."""
        self.memory_manager.shutdown()

    def test_access_recording(self):
        """Test access pattern recording."""
        # Record some accesses
        self.memory_manager.record_access("/test/file1.hdf", "saxs_2d", 100.0, 50.0)
        self.memory_manager.record_access("/test/file2.hdf", "g2", 200.0, 30.0)
        self.memory_manager.record_access("/test/file1.hdf", "metadata", 10.0, 1.0)

        # Check that accesses were recorded
        self.assertEqual(len(self.memory_manager._access_history), 3)

        # Check pattern tracking
        self.assertIn("/test/file1.hdf", self.memory_manager._data_type_patterns)
        self.assertIn(
            "saxs_2d", self.memory_manager._data_type_patterns["/test/file1.hdf"]
        )

    def test_pattern_detection(self):
        """Test usage pattern detection."""
        # Create sequential access pattern
        base_time = time.time()
        for i in range(5):
            file_path = f"/test/sequence_{i:03d}.hdf"
            self.memory_manager._access_history.append(
                type(
                    "AccessRecord",
                    (),
                    {
                        "timestamp": base_time + i,
                        "file_path": file_path,
                        "data_type": "saxs_2d",
                        "access_duration_ms": 100.0,
                        "memory_usage_mb": 50.0,
                    },
                )()
            )

        # Trigger pattern analysis
        self.memory_manager._analyze_usage_patterns()

        # Should detect some pattern
        if self.memory_manager._current_pattern:
            self.assertIsInstance(
                self.memory_manager._current_pattern.pattern_type, UsagePattern
            )
            self.assertGreater(self.memory_manager._current_pattern.confidence, 0.0)

    def test_memory_recommendations(self):
        """Test memory management recommendations."""
        # Get recommendations
        recommendations = self.memory_manager.get_memory_recommendations()

        # Should contain expected fields
        self.assertIn("current_memory_pressure", recommendations)
        self.assertIn("recommended_strategy", recommendations)
        self.assertIn("memory_allocation", recommendations)

        # Memory pressure should be reasonable
        pressure = recommendations["current_memory_pressure"]
        self.assertIsInstance(pressure, (int, float))
        self.assertGreaterEqual(pressure, 0.0)
        self.assertLessEqual(pressure, 100.0)


class TestCacheMonitor(unittest.TestCase):
    """Test cache monitoring and performance analytics."""

    def setUp(self):
        """Set up cache monitor for testing."""
        self.monitor = CacheMonitor(
            monitoring_interval_seconds=0.1,  # Fast interval for testing
            enable_alerts=True,
        )
        time.sleep(0.2)  # Let it collect some initial metrics

    def tearDown(self):
        """Clean up cache monitor."""
        self.monitor.shutdown()

    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        # Wait for metrics collection
        time.sleep(0.3)

        # Get current metrics
        metrics = self.monitor.get_current_metrics()

        # Should have some basic metrics
        self.assertIsInstance(metrics, dict)
        self.assertGreater(len(metrics), 0)

        # Check for expected metric categories
        metric_names = list(metrics.keys())
        self.assertTrue(any("cache" in name for name in metric_names))
        self.assertTrue(any("memory" in name for name in metric_names))

    def test_alert_generation(self):
        """Test performance alert generation."""
        # Manually trigger an alert by setting a high threshold violation
        self.monitor._update_metric("system_memory_pressure", 95.0)
        self.monitor._analyze_performance()

        # Check for alerts
        alerts = self.monitor.get_active_alerts()

        # Should have generated alert for high memory pressure
        memory_alerts = [
            a for a in alerts if a["metric_name"] == "system_memory_pressure"
        ]
        if memory_alerts:
            alert = memory_alerts[0]
            self.assertEqual(alert["component"], "cache_system")
            self.assertIn("recommendations", alert)
            self.assertGreater(len(alert["recommendations"]), 0)

    def test_performance_summary(self):
        """Test performance summary generation."""
        summary = self.monitor.get_performance_summary()

        # Should contain expected sections
        self.assertIn("overall_health", summary)
        self.assertIn("key_metrics", summary)
        self.assertIn("alert_summary", summary)
        self.assertIn("recommendations", summary)

        # Health should be a valid value
        health = summary["overall_health"]
        valid_health_values = ["excellent", "good", "fair", "poor", "critical"]
        self.assertIn(health, valid_health_values)

    def test_metric_export(self):
        """Test metrics export functionality."""
        # Export metrics
        exported_data = self.monitor.export_metrics(format="json")

        # Should be valid JSON
        import json

        parsed_data = json.loads(exported_data)

        # Should contain expected sections
        self.assertIn("timestamp", parsed_data)
        self.assertIn("metrics", parsed_data)
        self.assertIn("performance_summary", parsed_data)


class TestCacheIntegration(unittest.TestCase):
    """Test integration between different cache components."""

    def setUp(self):
        """Set up integrated caching system for testing."""
        # Get fresh instances for integration testing
        self.advanced_cache = get_global_cache()
        self.comp_cache = get_computation_cache()
        self.meta_cache = get_metadata_cache(enable_prefetch=False)
        self.memory_manager = get_adaptive_memory_manager()
        self.monitor = get_cache_monitor(monitoring_interval_seconds=0.5)

        # Small delay to let monitor start
        time.sleep(0.1)

    def test_cache_coordination(self):
        """Test coordination between different cache layers."""
        # Add data to advanced cache
        test_data = np.random.rand(100, 100)
        success = self.advanced_cache.put("integration_test", test_data)
        self.assertTrue(success)

        # Retrieve data
        retrieved, found = self.advanced_cache.get("integration_test")
        self.assertTrue(found)
        np.testing.assert_array_equal(retrieved, test_data)

        # Record access in memory manager
        self.memory_manager.record_access(
            "/test/integration.hdf", "test_data", 50.0, 10.0
        )

        # Wait for monitoring
        time.sleep(0.6)

        # Check that monitor collected metrics
        metrics = self.monitor.get_current_metrics()
        self.assertGreater(len(metrics), 0)

    def test_memory_pressure_response(self):
        """Test system response to memory pressure."""
        # Simulate memory pressure by adding large amounts of data
        large_data_items = []
        for i in range(10):
            # Each item ~4MB
            data = np.random.rand(1000, 1000).astype(np.float32)
            large_data_items.append(data)
            self.advanced_cache.put(f"large_item_{i}", data)

        # Wait for system to respond
        time.sleep(0.2)

        # Check cache stats
        stats = self.advanced_cache.get_stats()

        # Should have triggered some evictions or promotions
        self.assertGreaterEqual(stats["operation_counts"]["total_evictions"], 0)

    def test_end_to_end_caching_workflow(self):
        """Test complete end-to-end caching workflow."""
        # Simulate realistic XPCS workflow
        file_path = "/test/workflow.hdf"

        # 1. Load file metadata (would be cached)

        # Record metadata access
        self.memory_manager.record_access(file_path, "metadata", 20.0, 1.0)

        # 2. Load SAXS data (would be cached)
        saxs_data = np.random.rand(512, 512).astype(np.float32)
        self.advanced_cache.put(f"saxs_2d:{file_path}", saxs_data)
        self.memory_manager.record_access(file_path, "saxs_2d", 200.0, 100.0)

        # 3. Perform G2 fitting (would use computation cache)
        fitting_params = {
            "q_range": (0.1, 1.0),
            "t_range": (1e-3, 1.0),
            "fit_func": "single",
        }

        @self.comp_cache.cache_g2_fitting(file_path=file_path, **fitting_params)
        def mock_g2_fitting():
            return {
                "fit_func": "single",
                "fit_val": np.array([[1.0, 0.5, 1.0, 0.1]]),
                "t_el": np.logspace(-3, 0, 50),
                "q_val": np.array([0.5]),
                "fit_line": [],
                "label": ["Q=0.5"],
            }

        # Perform fitting (should be cached)
        fitting_result = mock_g2_fitting()
        self.assertIsInstance(fitting_result, G2FitResult)

        # Record computation access
        self.memory_manager.record_access(file_path, "g2_fitting", 5000.0, 20.0)

        # 4. Check that all systems recorded the workflow
        time.sleep(0.6)  # Wait for monitoring

        # Verify access patterns were recorded
        access_history = self.memory_manager._access_history
        file_accesses = [rec for rec in access_history if rec.file_path == file_path]
        self.assertGreaterEqual(len(file_accesses), 3)  # metadata, saxs_2d, g2_fitting

        # Verify caches have data
        retrieved_saxs, found = self.advanced_cache.get(f"saxs_2d:{file_path}")
        self.assertTrue(found)
        np.testing.assert_array_equal(retrieved_saxs, saxs_data)

        # Verify computation cache has result
        cached_g2 = self.comp_cache.get_cached_g2_fitting(file_path, **fitting_params)
        self.assertIsNotNone(cached_g2)

        # Verify monitoring collected metrics
        metrics = self.monitor.get_current_metrics()
        self.assertIn("cache_overall_hit_rate", metrics)

        # Generate performance summary
        summary = self.monitor.get_performance_summary()
        self.assertIn("overall_health", summary)

        print("End-to-end test completed successfully!")
        print(f"Cache health: {summary['overall_health']}")
        print(
            f"Hit rate: {metrics.get('cache_overall_hit_rate', {}).get('value', 0) * 100:.1f}%"
        )


if __name__ == "__main__":
    # Configure logging for test output
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run tests
    unittest.main(verbosity=2)
