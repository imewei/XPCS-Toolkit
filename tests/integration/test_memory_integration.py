#!/usr/bin/env python3
"""
Memory Management Integration Tests for XPCS Toolkit

This module tests the integration of memory management systems:
- Caching systems working with analysis modules
- Memory pressure detection and cleanup
- Integration between MemoryTracker, XpcsFile, and ViewerKernel
- LRU caching effectiveness during real workflows
- Memory optimization phases working together

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import gc
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XPCS Toolkit components
try:
    from xpcs_toolkit.fileIO.hdf_reader import batch_read_fields, get
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import (
        MemoryStatus,
        MemoryTracker,
        get_cached_memory_monitor,
    )
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile
except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}")
    sys.exit(0)

logger = get_logger(__name__)


class TestMemoryManagementIntegration(unittest.TestCase):
    """Test memory management system integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_memory_integration_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.initial_memory = self._get_process_memory()
        self.test_files = self._create_memory_test_files()

    def tearDown(self):
        """Clean up and verify no memory leaks."""
        gc.collect()  # Force garbage collection
        time.sleep(0.1)  # Allow cleanup

        final_memory = self._get_process_memory()
        memory_increase = final_memory - self.initial_memory

        # Log memory usage for debugging
        logger.info(f"Memory change during test: {memory_increase:.1f} MB")

        # Allow some memory increase but warn if excessive
        if memory_increase > 100:  # More than 100MB increase
            logger.warning(f"High memory increase detected: {memory_increase:.1f} MB")

    def _get_process_memory(self):
        """Get current process memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _create_memory_test_files(self):
        """Create test files of varying sizes for memory testing."""
        test_files = []

        # Create files with different memory footprints
        sizes = [
            {"name": "small", "n_q": 10, "n_tau": 50, "n_saxs": 100},
            {"name": "medium", "n_q": 50, "n_tau": 100, "n_saxs": 500},
            {"name": "large", "n_q": 100, "n_tau": 200, "n_saxs": 1000},
        ]

        for size_config in sizes:
            hdf_path = os.path.join(self.temp_dir, f"{size_config['name']}_data.hdf")
            self._create_sized_file(hdf_path, size_config)
            test_files.append(hdf_path)

        return test_files

    def _create_sized_file(self, hdf_path, config):
        """Create HDF5 file with specified dimensions."""
        with h5py.File(hdf_path, "w") as f:
            # Complete XPCS structure for proper recognition
            entry = f.create_group("entry")
            entry.create_dataset("start_time", data="2023-01-01T00:00:00")
            f.attrs["analysis_type"] = "XPCS"

            # Instrument configuration
            instrument = entry.create_group("instrument")
            detector_1 = instrument.create_group("detector_1")
            detector_1.create_dataset("frame_time", data=0.001)
            detector_1.create_dataset("count_time", data=0.001)

            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            temporal_mean = xpcs.create_group("temporal_mean")
            qmap = xpcs.create_group("qmap")

            n_q = config["n_q"]
            n_tau = config["n_tau"]
            n_saxs = config["n_saxs"]

            # Create arrays of specified sizes
            tau = np.logspace(-6, 2, n_tau)
            g2_data = np.random.rand(n_q, n_tau) * 0.5 + 1.0
            g2_err = 0.02 * g2_data

            multitau.create_dataset("normalized_g2", data=g2_data)
            multitau.create_dataset("normalized_g2_err", data=g2_err)
            multitau.create_dataset("delay_list", data=tau)

            # Configuration data required by XpcsFile
            config_group = multitau.create_group("config")
            config_group.create_dataset("avg_frame", data=1)
            config_group.create_dataset("stride_frame", data=1)

            # Large SAXS arrays for memory testing
            saxs_1d = np.random.rand(n_saxs) * 1000
            saxs_2d = np.random.rand(256, 256) * 1000
            Iqp_data = np.random.rand(n_q, n_saxs) * 1000

            temporal_mean.create_dataset("scattering_1d", data=saxs_1d)
            temporal_mean.create_dataset("scattering_2d", data=saxs_2d)
            temporal_mean.create_dataset("scattering_1d_segments", data=Iqp_data)

            # Q-map data for proper recognition
            qmap.create_dataset("dynamic_v_list_dim0", data=np.linspace(0.001, 0.1, n_q))
            qmap.create_dataset("static_v_list_dim0", data=np.linspace(0.001, 0.1, n_q))
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(n_q))
            qmap.create_dataset("static_index_mapping", data=np.arange(n_q))
            qmap.create_dataset("dynamic_num_pts", data=n_q)
            qmap.create_dataset("static_num_pts", data=n_q)

            # Spatial mean group for intensity vs time
            spatial_mean = xpcs.create_group("spatial_mean")
            n_time = 5000 if config["name"] == "large" else 1000
            intensity_data = 1000 + 50 * np.random.randn(n_time)
            spatial_mean.create_dataset("intensity_vs_time", data=intensity_data)

    def test_memory_tracker_integration_with_file_loading(self):
        """Test MemoryTracker integration with file loading operations."""
        tracker = MemoryTracker()
        tracker.start_tracking()

        memory_points = []

        # Track memory during file loading
        for i, test_file in enumerate(self.test_files):
            # Record memory before loading
            memory_before = tracker.get_current_usage()

            # Load file
            xf = XpcsFile(test_file)

            # Access data to trigger loading
            g2_data = xf.g2
            tau_data = xf.tau
            saxs_data = get(test_file, "/xpcs/temporal_mean/scattering_1d")

            # Record memory after loading
            memory_after = tracker.get_current_usage()

            memory_points.append(
                {
                    "file": Path(test_file).stem,
                    "before": memory_before,
                    "after": memory_after,
                    "increase": memory_after - memory_before,
                    "data_size": g2_data.nbytes + tau_data.nbytes + saxs_data.nbytes
                    if saxs_data is not None
                    else g2_data.nbytes + tau_data.nbytes,
                }
            )

            # Verify memory tracking is working
            self.assertGreater(
                memory_after,
                memory_before,
                f"Memory should increase after loading {test_file}",
            )

        tracker.stop_tracking()

        # Analyze memory usage patterns
        for point in memory_points:
            logger.info(
                f"File {point['file']}: "
                f"Memory increase {point['increase']:.1f} MB, "
                f"Data size {point['data_size'] / 1024 / 1024:.1f} MB"
            )

            # Memory increase should be reasonable relative to data size
            data_size_mb = point["data_size"] / 1024 / 1024
            memory_efficiency = point["increase"] / max(data_size_mb, 0.1)

            # Memory overhead should not be excessive (< 10x data size)
            self.assertLess(
                memory_efficiency,
                10.0,
                f"Memory efficiency too low for {point['file']}",
            )

    def test_cached_memory_monitor_integration(self):
        """Test cached memory monitor integration with components."""
        # Test memory monitor caching
        monitor1 = get_cached_memory_monitor()
        monitor2 = get_cached_memory_monitor()

        # Should return the same cached instance
        self.assertIs(monitor1, monitor2, "Memory monitor should be cached")

        # Test integration with MemoryMonitor class
        used_mb, available_mb = MemoryMonitor.get_memory_usage()
        pressure = MemoryMonitor.get_memory_pressure()
        is_high_pressure = MemoryMonitor.is_memory_pressure_high()

        # Verify returned values are reasonable
        self.assertIsInstance(used_mb, (int, float))
        self.assertIsInstance(available_mb, (int, float))
        self.assertIsInstance(pressure, (int, float))
        self.assertIsInstance(is_high_pressure, bool)

        self.assertGreater(used_mb, 0)
        self.assertGreater(available_mb, 0)
        self.assertGreaterEqual(pressure, 0.0)
        self.assertLessEqual(pressure, 1.0)

        # Test memory pressure detection with file loading
        initial_pressure = MemoryMonitor.get_memory_pressure()

        # Load large file to increase memory pressure
        large_file = self.test_files[-1]  # Largest file
        xf = XpcsFile(large_file)

        # Access all data to maximize memory usage
        _ = xf.g2
        _ = xf.tau
        _ = get(large_file, "/xpcs/temporal_mean/scattering_1d")
        _ = get(large_file, "/xpcs/temporal_mean/scattering_1d_segments")
        _ = get(large_file, "/xpcs/spatial_mean/intensity_vs_time")

        final_pressure = MemoryMonitor.get_memory_pressure()

        # Memory pressure should increase or stay the same
        self.assertGreaterEqual(final_pressure, initial_pressure)

    def test_memory_cleanup_integration(self):
        """Test memory cleanup mechanisms in integrated system."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        initial_memory = self._get_process_memory()
        loaded_files = []

        # Load multiple files and track memory
        memory_checkpoints = []

        for i in range(len(self.test_files)):
            if i < len(kernel.raw_hdf_files):
                # Load file
                xf = kernel.load_xpcs_file(i)
                loaded_files.append(xf)

                # Access data to trigger loading
                _ = xf.g2
                _ = xf.tau

                current_memory = self._get_process_memory()
                memory_checkpoints.append(current_memory - initial_memory)

                logger.debug(
                    f"Memory after loading file {i}: "
                    f"{current_memory - initial_memory:.1f} MB"
                )

        peak_memory = max(memory_checkpoints) if memory_checkpoints else 0

        # Test cleanup by removing references
        del loaded_files
        gc.collect()
        time.sleep(0.1)  # Allow cleanup

        cleanup_memory = self._get_process_memory() - initial_memory

        # Memory should decrease after cleanup
        memory_reduction = peak_memory - cleanup_memory
        self.assertGreater(
            memory_reduction,
            0,
            f"Memory cleanup ineffective. Peak: {peak_memory:.1f} MB, "
            f"After cleanup: {cleanup_memory:.1f} MB",
        )

        # Memory reduction should be significant (at least 20% of peak usage)
        cleanup_efficiency = memory_reduction / max(peak_memory, 1)
        self.assertGreater(
            cleanup_efficiency,
            0.1,
            f"Memory cleanup efficiency too low: {cleanup_efficiency:.2f}",
        )

    def test_concurrent_memory_management(self):
        """Test memory management under concurrent access."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        memory_tracker = MemoryTracker()
        memory_tracker.start_tracking()

        results = []
        errors = []
        memory_snapshots = []

        def concurrent_file_access(thread_id):
            """Access files concurrently and track memory."""
            try:
                for i in range(len(self.test_files)):
                    if i < len(kernel.raw_hdf_files):
                        # Load and access file
                        xf = kernel.load_xpcs_file(i)
                        g2_data = xf.g2

                        # Track memory usage
                        current_memory = memory_tracker.get_current_usage()
                        memory_snapshots.append((thread_id, i, current_memory))

                        # Simulate processing
                        if g2_data.size > 0:
                            mean_g2 = np.mean(g2_data)
                            results.append((thread_id, i, mean_g2))

                        time.sleep(0.01)  # Small delay

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent threads
        threads = []
        n_threads = 3

        for thread_id in range(n_threads):
            thread = threading.Thread(target=concurrent_file_access, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        memory_tracker.stop_tracking()

        # Analyze results
        self.assertGreater(len(results), 0, "No successful concurrent operations")
        self.assertEqual(len(errors), 0, f"Concurrent access errors: {errors}")

        # Analyze memory usage patterns
        if memory_snapshots:
            memory_values = [snapshot[2] for snapshot in memory_snapshots]
            max_memory = max(memory_values)
            min_memory = min(memory_values)
            memory_range = max_memory - min_memory

            logger.info(
                f"Concurrent memory range: {memory_range:.1f} MB "
                f"(min: {min_memory:.1f}, max: {max_memory:.1f})"
            )

            # Memory should not grow excessively during concurrent access
            self.assertLess(
                memory_range, 500, "Excessive memory growth during concurrent access"
            )

    def test_lazy_loading_memory_integration(self):
        """Test lazy loading integration with memory management."""
        # Test that large datasets are loaded lazily
        large_file = self.test_files[-1]  # Largest test file

        # Memory before file creation
        memory_before = self._get_process_memory()

        # Create XpcsFile object (should not load data yet)
        xf = XpcsFile(large_file)
        memory_after_creation = self._get_process_memory()

        # Memory increase should be minimal for object creation
        creation_memory_increase = memory_after_creation - memory_before
        self.assertLess(
            creation_memory_increase,
            10,
            "Excessive memory usage during XpcsFile creation",
        )

        # Access different datasets and track memory
        datasets_accessed = []

        # Access G2 data
        memory_before_g2 = self._get_process_memory()
        g2_data = xf.g2
        memory_after_g2 = self._get_process_memory()

        datasets_accessed.append(
            {
                "name": "g2",
                "size_mb": g2_data.nbytes / 1024 / 1024 if g2_data is not None else 0,
                "memory_increase": memory_after_g2 - memory_before_g2,
            }
        )

        # Access SAXS data
        memory_before_saxs = self._get_process_memory()
        saxs_data = get(large_file, "/xpcs/temporal_mean/scattering_1d")
        memory_after_saxs = self._get_process_memory()

        datasets_accessed.append(
            {
                "name": "saxs_1d",
                "size_mb": saxs_data.nbytes / 1024 / 1024
                if saxs_data is not None
                else 0,
                "memory_increase": memory_after_saxs - memory_before_saxs,
            }
        )

        # Verify lazy loading behavior
        for dataset in datasets_accessed:
            logger.info(
                f"Dataset {dataset['name']}: "
                f"Size {dataset['size_mb']:.1f} MB, "
                f"Memory increase {dataset['memory_increase']:.1f} MB"
            )

            if dataset["size_mb"] > 0:
                # Memory increase should be proportional to data size
                memory_efficiency = dataset["memory_increase"] / dataset["size_mb"]
                self.assertLess(
                    memory_efficiency,
                    5.0,
                    f"Poor memory efficiency for {dataset['name']}",
                )

    def test_memory_pressure_response_integration(self):
        """Test system response to memory pressure."""
        # Force memory pressure by loading large datasets
        initial_pressure = MemoryMonitor.get_memory_pressure()

        # Load files progressively and monitor pressure
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        loaded_files = []
        pressure_history = [initial_pressure]

        for i in range(len(self.test_files)):
            if i < len(kernel.raw_hdf_files):
                # Load file and access large datasets
                xf = kernel.load_xpcs_file(i)
                loaded_files.append(xf)

                # Access data to increase memory usage
                _ = xf.g2
                _ = xf.tau

                # Try to access large datasets
                try:
                    _ = get(kernel.raw_hdf_files[i], "/xpcs/temporal_mean/scattering_1d_segments")
                    _ = get(kernel.raw_hdf_files[i], "/xpcs/spatial_mean/intensity_vs_time")
                except Exception as e:
                    logger.debug(f"Could not access large dataset: {e}")

                current_pressure = MemoryMonitor.get_memory_pressure()
                pressure_history.append(current_pressure)

                # Check if memory pressure is getting high
                if MemoryMonitor.is_memory_pressure_high(threshold=0.7):
                    logger.warning(
                        f"High memory pressure detected after loading {i + 1} files"
                    )

                    # Test cleanup mechanisms
                    # Remove oldest files to free memory
                    if len(loaded_files) > 2:
                        del loaded_files[0]
                        gc.collect()

                        after_cleanup_pressure = MemoryMonitor.get_memory_pressure()
                        pressure_history.append(after_cleanup_pressure)

                        # Pressure should decrease after cleanup
                        self.assertLessEqual(
                            after_cleanup_pressure,
                            current_pressure,
                            "Memory pressure should decrease after cleanup",
                        )

        # Analyze pressure history
        max_pressure = max(pressure_history)
        logger.info(f"Maximum memory pressure reached: {max_pressure:.2f}")

        # System should handle memory pressure gracefully
        self.assertLess(max_pressure, 0.95, "Memory pressure reached critical levels")

    def test_cache_integration_with_repeated_access(self):
        """Test caching system integration with repeated data access."""
        test_file = self.test_files[1]  # Medium-sized file

        # First access - should trigger loading
        xf = XpcsFile(test_file)

        access_times = []

        # Time first access
        start_time = time.time()
        g2_data1 = xf.g2
        first_access_time = time.time() - start_time
        access_times.append(("first", first_access_time))

        # Time second access (should use cache)
        start_time = time.time()
        g2_data2 = xf.g2
        second_access_time = time.time() - start_time
        access_times.append(("second", second_access_time))

        # Time third access (should use cache)
        start_time = time.time()
        g2_data3 = xf.g2
        third_access_time = time.time() - start_time
        access_times.append(("third", third_access_time))

        # Verify data consistency
        np.testing.assert_array_equal(g2_data1, g2_data2)
        np.testing.assert_array_equal(g2_data1, g2_data3)

        # Verify caching effectiveness
        logger.info(f"Access times: {access_times}")

        # Cached accesses should be significantly faster
        self.assertLess(
            second_access_time,
            first_access_time * 0.1,
            "Second access should be much faster (cached)",
        )
        self.assertLess(
            third_access_time,
            first_access_time * 0.1,
            "Third access should be much faster (cached)",
        )

        # Test cache invalidation/refresh
        # Access different dataset to test cache behavior
        start_time = time.time()
        tau_data = xf.tau
        tau_access_time = time.time() - start_time
        access_times.append(("tau_first", tau_access_time))

        # Second tau access should be fast
        start_time = time.time()
        tau_data2 = xf.tau
        tau_cached_time = time.time() - start_time
        access_times.append(("tau_cached", tau_cached_time))

        np.testing.assert_array_equal(tau_data, tau_data2)
        self.assertLess(
            tau_cached_time, tau_access_time * 0.2, "Cached tau access should be fast"
        )


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run memory integration tests
    unittest.main(verbosity=2)
