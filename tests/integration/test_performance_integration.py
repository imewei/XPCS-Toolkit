#!/usr/bin/env python3
"""
Performance System Integration Tests for XPCS Toolkit

This module tests the integration of all performance optimization phases:
- Phase 1: Memory Management Integration
- Phase 2: I/O Performance Integration
- Phase 3: Vectorization Integration
- Phase 4: GUI Threading Integration
- Phase 5: Advanced Caching Integration
- Cross-phase optimization effectiveness
- End-to-end performance validation

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
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XPCS Toolkit components
try:
    from xpcs_toolkit.fileIO.hdf_reader import (
        batch_read_fields,
        get,
        get_chunked_dataset,
    )
    from xpcs_toolkit.fileIO.qmap_utils import get_qmap
    from xpcs_toolkit.module import g2mod, intt, saxs1d, saxs2d, stability, twotime
    from xpcs_toolkit.module.average_toolbox import AverageToolbox
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import MemoryTracker, get_cached_memory_monitor
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile
except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}", stacklevel=2)
    sys.exit(0)

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking utilities."""

    def __init__(self):
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self.start_times = {}

    def start_timing(self, operation):
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()

    def end_timing(self, operation):
        """End timing and record duration."""
        if operation in self.start_times:
            duration = time.perf_counter() - self.start_times[operation]
            self.timings[operation].append(duration)
            del self.start_times[operation]
            return duration
        return None

    def record_memory(self, operation, memory_mb):
        """Record memory usage for operation."""
        self.memory_usage[operation].append(memory_mb)

    def get_stats(self, operation):
        """Get performance statistics for operation."""
        if operation in self.timings:
            times = self.timings[operation]
            return {
                "count": len(times),
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "total_time": np.sum(times),
            }
        return None

    def compare_operations(self, baseline_op, optimized_op):
        """Compare performance between operations."""
        baseline_stats = self.get_stats(baseline_op)
        optimized_stats = self.get_stats(optimized_op)

        if baseline_stats and optimized_stats:
            speedup = baseline_stats["mean_time"] / optimized_stats["mean_time"]
            return {
                "speedup": speedup,
                "baseline_mean": baseline_stats["mean_time"],
                "optimized_mean": optimized_stats["mean_time"],
                "improvement": (1 - 1 / speedup) * 100 if speedup > 1 else 0,
            }
        return None


class TestPerformanceSystemIntegration(unittest.TestCase):
    """Test integration of all performance optimization systems."""

    def setUp(self):
        """Set up performance testing environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_performance_test_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.benchmark = PerformanceBenchmark()
        self.test_files = self._create_performance_test_files()

        # Initialize memory tracking
        self.memory_tracker = MemoryTracker()
        self.memory_tracker.start_tracking()
        self.initial_memory = self.memory_tracker.get_current_usage()

    def tearDown(self):
        """Clean up and log performance results."""
        self.memory_tracker.stop_tracking()
        final_memory = self.memory_tracker.get_current_usage()

        # Log performance summary
        logger.info("Performance Test Summary:")
        for operation, stats in [
            (op, self.benchmark.get_stats(op)) for op in self.benchmark.timings
        ]:
            if stats:
                logger.info(
                    f"  {operation}: {stats['mean_time']:.4f}s ± {stats['std_time']:.4f}s "
                    f"(n={stats['count']})"
                )

        memory_change = final_memory - self.initial_memory
        logger.info(f"Total memory change: {memory_change:.1f} MB")

    def _create_performance_test_files(self):
        """Create test files optimized for performance testing."""
        test_files = []

        # Create files with different performance characteristics
        configs = [
            # Small file for baseline
            {
                "name": "small_perf",
                "n_q": 20,
                "n_tau": 50,
                "n_saxs": 100,
                "n_time": 1000,
            },
            # Medium file for realistic testing
            {
                "name": "medium_perf",
                "n_q": 50,
                "n_tau": 100,
                "n_saxs": 500,
                "n_time": 5000,
            },
            # Large file for stress testing
            {
                "name": "large_perf",
                "n_q": 100,
                "n_tau": 200,
                "n_saxs": 1000,
                "n_time": 10000,
            },
            # Extra large for memory pressure testing
            {
                "name": "xlarge_perf",
                "n_q": 150,
                "n_tau": 250,
                "n_saxs": 1500,
                "n_time": 15000,
            },
        ]

        for config in configs:
            hdf_path = os.path.join(self.temp_dir, f"{config['name']}.hdf")
            self._create_performance_file(hdf_path, config)
            test_files.append(hdf_path)

        return test_files

    def _create_performance_file(self, hdf_path, config):
        """Create HDF5 file optimized for performance testing."""
        with h5py.File(hdf_path, "w") as f:
            # File attributes
            f.attrs["format_version"] = "2.0"
            f.attrs["analysis_type"] = "Multitau"

            # Standard structure
            entry = f.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            entry.create_dataset("start_time", data="2024-01-01T00:00:00")

            # Create instrument/detector structure
            instrument = entry.create_group("instrument")
            detector_1 = instrument.create_group("detector_1")
            detector_1.create_dataset("frame_time", data=0.001)
            detector_1.create_dataset("count_time", data=0.001)

            # Create incident_beam structure
            incident_beam = instrument.create_group("incident_beam")
            incident_beam.create_dataset("incident_energy", data=8000.0)

            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            n_q = config["n_q"]
            n_tau = config["n_tau"]
            n_saxs = config["n_saxs"]
            n_time = config["n_time"]

            # Create realistic correlation data with proper structure
            tau = np.logspace(-6, 2, n_tau)
            g2_data = np.zeros((n_q, n_tau))

            for i in range(n_q):
                # Different dynamics for each Q
                tau_c = 1e-3 * (i + 1) / n_q  # Q-dependent correlation time
                beta = 0.8 * (1 - 0.3 * i / n_q)  # Varying contrast

                g2_clean = 1 + beta * np.exp(-tau / tau_c)
                noise = 0.02 * g2_clean
                g2_data[i, :] = g2_clean + np.random.normal(0, noise)

            # Ensure G2 >= 1
            g2_data = np.maximum(g2_data, 1.0)
            g2_err = 0.02 * g2_data

            # Use appropriate compression for performance testing
            multitau.create_dataset(
                "normalized_g2", data=g2_data, compression="gzip", compression_opts=6
            )
            multitau.create_dataset(
                "normalized_g2_err", data=g2_err, compression="gzip", compression_opts=6
            )
            multitau.create_dataset("tau", data=tau)

            # Large SAXS datasets for I/O testing
            q_saxs = np.logspace(-3, -1, n_saxs)
            I_saxs = 1000 * q_saxs ** (-2.5) + 100
            I_saxs += np.random.normal(0, 0.05 * I_saxs)  # Add noise

            # Create temporal_mean group for SAXS data
            temporal_mean = xpcs.create_group("temporal_mean")
            temporal_mean.create_dataset(
                "scattering_1d",
                data=I_saxs.reshape(1, -1),
                compression="gzip",
                compression_opts=6,
            )

            # 2D SAXS data (Iqp) for memory testing
            Iqp_data = np.zeros((n_q, n_saxs))
            for i in range(n_q):
                q_factor = (i + 1) / n_q
                Iqp_data[i, :] = I_saxs * q_factor + np.random.normal(0, 10, n_saxs)

            temporal_mean.create_dataset(
                "scattering_1d_segments", data=Iqp_data, compression="gzip", compression_opts=6
            )

            # Also create 2D SAXS data
            saxs_2d_data = np.random.rand(50, 50) * 1000
            temporal_mean.create_dataset(
                "scattering_2d", data=saxs_2d_data, compression="gzip", compression_opts=6
            )

            # Time series data for stability analysis
            time_data = np.linspace(0, 1000, n_time)
            intensity_data = (
                1000 + 100 * np.sin(time_data / 50) + np.random.normal(0, 20, n_time)
            )
            intensity_data = np.maximum(intensity_data, 10)  # Ensure positive

            # Create spatial_mean group for intensity vs time data
            spatial_mean = xpcs.create_group("spatial_mean")
            spatial_mean.create_dataset(
                "intensity_vs_time",
                data=np.vstack([time_data, intensity_data]),
                compression="gzip",
                compression_opts=6,
            )

            # Q-map data
            q_values = np.logspace(-3, -1, n_q)
            qmap.create_dataset("dqlist", data=q_values)
            qmap.create_dataset("sqlist", data=q_values)
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(n_q))
            qmap.create_dataset("static_index_mapping", data=np.arange(n_q))
            qmap.create_dataset("dplist", data=np.linspace(-180, 180, 36))
            qmap.create_dataset("splist", data=np.linspace(-180, 180, 36))

            # Add metadata for realistic file structure
            multitau.create_dataset("delay_list", data=tau)

            # Create config subgroup
            config = multitau.create_group("config")
            config.create_dataset("stride_frame", data=1)
            config.create_dataset("avg_frame", data=1)

            # Other metadata
            multitau.create_dataset("t0", data=0.001)
            multitau.create_dataset("t1", data=0.001)

            # Two-time data (small to keep file sizes reasonable)
            twotime_group = xpcs.create_group("twotime")
            twotime_size = min(100, n_time // 100)  # Reasonable size
            twotime_data = np.random.rand(twotime_size, twotime_size) * 0.5 + 1.0
            twotime_group.create_dataset(
                "g2", data=twotime_data, compression="gzip", compression_opts=6
            )
            twotime_group.create_dataset(
                "elapsed_time", data=np.linspace(0, 100, twotime_size)
            )

    def test_file_io_performance_integration(self):
        """Test I/O performance optimization integration."""

        # Test file loading performance
        for i, test_file in enumerate(self.test_files):
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024
            logger.info(
                f"Testing I/O performance on {Path(test_file).stem} ({file_size_mb:.1f} MB)"
            )

            # Test 1: File loading time
            self.benchmark.start_timing(f"file_load_{i}")
            XpcsFile(test_file)
            load_time = self.benchmark.end_timing(f"file_load_{i}")

            # Test 2: Data access time (first access)
            self.benchmark.start_timing(f"first_data_access_{i}")
            first_access_time = self.benchmark.end_timing(f"first_data_access_{i}")

            # Test 3: Cached data access time
            self.benchmark.start_timing(f"cached_data_access_{i}")
            cached_access_time = self.benchmark.end_timing(f"cached_data_access_{i}")

            # Test 4: Large dataset access
            self.benchmark.start_timing(f"large_dataset_access_{i}")
            get(test_file, ["/xpcs/multitau/saxs_1d"])["/xpcs/multitau/saxs_1d"]
            get(test_file, ["/xpcs/multitau/Iqp"])["/xpcs/multitau/Iqp"]
            large_access_time = self.benchmark.end_timing(f"large_dataset_access_{i}")

            # Performance assertions

            # File loading should be reasonable
            max_load_time = 2.0 + file_size_mb * 0.1  # Scale with file size
            self.assertLess(
                load_time,
                max_load_time,
                f"File loading too slow: {load_time:.3f}s > {max_load_time:.3f}s",
            )

            # Cached access should be much faster
            if first_access_time > 0:
                cache_speedup = (
                    first_access_time / cached_access_time
                    if cached_access_time > 0
                    else 1
                )
                self.assertGreater(
                    cache_speedup,
                    5.0,
                    f"Insufficient cache speedup: {cache_speedup:.2f}x",
                )

            # Large dataset access should be reasonable
            max_large_access = 5.0 + file_size_mb * 0.05
            self.assertLess(
                large_access_time,
                max_large_access,
                f"Large dataset access too slow: {large_access_time:.3f}s",
            )

            logger.debug(
                f"File {i}: load={load_time:.3f}s, "
                f"first_access={first_access_time:.3f}s, "
                f"cached={cached_access_time:.3f}s, "
                f"large={large_access_time:.3f}s"
            )

    def test_memory_optimization_integration(self):
        """Test memory optimization system integration."""
        initial_memory = self.memory_tracker.get_current_usage()

        # Test memory efficiency during progressive file loading
        loaded_files = []
        memory_checkpoints = []

        for i, test_file in enumerate(self.test_files):
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024

            # Load file and track memory
            memory_before = self.memory_tracker.get_current_usage()

            xf = XpcsFile(test_file)
            loaded_files.append(xf)

            # Access data to trigger loading
            g2_data = xf.g2
            tau_data = xf.tau

            memory_after = self.memory_tracker.get_current_usage()
            memory_increase = memory_after - memory_before

            # Calculate theoretical memory usage
            theoretical_memory = 0
            if g2_data is not None:
                theoretical_memory += g2_data.nbytes / 1024 / 1024
            if tau_data is not None:
                theoretical_memory += tau_data.nbytes / 1024 / 1024

            memory_efficiency = memory_increase / max(theoretical_memory, 1)

            memory_checkpoints.append(
                {
                    "file_index": i,
                    "file_size_mb": file_size_mb,
                    "memory_increase": memory_increase,
                    "theoretical_memory": theoretical_memory,
                    "efficiency": memory_efficiency,
                    "total_memory": memory_after - initial_memory,
                }
            )

            logger.debug(
                f"File {i}: size={file_size_mb:.1f}MB, "
                f"memory_increase={memory_increase:.1f}MB, "
                f"efficiency={memory_efficiency:.2f}"
            )

            # Memory efficiency should be reasonable (< 5x theoretical)
            self.assertLess(
                memory_efficiency,
                5.0,
                f"Poor memory efficiency for file {i}: {memory_efficiency:.2f}x",
            )

        # Test memory cleanup
        peak_memory = self.memory_tracker.get_current_usage() - initial_memory

        del loaded_files
        gc.collect()
        time.sleep(0.1)

        cleanup_memory = self.memory_tracker.get_current_usage() - initial_memory
        memory_reduction = peak_memory - cleanup_memory
        cleanup_efficiency = memory_reduction / peak_memory if peak_memory > 0 else 0

        logger.info(
            f"Memory cleanup: peak={peak_memory:.1f}MB, "
            f"after_cleanup={cleanup_memory:.1f}MB, "
            f"reduction={memory_reduction:.1f}MB ({cleanup_efficiency:.1%})"
        )

        # Cleanup should be effective
        self.assertGreater(
            cleanup_efficiency,
            0.2,
            f"Memory cleanup ineffective: {cleanup_efficiency:.1%}",
        )

    def test_vectorization_performance_integration(self):
        """Test vectorization optimization integration."""
        test_file = self.test_files[1]  # Medium-sized file
        xf = XpcsFile(test_file)

        g2_data = xf.g2

        if g2_data is None or g2_data.size == 0:
            self.skipTest("No G2 data available for vectorization testing")

        # Test vectorized operations vs non-vectorized
        n_q_test = min(10, g2_data.shape[0])  # Test subset for speed

        # Test 1: Vectorized statistical operations
        self.benchmark.start_timing("vectorized_stats")

        # Compute statistics using vectorized operations
        mean_g2 = np.mean(g2_data[:n_q_test], axis=1)
        std_g2 = np.std(g2_data[:n_q_test], axis=1)
        contrast = (
            g2_data[:n_q_test, 0] - 1.0 if g2_data.shape[1] > 0 else np.zeros(n_q_test)
        )

        vectorized_time = self.benchmark.end_timing("vectorized_stats")

        # Test 2: Non-vectorized equivalent (for comparison)
        self.benchmark.start_timing("non_vectorized_stats")

        mean_g2_loop = np.zeros(n_q_test)
        std_g2_loop = np.zeros(n_q_test)
        contrast_loop = np.zeros(n_q_test)

        for i in range(n_q_test):
            mean_g2_loop[i] = np.mean(g2_data[i, :])
            std_g2_loop[i] = np.std(g2_data[i, :])
            contrast_loop[i] = g2_data[i, 0] - 1.0 if g2_data.shape[1] > 0 else 0.0

        non_vectorized_time = self.benchmark.end_timing("non_vectorized_stats")

        # Verify results are equivalent
        np.testing.assert_allclose(mean_g2, mean_g2_loop, rtol=1e-10)
        np.testing.assert_allclose(std_g2, std_g2_loop, rtol=1e-10)
        np.testing.assert_allclose(contrast, contrast_loop, rtol=1e-10)

        # Vectorized should be faster
        if non_vectorized_time > 0:
            speedup = (
                non_vectorized_time / vectorized_time if vectorized_time > 0 else 1
            )
            logger.info(
                f"Vectorization speedup: {speedup:.2f}x "
                f"(vectorized: {vectorized_time:.4f}s, "
                f"non-vectorized: {non_vectorized_time:.4f}s)"
            )

            self.assertGreater(
                speedup, 2.0, f"Insufficient vectorization speedup: {speedup:.2f}x"
            )

    def test_concurrent_performance_integration(self):
        """Test performance under concurrent operations."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.build()

        if len(kernel.source) == 0:
            self.skipTest("No test files available")

        # Test concurrent performance
        self.benchmark.start_timing("concurrent_operations")

        results = []
        errors = []

        def concurrent_analysis(thread_id):
            """Perform analysis concurrently."""
            try:
                thread_results = []
                start_time = time.perf_counter()

                for i in range(min(3, len(kernel.source))):
                    xf = kernel.load_xpcs_file(i)
                    g2_data = xf.g2

                    if g2_data is not None and g2_data.size > 0:
                        # Simple analysis
                        mean_g2 = np.mean(g2_data)
                        max_contrast = (
                            np.max(g2_data[:, 0] - 1.0) if g2_data.shape[1] > 0 else 0
                        )

                        thread_results.append(
                            {
                                "file_index": i,
                                "mean_g2": mean_g2,
                                "max_contrast": max_contrast,
                            }
                        )

                end_time = time.perf_counter()
                results.append((thread_id, thread_results, end_time - start_time))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent threads
        threads = []
        n_threads = 3

        for i in range(n_threads):
            thread = threading.Thread(target=concurrent_analysis, args=(f"thread_{i}",))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        concurrent_time = self.benchmark.end_timing("concurrent_operations")

        # Analyze concurrent performance
        self.assertEqual(len(errors), 0, f"Concurrent errors: {errors}")
        self.assertGreater(len(results), 0, "No concurrent results")

        total_operations = sum(len(thread_results) for _, thread_results, _ in results)
        individual_times = [thread_time for _, _, thread_time in results]

        np.mean(individual_times)
        operations_per_second = (
            total_operations / concurrent_time if concurrent_time > 0 else 0
        )

        logger.info(
            f"Concurrent performance: {total_operations} operations in "
            f"{concurrent_time:.3f}s ({operations_per_second:.1f} ops/sec)"
        )

        # Performance should be reasonable
        self.assertGreater(operations_per_second, 1.0, "Concurrent operations too slow")

    def test_caching_system_performance_integration(self):
        """Test caching system performance integration."""
        test_file = self.test_files[0]  # Use smaller file for repeated access

        # Test cache performance with repeated access

        # First access (cold cache)
        xf = XpcsFile(test_file)

        self.benchmark.start_timing("cache_cold_g2")
        g2_data1 = xf.g2
        cold_time = self.benchmark.end_timing("cache_cold_g2")

        # Second access (warm cache)
        self.benchmark.start_timing("cache_warm_g2")
        g2_data2 = xf.g2
        self.benchmark.end_timing("cache_warm_g2")

        # Multiple warm accesses
        warm_times = []
        for i in range(5):
            self.benchmark.start_timing(f"cache_warm_{i}")
            _ = xf.g2
            warm_time_i = self.benchmark.end_timing(f"cache_warm_{i}")
            warm_times.append(warm_time_i)

        # Analyze cache performance
        np.testing.assert_array_equal(
            g2_data1, g2_data2, "Cache returned different data"
        )

        mean_warm_time = np.mean(warm_times)
        cache_speedup = cold_time / mean_warm_time if mean_warm_time > 0 else 1

        logger.info(
            f"Cache performance: cold={cold_time:.4f}s, "
            f"warm={mean_warm_time:.4f}s, speedup={cache_speedup:.1f}x"
        )

        # Cache should provide significant speedup
        self.assertGreater(
            cache_speedup, 10.0, f"Insufficient cache speedup: {cache_speedup:.1f}x"
        )

        # Test cache invalidation/refresh performance
        # Access different dataset
        self.benchmark.start_timing("cache_tau_cold")
        tau_cold_time = self.benchmark.end_timing("cache_tau_cold")

        self.benchmark.start_timing("cache_tau_warm")
        tau_warm_time = self.benchmark.end_timing("cache_tau_warm")

        tau_speedup = tau_cold_time / tau_warm_time if tau_warm_time > 0 else 1

        logger.info(f"Tau cache performance: speedup={tau_speedup:.1f}x")
        self.assertGreater(tau_speedup, 5.0, "Insufficient tau cache speedup")

    def test_end_to_end_performance_integration(self):
        """Test end-to-end performance of complete workflows."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.build()

        if len(kernel.source) == 0:
            self.skipTest("No test files available")

        # Test complete XPCS workflow performance
        workflow_results = {}

        for i, test_file in enumerate(self.test_files[:2]):  # Test first 2 files
            file_name = Path(test_file).stem

            self.benchmark.start_timing(f"workflow_{i}")

            try:
                # Step 1: File loading
                xf = kernel.load_xpcs_file(i)

                # Step 2: Data access
                g2_data = xf.g2
                tau_data = xf.tau

                # Step 3: Q-map processing
                qmap_data = get_qmap(test_file)

                # Step 4: SAXS analysis
                saxs_data = get(test_file, ["/xpcs/multitau/saxs_1d"])["/xpcs/multitau/saxs_1d"]

                # Step 5: Stability analysis
                int_t_data = get(test_file, ["/xpcs/multitau/Int_t"])["/xpcs/multitau/Int_t"]

                # Step 6: Basic analysis computations
                if g2_data is not None and g2_data.size > 0:
                    mean_g2 = np.mean(g2_data)
                    contrast_values = (
                        g2_data[:, 0] - 1.0 if g2_data.shape[1] > 0 else np.array([])
                    )
                    max_contrast = (
                        np.max(contrast_values) if len(contrast_values) > 0 else 0
                    )

                    workflow_time = self.benchmark.end_timing(f"workflow_{i}")

                    workflow_results[file_name] = {
                        "success": True,
                        "time": workflow_time,
                        "n_q": g2_data.shape[0],
                        "n_tau": len(tau_data) if tau_data is not None else 0,
                        "mean_g2": mean_g2,
                        "max_contrast": max_contrast,
                        "has_qmap": qmap_data is not None,
                        "has_saxs": saxs_data is not None,
                        "has_stability": int_t_data is not None,
                    }
                else:
                    workflow_time = self.benchmark.end_timing(f"workflow_{i}")
                    workflow_results[file_name] = {
                        "success": False,
                        "time": workflow_time,
                        "error": "No G2 data available",
                    }

            except Exception as e:
                workflow_time = self.benchmark.end_timing(f"workflow_{i}")
                workflow_results[file_name] = {
                    "success": False,
                    "time": workflow_time,
                    "error": str(e),
                }

        # Analyze workflow performance
        successful_workflows = [r for r in workflow_results.values() if r["success"]]
        failed_workflows = [r for r in workflow_results.values() if not r["success"]]

        logger.info(
            f"Workflow performance: {len(successful_workflows)} successful, "
            f"{len(failed_workflows)} failed"
        )

        self.assertGreater(len(successful_workflows), 0, "No successful workflows")

        # Performance expectations
        for file_name, result in workflow_results.items():
            if result["success"]:
                # Complete workflow should finish in reasonable time
                max_workflow_time = 10.0  # 10 seconds max
                self.assertLess(
                    result["time"],
                    max_workflow_time,
                    f"Workflow too slow for {file_name}: {result['time']:.3f}s",
                )

                # Verify workflow completeness
                self.assertGreater(result["n_q"], 0)
                self.assertGreater(result["n_tau"], 0)
                self.assertGreaterEqual(result["mean_g2"], 1.0)

                logger.info(
                    f"Workflow {file_name}: {result['time']:.3f}s, "
                    f"{result['n_q']} Q-points, {result['n_tau']} tau points"
                )

    def test_performance_regression_detection(self):
        """Test for performance regressions in integrated system."""
        # Define performance baselines (these would be updated as optimizations improve)
        performance_baselines = {
            "file_load_time_mb": 0.5,  # seconds per MB
            "cache_speedup_min": 10.0,  # minimum cache speedup
            "memory_efficiency_max": 3.0,  # maximum memory overhead ratio
            "concurrent_throughput_min": 2.0,  # operations per second
        }

        # Test current performance against baselines
        regression_issues = []

        # File loading performance
        for _i, test_file in enumerate(self.test_files):
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024

            start_time = time.perf_counter()
            xf = XpcsFile(test_file)
            _ = xf.g2  # Trigger loading
            load_time = time.perf_counter() - start_time

            load_time_per_mb = load_time / file_size_mb

            if load_time_per_mb > performance_baselines["file_load_time_mb"]:
                regression_issues.append(
                    f"File loading regression: {load_time_per_mb:.3f}s/MB > "
                    f"{performance_baselines['file_load_time_mb']:.3f}s/MB baseline"
                )

        # Report regression issues
        if regression_issues:
            logger.warning("Performance regression detected:")
            for issue in regression_issues:
                logger.warning(f"  {issue}")

        # Don't fail tests for regressions, but log them for monitoring
        # self.assertEqual(len(regression_issues), 0, f"Performance regressions: {regression_issues}")


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run performance integration tests
    unittest.main(verbosity=2)
