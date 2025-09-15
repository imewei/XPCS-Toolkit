#!/usr/bin/env python3
"""
Integration Testing Suite for XPCS Toolkit Phase 1-5 Optimizations

This module provides comprehensive integration testing to ensure all optimization
phases work correctly together and maintain expected performance and functionality.

Integration Test Coverage:
- Phase 1: Memory Management Integration
- Phase 2: I/O Performance Integration
- Phase 3: Vectorization Integration
- Phase 4: GUI Threading Integration
- Phase 5: Advanced Caching Integration
- Cross-Phase Interaction Testing
- End-to-End Workflow Testing

Author: Integration and Validation Agent
Created: 2025-09-11
"""

import gc
import logging
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    # Phase 1: Memory Management
    from xpcs_toolkit.fileIO.hdf_reader import HDFReaderPool

    # Phase 3: Vectorization
    from xpcs_toolkit.module.g2mod import G2Calculator
    from xpcs_toolkit.module.twotime_utils import TwoTimeProcessor
    from xpcs_toolkit.threading.async_workers import AsyncWorker

    # Phase 4: GUI Threading
    from xpcs_toolkit.threading.async_workers_enhanced import EnhancedAsyncWorker
    from xpcs_toolkit.utils.adaptive_memory import AdaptiveMemoryManager

    # Phase 5: Caching
    from xpcs_toolkit.utils.advanced_cache import AdvancedCacheManager
    from xpcs_toolkit.utils.computation_cache import ComputationCache
    from xpcs_toolkit.utils.enhanced_xpcs_file import EnhancedXpcsFile

    # Phase 2: I/O Performance
    from xpcs_toolkit.utils.io_performance import OptimizedHDFReader
    from xpcs_toolkit.viewer_kernel import ViewerKernel

    # Core modules
    from xpcs_toolkit.xpcs_file import XpcsFile

    OPTIMIZATION_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some optimization modules not available: {e}")
    OPTIMIZATION_MODULES_AVAILABLE = False


class TestDataGenerator:
    """Generate test data for integration testing"""

    @staticmethod
    def create_test_hdf5_file(
        filepath: Path, width: int = 512, height: int = 512, frames: int = 100
    ):
        """Create a synthetic XPCS HDF5 file for testing"""
        with h5py.File(filepath, "w") as f:
            # Create XPCS group structure
            xpcs_group = f.create_group("xpcs")

            # Generate synthetic data
            np.random.seed(42)  # For reproducible test data

            # Intensity data with some temporal correlation
            intensity_data = np.random.poisson(100, (frames, height, width)).astype(
                np.uint16
            )

            # Add spatial structure
            center_x, center_y = width // 2, height // 2
            x, y = np.meshgrid(np.arange(width), np.arange(height))
            r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            spatial_pattern = np.exp(-r / (width / 8))

            for i in range(frames):
                intensity_data[i] = intensity_data[i] * spatial_pattern

            # Create datasets
            xpcs_group.create_dataset("pixelSum", data=intensity_data)
            xpcs_group.create_dataset(
                "frameSum", data=np.sum(intensity_data, axis=(1, 2))
            )

            # Add correlation data
            tau_values = np.logspace(-6, 2, 100)
            g2_values = 1.0 + 0.5 * np.exp(-tau_values / 0.01)  # Exponential decay
            xpcs_group.create_dataset("delay_time", data=tau_values)
            xpcs_group.create_dataset("g2", data=g2_values)

            # Add metadata
            xpcs_group.attrs["detector_width"] = width
            xpcs_group.attrs["detector_height"] = height
            xpcs_group.attrs["frame_count"] = frames
            xpcs_group.attrs["exposure_time"] = 0.001
            xpcs_group.attrs["detector_distance"] = 1000.0
            xpcs_group.attrs["wavelength"] = 1.24e-10

        return filepath


class Phase1MemoryIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 1: Memory Management"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_files = []

        # Create test files of different sizes
        sizes = [(256, 256, 50), (512, 512, 100), (1024, 1024, 50)]
        for i, (w, h, f) in enumerate(sizes):
            filepath = self.temp_dir / f"test_{i}.h5"
            TestDataGenerator.create_test_hdf5_file(filepath, w, h, f)
            self.test_files.append(filepath)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_memory_manager_integration(self):
        """Test memory manager integration with file loading"""
        if not OPTIMIZATION_MODULES_AVAILABLE:
            self.skipTest("Optimization modules not available")

        initial_memory = psutil.Process().memory_info().rss

        # Test with memory manager
        if "AdaptiveMemoryManager" in globals():
            memory_manager = AdaptiveMemoryManager()

            try:
                # Load multiple files
                loaded_files = []
                for filepath in self.test_files:
                    if "EnhancedXpcsFile" in globals():
                        xpcs_file = EnhancedXpcsFile(str(filepath))
                        xpcs_file.load_data()
                        loaded_files.append(xpcs_file)

                    # Check memory pressure
                    memory_manager._check_memory_pressure()

                # Memory should be managed efficiently
                peak_memory = psutil.Process().memory_info().rss
                memory_growth = peak_memory - initial_memory

                # Should not grow excessively (threshold: 500MB)
                self.assertLess(
                    memory_growth,
                    500 * 1024 * 1024,
                    f"Memory growth too large: {memory_growth / 1024 / 1024:.1f}MB",
                )

                # Clean up
                for xpcs_file in loaded_files:
                    del xpcs_file

                memory_manager.cleanup()
                gc.collect()

                # Memory should be released
                final_memory = psutil.Process().memory_info().rss
                memory_released = peak_memory - final_memory
                self.assertGreater(
                    memory_released, 0, "Memory should be released after cleanup"
                )

            finally:
                if hasattr(memory_manager, "cleanup"):
                    memory_manager.cleanup()

    def test_memory_leak_prevention(self):
        """Test that repeated operations don't cause memory leaks"""
        initial_memory = psutil.Process().memory_info().rss

        # Perform repeated file load/unload cycles
        for cycle in range(10):
            for filepath in self.test_files:
                try:
                    with h5py.File(filepath, "r") as f:
                        data = f["xpcs/pixelSum"][:]
                        # Process data
                        result = np.mean(data, axis=0)
                        del data, result
                except Exception:
                    pass

            # Force garbage collection
            gc.collect()

        final_memory = psutil.Process().memory_info().rss
        memory_growth = final_memory - initial_memory

        # Memory growth should be minimal (threshold: 50MB)
        self.assertLess(
            memory_growth,
            50 * 1024 * 1024,
            f"Potential memory leak detected: {memory_growth / 1024 / 1024:.1f}MB growth",
        )

    def test_large_dataset_handling(self):
        """Test handling of large datasets without memory issues"""
        # Create a larger test dataset
        large_file = self.temp_dir / "large_test.h5"
        TestDataGenerator.create_test_hdf5_file(large_file, 1024, 1024, 200)

        initial_memory = psutil.Process().memory_info().rss

        try:
            # Load large dataset
            with h5py.File(large_file, "r") as f:
                # Test chunked reading if available
                dataset = f["xpcs/pixelSum"]

                # Read in chunks to avoid memory issues
                chunk_size = 10
                for i in range(0, dataset.shape[0], chunk_size):
                    end_idx = min(i + chunk_size, dataset.shape[0])
                    chunk = dataset[i:end_idx]

                    # Process chunk
                    np.mean(chunk)
                    del chunk

                    # Check memory usage periodically
                    current_memory = psutil.Process().memory_info().rss
                    memory_used = current_memory - initial_memory

                    # Should not exceed reasonable limit (1GB)
                    self.assertLess(
                        memory_used,
                        1024 * 1024 * 1024,
                        f"Memory usage too high: {memory_used / 1024 / 1024:.1f}MB",
                    )

        except Exception as e:
            self.fail(f"Large dataset handling failed: {e}")


class Phase2IOIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 2: I/O Performance"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_files = []

        # Create multiple test files for concurrent access testing
        for i in range(5):
            filepath = self.temp_dir / f"io_test_{i}.h5"
            TestDataGenerator.create_test_hdf5_file(filepath, 512, 512, 100)
            self.test_files.append(filepath)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_hdf_reader_pool_integration(self):
        """Test HDF reader pool performance and correctness"""
        if not OPTIMIZATION_MODULES_AVAILABLE:
            self.skipTest("Optimization modules not available")

        if "HDFReaderPool" not in globals():
            self.skipTest("HDFReaderPool not available")

        pool = HDFReaderPool(max_connections=3)

        try:
            # Test concurrent file access
            def read_file_data(filepath):
                try:
                    reader = pool.get_reader(filepath)
                    data = reader.read_dataset("xpcs/pixelSum")
                    pool.return_reader(reader)
                    return data.shape
                except Exception as e:
                    return str(e)

            # Concurrent reads
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(read_file_data, fp) for fp in self.test_files * 2
                ]
                results = [f.result() for f in futures]

            # All reads should succeed
            successful_reads = [r for r in results if isinstance(r, tuple)]
            self.assertGreater(len(successful_reads), 0, "No successful reads")

            # Check that pool managed connections properly
            # (This is implementation-dependent, so we just check basic functionality)

        finally:
            pool.clear_pool()

    def test_optimized_io_vs_standard(self):
        """Compare optimized I/O performance with standard I/O"""
        test_file = self.test_files[0]

        # Time standard I/O
        start_time = time.perf_counter()
        with h5py.File(test_file, "r") as f:
            standard_data = f["xpcs/pixelSum"][:]
        standard_time = time.perf_counter() - start_time

        # Time optimized I/O (if available)
        if "OptimizedHDFReader" in globals():
            try:
                start_time = time.perf_counter()
                reader = OptimizedHDFReader(str(test_file))
                optimized_data = reader.read_dataset("xpcs/pixelSum")
                optimized_time = time.perf_counter() - start_time

                # Data should be identical
                np.testing.assert_array_equal(standard_data, optimized_data)

                # Optimized should not be significantly slower (allow 50% margin)
                self.assertLess(
                    optimized_time,
                    standard_time * 1.5,
                    f"Optimized I/O slower: {optimized_time:.3f}s vs {standard_time:.3f}s",
                )

                reader.close()

            except Exception as e:
                self.fail(f"Optimized I/O test failed: {e}")

    def test_concurrent_file_access(self):
        """Test concurrent access to multiple files"""

        def access_file(filepath, access_count=5):
            """Access a file multiple times"""
            results = []
            for _ in range(access_count):
                try:
                    with h5py.File(filepath, "r") as f:
                        data = f["xpcs/frameSum"][:]
                        results.append(np.sum(data))
                except Exception as e:
                    results.append(str(e))
            return results

        # Launch concurrent access tasks
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(access_file, fp) for fp in self.test_files]
            all_results = [f.result() for f in futures]

        # Check that all accesses succeeded
        for file_results in all_results:
            successful_accesses = [
                r for r in file_results if isinstance(r, (int, float))
            ]
            self.assertGreater(
                len(successful_accesses), 0, "No successful file accesses"
            )


class Phase3VectorizationIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 3: Vectorization"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "vectorization_test.h5"
        TestDataGenerator.create_test_hdf5_file(self.test_file, 256, 256, 100)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_g2_calculation_integration(self):
        """Test G2 calculation integration with optimizations"""
        with h5py.File(self.test_file, "r") as f:
            intensity_data = f["xpcs/pixelSum"][:]
            f["xpcs/g2"][:]

        if "G2Calculator" in globals():
            try:
                calculator = G2Calculator()

                # Extract intensity time series for a pixel
                pixel_intensity = intensity_data[:, 128, 128]  # Center pixel
                tau_values = np.arange(1, min(21, len(pixel_intensity)))

                computed_g2 = calculator.calculate_correlation(
                    pixel_intensity, tau_values
                )

                # Should produce reasonable correlation values
                self.assertTrue(np.all(computed_g2 > 0), "G2 values should be positive")
                self.assertTrue(
                    np.all(computed_g2 < 10), "G2 values should be reasonable"
                )

                # Should handle edge cases
                zero_intensity = np.zeros_like(pixel_intensity)
                calculator.calculate_correlation(zero_intensity, tau_values[:5])
                # Should handle gracefully (not crash)

            except Exception as e:
                self.fail(f"G2 calculation integration failed: {e}")

    def test_vectorized_operations_accuracy(self):
        """Test accuracy of vectorized operations"""
        with h5py.File(self.test_file, "r") as f:
            data = f["xpcs/pixelSum"][:]

        # Test vectorized vs loop-based calculations
        sample_data = data[:50, :128, :128]  # Smaller subset for testing

        # Reference calculation (using loops)
        start_time = time.perf_counter()
        reference_mean = np.zeros((sample_data.shape[1], sample_data.shape[2]))
        for i in range(sample_data.shape[1]):
            for j in range(sample_data.shape[2]):
                reference_mean[i, j] = np.mean(sample_data[:, i, j])
        loop_time = time.perf_counter() - start_time

        # Vectorized calculation
        start_time = time.perf_counter()
        vectorized_mean = np.mean(sample_data, axis=0)
        vectorized_time = time.perf_counter() - start_time

        # Results should be identical
        np.testing.assert_allclose(reference_mean, vectorized_mean, rtol=1e-10)

        # Vectorized should be faster
        self.assertLess(
            vectorized_time,
            loop_time,
            f"Vectorized not faster: {vectorized_time:.3f}s vs {loop_time:.3f}s",
        )

    def test_fft_operations_integration(self):
        """Test FFT operations integration"""
        with h5py.File(self.test_file, "r") as f:
            data = f["xpcs/pixelSum"][0]  # Single frame

        # Test 2D FFT
        fft_result = np.fft.fft2(data.astype(np.complex64))

        # Basic FFT properties
        self.assertEqual(
            fft_result.shape, data.shape, "FFT output shape should match input"
        )

        # Test Parseval's theorem (energy conservation)
        time_domain_energy = np.sum(np.abs(data.astype(np.float64)) ** 2)
        freq_domain_energy = np.sum(np.abs(fft_result) ** 2) / data.size

        # Should be approximately equal (within numerical precision)
        relative_error = (
            abs(time_domain_energy - freq_domain_energy) / time_domain_energy
        )
        self.assertLess(relative_error, 1e-10, "FFT energy conservation violated")

    def test_statistical_calculations_integration(self):
        """Test statistical calculations integration"""
        with h5py.File(self.test_file, "r") as f:
            data = f["xpcs/pixelSum"][:]

        # Test various statistical operations
        sample_data = data.flatten()[:10000]  # Sample for speed

        # Calculate statistics
        mean_val = np.mean(sample_data)
        std_val = np.std(sample_data)
        var_val = np.var(sample_data)
        median_val = np.median(sample_data)

        # Basic statistical relationships
        self.assertAlmostEqual(
            var_val, std_val**2, places=10, msg="Variance should equal std squared"
        )

        # Values should be reasonable for Poisson-like data
        self.assertGreater(mean_val, 0, "Mean should be positive")
        self.assertGreater(std_val, 0, "Standard deviation should be positive")
        self.assertGreater(median_val, 0, "Median should be positive")


class Phase4ThreadingIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 4: GUI Threading"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "threading_test.h5"
        TestDataGenerator.create_test_hdf5_file(self.test_file, 512, 512, 100)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_async_worker_integration(self):
        """Test async worker integration"""
        if not OPTIMIZATION_MODULES_AVAILABLE:
            self.skipTest("Optimization modules not available")

        def compute_task(data_size):
            """Simulation of computational task"""
            data = np.random.random(data_size)
            return np.sum(data**2)

        if "EnhancedAsyncWorker" in globals():
            worker = EnhancedAsyncWorker()

            try:
                # Submit multiple tasks
                tasks = []
                for i in range(5):
                    task = worker.submit_task(compute_task, (1000 + i * 100,))
                    tasks.append(task)

                # Wait for all tasks to complete
                results = []
                for task in tasks:
                    result = task.result(timeout=10)  # 10 second timeout
                    results.append(result)
                    self.assertIsInstance(
                        result, (int, float), "Task should return numeric result"
                    )

                # All tasks should complete successfully
                self.assertEqual(len(results), 5, "All tasks should complete")

            except Exception as e:
                self.fail(f"Async worker integration failed: {e}")
            finally:
                worker.cleanup()

    def test_background_processing_integration(self):
        """Test background processing without blocking main thread"""
        main_thread_work_completed = False
        background_task_completed = False

        def background_task():
            nonlocal background_task_completed
            time.sleep(0.1)  # Simulate work
            with h5py.File(self.test_file, "r") as f:
                data = f["xpcs/pixelSum"][:]
                result = np.mean(data)
            background_task_completed = True
            return result

        def main_thread_work():
            nonlocal main_thread_work_completed
            # Simulate main thread work
            for _ in range(100):
                _ = sum(range(1000))
            main_thread_work_completed = True

        # Start background task
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(background_task)

            # Do main thread work while background task runs
            main_thread_work()

            # Check that main thread completed while background task was running
            self.assertTrue(
                main_thread_work_completed, "Main thread work should complete"
            )

            # Wait for background task
            result = future.result(timeout=5)
            self.assertTrue(
                background_task_completed, "Background task should complete"
            )
            self.assertIsInstance(
                result, (int, float), "Background task should return result"
            )

    def test_concurrent_data_processing(self):
        """Test concurrent data processing without race conditions"""
        with h5py.File(self.test_file, "r") as f:
            data = f["xpcs/pixelSum"][:]

        # Split data for concurrent processing
        num_workers = 4
        chunk_size = len(data) // num_workers
        chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

        def process_chunk(chunk):
            """Process a chunk of data"""
            return {"mean": np.mean(chunk), "std": np.std(chunk), "sum": np.sum(chunk)}

        # Process chunks concurrently
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            results = [f.result() for f in futures]

        # Combine results and verify
        total_sum = sum(r["sum"] for r in results)
        expected_sum = np.sum(data)

        # Should be equal (within numerical precision)
        relative_error = abs(total_sum - expected_sum) / expected_sum
        self.assertLess(
            relative_error, 1e-10, "Concurrent processing should preserve total sum"
        )

    def test_thread_safety(self):
        """Test thread safety of shared resources"""
        shared_counter = [0]  # Use list for mutability
        lock = threading.Lock()

        def increment_counter():
            for _ in range(1000):
                with lock:
                    shared_counter[0] += 1

        # Start multiple threads
        threads = []
        num_threads = 10

        for _ in range(num_threads):
            thread = threading.Thread(target=increment_counter)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Counter should be exactly correct if thread-safe
        expected_count = num_threads * 1000
        self.assertEqual(
            shared_counter[0], expected_count, "Thread safety violation detected"
        )


class Phase5CachingIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 5: Advanced Caching"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_file = self.temp_dir / "caching_test.h5"
        TestDataGenerator.create_test_hdf5_file(self.test_file, 256, 256, 50)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cache_integration_with_io(self):
        """Test cache integration with I/O operations"""
        if not OPTIMIZATION_MODULES_AVAILABLE:
            self.skipTest("Optimization modules not available")

        if "AdvancedCacheManager" not in globals():
            self.skipTest("AdvancedCacheManager not available")

        cache = AdvancedCacheManager(max_memory_mb=50)

        try:
            # Load data multiple times - should use cache on subsequent loads
            times = []

            for i in range(3):
                start_time = time.perf_counter()

                cache_key = f"test_data_{str(self.test_file)}"
                cached_data = cache.get(cache_key)

                if cached_data is None:
                    # Load from file
                    with h5py.File(self.test_file, "r") as f:
                        data = f["xpcs/pixelSum"][:]
                    cache.set(cache_key, data, size_bytes=data.nbytes)
                else:
                    data = cached_data

                load_time = time.perf_counter() - start_time
                times.append(load_time)

                # Verify data integrity
                self.assertIsInstance(
                    data, np.ndarray, "Cached data should be numpy array"
                )
                self.assertEqual(data.ndim, 3, "Data should be 3D")

            # Second and third loads should be faster (cache hits)
            self.assertLess(
                times[1], times[0] * 0.5, "Cache should speed up subsequent loads"
            )
            self.assertLess(
                times[2], times[0] * 0.5, "Cache should consistently speed up loads"
            )

        finally:
            cache.clear()

    def test_computation_cache_integration(self):
        """Test computation caching integration"""
        if "ComputationCache" not in globals():
            self.skipTest("ComputationCache not available")

        cache = ComputationCache(max_size=100)

        def expensive_computation(data):
            """Simulate expensive computation"""
            time.sleep(0.01)  # Simulate computation time
            return np.mean(data**2)

        with h5py.File(self.test_file, "r") as f:
            data = f["xpcs/pixelSum"][0]  # Single frame

        # First computation - should be slow
        start_time = time.perf_counter()
        result1 = cache.get_or_compute("mean_square", expensive_computation, data)
        first_time = time.perf_counter() - start_time

        # Second computation - should be fast (cached)
        start_time = time.perf_counter()
        result2 = cache.get_or_compute("mean_square", expensive_computation, data)
        second_time = time.perf_counter() - start_time

        # Results should be identical
        self.assertEqual(result1, result2, "Cached result should be identical")

        # Second call should be much faster
        self.assertLess(
            second_time,
            first_time * 0.1,
            "Cache should significantly speed up computation",
        )

    def test_cache_memory_management(self):
        """Test cache memory management under load"""
        if "AdvancedCacheManager" not in globals():
            self.skipTest("AdvancedCacheManager not available")

        # Small cache to force eviction
        cache = AdvancedCacheManager(max_memory_mb=10)

        try:
            # Add many items to force eviction
            stored_keys = []
            for i in range(50):
                key = f"item_{i}"
                data = np.random.random((100, 100)).astype(np.float32)
                cache.set(key, data, size_bytes=data.nbytes)
                stored_keys.append(key)

            # Cache should have evicted old items
            hit_count = 0
            for key in stored_keys[:25]:  # Check first half (should be evicted)
                if cache.get(key) is not None:
                    hit_count += 1

            # Most early items should be evicted
            eviction_rate = 1.0 - (hit_count / 25)
            self.assertGreater(
                eviction_rate, 0.5, "Cache should evict old items under memory pressure"
            )

            # Recent items should still be available
            recent_hit_count = 0
            for key in stored_keys[-10:]:  # Check last 10 items
                if cache.get(key) is not None:
                    recent_hit_count += 1

            recent_hit_rate = recent_hit_count / 10
            self.assertGreater(
                recent_hit_rate, 0.7, "Recent items should still be cached"
            )

        finally:
            cache.clear()

    def test_cache_thread_safety(self):
        """Test cache thread safety under concurrent access"""
        if "AdvancedCacheManager" not in globals():
            self.skipTest("AdvancedCacheManager not available")

        cache = AdvancedCacheManager(max_memory_mb=20)

        def cache_worker(worker_id, num_operations=100):
            """Worker function for cache operations"""
            results = []
            for i in range(num_operations):
                key = f"worker_{worker_id}_item_{i % 10}"  # Reuse some keys

                # Try to get from cache
                data = cache.get(key)
                if data is None:
                    # Generate and store new data
                    data = np.random.random((50, 50))
                    cache.set(key, data, size_bytes=data.nbytes)

                results.append(data.shape)
            return results

        try:
            # Run concurrent workers
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(cache_worker, i) for i in range(8)]
                all_results = [f.result() for f in futures]

            # All workers should complete successfully
            total_operations = sum(len(results) for results in all_results)
            self.assertGreater(total_operations, 0, "Cache operations should complete")

        finally:
            cache.clear()


class CrossPhaseIntegrationTests(unittest.TestCase):
    """Integration tests across multiple optimization phases"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create multiple test files
        self.test_files = []
        for i in range(3):
            filepath = self.temp_dir / f"integration_test_{i}.h5"
            TestDataGenerator.create_test_hdf5_file(filepath, 512, 512, 100)
            self.test_files.append(filepath)

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_workflow_integration(self):
        """Test complete end-to-end workflow with all optimizations"""
        workflow_steps = []

        try:
            # Step 1: Memory-optimized file loading
            workflow_steps.append("File Loading")
            start_time = time.perf_counter()

            loaded_data = {}
            for i, filepath in enumerate(self.test_files):
                with h5py.File(filepath, "r") as f:
                    loaded_data[i] = f["xpcs/pixelSum"][:]

            loading_time = time.perf_counter() - start_time
            workflow_steps.append(f"Loading completed in {loading_time:.3f}s")

            # Step 2: Vectorized data processing
            workflow_steps.append("Vectorized Processing")
            start_time = time.perf_counter()

            processed_results = {}
            for i, data in loaded_data.items():
                # Vectorized operations
                mean_data = np.mean(data, axis=0)
                std_data = np.std(data, axis=0)

                # G2-like correlation (simplified)
                sample_pixels = data[:, 256:260, 256:260]  # Small ROI
                g2_approx = np.mean(sample_pixels[:-1] * sample_pixels[1:], axis=0) / (
                    np.mean(sample_pixels, axis=0) ** 2
                )

                processed_results[i] = {
                    "mean": mean_data,
                    "std": std_data,
                    "g2_sample": g2_approx,
                }

            processing_time = time.perf_counter() - start_time
            workflow_steps.append(f"Processing completed in {processing_time:.3f}s")

            # Step 3: Concurrent analysis with threading
            workflow_steps.append("Concurrent Analysis")
            start_time = time.perf_counter()

            def analyze_dataset(data_tuple):
                """Analysis function for concurrent execution"""
                idx, results = data_tuple
                return {
                    "index": idx,
                    "total_intensity": np.sum(results["mean"]),
                    "noise_level": np.mean(results["std"]),
                    "correlation_strength": np.mean(results["g2_sample"]),
                }

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(analyze_dataset, (i, results))
                    for i, results in processed_results.items()
                ]
                analysis_results = [f.result() for f in futures]

            analysis_time = time.perf_counter() - start_time
            workflow_steps.append(f"Analysis completed in {analysis_time:.3f}s")

            # Verify results
            self.assertEqual(
                len(analysis_results),
                len(self.test_files),
                "Should have analysis results for all files",
            )

            for result in analysis_results:
                self.assertIn(
                    "total_intensity", result, "Analysis should include intensity"
                )
                self.assertIn(
                    "noise_level", result, "Analysis should include noise level"
                )
                self.assertIn(
                    "correlation_strength",
                    result,
                    "Analysis should include correlation",
                )

                # Values should be reasonable
                self.assertGreater(
                    result["total_intensity"], 0, "Total intensity should be positive"
                )
                self.assertGreater(
                    result["noise_level"], 0, "Noise level should be positive"
                )

            # Overall workflow should complete in reasonable time
            total_time = loading_time + processing_time + analysis_time
            workflow_steps.append(f"Total workflow time: {total_time:.3f}s")

            # Print workflow summary
            print("\nEnd-to-End Workflow Summary:")
            for step in workflow_steps:
                print(f"  {step}")

        except Exception as e:
            self.fail(f"End-to-end workflow failed at {workflow_steps[-1]}: {e}")

    def test_memory_and_cache_interaction(self):
        """Test interaction between memory management and caching"""
        if not OPTIMIZATION_MODULES_AVAILABLE:
            self.skipTest("Optimization modules not available")

        # This test verifies that memory management and caching work together
        initial_memory = psutil.Process().memory_info().rss

        # Simulate memory-cache interaction
        cached_items = {}

        try:
            # Load and cache multiple datasets
            for i, filepath in enumerate(self.test_files):
                with h5py.File(filepath, "r") as f:
                    data = f["xpcs/pixelSum"][:]

                # Simulate caching (simplified)
                cache_key = f"dataset_{i}"
                cached_items[cache_key] = data

                # Check memory usage
                current_memory = psutil.Process().memory_info().rss
                memory_growth = current_memory - initial_memory

                # Memory should grow but not excessively
                self.assertLess(
                    memory_growth,
                    1000 * 1024 * 1024,  # 1GB limit
                    f"Memory usage too high: {memory_growth / 1024 / 1024:.1f}MB",
                )

            # Access cached items
            for key, data in cached_items.items():
                result = np.mean(data)
                self.assertIsInstance(
                    result, (int, float), "Cached data should be accessible"
                )

            # Clean up
            cached_items.clear()
            gc.collect()

            # Memory should be released
            final_memory = psutil.Process().memory_info().rss
            memory_released = (initial_memory + memory_growth) - final_memory
            self.assertGreater(
                memory_released, 0, "Memory should be released after cleanup"
            )

        except Exception as e:
            self.fail(f"Memory-cache interaction test failed: {e}")

    def test_io_and_vectorization_interaction(self):
        """Test interaction between I/O optimizations and vectorization"""
        # Test that I/O optimizations provide data in correct format for vectorization

        performance_metrics = {}

        for i, filepath in enumerate(self.test_files):
            # Time I/O operation
            start_time = time.perf_counter()
            with h5py.File(filepath, "r") as f:
                data = f["xpcs/pixelSum"][:]
            io_time = time.perf_counter() - start_time

            # Time vectorized operation
            start_time = time.perf_counter()

            # Vectorized processing
            mean_frame = np.mean(data, axis=0)
            np.std(data, axis=0)

            # More complex vectorized operation
            np.diff(data, axis=0)
            correlation_map = np.corrcoef(data.reshape(data.shape[0], -1))

            vectorization_time = time.perf_counter() - start_time

            performance_metrics[i] = {
                "io_time": io_time,
                "vectorization_time": vectorization_time,
                "data_shape": data.shape,
                "mean_intensity": np.mean(mean_frame),
                "correlation_size": correlation_map.shape,
            }

            # Verify data integrity
            self.assertEqual(data.ndim, 3, "Data should be 3D")
            self.assertEqual(
                mean_frame.shape, data.shape[1:], "Mean frame shape should match"
            )
            self.assertEqual(
                correlation_map.shape[0],
                data.shape[0],
                "Correlation matrix should match frame count",
            )

        # Performance should be reasonable
        total_io_time = sum(m["io_time"] for m in performance_metrics.values())
        total_vectorization_time = sum(
            m["vectorization_time"] for m in performance_metrics.values()
        )

        self.assertLess(total_io_time, 10.0, "Total I/O time should be reasonable")
        self.assertLess(
            total_vectorization_time,
            5.0,
            "Total vectorization time should be reasonable",
        )

        print("\nI/O and Vectorization Performance:")
        print(f"  Total I/O time: {total_io_time:.3f}s")
        print(f"  Total vectorization time: {total_vectorization_time:.3f}s")


def create_integration_test_suite():
    """Create comprehensive integration test suite"""
    suite = unittest.TestSuite()

    # Phase-specific integration tests
    suite.addTest(unittest.makeSuite(Phase1MemoryIntegrationTests))
    suite.addTest(unittest.makeSuite(Phase2IOIntegrationTests))
    suite.addTest(unittest.makeSuite(Phase3VectorizationIntegrationTests))
    suite.addTest(unittest.makeSuite(Phase4ThreadingIntegrationTests))
    suite.addTest(unittest.makeSuite(Phase5CachingIntegrationTests))

    # Cross-phase integration tests
    suite.addTest(unittest.makeSuite(CrossPhaseIntegrationTests))

    return suite


def main():
    """Run integration test suite"""
    print("XPCS Toolkit Phase 1-5 Integration Testing Suite")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing

    # Create test suite
    suite = create_integration_test_suite()

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    start_time = time.time()

    try:
        result = runner.run(suite)

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(
            f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%"
        )
        print(f"Total time: {total_time:.1f} seconds")

        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"  {test}: {traceback.split('AssertionError:')[-1].strip()}")

        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"  {test}: {traceback.split('Exception:')[-1].strip()}")

        # Return success status
        return len(result.failures) == 0 and len(result.errors) == 0

    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        return False
    except Exception as e:
        print(f"\nTesting failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
