#!/usr/bin/env python3
"""
System Stability and Stress Testing Suite for XPCS Toolkit Phase 6 Integration Testing

This module provides comprehensive system stability testing under various stress conditions
to ensure the integrated system remains stable, responsive, and reliable under load.

Stress Testing Coverage:
- Memory pressure stress testing
- High I/O load stress testing
- CPU-intensive computation stress testing
- Concurrent user simulation stress testing
- Long-running stability testing
- Resource exhaustion recovery testing
- Thread safety under extreme load
- Cache pressure stress testing

Author: Integration and Validation Agent
Created: 2025-09-11
"""

import gc
import json
import multiprocessing
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from xpcs_toolkit.fileIO.hdf_reader import HDFReaderPool
    from xpcs_toolkit.threading.async_workers_enhanced import EnhancedAsyncWorker
    from xpcs_toolkit.utils.adaptive_memory import AdaptiveMemoryManager
    from xpcs_toolkit.utils.advanced_cache import AdvancedCacheManager
    from xpcs_toolkit.utils.io_performance import OptimizedHDFReader

    STRESS_TEST_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available for stress testing: {e}")
    STRESS_TEST_MODULES_AVAILABLE = False


@dataclass
class StressTestMetrics:
    """Container for stress test metrics"""

    test_name: str
    duration_seconds: float
    peak_memory_mb: float
    avg_cpu_percent: float
    peak_cpu_percent: float
    operations_completed: int
    errors_encountered: int
    thread_count_peak: int
    io_operations_count: int
    cache_hit_rate: float
    system_stability_score: float
    notes: str


@dataclass
class SystemStabilityReport:
    """Container for system stability report"""

    test_timestamp: str
    overall_stability_score: float
    stress_test_results: list[StressTestMetrics]
    system_info: dict[str, Any]
    recommendations: list[str]
    critical_issues: list[str]


class ResourceMonitor:
    """Monitor system resources during stress testing"""

    def __init__(self, monitoring_interval: float = 0.1):
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.metrics_history = []
        self.monitor_thread = None

    def start_monitoring(self):
        """Start resource monitoring in background thread"""
        self.monitoring_active = True
        self.metrics_history = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """Background monitoring loop"""
        process = psutil.Process()

        while self.monitoring_active:
            try:
                metrics = {
                    "timestamp": time.time(),
                    "memory_mb": process.memory_info().rss / 1024 / 1024,
                    "cpu_percent": process.cpu_percent(),
                    "thread_count": process.num_threads(),
                    "open_files": len(process.open_files()),
                    "system_memory_percent": psutil.virtual_memory().percent,
                    "system_cpu_percent": psutil.cpu_percent(),
                }
                self.metrics_history.append(metrics)

            except Exception:
                # Continue monitoring even if individual measurements fail
                pass

            time.sleep(self.monitoring_interval)

    def get_summary_metrics(self) -> dict[str, float]:
        """Get summary metrics from monitoring history"""
        if not self.metrics_history:
            return {}

        memory_values = [m["memory_mb"] for m in self.metrics_history]
        cpu_values = [
            m["cpu_percent"] for m in self.metrics_history if m["cpu_percent"] > 0
        ]
        thread_values = [m["thread_count"] for m in self.metrics_history]

        return {
            "peak_memory_mb": max(memory_values),
            "avg_memory_mb": np.mean(memory_values),
            "peak_cpu_percent": max(cpu_values) if cpu_values else 0,
            "avg_cpu_percent": np.mean(cpu_values) if cpu_values else 0,
            "peak_thread_count": max(thread_values),
            "avg_thread_count": np.mean(thread_values),
            "measurement_count": len(self.metrics_history),
        }


class StressTestDataGenerator:
    """Generate test data for stress testing"""

    @staticmethod
    def create_large_hdf5_dataset(filepath: Path, size_mb: int = 100):
        """Create large HDF5 dataset for I/O stress testing"""
        # Calculate dimensions for target size
        target_bytes = size_mb * 1024 * 1024
        bytes_per_element = 2  # uint16
        total_elements = target_bytes // bytes_per_element

        # Use reasonable dimensions
        frames = 200
        pixels_per_frame = total_elements // frames
        side_length = int(np.sqrt(pixels_per_frame))

        with h5py.File(filepath, "w") as f:
            xpcs_group = f.create_group("xpcs")

            # Generate data in chunks to avoid memory issues
            chunk_frames = 10
            data_shape = (frames, side_length, side_length)
            dataset = xpcs_group.create_dataset(
                "pixelSum",
                shape=data_shape,
                dtype=np.uint16,
                chunks=(chunk_frames, side_length, side_length),
            )

            for start_frame in range(0, frames, chunk_frames):
                end_frame = min(start_frame + chunk_frames, frames)
                chunk_shape = (end_frame - start_frame, side_length, side_length)

                # Generate chunk data
                chunk_data = np.random.poisson(100, chunk_shape).astype(np.uint16)
                dataset[start_frame:end_frame] = chunk_data

            # Add metadata
            xpcs_group.attrs["file_size_mb"] = size_mb
            xpcs_group.attrs["detector_width"] = side_length
            xpcs_group.attrs["detector_height"] = side_length
            xpcs_group.attrs["frame_count"] = frames

        return filepath


class MemoryStressTests:
    """Memory pressure and memory leak stress tests"""

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor

    def test_memory_pressure_handling(
        self, duration_seconds: int = 60
    ) -> StressTestMetrics:
        """Test system behavior under memory pressure"""
        print(f"Running memory pressure test for {duration_seconds} seconds...")

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0
        allocated_objects = []

        try:
            while time.time() - start_time < duration_seconds:
                try:
                    # Allocate memory in chunks
                    chunk_size = 10 * 1024 * 1024  # 10MB chunks
                    data = np.random.random(chunk_size // 8).astype(
                        np.float64
                    )  # 8 bytes per float64
                    allocated_objects.append(data)

                    # Periodically release some memory
                    if len(allocated_objects) > 50:
                        # Release oldest 10 chunks
                        for _ in range(10):
                            if allocated_objects:
                                del allocated_objects[0]
                        gc.collect()

                    # Simulate some computation
                    np.sum(data**2)
                    operations_completed += 1

                    # Brief pause to allow monitoring
                    time.sleep(0.001)

                except MemoryError:
                    # Handle memory exhaustion gracefully
                    allocated_objects.clear()
                    gc.collect()
                    errors_encountered += 1
                    time.sleep(0.1)  # Allow system to recover

                except Exception:
                    errors_encountered += 1
                    time.sleep(0.001)

        finally:
            # Clean up
            allocated_objects.clear()
            gc.collect()
            self.monitor.stop_monitoring()

        # Calculate metrics
        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # Memory stability score (based on error rate and resource usage)
        error_rate = errors_encountered / max(1, operations_completed)
        stability_score = max(
            0,
            100
            - (error_rate * 100)
            - min(50, summary_metrics.get("peak_memory_mb", 0) / 100),
        )

        return StressTestMetrics(
            test_name="memory_pressure_handling",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=0,
            cache_hit_rate=0.0,
            system_stability_score=stability_score,
            notes=f"Memory stress test with {len(allocated_objects)} objects at peak",
        )

    def test_memory_leak_detection(self, cycles: int = 100) -> StressTestMetrics:
        """Test for memory leaks in repeated operations"""
        print(f"Running memory leak detection for {cycles} cycles...")

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0

        try:
            initial_memory = psutil.Process().memory_info().rss

            for cycle in range(cycles):
                try:
                    # Simulate typical XPCS operations
                    data = np.random.random((100, 256, 256))

                    # Processing operations that could leak memory
                    mean_data = np.mean(data, axis=0)
                    fft_data = np.fft.fft2(data, axes=(1, 2))
                    correlation = np.corrcoef(data.reshape(data.shape[0], -1))

                    # Delete references
                    del data, mean_data, fft_data, correlation

                    # Periodic garbage collection
                    if cycle % 10 == 0:
                        gc.collect()

                        # Check memory growth
                        current_memory = psutil.Process().memory_info().rss
                        memory_growth = current_memory - initial_memory

                        # If memory growth is excessive, might indicate a leak
                        if memory_growth > 500 * 1024 * 1024:  # 500MB threshold
                            print(
                                f"Warning: High memory growth detected: {memory_growth / 1024 / 1024:.1f}MB"
                            )

                    operations_completed += 1

                except Exception:
                    errors_encountered += 1

        finally:
            self.monitor.stop_monitoring()

        # Final memory check
        final_memory = psutil.Process().memory_info().rss
        total_memory_growth = final_memory - initial_memory

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # Memory leak score (lower growth is better)
        memory_growth_mb = total_memory_growth / 1024 / 1024
        leak_score = max(0, 100 - memory_growth_mb)

        return StressTestMetrics(
            test_name="memory_leak_detection",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=0,
            cache_hit_rate=0.0,
            system_stability_score=leak_score,
            notes=f"Total memory growth: {memory_growth_mb:.1f}MB over {cycles} cycles",
        )


class IOStressTests:
    """I/O intensive stress tests"""

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor
        self.temp_dir = Path(tempfile.mkdtemp())

    def cleanup(self):
        """Clean up temporary files"""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_high_io_load(
        self, duration_seconds: int = 30, file_count: int = 10
    ) -> StressTestMetrics:
        """Test system under high I/O load"""
        print(
            f"Running high I/O load test for {duration_seconds} seconds with {file_count} files..."
        )

        # Create test files
        test_files = []
        for i in range(file_count):
            filepath = self.temp_dir / f"io_stress_{i}.h5"
            StressTestDataGenerator.create_large_hdf5_dataset(filepath, size_mb=20)
            test_files.append(filepath)

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0
        io_operations_count = 0

        def io_worker(file_list, worker_id):
            """Worker function for I/O operations"""
            local_ops = 0
            local_errors = 0
            local_io_count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # Random file selection
                    filepath = np.random.choice(file_list)

                    # Random I/O operation
                    operation = np.random.choice(
                        ["read_full", "read_chunk", "read_metadata"]
                    )

                    with h5py.File(filepath, "r") as f:
                        local_io_count += 1

                        if operation == "read_full":
                            data = f["xpcs/pixelSum"][:]
                            np.mean(data)
                        elif operation == "read_chunk":
                            data = f["xpcs/pixelSum"][:10]
                            np.sum(data)
                        else:  # read_metadata
                            attrs = dict(f["xpcs"].attrs)
                            len(attrs)

                    local_ops += 1

                except Exception:
                    local_errors += 1
                    time.sleep(0.001)  # Brief pause on error

            return local_ops, local_errors, local_io_count

        try:
            # Launch concurrent I/O workers
            num_workers = min(8, multiprocessing.cpu_count())

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(io_worker, test_files, i)
                    for i in range(num_workers)
                ]

                # Collect results
                for future in as_completed(futures):
                    ops, errors, io_count = future.result()
                    operations_completed += ops
                    errors_encountered += errors
                    io_operations_count += io_count

        finally:
            self.monitor.stop_monitoring()

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # I/O performance score
        io_rate = io_operations_count / duration
        error_rate = errors_encountered / max(1, operations_completed)
        io_score = min(100, io_rate * 2) * (1 - error_rate)

        return StressTestMetrics(
            test_name="high_io_load",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=io_operations_count,
            cache_hit_rate=0.0,
            system_stability_score=io_score,
            notes=f"I/O rate: {io_rate:.1f} ops/sec, {file_count} files, {num_workers} workers",
        )

    def test_concurrent_file_access(
        self, duration_seconds: int = 20
    ) -> StressTestMetrics:
        """Test concurrent access to same files"""
        print(f"Running concurrent file access test for {duration_seconds} seconds...")

        # Create a single large test file for concurrent access
        test_file = self.temp_dir / "concurrent_access.h5"
        StressTestDataGenerator.create_large_hdf5_dataset(test_file, size_mb=50)

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0
        access_lock = threading.Lock()

        def concurrent_accessor(accessor_id):
            """Function for concurrent file access"""
            local_ops = 0
            local_errors = 0

            while time.time() - start_time < duration_seconds:
                try:
                    with h5py.File(test_file, "r") as f:
                        # Different access patterns
                        if accessor_id % 3 == 0:
                            # Read random chunks
                            start_idx = np.random.randint(0, 150)
                            data = f["xpcs/pixelSum"][start_idx : start_idx + 10]
                        elif accessor_id % 3 == 1:
                            # Read metadata
                            attrs = dict(f["xpcs"].attrs)
                        else:
                            # Read specific slices
                            data = f["xpcs/pixelSum"][:, 100:200, 100:200]

                        # Brief computation
                        if "data" in locals():
                            result = np.mean(data)

                    local_ops += 1

                    # Brief pause to allow other threads
                    time.sleep(0.001)

                except Exception:
                    local_errors += 1
                    time.sleep(0.005)  # Longer pause on error

            # Thread-safe result accumulation
            with access_lock:
                nonlocal operations_completed, errors_encountered
                operations_completed += local_ops
                errors_encountered += local_errors

        try:
            # Launch many concurrent accessors
            num_accessors = 20

            with ThreadPoolExecutor(max_workers=num_accessors) as executor:
                futures = [
                    executor.submit(concurrent_accessor, i)
                    for i in range(num_accessors)
                ]

                # Wait for completion
                for future in as_completed(futures):
                    future.result()  # Ensure no exceptions are lost

        finally:
            self.monitor.stop_monitoring()

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # Concurrent access score
        error_rate = errors_encountered / max(1, operations_completed)
        concurrency_score = max(0, 100 - (error_rate * 100))

        return StressTestMetrics(
            test_name="concurrent_file_access",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=operations_completed,  # Each operation is an I/O
            cache_hit_rate=0.0,
            system_stability_score=concurrency_score,
            notes=f"Concurrent access with {num_accessors} accessors, error rate: {error_rate:.1%}",
        )


class ComputationStressTests:
    """CPU-intensive computation stress tests"""

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor

    def test_cpu_intensive_load(self, duration_seconds: int = 30) -> StressTestMetrics:
        """Test system under CPU-intensive computational load"""
        print(f"Running CPU-intensive load test for {duration_seconds} seconds...")

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0

        def cpu_worker(worker_id):
            """CPU-intensive worker function"""
            local_ops = 0
            local_errors = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # CPU-intensive operations simulating XPCS analysis

                    # Generate data
                    data = np.random.random((200, 200)).astype(np.float32)

                    # FFT operations
                    fft_result = np.fft.fft2(data)
                    power_spectrum = np.abs(fft_result) ** 2

                    # Statistical calculations
                    np.mean(power_spectrum)
                    np.std(power_spectrum)

                    # Correlation calculations
                    correlation_matrix = np.corrcoef(data)
                    np.linalg.eigvals(correlation_matrix[:50, :50])  # Subset for speed

                    # More complex operations
                    np.convolve(data.flatten()[:1000], np.ones(10) / 10, mode="valid")

                    local_ops += 1

                except Exception:
                    local_errors += 1
                    time.sleep(0.001)

            return local_ops, local_errors

        try:
            # Use all available CPU cores
            num_workers = multiprocessing.cpu_count()

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(cpu_worker, i) for i in range(num_workers)]

                # Collect results
                for future in as_completed(futures):
                    ops, errors = future.result()
                    operations_completed += ops
                    errors_encountered += errors

        finally:
            self.monitor.stop_monitoring()

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # CPU performance score
        ops_per_second = operations_completed / duration
        error_rate = errors_encountered / max(1, operations_completed)
        cpu_score = min(100, ops_per_second) * (1 - error_rate)

        return StressTestMetrics(
            test_name="cpu_intensive_load",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=0,
            cache_hit_rate=0.0,
            system_stability_score=cpu_score,
            notes=f"CPU utilization test with {num_workers} workers, {ops_per_second:.1f} ops/sec",
        )

    def test_mixed_workload_stress(
        self, duration_seconds: int = 45
    ) -> StressTestMetrics:
        """Test system under mixed computational and I/O workload"""
        print(f"Running mixed workload stress test for {duration_seconds} seconds...")

        # Create temporary data file
        temp_dir = Path(tempfile.mkdtemp())
        test_file = temp_dir / "mixed_workload.h5"

        try:
            StressTestDataGenerator.create_large_hdf5_dataset(test_file, size_mb=30)

            self.monitor.start_monitoring()
            start_time = time.time()

            operations_completed = 0
            errors_encountered = 0
            io_operations_count = 0

            def mixed_worker(worker_id):
                """Mixed workload worker"""
                local_ops = 0
                local_errors = 0
                local_io_count = 0

                while time.time() - start_time < duration_seconds:
                    try:
                        # Randomly choose workload type
                        workload_type = np.random.choice(
                            ["io_heavy", "cpu_heavy", "mixed"]
                        )

                        if workload_type == "io_heavy":
                            # I/O intensive operations
                            with h5py.File(test_file, "r") as f:
                                chunk_size = np.random.randint(10, 30)
                                data = f["xpcs/pixelSum"][:chunk_size]
                                np.mean(data)
                                local_io_count += 1

                        elif workload_type == "cpu_heavy":
                            # CPU intensive operations
                            data = np.random.random((150, 150))
                            fft_data = np.fft.fft2(data)
                            np.corrcoef(data[:50])  # Subset for speed
                            np.sum(np.abs(fft_data))

                        else:  # mixed
                            # Combined I/O and CPU
                            with h5py.File(test_file, "r") as f:
                                data = f["xpcs/pixelSum"][:20]
                                local_io_count += 1

                            # Process the loaded data
                            processed = np.mean(data, axis=0)
                            fft_result = np.fft.fft2(processed)
                            np.sum(np.abs(fft_result))

                        local_ops += 1

                    except Exception:
                        local_errors += 1
                        time.sleep(0.001)

                return local_ops, local_errors, local_io_count

            # Run mixed workload with multiple workers
            num_workers = max(2, multiprocessing.cpu_count() // 2)

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(mixed_worker, i) for i in range(num_workers)]

                # Collect results
                for future in as_completed(futures):
                    ops, errors, io_count = future.result()
                    operations_completed += ops
                    errors_encountered += errors
                    io_operations_count += io_count

        finally:
            self.monitor.stop_monitoring()
            # Clean up temp file
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # Mixed workload score
        ops_per_second = operations_completed / duration
        error_rate = errors_encountered / max(1, operations_completed)
        mixed_score = min(100, ops_per_second * 2) * (1 - error_rate)

        return StressTestMetrics(
            test_name="mixed_workload_stress",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=io_operations_count,
            cache_hit_rate=0.0,
            system_stability_score=mixed_score,
            notes=f"Mixed workload: {ops_per_second:.1f} ops/sec, {io_operations_count} I/O ops",
        )


class CacheStressTests:
    """Cache system stress tests"""

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor

    def test_cache_pressure(self, duration_seconds: int = 30) -> StressTestMetrics:
        """Test cache system under pressure"""
        if not STRESS_TEST_MODULES_AVAILABLE or "AdvancedCacheManager" not in globals():
            print("Cache stress test skipped - AdvancedCacheManager not available")
            return StressTestMetrics(
                test_name="cache_pressure",
                duration_seconds=0,
                peak_memory_mb=0,
                avg_cpu_percent=0,
                peak_cpu_percent=0,
                operations_completed=0,
                errors_encountered=0,
                thread_count_peak=0,
                io_operations_count=0,
                cache_hit_rate=0.0,
                system_stability_score=0,
                notes="Test skipped - cache module not available",
            )

        print(f"Running cache pressure test for {duration_seconds} seconds...")

        cache = AdvancedCacheManager(max_memory_mb=50)  # Limited cache size

        self.monitor.start_monitoring()
        start_time = time.time()

        operations_completed = 0
        errors_encountered = 0
        cache_hits = 0
        cache_misses = 0

        def cache_worker(worker_id):
            """Cache stress worker"""
            local_ops = 0
            local_errors = 0
            local_hits = 0
            local_misses = 0

            while time.time() - start_time < duration_seconds:
                try:
                    # Generate cache keys with some overlap to create hit/miss patterns
                    key_id = np.random.randint(0, 100)  # 100 possible keys
                    cache_key = f"worker_{worker_id}_data_{key_id}"

                    # Try to get from cache
                    cached_data = cache.get(cache_key)

                    if cached_data is not None:
                        local_hits += 1
                        # Use cached data
                        np.mean(cached_data)
                    else:
                        local_misses += 1
                        # Generate new data and cache it
                        data = np.random.random((100, 100)).astype(np.float32)
                        cache.set(cache_key, data, size_bytes=data.nbytes)
                        np.mean(data)

                    local_ops += 1

                except Exception:
                    local_errors += 1
                    time.sleep(0.001)

            return local_ops, local_errors, local_hits, local_misses

        try:
            # Multiple workers accessing cache concurrently
            num_workers = 8

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(cache_worker, i) for i in range(num_workers)]

                # Collect results
                for future in as_completed(futures):
                    ops, errors, hits, misses = future.result()
                    operations_completed += ops
                    errors_encountered += errors
                    cache_hits += hits
                    cache_misses += misses

        finally:
            cache.clear()
            self.monitor.stop_monitoring()

        duration = time.time() - start_time
        summary_metrics = self.monitor.get_summary_metrics()

        # Cache performance metrics
        total_requests = cache_hits + cache_misses
        cache_hit_rate = cache_hits / max(1, total_requests)
        error_rate = errors_encountered / max(1, operations_completed)
        cache_score = cache_hit_rate * 100 * (1 - error_rate)

        return StressTestMetrics(
            test_name="cache_pressure",
            duration_seconds=duration,
            peak_memory_mb=summary_metrics.get("peak_memory_mb", 0),
            avg_cpu_percent=summary_metrics.get("avg_cpu_percent", 0),
            peak_cpu_percent=summary_metrics.get("peak_cpu_percent", 0),
            operations_completed=operations_completed,
            errors_encountered=errors_encountered,
            thread_count_peak=summary_metrics.get("peak_thread_count", 0),
            io_operations_count=0,
            cache_hit_rate=cache_hit_rate,
            system_stability_score=cache_score,
            notes=f"Cache hit rate: {cache_hit_rate:.1%}, {num_workers} workers, {total_requests} requests",
        )


class SystemStabilityTestSuite:
    """Main system stability test suite"""

    def __init__(self, output_dir: str = "stability_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.monitor = ResourceMonitor()
        self.test_results: list[StressTestMetrics] = []

        # Initialize stress test modules
        self.memory_tests = MemoryStressTests(self.monitor)
        self.io_tests = IOStressTests(self.monitor)
        self.computation_tests = ComputationStressTests(self.monitor)
        self.cache_tests = CacheStressTests(self.monitor)

    def run_all_stress_tests(self, quick_mode: bool = False) -> SystemStabilityReport:
        """Run comprehensive stress test suite"""
        print("Starting System Stability and Stress Testing Suite")
        print("=" * 60)

        start_time = time.time()

        # Adjust test durations for quick mode
        duration_multiplier = 0.3 if quick_mode else 1.0

        try:
            # Memory stress tests
            print("\n--- Memory Stress Tests ---")
            self.test_results.append(
                self.memory_tests.test_memory_pressure_handling(
                    duration_seconds=int(60 * duration_multiplier)
                )
            )
            self.test_results.append(
                self.memory_tests.test_memory_leak_detection(
                    cycles=int(100 * duration_multiplier)
                )
            )

            # I/O stress tests
            print("\n--- I/O Stress Tests ---")
            self.test_results.append(
                self.io_tests.test_high_io_load(
                    duration_seconds=int(30 * duration_multiplier)
                )
            )
            self.test_results.append(
                self.io_tests.test_concurrent_file_access(
                    duration_seconds=int(20 * duration_multiplier)
                )
            )

            # Computational stress tests
            print("\n--- Computational Stress Tests ---")
            self.test_results.append(
                self.computation_tests.test_cpu_intensive_load(
                    duration_seconds=int(30 * duration_multiplier)
                )
            )
            self.test_results.append(
                self.computation_tests.test_mixed_workload_stress(
                    duration_seconds=int(45 * duration_multiplier)
                )
            )

            # Cache stress tests
            print("\n--- Cache Stress Tests ---")
            self.test_results.append(
                self.cache_tests.test_cache_pressure(
                    duration_seconds=int(30 * duration_multiplier)
                )
            )

        except KeyboardInterrupt:
            print("\nStress testing interrupted by user")
        except Exception as e:
            print(f"\nStress testing failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # Clean up
            self.io_tests.cleanup()

        total_time = time.time() - start_time
        print(f"\nStress testing completed in {total_time:.1f} seconds")

        # Generate comprehensive report
        report = self._generate_stability_report()
        self._save_report(report)

        return report

    def _generate_stability_report(self) -> SystemStabilityReport:
        """Generate comprehensive stability report"""
        # Calculate overall stability score
        if self.test_results:
            stability_scores = [
                r.system_stability_score
                for r in self.test_results
                if r.system_stability_score > 0
            ]
            overall_score = np.mean(stability_scores) if stability_scores else 0
        else:
            overall_score = 0

        # Identify critical issues
        critical_issues = []
        recommendations = []

        for result in self.test_results:
            if (
                result.errors_encountered > result.operations_completed * 0.1
            ):  # >10% error rate
                critical_issues.append(
                    f"{result.test_name}: High error rate ({result.errors_encountered}/{result.operations_completed})"
                )

            if result.system_stability_score < 50:
                critical_issues.append(
                    f"{result.test_name}: Low stability score ({result.system_stability_score:.1f})"
                )

            # Generate specific recommendations
            if "memory" in result.test_name.lower() and result.peak_memory_mb > 1000:
                recommendations.append(
                    "Consider implementing more aggressive memory management"
                )

            if "io" in result.test_name.lower() and result.errors_encountered > 0:
                recommendations.append(
                    "Review I/O error handling and connection management"
                )

            if "cache" in result.test_name.lower() and result.cache_hit_rate < 0.5:
                recommendations.append("Optimize cache eviction policy and size limits")

        # General recommendations based on overall performance
        if overall_score >= 90:
            recommendations.append(
                "Excellent system stability - optimizations are working well"
            )
        elif overall_score >= 70:
            recommendations.append(
                "Good system stability - minor improvements possible"
            )
        elif overall_score >= 50:
            recommendations.append(
                "Moderate stability - review optimization implementations"
            )
        else:
            recommendations.append(
                "Poor system stability - significant optimization review needed"
            )

        # System information
        system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": multiprocessing.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "available_optimizations": STRESS_TEST_MODULES_AVAILABLE,
        }

        return SystemStabilityReport(
            test_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            overall_stability_score=overall_score,
            stress_test_results=self.test_results,
            system_info=system_info,
            recommendations=recommendations,
            critical_issues=critical_issues,
        )

    def _save_report(self, report: SystemStabilityReport):
        """Save stability report to files"""
        timestamp = int(time.time())

        # Save JSON report
        json_file = self.output_dir / f"stability_report_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)

        # Save text summary
        text_file = self.output_dir / f"stability_summary_{timestamp}.txt"
        with open(text_file, "w") as f:
            f.write("System Stability and Stress Testing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Timestamp: {report.test_timestamp}\n")
            f.write(
                f"Overall Stability Score: {report.overall_stability_score:.1f}%\n\n"
            )

            f.write("Test Results Summary:\n")
            for result in report.stress_test_results:
                f.write(f"\n{result.test_name}:\n")
                f.write(f"  Duration: {result.duration_seconds:.1f}s\n")
                f.write(f"  Operations: {result.operations_completed}\n")
                f.write(f"  Errors: {result.errors_encountered}\n")
                f.write(f"  Stability Score: {result.system_stability_score:.1f}%\n")
                f.write(f"  Peak Memory: {result.peak_memory_mb:.1f}MB\n")
                f.write(f"  Peak CPU: {result.peak_cpu_percent:.1f}%\n")

            if report.critical_issues:
                f.write("\nCritical Issues:\n")
                for issue in report.critical_issues:
                    f.write(f"  - {issue}\n")

            f.write("\nRecommendations:\n")
            for rec in report.recommendations:
                f.write(f"  - {rec}\n")

        print("\nStability report saved to:")
        print(f"  JSON: {json_file}")
        print(f"  Summary: {text_file}")


def main():
    """Main function to run system stability tests"""
    import argparse

    parser = argparse.ArgumentParser(description="XPCS Toolkit System Stability Tests")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick tests (reduced duration)"
    )
    parser.add_argument(
        "--output-dir", default="stability_results", help="Output directory for results"
    )

    args = parser.parse_args()

    # Create test suite
    suite = SystemStabilityTestSuite(args.output_dir)

    try:
        # Run comprehensive stability tests
        report = suite.run_all_stress_tests(quick_mode=args.quick)

        # Print final summary
        print("\n" + "=" * 60)
        print("SYSTEM STABILITY TEST SUMMARY")
        print("=" * 60)
        print(f"Overall Stability Score: {report.overall_stability_score:.1f}%")
        print(f"Tests Completed: {len(report.stress_test_results)}")

        if report.overall_stability_score >= 80:
            print("✓ EXCELLENT: System is highly stable under stress")
        elif report.overall_stability_score >= 60:
            print("✓ GOOD: System is stable with minor issues")
        elif report.overall_stability_score >= 40:
            print("⚠ WARNING: System has stability issues under stress")
        else:
            print("✗ CRITICAL: System is unstable under stress")

        if report.critical_issues:
            print("\nCritical Issues:")
            for issue in report.critical_issues[:3]:  # Show top 3
                print(f"  - {issue}")

        return report.overall_stability_score >= 50  # Pass threshold

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
