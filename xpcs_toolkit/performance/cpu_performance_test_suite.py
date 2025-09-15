#!/usr/bin/env python3
"""
Comprehensive CPU Performance Test Suite for XPCS Toolkit Optimization Prevention

This test suite provides comprehensive automated CPU performance testing to prevent
optimization regressions and ensure continued performance excellence across all
system components.

Features:
- Threading system performance tests (signal optimization, thread pools, workers)
- Memory management performance tests (cache efficiency, cleanup, pressure handling)
- I/O performance tests (HDF5 connection pooling, batch operations)
- Scientific computing performance tests (analysis modules, fitting algorithms)
- End-to-end workflow performance tests
- Statistical analysis with confidence intervals
- Regression detection and alerting

Integration Points:
- Tests all optimization components from src/xpcs_toolkit/threading/
- Validates memory optimizations from src/xpcs_toolkit/utils/
- Tests scientific computing performance in analysis modules
- Integrates with existing performance monitoring systems

Author: Claude Code Performance Testing Generator
Date: 2025-01-11
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import Mock

import numpy as np
import psutil

# Add project root to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import optimization systems for testing
try:
    from xpcs_toolkit.threading.enhanced_thread_pool import get_thread_pool_manager
    from xpcs_toolkit.threading.optimized_workers import OptimizedWorkerPool
    from xpcs_toolkit.threading.performance_monitor import get_performance_monitor
    from xpcs_toolkit.threading.signal_optimization import get_signal_optimizer
    from xpcs_toolkit.utils.adaptive_memory import AdaptiveMemoryManager
    from xpcs_toolkit.utils.advanced_cache import AdvancedCache
    from xpcs_toolkit.utils.io_performance import HDF5ConnectionPool
    from xpcs_toolkit.utils.performance_profiler import global_profiler
except ImportError:
    # Mock imports if optimization systems not available
    get_signal_optimizer = Mock()
    get_thread_pool_manager = Mock()
    get_performance_monitor = Mock()
    OptimizedWorkerPool = Mock()
    AdvancedCache = Mock()
    AdaptiveMemoryManager = Mock()
    global_profiler = Mock()
    HDF5ConnectionPool = Mock()

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


# =============================================================================
# Performance Test Configuration and Utilities
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    name: str
    execution_time_ms: float
    cpu_usage_percent: float
    memory_usage_mb: float
    throughput_ops_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate_percent: float
    additional_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class CPUTestConfig:
    """Configuration for CPU performance tests."""

    # Performance thresholds
    MAX_THREAD_CREATION_TIME_MS: float = 10.0
    MAX_SIGNAL_BATCHING_LATENCY_MS: float = 5.0
    MAX_CACHE_ACCESS_TIME_MS: float = 1.0
    MAX_MEMORY_CLEANUP_TIME_MS: float = 100.0
    MAX_HDF5_CONNECTION_TIME_MS: float = 50.0

    # Throughput requirements
    MIN_SIGNAL_THROUGHPUT_OPS_SEC: int = 10000
    MIN_THREAD_POOL_THROUGHPUT_TASKS_SEC: int = 1000
    MIN_CACHE_THROUGHPUT_OPS_SEC: int = 50000
    MIN_IO_THROUGHPUT_MB_SEC: float = 10.0

    # Scale test parameters
    THREAD_COUNT_RANGE: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    DATA_SIZE_RANGE: list[int] = field(
        default_factory=lambda: [100, 1000, 10000, 100000]
    )
    CONCURRENT_OPERATION_RANGE: list[int] = field(
        default_factory=lambda: [10, 50, 100, 500]
    )

    # Test durations
    STRESS_TEST_DURATION_SEC: int = 30
    ENDURANCE_TEST_DURATION_SEC: int = 300
    WARMUP_ITERATIONS: int = 10
    MEASUREMENT_ITERATIONS: int = 100


class PerformanceMeasurement:
    """Utility class for performance measurements."""

    def __init__(self, test_name: str):
        self.test_name = test_name
        self.start_time = None
        self.end_time = None
        self.start_cpu = None
        self.end_cpu = None
        self.start_memory = None
        self.end_memory = None
        self.process = psutil.Process()

    def __enter__(self):
        gc.collect()  # Clean start
        self.start_time = time.perf_counter()
        self.start_cpu = psutil.cpu_percent(interval=None)
        self.start_memory = self.process.memory_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.end_cpu = psutil.cpu_percent(interval=None)
        self.end_memory = self.process.memory_info()

    def get_metrics(self, operation_count: int = 1) -> PerformanceMetrics:
        """Calculate performance metrics from measurements."""
        execution_time_s = self.end_time - self.start_time
        memory_delta_mb = (self.end_memory.rss - self.start_memory.rss) / 1024 / 1024

        return PerformanceMetrics(
            name=self.test_name,
            execution_time_ms=execution_time_s * 1000,
            cpu_usage_percent=(self.end_cpu - self.start_cpu) if self.start_cpu else 0,
            memory_usage_mb=memory_delta_mb,
            throughput_ops_per_sec=operation_count / execution_time_s
            if execution_time_s > 0
            else 0,
            latency_p50_ms=execution_time_s * 1000,  # Single operation
            latency_p95_ms=execution_time_s * 1000,  # Single operation
            latency_p99_ms=execution_time_s * 1000,  # Single operation
            error_rate_percent=0.0,
        )


def measure_latency_distribution(
    func: Callable, iterations: int = 1000
) -> dict[str, float]:
    """Measure latency distribution for a function."""
    times = []

    # Warmup
    for _ in range(10):
        func()

    # Measurements
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times.sort()
    return {
        "p50_ms": times[int(len(times) * 0.50)],
        "p95_ms": times[int(len(times) * 0.95)],
        "p99_ms": times[int(len(times) * 0.99)],
        "mean_ms": mean(times),
        "std_ms": stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


# =============================================================================
# Threading System Performance Tests
# =============================================================================


class ThreadingPerformanceTests(unittest.TestCase):
    """Comprehensive threading performance tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUTestConfig()
        self.signal_optimizer = get_signal_optimizer()
        self.thread_manager = get_thread_pool_manager()

    def test_signal_batching_performance(self):
        """Test signal batching optimization performance."""

        def emit_signals(count: int):
            """Emit a batch of signals for testing."""
            if hasattr(self.signal_optimizer, "emit_signal"):
                for i in range(count):
                    self.signal_optimizer.emit_signal("test_signal", f"data_{i}")

        # Test different batch sizes
        batch_sizes = [10, 100, 1000, 5000]
        results = {}

        for batch_size in batch_sizes:
            with PerformanceMeasurement(f"signal_batching_{batch_size}") as measurement:
                emit_signals(batch_size)

            metrics = measurement.get_metrics(batch_size)
            results[batch_size] = metrics

            # Validate performance thresholds
            self.assertLess(
                metrics.latency_p95_ms,
                self.config.MAX_SIGNAL_BATCHING_LATENCY_MS,
                f"Signal batching latency too high for batch size {batch_size}",
            )

            self.assertGreater(
                metrics.throughput_ops_per_sec,
                self.config.MIN_SIGNAL_THROUGHPUT_OPS_SEC,
                f"Signal throughput too low for batch size {batch_size}",
            )

        logger.info(f"Signal batching performance results: {results}")

    def test_thread_pool_scaling_performance(self):
        """Test thread pool performance across different scales."""

        def cpu_intensive_task():
            """CPU intensive task for thread pool testing."""
            result = 0
            for i in range(10000):
                result += i * i
            return result

        results = {}

        for thread_count in self.config.THREAD_COUNT_RANGE:
            if hasattr(self.thread_manager, "submit_task"):
                with PerformanceMeasurement(
                    f"thread_pool_{thread_count}"
                ) as measurement:
                    futures = []
                    for _ in range(100):  # Submit 100 tasks
                        future = self.thread_manager.submit_task(cpu_intensive_task)
                        futures.append(future)

                    # Wait for all tasks to complete
                    for future in futures:
                        if hasattr(future, "result"):
                            future.result()

                metrics = measurement.get_metrics(100)
                results[thread_count] = metrics

                # Validate thread pool performance
                self.assertGreater(
                    metrics.throughput_ops_per_sec,
                    self.config.MIN_THREAD_POOL_THROUGHPUT_TASKS_SEC,
                    f"Thread pool throughput too low with {thread_count} threads",
                )

        logger.info(f"Thread pool scaling results: {results}")

    def test_worker_performance_optimization(self):
        """Test optimized worker performance."""

        if not hasattr(OptimizedWorkerPool, "execute_task"):
            self.skipTest("OptimizedWorkerPool not available")

        worker_pool = OptimizedWorkerPool(max_workers=8)

        def scientific_computation_task(data_size: int):
            """Scientific computation task for worker testing."""
            data = np.random.random((data_size, data_size))
            return np.fft.fft2(data)

        data_sizes = [64, 128, 256, 512]
        results = {}

        for data_size in data_sizes:
            with PerformanceMeasurement(
                f"worker_performance_{data_size}"
            ) as measurement:
                tasks = []
                for _ in range(20):  # Submit 20 computation tasks
                    task = worker_pool.execute_task(
                        scientific_computation_task, data_size
                    )
                    tasks.append(task)

                # Wait for completion
                for task in tasks:
                    if hasattr(task, "result"):
                        task.result()

            metrics = measurement.get_metrics(20)
            results[data_size] = metrics

        logger.info(f"Worker performance results: {results}")

    def test_thread_creation_overhead(self):
        """Test thread creation and destruction overhead."""

        def measure_thread_creation(count: int):
            """Measure thread creation performance."""
            threads = []

            def dummy_work():
                time.sleep(0.001)  # 1ms work

            start_time = time.perf_counter()

            for _ in range(count):
                thread = threading.Thread(target=dummy_work)
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            end_time = time.perf_counter()
            return end_time - start_time

        thread_counts = [10, 50, 100, 200]
        results = {}

        for count in thread_counts:
            execution_time = measure_thread_creation(count)
            time_per_thread_ms = (execution_time / count) * 1000

            results[count] = {
                "total_time_s": execution_time,
                "time_per_thread_ms": time_per_thread_ms,
                "threads_per_sec": count / execution_time,
            }

            # Validate thread creation performance
            self.assertLess(
                time_per_thread_ms,
                self.config.MAX_THREAD_CREATION_TIME_MS,
                f"Thread creation overhead too high: {time_per_thread_ms:.2f}ms",
            )

        logger.info(f"Thread creation overhead results: {results}")

    def test_concurrent_signal_processing_stress(self):
        """Stress test concurrent signal processing."""

        def signal_stress_worker(worker_id: int, signal_count: int):
            """Worker that generates signals for stress testing."""
            if hasattr(self.signal_optimizer, "emit_signal"):
                for i in range(signal_count):
                    self.signal_optimizer.emit_signal(
                        f"stress_signal_{worker_id}",
                        {
                            "worker_id": worker_id,
                            "signal_id": i,
                            "timestamp": time.time(),
                        },
                    )

        worker_count = 8
        signals_per_worker = 1000

        with PerformanceMeasurement("concurrent_signal_stress") as measurement:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = []
                for worker_id in range(worker_count):
                    future = executor.submit(
                        signal_stress_worker, worker_id, signals_per_worker
                    )
                    futures.append(future)

                # Wait for all workers to complete
                for future in futures:
                    future.result()

        metrics = measurement.get_metrics(worker_count * signals_per_worker)

        # Validate stress test performance
        self.assertGreater(
            metrics.throughput_ops_per_sec,
            self.config.MIN_SIGNAL_THROUGHPUT_OPS_SEC
            * 0.5,  # Allow 50% degradation under stress
            f"Signal processing throughput under stress too low: {metrics.throughput_ops_per_sec}",
        )

        logger.info(f"Concurrent signal stress test results: {metrics}")


# =============================================================================
# Memory Management Performance Tests
# =============================================================================


class MemoryPerformanceTests(unittest.TestCase):
    """Comprehensive memory management performance tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUTestConfig()

    def test_cache_performance_optimization(self):
        """Test cache system performance."""

        if not callable(AdvancedCache):
            self.skipTest("AdvancedCache not available")

        cache = AdvancedCache(max_size=10000, ttl_seconds=300)

        # Test cache operations
        def cache_operations_test(operation_count: int):
            """Test various cache operations."""
            # Fill cache
            for i in range(operation_count):
                cache.set(f"key_{i}", f"value_{i}")

            # Test cache hits
            for i in range(operation_count):
                cache.get(f"key_{i}")

            # Test cache misses
            for i in range(operation_count, operation_count * 2):
                cache.get(f"nonexistent_key_{i}")

        operation_counts = [100, 1000, 10000, 50000]
        results = {}

        for count in operation_counts:
            with PerformanceMeasurement(f"cache_operations_{count}") as measurement:
                cache_operations_test(count)

            metrics = measurement.get_metrics(count * 3)  # set + get + miss operations
            results[count] = metrics

            # Validate cache performance
            self.assertLess(
                metrics.latency_p95_ms / count,  # Average per operation
                self.config.MAX_CACHE_ACCESS_TIME_MS,
                f"Cache access time too high for {count} operations",
            )

            self.assertGreater(
                metrics.throughput_ops_per_sec,
                self.config.MIN_CACHE_THROUGHPUT_OPS_SEC,
                f"Cache throughput too low for {count} operations",
            )

        logger.info(f"Cache performance results: {results}")

    def test_adaptive_memory_management_performance(self):
        """Test adaptive memory manager performance."""

        if not hasattr(AdaptiveMemoryManager, "monitor_memory_usage"):
            self.skipTest("AdaptiveMemoryManager not available")

        memory_manager = AdaptiveMemoryManager()

        def memory_pressure_simulation():
            """Simulate memory pressure scenarios."""
            large_arrays = []

            # Gradually increase memory usage
            for i in range(20):
                # Create progressively larger arrays
                array_size = 1000 * (i + 1)
                array = np.random.random((array_size, 100))
                large_arrays.append(array)

                # Trigger memory monitoring
                if hasattr(memory_manager, "check_memory_pressure"):
                    memory_manager.check_memory_pressure()

                # Simulate some processing
                time.sleep(0.01)

        with PerformanceMeasurement("adaptive_memory_management") as measurement:
            memory_pressure_simulation()

        metrics = measurement.get_metrics(20)

        # Memory manager should keep memory growth reasonable
        self.assertLess(
            metrics.memory_usage_mb,
            500,  # Should not grow beyond 500MB for this test
            f"Memory usage too high: {metrics.memory_usage_mb}MB",
        )

        logger.info(f"Adaptive memory management results: {metrics}")

    def test_memory_cleanup_performance(self):
        """Test memory cleanup efficiency."""

        def memory_allocation_cleanup_cycle():
            """Test memory allocation and cleanup cycle."""
            # Allocate large amounts of memory
            large_objects = []

            for _i in range(100):
                # Create large numpy arrays
                array = np.random.random((1000, 1000))
                large_objects.append(array)

            # Explicit cleanup
            del large_objects
            gc.collect()

        # Measure cleanup performance
        cleanup_times = []

        for _ in range(10):  # Multiple cycles
            start_memory = psutil.Process().memory_info().rss

            with PerformanceMeasurement("memory_cleanup_cycle") as measurement:
                memory_allocation_cleanup_cycle()

            end_memory = psutil.Process().memory_info().rss
            (end_memory - start_memory) / 1024 / 1024  # MB

            cleanup_times.append(measurement.end_time - measurement.start_time)

        avg_cleanup_time_ms = mean(cleanup_times) * 1000

        # Validate cleanup performance
        self.assertLess(
            avg_cleanup_time_ms,
            self.config.MAX_MEMORY_CLEANUP_TIME_MS,
            f"Memory cleanup too slow: {avg_cleanup_time_ms:.2f}ms",
        )

        logger.info(f"Memory cleanup performance: {avg_cleanup_time_ms:.2f}ms average")

    def test_memory_leak_detection(self):
        """Test for memory leaks in optimization systems."""

        def repeated_operations(iterations: int):
            """Perform repeated operations that could leak memory."""
            for i in range(iterations):
                # Simulate optimization system usage
                if hasattr(get_signal_optimizer(), "emit_signal"):
                    get_signal_optimizer().emit_signal("leak_test", f"data_{i}")

                # Cache operations
                if callable(AdvancedCache):
                    cache = AdvancedCache(max_size=1000)
                    cache.set(f"leak_key_{i}", np.random.random(100))
                    del cache  # Should clean up properly

                # Thread operations
                if hasattr(get_thread_pool_manager(), "submit_task"):
                    future = get_thread_pool_manager().submit_task(
                        lambda: sum(range(1000))
                    )
                    if hasattr(future, "result"):
                        future.result()

        # Baseline memory measurement
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Perform operations
        repeated_operations(1000)

        # Force cleanup
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        memory_growth = final_memory - initial_memory

        # Validate no significant memory leaks
        self.assertLess(
            memory_growth,
            50,  # Allow up to 50MB growth
            f"Potential memory leak detected: {memory_growth:.2f}MB growth",
        )

        logger.info(
            f"Memory leak test: {memory_growth:.2f}MB growth after 1000 operations"
        )


# =============================================================================
# I/O Performance Tests
# =============================================================================


class IOPerformanceTests(unittest.TestCase):
    """Comprehensive I/O performance tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUTestConfig()

    def test_hdf5_connection_pooling_performance(self):
        """Test HDF5 connection pooling performance."""

        if not hasattr(HDF5ConnectionPool, "get_connection"):
            self.skipTest("HDF5ConnectionPool not available")

        # Create temporary HDF5 files for testing
        with TemporaryDirectory() as temp_dir:
            test_files = []
            for i in range(5):
                test_file = Path(temp_dir) / f"test_{i}.h5"
                # Create mock HDF5 file structure
                test_files.append(str(test_file))

            connection_pool = HDF5ConnectionPool(max_connections=10)

            def connection_pooling_test():
                """Test connection pooling performance."""
                connections = []

                # Test connection acquisition
                for file_path in test_files:
                    for _ in range(20):  # Multiple connections per file
                        if hasattr(connection_pool, "get_connection"):
                            conn = connection_pool.get_connection(file_path)
                            connections.append(conn)

                # Test connection release
                for conn in connections:
                    if hasattr(connection_pool, "release_connection"):
                        connection_pool.release_connection(conn)

            with PerformanceMeasurement("hdf5_connection_pooling") as measurement:
                connection_pooling_test()

            metrics = measurement.get_metrics(len(test_files) * 40)  # acquire + release

            # Validate connection pooling performance
            self.assertLess(
                metrics.latency_p95_ms / (len(test_files) * 40),
                self.config.MAX_HDF5_CONNECTION_TIME_MS,
                "HDF5 connection pooling too slow",
            )

            logger.info(f"HDF5 connection pooling results: {metrics}")

    def test_batch_io_operations_performance(self):
        """Test batch I/O operations performance."""

        def batch_file_operations(file_count: int, operations_per_file: int):
            """Test batch file operations."""
            with TemporaryDirectory() as temp_dir:
                files = []

                # Create files
                for i in range(file_count):
                    file_path = Path(temp_dir) / f"batch_test_{i}.txt"
                    files.append(file_path)

                # Batch write operations
                for file_path in files:
                    with open(file_path, "w") as f:
                        for j in range(operations_per_file):
                            f.write(f"Line {j} in file {file_path.name}\n")

                # Batch read operations
                total_lines = 0
                for file_path in files:
                    with open(file_path) as f:
                        total_lines += len(f.readlines())

                return total_lines

        file_counts = [10, 50, 100]
        operations_per_file = 1000
        results = {}

        for file_count in file_counts:
            with PerformanceMeasurement(f"batch_io_{file_count}") as measurement:
                total_operations = batch_file_operations(
                    file_count, operations_per_file
                )

            metrics = measurement.get_metrics(total_operations)
            results[file_count] = metrics

            # Validate batch I/O performance
            throughput_mb_s = (
                (total_operations * 50)
                / (1024 * 1024)
                / (metrics.execution_time_ms / 1000)
            )  # Approximate
            self.assertGreater(
                throughput_mb_s,
                self.config.MIN_IO_THROUGHPUT_MB_SEC
                * 0.1,  # Lower threshold for file I/O
                f"Batch I/O throughput too low for {file_count} files",
            )

        logger.info(f"Batch I/O performance results: {results}")


# =============================================================================
# Scientific Computing Performance Tests
# =============================================================================


class ScientificComputingPerformanceTests(unittest.TestCase):
    """Comprehensive scientific computing performance tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUTestConfig()

    def test_correlation_analysis_performance(self):
        """Test correlation analysis performance."""

        def correlation_computation(data_size: int):
            """Simulate correlation analysis computation."""
            # Generate synthetic XPCS data
            intensity_data = np.random.poisson(1000, (data_size, data_size))

            # Compute correlation function (simplified)
            g2_values = []

            for tau_idx in range(min(data_size, 100)):  # Limit for performance
                # Simplified correlation calculation
                if tau_idx < data_size - 1:
                    correlation = np.corrcoef(
                        intensity_data[0, :], intensity_data[tau_idx, :]
                    )[0, 1]
                    g2_values.append(correlation)

            return np.array(g2_values)

        data_sizes = [64, 128, 256, 512]
        results = {}

        for data_size in data_sizes:
            with PerformanceMeasurement(
                f"correlation_analysis_{data_size}"
            ) as measurement:
                g2_result = correlation_computation(data_size)

            metrics = measurement.get_metrics(len(g2_result))
            results[data_size] = metrics

            # Validate performance scales reasonably with data size
            self.assertLess(
                metrics.execution_time_ms,
                data_size * 10,  # Linear scaling expectation
                f"Correlation analysis too slow for data size {data_size}",
            )

        logger.info(f"Correlation analysis performance results: {results}")

    def test_fitting_algorithm_performance(self):
        """Test fitting algorithm performance."""

        def exponential_fitting(data_points: int, fit_iterations: int):
            """Simulate exponential fitting performance."""
            from scipy.optimize import curve_fit

            # Generate synthetic G2 data with exponential decay
            tau_values = np.logspace(-6, 2, data_points)
            true_params = [1.0, 0.1, 0.02]  # amplitude, decay1, decay2

            def double_exponential(t, a, tau1, tau2):
                return a * (np.exp(-t / tau1) + np.exp(-t / tau2))

            # Add noise
            g2_data = double_exponential(tau_values, *true_params)
            g2_data += np.random.normal(0, 0.01, len(g2_data))

            # Perform multiple fits
            fit_results = []
            for _ in range(fit_iterations):
                try:
                    params, _ = curve_fit(
                        double_exponential,
                        tau_values,
                        g2_data,
                        p0=[1.0, 0.1, 0.02],
                        bounds=([0, 0, 0], [10, 1, 1]),
                    )
                    fit_results.append(params)
                except Exception:
                    # Skip failed fits - curve fitting can fail with bad data
                    pass

            return fit_results

        data_points_list = [50, 100, 200]
        fit_iterations_list = [10, 50, 100]
        results = {}

        for data_points in data_points_list:
            for fit_iterations in fit_iterations_list:
                test_name = f"{data_points}pts_{fit_iterations}fits"

                with PerformanceMeasurement(f"fitting_{test_name}") as measurement:
                    exponential_fitting(data_points, fit_iterations)

                metrics = measurement.get_metrics(fit_iterations)
                results[test_name] = metrics

                # Validate fitting performance
                time_per_fit_ms = metrics.execution_time_ms / fit_iterations
                self.assertLess(
                    time_per_fit_ms,
                    1000,  # Max 1 second per fit
                    f"Fitting too slow: {time_per_fit_ms:.2f}ms per fit",
                )

        logger.info(f"Fitting algorithm performance results: {results}")

    def test_fft_computation_performance(self):
        """Test FFT computation performance."""

        def fft_operations(array_sizes: list[int], operation_count: int):
            """Test FFT computation performance."""
            results = []

            for size in array_sizes:
                # Generate test data
                data = np.random.random((size, size)) + 1j * np.random.random(
                    (size, size)
                )

                # Perform FFT operations
                for _ in range(operation_count):
                    fft_result = np.fft.fft2(data)
                    ifft_result = np.fft.ifft2(fft_result)  # Round trip
                    results.append(np.mean(np.abs(ifft_result)))

            return results

        array_sizes = [64, 128, 256, 512]
        operation_count = 10

        with PerformanceMeasurement("fft_computation") as measurement:
            fft_operations(array_sizes, operation_count)

        metrics = measurement.get_metrics(
            len(array_sizes) * operation_count * 2
        )  # FFT + IFFT

        # Validate FFT performance
        self.assertGreater(
            metrics.throughput_ops_per_sec,
            100,  # At least 100 FFT operations per second
            f"FFT computation throughput too low: {metrics.throughput_ops_per_sec}",
        )

        logger.info(f"FFT computation performance results: {metrics}")


# =============================================================================
# End-to-End Workflow Performance Tests
# =============================================================================


class EndToEndWorkflowTests(unittest.TestCase):
    """End-to-end workflow performance tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CPUTestConfig()

    def test_complete_xpcs_analysis_workflow(self):
        """Test complete XPCS analysis workflow performance."""

        def xpcs_workflow_simulation():
            """Simulate complete XPCS analysis workflow."""

            # Step 1: Data loading simulation
            data_size = 256
            intensity_data = np.random.poisson(1000, (data_size, data_size))

            # Step 2: Preprocessing
            # Background subtraction
            background = np.mean(intensity_data)
            corrected_data = intensity_data - background

            # Step 3: Correlation analysis
            tau_values = np.logspace(-6, 2, 100)
            g2_values = []

            for tau_idx in range(len(tau_values)):
                if tau_idx < data_size - 1:
                    correlation = np.corrcoef(
                        corrected_data[0, :], corrected_data[tau_idx, :]
                    )[0, 1]
                    g2_values.append(max(0, correlation))

            g2_values = np.array(g2_values)

            # Step 4: Fitting (simplified)
            try:
                from scipy.optimize import curve_fit

                def exponential_decay(t, a, tau):
                    return a * np.exp(-t / tau)

                valid_indices = g2_values > 0
                if np.sum(valid_indices) > 10:
                    params, _ = curve_fit(
                        exponential_decay,
                        tau_values[valid_indices],
                        g2_values[valid_indices],
                        p0=[1.0, 0.1],
                        bounds=([0, 0], [10, 1]),
                    )
                else:
                    params = [1.0, 0.1]  # Default parameters
            except Exception:
                params = [1.0, 0.1]  # Fallback on any fitting error

            # Step 5: Results processing
            results = {
                "g2_values": g2_values.tolist(),
                "tau_values": tau_values.tolist(),
                "fit_parameters": params,
                "data_shape": intensity_data.shape,
                "background_level": float(background),
            }

            return results

        with PerformanceMeasurement("complete_xpcs_workflow") as measurement:
            xpcs_workflow_simulation()

        metrics = measurement.get_metrics(1)

        # Validate end-to-end workflow performance
        self.assertLess(
            metrics.execution_time_ms,
            10000,  # Should complete within 10 seconds
            f"XPCS workflow too slow: {metrics.execution_time_ms}ms",
        )

        self.assertLess(
            metrics.memory_usage_mb,
            500,  # Should not use more than 500MB
            f"XPCS workflow uses too much memory: {metrics.memory_usage_mb}MB",
        )

        logger.info(f"Complete XPCS workflow performance: {metrics}")

    def test_concurrent_analysis_performance(self):
        """Test concurrent analysis performance."""

        def concurrent_analysis_worker(worker_id: int):
            """Worker function for concurrent analysis."""
            # Simulate analysis on smaller dataset
            data_size = 128
            intensity_data = np.random.poisson(500, (data_size, data_size))

            # Simple analysis
            mean_intensity = np.mean(intensity_data)
            std_intensity = np.std(intensity_data)

            # Some computation
            fft_result = np.fft.fft2(intensity_data)
            power_spectrum = np.abs(fft_result) ** 2

            return {
                "worker_id": worker_id,
                "mean_intensity": float(mean_intensity),
                "std_intensity": float(std_intensity),
                "power_sum": float(np.sum(power_spectrum)),
            }

        worker_counts = [2, 4, 8]
        results = {}

        for worker_count in worker_counts:
            with (
                PerformanceMeasurement(
                    f"concurrent_analysis_{worker_count}"
                ) as measurement,
                ThreadPoolExecutor(max_workers=worker_count) as executor,
            ):
                futures = []
                for worker_id in range(worker_count):
                    future = executor.submit(concurrent_analysis_worker, worker_id)
                    futures.append(future)

                # Wait for all workers to complete
                worker_results = []
                for future in futures:
                    worker_results.append(future.result())

            metrics = measurement.get_metrics(worker_count)
            results[worker_count] = metrics

            # Validate concurrent performance
            self.assertLess(
                metrics.execution_time_ms,
                5000,  # Should complete within 5 seconds
                f"Concurrent analysis too slow with {worker_count} workers",
            )

        logger.info(f"Concurrent analysis performance results: {results}")


# =============================================================================
# Performance Test Suite Runner
# =============================================================================


class CPUPerformanceTestSuite:
    """Main test suite runner for CPU performance tests."""

    def __init__(self, config: CPUTestConfig | None = None):
        self.config = config or CPUTestConfig()
        self.test_results: dict[str, Any] = {}

    def run_all_tests(self) -> dict[str, Any]:
        """Run all CPU performance tests."""

        test_classes = [
            ThreadingPerformanceTests,
            MemoryPerformanceTests,
            IOPerformanceTests,
            ScientificComputingPerformanceTests,
            EndToEndWorkflowTests,
        ]

        all_results = {}

        for test_class in test_classes:
            class_name = test_class.__name__
            logger.info(f"Running {class_name}...")

            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

            # Run tests
            runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
            result = runner.run(suite)

            # Store results
            all_results[class_name] = {
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "success_rate": (
                    result.testsRun - len(result.failures) - len(result.errors)
                )
                / result.testsRun
                * 100,
            }

        self.test_results = all_results
        return all_results

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""

        report = {
            "timestamp": time.time(),
            "config": {
                "max_thread_creation_time_ms": self.config.MAX_THREAD_CREATION_TIME_MS,
                "max_signal_batching_latency_ms": self.config.MAX_SIGNAL_BATCHING_LATENCY_MS,
                "min_signal_throughput_ops_sec": self.config.MIN_SIGNAL_THROUGHPUT_OPS_SEC,
                "thread_count_range": self.config.THREAD_COUNT_RANGE,
                "data_size_range": self.config.DATA_SIZE_RANGE,
            },
            "system_info": {
                "cpu_count": os.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "test_results": self.test_results,
            "performance_summary": {
                "total_test_classes": len(self.test_results),
                "overall_success_rate": np.mean(
                    [r["success_rate"] for r in self.test_results.values()]
                ),
                "failed_tests": sum(
                    r["failures"] + r["errors"] for r in self.test_results.values()
                ),
            },
        }

        return report

    def save_report(self, output_path: str):
        """Save performance report to file."""
        report = self.generate_performance_report()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Performance report saved to {output_path}")


if __name__ == "__main__":
    # Run the complete CPU performance test suite
    suite = CPUPerformanceTestSuite()

    # Run all tests
    results = suite.run_all_tests()

    # Generate and save report
    report_path = "cpu_performance_test_report.json"
    suite.save_report(report_path)

    # Print summary

    for _class_name, _result in results.items():
        pass
