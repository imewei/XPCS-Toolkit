#!/usr/bin/env python3
"""
Test Performance Optimization and Monitoring for XPCS Toolkit

This module provides performance monitoring, optimization utilities, and
benchmarking tools for the XPCS Toolkit test suite.

Created: 2025-09-16
"""

import gc
import os
import psutil
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import pytest


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    test_name: str
    duration: float
    memory_peak_mb: float
    memory_start_mb: float
    memory_end_mb: float
    cpu_percent: float
    disk_io_read: int
    disk_io_write: int
    timestamp: float = field(default_factory=time.time)

    @property
    def memory_delta_mb(self) -> float:
        """Memory usage change during test."""
        return self.memory_end_mb - self.memory_start_mb

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'test_name': self.test_name,
            'duration': self.duration,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_start_mb': self.memory_start_mb,
            'memory_end_mb': self.memory_end_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'cpu_percent': self.cpu_percent,
            'disk_io_read': self.disk_io_read,
            'disk_io_write': self.disk_io_write,
            'timestamp': self.timestamp
        }


class PerformanceMonitor:
    """Advanced performance monitoring for tests."""

    def __init__(self, slow_threshold: float = 1.0, memory_threshold_mb: float = 100.0):
        self.slow_threshold = slow_threshold
        self.memory_threshold_mb = memory_threshold_mb
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        self._monitoring_active = False
        self._peak_memory_mb = 0.0
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()

    def start_monitoring(self, test_name: str) -> Dict[str, Any]:
        """Start monitoring performance for a test."""
        if self._monitoring_active:
            self.stop_monitoring()

        self._monitoring_active = True
        self._stop_monitoring.clear()
        self._peak_memory_mb = 0.0

        # Get initial measurements
        initial_state = {
            'test_name': test_name,
            'start_time': time.time(),
            'start_memory_mb': self._get_memory_mb(),
            'start_disk_io': self._get_disk_io(),
            'start_cpu_time': self.process.cpu_times()
        }

        # Start background monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._background_monitoring,
            daemon=True
        )
        self._monitor_thread.start()

        return initial_state

    def stop_monitoring(self, initial_state: Optional[Dict[str, Any]] = None) -> Optional[PerformanceMetrics]:
        """Stop monitoring and return metrics."""
        if not self._monitoring_active:
            return None

        self._monitoring_active = False
        self._stop_monitoring.set()

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        if initial_state is None:
            return None

        # Calculate final metrics
        end_time = time.time()
        end_memory_mb = self._get_memory_mb()
        end_disk_io = self._get_disk_io()
        end_cpu_time = self.process.cpu_times()

        # Calculate CPU usage
        cpu_delta = (
            (end_cpu_time.user + end_cpu_time.system) -
            (initial_state['start_cpu_time'].user + initial_state['start_cpu_time'].system)
        )
        wall_time = end_time - initial_state['start_time']
        cpu_percent = (cpu_delta / wall_time * 100) if wall_time > 0 else 0

        # Calculate disk I/O
        start_disk = initial_state['start_disk_io']
        disk_read = end_disk_io.read_bytes - start_disk.read_bytes
        disk_write = end_disk_io.write_bytes - start_disk.write_bytes

        metrics = PerformanceMetrics(
            test_name=initial_state['test_name'],
            duration=wall_time,
            memory_peak_mb=self._peak_memory_mb,
            memory_start_mb=initial_state['start_memory_mb'],
            memory_end_mb=end_memory_mb,
            cpu_percent=cpu_percent,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            timestamp=initial_state['start_time']
        )

        self.metrics.append(metrics)
        return metrics

    def _background_monitoring(self) -> None:
        """Background thread for continuous monitoring."""
        while not self._stop_monitoring.wait(0.1):  # Check every 100ms
            try:
                current_memory = self._get_memory_mb()
                self._peak_memory_mb = max(self._peak_memory_mb, current_memory)
            except (psutil.Error, OSError):
                pass  # Ignore monitoring errors

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / (1024 * 1024)
        except (psutil.Error, OSError):
            return 0.0

    def _get_disk_io(self):
        """Get current disk I/O counters."""
        try:
            disk_io = psutil.disk_io_counters()
            if disk_io is None:
                from collections import namedtuple
                DummyIO = namedtuple("DummyIO", ["read_bytes", "write_bytes"])
                return DummyIO(0, 0)
            return disk_io
        except (psutil.Error, OSError):
            # Return dummy object if unavailable
            from collections import namedtuple
            DummyIO = namedtuple('DummyIO', ['read_bytes', 'write_bytes'])
            return DummyIO(0, 0)

    def get_slow_tests(self) -> List[PerformanceMetrics]:
        """Get tests that exceed the slow threshold."""
        return [m for m in self.metrics if m.duration > self.slow_threshold]

    def get_memory_intensive_tests(self) -> List[PerformanceMetrics]:
        """Get tests that use excessive memory."""
        return [
            m for m in self.metrics
            if m.memory_peak_mb > self.memory_threshold_mb or
               abs(m.memory_delta_mb) > self.memory_threshold_mb
        ]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return {'total_tests': 0, 'message': 'No performance data collected'}

        durations = [m.duration for m in self.metrics]
        memory_peaks = [m.memory_peak_mb for m in self.metrics]
        memory_deltas = [m.memory_delta_mb for m in self.metrics]

        return {
            'total_tests': len(self.metrics),
            'duration_stats': {
                'min': min(durations),
                'max': max(durations),
                'mean': sum(durations) / len(durations),
                'total': sum(durations)
            },
            'memory_stats': {
                'peak_max_mb': max(memory_peaks),
                'peak_mean_mb': sum(memory_peaks) / len(memory_peaks),
                'delta_max_mb': max(memory_deltas),
                'delta_mean_mb': sum(memory_deltas) / len(memory_deltas)
            },
            'slow_tests': len(self.get_slow_tests()),
            'memory_intensive_tests': len(self.get_memory_intensive_tests())
        }

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics.clear()


class TestOptimizer:
    """Utilities for optimizing test performance."""

    @staticmethod
    def optimize_numpy_operations():
        """Optimize NumPy operations for testing."""
        try:
            import numpy as np
            # Set optimal thread count for testing
            cpu_count = os.cpu_count() or 1
            optimal_threads = min(cpu_count, 4)  # Don't use all cores in tests

            # Configure NumPy threading
            os.environ['OPENBLAS_NUM_THREADS'] = str(optimal_threads)
            os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
            os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)

        except ImportError:
            pass

    @staticmethod
    def optimize_memory_usage():
        """Optimize memory usage for tests."""
        # Force garbage collection
        gc.collect()

        # Set aggressive garbage collection
        gc.set_threshold(700, 10, 10)

        # Disable memory debugging if enabled
        if hasattr(gc, 'set_debug'):
            gc.set_debug(0)

    @staticmethod
    def optimize_io_operations():
        """Optimize I/O operations for tests."""
        # Set optimal I/O buffer sizes
        import io
        if hasattr(io, 'DEFAULT_BUFFER_SIZE'):
            # Use smaller buffers for tests to reduce memory usage
            io.DEFAULT_BUFFER_SIZE = 8192  # 8KB instead of default

    @staticmethod
    @contextmanager
    def memory_limit(max_mb: float):
        """Context manager to enforce memory limits during tests."""
        import tracemalloc

        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            peak_mb = peak / (1024 * 1024)
            if peak_mb > max_mb:
                raise MemoryError(
                    f"Test exceeded memory limit: {peak_mb:.1f}MB > {max_mb:.1f}MB"
                )

    @staticmethod
    @contextmanager
    def timeout_limit(max_seconds: float):
        """Context manager to enforce time limits during tests."""
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Test exceeded time limit: {max_seconds}s")

        # Set up timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(max_seconds))
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for Windows
            start_time = time.time()
            try:
                yield
            finally:
                if time.time() - start_time > max_seconds:
                    raise TimeoutError(f"Test exceeded time limit: {max_seconds}s")


class TestCacheManager:
    """Manage caching for test performance optimization."""

    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: float = 100.0):
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'xpcs_toolkit_tests'

        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self._cache_index: Dict[str, Dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        if index_file.exists():
            try:
                import json
                with open(index_file, 'r') as f:
                    self._cache_index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache_index = {}

    def _save_index(self) -> None:
        """Save cache index."""
        index_file = self.cache_dir / 'cache_index.json'
        try:
            import json
            with open(index_file, 'w') as f:
                json.dump(self._cache_index, f, indent=2)
        except IOError:
            pass

    def get_cache_size_mb(self) -> float:
        """Get current cache size in MB."""
        total_size = 0
        try:
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except OSError:
            pass
        return total_size / (1024 * 1024)

    def cleanup_cache(self) -> None:
        """Clean up cache to stay within size limits."""
        current_size = self.get_cache_size_mb()

        if current_size <= self.max_size_mb:
            return

        # Remove oldest files first
        cache_files = []
        try:
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    stat = file_path.stat()
                    cache_files.append((stat.st_mtime, stat.st_size, file_path))
        except OSError:
            return

        cache_files.sort()  # Sort by modification time

        removed_size = 0
        target_removal = (current_size - self.max_size_mb * 0.8) * 1024 * 1024  # MB to bytes

        for mtime, size, file_path in cache_files:
            if removed_size >= target_removal:
                break

            try:
                file_path.unlink()
                removed_size += size
            except OSError:
                continue

    def clear_cache(self) -> None:
        """Clear all cached data."""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._cache_index.clear()
                self._save_index()
        except OSError:
            pass


# Global instances
_performance_monitor = PerformanceMonitor()
_cache_manager = TestCacheManager()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    return _performance_monitor


def get_cache_manager() -> TestCacheManager:
    """Get the global cache manager."""
    return _cache_manager


# Performance optimization decorators
def optimize_for_speed(func: Callable) -> Callable:
    """Decorator to optimize test for speed."""
    def wrapper(*args, **kwargs):
        TestOptimizer.optimize_numpy_operations()
        TestOptimizer.optimize_memory_usage()
        TestOptimizer.optimize_io_operations()

        return func(*args, **kwargs)

    return wrapper


def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor test performance."""
    def wrapper(*args, **kwargs):
        monitor = get_performance_monitor()
        test_name = f"{func.__module__}.{func.__name__}"

        initial_state = monitor.start_monitoring(test_name)

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            metrics = monitor.stop_monitoring(initial_state)
            if metrics and (
                metrics.duration > monitor.slow_threshold or
                metrics.memory_peak_mb > monitor.memory_threshold_mb
            ):
                print(f"\n⚠️  Performance warning for {test_name}:")
                print(f"   Duration: {metrics.duration:.2f}s")
                print(f"   Peak Memory: {metrics.memory_peak_mb:.1f}MB")

    return wrapper


def memory_limit(max_mb: float):
    """Decorator to enforce memory limits."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with TestOptimizer.memory_limit(max_mb):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def timeout_limit(max_seconds: float):
    """Decorator to enforce time limits."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with TestOptimizer.timeout_limit(max_seconds):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Pytest fixtures for performance testing
@pytest.fixture(autouse=True)
def auto_performance_optimization():
    """Automatically apply performance optimizations."""
    TestOptimizer.optimize_numpy_operations()
    TestOptimizer.optimize_memory_usage()
    TestOptimizer.optimize_io_operations()

    # Clean up cache periodically
    cache_manager = get_cache_manager()
    if cache_manager.get_cache_size_mb() > cache_manager.max_size_mb:
        cache_manager.cleanup_cache()

    yield

    # Force cleanup after test
    gc.collect()


@pytest.fixture(scope="session", autouse=True)
def performance_report():
    """Generate performance report at end of session."""
    yield

    monitor = get_performance_monitor()
    summary = monitor.get_performance_summary()

    if summary['total_tests'] > 0:
        print("\n" + "="*80)
        print("TEST PERFORMANCE SUMMARY")
        print("="*80)

        duration_stats = summary['duration_stats']
        memory_stats = summary['memory_stats']

        print(f"\nTests analyzed: {summary['total_tests']}")
        print(f"Total runtime: {duration_stats['total']:.2f}s")
        print(f"Average test time: {duration_stats['mean']:.3f}s")
        print(f"Slowest test: {duration_stats['max']:.2f}s")

        print(f"\nMemory usage:")
        print(f"Peak memory: {memory_stats['peak_max_mb']:.1f}MB")
        print(f"Average peak: {memory_stats['peak_mean_mb']:.1f}MB")
        print(f"Max memory delta: {abs(memory_stats['delta_max_mb']):.1f}MB")

        if summary['slow_tests'] > 0:
            print(f"\n⚠️  {summary['slow_tests']} slow tests detected")

        if summary['memory_intensive_tests'] > 0:
            print(f"⚠️  {summary['memory_intensive_tests']} memory-intensive tests detected")

        print("\nUse @monitor_performance decorator for detailed analysis")
        print("="*80)


# Convenience functions for performance testing
def benchmark_function(func: Callable, *args, iterations: int = 100, **kwargs) -> Dict[str, float]:
    """Benchmark a function's performance."""
    times = []
    memory_usage = []

    for _ in range(iterations):
        start_memory = psutil.Process().memory_info().rss

        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()

        end_memory = psutil.Process().memory_info().rss

        times.append(end_time - start_time)
        memory_usage.append((end_memory - start_memory) / (1024 * 1024))  # MB

    return {
        'mean_time': sum(times) / len(times),
        'min_time': min(times),
        'max_time': max(times),
        'mean_memory_delta_mb': sum(memory_usage) / len(memory_usage),
        'max_memory_delta_mb': max(memory_usage)
    }