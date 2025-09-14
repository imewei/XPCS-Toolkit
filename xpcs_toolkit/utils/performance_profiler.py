"""
Performance profiling utilities for XPCS Toolkit algorithm optimization.

This module provides decorators and tools for profiling computational functions
to identify bottlenecks and measure optimization improvements.
"""

from __future__ import annotations

import cProfile
import functools
import io
import pstats
import time
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class PerformanceProfiler:
    """
    A comprehensive performance profiler for XPCS Toolkit algorithms.
    """

    def __init__(self):
        self.timing_results: dict[str, list[float]] = {}
        self.memory_usage: dict[str, list[float]] = {}
        self.call_counts: dict[str, int] = {}
        self.enabled = True

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def clear_stats(self):
        """Clear all profiling statistics."""
        self.timing_results.clear()
        self.memory_usage.clear()
        self.call_counts.clear()

    def profile_function(self, func_name: str | None = None):
        """
        Decorator to profile function execution time and call frequency.

        Args:
            func_name: Optional name for the function (defaults to function.__name__)
        """

        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return func(*args, **kwargs)

                # Record call count
                self.call_counts[name] = self.call_counts.get(name, 0) + 1

                # Time the function execution
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time

                    # Store timing result
                    if name not in self.timing_results:
                        self.timing_results[name] = []
                    self.timing_results[name].append(execution_time)

                    # Log slow functions
                    if execution_time > 1.0:  # Log functions taking > 1 second
                        logger.info(
                            f"Slow function detected: {name} took {execution_time:.3f}s"
                        )

            return wrapper

        return decorator

    @contextmanager
    def profile_block(self, block_name: str):
        """
        Context manager to profile a block of code.

        Args:
            block_name: Name to identify the code block
        """
        if not self.enabled:
            yield
            return

        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            if block_name not in self.timing_results:
                self.timing_results[block_name] = []
            self.timing_results[block_name].append(execution_time)

    def detailed_profile(self, func: Callable, *args, **kwargs):
        """
        Perform detailed profiling using cProfile.

        Args:
            func: Function to profile
            *args, **kwargs: Arguments to pass to the function

        Returns:
            Function result and profiling statistics
        """
        if not self.enabled:
            return func(*args, **kwargs), None

        pr = cProfile.Profile()
        pr.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            pr.disable()

        # Capture profiling output
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats()

        return result, s.getvalue()

    def get_performance_summary(self) -> dict[str, dict[str, Any]]:
        """
        Get a comprehensive performance summary.

        Returns:
            Dictionary with performance statistics for each profiled function
        """
        summary = {}

        for func_name in self.timing_results:
            times = np.array(self.timing_results[func_name])
            summary[func_name] = {
                "call_count": self.call_counts.get(func_name, 0),
                "total_time": np.sum(times),
                "average_time": np.mean(times),
                "median_time": np.median(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "std_time": np.std(times),
                "time_per_call": np.sum(times) / len(times) if len(times) > 0 else 0,
            }

        return summary

    def print_performance_report(self):
        """Print a formatted performance report."""
        summary = self.get_performance_summary()

        if not summary:
            print("No profiling data available.")
            return

        print("\n" + "=" * 80)
        print("PERFORMANCE PROFILING REPORT")
        print("=" * 80)

        # Sort functions by total time
        sorted_funcs = sorted(
            summary.items(), key=lambda x: x[1]["total_time"], reverse=True
        )

        print(
            f"{'Function':<30} {'Calls':<8} {'Total(s)':<10} {'Avg(s)':<10} {'Max(s)':<10}"
        )
        print("-" * 80)

        for func_name, stats in sorted_funcs:
            print(
                f"{func_name:<30} {stats['call_count']:<8} "
                f"{stats['total_time']:<10.3f} {stats['average_time']:<10.6f} "
                f"{stats['max_time']:<10.6f}"
            )

        print("-" * 80)
        print(f"Total profiled functions: {len(summary)}")
        print(
            f"Total execution time: {sum(s['total_time'] for s in summary.values()):.3f}s"
        )

    def get_bottlenecks(self, threshold: float = 0.1) -> list[str]:
        """
        Identify performance bottlenecks.

        Args:
            threshold: Minimum execution time (seconds) to consider a bottleneck

        Returns:
            List of function names that are bottlenecks
        """
        summary = self.get_performance_summary()
        bottlenecks = []

        for func_name, stats in summary.items():
            if stats["total_time"] > threshold or stats["max_time"] > threshold:
                bottlenecks.append(func_name)

        return bottlenecks

    def compare_performance(
        self, other: PerformanceProfiler
    ) -> dict[str, dict[str, float]]:
        """
        Compare performance with another profiler instance.

        Args:
            other: Another PerformanceProfiler instance

        Returns:
            Dictionary with performance comparisons
        """
        self_summary = self.get_performance_summary()
        other_summary = other.get_performance_summary()

        comparison = {}

        common_functions = set(self_summary.keys()) & set(other_summary.keys())

        for func_name in common_functions:
            self_stats = self_summary[func_name]
            other_stats = other_summary[func_name]

            comparison[func_name] = {
                "speedup_factor": other_stats["average_time"]
                / self_stats["average_time"]
                if self_stats["average_time"] > 0
                else 0,
                "time_saved": other_stats["total_time"] - self_stats["total_time"],
                "call_count_change": self_stats["call_count"]
                - other_stats["call_count"],
            }

        return comparison


# Global profiler instance
global_profiler = PerformanceProfiler()


def profile_algorithm(func_name: str | None = None):
    """
    Convenience decorator using the global profiler.

    Args:
        func_name: Optional name for the function
    """
    return global_profiler.profile_function(func_name)


@contextmanager
def profile_block(block_name: str):
    """
    Convenience context manager using the global profiler.

    Args:
        block_name: Name to identify the code block
    """
    with global_profiler.profile_block(block_name):
        yield


def benchmark_function(func: Callable, iterations: int = 10, *args, **kwargs):
    """
    Benchmark a function over multiple iterations.

    Args:
        func: Function to benchmark
        iterations: Number of iterations to run
        *args, **kwargs: Arguments to pass to the function

    Returns:
        Dictionary with benchmark statistics
    """
    times = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    times = np.array(times)

    return {
        "iterations": iterations,
        "total_time": np.sum(times),
        "average_time": np.mean(times),
        "median_time": np.median(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "std_time": np.std(times),
        "times": times.tolist(),
    }


def memory_usage_profiler(func: Callable):
    """
    Decorator to profile memory usage (requires psutil).

    Args:
        func: Function to profile
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import os

            import psutil

            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            result = func(*args, **kwargs)

            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before

            func_name = func.__name__
            if func_name not in global_profiler.memory_usage:
                global_profiler.memory_usage[func_name] = []
            global_profiler.memory_usage[func_name].append(memory_used)

            if memory_used > 100:  # Log functions using > 100MB
                logger.info(f"High memory usage: {func_name} used {memory_used:.1f}MB")

            return result

        except ImportError:
            logger.warning("psutil not available for memory profiling")
            return func(*args, **kwargs)

    return wrapper


# Example usage and testing functions
def example_slow_function():
    """Example function that simulates slow computation."""
    import time

    time.sleep(0.1)
    return sum(range(10000))


def example_fast_function():
    """Example function that is fast."""
    return sum(range(100))


if __name__ == "__main__":
    # Example usage
    profiler = PerformanceProfiler()

    # Profile some example functions
    @profiler.profile_function("slow_computation")
    def slow_computation():
        return example_slow_function()

    @profiler.profile_function("fast_computation")
    def fast_computation():
        return example_fast_function()

    # Run the functions
    for _ in range(5):
        slow_computation()
        fast_computation()

    # Print performance report
    profiler.print_performance_report()

    # Identify bottlenecks
    bottlenecks = profiler.get_bottlenecks(0.05)
    print(f"\nBottlenecks: {bottlenecks}")

    # Benchmark example
    benchmark_results = benchmark_function(example_fast_function, iterations=100)
    print(f"\nBenchmark results: {benchmark_results}")
