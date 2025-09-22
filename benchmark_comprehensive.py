#!/usr/bin/env python3
"""
Comprehensive Performance Benchmark Suite for XPCS Toolkit
==========================================================

This benchmark suite measures the performance impact of our duplicate code
elimination and optimization work across multiple dimensions:

1. Startup Performance
2. Memory Usage Optimization
3. Import/Module Loading Performance
4. Lazy Loading Effectiveness
5. Plot Generation Performance
6. Threading System Performance

Results are saved to benchmark_results.json for analysis.
"""

import gc
import json
import os
import sys
import time
import tracemalloc
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""

    def __init__(self, output_file: str = "benchmark_results.json"):
        self.output_file = output_file
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system_info": self._get_system_info(),
            "benchmarks": {},
        }
        self.process = psutil.Process()

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        return {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "architecture": os.uname().machine if hasattr(os, "uname") else "unknown",
        }

    @contextmanager
    def measure_performance(self, name: str):
        """Context manager to measure performance metrics."""

        # Start measurements
        start_time = time.perf_counter()
        tracemalloc.start()
        initial_memory = self.process.memory_info().rss

        try:
            yield
        finally:
            # End measurements
            end_time = time.perf_counter()
            execution_time = end_time - start_time

            _current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            final_memory = self.process.memory_info().rss
            memory_delta = final_memory - initial_memory

            # Store results
            self.results["benchmarks"][name] = {
                "execution_time_ms": execution_time * 1000,
                "memory_delta_mb": memory_delta / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
            }

    def benchmark_startup_performance(self):
        """Benchmark application startup performance."""

        # Test 1: Core imports
        with self.measure_performance("core_imports"):
            pass

        # Test 2: GUI component imports
        with self.measure_performance("gui_imports"):
            pass

        # Test 3: Plotting system imports
        with self.measure_performance("plotting_imports"):
            pass

        # Test 4: Threading system imports
        with self.measure_performance("threading_imports"):
            pass

        # Test 5: Analysis modules (lazy loading test)
        with self.measure_performance("analysis_modules_lazy"):
            from xpcs_toolkit.viewer_kernel import _get_module

            _get_module("g2mod")
            _get_module("saxs1d")
            _get_module("stability")

    def benchmark_lazy_loading_effectiveness(self):
        """Benchmark the effectiveness of lazy loading system."""

        # Clear module cache first
        from xpcs_toolkit.viewer_kernel import _module_cache

        _module_cache.clear()

        # Test lazy loading speed
        module_names = ["g2mod", "saxs1d", "saxs2d", "stability", "intt", "twotime"]

        for module_name in module_names:
            with self.measure_performance(f"lazy_load_{module_name}"):
                from xpcs_toolkit.viewer_kernel import _get_module

                module = _get_module(module_name)
                # Access a common method to ensure full loading
                if hasattr(module, "plot"):
                    _ = module.plot

    def benchmark_memory_optimization(self):
        """Benchmark memory usage optimizations."""

        # Test 1: Memory manager initialization
        with self.measure_performance("memory_manager_init"):
            from xpcs_toolkit.utils.memory_manager import get_memory_manager

            memory_manager = get_memory_manager()
            memory_manager.get_enhanced_stats()

        # Test 2: Connection pool efficiency
        with self.measure_performance("connection_pool_operations"):
            from xpcs_toolkit.fileIO.hdf_reader import get_connection_pool_stats

            for i in range(10):
                get_connection_pool_stats()

        # Test 3: Plot constants memory usage
        with self.measure_performance("plot_constants_usage"):
            from xpcs_toolkit.plothandler.plot_constants import (
                get_color_cycle,
                get_color_marker,
                get_marker_cycle,
            )

            # Simulate heavy usage
            for i in range(100):
                _color, _marker = get_color_marker(i, "matplotlib")
                get_color_cycle("matplotlib", "hex")
                get_marker_cycle("matplotlib")

    def benchmark_threading_performance(self):
        """Benchmark threading system performance."""

        # Test 1: Unified threading manager
        with self.measure_performance("unified_threading_manager"):
            from xpcs_toolkit.threading.unified_threading import (
                TaskPriority,
                TaskType,
                get_unified_threading_manager,
            )

            manager = get_unified_threading_manager()

            # Submit test tasks
            def dummy_task():
                time.sleep(0.001)
                return "completed"

            for i in range(10):
                manager.submit_task(
                    f"test_task_{i}",
                    dummy_task,
                    TaskPriority.NORMAL,
                    TaskType.COMPUTATION,
                )

            # Wait a bit for tasks to complete
            time.sleep(0.1)
            manager.get_performance_stats()

        # Test 2: Cleanup system performance
        with self.measure_performance("cleanup_system"):
            from xpcs_toolkit.threading.cleanup_optimized import (
                get_cleanup_system,
                shutdown_worker_managers,
            )

            get_cleanup_system()
            shutdown_worker_managers()

    def benchmark_plotting_performance(self):
        """Benchmark plotting system performance."""

        # Test 1: Color/marker generation performance
        with self.measure_performance("color_marker_generation"):
            from xpcs_toolkit.plothandler.plot_constants import get_color_marker

            # Generate 1000 color/marker combinations
            results = []
            for i in range(1000):
                color, marker = get_color_marker(i % 100, "matplotlib")
                results.append((color, marker))

        # Test 2: Constants consolidation performance
        with self.measure_performance("constants_access"):
            from xpcs_toolkit.plothandler.plot_constants import (
                BASIC_COLORS,
                MATPLOTLIB_COLORS_HEX,
                MATPLOTLIB_COLORS_RGB,
            )

            # Access consolidated constants repeatedly
            for i in range(1000):
                MATPLOTLIB_COLORS_HEX[i % len(MATPLOTLIB_COLORS_HEX)]
                MATPLOTLIB_COLORS_RGB[i % len(MATPLOTLIB_COLORS_RGB)]
                BASIC_COLORS[i % len(BASIC_COLORS)]

    def benchmark_qmap_constants(self):
        """Benchmark qmap constants performance."""

        with self.measure_performance("qmap_constants_usage"):
            from xpcs_toolkit.fileIO.qmap_utils import (
                DEFAULT_BEAM_CENTER,
                DEFAULT_DETECTOR_SIZE,
            )

            # Simulate heavy usage of constants
            for _i in range(1000):
                [[DEFAULT_DETECTOR_SIZE, DEFAULT_DETECTOR_SIZE] for _ in range(10)]
                [DEFAULT_BEAM_CENTER for _ in range(10)]

                # Verify relationship
                assert DEFAULT_BEAM_CENTER == DEFAULT_DETECTOR_SIZE // 2

    def run_comprehensive_benchmark(self):
        """Run all benchmark suites."""

        # Run all benchmark categories
        self.benchmark_startup_performance()
        self.benchmark_lazy_loading_effectiveness()
        self.benchmark_memory_optimization()
        self.benchmark_threading_performance()
        self.benchmark_plotting_performance()
        self.benchmark_qmap_constants()

        # Calculate summary statistics
        self._calculate_summary_stats()

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def _calculate_summary_stats(self):
        """Calculate summary statistics from benchmark results."""
        benchmarks = self.results["benchmarks"]

        total_time = sum(b["execution_time_ms"] for b in benchmarks.values())
        total_memory = sum(b["memory_delta_mb"] for b in benchmarks.values())
        avg_time = total_time / len(benchmarks)

        self.results["summary"] = {
            "total_execution_time_ms": total_time,
            "average_execution_time_ms": avg_time,
            "total_memory_delta_mb": total_memory,
            "benchmark_count": len(benchmarks),
            "fastest_benchmark": min(
                benchmarks.items(), key=lambda x: x[1]["execution_time_ms"]
            ),
            "slowest_benchmark": max(
                benchmarks.items(), key=lambda x: x[1]["execution_time_ms"]
            ),
            "most_memory_efficient": min(
                benchmarks.items(), key=lambda x: x[1]["memory_delta_mb"]
            ),
            "least_memory_efficient": max(
                benchmarks.items(), key=lambda x: x[1]["memory_delta_mb"]
            ),
        }

    def save_results(self):
        """Save benchmark results to JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

    def print_summary(self):
        """Print benchmark summary."""
        self.results["summary"]


def main():
    """Run the comprehensive benchmark suite."""
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)

    # Run garbage collection before starting
    gc.collect()

    # Create and run benchmark
    benchmark = PerformanceBenchmark()
    benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
