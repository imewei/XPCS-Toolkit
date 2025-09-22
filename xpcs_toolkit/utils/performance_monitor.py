"""
Performance Monitoring and Benchmarking Infrastructure for XPCS Toolkit

This module provides comprehensive performance monitoring capabilities to track,
analyze, and benchmark the various optimizations implemented in the XPCS toolkit.
"""

import builtins
import json
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from .logging_config import get_logger
from .memory_manager import get_memory_manager

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0
    success: bool = True
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def throughput_mbs(self) -> float:
        """Calculate throughput in MB/s."""
        if self.duration > 0 and self.input_size_mb > 0:
            return self.input_size_mb / self.duration
        return 0.0

    @property
    def memory_efficiency(self) -> float:
        """Calculate memory efficiency (input/peak memory)."""
        if self.memory_peak_mb > 0:
            return self.input_size_mb / self.memory_peak_mb
        return 0.0


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    benchmark_name: str
    timestamp: str
    system_info: dict[str, Any]
    metrics: list[PerformanceMetrics]
    summary: dict[str, Any]
    duration_total: float
    success_rate: float


class PerformanceProfiler:
    """Context manager for profiling performance of operations."""

    def __init__(
        self,
        operation_name: str,
        input_size_mb: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.operation_name = operation_name
        self.input_size_mb = input_size_mb
        self.metadata = metadata or {}

        # Metrics tracking
        self.start_time = None
        self.end_time = None
        self.memory_before_mb = None
        self.memory_after_mb = None
        self.memory_peak_mb = None
        self.cpu_percent = None
        self.success = True
        self.error_message = ""

        # Memory manager for additional context
        self.memory_manager = get_memory_manager()

    def __enter__(self):
        """Start profiling."""
        self.start_time = time.time()

        # Capture initial memory state
        memory = psutil.virtual_memory()
        self.memory_before_mb = (memory.total - memory.available) / (1024 * 1024)
        self.memory_peak_mb = self.memory_before_mb

        # Start CPU monitoring
        psutil.cpu_percent()  # First call to initialize

        logger.debug(f"Started profiling operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End profiling and create metrics."""
        self.end_time = time.time()

        # Capture final memory state
        memory = psutil.virtual_memory()
        self.memory_after_mb = (memory.total - memory.available) / (1024 * 1024)

        # Get CPU usage
        self.cpu_percent = psutil.cpu_percent()

        # Handle exceptions
        if exc_type is not None:
            self.success = False
            self.error_message = str(exc_val)
            logger.warning(f"Operation {self.operation_name} failed: {exc_val}")

        # Create performance metrics
        metrics = PerformanceMetrics(
            operation_name=self.operation_name,
            start_time=self.start_time,
            end_time=self.end_time,
            duration=self.end_time - self.start_time,
            memory_before_mb=self.memory_before_mb,
            memory_after_mb=self.memory_after_mb,
            memory_peak_mb=max(self.memory_peak_mb, self.memory_after_mb),
            memory_delta_mb=self.memory_after_mb - self.memory_before_mb,
            cpu_percent=self.cpu_percent,
            input_size_mb=self.input_size_mb,
            success=self.success,
            error_message=self.error_message,
            metadata=self.metadata,
        )

        # Register metrics with global monitor
        PerformanceMonitor.get_instance().record_metrics(metrics)

        logger.debug(
            f"Completed profiling {self.operation_name}: "
            f"{metrics.duration:.3f}s, memory: {metrics.memory_delta_mb:+.1f}MB"
        )

    def update_memory_peak(self, current_memory_mb: float):
        """Update peak memory usage during operation."""
        self.memory_peak_mb = max(self.memory_peak_mb, current_memory_mb)


class PerformanceMonitor:
    """Global performance monitoring system."""

    _instance = None

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.operation_stats: dict[str, list[PerformanceMetrics]] = defaultdict(list)
        self.benchmarks: list[BenchmarkResult] = []

        # System information
        self.system_info = self._get_system_info()

        logger.info("PerformanceMonitor initialized")

    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        """Get or create global performance monitor instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history.append(metrics)
        self.operation_stats[metrics.operation_name].append(metrics)

        # Limit per-operation history to prevent unbounded growth
        if len(self.operation_stats[metrics.operation_name]) > 1000:
            self.operation_stats[metrics.operation_name].pop(0)

    def get_operation_stats(self, operation_name: str) -> dict[str, Any]:
        """Get statistics for a specific operation."""
        metrics_list = self.operation_stats.get(operation_name, [])
        if not metrics_list:
            return {}

        # Calculate statistics
        durations = [m.duration for m in metrics_list if m.success]
        memory_deltas = [m.memory_delta_mb for m in metrics_list if m.success]
        throughputs = [
            m.throughput_mbs for m in metrics_list if m.success and m.throughput_mbs > 0
        ]

        stats = {
            "operation_name": operation_name,
            "total_calls": len(metrics_list),
            "successful_calls": len(durations),
            "success_rate": len(durations) / len(metrics_list) if metrics_list else 0.0,
            "duration_stats": self._calculate_stats(durations),
            "memory_stats": self._calculate_stats(memory_deltas),
            "throughput_stats": self._calculate_stats(throughputs),
            "recent_performance": durations[-10:] if durations else [],
        }

        return stats

    def get_overall_stats(self) -> dict[str, Any]:
        """Get overall performance statistics."""
        all_operations = list(self.operation_stats.keys())
        total_metrics = len(self.metrics_history)
        successful_operations = sum(1 for m in self.metrics_history if m.success)

        stats = {
            "total_operations": len(all_operations),
            "total_metrics": total_metrics,
            "overall_success_rate": successful_operations / total_metrics
            if total_metrics
            else 0.0,
            "operations_summary": {
                op: self.get_operation_stats(op) for op in all_operations
            },
            "system_info": self.system_info,
            "monitoring_period": {
                "start_time": min(m.start_time for m in self.metrics_history)
                if self.metrics_history
                else None,
                "end_time": max(m.end_time for m in self.metrics_history)
                if self.metrics_history
                else None,
            },
        }

        return stats

    def export_metrics(self, filepath: str):
        """Export performance metrics to JSON file."""
        stats = self.get_overall_stats()

        # Convert metrics to serializable format
        serializable_stats = self._make_serializable(stats)

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Performance metrics exported to {filepath}")

    def clear_history(self):
        """Clear performance metrics history."""
        self.metrics_history.clear()
        self.operation_stats.clear()
        logger.info("Performance metrics history cleared")

    def _get_system_info(self) -> dict[str, Any]:
        """Collect system information."""
        memory = psutil.virtual_memory()
        cpu_info = {}

        with suppress(builtins.BaseException):
            cpu_info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

        return {
            "platform": psutil.WINDOWS if hasattr(psutil, "WINDOWS") else "unix",
            "memory_total_gb": memory.total / (1024**3),
            "cpu_info": cpu_info,
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            "numpy_version": np.__version__,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_stats(self, values: list[float]) -> dict[str, float]:
        """Calculate basic statistics for a list of values."""
        if not values:
            return {}

        values_array = np.array(values)
        return {
            "count": len(values),
            "mean": float(np.mean(values_array)),
            "median": float(np.median(values_array)),
            "std": float(np.std(values_array)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
            "p95": float(np.percentile(values_array, 95)),
            "p99": float(np.percentile(values_array, 99)),
        }

    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "_asdict"):  # namedtuple
            return self._make_serializable(obj._asdict())
        return obj


class XPCSBenchmarkSuite:
    """Comprehensive benchmark suite for XPCS performance testing."""

    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = PerformanceMonitor.get_instance()

    def run_comprehensive_benchmark(
        self, sample_data_path: str | None = None
    ) -> BenchmarkResult:
        """Run comprehensive benchmark of all XPCS optimizations."""
        benchmark_start = time.time()
        logger.info("Starting comprehensive XPCS benchmark suite")

        # Clear previous metrics
        self.monitor.clear_history()

        benchmark_metrics = []

        try:
            # Memory management benchmarks
            benchmark_metrics.extend(self._benchmark_memory_management())

            # I/O performance benchmarks
            if sample_data_path:
                benchmark_metrics.extend(
                    self._benchmark_io_operations(sample_data_path)
                )

            # G2 fitting benchmarks
            benchmark_metrics.extend(self._benchmark_g2_fitting())

            # ROI calculation benchmarks
            benchmark_metrics.extend(self._benchmark_roi_calculations())

            # SAXS processing benchmarks
            benchmark_metrics.extend(self._benchmark_saxs_processing())

        except Exception as e:
            logger.error(f"Benchmark suite failed: {e}")
            logger.error(traceback.format_exc())

        # Calculate overall results
        benchmark_end = time.time()
        duration_total = benchmark_end - benchmark_start
        successful_metrics = [m for m in benchmark_metrics if m.success]
        success_rate = (
            len(successful_metrics) / len(benchmark_metrics)
            if benchmark_metrics
            else 0.0
        )

        # Create summary
        summary = self._create_benchmark_summary(benchmark_metrics)

        result = BenchmarkResult(
            benchmark_name="comprehensive_xpcs_benchmark",
            timestamp=datetime.now().isoformat(),
            system_info=self.monitor.system_info,
            metrics=benchmark_metrics,
            summary=summary,
            duration_total=duration_total,
            success_rate=success_rate,
        )

        # Export results
        self._export_benchmark_result(result)

        logger.info(
            f"Comprehensive benchmark completed: {duration_total:.1f}s, "
            f"success rate: {success_rate:.1%}"
        )

        return result

    def _benchmark_memory_management(self) -> list[PerformanceMetrics]:
        """Benchmark memory management optimizations."""
        logger.info("Benchmarking memory management...")

        memory_manager = get_memory_manager()

        # Test cache operations
        with PerformanceProfiler(
            "memory_cache_operations", metadata={"test": "cache_ops"}
        ) as profiler:
            # Create test data
            test_data = np.random.random((1000, 1000)).astype(np.float32)
            data_size_mb = test_data.nbytes / (1024 * 1024)
            profiler.input_size_mb = data_size_mb

            # Cache operations
            for i in range(100):
                cache_key = f"test_data_{i}"
                memory_manager.cache_put(cache_key, test_data, "array_data")
                retrieved = memory_manager.cache_get(cache_key, "array_data")
                assert retrieved is not None

        # Test memory pressure detection
        with PerformanceProfiler(
            "memory_pressure_detection", metadata={"test": "pressure"}
        ) as profiler:
            for i in range(1000):
                memory_manager.get_memory_pressure()
                memory_manager.get_cache_stats()

        # Test cleanup operations
        with PerformanceProfiler(
            "memory_cleanup", metadata={"test": "cleanup"}
        ) as profiler:
            memory_manager._aggressive_cleanup()

        return [
            m
            for m in self.monitor.metrics_history
            if m.operation_name.startswith("memory_")
        ]

    def _benchmark_io_operations(
        self, sample_data_path: str
    ) -> list[PerformanceMetrics]:
        """Benchmark I/O operations with enhanced HDF5 reader."""
        logger.info("Benchmarking I/O operations...")

        from ..fileIO.hdf_reader import batch_read_fields
        from ..fileIO.hdf_reader_enhanced import get_enhanced_reader

        enhanced_reader = get_enhanced_reader()

        # Benchmark enhanced vs standard HDF5 reading
        sample_path = Path(sample_data_path)
        if sample_path.exists():
            # Test enhanced reader
            with PerformanceProfiler(
                "hdf5_enhanced_read", metadata={"file": str(sample_path)}
            ) as profiler:
                try:
                    data = enhanced_reader.read_dataset(
                        str(sample_path), "/xpcs/scattering_2d", enable_read_ahead=True
                    )
                    if hasattr(data, "nbytes"):
                        profiler.input_size_mb = data.nbytes / (1024 * 1024)
                except Exception as e:
                    logger.warning(f"Enhanced HDF5 read test failed: {e}")

            # Test standard reader for comparison
            with PerformanceProfiler(
                "hdf5_standard_read", metadata={"file": str(sample_path)}
            ) as profiler:
                try:
                    result = batch_read_fields(
                        str(sample_path), ["saxs_2d"], "alias", ftype="nexus"
                    )
                    if "saxs_2d" in result and hasattr(result["saxs_2d"], "nbytes"):
                        profiler.input_size_mb = result["saxs_2d"].nbytes / (
                            1024 * 1024
                        )
                except Exception as e:
                    logger.warning(f"Standard HDF5 read test failed: {e}")

        return [m for m in self.monitor.metrics_history if "hdf5" in m.operation_name]

    def _benchmark_g2_fitting(self) -> list[PerformanceMetrics]:
        """Benchmark G2 fitting performance (sequential vs parallel)."""
        logger.info("Benchmarking G2 fitting...")

        # Create synthetic G2 data for testing
        num_tau = 100
        num_q = 50

        tau = np.logspace(-6, 2, num_tau)
        g2_synthetic = np.zeros((num_tau, num_q))
        g2_err_synthetic = np.zeros((num_tau, num_q))

        # Generate synthetic G2 curves
        for q in range(num_q):
            # Single exponential with noise
            true_tau = 10 ** (np.random.uniform(-3, 1))  # Random relaxation time
            contrast = np.random.uniform(0.1, 0.5)
            baseline = 1.0

            g2_true = contrast * np.exp(-2 * tau / true_tau) + baseline
            noise = np.random.normal(0, 0.01, num_tau)
            g2_synthetic[:, q] = g2_true + noise
            g2_err_synthetic[:, q] = np.abs(noise) + 0.001

        # Import fitting functions
        from ..helper.fitting import fit_with_fixed, fit_with_fixed_parallel, single_exp

        # Prepare fitting parameters
        bounds = np.array(
            [[0.01, 1e-6, 0.8], [1.0, 1e6, 1.2]]
        )  # [contrast, tau, baseline]
        fit_flag = np.array([True, True, True])
        fit_x = np.logspace(
            np.log10(np.min(tau)) - 0.5, np.log10(np.max(tau)) + 0.5, 128
        )

        # Benchmark sequential fitting
        with PerformanceProfiler(
            "g2_fitting_sequential",
            input_size_mb=g2_synthetic.nbytes / (1024 * 1024),
            metadata={"num_q": num_q, "num_tau": num_tau},
        ):
            try:
                _fit_line_seq, _fit_val_seq = fit_with_fixed(
                    single_exp,
                    tau,
                    g2_synthetic,
                    g2_err_synthetic,
                    bounds,
                    fit_flag,
                    fit_x,
                )
            except Exception as e:
                logger.warning(f"Sequential G2 fitting test failed: {e}")

        # Benchmark parallel fitting
        with PerformanceProfiler(
            "g2_fitting_parallel",
            input_size_mb=g2_synthetic.nbytes / (1024 * 1024),
            metadata={"num_q": num_q, "num_tau": num_tau},
        ):
            try:
                _fit_line_par, _fit_val_par = fit_with_fixed_parallel(
                    single_exp,
                    tau,
                    g2_synthetic,
                    g2_err_synthetic,
                    bounds,
                    fit_flag,
                    fit_x,
                    max_workers=4,
                )
            except Exception as e:
                logger.warning(f"Parallel G2 fitting test failed: {e}")

        return [
            m for m in self.monitor.metrics_history if "g2_fitting" in m.operation_name
        ]

    def _benchmark_roi_calculations(self) -> list[PerformanceMetrics]:
        """Benchmark ROI calculation performance."""
        logger.info("Benchmarking ROI calculations...")

        # Create synthetic SAXS data
        shape = (10, 512, 512)  # 10 time frames, 512x512 detector
        saxs_data = np.random.poisson(100, shape).astype(np.float32)
        data_size_mb = saxs_data.nbytes / (1024 * 1024)

        # Create geometry data
        y, x = np.mgrid[: shape[1], : shape[2]]
        center_x, center_y = shape[2] // 2, shape[1] // 2

        geometry_data = {
            "qmap": np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            * 0.01,  # Synthetic q-map
            "pmap": np.degrees(np.arctan2(y - center_y, x - center_x))
            % 360,  # Angular map
            "rmap": np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2),  # Radial map
            "mask": np.ones(shape[1:], dtype=bool),
        }

        # Import ROI calculators
        from ..utils.vectorized_roi import (
            PieROICalculator,
            RingROICalculator,
            ROIParameters,
            ROIType,
        )

        # Benchmark pie ROI calculation
        pie_params = ROIParameters(
            roi_type=ROIType.PIE,
            parameters={
                "angle_range": (45, 135),
                "qmap_idx": np.arange(np.max(geometry_data["qmap"]) + 1),
                "qsize": 100,
                "qspan": np.linspace(0, np.max(geometry_data["qmap"]), 101),
            },
        )

        with PerformanceProfiler(
            "roi_pie_vectorized",
            input_size_mb=data_size_mb,
            metadata={"roi_type": "pie", "shape": shape},
        ):
            try:
                pie_calculator = PieROICalculator()
                pie_calculator.calculate_roi(saxs_data, geometry_data, pie_params)
            except Exception as e:
                logger.warning(f"Pie ROI benchmark failed: {e}")

        # Benchmark ring ROI calculation
        ring_params = ROIParameters(
            roi_type=ROIType.RING,
            parameters={
                "radius_range": (50, 150),
                "pmap": geometry_data["pmap"],
                "phi_num": 180,
            },
        )

        with PerformanceProfiler(
            "roi_ring_vectorized",
            input_size_mb=data_size_mb,
            metadata={"roi_type": "ring", "shape": shape},
        ):
            try:
                ring_calculator = RingROICalculator()
                ring_calculator.calculate_roi(saxs_data, geometry_data, ring_params)
            except Exception as e:
                logger.warning(f"Ring ROI benchmark failed: {e}")

        return [m for m in self.monitor.metrics_history if "roi_" in m.operation_name]

    def _benchmark_saxs_processing(self) -> list[PerformanceMetrics]:
        """Benchmark SAXS data processing (log computation, streaming)."""
        logger.info("Benchmarking SAXS processing...")

        # Create synthetic SAXS data
        shape = (100, 256, 256)  # Large SAXS dataset
        saxs_data = np.random.poisson(50, shape).astype(np.float32)
        data_size_mb = saxs_data.nbytes / (1024 * 1024)

        # Benchmark standard log computation
        with PerformanceProfiler(
            "saxs_log_standard", input_size_mb=data_size_mb, metadata={"shape": shape}
        ):
            try:
                # Standard log computation
                saxs_copy = np.copy(saxs_data)
                roi = saxs_copy > 0
                if np.sum(roi) > 0:
                    min_val = np.min(saxs_copy[roi])
                    saxs_copy[~roi] = min_val
                    np.log10(saxs_copy).astype(np.float32)
            except Exception as e:
                logger.warning(f"Standard SAXS log benchmark failed: {e}")

        # Benchmark streaming log computation
        with PerformanceProfiler(
            "saxs_log_streaming", input_size_mb=data_size_mb, metadata={"shape": shape}
        ):
            try:
                from ..utils.streaming_processor import process_saxs_log_streaming

                process_saxs_log_streaming(saxs_data, chunk_size_mb=20.0)
            except Exception as e:
                logger.warning(f"Streaming SAXS log benchmark failed: {e}")

        return [m for m in self.monitor.metrics_history if "saxs_" in m.operation_name]

    def _create_benchmark_summary(
        self, metrics: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Create summary of benchmark results."""
        if not metrics:
            return {}

        # Group metrics by operation
        operation_groups = defaultdict(list)
        for metric in metrics:
            operation_groups[metric.operation_name].append(metric)

        summary = {
            "total_operations": len(operation_groups),
            "total_metrics": len(metrics),
            "overall_success_rate": sum(1 for m in metrics if m.success) / len(metrics),
            "performance_comparison": {},
            "memory_efficiency": {},
            "throughput_analysis": {},
        }

        # Analyze performance comparisons
        for op_name, op_metrics in operation_groups.items():
            successful_metrics = [m for m in op_metrics if m.success]
            if successful_metrics:
                durations = [m.duration for m in successful_metrics]
                memory_deltas = [m.memory_delta_mb for m in successful_metrics]
                throughputs = [
                    m.throughput_mbs for m in successful_metrics if m.throughput_mbs > 0
                ]

                summary["performance_comparison"][op_name] = {
                    "average_duration": np.mean(durations),
                    "average_memory_delta": np.mean(memory_deltas),
                    "average_throughput": np.mean(throughputs) if throughputs else 0.0,
                    "sample_count": len(successful_metrics),
                }

        # Highlight performance improvements
        if (
            "g2_fitting_sequential" in summary["performance_comparison"]
            and "g2_fitting_parallel" in summary["performance_comparison"]
        ):
            seq_time = summary["performance_comparison"]["g2_fitting_sequential"][
                "average_duration"
            ]
            par_time = summary["performance_comparison"]["g2_fitting_parallel"][
                "average_duration"
            ]
            if par_time > 0:
                speedup = seq_time / par_time
                summary["g2_fitting_speedup"] = speedup

        return summary

    def _export_benchmark_result(self, result: BenchmarkResult):
        """Export benchmark result to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"xpcs_benchmark_{timestamp}.json"
        filepath = self.output_dir / filename

        # Convert to serializable format
        result_dict = asdict(result)
        serializable_result = self.monitor._make_serializable(result_dict)

        with open(filepath, "w") as f:
            json.dump(serializable_result, f, indent=2)

        logger.info(f"Benchmark results exported to {filepath}")


# Convenience decorators for easy performance monitoring
def profile_performance(
    operation_name: str | None = None, track_input_size: bool = False
):
    """Decorator to automatically profile function performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Determine operation name
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Try to estimate input size if requested
            input_size_mb = 0.0
            if track_input_size and args:
                try:
                    # Look for numpy arrays in arguments
                    for arg in args:
                        if hasattr(arg, "nbytes"):
                            input_size_mb += arg.nbytes / (1024 * 1024)
                except:
                    pass

            # Profile the function execution
            with PerformanceProfiler(op_name, input_size_mb=input_size_mb):
                return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def benchmark_context(
    operation_name: str,
    input_size_mb: float = 0.0,
    metadata: dict[str, Any] | None = None,
):
    """Context manager for benchmarking code blocks."""
    with PerformanceProfiler(operation_name, input_size_mb, metadata):
        yield


# Global performance monitor instance
def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return PerformanceMonitor.get_instance()


def run_xpcs_benchmark(
    sample_data_path: str | None = None, output_dir: str = "benchmarks"
) -> BenchmarkResult:
    """Convenience function to run the complete XPCS benchmark suite."""
    suite = XPCSBenchmarkSuite(output_dir)
    return suite.run_comprehensive_benchmark(sample_data_path)
