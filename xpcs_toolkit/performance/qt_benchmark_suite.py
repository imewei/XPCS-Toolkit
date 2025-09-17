"""
Comprehensive Qt Compliance Benchmarking Suite.

This module provides comprehensive benchmarking capabilities for the Qt
compliance system, measuring performance across different scenarios,
data volumes, and system configurations.
"""

import gc
import json
import statistics
import threading
import time
import traceback
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from PySide6.QtCore import QCoreApplication, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QWidget

from ..monitoring import (
    get_qt_error_detector,
    get_integrated_monitoring_system,
    initialize_integrated_monitoring,
    shutdown_integrated_monitoring
)
from ..threading import get_qt_compliant_thread_manager, SafeWorkerBase
from .qt_performance_profiler import get_qt_performance_profiler, ProfilerConfiguration, ProfilerType
from .qt_optimization_engine import get_qt_optimization_engine, OptimizationConfiguration
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of benchmarks available."""

    PERFORMANCE = "performance"
    SCALABILITY = "scalability"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    STRESS = "stress"
    REGRESSION = "regression"


class BenchmarkScope(Enum):
    """Scope of benchmark execution."""

    UNIT = "unit"           # Single component
    INTEGRATION = "integration"  # Multiple components
    SYSTEM = "system"       # Full system
    END_TO_END = "end_to_end"   # Complete user workflow


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark execution."""

    benchmark_type: BenchmarkType = BenchmarkType.PERFORMANCE
    scope: BenchmarkScope = BenchmarkScope.INTEGRATION
    duration_seconds: float = 30.0
    iterations: int = 100
    warmup_iterations: int = 10
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    data_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    enable_profiling: bool = True
    enable_optimization: bool = False
    output_directory: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkMetrics:
    """Metrics collected during benchmark execution."""

    timestamp: float
    benchmark_name: str
    configuration: BenchmarkConfiguration

    # Performance metrics
    execution_time_ms: float = 0.0
    throughput_ops_per_second: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0

    # Resource metrics
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    file_descriptors: int = 0
    thread_count: int = 0

    # Qt-specific metrics
    qt_errors_count: int = 0
    qt_warnings_processed: int = 0
    qt_operations_per_second: float = 0.0
    signal_emissions: int = 0

    # Quality metrics
    success_rate: float = 0.0
    error_rate: float = 0.0
    stability_score: float = 0.0

    # Raw data
    execution_times: List[float] = field(default_factory=list)
    memory_samples: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of a benchmark execution."""

    benchmark_name: str
    timestamp: float
    success: bool
    metrics: BenchmarkMetrics
    comparison_baseline: Optional[BenchmarkMetrics] = None
    performance_delta: Optional[float] = None  # Percentage change from baseline
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        if self.comparison_baseline:
            result['comparison_baseline'] = self.comparison_baseline.to_dict()
        return result


class QtComplianceBenchmarkSuite:
    """
    Comprehensive benchmarking suite for Qt compliance system.

    Provides:
    - Performance benchmarking across different scenarios
    - Scalability testing with varying loads
    - Memory usage analysis
    - Throughput and latency measurements
    - Stress testing capabilities
    - Regression detection
    """

    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        """Initialize benchmark suite."""
        self.config = config or BenchmarkConfiguration()

        # Results storage
        self.benchmark_results: Dict[str, BenchmarkResult] = {}
        self.baseline_results: Dict[str, BenchmarkMetrics] = {}

        # Components
        self._profiler = get_qt_performance_profiler()
        self._optimization_engine = get_qt_optimization_engine()

        # Benchmark registry
        self._benchmarks: Dict[str, Callable] = {}
        self._register_standard_benchmarks()

        logger.info("Qt compliance benchmark suite initialized")

    def _register_standard_benchmarks(self):
        """Register standard benchmarks."""
        self._benchmarks.update({
            "qt_error_detection_performance": self._benchmark_qt_error_detection,
            "widget_creation_throughput": self._benchmark_widget_creation,
            "timer_management_scalability": self._benchmark_timer_management,
            "signal_slot_latency": self._benchmark_signal_slot_performance,
            "thread_pool_efficiency": self._benchmark_thread_pool_performance,
            "memory_usage_patterns": self._benchmark_memory_patterns,
            "monitoring_system_overhead": self._benchmark_monitoring_overhead,
            "filtering_algorithm_performance": self._benchmark_filtering_performance,
            "optimization_effectiveness": self._benchmark_optimization_impact,
            "full_system_stress_test": self._benchmark_full_system_stress
        })

    def run_benchmark(self, benchmark_name: str, config: Optional[BenchmarkConfiguration] = None) -> BenchmarkResult:
        """Run a specific benchmark."""
        if benchmark_name not in self._benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        benchmark_config = config or self.config
        logger.info(f"Running benchmark: {benchmark_name}")

        # Initialize monitoring if enabled
        monitoring_system = None
        if benchmark_config.enable_profiling:
            monitoring_system = initialize_integrated_monitoring()

        try:
            # Run benchmark
            benchmark_func = self._benchmarks[benchmark_name]
            metrics = benchmark_func(benchmark_config)

            # Create result
            result = BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp=time.perf_counter(),
                success=True,
                metrics=metrics
            )

            # Compare with baseline if available
            if benchmark_name in self.baseline_results:
                result.comparison_baseline = self.baseline_results[benchmark_name]
                result.performance_delta = self._calculate_performance_delta(
                    metrics, result.comparison_baseline
                )

            # Generate recommendations
            result.recommendations = self._generate_recommendations(metrics)

            # Store result
            self.benchmark_results[benchmark_name] = result

            logger.info(f"Benchmark completed: {benchmark_name} "
                       f"({metrics.execution_time_ms:.1f}ms, "
                       f"{metrics.throughput_ops_per_second:.1f} ops/sec)")

            return result

        except Exception as e:
            logger.error(f"Benchmark failed: {benchmark_name} - {e}")
            error_result = BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp=time.perf_counter(),
                success=False,
                metrics=BenchmarkMetrics(
                    timestamp=time.perf_counter(),
                    benchmark_name=benchmark_name,
                    configuration=benchmark_config,
                    errors=[str(e)]
                )
            )
            return error_result

        finally:
            if monitoring_system:
                shutdown_integrated_monitoring()

    def _benchmark_qt_error_detection(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark Qt error detection performance."""
        qt_detector = get_qt_error_detector()
        qt_detector.clear_error_history()

        # Prepare test messages
        test_messages = self._generate_test_qt_messages(max(config.data_sizes))

        execution_times = []
        total_errors_detected = 0

        # Warmup
        for _ in range(config.warmup_iterations):
            qt_detector._process_qt_message(test_messages[0]["msg_type"], test_messages[0]["message"])

        # Benchmark iterations
        for iteration in range(config.iterations):
            start_time = time.perf_counter()

            # Process batch of messages
            batch_size = config.data_sizes[iteration % len(config.data_sizes)]
            batch_messages = test_messages[:batch_size]

            for msg in batch_messages:
                error = qt_detector._process_qt_message(msg["msg_type"], msg["message"])
                if error:
                    total_errors_detected += 1

            execution_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(execution_time)

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = (sum(config.data_sizes) * config.iterations) / (sum(execution_times) / 1000)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="qt_error_detection_performance",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            latency_p50_ms=statistics.median(execution_times),
            latency_p95_ms=statistics.quantiles(execution_times, n=20)[18] if len(execution_times) >= 20 else max(execution_times),
            latency_p99_ms=statistics.quantiles(execution_times, n=100)[98] if len(execution_times) >= 100 else max(execution_times),
            qt_errors_count=total_errors_detected,
            qt_warnings_processed=sum(config.data_sizes) * config.iterations,
            execution_times=execution_times
        )

    def _benchmark_widget_creation(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark widget creation throughput."""
        execution_times = []
        memory_samples = []
        widgets_created = 0

        # Warmup
        for _ in range(config.warmup_iterations):
            widget = QWidget()
            widget.deleteLater()
            QCoreApplication.processEvents()

        # Benchmark iterations
        for iteration in range(config.iterations):
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()

            # Create batch of widgets
            batch_size = config.data_sizes[iteration % len(config.data_sizes)]
            widgets = []

            for i in range(batch_size):
                widget = QWidget()
                widget.setObjectName(f"benchmark_widget_{i}")
                widgets.append(widget)
                widgets_created += 1

            # Cleanup widgets
            for widget in widgets:
                widget.deleteLater()

            QCoreApplication.processEvents()

            execution_time = (time.perf_counter() - start_time) * 1000
            memory_usage = self._get_memory_usage()

            execution_times.append(execution_time)
            memory_samples.append(memory_usage - start_memory)

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = widgets_created / (sum(execution_times) / 1000)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="widget_creation_throughput",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            latency_p50_ms=statistics.median(execution_times),
            peak_memory_mb=max(memory_samples) if memory_samples else 0.0,
            execution_times=execution_times,
            memory_samples=memory_samples
        )

    def _benchmark_timer_management(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark timer management scalability."""
        execution_times = []
        timers_managed = 0

        # Benchmark with different concurrency levels
        for concurrency in config.concurrency_levels:
            timers = []

            start_time = time.perf_counter()

            # Create timers
            for i in range(concurrency * 10):  # 10 timers per concurrency level
                timer = QTimer()
                timer.setObjectName(f"benchmark_timer_{i}")
                timer.timeout.connect(lambda: None)
                timer.start(1000)  # 1 second interval
                timers.append(timer)
                timers_managed += 1

            # Let timers run briefly
            time.sleep(0.1)

            # Stop and cleanup timers
            for timer in timers:
                timer.stop()
                timer.deleteLater()

            execution_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(execution_time)

            QCoreApplication.processEvents()

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = timers_managed / (sum(execution_times) / 1000)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="timer_management_scalability",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            latency_p50_ms=statistics.median(execution_times),
            execution_times=execution_times
        )

    def _benchmark_signal_slot_performance(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark signal/slot connection and emission performance."""
        from PySide6.QtCore import QObject, Signal

        class BenchmarkObject(QObject):
            test_signal = Signal(int)

            def __init__(self):
                super().__init__()
                self.signal_count = 0

            def slot_handler(self, value):
                self.signal_count += 1

        execution_times = []
        signal_emissions = 0

        # Benchmark iterations
        for iteration in range(config.iterations):
            start_time = time.perf_counter()

            # Create objects and connections
            obj = BenchmarkObject()
            obj.test_signal.connect(obj.slot_handler)

            # Emit signals
            batch_size = config.data_sizes[iteration % len(config.data_sizes)]
            for i in range(batch_size):
                obj.test_signal.emit(i)
                signal_emissions += 1

            QCoreApplication.processEvents()

            execution_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(execution_time)

            # Cleanup
            obj.deleteLater()

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = signal_emissions / (sum(execution_times) / 1000)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="signal_slot_latency",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            latency_p50_ms=statistics.median(execution_times),
            signal_emissions=signal_emissions,
            execution_times=execution_times
        )

    def _benchmark_thread_pool_performance(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark thread pool efficiency."""
        thread_manager = get_qt_compliant_thread_manager()

        class BenchmarkWorker(SafeWorkerBase):
            def __init__(self, work_duration=0.01):
                super().__init__()
                self.work_duration = work_duration
                self.completed = False

            def do_work(self):
                time.sleep(self.work_duration)
                self.completed = True

        execution_times = []
        workers_completed = 0

        # Benchmark with different concurrency levels
        for concurrency in config.concurrency_levels:
            workers = []

            start_time = time.perf_counter()

            # Create and start workers
            for i in range(concurrency * 5):  # 5 workers per concurrency level
                worker = BenchmarkWorker()
                workers.append(worker)
                thread_manager.start_worker(worker, f"benchmark_worker_{i}")

            # Wait for completion
            timeout = 10.0
            while time.perf_counter() - start_time < timeout:
                completed_count = sum(1 for w in workers if w.completed)
                if completed_count == len(workers):
                    break
                time.sleep(0.01)

            execution_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(execution_time)

            workers_completed += sum(1 for w in workers if w.completed)

            # Cleanup
            thread_manager.cleanup_finished_threads()

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = workers_completed / (sum(execution_times) / 1000)
        efficiency = workers_completed / (sum(config.concurrency_levels) * 5)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="thread_pool_efficiency",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            success_rate=efficiency,
            execution_times=execution_times
        )

    def _benchmark_memory_patterns(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark memory usage patterns."""
        initial_memory = self._get_memory_usage()
        memory_samples = []
        peak_memory = initial_memory

        # Memory stress test
        for iteration in range(config.iterations):
            start_memory = self._get_memory_usage()

            # Create objects
            objects = []
            batch_size = config.data_sizes[iteration % len(config.data_sizes)]

            for i in range(batch_size):
                widget = QWidget()
                timer = QTimer(widget)
                objects.append((widget, timer))

            current_memory = self._get_memory_usage()
            peak_memory = max(peak_memory, current_memory)
            memory_samples.append(current_memory - start_memory)

            # Cleanup
            for widget, timer in objects:
                widget.deleteLater()

            objects.clear()
            gc.collect()

            # Process cleanup events
            for _ in range(5):
                QCoreApplication.processEvents()
                time.sleep(0.001)

        final_memory = self._get_memory_usage()
        memory_growth = final_memory - initial_memory
        memory_leaked = max(0, memory_growth)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="memory_usage_patterns",
            configuration=config,
            peak_memory_mb=peak_memory,
            memory_samples=memory_samples
        )

    def _benchmark_monitoring_overhead(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark monitoring system overhead."""
        # Baseline measurement without monitoring
        baseline_times = []
        for _ in range(config.iterations // 2):
            start_time = time.perf_counter()

            # Perform standard operations
            widget = QWidget()
            timer = QTimer(widget)
            timer.start(100)
            timer.stop()
            widget.deleteLater()
            QCoreApplication.processEvents()

            baseline_times.append((time.perf_counter() - start_time) * 1000)

        # Measurement with monitoring enabled
        monitoring_system = initialize_integrated_monitoring()
        monitoring_times = []

        try:
            for _ in range(config.iterations // 2):
                start_time = time.perf_counter()

                # Same operations with monitoring
                widget = QWidget()
                timer = QTimer(widget)
                timer.start(100)
                timer.stop()
                widget.deleteLater()
                QCoreApplication.processEvents()

                monitoring_times.append((time.perf_counter() - start_time) * 1000)

        finally:
            shutdown_integrated_monitoring()

        # Calculate overhead
        baseline_avg = statistics.mean(baseline_times)
        monitoring_avg = statistics.mean(monitoring_times)
        overhead_percent = ((monitoring_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="monitoring_system_overhead",
            configuration=config,
            execution_time_ms=monitoring_avg,
            execution_times=monitoring_times + baseline_times
        )

    def _benchmark_filtering_performance(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark filtering algorithm performance."""
        optimization_engine = get_qt_optimization_engine()
        message_filter = optimization_engine.message_filter

        # Generate test messages
        test_messages = [
            "QStyleHints::colorSchemeChanged() connection warning",
            "QObject::connect: Unique connection requires a pointer to member function",
            "qt.svg renderer warning about invalid element",
            "QTimer::start: Timers can only be used with threads started with QThread",
            "Normal message that should not be filtered"
        ] * (max(config.data_sizes) // 5)

        execution_times = []
        messages_processed = 0

        # Benchmark filtering performance
        for iteration in range(config.iterations):
            batch_size = config.data_sizes[iteration % len(config.data_sizes)]
            batch_messages = test_messages[:batch_size]

            start_time = time.perf_counter()

            # Filter messages
            if config.enable_optimization:
                results = message_filter.filter_messages_batch(batch_messages)
            else:
                results = [message_filter.filter_message(msg) for msg in batch_messages]

            execution_time = (time.perf_counter() - start_time) * 1000
            execution_times.append(execution_time)
            messages_processed += len(batch_messages)

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times)
        throughput = messages_processed / (sum(execution_times) / 1000)

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="filtering_algorithm_performance",
            configuration=config,
            execution_time_ms=avg_execution_time,
            throughput_ops_per_second=throughput,
            qt_warnings_processed=messages_processed,
            execution_times=execution_times
        )

    def _benchmark_optimization_impact(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark optimization effectiveness."""
        optimization_engine = get_qt_optimization_engine()

        # Measure baseline performance
        baseline_start = time.perf_counter()
        self._perform_standard_qt_operations(config.iterations)
        baseline_time = (time.perf_counter() - baseline_start) * 1000

        # Apply optimizations
        optimization_results = optimization_engine.apply_comprehensive_optimization()

        # Measure optimized performance
        optimized_start = time.perf_counter()
        self._perform_standard_qt_operations(config.iterations)
        optimized_time = (time.perf_counter() - optimized_start) * 1000

        # Calculate improvement
        improvement_percent = ((baseline_time - optimized_time) / baseline_time) * 100 if baseline_time > 0 else 0

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="optimization_effectiveness",
            configuration=config,
            execution_time_ms=optimized_time,
            execution_times=[baseline_time, optimized_time]
        )

    def _benchmark_full_system_stress(self, config: BenchmarkConfiguration) -> BenchmarkMetrics:
        """Benchmark full system under stress conditions."""
        monitoring_system = initialize_integrated_monitoring()
        execution_times = []
        errors = []
        peak_memory = 0.0

        try:
            for iteration in range(config.iterations):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()

                try:
                    # Stress test operations
                    self._perform_stress_operations(
                        config.data_sizes[iteration % len(config.data_sizes)]
                    )

                    execution_time = (time.perf_counter() - start_time) * 1000
                    execution_times.append(execution_time)

                    current_memory = self._get_memory_usage()
                    peak_memory = max(peak_memory, current_memory)

                except Exception as e:
                    errors.append(str(e))

        finally:
            shutdown_integrated_monitoring()

        # Calculate metrics
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        success_rate = (config.iterations - len(errors)) / config.iterations if config.iterations > 0 else 0

        return BenchmarkMetrics(
            timestamp=time.perf_counter(),
            benchmark_name="full_system_stress_test",
            configuration=config,
            execution_time_ms=avg_execution_time,
            peak_memory_mb=peak_memory,
            success_rate=success_rate,
            error_rate=len(errors) / config.iterations if config.iterations > 0 else 0,
            execution_times=execution_times,
            errors=errors
        )

    def _perform_standard_qt_operations(self, iterations: int):
        """Perform standard Qt operations for benchmarking."""
        for i in range(iterations):
            widget = QWidget()
            widget.setObjectName(f"standard_widget_{i}")

            timer = QTimer(widget)
            timer.timeout.connect(lambda: None)
            timer.start(100)
            timer.stop()

            widget.deleteLater()

            if i % 10 == 0:
                QCoreApplication.processEvents()

    def _perform_stress_operations(self, operation_count: int):
        """Perform stress operations for benchmarking."""
        objects = []

        # Create many objects
        for i in range(operation_count):
            widget = QWidget()
            timer = QTimer(widget)
            timer.timeout.connect(lambda: None)
            timer.start(50)
            objects.append((widget, timer))

        # Process events
        for _ in range(10):
            QCoreApplication.processEvents()
            time.sleep(0.001)

        # Cleanup
        for widget, timer in objects:
            timer.stop()
            widget.deleteLater()

        QCoreApplication.processEvents()

    def _generate_test_qt_messages(self, count: int) -> List[Dict[str, Any]]:
        """Generate test Qt messages for benchmarking."""
        from PySide6.QtCore import QtMsgType

        message_templates = [
            "QStyleHints::colorSchemeChanged() connection warning #{i}",
            "QObject::connect: Unique connection requires a pointer to member function #{i}",
            "qt.svg renderer warning about invalid element #{i}",
            "QTimer::start: Timers can only be used with threads started with QThread #{i}",
            "QWidget: Cannot create a QWidget when no GUI thread #{i}"
        ]

        messages = []
        for i in range(count):
            template = message_templates[i % len(message_templates)]
            messages.append({
                "msg_type": QtMsgType.QtWarningMsg,
                "message": template.format(i=i)
            })

        return messages

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def _calculate_performance_delta(self, current: BenchmarkMetrics, baseline: BenchmarkMetrics) -> float:
        """Calculate performance delta from baseline."""
        if baseline.execution_time_ms == 0:
            return 0.0

        return ((current.execution_time_ms - baseline.execution_time_ms) / baseline.execution_time_ms) * 100

    def _generate_recommendations(self, metrics: BenchmarkMetrics) -> List[str]:
        """Generate optimization recommendations based on metrics."""
        recommendations = []

        # Performance recommendations
        if metrics.latency_p95_ms > 100:
            recommendations.append("High P95 latency detected - consider optimization")

        if metrics.throughput_ops_per_second < 1000:
            recommendations.append("Low throughput - review algorithm efficiency")

        # Memory recommendations
        if metrics.peak_memory_mb > 500:
            recommendations.append("High memory usage - implement memory optimization")

        # Qt-specific recommendations
        if metrics.qt_errors_count > 0:
            recommendations.append("Qt errors detected - review Qt compliance implementation")

        if metrics.error_rate > 0.05:  # More than 5% error rate
            recommendations.append("High error rate - improve error handling and stability")

        return recommendations

    def run_comprehensive_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run all benchmarks in the suite."""
        logger.info("Running comprehensive Qt compliance benchmark suite")

        results = {}
        benchmark_names = list(self._benchmarks.keys())

        for benchmark_name in benchmark_names:
            try:
                result = self.run_benchmark(benchmark_name)
                results[benchmark_name] = result

                # Brief pause between benchmarks
                time.sleep(1.0)

            except Exception as e:
                logger.error(f"Benchmark {benchmark_name} failed: {e}")

        logger.info(f"Benchmark suite completed: {len(results)} benchmarks executed")
        return results

    def establish_performance_baseline(self) -> Dict[str, BenchmarkMetrics]:
        """Establish performance baseline for all benchmarks."""
        logger.info("Establishing performance baseline")

        baseline_results = {}

        for benchmark_name in self._benchmarks.keys():
            try:
                result = self.run_benchmark(benchmark_name)
                if result.success:
                    baseline_results[benchmark_name] = result.metrics
                    self.baseline_results[benchmark_name] = result.metrics

            except Exception as e:
                logger.error(f"Failed to establish baseline for {benchmark_name}: {e}")

        logger.info(f"Baseline established for {len(baseline_results)} benchmarks")
        return baseline_results

    def generate_benchmark_report(self, results: Optional[Dict[str, BenchmarkResult]] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if results is None:
            results = self.benchmark_results

        # Calculate summary statistics
        successful_benchmarks = [r for r in results.values() if r.success]
        total_benchmarks = len(results)
        success_rate = len(successful_benchmarks) / total_benchmarks if total_benchmarks > 0 else 0

        # Performance metrics summary
        avg_execution_time = statistics.mean([r.metrics.execution_time_ms for r in successful_benchmarks]) if successful_benchmarks else 0
        avg_throughput = statistics.mean([r.metrics.throughput_ops_per_second for r in successful_benchmarks]) if successful_benchmarks else 0
        peak_memory = max([r.metrics.peak_memory_mb for r in successful_benchmarks], default=0)

        # Regression analysis
        regressions = []
        improvements = []
        for result in successful_benchmarks:
            if result.performance_delta is not None:
                if result.performance_delta > 5:  # 5% performance degradation
                    regressions.append({
                        "benchmark": result.benchmark_name,
                        "delta_percent": result.performance_delta
                    })
                elif result.performance_delta < -5:  # 5% performance improvement
                    improvements.append({
                        "benchmark": result.benchmark_name,
                        "delta_percent": abs(result.performance_delta)
                    })

        return {
            "summary": {
                "total_benchmarks": total_benchmarks,
                "successful_benchmarks": len(successful_benchmarks),
                "success_rate": success_rate,
                "avg_execution_time_ms": avg_execution_time,
                "avg_throughput_ops_per_second": avg_throughput,
                "peak_memory_mb": peak_memory
            },
            "performance_analysis": {
                "regressions": regressions,
                "improvements": improvements,
                "regression_count": len(regressions),
                "improvement_count": len(improvements)
            },
            "detailed_results": {name: result.to_dict() for name, result in results.items()},
            "recommendations": self._generate_suite_recommendations(successful_benchmarks)
        }

    def _generate_suite_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate recommendations for the entire suite."""
        recommendations = []

        # Collect all individual recommendations
        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)

        # Count recommendation frequency
        recommendation_counts = defaultdict(int)
        for rec in all_recommendations:
            recommendation_counts[rec] += 1

        # Prioritize frequent recommendations
        frequent_recommendations = [
            rec for rec, count in recommendation_counts.items()
            if count >= len(results) * 0.3  # Appears in 30% or more of benchmarks
        ]

        recommendations.extend(frequent_recommendations)

        # Add suite-level recommendations
        avg_performance = statistics.mean([r.metrics.execution_time_ms for r in results])
        if avg_performance > 50:  # More than 50ms average
            recommendations.append("Overall performance could be improved - consider system-wide optimization")

        return recommendations

    def save_benchmark_results(self, results: Dict[str, BenchmarkResult], output_path: Optional[Path] = None) -> str:
        """Save benchmark results to file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config.output_directory) if self.config.output_directory else Path.home() / ".xpcs_toolkit" / "benchmarks"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"qt_benchmark_results_{timestamp}.json"

        # Generate report
        report = self.generate_benchmark_report(results)

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Benchmark results saved to: {output_path}")
        return str(output_path)


def run_qt_compliance_benchmarks(config: Optional[BenchmarkConfiguration] = None) -> Dict[str, Any]:
    """Run complete Qt compliance benchmark suite."""
    suite = QtComplianceBenchmarkSuite(config)

    try:
        # Establish baseline if not exists
        if not suite.baseline_results:
            suite.establish_performance_baseline()

        # Run comprehensive benchmarks
        results = suite.run_comprehensive_benchmark_suite()

        # Generate and save report
        report = suite.generate_benchmark_report(results)
        report_path = suite.save_benchmark_results(results)
        report["report_path"] = report_path

        return report

    except Exception as e:
        logger.error(f"Benchmark suite execution failed: {e}")
        return {"error": str(e), "success": False}


if __name__ == "__main__":
    # Run benchmark suite
    config = BenchmarkConfiguration(
        benchmark_type=BenchmarkType.PERFORMANCE,
        duration_seconds=10.0,
        iterations=50
    )

    report = run_qt_compliance_benchmarks(config)

    print("Qt Compliance Benchmark Report")
    print("=" * 50)
    print(f"Success Rate: {report.get('summary', {}).get('success_rate', 0):.1%}")
    print(f"Average Execution Time: {report.get('summary', {}).get('avg_execution_time_ms', 0):.1f}ms")
    print(f"Average Throughput: {report.get('summary', {}).get('avg_throughput_ops_per_second', 0):.1f} ops/sec")
    print(f"Peak Memory: {report.get('summary', {}).get('peak_memory_mb', 0):.1f}MB")

    # Show regressions and improvements
    regressions = report.get('performance_analysis', {}).get('regressions', [])
    improvements = report.get('performance_analysis', {}).get('improvements', [])

    if regressions:
        print(f"\nPerformance Regressions ({len(regressions)}):")
        for reg in regressions:
            print(f"  ↓ {reg['benchmark']}: {reg['delta_percent']:.1f}% slower")

    if improvements:
        print(f"\nPerformance Improvements ({len(improvements)}):")
        for imp in improvements:
            print(f"  ↑ {imp['benchmark']}: {imp['delta_percent']:.1f}% faster")