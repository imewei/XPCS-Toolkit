"""
Performance benchmarks and validation tests for threading optimizations.

This module provides comprehensive benchmarks to validate the effectiveness
of Qt signal handling and thread pool optimizations in the XPCS Toolkit.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List

from PySide6.QtCore import QCoreApplication, QObject, Signal

from ..utils.logging_config import get_logger
from .enhanced_thread_pool import initialize_enhanced_threading
from .optimized_workers import OptimizedBaseWorker, submit_optimized_worker
from .performance_monitor import initialize_performance_monitoring

# Import our optimization systems
from .signal_optimization import SignalPriority, initialize_signal_optimization

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark test."""

    test_name: str
    baseline_time: float
    optimized_time: float
    baseline_memory: float = 0.0
    optimized_memory: float = 0.0
    baseline_signals: int = 0
    optimized_signals: int = 0
    improvement_percent: float = field(init=False)

    def __post_init__(self):
        if self.baseline_time > 0:
            self.improvement_percent = (
                (self.baseline_time - self.optimized_time) / self.baseline_time
            ) * 100
        else:
            self.improvement_percent = 0.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    suite_name: str
    results: List[BenchmarkResult] = field(default_factory=list)
    total_improvement: float = field(init=False, default=0.0)

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        self._calculate_total_improvement()

    def _calculate_total_improvement(self):
        """Calculate overall improvement across all tests."""
        if not self.results:
            self.total_improvement = 0.0
            return

        # Weight improvements by baseline execution time
        weighted_improvements = []
        total_weight = 0

        for result in self.results:
            weight = result.baseline_time
            weighted_improvements.append(result.improvement_percent * weight)
            total_weight += weight

        if total_weight > 0:
            self.total_improvement = sum(weighted_improvements) / total_weight
        else:
            self.total_improvement = 0.0


class SignalBenchmarkHelper(QObject):
    """Helper class for signal emission benchmarks."""

    # Test signals
    progress_signal = Signal(str, int, int, str, float)
    status_signal = Signal(str, str, int)
    resource_signal = Signal(str, float, float)
    result_signal = Signal(str, object, bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.received_signals = []
        self.connect_signals()

    def connect_signals(self):
        """Connect signals to receivers."""
        self.progress_signal.connect(self.on_progress)
        self.status_signal.connect(self.on_status)
        self.resource_signal.connect(self.on_resource)
        self.result_signal.connect(self.on_result)

    def on_progress(
        self, worker_id: str, current: int, total: int, message: str, eta: float
    ):
        self.received_signals.append(("progress", time.perf_counter()))

    def on_status(self, worker_id: str, status: str, level: int):
        self.received_signals.append(("status", time.perf_counter()))

    def on_resource(self, worker_id: str, cpu: float, memory: float):
        self.received_signals.append(("resource", time.perf_counter()))

    def on_result(self, worker_id: str, result: Any, is_final: bool):
        self.received_signals.append(("result", time.perf_counter()))

    def reset_counters(self):
        """Reset signal counters."""
        self.received_signals.clear()


class BenchmarkWorker(OptimizedBaseWorker):
    """Worker for benchmarking purposes."""

    def __init__(
        self,
        work_duration: float = 0.1,
        signal_frequency: float = 0.01,
        worker_id: str = None,
    ):
        super().__init__(worker_id)
        self.work_duration = work_duration
        self.signal_frequency = signal_frequency

    def do_work(self) -> str:
        """Simulate work with frequent signal emissions."""
        start_time = time.perf_counter()
        end_time = start_time + self.work_duration

        progress_count = 0
        last_signal_time = start_time

        while time.perf_counter() < end_time:
            current_time = time.perf_counter()

            # Check cancellation
            self.check_cancelled()

            # Emit signals at specified frequency
            if current_time - last_signal_time >= self.signal_frequency:
                progress = int(((current_time - start_time) / self.work_duration) * 100)
                self.emit_progress(progress, 100, f"Working... {progress_count}")
                self.emit_status(f"Status update {progress_count}", 1)
                self.emit_resource_usage(
                    20.0 + (progress_count % 50), 100.0 + progress_count
                )

                progress_count += 1
                last_signal_time = current_time

            # Small delay to prevent CPU spinning
            time.sleep(0.001)

        return f"Work completed with {progress_count} signal emissions"


class ThreadingBenchmarks:
    """
    Comprehensive benchmarks for threading system performance.
    """

    def __init__(self):
        self.app = None
        self.signal_helper = None
        self.benchmark_results = BenchmarkSuite("Threading Performance Benchmarks")

        # Initialize optimization systems
        self.signal_optimizer = initialize_signal_optimization()
        self.thread_pool_manager = initialize_enhanced_threading()
        self.performance_monitor = initialize_performance_monitoring(
            monitoring_interval=1.0
        )

    def setup_qt_application(self):
        """Setup Qt application if needed."""
        if QCoreApplication.instance() is None:
            self.app = QCoreApplication([])

        self.signal_helper = SignalBenchmarkHelper()

    def benchmark_signal_emission_patterns(self) -> BenchmarkResult:
        """Benchmark signal emission with and without optimization."""
        logger.info("Running signal emission benchmark...")

        # Test parameters
        num_signals = 10000

        # Baseline: Direct signal emission
        self.signal_optimizer.set_optimization_enabled(False)
        baseline_start = time.perf_counter()

        for i in range(num_signals):
            worker_id = f"test_worker_{i % 10}"
            self.signal_helper.progress_signal.emit(
                worker_id, i, num_signals, f"Message {i}", 1.0
            )
            self.signal_helper.status_signal.emit(worker_id, f"Status {i}", 1)
            self.signal_helper.resource_signal.emit(
                worker_id, float(i % 100), float(i % 200)
            )
            self.signal_helper.result_signal.emit(
                worker_id, f"Result {i}", i == num_signals - 1
            )

        # Process any remaining events
        if self.app:
            self.app.processEvents()

        baseline_time = time.perf_counter() - baseline_start
        baseline_received = len(self.signal_helper.received_signals)

        # Reset for optimized test
        self.signal_helper.reset_counters()

        # Optimized: With signal batching and optimization
        self.signal_optimizer.set_optimization_enabled(True)
        optimized_start = time.perf_counter()

        for i in range(num_signals):
            worker_id = f"test_worker_{i % 10}"

            # Use optimized emission
            self.signal_optimizer.emit_optimized(
                self.signal_helper.progress_signal,
                "progress",
                (worker_id, i, num_signals, f"Message {i}", 1.0),
                SignalPriority.NORMAL,
            )
            self.signal_optimizer.emit_optimized(
                self.signal_helper.status_signal,
                "status",
                (worker_id, f"Status {i}", 1),
                SignalPriority.NORMAL,
            )
            self.signal_optimizer.emit_optimized(
                self.signal_helper.resource_signal,
                "resource_usage",
                (worker_id, float(i % 100), float(i % 200)),
                SignalPriority.LOW,
            )
            self.signal_optimizer.emit_optimized(
                self.signal_helper.result_signal,
                "partial_result",
                (worker_id, f"Result {i}", i == num_signals - 1),
                SignalPriority.NORMAL,
            )

        # Allow batched signals to be processed
        time.sleep(0.2)  # Wait for batching timers
        if self.app:
            self.app.processEvents()

        optimized_time = time.perf_counter() - optimized_start
        optimized_received = len(self.signal_helper.received_signals)

        result = BenchmarkResult(
            test_name="Signal Emission Patterns",
            baseline_time=baseline_time,
            optimized_time=optimized_time,
            baseline_signals=baseline_received,
            optimized_signals=optimized_received,
        )

        logger.info(f"Signal benchmark: {result.improvement_percent:.1f}% improvement")
        logger.info(
            f"Signals: baseline={baseline_received}, optimized={optimized_received}"
        )

        return result

    def benchmark_worker_execution(self) -> BenchmarkResult:
        """Benchmark worker execution performance."""
        logger.info("Running worker execution benchmark...")

        num_workers = 50
        work_duration = 0.1  # 100ms per worker

        # Baseline: Use regular thread pool
        import concurrent.futures

        baseline_start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(num_workers):
                worker = BenchmarkWorker(work_duration, 0.005, f"baseline_worker_{i}")
                # Disable optimizations for baseline
                worker.signals = None  # Disable signal emission for fair comparison
                future = executor.submit(worker.do_work)
                futures.append(future)

            # Wait for completion
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Baseline worker failed: {e}")

        baseline_time = time.perf_counter() - baseline_start

        # Optimized: Use smart thread pool with optimizations
        optimized_start = time.perf_counter()

        completed_workers = []
        for i in range(num_workers):
            worker = BenchmarkWorker(work_duration, 0.005, f"optimized_worker_{i}")

            # Connect completion tracking
            def on_completion(result, worker_id=f"optimized_worker_{i}"):
                completed_workers.append(worker_id)

            worker.signals.finished.connect(on_completion)

            # Submit to optimized thread pool
            submit_optimized_worker(worker, priority=3)

        # Wait for completion
        while len(completed_workers) < num_workers:
            time.sleep(0.01)
            if self.app:
                self.app.processEvents()

        optimized_time = time.perf_counter() - optimized_start

        result = BenchmarkResult(
            test_name="Worker Execution Performance",
            baseline_time=baseline_time,
            optimized_time=optimized_time,
        )

        logger.info(f"Worker benchmark: {result.improvement_percent:.1f}% improvement")

        return result

    def benchmark_thread_pool_scaling(self) -> BenchmarkResult:
        """Benchmark thread pool dynamic scaling."""
        logger.info("Running thread pool scaling benchmark...")

        # Test with varying workload
        light_load_tasks = 20
        heavy_load_tasks = 100
        task_duration = 0.05

        # Baseline: Fixed size thread pool
        baseline_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Light load
            futures = []
            for i in range(light_load_tasks):
                future = executor.submit(time.sleep, task_duration)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

            # Heavy load
            futures = []
            for i in range(heavy_load_tasks):
                future = executor.submit(time.sleep, task_duration)
                futures.append(future)

            for future in as_completed(futures):
                future.result()

        baseline_time = time.perf_counter() - baseline_start

        # Optimized: Smart thread pool with dynamic scaling
        optimized_start = time.perf_counter()

        # Get the smart thread pool
        smart_pool = self.thread_pool_manager.get_pool("default")

        # Light load phase
        completed_count = 0
        target_count = light_load_tasks

        def on_light_completion():
            nonlocal completed_count
            completed_count += 1

        for i in range(light_load_tasks):
            worker = BenchmarkWorker(task_duration, 0.1, f"light_worker_{i}")
            worker.signals.finished.connect(on_light_completion)
            smart_pool.start_with_priority(worker, priority=5)

        # Wait for light load completion
        while completed_count < target_count:
            time.sleep(0.01)
            if self.app:
                self.app.processEvents()

        # Heavy load phase
        completed_count = 0
        target_count = heavy_load_tasks

        def on_heavy_completion():
            nonlocal completed_count
            completed_count += 1

        for i in range(heavy_load_tasks):
            worker = BenchmarkWorker(task_duration, 0.1, f"heavy_worker_{i}")
            worker.signals.finished.connect(on_heavy_completion)
            smart_pool.start_with_priority(worker, priority=5)

        # Wait for heavy load completion
        while completed_count < target_count:
            time.sleep(0.01)
            if self.app:
                self.app.processEvents()

        optimized_time = time.perf_counter() - optimized_start

        result = BenchmarkResult(
            test_name="Thread Pool Scaling",
            baseline_time=baseline_time,
            optimized_time=optimized_time,
        )

        logger.info(
            f"Thread pool scaling: {result.improvement_percent:.1f}% improvement"
        )

        return result

    def benchmark_attribute_caching(self) -> BenchmarkResult:
        """Benchmark worker attribute caching performance."""
        logger.info("Running attribute caching benchmark...")

        num_accesses = 100000

        # Create test worker
        worker = BenchmarkWorker(0.1, 0.01, "cache_test_worker")

        # Baseline: Direct attribute access
        baseline_start = time.perf_counter()

        for i in range(num_accesses):
            # Simulate frequent attribute access patterns
            _ = worker._is_cancelled
            _ = worker.worker_id
            _ = time.perf_counter() - (worker._start_time or time.perf_counter())
            _ = worker.pool_id

        baseline_time = time.perf_counter() - baseline_start

        # Optimized: Cached attribute access
        worker._start_time = time.perf_counter()  # Set for execution time calculation
        optimized_start = time.perf_counter()

        for i in range(num_accesses):
            # Use cached attribute access
            _ = worker.is_cancelled  # Uses cache
            _ = worker.get_cached_attribute("worker_id", lambda: worker.worker_id)
            _ = worker.execution_time  # Uses cache
            _ = worker.get_cached_attribute("pool_id", lambda: worker.pool_id)

        optimized_time = time.perf_counter() - optimized_start

        # Get cache statistics
        cache_stats = worker.get_performance_metrics()

        result = BenchmarkResult(
            test_name="Attribute Caching",
            baseline_time=baseline_time,
            optimized_time=optimized_time,
        )

        logger.info(f"Attribute caching: {result.improvement_percent:.1f}% improvement")
        logger.info(f"Cache hit rate: {cache_stats.cache_hit_rate:.1f}%")

        return result

    def run_comprehensive_benchmarks(self) -> BenchmarkSuite:
        """Run all benchmarks and return comprehensive results."""
        logger.info("Starting comprehensive threading benchmarks...")

        self.setup_qt_application()

        # Run individual benchmarks
        benchmark_methods = [
            self.benchmark_signal_emission_patterns,
            self.benchmark_worker_execution,
            self.benchmark_thread_pool_scaling,
            self.benchmark_attribute_caching,
        ]

        for benchmark_method in benchmark_methods:
            try:
                result = benchmark_method()
                self.benchmark_results.add_result(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark_method.__name__} failed: {e}")
                # Add failed result
                failed_result = BenchmarkResult(
                    test_name=benchmark_method.__name__,
                    baseline_time=1.0,
                    optimized_time=1.0,  # No improvement for failed test
                )
                self.benchmark_results.add_result(failed_result)

        # Generate summary
        logger.info("Benchmark suite completed!")
        logger.info(
            f"Overall improvement: {self.benchmark_results.total_improvement:.1f}%"
        )

        return self.benchmark_results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""

        # Get optimization statistics
        signal_stats = self.signal_optimizer.get_comprehensive_statistics()
        thread_stats = self.thread_pool_manager.get_global_statistics()

        report = {
            "benchmark_summary": {
                "suite_name": self.benchmark_results.suite_name,
                "total_improvement_percent": self.benchmark_results.total_improvement,
                "num_tests": len(self.benchmark_results.results),
            },
            "individual_results": [
                {
                    "test_name": result.test_name,
                    "baseline_time": result.baseline_time,
                    "optimized_time": result.optimized_time,
                    "improvement_percent": result.improvement_percent,
                    "baseline_signals": result.baseline_signals,
                    "optimized_signals": result.optimized_signals,
                }
                for result in self.benchmark_results.results
            ],
            "optimization_effectiveness": {
                "signal_batching": signal_stats.get("signal_batching", {}),
                "connection_pooling": signal_stats.get("connection_pool", {}),
                "attribute_caching": signal_stats.get("attribute_cache", {}),
                "thread_pool_management": thread_stats,
            },
            "performance_analysis": {
                "most_improved_test": max(
                    self.benchmark_results.results, key=lambda r: r.improvement_percent
                ).test_name
                if self.benchmark_results.results
                else None,
                "average_improvement": statistics.mean(
                    [r.improvement_percent for r in self.benchmark_results.results]
                )
                if self.benchmark_results.results
                else 0.0,
                "std_deviation": statistics.stdev(
                    [r.improvement_percent for r in self.benchmark_results.results]
                )
                if len(self.benchmark_results.results) > 1
                else 0.0,
            },
        }

        return report


def run_optimization_benchmarks() -> Dict[str, Any]:
    """
    Run comprehensive optimization benchmarks and return results.

    Returns:
        Dictionary containing benchmark results and performance analysis
    """
    benchmarks = ThreadingBenchmarks()

    try:
        # Run benchmarks
        results = benchmarks.run_comprehensive_benchmarks()

        # Generate report
        report = benchmarks.generate_performance_report()

        # Log summary
        logger.info("=" * 60)
        logger.info("THREADING OPTIMIZATION BENCHMARK RESULTS")
        logger.info("=" * 60)
        logger.info(
            f"Overall Performance Improvement: {results.total_improvement:.1f}%"
        )
        logger.info(f"Tests Completed: {len(results.results)}")

        for result in results.results:
            logger.info(
                f"  {result.test_name}: {result.improvement_percent:.1f}% improvement"
            )

        logger.info("=" * 60)

        return report

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        return {"error": str(e), "benchmark_results": {}}


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Run benchmarks
    results = run_optimization_benchmarks()

    # Print results
    import json

    print(json.dumps(results, indent=2))
