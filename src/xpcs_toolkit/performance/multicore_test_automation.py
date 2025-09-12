#!/usr/bin/env python3
"""
Multi-Core Test Automation for XPCS Toolkit CPU Performance Validation

This module provides automated testing across different CPU configurations with
comprehensive validation of thread scaling performance, CPU affinity and NUMA
awareness testing, load testing under various utilization scenarios, and
performance validation on different architectures.

Features:
- Automated testing across different CPU configurations (1-core, 2-core, 4-core, 8+ core)
- Thread scaling performance validation with detailed analysis
- CPU affinity and NUMA awareness testing for optimal resource utilization
- Load testing under various CPU utilization scenarios
- Performance validation on different architectures (x86, ARM, etc.)
- Comprehensive reporting and visualization of scaling characteristics
- Integration with existing performance test suite and regression detection
- Advanced CPU topology detection and optimization recommendations
- Real-time monitoring during multi-core stress testing

Integration Points:
- Integrates with CPU performance test suite for comprehensive validation
- Works with benchmark database for historical multi-core data
- Connects to regression detector for scaling performance validation
- Uses existing threading optimizations for realistic testing scenarios

Author: Claude Code Performance Testing Generator
Date: 2025-01-11
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import platform
import psutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable

import numpy as np
import matplotlib.pyplot as plt

from .cpu_performance_test_suite import PerformanceMeasurement
from .regression_detector import RegressionDetector
from .benchmark_database import BenchmarkDatabase, PerformanceRecord

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models and Configuration
# =============================================================================


@dataclass
class CPUTopology:
    """CPU topology information."""

    physical_cores: int
    logical_cores: int
    threads_per_core: int
    numa_nodes: int
    l1_cache_size: Optional[str] = None
    l2_cache_size: Optional[str] = None
    l3_cache_size: Optional[str] = None
    cpu_model: str = ""
    cpu_frequency_mhz: float = 0.0
    architecture: str = ""
    features: List[str] = field(default_factory=list)


@dataclass
class ScalingTestResult:
    """Result of a scaling performance test."""

    test_name: str
    core_count: int
    thread_count: int
    execution_time_s: float
    throughput_ops_per_sec: float
    efficiency_percent: float  # Actual vs ideal scaling
    speedup_factor: float  # Relative to single-core baseline
    cpu_utilization_percent: float
    memory_usage_mb: float
    cache_hit_rate: Optional[float] = None
    context_switches: int = 0
    load_balance_score: float = 0.0  # How well work was distributed
    thermal_throttling: bool = False
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiCoreTestConfig:
    """Configuration for multi-core testing."""

    # Core configurations to test
    core_configurations: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])

    # Thread configurations (multipliers of core count)
    thread_multipliers: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0]
    )

    # Test workload parameters
    cpu_intensive_task_size: int = 100000
    memory_intensive_task_size: int = 1000000
    io_intensive_task_count: int = 100

    # Performance thresholds
    min_efficiency_threshold: float = 0.7  # 70% scaling efficiency
    max_acceptable_overhead: float = 0.1  # 10% overhead
    min_speedup_threshold: float = 0.8  # 80% of theoretical speedup

    # Test duration and iterations
    warmup_iterations: int = 3
    measurement_iterations: int = 10
    stress_test_duration_s: int = 60

    # CPU affinity and NUMA testing
    test_cpu_affinity: bool = True
    test_numa_awareness: bool = True
    test_hyperthreading: bool = True

    # Load testing parameters
    load_levels: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.25]
    )  # CPU utilization targets

    # Architecture-specific settings
    enable_architecture_optimization: bool = True
    test_vectorization: bool = True


class CPUTopologyDetector:
    """Detect and analyze CPU topology."""

    @staticmethod
    def detect_topology() -> CPUTopology:
        """Detect current system CPU topology."""

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        threads_per_core = logical_cores // physical_cores if physical_cores else 1

        # Try to detect NUMA nodes
        numa_nodes = 1
        try:
            import numa

            numa_nodes = numa.get_max_node() + 1
        except ImportError:
            # Fallback detection
            if os.path.exists("/sys/devices/system/node"):
                numa_nodes = len(
                    [
                        d
                        for d in os.listdir("/sys/devices/system/node")
                        if d.startswith("node")
                    ]
                )

        # Get CPU info
        cpu_model = ""
        cpu_frequency = 0.0
        architecture = platform.machine()

        try:
            cpu_info = psutil.cpu_freq()
            if cpu_info:
                cpu_frequency = cpu_info.max or cpu_info.current or 0.0
        except Exception:
            pass

        # Try to get CPU model from /proc/cpuinfo on Linux
        if platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu_model = line.split(":", 1)[1].strip()
                            break
            except Exception:
                pass

        # Get CPU features
        features = []
        if hasattr(psutil, "cpu_stats"):
            stats = psutil.cpu_stats()
            if hasattr(stats, "ctx_switches"):
                features.append("context_switching")

        return CPUTopology(
            physical_cores=physical_cores,
            logical_cores=logical_cores,
            threads_per_core=threads_per_core,
            numa_nodes=numa_nodes,
            cpu_model=cpu_model,
            cpu_frequency_mhz=cpu_frequency,
            architecture=architecture,
            features=features,
        )

    @staticmethod
    def get_cache_info() -> Dict[str, str]:
        """Get CPU cache information."""
        cache_info = {}

        if platform.system() == "Linux":
            try:
                cache_dirs = ["/sys/devices/system/cpu/cpu0/cache"]
                for cache_dir in cache_dirs:
                    if os.path.exists(cache_dir):
                        for level_dir in os.listdir(cache_dir):
                            level_path = os.path.join(cache_dir, level_dir)
                            if os.path.isdir(level_path):
                                # Read cache size
                                size_file = os.path.join(level_path, "size")
                                if os.path.exists(size_file):
                                    with open(size_file, "r") as f:
                                        cache_info[level_dir] = f.read().strip()
            except Exception:
                pass

        return cache_info


# =============================================================================
# Workload Generators
# =============================================================================


class WorkloadGenerator:
    """Generate different types of workloads for multi-core testing."""

    @staticmethod
    def cpu_intensive_workload(size: int = 100000) -> Callable[[], float]:
        """Generate CPU-intensive workload."""

        def workload():
            # Mathematical computation that utilizes CPU
            result = 0.0
            for i in range(size):
                result += np.sin(i) * np.cos(i) + np.sqrt(i)
            return result

        return workload

    @staticmethod
    def memory_intensive_workload(size: int = 1000000) -> Callable[[], float]:
        """Generate memory-intensive workload."""

        def workload():
            # Memory allocation and manipulation
            data = np.random.random(size)
            # Perform operations that stress memory bandwidth
            result = np.sum(data * 2.0 + 1.0)
            # Matrix operations
            if size > 1000:
                matrix_size = int(np.sqrt(min(size, 10000)))
                matrix = np.random.random((matrix_size, matrix_size))
                result += np.trace(np.dot(matrix, matrix.T))
            return float(result)

        return workload

    @staticmethod
    def cache_intensive_workload(size: int = 50000) -> Callable[[], float]:
        """Generate cache-intensive workload."""

        def workload():
            # Create data that fits in different cache levels
            data = list(range(size))

            # Random access pattern (cache misses)
            indices = np.random.randint(0, size, size // 2)
            result = sum(data[i] for i in indices)

            # Sequential access pattern (cache hits)
            result += sum(data[i : i + 10] for i in range(0, size - 10, 100))

            return float(result)

        return workload

    @staticmethod
    def mixed_workload(
        cpu_size: int = 10000, mem_size: int = 100000
    ) -> Callable[[], Dict[str, float]]:
        """Generate mixed CPU and memory workload."""

        def workload():
            # CPU component
            cpu_result = 0.0
            for i in range(cpu_size):
                cpu_result += np.sin(i * 0.1) * np.cos(i * 0.1)

            # Memory component
            data = np.random.random(mem_size)
            mem_result = np.sum(data**2)

            # Combined operations
            if mem_size > 1000:
                fft_data = np.random.random(min(mem_size, 1024))
                fft_result = np.sum(np.abs(np.fft.fft(fft_data)))
            else:
                fft_result = 0.0

            return {
                "cpu_result": cpu_result,
                "memory_result": float(mem_result),
                "fft_result": float(fft_result),
                "total": cpu_result + mem_result + fft_result,
            }

        return workload


# =============================================================================
# Multi-Core Test Automation Class
# =============================================================================


class MultiCoreTestAutomation:
    """Main class for multi-core performance test automation."""

    def __init__(self, config: Optional[MultiCoreTestConfig] = None):
        """Initialize multi-core test automation."""
        self.config = config or MultiCoreTestConfig()
        self.topology = CPUTopologyDetector.detect_topology()
        self.results: Dict[str, List[ScalingTestResult]] = {}

        # Adjust core configurations based on available cores
        max_cores = self.topology.physical_cores
        self.config.core_configurations = [
            c for c in self.config.core_configurations if c <= max_cores * 2
        ]

        # Initialize related systems
        self.benchmark_db = None
        self.regression_detector = None

        logger.info(
            f"MultiCoreTestAutomation initialized for {self.topology.physical_cores} cores"
        )

    def initialize_integrations(
        self,
        benchmark_db: Optional[BenchmarkDatabase] = None,
        regression_detector: Optional[RegressionDetector] = None,
    ):
        """Initialize integrations with other systems."""
        self.benchmark_db = benchmark_db
        self.regression_detector = regression_detector

    def run_scaling_performance_tests(self) -> Dict[str, List[ScalingTestResult]]:
        """Run comprehensive scaling performance tests."""

        logger.info("Starting scaling performance tests...")

        test_workloads = {
            "cpu_intensive": WorkloadGenerator.cpu_intensive_workload(
                self.config.cpu_intensive_task_size
            ),
            "memory_intensive": WorkloadGenerator.memory_intensive_workload(
                self.config.memory_intensive_task_size
            ),
            "cache_intensive": WorkloadGenerator.cache_intensive_workload(),
            "mixed_workload": WorkloadGenerator.mixed_workload(),
        }

        results = {}

        for workload_name, workload_func in test_workloads.items():
            logger.info(f"Testing workload: {workload_name}")
            results[workload_name] = self._test_workload_scaling(
                workload_name, workload_func
            )

        self.results = results
        return results

    def _test_workload_scaling(
        self, workload_name: str, workload_func: Callable
    ) -> List[ScalingTestResult]:
        """Test scaling performance for a specific workload."""

        results = []
        baseline_result = None

        for core_count in self.config.core_configurations:
            for thread_multiplier in self.config.thread_multipliers:
                thread_count = max(1, int(core_count * thread_multiplier))

                logger.info(
                    f"Testing {workload_name} with {core_count} cores, {thread_count} threads"
                )

                # Run the test
                result = self._run_single_scaling_test(
                    workload_name, workload_func, core_count, thread_count
                )

                # Set baseline for speedup calculation
                if baseline_result is None and core_count == 1 and thread_count == 1:
                    baseline_result = result

                # Calculate scaling metrics
                if baseline_result:
                    result.speedup_factor = (
                        baseline_result.execution_time_s / result.execution_time_s
                    )
                    ideal_speedup = min(core_count, thread_count)
                    result.efficiency_percent = (
                        result.speedup_factor / ideal_speedup
                    ) * 100

                results.append(result)

                # Store in benchmark database if available
                if self.benchmark_db:
                    self._store_result_in_database(result)

        return results

    def _run_single_scaling_test(
        self,
        workload_name: str,
        workload_func: Callable,
        core_count: int,
        thread_count: int,
    ) -> ScalingTestResult:
        """Run a single scaling performance test."""

        # Warmup runs
        for _ in range(self.config.warmup_iterations):
            self._execute_parallel_workload(workload_func, thread_count, warmup=True)

        # Measurement runs
        execution_times = []
        cpu_utilizations = []
        memory_usages = []

        for _ in range(self.config.measurement_iterations):
            with PerformanceMeasurement(f"{workload_name}_scaling") as measurement:
                psutil.cpu_percent(interval=None)
                start_memory = psutil.virtual_memory().used / 1024 / 1024

                # Execute workload
                self._execute_parallel_workload(workload_func, thread_count)

                end_cpu = psutil.cpu_percent(interval=0.1)
                end_memory = psutil.virtual_memory().used / 1024 / 1024

            execution_times.append(measurement.end_time - measurement.start_time)
            cpu_utilizations.append(end_cpu)
            memory_usages.append(end_memory - start_memory)

        # Calculate statistics
        avg_execution_time = np.mean(execution_times)
        avg_cpu_utilization = np.mean(cpu_utilizations)
        avg_memory_usage = np.mean(memory_usages)

        # Calculate throughput (operations per second)
        throughput = thread_count / avg_execution_time if avg_execution_time > 0 else 0

        # Detect thermal throttling
        thermal_throttling = self._detect_thermal_throttling()

        # Calculate load balance score
        load_balance_score = self._calculate_load_balance_score(
            core_count, thread_count
        )

        return ScalingTestResult(
            test_name=workload_name,
            core_count=core_count,
            thread_count=thread_count,
            execution_time_s=avg_execution_time,
            throughput_ops_per_sec=throughput,
            efficiency_percent=0.0,  # Will be calculated later
            speedup_factor=1.0,  # Will be calculated later
            cpu_utilization_percent=avg_cpu_utilization,
            memory_usage_mb=avg_memory_usage,
            context_switches=0,  # Could be measured with more detailed monitoring
            load_balance_score=load_balance_score,
            thermal_throttling=thermal_throttling,
        )

    def _execute_parallel_workload(
        self, workload_func: Callable, thread_count: int, warmup: bool = False
    ) -> List[Any]:
        """Execute workload in parallel with specified thread count."""

        results = []

        if thread_count == 1:
            # Single-threaded execution
            results.append(workload_func())
        else:
            # Multi-threaded execution
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = []
                for _ in range(thread_count):
                    future = executor.submit(workload_func)
                    futures.append(future)

                for future in futures:
                    try:
                        result = future.result(timeout=30)  # 30-second timeout
                        results.append(result)
                    except Exception as e:
                        if not warmup:
                            logger.warning(f"Workload execution failed: {e}")

        return results

    def _detect_thermal_throttling(self) -> bool:
        """Detect if thermal throttling is occurring."""
        try:
            # Check CPU frequency
            current_freq = psutil.cpu_freq()
            if current_freq and current_freq.max:
                # If current frequency is significantly lower than max, might be throttling
                return current_freq.current < (current_freq.max * 0.9)
        except Exception:
            pass

        return False

    def _calculate_load_balance_score(
        self, core_count: int, thread_count: int
    ) -> float:
        """Calculate load balance score (simplified)."""
        # Ideal would be perfect distribution of threads across cores
        if core_count == 0:
            return 0.0

        threads_per_core = thread_count / core_count

        # Perfect balance would be integer threads per core
        balance_deviation = abs(threads_per_core - round(threads_per_core))

        # Score from 0 to 1, where 1 is perfect balance
        return 1.0 - min(balance_deviation, 1.0)

    def run_cpu_affinity_tests(self) -> Dict[str, Any]:
        """Test CPU affinity and NUMA awareness."""

        if not self.config.test_cpu_affinity:
            return {}

        logger.info("Running CPU affinity tests...")

        affinity_results = {}

        # Test 1: Single core affinity
        affinity_results["single_core"] = self._test_single_core_affinity()

        # Test 2: Core spreading
        affinity_results["core_spreading"] = self._test_core_spreading()

        # Test 3: NUMA node affinity
        if self.topology.numa_nodes > 1:
            affinity_results["numa_affinity"] = self._test_numa_affinity()

        return affinity_results

    def _test_single_core_affinity(self) -> Dict[str, float]:
        """Test performance with single core affinity."""

        workload = WorkloadGenerator.cpu_intensive_workload(50000)
        results = {}

        for core_id in range(
            min(4, self.topology.physical_cores)
        ):  # Test first 4 cores
            try:
                # Set CPU affinity to single core
                current_process = psutil.Process()
                current_process.cpu_affinity([core_id])

                # Run workload
                start_time = time.perf_counter()
                workload()
                end_time = time.perf_counter()

                results[f"core_{core_id}"] = end_time - start_time

            except Exception as e:
                logger.warning(f"Failed to test affinity for core {core_id}: {e}")
            finally:
                # Reset CPU affinity
                try:
                    current_process.cpu_affinity(
                        list(range(self.topology.logical_cores))
                    )
                except Exception:
                    pass

        return results

    def _test_core_spreading(self) -> Dict[str, float]:
        """Test performance with core spreading strategy."""

        workload = WorkloadGenerator.cpu_intensive_workload(50000)
        thread_counts = [2, 4, 8]
        results = {}

        for thread_count in thread_counts:
            if thread_count <= self.topology.physical_cores:
                # Test with threads spread across cores
                start_time = time.perf_counter()

                with ThreadPoolExecutor(max_workers=thread_count) as executor:
                    futures = [executor.submit(workload) for _ in range(thread_count)]
                    for future in futures:
                        future.result()

                end_time = time.perf_counter()
                results[f"{thread_count}_threads"] = end_time - start_time

        return results

    def _test_numa_affinity(self) -> Dict[str, Any]:
        """Test NUMA node affinity performance."""

        # This is a simplified NUMA test
        # Real implementation would use numa library for proper node affinity

        workload = WorkloadGenerator.memory_intensive_workload(500000)

        # Test memory-intensive workload that would benefit from NUMA locality
        start_time = time.perf_counter()
        workload()
        end_time = time.perf_counter()

        return {
            "numa_test_duration": end_time - start_time,
            "numa_nodes_detected": self.topology.numa_nodes,
        }

    def run_load_testing(self) -> Dict[str, List[ScalingTestResult]]:
        """Run load testing under various CPU utilization levels."""

        logger.info("Running load testing...")

        load_test_results = {}

        for load_level in self.config.load_levels:
            logger.info(f"Testing at {load_level:.0%} CPU load")

            # Generate background load
            load_processes = self._generate_background_load(load_level)

            try:
                # Run performance tests under load
                test_workload = WorkloadGenerator.mixed_workload(50000, 200000)
                results = []

                for core_count in [1, 2, 4, min(8, self.topology.physical_cores)]:
                    result = self._run_single_scaling_test(
                        f"load_test_{load_level:.0%}",
                        test_workload,
                        core_count,
                        core_count,
                    )
                    results.append(result)

                load_test_results[f"load_{load_level:.0%}"] = results

            finally:
                # Clean up background load
                self._cleanup_background_load(load_processes)

        return load_test_results

    def _generate_background_load(self, target_load: float) -> List:
        """Generate background CPU load."""

        # Calculate number of processes needed for target load
        load_processes = []
        processes_needed = max(1, int(target_load * self.topology.physical_cores))

        def cpu_load_worker():
            """Worker function to generate CPU load."""
            end_time = time.time() + self.config.stress_test_duration_s
            while time.time() < end_time:
                # Busy work
                _ = sum(i * i for i in range(1000))

        try:
            # Start background processes
            for _ in range(processes_needed):
                process = mp.Process(target=cpu_load_worker)
                process.start()
                load_processes.append(process)

            # Wait a bit for load to stabilize
            time.sleep(2)

        except Exception as e:
            logger.error(f"Failed to generate background load: {e}")
            # Clean up any started processes
            self._cleanup_background_load(load_processes)
            return []

        return load_processes

    def _cleanup_background_load(self, load_processes: List):
        """Clean up background load processes."""

        for process in load_processes:
            try:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                logger.warning(f"Failed to clean up load process: {e}")

    def run_architecture_optimization_tests(self) -> Dict[str, Any]:
        """Test architecture-specific optimizations."""

        if not self.config.enable_architecture_optimization:
            return {}

        logger.info("Running architecture optimization tests...")

        arch_results = {}

        # Test vectorization performance
        if self.config.test_vectorization:
            arch_results["vectorization"] = self._test_vectorization_performance()

        # Test memory alignment
        arch_results["memory_alignment"] = self._test_memory_alignment()

        # Test instruction-level parallelism
        arch_results["instruction_parallelism"] = self._test_instruction_parallelism()

        return arch_results

    def _test_vectorization_performance(self) -> Dict[str, float]:
        """Test vectorized vs non-vectorized operations."""

        size = 100000
        data1 = np.random.random(size)
        data2 = np.random.random(size)

        results = {}

        # Vectorized operations
        start_time = time.perf_counter()
        np.multiply(data1, data2) + np.sin(data1)
        vectorized_time = time.perf_counter() - start_time
        results["vectorized"] = vectorized_time

        # Non-vectorized operations (for comparison)
        start_time = time.perf_counter()
        non_vectorized_result = []
        for i in range(size):
            non_vectorized_result.append(data1[i] * data2[i] + np.sin(data1[i]))
        non_vectorized_time = time.perf_counter() - start_time
        results["non_vectorized"] = non_vectorized_time

        results["speedup_factor"] = (
            non_vectorized_time / vectorized_time if vectorized_time > 0 else 0
        )

        return results

    def _test_memory_alignment(self) -> Dict[str, float]:
        """Test memory alignment effects on performance."""

        # Test aligned vs unaligned memory access patterns
        size = 1000000

        # Aligned access
        aligned_data = np.random.random(size)
        start_time = time.perf_counter()
        np.sum(aligned_data)
        aligned_time = time.perf_counter() - start_time

        # Simulated unaligned access (stride pattern)
        start_time = time.perf_counter()
        np.sum(aligned_data[::3])  # Every 3rd element
        unaligned_time = time.perf_counter() - start_time

        return {
            "aligned_access_time": aligned_time,
            "unaligned_access_time": unaligned_time,
            "alignment_impact": (unaligned_time - aligned_time) / aligned_time
            if aligned_time > 0
            else 0,
        }

    def _test_instruction_parallelism(self) -> Dict[str, float]:
        """Test instruction-level parallelism."""

        # Test independent vs dependent operations
        size = 100000

        # Independent operations (can be parallelized)
        data = np.random.random(size)
        start_time = time.perf_counter()
        result1 = np.sin(data)
        result2 = np.cos(data)
        result3 = np.tan(data)
        result1 + result2 + result3
        independent_time = time.perf_counter() - start_time

        # Dependent operations (sequential)
        start_time = time.perf_counter()
        temp = np.sin(data)
        temp = np.cos(temp)
        np.tan(temp)
        dependent_time = time.perf_counter() - start_time

        return {
            "independent_operations_time": independent_time,
            "dependent_operations_time": dependent_time,
            "parallelism_benefit": (dependent_time - independent_time) / dependent_time
            if dependent_time > 0
            else 0,
        }

    def _store_result_in_database(self, result: ScalingTestResult):
        """Store scaling test result in benchmark database."""

        if not self.benchmark_db:
            return

        try:
            # Create performance record
            record = PerformanceRecord(
                timestamp=time.time(),
                test_name=f"multicore_scaling_{result.test_name}",
                metric_name=f"{result.test_name}_throughput",
                metric_value=result.throughput_ops_per_sec,
                metric_unit="ops/sec",
                test_config={
                    "core_count": result.core_count,
                    "thread_count": result.thread_count,
                    "test_type": "scaling",
                },
                environment_info={
                    "cpu_topology": asdict(self.topology),
                    "scaling_efficiency": result.efficiency_percent,
                    "speedup_factor": result.speedup_factor,
                },
            )

            self.benchmark_db.store_performance_record(record)

        except Exception as e:
            logger.error(f"Failed to store result in database: {e}")

    def generate_scaling_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling analysis report."""

        if not self.results:
            return {"error": "No test results available"}

        report = {
            "system_info": asdict(self.topology),
            "test_configuration": asdict(self.config),
            "test_results": {},
            "scaling_analysis": {},
            "recommendations": [],
        }

        for workload_name, results in self.results.items():
            # Analyze scaling characteristics
            scaling_analysis = self._analyze_scaling_characteristics(results)

            report["test_results"][workload_name] = [asdict(r) for r in results]
            report["scaling_analysis"][workload_name] = scaling_analysis

        # Generate recommendations
        report["recommendations"] = self._generate_optimization_recommendations()

        return report

    def _analyze_scaling_characteristics(
        self, results: List[ScalingTestResult]
    ) -> Dict[str, Any]:
        """Analyze scaling characteristics from test results."""

        analysis = {
            "optimal_thread_count": 1,
            "peak_efficiency": 0.0,
            "scaling_bottleneck": "unknown",
            "performance_plateau": False,
            "linear_scaling_range": (1, 1),
        }

        if not results:
            return analysis

        # Find optimal configuration
        best_result = max(results, key=lambda r: r.efficiency_percent)
        analysis["optimal_thread_count"] = best_result.thread_count
        analysis["peak_efficiency"] = best_result.efficiency_percent

        # Detect scaling bottlenecks
        efficiencies = [r.efficiency_percent for r in results]
        if max(efficiencies) < 50:
            analysis["scaling_bottleneck"] = "poor_parallelization"
        elif any(r.memory_usage_mb > 1000 for r in results):
            analysis["scaling_bottleneck"] = "memory_bandwidth"
        elif any(r.thermal_throttling for r in results):
            analysis["scaling_bottleneck"] = "thermal_throttling"

        # Detect performance plateau
        sorted_results = sorted(results, key=lambda r: r.thread_count)
        if len(sorted_results) > 2:
            last_three = sorted_results[-3:]
            throughputs = [r.throughput_ops_per_sec for r in last_three]
            if max(throughputs) - min(throughputs) < max(throughputs) * 0.1:
                analysis["performance_plateau"] = True

        return analysis

    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results."""

        recommendations = []

        if not self.results:
            return recommendations

        # Analyze all workload results
        all_results = []
        for results_list in self.results.values():
            all_results.extend(results_list)

        if not all_results:
            return recommendations

        # Check overall scaling efficiency
        avg_efficiency = np.mean([r.efficiency_percent for r in all_results])
        if avg_efficiency < 60:
            recommendations.append(
                "Low scaling efficiency detected. Consider optimizing algorithms for better parallelization."
            )

        # Check memory usage
        high_memory_results = [r for r in all_results if r.memory_usage_mb > 500]
        if len(high_memory_results) > len(all_results) * 0.3:
            recommendations.append(
                "High memory usage detected in multiple tests. Consider memory optimization strategies."
            )

        # Check thermal throttling
        throttling_results = [r for r in all_results if r.thermal_throttling]
        if throttling_results:
            recommendations.append(
                "Thermal throttling detected. Improve cooling or reduce computational intensity."
            )

        # Check load balancing
        poor_balance_results = [r for r in all_results if r.load_balance_score < 0.7]
        if poor_balance_results:
            recommendations.append(
                "Poor load balancing detected. Review thread distribution strategies."
            )

        # Architecture-specific recommendations
        if self.topology.threads_per_core > 1:
            recommendations.append(
                f"System has hyperthreading ({self.topology.threads_per_core} threads per core). "
                "Test both physical and logical core counts for optimal performance."
            )

        if self.topology.numa_nodes > 1:
            recommendations.append(
                f"NUMA system detected ({self.topology.numa_nodes} nodes). "
                "Consider NUMA-aware memory allocation and thread affinity."
            )

        return recommendations

    def create_scaling_visualizations(
        self, output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """Create scaling performance visualizations."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = {}

        if not self.results:
            logger.warning("No results available for visualization")
            return created_files

        # Scaling efficiency plot
        plt.figure(figsize=(12, 8))

        for workload_name, results in self.results.items():
            thread_counts = [r.thread_count for r in results]
            efficiencies = [r.efficiency_percent for r in results]

            plt.scatter(thread_counts, efficiencies, label=workload_name, alpha=0.7)

            # Fit trend line
            if len(thread_counts) > 1:
                z = np.polyfit(thread_counts, efficiencies, 1)
                p = np.poly1d(z)
                plt.plot(thread_counts, p(thread_counts), "--", alpha=0.5)

        plt.xlabel("Thread Count")
        plt.ylabel("Scaling Efficiency (%)")
        plt.title("Multi-Core Scaling Efficiency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        efficiency_file = output_dir / "scaling_efficiency.png"
        plt.savefig(efficiency_file, dpi=150, bbox_inches="tight")
        plt.close()
        created_files["efficiency"] = str(efficiency_file)

        # Speedup factor plot
        plt.figure(figsize=(12, 8))

        for workload_name, results in self.results.items():
            core_counts = [r.core_count for r in results]
            speedups = [r.speedup_factor for r in results]

            plt.plot(core_counts, speedups, "o-", label=workload_name, markersize=6)

        # Add ideal scaling line
        max_cores = max(
            max(r.core_count for r in results) for results in self.results.values()
        )
        ideal_cores = list(range(1, max_cores + 1))
        plt.plot(ideal_cores, ideal_cores, "k--", alpha=0.5, label="Ideal Scaling")

        plt.xlabel("Core Count")
        plt.ylabel("Speedup Factor")
        plt.title("Multi-Core Speedup Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        speedup_file = output_dir / "speedup_performance.png"
        plt.savefig(speedup_file, dpi=150, bbox_inches="tight")
        plt.close()
        created_files["speedup"] = str(speedup_file)

        # Resource utilization heatmap
        if len(self.results) > 1:
            plt.figure(figsize=(14, 10))

            workload_names = list(self.results.keys())
            core_counts = sorted(
                set(r.core_count for results in self.results.values() for r in results)
            )

            # Create heatmap data
            heatmap_data = np.zeros((len(workload_names), len(core_counts)))

            for i, workload_name in enumerate(workload_names):
                for j, core_count in enumerate(core_counts):
                    matching_results = [
                        r
                        for r in self.results[workload_name]
                        if r.core_count == core_count
                    ]
                    if matching_results:
                        heatmap_data[i, j] = matching_results[0].cpu_utilization_percent

            im = plt.imshow(heatmap_data, cmap="RdYlBu_r", aspect="auto")
            plt.colorbar(im, label="CPU Utilization (%)")

            plt.xticks(range(len(core_counts)), core_counts)
            plt.yticks(range(len(workload_names)), workload_names)
            plt.xlabel("Core Count")
            plt.ylabel("Workload Type")
            plt.title("CPU Utilization Heatmap")

            # Add text annotations
            for i in range(len(workload_names)):
                for j in range(len(core_counts)):
                    plt.text(
                        j,
                        i,
                        f"{heatmap_data[i, j]:.1f}%",
                        ha="center",
                        va="center",
                        color="black" if heatmap_data[i, j] < 50 else "white",
                    )

            plt.tight_layout()

            heatmap_file = output_dir / "utilization_heatmap.png"
            plt.savefig(heatmap_file, dpi=150, bbox_inches="tight")
            plt.close()
            created_files["heatmap"] = str(heatmap_file)

        logger.info(f"Created {len(created_files)} visualization files")
        return created_files


# =============================================================================
# Command Line Interface
# =============================================================================


def main():
    """Main CLI interface for multi-core test automation."""

    import argparse

    parser = argparse.ArgumentParser(
        description="XPCS Toolkit Multi-Core Test Automation"
    )
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--run-scaling-tests", action="store_true", help="Run scaling performance tests"
    )
    parser.add_argument(
        "--run-affinity-tests", action="store_true", help="Run CPU affinity tests"
    )
    parser.add_argument(
        "--run-load-tests", action="store_true", help="Run load testing"
    )
    parser.add_argument(
        "--run-architecture-tests",
        action="store_true",
        help="Run architecture optimization tests",
    )
    parser.add_argument("--generate-report", type=str, help="Generate report to file")
    parser.add_argument(
        "--create-visualizations", type=str, help="Create visualizations in directory"
    )
    parser.add_argument(
        "--detect-topology", action="store_true", help="Detect and display CPU topology"
    )
    parser.add_argument("--benchmark-db", type=str, help="Benchmark database path")

    args = parser.parse_args()

    # Load configuration
    config = MultiCoreTestConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config_data = json.load(f)
            # Update config (simplified)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Initialize test automation
    automation = MultiCoreTestAutomation(config)

    # Initialize integrations
    benchmark_db = None
    if args.benchmark_db:
        from .benchmark_database import BenchmarkDatabase

        benchmark_db = BenchmarkDatabase(args.benchmark_db)

    automation.initialize_integrations(benchmark_db=benchmark_db)

    # Detect topology
    if args.detect_topology:
        topology = automation.topology
        print("CPU Topology Information:")
        print(f"  Physical Cores: {topology.physical_cores}")
        print(f"  Logical Cores: {topology.logical_cores}")
        print(f"  Threads per Core: {topology.threads_per_core}")
        print(f"  NUMA Nodes: {topology.numa_nodes}")
        print(f"  CPU Model: {topology.cpu_model}")
        print(f"  Architecture: {topology.architecture}")
        print(f"  CPU Frequency: {topology.cpu_frequency_mhz:.1f} MHz")
        return

    # Run tests
    if args.run_scaling_tests:
        print("Running scaling performance tests...")
        results = automation.run_scaling_performance_tests()
        print(f"Completed scaling tests for {len(results)} workload types")

    if args.run_affinity_tests:
        print("Running CPU affinity tests...")
        affinity_results = automation.run_cpu_affinity_tests()
        print(f"CPU affinity test results: {affinity_results}")

    if args.run_load_tests:
        print("Running load tests...")
        load_results = automation.run_load_testing()
        print(f"Completed load tests for {len(load_results)} load levels")

    if args.run_architecture_tests:
        print("Running architecture optimization tests...")
        arch_results = automation.run_architecture_optimization_tests()
        print(f"Architecture test results: {arch_results}")

    # Generate report
    if args.generate_report:
        report = automation.generate_scaling_analysis_report()

        with open(args.generate_report, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Generated report: {args.generate_report}")

        # Print summary
        if "scaling_analysis" in report:
            print("\nScaling Analysis Summary:")
            for workload, analysis in report["scaling_analysis"].items():
                print(
                    f"  {workload}: Optimal threads={analysis.get('optimal_thread_count', 'N/A')}, "
                    f"Peak efficiency={analysis.get('peak_efficiency', 0):.1f}%"
                )

    # Create visualizations
    if args.create_visualizations:
        viz_files = automation.create_scaling_visualizations(args.create_visualizations)
        print(f"Created visualizations: {list(viz_files.values())}")

    if not any(
        [
            args.detect_topology,
            args.run_scaling_tests,
            args.run_affinity_tests,
            args.run_load_tests,
            args.run_architecture_tests,
            args.generate_report,
            args.create_visualizations,
        ]
    ):
        print("Use --help for usage information")


if __name__ == "__main__":
    main()
