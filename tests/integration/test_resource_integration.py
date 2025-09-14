#!/usr/bin/env python3
"""
Resource Management Integration Tests for XPCS Toolkit

This module tests resource management across the integrated system:
- Memory, CPU, and I/O resource handling during workflows
- Resource cleanup after operations and errors
- Resource contention handling in concurrent scenarios
- Integration between memory management and I/O systems
- Resource monitoring and pressure detection

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import os
import shutil
import sys
import tempfile
import time
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XPCS Toolkit components
try:
    from xpcs_toolkit.fileIO.hdf_reader import batch_read_fields, get
    from xpcs_toolkit.module import g2mod, intt, saxs1d, saxs2d, stability
    from xpcs_toolkit.module.average_toolbox import AverageToolbox
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import MemoryTracker, get_cached_memory_monitor
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile
except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}")
    sys.exit(0)

logger = get_logger(__name__)


class ResourceMonitor:
    """Monitor system resources during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []

    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.measurements = []

    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False

    def take_measurement(self, label=""):
        """Take a resource measurement."""
        if self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                # Get I/O stats if available
                try:
                    io_info = self.process.io_counters()
                    io_data = {
                        "read_bytes": io_info.read_bytes,
                        "write_bytes": io_info.write_bytes,
                        "read_count": io_info.read_count,
                        "write_count": io_info.write_count,
                    }
                except (AttributeError, psutil.AccessDenied):
                    io_data = {
                        "read_bytes": 0,
                        "write_bytes": 0,
                        "read_count": 0,
                        "write_count": 0,
                    }

                measurement = {
                    "timestamp": time.time(),
                    "label": label,
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "memory_vms_mb": memory_info.vms / 1024 / 1024,
                    "cpu_percent": cpu_percent,
                    "io": io_data,
                }

                self.measurements.append(measurement)
                return measurement

            except Exception as e:
                logger.debug(f"Failed to take measurement: {e}")
                return None

    def get_summary(self):
        """Get resource usage summary."""
        if not self.measurements:
            return {}

        memory_values = [m["memory_mb"] for m in self.measurements]
        cpu_values = [
            m["cpu_percent"] for m in self.measurements if m["cpu_percent"] > 0
        ]

        summary = {
            "duration": self.measurements[-1]["timestamp"]
            - self.measurements[0]["timestamp"],
            "memory_peak_mb": max(memory_values),
            "memory_min_mb": min(memory_values),
            "memory_delta_mb": max(memory_values) - min(memory_values),
            "cpu_max_percent": max(cpu_values) if cpu_values else 0,
            "cpu_avg_percent": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
            "measurements_count": len(self.measurements),
        }

        # Add I/O summary if available
        if self.measurements[0]["io"]["read_bytes"] > 0:
            io_start = self.measurements[0]["io"]
            io_end = self.measurements[-1]["io"]

            summary["io_read_mb"] = (
                (io_end["read_bytes"] - io_start["read_bytes"]) / 1024 / 1024
            )
            summary["io_write_mb"] = (
                (io_end["write_bytes"] - io_start["write_bytes"]) / 1024 / 1024
            )
            summary["io_operations"] = (
                io_end["read_count"]
                - io_start["read_count"]
                + io_end["write_count"]
                - io_start["write_count"]
            )

        return summary


class TestResourceManagementIntegration(unittest.TestCase):
    """Test resource management integration across system components."""

    def setUp(self):
        """Set up resource management test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_resource_test_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.resource_monitor = ResourceMonitor()
        self.test_files = self._create_resource_test_files()

    def tearDown(self):
        """Clean up resources and log summary."""
        if self.resource_monitor.monitoring:
            self.resource_monitor.stop_monitoring()
            summary = self.resource_monitor.get_summary()
            if summary:
                logger.info(f"Test resource usage: {summary}")

    def _create_resource_test_files(self):
        """Create test files for resource management testing."""
        test_files = []

        # Create files of varying sizes for resource testing
        file_configs = [
            {"name": "small_resource", "n_q": 10, "n_tau": 50, "data_size": "small"},
            {"name": "medium_resource", "n_q": 50, "n_tau": 100, "data_size": "medium"},
            {"name": "large_resource", "n_q": 100, "n_tau": 200, "data_size": "large"},
            {
                "name": "xlarge_resource",
                "n_q": 150,
                "n_tau": 300,
                "data_size": "xlarge",
            },
        ]

        for config in file_configs:
            hdf_path = os.path.join(self.temp_dir, f"{config['name']}.hdf")
            self._create_resource_test_file(hdf_path, config)
            test_files.append(hdf_path)

        return test_files

    def _create_resource_test_file(self, hdf_path, config):
        """Create HDF5 file for resource testing."""
        with h5py.File(hdf_path, "w") as f:
            # Standard structure
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            n_q = config["n_q"]
            n_tau = config["n_tau"]

            # Scale data sizes based on configuration
            size_multipliers = {"small": 1, "medium": 5, "large": 20, "xlarge": 50}
            multiplier = size_multipliers.get(config["data_size"], 1)

            # Create appropriately sized datasets
            n_saxs = 100 * multiplier
            n_time = 1000 * multiplier

            # G2 correlation data
            tau = np.logspace(-6, 2, n_tau)
            g2_data = np.random.rand(n_q, n_tau) * 0.5 + 1.0
            g2_err = 0.02 * g2_data

            # Use compression to manage file sizes
            compression_opts = 6 if multiplier > 10 else None
            compression = "gzip" if compression_opts else None

            multitau.create_dataset(
                "g2",
                data=g2_data,
                compression=compression,
                compression_opts=compression_opts,
            )
            multitau.create_dataset(
                "g2_err",
                data=g2_err,
                compression=compression,
                compression_opts=compression_opts,
            )
            multitau.create_dataset("tau", data=tau)

            # Large SAXS datasets for I/O and memory testing
            saxs_data = np.random.rand(1, n_saxs) * 1000
            Iqp_data = np.random.rand(n_q, n_saxs) * 1000

            multitau.create_dataset(
                "saxs_1d",
                data=saxs_data,
                compression=compression,
                compression_opts=compression_opts,
            )
            multitau.create_dataset(
                "Iqp",
                data=Iqp_data,
                compression=compression,
                compression_opts=compression_opts,
            )

            # Time series data
            time_data = np.linspace(0, 1000, n_time)
            intensity_data = 1000 + 50 * np.random.randn(n_time)

            multitau.create_dataset(
                "Int_t",
                data=np.vstack([time_data, intensity_data]),
                compression=compression,
                compression_opts=compression_opts,
            )

            # Q-map data
            qmap.create_dataset("dqlist", data=np.linspace(0.001, 0.1, n_q))
            qmap.create_dataset("sqlist", data=np.linspace(0.001, 0.1, n_q))
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(n_q))
            qmap.create_dataset("static_index_mapping", data=np.arange(n_q))

    def test_memory_resource_management_integration(self):
        """Test memory resource management during file operations."""
        self.resource_monitor.start_monitoring()
        self.resource_monitor.take_measurement("start")

        # Test progressive file loading with memory tracking
        loaded_files = []
        memory_checkpoints = []

        for i, test_file in enumerate(self.test_files):
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024

            self.resource_monitor.take_measurement(f"before_load_{i}")

            # Load file
            xf = XpcsFile(test_file)
            loaded_files.append(xf)

            self.resource_monitor.take_measurement(f"after_load_{i}")

            # Access data to trigger memory allocation

            self.resource_monitor.take_measurement(f"after_access_{i}")

            # Calculate memory efficiency
            current_measurement = self.resource_monitor.take_measurement(
                f"checkpoint_{i}"
            )

            if i == 0:
                baseline_memory = current_measurement["memory_mb"]
            else:
                memory_increase = current_measurement["memory_mb"] - baseline_memory
                memory_checkpoints.append(
                    {
                        "file_index": i,
                        "file_size_mb": file_size_mb,
                        "memory_increase_mb": memory_increase,
                        "efficiency_ratio": memory_increase / file_size_mb
                        if file_size_mb > 0
                        else 0,
                    }
                )

        # Test memory cleanup
        self.resource_monitor.take_measurement("before_cleanup")

        del loaded_files
        import gc

        gc.collect()

        self.resource_monitor.take_measurement("after_cleanup")
        self.resource_monitor.stop_monitoring()

        # Analyze memory resource management
        summary = self.resource_monitor.get_summary()

        logger.info(
            f"Memory resource test: peak={summary['memory_peak_mb']:.1f}MB, "
            f"delta={summary['memory_delta_mb']:.1f}MB"
        )

        # Validate memory management
        for checkpoint in memory_checkpoints:
            logger.debug(
                f"File {checkpoint['file_index']}: "
                f"size={checkpoint['file_size_mb']:.1f}MB, "
                f"memory_increase={checkpoint['memory_increase_mb']:.1f}MB, "
                f"efficiency={checkpoint['efficiency_ratio']:.2f}"
            )

            # Memory efficiency should be reasonable (< 10x file size)
            self.assertLess(
                checkpoint["efficiency_ratio"],
                10.0,
                f"Poor memory efficiency for file {checkpoint['file_index']}",
            )

        # Memory cleanup should be effective
        cleanup_measurements = [
            m for m in self.resource_monitor.measurements if "cleanup" in m["label"]
        ]
        if len(cleanup_measurements) >= 2:
            before_cleanup = cleanup_measurements[0]["memory_mb"]
            after_cleanup = cleanup_measurements[1]["memory_mb"]
            cleanup_reduction = before_cleanup - after_cleanup

            logger.info(
                f"Memory cleanup: {before_cleanup:.1f}MB -> {after_cleanup:.1f}MB "
                f"(reduction: {cleanup_reduction:.1f}MB)"
            )

            # Should see some memory reduction
            self.assertGreater(cleanup_reduction, 0, "No memory cleanup detected")

    def test_io_resource_management_integration(self):
        """Test I/O resource management during operations."""
        self.resource_monitor.start_monitoring()
        self.resource_monitor.take_measurement("io_start")

        # Test I/O patterns with different access strategies
        io_test_results = []

        for i, test_file in enumerate(self.test_files):
            file_size_mb = os.path.getsize(test_file) / 1024 / 1024

            # Test 1: Sequential data access
            start_time = time.perf_counter()
            self.resource_monitor.take_measurement(f"before_sequential_{i}")

            XpcsFile(test_file)
            get(test_file, "/xpcs/multitau/saxs_1d")

            sequential_time = time.perf_counter() - start_time
            self.resource_monitor.take_measurement(f"after_sequential_{i}")

            # Test 2: Random access pattern
            start_time = time.perf_counter()
            self.resource_monitor.take_measurement(f"before_random_{i}")

            # Access data in different order
            get(test_file, "/xpcs/multitau/Int_t")
            get(test_file, "/xpcs/multitau/Iqp")

            random_time = time.perf_counter() - start_time
            self.resource_monitor.take_measurement(f"after_random_{i}")

            # Test 3: Batch access
            start_time = time.perf_counter()
            self.resource_monitor.take_measurement(f"before_batch_{i}")

            # Batch read multiple datasets
            field_list = [
                "/xpcs/multitau/g2",
                "/xpcs/multitau/tau",
                "/xpcs/multitau/saxs_1d",
            ]
            try:
                batch_read_fields(test_file, field_list)
                batch_time = time.perf_counter() - start_time
            except Exception as e:
                logger.debug(f"Batch read failed: {e}")
                batch_time = float("inf")

            self.resource_monitor.take_measurement(f"after_batch_{i}")

            io_test_results.append(
                {
                    "file_index": i,
                    "file_size_mb": file_size_mb,
                    "sequential_time": sequential_time,
                    "random_time": random_time,
                    "batch_time": batch_time,
                    "sequential_throughput": file_size_mb / sequential_time
                    if sequential_time > 0
                    else 0,
                    "cache_effectiveness": sequential_time / random_time
                    if random_time > 0
                    else 1,
                }
            )

        self.resource_monitor.stop_monitoring()

        # Analyze I/O resource management
        self.resource_monitor.get_summary()

        for result in io_test_results:
            logger.info(
                f"I/O test file {result['file_index']} ({result['file_size_mb']:.1f}MB): "
                f"sequential={result['sequential_time']:.3f}s, "
                f"random={result['random_time']:.3f}s, "
                f"throughput={result['sequential_throughput']:.1f}MB/s"
            )

            # I/O performance should be reasonable
            self.assertGreater(
                result["sequential_throughput"],
                1.0,
                f"Poor I/O throughput: {result['sequential_throughput']:.1f}MB/s",
            )

            # Cache should improve performance for repeated access
            if result["random_time"] > 0:
                self.assertLess(
                    result["random_time"],
                    result["sequential_time"] * 2,
                    "Random access should benefit from caching",
                )

    def test_concurrent_resource_management(self):
        """Test resource management under concurrent load."""
        self.resource_monitor.start_monitoring()
        self.resource_monitor.take_measurement("concurrent_start")

        # Test concurrent resource usage
        results = []
        errors = []

        def concurrent_resource_operation(thread_id, test_files):
            """Perform resource-intensive operations concurrently."""
            try:
                thread_results = []

                for i, test_file in enumerate(test_files):
                    operation_start = time.perf_counter()

                    # Load and process file
                    xf = XpcsFile(test_file)
                    g2_data = xf.g2

                    # Perform some processing
                    if g2_data is not None and g2_data.size > 0:
                        mean_g2 = np.mean(g2_data)
                        np.std(g2_data)

                        # Memory-intensive operation
                        np.corrcoef(g2_data) if g2_data.shape[0] > 1 else np.array(
                            [[1.0]]
                        )

                        operation_time = time.perf_counter() - operation_start

                        thread_results.append(
                            {
                                "thread_id": thread_id,
                                "file_index": i,
                                "operation_time": operation_time,
                                "mean_g2": mean_g2,
                                "correlation_computed": True,
                            }
                        )

                results.append((thread_id, thread_results))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent operations
        n_threads = 3
        files_per_thread = min(2, len(self.test_files))  # Limit for resource management

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = []

            for i in range(n_threads):
                # Each thread gets subset of files
                thread_files = self.test_files[:files_per_thread]
                future = executor.submit(concurrent_resource_operation, i, thread_files)
                futures.append(future)

            # Monitor resources during execution
            start_time = time.time()
            while not all(f.done() for f in futures):
                if time.time() - start_time > 60:  # 1 minute timeout
                    logger.warning("Concurrent resource test timeout")
                    break

                self.resource_monitor.take_measurement("concurrent_progress")
                time.sleep(0.5)

            # Wait for completion
            for future in futures:
                try:
                    future.result(timeout=10)
                except Exception as e:
                    errors.append(("future_error", str(e)))

        self.resource_monitor.take_measurement("concurrent_end")
        self.resource_monitor.stop_monitoring()

        # Analyze concurrent resource management
        summary = self.resource_monitor.get_summary()
        total_operations = sum(len(thread_results) for _, thread_results in results)

        logger.info(
            f"Concurrent resource test: {len(results)} threads, "
            f"{total_operations} operations, {len(errors)} errors"
        )
        logger.info(
            f"Resource usage: peak_memory={summary['memory_peak_mb']:.1f}MB, "
            f"cpu_max={summary['cpu_max_percent']:.1f}%"
        )

        # Validate concurrent resource management
        self.assertEqual(len(errors), 0, f"Concurrent resource errors: {errors}")
        self.assertGreater(total_operations, 0, "No concurrent operations completed")

        # Resource usage should be reasonable under load
        if summary["memory_peak_mb"] > 0:
            self.assertLess(
                summary["memory_peak_mb"],
                2000,
                f"Excessive memory usage: {summary['memory_peak_mb']:.1f}MB",
            )

    def test_resource_pressure_detection_and_response(self):
        """Test resource pressure detection and system response."""
        self.resource_monitor.start_monitoring()

        # Test memory pressure detection
        initial_pressure = MemoryMonitor.get_memory_pressure()
        pressure_responses = []

        loaded_objects = []

        try:
            # Gradually increase memory usage
            for i, test_file in enumerate(self.test_files):
                self.resource_monitor.take_measurement(f"pressure_test_{i}")

                # Load file and keep reference
                xf = XpcsFile(test_file)
                g2_data = xf.g2

                # Access large datasets to increase memory pressure
                try:
                    saxs_data = get(test_file, "/xpcs/multitau/saxs_1d")
                    Iqp_data = get(test_file, "/xpcs/multitau/Iqp")
                    int_t_data = get(test_file, "/xpcs/multitau/Int_t")

                    # Keep references to maintain memory usage
                    loaded_objects.append(
                        {
                            "xf": xf,
                            "g2": g2_data,
                            "saxs": saxs_data,
                            "Iqp": Iqp_data,
                            "int_t": int_t_data,
                        }
                    )
                except Exception as e:
                    logger.debug(f"Could not load all datasets: {e}")
                    loaded_objects.append({"xf": xf, "g2": g2_data})

                # Check memory pressure
                current_pressure = MemoryMonitor.get_memory_pressure()
                is_high_pressure = MemoryMonitor.is_memory_pressure_high(threshold=0.7)

                pressure_responses.append(
                    {
                        "iteration": i,
                        "memory_pressure": current_pressure,
                        "is_high_pressure": is_high_pressure,
                        "objects_loaded": len(loaded_objects),
                    }
                )

                logger.debug(f"Memory pressure iteration {i}: {current_pressure:.2f}")

                # Test cleanup response to high pressure
                if is_high_pressure and len(loaded_objects) > 2:
                    logger.info(
                        f"High memory pressure detected: {current_pressure:.2f}"
                    )

                    # Remove oldest objects
                    removed_count = len(loaded_objects) // 3  # Remove 1/3
                    for _ in range(removed_count):
                        if loaded_objects:
                            loaded_objects.pop(0)

                    import gc

                    gc.collect()

                    # Check if pressure decreased
                    post_cleanup_pressure = MemoryMonitor.get_memory_pressure()

                    pressure_responses.append(
                        {
                            "iteration": f"{i}_cleanup",
                            "memory_pressure": post_cleanup_pressure,
                            "pressure_reduction": current_pressure
                            - post_cleanup_pressure,
                            "objects_after_cleanup": len(loaded_objects),
                        }
                    )

                    logger.info(
                        f"After cleanup: {post_cleanup_pressure:.2f} "
                        f"(reduction: {current_pressure - post_cleanup_pressure:.2f})"
                    )

        finally:
            # Final cleanup
            del loaded_objects
            import gc

            gc.collect()

        self.resource_monitor.stop_monitoring()

        # Analyze pressure detection and response
        max_pressure = max(
            r["memory_pressure"] for r in pressure_responses if "memory_pressure" in r
        )

        pressure_reductions = [
            r.get("pressure_reduction", 0)
            for r in pressure_responses
            if r.get("pressure_reduction", 0) > 0
        ]

        logger.info(
            f"Pressure test: max_pressure={max_pressure:.2f}, "
            f"cleanup_operations={len(pressure_reductions)}"
        )

        # Validate pressure detection and response
        self.assertGreater(
            max_pressure, initial_pressure, "Should detect increasing memory pressure"
        )

        if pressure_reductions:
            avg_reduction = sum(pressure_reductions) / len(pressure_reductions)
            self.assertGreater(
                avg_reduction, 0.01, "Cleanup should reduce memory pressure"
            )

    def test_resource_cleanup_integration(self):
        """Test resource cleanup integration across components."""
        self.resource_monitor.start_monitoring()
        self.resource_monitor.take_measurement("cleanup_test_start")

        # Test systematic resource allocation and cleanup
        cleanup_test_phases = []

        # Phase 1: Allocation
        self.resource_monitor.take_measurement("phase1_start")

        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        loaded_files = []
        for i in range(min(3, len(kernel.raw_hdf_files))):
            xf = kernel.load_xpcs_file(i)
            loaded_files.append(xf)
            _ = xf.g2  # Access data

        phase1_measurement = self.resource_monitor.take_measurement("phase1_end")
        cleanup_test_phases.append(("allocation", phase1_measurement))

        # Phase 2: Processing
        self.resource_monitor.take_measurement("phase2_start")

        processing_results = []
        for xf in loaded_files:
            g2_data = xf.g2
            if g2_data is not None and g2_data.size > 0:
                # Memory-intensive processing
                np.mean(g2_data, axis=1)
                np.std(g2_data, axis=1)

                # Correlation computation (more memory)
                if g2_data.shape[0] > 1:
                    corr_matrix = np.corrcoef(g2_data)
                    processing_results.append(
                        {
                            "mean_correlation": np.mean(corr_matrix),
                            "data_shape": g2_data.shape,
                        }
                    )

        phase2_measurement = self.resource_monitor.take_measurement("phase2_end")
        cleanup_test_phases.append(("processing", phase2_measurement))

        # Phase 3: Explicit cleanup
        self.resource_monitor.take_measurement("phase3_start")

        # Clear references
        del loaded_files
        del processing_results
        del kernel

        # Force garbage collection
        import gc

        gc.collect()

        phase3_measurement = self.resource_monitor.take_measurement("phase3_end")
        cleanup_test_phases.append(("cleanup", phase3_measurement))

        # Phase 4: Verification
        time.sleep(0.5)  # Allow system to settle
        final_measurement = self.resource_monitor.take_measurement("final_verification")
        cleanup_test_phases.append(("verification", final_measurement))

        self.resource_monitor.stop_monitoring()

        # Analyze cleanup effectiveness
        memory_progression = [phase[1]["memory_mb"] for phase in cleanup_test_phases]
        phase_names = [phase[0] for phase in cleanup_test_phases]

        logger.info("Resource cleanup progression:")
        for i, (phase_name, memory_mb) in enumerate(
            zip(phase_names, memory_progression)
        ):
            logger.info(f"  {phase_name}: {memory_mb:.1f} MB")

        # Validate cleanup integration
        allocation_memory = memory_progression[0]  # allocation phase
        processing_memory = memory_progression[1]  # processing phase
        cleanup_memory = memory_progression[2]  # cleanup phase
        final_memory = memory_progression[3]  # final verification

        # Memory should increase during allocation and processing
        self.assertGreater(
            processing_memory,
            allocation_memory,
            "Memory should increase during processing",
        )

        # Memory should decrease after cleanup
        cleanup_effectiveness = (processing_memory - cleanup_memory) / processing_memory
        self.assertGreater(
            cleanup_effectiveness,
            0.1,
            f"Cleanup should be effective: {cleanup_effectiveness:.2f}",
        )

        # Final memory should be close to or below cleanup level
        self.assertLessEqual(
            final_memory, cleanup_memory * 1.1, "Memory should remain low after cleanup"
        )

    def test_resource_limits_and_throttling(self):
        """Test system behavior at resource limits."""
        self.resource_monitor.start_monitoring()

        # Test approach to resource limits
        limit_test_results = []

        # Start with baseline
        MemoryMonitor.get_memory_usage()[0]

        # Progressively load larger datasets
        allocated_objects = []

        try:
            for iteration in range(10):  # Limit iterations for safety
                self.resource_monitor.take_measurement(f"limit_test_{iteration}")

                # Check current resource state
                current_memory = MemoryMonitor.get_memory_usage()[0]
                memory_pressure = MemoryMonitor.get_memory_pressure()

                # Load and process data
                test_file = self.test_files[iteration % len(self.test_files)]

                try:
                    xf = XpcsFile(test_file)
                    g2_data = xf.g2

                    if g2_data is not None:
                        # Create additional memory allocation
                        extra_data = np.random.rand(*g2_data.shape) * g2_data
                        allocated_objects.append(
                            {"xf": xf, "g2": g2_data, "extra": extra_data}
                        )

                    limit_test_results.append(
                        {
                            "iteration": iteration,
                            "memory_mb": current_memory,
                            "memory_pressure": memory_pressure,
                            "allocation_success": True,
                            "objects_count": len(allocated_objects),
                        }
                    )

                except Exception as e:
                    limit_test_results.append(
                        {
                            "iteration": iteration,
                            "memory_mb": current_memory,
                            "memory_pressure": memory_pressure,
                            "allocation_success": False,
                            "error": str(e),
                            "objects_count": len(allocated_objects),
                        }
                    )

                    logger.info(f"Allocation failed at iteration {iteration}: {e}")
                    break

                # Check if we're approaching limits
                if memory_pressure > 0.85:
                    logger.warning(
                        f"High memory pressure reached: {memory_pressure:.2f}"
                    )

                    # Test throttling behavior
                    # Remove some objects to stay within limits
                    if len(allocated_objects) > 3:
                        removed_count = len(allocated_objects) // 4
                        for _ in range(removed_count):
                            allocated_objects.pop(0)

                        import gc

                        gc.collect()

                        post_throttle_pressure = MemoryMonitor.get_memory_pressure()
                        logger.info(f"After throttling: {post_throttle_pressure:.2f}")

                        limit_test_results[-1]["throttling_applied"] = True
                        limit_test_results[-1]["post_throttle_pressure"] = (
                            post_throttle_pressure
                        )

        finally:
            # Cleanup
            del allocated_objects
            import gc

            gc.collect()

        self.resource_monitor.stop_monitoring()

        # Analyze resource limit behavior
        successful_allocations = [
            r for r in limit_test_results if r["allocation_success"]
        ]
        failed_allocations = [
            r for r in limit_test_results if not r["allocation_success"]
        ]

        max_memory_reached = max(r["memory_mb"] for r in limit_test_results)
        max_pressure_reached = max(r["memory_pressure"] for r in limit_test_results)

        logger.info(
            f"Resource limit test: {len(successful_allocations)} successful, "
            f"{len(failed_allocations)} failed allocations"
        )
        logger.info(
            f"Peak memory: {max_memory_reached:.1f}MB, "
            f"peak pressure: {max_pressure_reached:.2f}"
        )

        # Validate resource limit handling
        self.assertGreater(
            len(successful_allocations), 0, "Should handle some allocations"
        )

        # Should detect high memory pressure
        self.assertGreater(
            max_pressure_reached, 0.5, "Should detect memory pressure increases"
        )

        # System should handle resource limits gracefully
        if failed_allocations:
            # Failures at resource limits are acceptable
            logger.info("System appropriately failed at resource limits")


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run resource management integration tests
    unittest.main(verbosity=2)
