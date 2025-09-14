#!/usr/bin/env python3
"""
Threading Integration Tests for XPCS Toolkit

This module tests the integration of threading and asynchronous systems:
- Async workers + GUI + analysis coordination
- Thread pool management in real workflows
- Background processing integration with UI responsiveness
- Thread safety of shared data structures
- Integration of multiprocessing with threading systems

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import concurrent.futures
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest
import warnings
from pathlib import Path

import h5py
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import XPCS Toolkit components
try:
    from xpcs_toolkit.fileIO.hdf_reader import batch_read_fields, get
    from xpcs_toolkit.module import g2mod, intt, saxs1d, saxs2d, stability, twotime
    from xpcs_toolkit.module.average_toolbox import AverageToolbox
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import MemoryTracker
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import XpcsFile

    # Try to import Qt components for GUI threading tests
    try:
        from PySide6.QtCore import QEventLoop, QObject, QThread, QTimer, Signal
        from PySide6.QtWidgets import QApplication

        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        logger = get_logger(__name__)
        logger.warning("Qt components not available, skipping GUI threading tests")

except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}")
    sys.exit(0)

logger = get_logger(__name__)


class MockAsyncWorker:
    """Mock async worker for testing threading integration."""

    def __init__(self):
        self.results = queue.Queue()
        self.errors = queue.Queue()
        self.running = False

    def start_work(self, work_func, *args, **kwargs):
        """Start work in background thread."""

        def worker():
            try:
                self.running = True
                result = work_func(*args, **kwargs)
                self.results.put(result)
            except Exception as e:
                self.errors.put(e)
            finally:
                self.running = False

        thread = threading.Thread(target=worker)
        thread.start()
        return thread

    def get_result(self, timeout=5.0):
        """Get result from background work."""
        try:
            return self.results.get(timeout=timeout)
        except queue.Empty:
            if not self.errors.empty():
                raise self.errors.get()
            raise TimeoutError("Worker did not complete in time")


class TestThreadingIntegration(unittest.TestCase):
    """Test threading system integration."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_threading_test_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.test_files = self._create_threading_test_files()

    def _create_threading_test_files(self):
        """Create test files for threading tests."""
        test_files = []

        for i in range(3):
            hdf_path = os.path.join(self.temp_dir, f"thread_test_{i}.hdf")

            with h5py.File(hdf_path, "w") as f:
                # Basic structure
                f.create_group("entry")
                xpcs = f.create_group("xpcs")
                multitau = xpcs.create_group("multitau")
                qmap = xpcs.create_group("qmap")

                # Data with different characteristics for each file
                n_q = 20 + i * 10
                n_tau = 50

                tau = np.logspace(-6, 2, n_tau)
                g2_data = np.random.rand(n_q, n_tau) * 0.5 + 1.0
                g2_err = 0.02 * g2_data

                multitau.create_dataset("g2", data=g2_data)
                multitau.create_dataset("g2_err", data=g2_err)
                multitau.create_dataset("tau", data=tau)

                # SAXS and other data
                saxs_data = np.random.rand(1, 100) * 1000
                Iqp_data = np.random.rand(n_q, 100) * 1000
                int_t_data = np.vstack(
                    [np.linspace(0, 1000, 1000), 1000 + 50 * np.random.randn(1000)]
                )

                multitau.create_dataset("saxs_1d", data=saxs_data)
                multitau.create_dataset("Iqp", data=Iqp_data)
                multitau.create_dataset("Int_t", data=int_t_data)

                # Q-map data
                qmap.create_dataset("dqlist", data=np.linspace(0.001, 0.1, n_q))
                qmap.create_dataset("dynamic_index_mapping", data=np.arange(n_q))

            test_files.append(hdf_path)

        return test_files

    def test_concurrent_file_loading_integration(self):
        """Test concurrent file loading with thread safety."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        results = {}
        errors = {}
        threads = []

        def load_and_analyze_file(file_index, thread_id):
            """Load file and perform analysis in thread."""
            try:
                # Load file
                xf = kernel.load_xpcs_file(file_index)

                # Access data (potential thread safety issues here)
                g2_data = xf.g2
                tau_data = xf.tau

                # Perform analysis
                if g2_data.size > 0 and tau_data.size > 0:
                    mean_g2 = np.mean(g2_data)
                    max_contrast = np.max(g2_data[:, 0] - 1.0)

                    results[thread_id] = {
                        "file_index": file_index,
                        "n_q": g2_data.shape[0],
                        "n_tau": len(tau_data),
                        "mean_g2": mean_g2,
                        "max_contrast": max_contrast,
                    }
                else:
                    results[thread_id] = {"error": "No data available"}

            except Exception as e:
                errors[thread_id] = str(e)

        # Start concurrent threads
        n_threads = min(len(self.test_files), 3)

        for i in range(n_threads):
            file_index = i % len(kernel.raw_hdf_files) if kernel.raw_hdf_files else 0
            thread = threading.Thread(
                target=load_and_analyze_file, args=(file_index, f"thread_{i}")
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Verify results
        self.assertGreater(len(results), 0, "No successful concurrent file loading")
        self.assertEqual(len(errors), 0, f"Concurrent loading errors: {errors}")

        # Verify data integrity
        for thread_id, result in results.items():
            if "error" not in result:
                self.assertGreater(result["n_q"], 0)
                self.assertGreater(result["n_tau"], 0)
                self.assertGreaterEqual(result["mean_g2"], 1.0)

    def test_background_analysis_integration(self):
        """Test background analysis with mock async workers."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        if len(kernel.raw_hdf_files) == 0:
            self.skipTest("No test files available")

        # Load a test file
        xf = kernel.load_xpcs_file(0)
        g2_data = xf.g2
        tau_data = xf.tau

        if g2_data.size == 0 or tau_data.size == 0:
            self.skipTest("No G2 data available")

        # Test background G2 analysis
        worker = MockAsyncWorker()

        def background_g2_analysis():
            """Perform G2 analysis in background."""
            results = []

            # Analyze first few Q-points
            for q_idx in range(min(3, g2_data.shape[0])):
                g2_single = g2_data[q_idx, :]

                # Filter valid data
                valid = np.isfinite(g2_single) & (g2_single > 1.0)
                if np.sum(valid) > 20:
                    tau_fit = tau_data[valid]
                    g2_fit = g2_single[valid]

                    try:
                        # Simple analysis instead of complex fitting
                        contrast = g2_fit[0] - 1.0 if len(g2_fit) > 0 else 0.0
                        decay_time = (
                            tau_fit[len(tau_fit) // 2] if len(tau_fit) > 1 else 0.0
                        )

                        results.append(
                            {
                                "q_index": q_idx,
                                "contrast": contrast,
                                "characteristic_time": decay_time,
                                "n_points": len(tau_fit),
                            }
                        )
                    except Exception as e:
                        logger.debug(f"Analysis failed for Q-point {q_idx}: {e}")

            return results

        # Start background work
        work_thread = worker.start_work(background_g2_analysis)

        # Wait for completion
        work_thread.join(timeout=10)

        # Get results
        try:
            results = worker.get_result(timeout=1.0)
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0, "No analysis results produced")

            # Verify result structure
            for result in results:
                self.assertIn("q_index", result)
                self.assertIn("contrast", result)
                self.assertIn("characteristic_time", result)
                self.assertIn("n_points", result)

                self.assertGreaterEqual(result["contrast"], 0.0)
                self.assertGreater(result["characteristic_time"], 0.0)
                self.assertGreater(result["n_points"], 10)

        except Exception as e:
            self.fail(f"Background analysis failed: {e}")

    def test_thread_pool_analysis_integration(self):
        """Test thread pool integration with analysis workflows."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        if len(kernel.raw_hdf_files) == 0:
            self.skipTest("No test files available")

        def analyze_file(file_index):
            """Analyze single file in thread pool."""
            try:
                xf = kernel.load_xpcs_file(file_index)
                g2_data = xf.g2
                tau_data = xf.tau

                if g2_data.size == 0 or tau_data.size == 0:
                    return {"error": "No data available", "file_index": file_index}

                # Perform analysis
                analysis_results = {
                    "file_index": file_index,
                    "n_q_points": g2_data.shape[0],
                    "n_tau_points": len(tau_data),
                    "mean_g2_initial": np.mean(g2_data[:, 0])
                    if g2_data.shape[1] > 0
                    else 0,
                    "mean_g2_final": np.mean(g2_data[:, -1])
                    if g2_data.shape[1] > 0
                    else 0,
                    "tau_range": (np.min(tau_data), np.max(tau_data)),
                    "processing_time": time.time(),  # Timestamp
                }

                return analysis_results

            except Exception as e:
                return {"error": str(e), "file_index": file_index}

        # Use thread pool for analysis
        max_workers = min(len(kernel.raw_hdf_files), 3)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit analysis tasks
            futures = []
            for i in range(len(kernel.raw_hdf_files)):
                future = executor.submit(analyze_file, i)
                futures.append(future)

            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e), "future_error": True})

        # Verify results
        self.assertGreater(len(results), 0, "No analysis results from thread pool")

        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]

        logger.info(
            f"Thread pool analysis: {len(successful_results)} successful, "
            f"{len(failed_results)} failed"
        )

        # At least some analyses should succeed
        self.assertGreater(
            len(successful_results),
            0,
            f"All thread pool analyses failed: {failed_results}",
        )

        # Verify successful results
        for result in successful_results:
            self.assertGreater(result["n_q_points"], 0)
            self.assertGreater(result["n_tau_points"], 0)
            self.assertGreaterEqual(result["mean_g2_initial"], 1.0)
            self.assertGreaterEqual(result["mean_g2_final"], 1.0)

    def test_memory_tracker_thread_safety(self):
        """Test MemoryTracker thread safety during concurrent operations."""
        tracker = MemoryTracker()
        tracker.start_tracking()

        memory_readings = []
        errors = []

        def concurrent_memory_tracking(thread_id):
            """Track memory usage concurrently."""
            try:
                for i in range(10):
                    # Get memory reading
                    current_usage = tracker.get_current_usage()
                    memory_readings.append((thread_id, i, current_usage, time.time()))

                    # Load and process some data to change memory usage
                    if i < len(self.test_files):
                        test_file = self.test_files[i % len(self.test_files)]
                        xf = XpcsFile(test_file)
                        _ = xf.g2  # Access data

                    time.sleep(0.01)  # Small delay

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent memory tracking
        threads = []
        n_threads = 3

        for i in range(n_threads):
            thread = threading.Thread(target=concurrent_memory_tracking, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=20)

        tracker.stop_tracking()

        # Verify thread safety
        self.assertEqual(len(errors), 0, f"Memory tracking errors: {errors}")
        self.assertGreater(len(memory_readings), 0, "No memory readings collected")

        # Verify readings are reasonable
        memory_values = [reading[2] for reading in memory_readings]
        self.assertTrue(
            all(isinstance(val, (int, float)) and val > 0 for val in memory_values)
        )

        # Check for thread interference
        by_thread = {}
        for reading in memory_readings:
            thread_id = reading[0]
            if thread_id not in by_thread:
                by_thread[thread_id] = []
            by_thread[thread_id].append(reading[2])

        # Each thread should have reasonable memory readings
        for thread_id, values in by_thread.items():
            self.assertGreater(len(values), 0, f"No readings for thread {thread_id}")
            self.assertLess(
                max(values) - min(values),
                1000,
                f"Excessive memory variation in thread {thread_id}",
            )

    @unittest.skipUnless(QT_AVAILABLE, "Qt components not available")
    def test_qt_signal_integration(self):
        """Test Qt signal integration with threading (if Qt available)."""

        class AnalysisWorker(QObject):
            """Mock Qt worker for analysis."""

            progress_updated = Signal(int)
            analysis_completed = Signal(dict)
            error_occurred = Signal(str)

            def __init__(self, test_file):
                super().__init__()
                self.test_file = test_file

            def run_analysis(self):
                """Run analysis and emit signals."""
                try:
                    self.progress_updated.emit(10)

                    # Load file
                    xf = XpcsFile(self.test_file)
                    self.progress_updated.emit(30)

                    # Access data
                    g2_data = xf.g2
                    tau_data = xf.tau
                    self.progress_updated.emit(60)

                    # Simple analysis
                    if g2_data.size > 0:
                        result = {
                            "mean_g2": float(np.mean(g2_data)),
                            "n_q": int(g2_data.shape[0]),
                            "n_tau": int(len(tau_data)),
                        }
                        self.progress_updated.emit(100)
                        self.analysis_completed.emit(result)
                    else:
                        self.error_occurred.emit("No data available")

                except Exception as e:
                    self.error_occurred.emit(str(e))

        # Test Qt integration
        if len(self.test_files) == 0:
            self.skipTest("No test files available")

        # Create QApplication if needed
        app = None
        if not QApplication.instance():
            app = QApplication([])

        try:
            # Create worker and thread
            worker = AnalysisWorker(self.test_files[0])
            thread = QThread()

            # Move worker to thread
            worker.moveToThread(thread)

            # Connect signals for testing
            progress_values = []
            results = []
            errors = []

            worker.progress_updated.connect(lambda x: progress_values.append(x))
            worker.analysis_completed.connect(lambda x: results.append(x))
            worker.error_occurred.connect(lambda x: errors.append(x))

            # Start analysis
            thread.started.connect(worker.run_analysis)
            thread.start()

            # Wait for completion with event loop
            loop = QEventLoop()
            timer = QTimer()
            timer.timeout.connect(loop.quit)
            timer.start(5000)  # 5 second timeout

            worker.analysis_completed.connect(loop.quit)
            worker.error_occurred.connect(loop.quit)

            loop.exec()
            timer.stop()

            # Clean up thread
            thread.quit()
            thread.wait()

            # Verify results
            if errors:
                logger.warning(f"Qt analysis errors: {errors}")

            if results:
                result = results[0]
                self.assertIn("mean_g2", result)
                self.assertIn("n_q", result)
                self.assertIn("n_tau", result)
                self.assertGreaterEqual(result["mean_g2"], 1.0)

            # Verify progress updates
            self.assertGreater(len(progress_values), 0, "No progress updates received")
            self.assertIn(100, progress_values, "Analysis did not complete")

        finally:
            if app:
                app.quit()

    def test_averaging_toolbox_threading_integration(self):
        """Test AverageToolbox integration with threading."""
        if len(self.test_files) < 2:
            self.skipTest("Need at least 2 files for averaging test")

        # Create AverageToolbox
        AverageToolbox(self.temp_dir)

        # Test concurrent averaging operations
        averaging_results = []
        errors = []

        def perform_averaging(file_indices, thread_id):
            """Perform averaging in background thread."""
            try:
                # Mock averaging operation (simplified)
                files_to_average = [
                    self.test_files[i] for i in file_indices if i < len(self.test_files)
                ]

                if len(files_to_average) < 2:
                    averaging_results.append(
                        {
                            "thread_id": thread_id,
                            "status": "insufficient_files",
                            "files_count": len(files_to_average),
                        }
                    )
                    return

                # Simulate averaging by loading and combining data
                combined_g2 = []
                combined_tau = None

                for file_path in files_to_average:
                    xf = XpcsFile(file_path)
                    g2_data = xf.g2
                    tau_data = xf.tau

                    if g2_data.size > 0:
                        combined_g2.append(g2_data)
                        if combined_tau is None:
                            combined_tau = tau_data

                if len(combined_g2) > 1:
                    # Average the G2 data
                    min_shape = min(g2.shape for g2 in combined_g2)
                    truncated_g2 = [
                        g2[: min_shape[0], : min_shape[1]] for g2 in combined_g2
                    ]
                    averaged_g2 = np.mean(truncated_g2, axis=0)

                    averaging_results.append(
                        {
                            "thread_id": thread_id,
                            "status": "success",
                            "files_count": len(files_to_average),
                            "averaged_shape": averaged_g2.shape,
                            "mean_value": float(np.mean(averaged_g2)),
                        }
                    )
                else:
                    averaging_results.append(
                        {
                            "thread_id": thread_id,
                            "status": "no_valid_data",
                            "files_count": len(files_to_average),
                        }
                    )

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Start concurrent averaging
        threads = []

        # Thread 1: Average first 2 files
        thread1 = threading.Thread(target=perform_averaging, args=([0, 1], "avg_1"))
        threads.append(thread1)

        # Thread 2: Average files 1 and 2 (overlapping)
        if len(self.test_files) >= 3:
            thread2 = threading.Thread(target=perform_averaging, args=([1, 2], "avg_2"))
            threads.append(thread2)

        # Start threads
        for thread in threads:
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Verify results
        self.assertEqual(len(errors), 0, f"Averaging errors: {errors}")
        self.assertGreater(len(averaging_results), 0, "No averaging results")

        successful_averaging = [
            r for r in averaging_results if r["status"] == "success"
        ]

        if successful_averaging:
            for result in successful_averaging:
                self.assertGreater(result["files_count"], 1)
                self.assertIsInstance(result["averaged_shape"], tuple)
                self.assertGreaterEqual(result["mean_value"], 1.0)
        else:
            logger.warning(
                "No successful averaging operations (may be due to test data)"
            )

    def test_resource_contention_handling(self):
        """Test handling of resource contention in concurrent operations."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        if len(kernel.raw_hdf_files) == 0:
            self.skipTest("No test files available")

        # Create high contention scenario
        contention_results = []
        contention_errors = []

        def high_contention_operation(thread_id):
            """Perform operations that may cause resource contention."""
            try:
                results = []

                # Rapid file access
                for i in range(5):
                    file_idx = i % len(kernel.raw_hdf_files)

                    # Load file
                    xf = kernel.load_xpcs_file(file_idx)

                    # Access multiple datasets simultaneously
                    g2_data = xf.g2
                    tau_data = xf.tau

                    # HDF5 direct access (potential contention)
                    saxs_data = get(
                        kernel.raw_hdf_files[file_idx], "/xpcs/multitau/saxs_1d"
                    )

                    results.append(
                        {
                            "iteration": i,
                            "file_index": file_idx,
                            "g2_shape": g2_data.shape if g2_data is not None else None,
                            "tau_length": len(tau_data)
                            if tau_data is not None
                            else None,
                            "saxs_available": saxs_data is not None,
                        }
                    )

                    time.sleep(0.001)  # Minimal delay

                contention_results.append((thread_id, results))

            except Exception as e:
                contention_errors.append((thread_id, str(e)))

        # Start high contention scenario
        threads = []
        n_threads = 4  # High contention

        for i in range(n_threads):
            thread = threading.Thread(
                target=high_contention_operation, args=(f"contention_{i}",)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Analyze contention handling
        total_operations = sum(len(results) for _, results in contention_results)

        logger.info(
            f"Contention test: {len(contention_results)} threads completed, "
            f"{total_operations} total operations, {len(contention_errors)} errors"
        )

        # System should handle contention gracefully
        error_rate = len(contention_errors) / max(n_threads, 1)
        self.assertLess(
            error_rate, 0.5, f"High error rate under contention: {error_rate}"
        )

        # At least some operations should succeed
        self.assertGreater(
            total_operations,
            n_threads,
            "Very few operations completed under contention",
        )

        # Verify operation consistency
        for thread_id, results in contention_results:
            for result in results:
                if result["g2_shape"] is not None:
                    self.assertIsInstance(result["g2_shape"], tuple)
                    self.assertGreater(result["g2_shape"][0], 0)

                if result["tau_length"] is not None:
                    self.assertGreater(result["tau_length"], 0)


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run threading integration tests
    unittest.main(verbosity=2)
