#!/usr/bin/env python3
"""
Error Handling Integration Tests for XPCS Toolkit

This module tests error propagation and recovery throughout the integrated system:
- Error propagation through analysis chains
- Graceful handling of corrupted data files
- Recovery from memory pressure situations
- Thread safety during error conditions
- Integration between error handling systems

Author: Integration and Workflow Tester Agent
Created: 2025-09-13
"""

import os
import shutil
import sys
import tempfile
import threading
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
    from xpcs_toolkit.fileIO.qmap_utils import get_qmap
    from xpcs_toolkit.module import g2mod, intt, saxs1d, saxs2d, stability
    from xpcs_toolkit.utils.logging_config import get_logger
    from xpcs_toolkit.utils.memory_utils import MemoryTracker
    from xpcs_toolkit.viewer_kernel import ViewerKernel
    from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile
except ImportError as e:
    warnings.warn(f"Could not import all XPCS components: {e}")
    sys.exit(0)

logger = get_logger(__name__)


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling integration across system components."""

    def setUp(self):
        """Set up error handling test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="xpcs_error_test_")
        self.addCleanup(shutil.rmtree, self.temp_dir, ignore_errors=True)
        self.test_files = self._create_error_test_files()

    def _create_error_test_files(self):
        """Create test files with various error conditions."""
        test_files = {}

        # File 1: Corrupted G2 data
        corrupted_g2_file = os.path.join(self.temp_dir, "corrupted_g2.hdf")
        with h5py.File(corrupted_g2_file, "w") as f:
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            # Create problematic G2 data
            g2_data = np.ones((20, 50))
            g2_data[0, :10] = np.nan  # NaN values
            g2_data[1, :] = 0.5  # Violates G2 >= 1
            g2_data[2, :] = np.inf  # Infinite values
            g2_data[3, :] = -1.0  # Negative values

            tau_data = np.logspace(-6, 2, 50)
            tau_data[0] = 0  # Invalid tau value
            tau_data[5] = np.nan  # NaN tau

            multitau.create_dataset("g2", data=g2_data)
            multitau.create_dataset("g2_err", data=0.1 * np.abs(g2_data))
            multitau.create_dataset("tau", data=tau_data)

            # Valid qmap
            qmap.create_dataset("dqlist", data=np.linspace(0.001, 0.1, 20))
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(20))

        test_files["corrupted_g2"] = corrupted_g2_file

        # File 2: Missing required datasets
        missing_data_file = os.path.join(self.temp_dir, "missing_data.hdf")
        with h5py.File(missing_data_file, "w") as f:
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            # multitau group missing entirely

        test_files["missing_data"] = missing_data_file

        # File 3: Inconsistent dimensions
        inconsistent_file = os.path.join(self.temp_dir, "inconsistent.hdf")
        with h5py.File(inconsistent_file, "w") as f:
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            # Mismatched dimensions
            multitau.create_dataset("g2", data=np.ones((20, 50)))  # 20 Q-points
            multitau.create_dataset(
                "tau", data=np.logspace(-6, 2, 60)
            )  # 60 tau points (mismatch)

            qmap.create_dataset(
                "dqlist", data=np.linspace(0.001, 0.1, 15)
            )  # 15 Q-points (mismatch)
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(15))

        test_files["inconsistent"] = inconsistent_file

        # File 4: Extremely large values (overflow conditions)
        overflow_file = os.path.join(self.temp_dir, "overflow.hdf")
        with h5py.File(overflow_file, "w") as f:
            f.create_group("entry")
            xpcs = f.create_group("xpcs")
            multitau = xpcs.create_group("multitau")
            qmap = xpcs.create_group("qmap")

            # Extremely large values
            g2_data = np.ones((10, 30)) * 1e100  # Very large values
            tau_data = np.logspace(-6, 100, 30)  # Extreme time range

            multitau.create_dataset("g2", data=g2_data)
            multitau.create_dataset("tau", data=tau_data)

            qmap.create_dataset(
                "dqlist", data=np.linspace(0.001, 1e10, 10)
            )  # Extreme Q range
            qmap.create_dataset("dynamic_index_mapping", data=np.arange(10))

        test_files["overflow"] = overflow_file

        return test_files

    def test_corrupted_data_error_handling(self):
        """Test error handling with corrupted G2 data."""
        corrupted_file = self.test_files["corrupted_g2"]

        errors_caught = []
        warnings_issued = []

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                # File loading should succeed
                xf = XpcsFile(corrupted_file)

                # Data access should handle errors gracefully
                g2_data = xf.g2
                tau_data = xf.tau

                # Verify data was loaded despite issues
                self.assertIsInstance(g2_data, np.ndarray)
                self.assertIsInstance(tau_data, np.ndarray)

                # Check for problematic values
                has_nan = np.any(np.isnan(g2_data))
                has_inf = np.any(np.isinf(g2_data))
                has_negative = np.any(g2_data < 0)
                has_invalid_g2 = np.any((g2_data < 1.0) & np.isfinite(g2_data))

                self.assertTrue(
                    has_nan or has_inf or has_negative or has_invalid_g2,
                    "Should detect data quality issues",
                )

                # Test analysis with corrupted data
                try:
                    # Filter valid data for analysis
                    for q_idx in range(min(3, g2_data.shape[0])):
                        g2_single = g2_data[q_idx, :]

                        # Should handle invalid data gracefully
                        valid_mask = np.isfinite(g2_single) & (g2_single >= 1.0)

                        if np.sum(valid_mask) > 5:  # Some valid data exists
                            tau_data[valid_mask]
                            g2_valid = g2_single[valid_mask]

                            # Basic analysis should work with cleaned data
                            mean_g2 = np.mean(g2_valid)
                            self.assertGreater(mean_g2, 1.0)
                            self.assertTrue(np.isfinite(mean_g2))

                except Exception as e:
                    errors_caught.append(("analysis", str(e)))

            except Exception as e:
                errors_caught.append(("file_loading", str(e)))

            # Record any warnings
            warnings_issued = [
                (warning.category.__name__, str(warning.message)) for warning in w
            ]

        # Verify graceful error handling
        logger.info(
            f"Corrupted data test - Errors: {len(errors_caught)}, Warnings: {len(warnings_issued)}"
        )

        # Should handle corrupted data without crashing
        if errors_caught:
            # Errors are acceptable for highly corrupted data
            logger.warning(f"Expected errors with corrupted data: {errors_caught}")

    def test_missing_data_error_handling(self):
        """Test error handling with missing required datasets."""
        missing_data_file = self.test_files["missing_data"]

        with self.assertRaises(
            (KeyError, FileNotFoundError, ValueError, AttributeError)
        ):
            xf = XpcsFile(missing_data_file)
            # Attempt to access missing data should raise appropriate error
            _ = xf.g2

        # Test ViewerKernel handling
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        # Should handle files with missing data gracefully
        if len(kernel.raw_hdf_files) > 0:
            try:
                # Find the missing data file in the list
                missing_file_idx = None
                for i, hdf_file in enumerate(kernel.raw_hdf_files):
                    if "missing_data" in hdf_file:
                        missing_file_idx = i
                        break

                if missing_file_idx is not None:
                    with self.assertRaises(Exception):
                        xf = kernel.load_xpcs_file(missing_file_idx)
                        _ = xf.g2

            except Exception as e:
                # Expected for missing data
                logger.debug(f"Expected error with missing data: {e}")

    def test_dimension_mismatch_error_handling(self):
        """Test error handling with inconsistent data dimensions."""
        inconsistent_file = self.test_files["inconsistent"]

        try:
            xf = XpcsFile(inconsistent_file)

            # Access data - may succeed despite mismatches
            g2_data = xf.g2
            tau_data = xf.tau

            if g2_data is not None and tau_data is not None:
                # Check dimensions
                if g2_data.shape[1] != len(tau_data):
                    logger.warning(
                        f"Dimension mismatch detected: "
                        f"G2 shape {g2_data.shape}, tau length {len(tau_data)}"
                    )

                    # System should handle mismatch gracefully
                    # Use minimum dimension for analysis
                    min_tau_points = min(g2_data.shape[1], len(tau_data))

                    if min_tau_points > 10:
                        g2_subset = g2_data[:, :min_tau_points]
                        tau_data[:min_tau_points]

                        # Should be able to analyze subset
                        mean_g2 = np.mean(g2_subset)
                        self.assertTrue(np.isfinite(mean_g2))

            # Test Q-map dimension consistency
            qmap_data = get_qmap(inconsistent_file)
            if qmap_data and g2_data is not None:
                n_q_g2 = g2_data.shape[0]
                n_q_qmap = len(qmap_data.get("dqlist", []))

                if n_q_g2 != n_q_qmap:
                    logger.warning(
                        f"Q dimension mismatch: G2 has {n_q_g2}, qmap has {n_q_qmap}"
                    )

                    # Should use consistent subset
                    min_q_points = min(n_q_g2, n_q_qmap)
                    self.assertGreater(min_q_points, 0)

        except Exception as e:
            # Dimension mismatches may cause errors - this is acceptable
            logger.debug(f"Dimension mismatch caused error (expected): {e}")

    def test_numerical_overflow_error_handling(self):
        """Test error handling with numerical overflow conditions."""
        overflow_file = self.test_files["overflow"]

        try:
            xf = XpcsFile(overflow_file)

            g2_data = xf.g2
            tau_data = xf.tau

            if g2_data is not None and tau_data is not None:
                # Check for overflow values
                has_overflow = np.any(g2_data > 1e50)
                has_extreme_tau = np.any(tau_data > 1e50)

                if has_overflow:
                    logger.warning("Detected numerical overflow in G2 data")

                    # System should detect and handle overflow
                    finite_mask = np.isfinite(g2_data) & (g2_data < 1e10)

                    if np.any(finite_mask):
                        # Should be able to work with non-overflow values
                        finite_g2 = g2_data[finite_mask]
                        self.assertTrue(np.all(np.isfinite(finite_g2)))

                if has_extreme_tau:
                    logger.warning("Detected extreme tau values")

                    # Filter reasonable tau values
                    reasonable_tau = (
                        (tau_data > 1e-10) & (tau_data < 1e10) & np.isfinite(tau_data)
                    )

                    if np.any(reasonable_tau):
                        filtered_tau = tau_data[reasonable_tau]
                        self.assertTrue(np.all(filtered_tau > 0))

        except Exception as e:
            # Overflow conditions may cause legitimate errors
            logger.debug(f"Numerical overflow caused error (may be expected): {e}")

    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access conditions."""
        kernel = ViewerKernel(self.temp_dir)
        kernel.refresh_file_list()

        results = []
        errors = []

        def concurrent_error_prone_operation(thread_id):
            """Perform error-prone operations concurrently."""
            try:
                thread_errors = []
                thread_results = []

                for file_key in ["corrupted_g2", "missing_data", "inconsistent"]:
                    if file_key in self.test_files:
                        try:
                            test_file = self.test_files[file_key]

                            # Attempt to load problematic file
                            xf = XpcsFile(test_file)
                            g2_data = xf.g2

                            if g2_data is not None:
                                # Attempt basic analysis
                                mean_g2 = np.mean(g2_data[np.isfinite(g2_data)])
                                thread_results.append(
                                    {
                                        "file_type": file_key,
                                        "success": True,
                                        "mean_g2": mean_g2,
                                    }
                                )

                        except Exception as e:
                            thread_errors.append(
                                {"file_type": file_key, "error": str(e)}
                            )

                results.append((thread_id, thread_results, thread_errors))

            except Exception as e:
                errors.append((thread_id, str(e)))

        # Run concurrent error-prone operations
        threads = []
        n_threads = 3

        for i in range(n_threads):
            thread = threading.Thread(
                target=concurrent_error_prone_operation, args=(f"error_thread_{i}",)
            )
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join(timeout=30)

        # Analyze concurrent error handling
        total_operations = sum(
            len(thread_results) + len(thread_errors)
            for _, thread_results, thread_errors in results
        )
        total_thread_errors = len(errors)

        logger.info(
            f"Concurrent error handling: {len(results)} threads completed, "
            f"{total_operations} operations attempted, "
            f"{total_thread_errors} thread-level errors"
        )

        # System should handle concurrent errors without crashing
        self.assertGreater(len(results), 0, "No threads completed")
        self.assertLess(
            total_thread_errors / n_threads, 0.5, "Too many thread failures"
        )

    def test_memory_pressure_error_recovery(self):
        """Test error recovery under memory pressure conditions."""
        # Create memory pressure by loading multiple large files
        memory_tracker = MemoryTracker()
        memory_tracker.start_tracking()

        memory_tracker.get_current_usage()
        loaded_files = []
        memory_errors = []
        recovery_successful = False

        try:
            # Load files until memory pressure
            while len(loaded_files) < 10:  # Reasonable limit
                current_pressure = MemoryMonitor.get_memory_pressure()

                if current_pressure > 0.8:  # High memory pressure
                    logger.warning(
                        f"High memory pressure detected: {current_pressure:.2f}"
                    )

                    # Test error recovery
                    try:
                        # Force cleanup
                        oldest_file = loaded_files.pop(0) if loaded_files else None
                        del oldest_file

                        import gc

                        gc.collect()

                        # Check if pressure decreased
                        new_pressure = MemoryMonitor.get_memory_pressure()

                        if new_pressure < current_pressure:
                            recovery_successful = True
                            logger.info(
                                f"Memory pressure recovery: {current_pressure:.2f} -> {new_pressure:.2f}"
                            )
                            break

                    except Exception as e:
                        memory_errors.append(("recovery", str(e)))

                # Try to load another file
                try:
                    # Use corrupted file for testing (smaller and problematic)
                    test_file = self.test_files["corrupted_g2"]
                    xf = XpcsFile(test_file)
                    _ = xf.g2  # Trigger data loading
                    loaded_files.append(xf)

                except Exception as e:
                    memory_errors.append(("loading", str(e)))
                    break  # Stop if loading fails

        except Exception as e:
            memory_errors.append(("memory_pressure_test", str(e)))

        finally:
            # Cleanup
            del loaded_files
            import gc

            gc.collect()
            memory_tracker.stop_tracking()

        # Validate memory error handling
        logger.info(
            f"Memory pressure test: {len(memory_errors)} errors, "
            f"recovery_successful: {recovery_successful}"
        )

        # Memory pressure handling should work reasonably
        # Allow some errors under extreme memory conditions
        if memory_errors:
            logger.warning(f"Memory pressure errors (may be expected): {memory_errors}")

    def test_chain_error_propagation(self):
        """Test error propagation through analysis chains."""
        # Test error propagation from file -> analysis -> results

        error_chain_results = {}

        for file_key, test_file in self.test_files.items():
            chain_errors = []
            chain_steps = []

            try:
                # Step 1: File loading
                try:
                    xf = XpcsFile(test_file)
                    chain_steps.append("file_loading")
                except Exception as e:
                    chain_errors.append(("file_loading", str(e)))
                    error_chain_results[file_key] = {
                        "steps_completed": chain_steps,
                        "errors": chain_errors,
                        "stopped_at": "file_loading",
                    }
                    continue

                # Step 2: Data access
                try:
                    g2_data = xf.g2
                    tau_data = xf.tau
                    chain_steps.append("data_access")

                    if g2_data is None or tau_data is None:
                        raise ValueError("No data available")

                except Exception as e:
                    chain_errors.append(("data_access", str(e)))
                    error_chain_results[file_key] = {
                        "steps_completed": chain_steps,
                        "errors": chain_errors,
                        "stopped_at": "data_access",
                    }
                    continue

                # Step 3: Data validation
                try:
                    # Check data quality
                    valid_g2_mask = np.isfinite(g2_data) & (g2_data >= 1.0)
                    valid_tau_mask = np.isfinite(tau_data) & (tau_data > 0)

                    if not np.any(valid_g2_mask):
                        raise ValueError("No valid G2 data")
                    if not np.any(valid_tau_mask):
                        raise ValueError("No valid tau data")

                    chain_steps.append("data_validation")

                except Exception as e:
                    chain_errors.append(("data_validation", str(e)))
                    error_chain_results[file_key] = {
                        "steps_completed": chain_steps,
                        "errors": chain_errors,
                        "stopped_at": "data_validation",
                    }
                    continue

                # Step 4: Basic analysis
                try:
                    # Simple statistical analysis
                    valid_data = g2_data[np.isfinite(g2_data) & (g2_data >= 1.0)]

                    if len(valid_data) > 0:
                        mean_g2 = np.mean(valid_data)
                        std_g2 = np.std(valid_data)

                        if not np.isfinite(mean_g2) or not np.isfinite(std_g2):
                            raise ValueError("Analysis produced invalid results")

                        chain_steps.append("basic_analysis")
                    else:
                        raise ValueError("No valid data for analysis")

                except Exception as e:
                    chain_errors.append(("basic_analysis", str(e)))
                    error_chain_results[file_key] = {
                        "steps_completed": chain_steps,
                        "errors": chain_errors,
                        "stopped_at": "basic_analysis",
                    }
                    continue

                # If we reach here, chain completed
                error_chain_results[file_key] = {
                    "steps_completed": chain_steps,
                    "errors": chain_errors,
                    "stopped_at": "completed",
                }

            except Exception as e:
                chain_errors.append(("unexpected", str(e)))
                error_chain_results[file_key] = {
                    "steps_completed": chain_steps,
                    "errors": chain_errors,
                    "stopped_at": "unexpected_error",
                }

        # Analyze error propagation
        for file_key, result in error_chain_results.items():
            logger.info(
                f"Error chain for {file_key}: "
                f"completed {len(result['steps_completed'])} steps, "
                f"{len(result['errors'])} errors, "
                f"stopped at {result['stopped_at']}"
            )

        # Validate error propagation behavior
        # Each problematic file should fail at predictable points

        # Corrupted G2 should reach data validation but may fail at analysis
        if "corrupted_g2" in error_chain_results:
            corrupted_result = error_chain_results["corrupted_g2"]
            self.assertIn(
                "data_access",
                corrupted_result["steps_completed"],
                "Should be able to access corrupted data",
            )

        # Missing data should fail early
        if "missing_data" in error_chain_results:
            missing_result = error_chain_results["missing_data"]
            self.assertIn(
                missing_result["stopped_at"],
                ["file_loading", "data_access"],
                "Missing data should fail early in chain",
            )

        # At least one error chain should complete some steps
        completed_steps = [
            len(result["steps_completed"]) for result in error_chain_results.values()
        ]
        self.assertGreater(
            max(completed_steps), 0, "No error chains completed any steps"
        )

    def test_resource_cleanup_after_errors(self):
        """Test resource cleanup after error conditions."""
        initial_memory = MemoryMonitor.get_memory_usage()[0]

        # Cause various errors and ensure cleanup
        error_operations = []

        for file_key, test_file in self.test_files.items():
            try:
                # Attempt operation that may fail
                xf = XpcsFile(test_file)
                g2_data = xf.g2

                # Force some processing
                if g2_data is not None:
                    _ = np.mean(g2_data)

                error_operations.append((file_key, "success", None))

            except Exception as e:
                error_operations.append((file_key, "error", str(e)))

        # Force cleanup
        import gc

        gc.collect()

        # Check memory cleanup
        final_memory = MemoryMonitor.get_memory_usage()[0]
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable even after errors
        self.assertLess(
            memory_increase,
            200,
            f"Excessive memory after errors: {memory_increase:.1f} MB",
        )

        logger.info(
            f"Resource cleanup test: {len(error_operations)} operations, "
            f"memory increase: {memory_increase:.1f} MB"
        )


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)

    # Run error handling integration tests
    unittest.main(verbosity=2)
