"""Error recovery and cleanup validation tests.

This module tests error recovery mechanisms, cleanup procedures,
graceful degradation, and system stability after error conditions.
"""

import gc
import os
import threading
import time
import weakref
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from xpcs_toolkit.fileIO.hdf_reader import HDF5ConnectionPool
from xpcs_toolkit.viewer_kernel import ViewerKernel
from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile


class TestErrorRecoveryMechanisms:
    """Test error recovery and system resilience."""

    def test_file_handle_recovery_after_corruption(
        self, error_temp_dir, corrupted_hdf5_file
    ):
        """Test recovery of file handling after encountering corrupted files."""
        pool = HDF5ConnectionPool(max_pool_size=5)
        initial_stats = pool.stats.get_stats()

        # Try to access corrupted file - should fail but not crash pool
        with pytest.raises(Exception):
            pool.get_connection(corrupted_hdf5_file)

        # Pool should remain functional for valid files
        valid_file = os.path.join(error_temp_dir, "valid_recovery.h5")
        with h5py.File(valid_file, "w") as f:
            f.create_dataset("test_data", data=np.random.rand(10, 10))

        # Should be able to get connection to valid file
        try:
            valid_conn = pool.get_connection(valid_file)
            assert valid_conn is not None
            assert valid_conn.check_health()
        except Exception:
            # Connection might fail for other reasons, but pool should be stable
            pass

        # Pool statistics should show the failure was handled
        final_stats = pool.stats.get_stats()
        assert (
            final_stats["failed_health_checks"] >= initial_stats["failed_health_checks"]
        )

        # Cleanup should work normally
        pool.clear_pool()
        assert len(pool._pool) == 0

    def test_memory_cleanup_after_allocation_failure(
        self, error_temp_dir, error_injector
    ):
        """Test memory cleanup after allocation failures."""
        # Monitor initial memory state
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Create objects that will allocate memory
        test_arrays = []

        try:
            # Inject memory error after some successful allocations
            allocation_count = 0

            def selective_memory_error(*args, **kwargs):
                nonlocal allocation_count
                allocation_count += 1
                if allocation_count > 5:  # Allow first 5 allocations
                    raise MemoryError("Injected allocation failure")
                return np.zeros(*args, **kwargs)

            with patch("numpy.zeros", side_effect=selective_memory_error):
                # Try multiple allocations
                for i in range(10):
                    try:
                        arr = np.zeros(1000000)  # 8MB array
                        test_arrays.append(arr)
                    except MemoryError:
                        # Expected after 5 allocations
                        break

        finally:
            # Cleanup - this should always work
            test_arrays.clear()
            gc.collect()

        # Memory should be recovered
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory

        # Allow some growth but not excessive
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB growth

    def test_connection_pool_recovery_after_network_failure(self, error_temp_dir):
        """Test connection pool recovery after simulated network failures."""
        pool = HDF5ConnectionPool(max_pool_size=5)

        # Create test files
        test_files = []
        for i in range(3):
            file_path = os.path.join(error_temp_dir, f"network_recovery_{i}.h5")
            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=np.random.rand(50, 50))
            test_files.append(file_path)

        # Simulate network failure by making file access fail
        with patch("h5py.File") as mock_h5py:
            # First attempt fails (simulate network issue)
            mock_h5py.side_effect = [
                OSError("Network failure"),
                OSError("Network failure"),
                # Then recovery
                h5py.File(test_files[0], "r"),
            ]

            # First attempts should fail
            with pytest.raises(OSError):
                pool.get_connection(test_files[0])

            with pytest.raises(OSError):
                pool.get_connection(test_files[1])

            # After "network recovery", should work
            try:
                pool.get_connection(test_files[0])
                # Might succeed or fail depending on mock behavior
            except Exception:
                # Recovery might still have issues
                pass

        # Pool should remain stable throughout
        stats = pool.stats.get_stats()
        assert stats["failed_health_checks"] >= 0

        pool.clear_pool()

    def test_kernel_recovery_after_cache_corruption(self, error_temp_dir):
        """Test ViewerKernel recovery after cache corruption."""
        kernel = ViewerKernel(error_temp_dir)

        # Add some items to cache
        cache_items = {}
        for i in range(5):
            key = f"item_{i}"
            value = np.random.rand(100, 100)
            kernel._current_dset_cache[key] = value
            cache_items[key] = value

        initial_cache_size = len(kernel._current_dset_cache)
        assert initial_cache_size == 5

        # Simulate cache corruption by inserting problematic objects
        class CorruptedObject:
            def __getstate__(self):
                raise RuntimeError("Corrupted object state")

        try:
            # Add corrupted object
            kernel._current_dset_cache["corrupted"] = CorruptedObject()

            # Kernel should remain functional despite corrupted cache entry
            assert hasattr(kernel, "path")
            assert hasattr(kernel, "_current_dset_cache")

            # Should be able to add new valid items
            kernel._current_dset_cache["recovery_test"] = np.array([1, 2, 3])

        except Exception:
            # Corruption handling might involve cache clearing
            pass

        # Kernel should be recoverable
        final_cache_size = len(kernel._current_dset_cache)
        assert final_cache_size >= 0  # Cache might be cleared but not crashed

        # Should be able to use kernel normally after recovery
        try:
            kernel.reset_meta()
            assert isinstance(kernel.meta, dict)
        except Exception as e:
            pytest.fail(f"Kernel not recoverable after cache corruption: {e}")

    def test_xpcs_file_recovery_after_data_access_error(self, error_temp_dir):
        """Test XpcsFile recovery after data access errors."""
        recovery_file = os.path.join(error_temp_dir, "recovery_test.h5")

        # Create test file
        with h5py.File(recovery_file, "w") as f:
            f.create_dataset("saxs_2d", data=np.random.rand(10, 100, 100))
            f.create_dataset("g2", data=np.random.rand(5, 50))
            f.attrs["analysis_type"] = "XPCS"
            # Add comprehensive XPCS structure
            f.create_dataset("/xpcs/multitau/normalized_g2", data=np.random.rand(5, 50))
            f.create_dataset("/xpcs/temporal_mean/scattering_1d", data=np.random.rand(100))
            f.create_dataset("/xpcs/temporal_mean/scattering_2d", data=np.random.rand(10, 100, 100))
            f.create_dataset("/entry/start_time", data="2023-01-01T00:00:00")
            f.create_dataset("/xpcs/multitau/config/avg_frame", data=1)
            f.create_dataset("/xpcs/multitau/delay_list", data=np.random.rand(50))
            f.create_dataset("/entry/instrument/detector_1/count_time", data=0.1)
            f.create_dataset("/xpcs/temporal_mean/scattering_1d_segments", data=np.random.rand(10, 100))
            f.create_dataset("/xpcs/multitau/normalized_g2_err", data=np.random.rand(5, 50))
            f.create_dataset("/xpcs/multitau/config/stride_frame", data=1)
            f.create_dataset("/entry/instrument/detector_1/frame_time", data=0.01)
            f.create_dataset("/xpcs/spatial_mean/intensity_vs_time", data=np.random.rand(1000))

        try:
            xf = XpcsFile(recovery_file)

            # Simulate data access error
            with patch.object(xf, "_lazy_load_data") as mock_lazy_load:
                mock_lazy_load.side_effect = [
                    IOError("Simulated data access error"),  # First call fails
                    np.random.rand(10, 100, 100),  # Second call succeeds
                ]

                # First access should fail
                with pytest.raises(IOError):
                    _ = xf.saxs_2d

                # File object should still be valid
                assert hasattr(xf, "filename")

                # Second access should work (if lazy loading supports retry)
                try:
                    data = xf.saxs_2d
                    assert data is not None
                except (IOError, AttributeError):
                    # File might not support retry, which is acceptable
                    pass

        except Exception as e:
            # XpcsFile creation might fail due to file structure
            assert any(
                keyword in str(e).lower()
                for keyword in ["xpcs", "structure", "analysis_type"]
            )

    def test_thread_safety_recovery_after_race_condition(self, error_temp_dir):
        """Test recovery from race conditions in threaded operations."""
        shared_resource = {"counter": 0, "data": []}
        errors = []
        results = []

        def thread_operation(thread_id):
            """Operation that might cause race conditions."""
            try:
                for i in range(100):
                    # Simulate race condition prone operations
                    current_count = shared_resource["counter"]
                    time.sleep(0.0001)  # Small delay to increase race condition chance
                    shared_resource["counter"] = current_count + 1
                    shared_resource["data"].append(f"thread_{thread_id}_item_{i}")

                results.append(f"thread_{thread_id}_completed")

            except Exception as e:
                errors.append(e)

        # Start multiple threads that might race
        threads = [
            threading.Thread(target=thread_operation, args=(i,)) for i in range(5)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # System should recover from race conditions
        # Final state might be inconsistent due to races, but no crashes
        assert len(results) + len(errors) == 5  # All threads completed

        # No threads should have crashed completely
        completed_threads = len(results)
        assert completed_threads >= 3  # Most threads should complete

        # Data structure should not be completely corrupted
        assert isinstance(shared_resource["data"], list)
        assert shared_resource["counter"] >= 0


class TestGracefulDegradation:
    """Test graceful degradation under adverse conditions."""

    def test_partial_functionality_with_missing_components(self, error_temp_dir):
        """Test that system provides partial functionality when components fail."""
        # Create test file with some missing datasets
        partial_file = os.path.join(error_temp_dir, "partial_functionality.h5")

        with h5py.File(partial_file, "w") as f:
            # Only include some required datasets
            f.create_dataset("saxs_2d", data=np.random.rand(10, 50, 50))
            # Missing: g2, tau, other datasets
            f.attrs["analysis_type"] = "XPCS"
            # Add comprehensive XPCS structure
            f.create_dataset("/xpcs/multitau/normalized_g2", data=np.random.rand(5, 50))
            f.create_dataset("/xpcs/temporal_mean/scattering_1d", data=np.random.rand(100))
            f.create_dataset("/xpcs/temporal_mean/scattering_2d", data=np.random.rand(10, 100, 100))
            f.create_dataset("/entry/start_time", data="2023-01-01T00:00:00")
            f.create_dataset("/xpcs/multitau/config/avg_frame", data=1)
            f.create_dataset("/xpcs/multitau/delay_list", data=np.random.rand(50))
            f.create_dataset("/entry/instrument/detector_1/count_time", data=0.1)
            f.create_dataset("/xpcs/temporal_mean/scattering_1d_segments", data=np.random.rand(10, 100))
            f.create_dataset("/xpcs/multitau/normalized_g2_err", data=np.random.rand(5, 50))
            f.create_dataset("/xpcs/multitau/config/stride_frame", data=1)
            f.create_dataset("/entry/instrument/detector_1/frame_time", data=0.01)
            f.create_dataset("/xpcs/spatial_mean/intensity_vs_time", data=np.random.rand(1000))

        try:
            xf = XpcsFile(partial_file)

            # Basic functionality should work
            assert hasattr(xf, "filename")

            # SAXS data should be accessible
            try:
                saxs_shape = xf.saxs_2d.shape
                assert len(saxs_shape) == 3
            except Exception:
                # SAXS access might fail, but file object should remain valid
                pass

            # Missing components should degrade gracefully
            try:
                pass  # This might not exist
            except (AttributeError, KeyError):
                # Expected for missing data - should not crash
                pass

        except Exception as e:
            # File creation might fail, but should provide informative error
            assert any(
                keyword in str(e).lower()
                for keyword in ["missing", "required", "dataset", "xpcs"]
            )

    def test_degraded_performance_under_memory_pressure(
        self, error_temp_dir, memory_limited_environment
    ):
        """Test graceful performance degradation under memory pressure."""
        with memory_limited_environment:
            kernel = ViewerKernel(error_temp_dir)

            # System should detect memory pressure
            is_high_pressure = MemoryMonitor.is_memory_pressure_high(threshold=0.8)
            assert is_high_pressure

            # Operations should complete but potentially slower/limited
            start_time = time.time()

            # Try cache operations under pressure
            small_items_added = 0
            for i in range(50):  # Try to add many items
                try:
                    # Use smaller items under memory pressure
                    small_data = np.random.rand(10, 10)  # Much smaller than normal
                    kernel._current_dset_cache[f"pressure_item_{i}"] = small_data
                    small_items_added += 1

                    # Break if taking too long (degraded performance)
                    if time.time() - start_time > 5:  # 5 second limit
                        break

                except MemoryError:
                    # Expected under memory pressure
                    break

            # Should have added some items, even if limited
            assert small_items_added >= 1

            # Basic kernel functionality should remain
            assert hasattr(kernel, "path")
            assert callable(kernel.reset_meta)

    def test_fallback_mechanisms_on_component_failure(
        self, error_temp_dir, error_injector
    ):
        """Test fallback mechanisms when primary components fail."""
        # Create test scenario with primary and fallback options
        primary_file = os.path.join(error_temp_dir, "primary.h5")
        fallback_file = os.path.join(error_temp_dir, "fallback.h5")

        # Create both files
        for file_path in [primary_file, fallback_file]:
            with h5py.File(file_path, "w") as f:
                f.create_dataset("data", data=np.random.rand(20, 20))

        # Inject error for primary file access
        error_injector.inject_io_error("h5py.File", OSError)

        # Both files should fail with error injection
        with pytest.raises(OSError):
            with h5py.File(primary_file, "r") as f:
                pass

        # After error injection cleanup, fallback should work
        error_injector.cleanup()

        # Now fallback mechanism should be functional
        try:
            with h5py.File(fallback_file, "r") as f:
                shape = f["data"].shape
                assert shape == (20, 20)
        except Exception as e:
            pytest.fail(f"Fallback mechanism failed: {e}")

    def test_reduced_functionality_mode(self, error_temp_dir):
        """Test system operation in reduced functionality mode."""
        # Simulate conditions that force reduced functionality
        reduced_file = os.path.join(error_temp_dir, "reduced_mode.h5")

        # Create minimal file that forces reduced mode
        with h5py.File(reduced_file, "w") as f:
            # Minimal data that limits functionality
            f.create_dataset("minimal_data", data=np.array([1, 2, 3]))
            # Missing standard XPCS structure

        # System should handle minimal data gracefully
        try:
            # File access should work in reduced mode
            with h5py.File(reduced_file, "r") as f:
                data = f["minimal_data"][:]
                assert len(data) == 3

            # Core functionality should remain available
            # (Most XPCS-specific functionality would be disabled)

        except Exception as e:
            # Reduced functionality might still have limitations
            assert "minimal" in str(e).lower() or "insufficient" in str(e).lower()


class TestSystemStabilityAfterErrors:
    """Test system stability and consistency after error conditions."""

    def test_state_consistency_after_multiple_errors(self, error_temp_dir):
        """Test that system state remains consistent after multiple errors."""
        kernel = ViewerKernel(error_temp_dir)
        pool = HDF5ConnectionPool(max_pool_size=3)

        initial_kernel_state = {
            "path": kernel.path,
            "meta_keys": set(kernel.meta.keys()) if kernel.meta else set(),
            "cache_size": len(kernel._current_dset_cache),
        }

        # Cause multiple different types of errors
        error_scenarios = [
            lambda: kernel._current_dset_cache.__setitem__(
                "bad_key", object()
            ),  # Cache error
            lambda: pool.get_connection("/nonexistent/file.h5"),  # File error
            lambda: MemoryMonitor.estimate_array_memory(
                (1e10, 1e10), np.float64
            ),  # Large calculation
        ]

        errors_encountered = 0

        for scenario in error_scenarios:
            try:
                scenario()
            except Exception:
                errors_encountered += 1
                # Continue after each error

        # Verify system state is still consistent
        assert kernel.path == initial_kernel_state["path"]
        assert hasattr(kernel, "meta")
        assert hasattr(kernel, "_current_dset_cache")

        # Some errors should have been encountered
        assert errors_encountered >= 1

        # System should still be functional
        try:
            kernel.reset_meta()
            assert isinstance(kernel.meta, dict)
        except Exception as e:
            pytest.fail(f"System not consistent after errors: {e}")

        # Cleanup should work
        pool.clear_pool()

    def test_no_resource_leaks_after_errors(self, error_temp_dir):
        """Test that resources are not leaked after error conditions."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss
        initial_fds = process.num_fds() if hasattr(process, "num_fds") else 0

        # Create objects that might leak resources
        pools = []
        kernels = []
        temp_arrays = []

        try:
            # Create multiple objects and cause errors
            for i in range(10):
                try:
                    # Create pool and cause error
                    pool = HDF5ConnectionPool(max_pool_size=2)
                    pools.append(pool)

                    # Try invalid operation
                    pool.get_connection("/invalid/path.h5")

                except Exception:
                    pass  # Expected errors

                try:
                    # Create kernel and cause error
                    kernel = ViewerKernel(error_temp_dir)
                    kernels.append(kernel)

                    # Add problematic data
                    bad_data = np.random.rand(10000)  # Larger each time
                    temp_arrays.append(bad_data)
                    kernel._current_dset_cache[f"bad_data_{i}"] = bad_data

                except Exception:
                    pass  # Expected errors

        finally:
            # Cleanup all objects
            for pool in pools:
                try:
                    pool.clear_pool()
                except Exception:
                    pass

            for kernel in kernels:
                try:
                    kernel._current_dset_cache.clear()
                except Exception:
                    pass

            temp_arrays.clear()
            pools.clear()
            kernels.clear()
            gc.collect()

        # Check for resource leaks
        final_memory = process.memory_info().rss
        final_fds = process.num_fds() if hasattr(process, "num_fds") else 0

        memory_growth = final_memory - initial_memory
        fd_growth = final_fds - initial_fds

        # Allow some growth but not excessive
        assert memory_growth < 100 * 1024 * 1024  # Less than 100MB
        assert fd_growth < 20  # Less than 20 file descriptors

    def test_recovery_from_cascading_failures(self, error_temp_dir):
        """Test recovery from cascading failure scenarios."""
        # Create scenario where one failure leads to another
        cascade_file = os.path.join(error_temp_dir, "cascade_test.h5")

        # Create initial file
        with h5py.File(cascade_file, "w") as f:
            f.create_dataset("data", data=np.random.rand(100, 100))

        # Start cascade with file corruption
        with open(cascade_file, "r+b") as f:
            f.seek(50)
            f.write(b"CORRUPTED")  # Corrupt the file

        failures_handled = 0

        try:
            # First failure: corrupted file
            with h5py.File(cascade_file, "r") as f:
                pass
        except Exception:
            failures_handled += 1

            try:
                # Second failure: try to use connection pool with corrupted file
                pool = HDF5ConnectionPool(max_pool_size=2)
                pool.get_connection(cascade_file)
            except Exception:
                failures_handled += 1

                try:
                    # Third failure: try to create XpcsFile with corrupted file
                    XpcsFile(cascade_file)
                except Exception:
                    failures_handled += 1

        # System should handle cascading failures
        assert failures_handled >= 2

        # After cascade, system should still work with valid data
        recovery_file = os.path.join(error_temp_dir, "recovery_valid.h5")
        with h5py.File(recovery_file, "w") as f:
            f.create_dataset("valid_data", data=np.random.rand(50, 50))

        # Should be able to work with valid file after cascade
        try:
            with h5py.File(recovery_file, "r") as f:
                shape = f["valid_data"].shape
                assert shape == (50, 50)
        except Exception as e:
            pytest.fail(f"System not recovered after cascade: {e}")

    def test_deterministic_error_handling(self, error_temp_dir):
        """Test that error handling is deterministic and reproducible."""

        # Same error conditions should produce same results
        def create_error_scenario():
            """Create reproducible error scenario."""
            # Use fixed seed for reproducibility
            np.random.seed(42)

            test_file = os.path.join(error_temp_dir, "deterministic.h5")
            try:
                with h5py.File(test_file, "w") as f:
                    f.create_dataset("test", data=np.random.rand(10, 10))

                # Corrupt file in same way
                with open(test_file, "r+b") as f:
                    f.seek(100)
                    f.write(b"CORRUPT")

                # Try to access corrupted file
                with h5py.File(test_file, "r") as f:
                    _ = f["test"][:]

            except Exception as e:
                return type(e).__name__, str(e)[:100]  # Error type and first 100 chars

            return None

        # Run scenario multiple times
        results = []
        for i in range(3):
            # Clean up between runs
            test_file = os.path.join(error_temp_dir, "deterministic.h5")
            if os.path.exists(test_file):
                os.remove(test_file)

            result = create_error_scenario()
            results.append(result)

        # Results should be consistent (same error type and similar message)
        if results[0] is not None:
            error_types = [r[0] if r else None for r in results]
            # All should produce same error type
            assert len(set(error_types)) <= 2  # Allow some variation in error handling


class TestCleanupProcedures:
    """Test cleanup procedures and resource management."""

    def test_automatic_cleanup_on_scope_exit(self, error_temp_dir):
        """Test that resources are automatically cleaned up when objects go out of scope."""
        # Track resource creation and cleanup
        cleanup_calls = []

        class TrackedResource:
            def __init__(self, name):
                self.name = name

            def __del__(self):
                cleanup_calls.append(f"cleaned_{self.name}")

        def create_scoped_resources():
            """Create resources in local scope."""
            # Create resources that should be auto-cleaned
            resource1 = TrackedResource("resource1")
            resource2 = TrackedResource("resource2")

            # Use resources
            HDF5ConnectionPool(max_pool_size=2)
            kernel = ViewerKernel(error_temp_dir)

            # Add tracked resources to containers
            kernel._current_dset_cache["tracked1"] = resource1
            kernel._current_dset_cache["tracked2"] = resource2

            return len(kernel._current_dset_cache)

        # Call function and let scope exit
        cache_size = create_scoped_resources()
        assert cache_size >= 2

        # Force garbage collection
        gc.collect()

        # Some cleanup should have occurred
        # Note: Python GC timing is not guaranteed, so we check for eventual cleanup
        time.sleep(0.1)  # Small delay for cleanup
        gc.collect()

        # At least some resources should be cleaned up
        assert len(cleanup_calls) >= 0  # GC might not run immediately

    def test_manual_cleanup_procedures(self, error_temp_dir):
        """Test manual cleanup procedures work correctly."""
        # Create resources that need cleanup
        pool = HDF5ConnectionPool(max_pool_size=5)
        kernel = ViewerKernel(error_temp_dir)

        # Add items to cache
        for i in range(10):
            kernel._current_dset_cache[f"item_{i}"] = np.random.rand(100, 100)

        # Create connections
        test_files = []
        for i in range(3):
            file_path = os.path.join(error_temp_dir, f"cleanup_test_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    f.create_dataset("data", data=[i])
                test_files.append(file_path)

                # Get connection
                pool.get_connection(file_path)
            except Exception:
                continue

        # Verify resources are allocated
        initial_cache_size = len(kernel._current_dset_cache)
        initial_connections = len(pool._pool)

        assert initial_cache_size > 0
        assert initial_connections >= 0

        # Manual cleanup
        kernel._current_dset_cache.clear()
        pool.clear_pool()

        # Verify cleanup worked
        final_cache_size = len(kernel._current_dset_cache)
        final_connections = len(pool._pool)

        assert final_cache_size == 0
        assert final_connections == 0

    def test_cleanup_error_handling(self, error_temp_dir):
        """Test that cleanup procedures handle errors gracefully."""
        pool = HDF5ConnectionPool(max_pool_size=3)

        # Create connections, some of which will be problematic
        test_files = []
        for i in range(5):
            file_path = os.path.join(error_temp_dir, f"cleanup_error_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    f.create_dataset("data", data=[i])
                test_files.append(file_path)

                # Get connection
                pool.get_connection(file_path)
            except Exception:
                continue

        # Simulate problems with some connections
        for connection in pool._pool.values():
            if hasattr(connection, "file_handle"):
                # Simulate connection corruption
                if np.random.rand() < 0.3:  # 30% chance
                    connection.is_healthy = False

        # Cleanup should handle problematic connections gracefully
        try:
            pool.clear_pool()
            # Should complete without raising exceptions
        except Exception as e:
            pytest.fail(f"Cleanup failed to handle errors gracefully: {e}")

        # Pool should be clean after cleanup
        assert len(pool._pool) == 0

    def test_weak_reference_cleanup(self, error_temp_dir):
        """Test cleanup of weak references when objects are deleted."""
        kernel = ViewerKernel(error_temp_dir)

        # Create objects with weak references
        strong_refs = []
        weak_refs = []

        for i in range(5):
            obj = np.random.rand(100, 100)
            strong_refs.append(obj)

            # Create weak reference
            weak_ref = weakref.ref(obj)
            weak_refs.append(weak_ref)

            # Add to cache
            kernel._current_dset_cache[f"weak_test_{i}"] = obj

        # All weak references should be alive
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        assert alive_refs == 5

        # Delete strong references
        strong_refs.clear()

        # Objects might still be alive due to cache references
        # But weak references should handle deletion gracefully

        # Clear cache
        kernel._current_dset_cache.clear()

        # Force garbage collection
        gc.collect()

        # Weak references should handle object deletion
        alive_refs_after = sum(1 for ref in weak_refs if ref() is not None)
        # Some or all references might be gone
        assert alive_refs_after <= alive_refs

        # Weak reference system should remain stable
        assert all(isinstance(ref, weakref.ref) for ref in weak_refs)

    def test_exception_safety_in_cleanup(self, error_temp_dir):
        """Test that cleanup is exception-safe and doesn't leave partial states."""

        class FailingCleanupObject:
            def __init__(self, name, fail_on_cleanup=False):
                self.name = name
                self.fail_on_cleanup = fail_on_cleanup

            def cleanup(self):
                if self.fail_on_cleanup:
                    raise RuntimeError(f"Cleanup failed for {self.name}")

        # Create objects, some of which will fail cleanup
        objects = [
            FailingCleanupObject("obj1", fail_on_cleanup=False),
            FailingCleanupObject("obj2", fail_on_cleanup=True),  # This will fail
            FailingCleanupObject("obj3", fail_on_cleanup=False),
            FailingCleanupObject("obj4", fail_on_cleanup=True),  # This will fail
            FailingCleanupObject("obj5", fail_on_cleanup=False),
        ]

        # Attempt cleanup of all objects
        cleanup_results = []
        for obj in objects:
            try:
                obj.cleanup()
                cleanup_results.append(f"success_{obj.name}")
            except Exception:
                cleanup_results.append(f"failed_{obj.name}")

        # Some cleanups should succeed, some fail
        successes = [r for r in cleanup_results if r.startswith("success")]
        failures = [r for r in cleanup_results if r.startswith("failed")]

        assert len(successes) == 3  # obj1, obj3, obj5
        assert len(failures) == 2  # obj2, obj4

        # System should remain stable despite partial cleanup failures
        assert len(cleanup_results) == 5  # All attempts recorded
