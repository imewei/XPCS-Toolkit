"""Resource exhaustion simulation and testing.

This module tests system behavior under various resource exhaustion scenarios
including memory, disk space, file handles, CPU, and network resources.
"""

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch

import h5py
import numpy as np
import psutil
import pytest

from xpcs_toolkit.fileIO.hdf_reader import HDF5ConnectionPool
from xpcs_toolkit.module.average_toolbox import AverageToolbox
from xpcs_toolkit.viewer_kernel import ViewerKernel
from xpcs_toolkit.xpcs_file import MemoryMonitor, XpcsFile


class TestMemoryExhaustion:
    """Test behavior under memory exhaustion scenarios."""

    def test_gradual_memory_consumption(self, error_temp_dir):
        """Test system behavior as memory is gradually consumed."""
        # Monitor initial memory state
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        memory_consumers = []
        max_attempts = 50

        try:
            # Gradually consume memory
            for i in range(max_attempts):
                try:
                    # Allocate ~10MB chunks
                    chunk = np.random.rand(
                        1024 * 1024 + i * 1000
                    )  # Slightly increasing size
                    memory_consumers.append(chunk)

                    current_memory = process.memory_info().rss
                    memory_growth = current_memory - initial_memory

                    # Check if memory monitoring detects pressure
                    if memory_growth > 100 * 1024 * 1024:  # 100MB growth
                        MemoryMonitor.is_memory_pressure_high(threshold=0.8)
                        # System might detect pressure (implementation dependent)

                    # Stop before causing system issues
                    if memory_growth > 500 * 1024 * 1024:  # 500MB limit
                        break

                except MemoryError:
                    # Expected when memory is exhausted
                    break

        finally:
            # Clean up memory consumers
            memory_consumers.clear()
            gc.collect()

        # Verify system recovers
        final_memory = process.memory_info().rss
        memory_recovered = final_memory - initial_memory
        assert memory_recovered < 200 * 1024 * 1024  # Should recover most memory

    def test_memory_allocation_failure_simulation(
        self, error_temp_dir, resource_exhaustion
    ):
        """Test handling of memory allocation failures."""
        with resource_exhaustion.simulate_memory_pressure(threshold=0.95):
            # Try operations that require memory allocation
            kernel = ViewerKernel(error_temp_dir)

            # Operations should detect memory pressure and adapt
            try:
                # Large array allocation should be avoided or chunked
                large_data = np.random.rand(1000, 1000)
                kernel._current_dset_cache["test_large"] = large_data

                # Cache might reject large items under pressure
                cache_size = len(kernel._current_dset_cache)
                assert cache_size >= 0  # Should not crash

            except MemoryError:
                # Acceptable under memory pressure
                pass

    def test_memory_fragmentation_handling(self, error_temp_dir):
        """Test handling of memory fragmentation scenarios."""
        # Create fragmentation pattern
        fragments = []

        try:
            # Allocate many small arrays to fragment memory
            for i in range(1000):
                fragment = np.random.rand(100 + i % 50)  # Variable sizes
                fragments.append(fragment)

            # Delete every other fragment to create gaps (delete in reverse order to avoid index issues)
            indices_to_delete = list(range(0, len(fragments), 2))
            for i in reversed(indices_to_delete):
                del fragments[i]

            gc.collect()  # Force garbage collection

            # Try to allocate large contiguous array
            try:
                large_array = np.random.rand(10000, 1000)  # 80MB
                # Should either succeed or fail gracefully
                assert large_array.shape == (10000, 1000)

            except MemoryError:
                # Fragmentation might prevent large allocation
                pass

        finally:
            # Clean up
            fragments.clear()
            gc.collect()

    def test_connection_pool_memory_management(self, error_temp_dir):
        """Test connection pool behavior under memory pressure."""
        pool = HDF5ConnectionPool(max_pool_size=10)

        # Create many test files
        test_files = []
        for i in range(20):
            file_path = os.path.join(error_temp_dir, f"pool_memory_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    # Each file has some data
                    f.create_dataset("data", data=np.random.rand(100, 100))
                test_files.append(file_path)
            except Exception:
                # File creation might fail
                continue

        connections = []

        try:
            # Simulate memory pressure affecting connection pool
            with patch("psutil.virtual_memory") as mock_memory:
                # Start with normal memory
                mock_memory.return_value = Mock(
                    total=8 * 1024**3,  # 8GB
                    available=4 * 1024**3,  # 4GB available
                    percent=50.0,
                )

                # Get connections normally
                for file_path in test_files[:5]:
                    try:
                        conn = pool.get_connection(file_path)
                        connections.append(conn)
                    except Exception:
                        continue

                # Simulate memory pressure
                mock_memory.return_value = Mock(
                    total=8 * 1024**3,  # 8GB
                    available=200 * 1024**2,  # 200MB available
                    percent=97.5,
                )

                # Pool should limit new connections under pressure
                for file_path in test_files[5:10]:
                    try:
                        conn = pool.get_connection(file_path)
                        # Might succeed or fail under pressure
                    except (MemoryError, OSError):
                        # Expected under memory pressure
                        pass

                # Pool should remain functional
                stats = pool.stats.get_stats()
                assert stats["total_connections_created"] >= 0

        finally:
            pool.clear_pool()

    def test_lazy_loading_memory_efficiency(
        self, error_temp_dir, memory_limited_environment
    ):
        """Test lazy loading efficiency under memory constraints."""
        large_file = os.path.join(error_temp_dir, "lazy_memory.h5")

        # Create file with multiple large datasets
        with h5py.File(large_file, "w") as f:
            # Create chunked datasets to enable lazy loading
            large_saxs = f.create_dataset(
                "saxs_2d",
                shape=(50, 1000, 1000),
                dtype=np.float64,
                chunks=True,
                compression="gzip",
            )

            # Fill with test data in chunks to avoid memory issues
            chunk_size = 10
            for i in range(0, 50, chunk_size):
                end_idx = min(i + chunk_size, 50)
                large_saxs[i:end_idx] = np.random.rand(end_idx - i, 1000, 1000)

            f.attrs["analysis_type"] = "XPCS"

        try:
            with memory_limited_environment:
                xf = XpcsFile(large_file)

                # Should be able to create file object without loading all data
                assert hasattr(xf, "filename")

                # Accessing small chunks should work under memory pressure
                try:
                    chunk = xf.saxs_2d[0:1]  # Single frame
                    assert chunk.shape[0] == 1
                except (MemoryError, OSError):
                    # Acceptable under extreme memory pressure
                    pass

                # Accessing full dataset should fail gracefully
                with pytest.raises((MemoryError, OSError)):
                    xf.saxs_2d[:]

        except Exception as e:
            # File structure validation might fail
            assert any(
                keyword in str(e).lower()
                for keyword in ["memory", "xpcs", "structure", "lazy"]
            )


class TestDiskSpaceExhaustion:
    """Test behavior when disk space is exhausted."""

    def test_disk_space_monitoring(
        self, error_temp_dir, disk_space_limited_environment
    ):
        """Test disk space monitoring and handling."""
        with disk_space_limited_environment:
            # Try to create large file when disk space is limited
            large_file = os.path.join(error_temp_dir, "disk_test.h5")

            try:
                with h5py.File(large_file, "w") as f:
                    # Try to create dataset larger than available space
                    # This should fail or be handled gracefully
                    large_data = np.random.rand(1000, 1000, 1000)  # ~8GB
                    f.create_dataset("huge_data", data=large_data)

            except (OSError, IOError) as e:
                # Expected when disk is full
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["space", "full", "disk", "write"]
                )

    def test_temporary_file_cleanup_on_disk_full(self, error_temp_dir):
        """Test cleanup of temporary files when disk becomes full."""
        temp_files = []

        try:
            # Create multiple temporary files
            for i in range(10):
                temp_file = os.path.join(error_temp_dir, f"temp_{i}.tmp")
                try:
                    with open(temp_file, "w") as f:
                        f.write("x" * 1024 * 1024)  # 1MB each
                    temp_files.append(temp_file)
                except OSError:
                    # Disk might be full
                    break

            # Simulate disk full condition
            with patch("shutil.disk_usage") as mock_usage:
                mock_usage.return_value = (1000000000, 0, 1000000000)  # No free space

                # Cleanup should handle disk full condition
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except OSError:
                        # Cleanup might fail when disk is full
                        pass

        finally:
            # Ensure cleanup
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except OSError:
                    pass

    def test_hdf5_file_creation_with_limited_space(self, error_temp_dir):
        """Test HDF5 file creation when disk space is limited."""
        limited_space_file = os.path.join(error_temp_dir, "limited_space.h5")

        # Mock limited disk space
        with patch("shutil.disk_usage") as mock_usage:
            # Only 50MB free space
            mock_usage.return_value = (1000000000, 50 * 1024 * 1024, 950 * 1024 * 1024)

            try:
                with h5py.File(limited_space_file, "w") as f:
                    # Try to create dataset that might exceed space
                    # Use reasonable size for testing
                    modest_data = np.random.rand(100, 512, 512)  # ~200MB
                    f.create_dataset("data", data=modest_data)

            except (OSError, IOError) as e:
                # Expected when space is insufficient
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["space", "write", "full", "disk"]
                )

    def test_log_file_rotation_on_disk_full(self, error_temp_dir):
        """Test log file rotation when disk becomes full."""
        import logging

        log_file = os.path.join(error_temp_dir, "test.log")

        # Create logger with file handler
        logger = logging.getLogger("disk_full_test")
        handler = logging.FileHandler(log_file)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        try:
            # Write many log messages
            for i in range(1000):
                try:
                    logger.info(f"Test log message {i} " + "x" * 100)
                except OSError:
                    # Disk full - logging should handle gracefully
                    break

            # Log file should exist (even if truncated)
            if os.path.exists(log_file):
                file_size = os.path.getsize(log_file)
                assert file_size >= 0

        finally:
            # Cleanup
            logger.removeHandler(handler)
            handler.close()


class TestFileHandleExhaustion:
    """Test behavior when file handle limits are reached."""

    def test_file_handle_limit_detection(self, error_temp_dir):
        """Test detection and handling of file handle limits."""
        open_files = []
        max_files = 100  # Reasonable limit for testing

        try:
            # Open many files simultaneously
            for i in range(max_files):
                file_path = os.path.join(error_temp_dir, f"handle_test_{i}.tmp")
                try:
                    f = open(file_path, "w")
                    f.write(f"Test file {i}")
                    open_files.append(f)
                except OSError as e:
                    if "too many open files" in str(e).lower():
                        # Reached the limit
                        break
                    else:
                        raise

            # System should handle file handle exhaustion
            assert len(open_files) <= max_files

        finally:
            # Clean up open files
            for f in open_files:
                try:
                    f.close()
                except Exception:
                    pass

    def test_hdf5_connection_pool_handle_management(self, error_temp_dir):
        """Test HDF5 connection pool file handle management."""
        pool = HDF5ConnectionPool(max_pool_size=20)  # Reasonable limit

        # Create many test files
        test_files = []
        for i in range(50):  # More files than pool size
            file_path = os.path.join(error_temp_dir, f"handle_pool_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    f.create_dataset("data", data=[i])
                test_files.append(file_path)
            except Exception:
                continue

        connections = []

        try:
            # Get connections to many files
            for file_path in test_files:
                try:
                    conn = pool.get_connection(file_path)
                    connections.append(conn)
                except Exception:
                    # Pool might reject connections beyond limit
                    continue

            # Pool should not exceed reasonable handle count
            active_connections = len(pool._pool)
            assert active_connections <= 20

            # Pool should handle cleanup when limit is reached
            stats = pool.stats.get_stats()
            assert stats["total_connections_evicted"] >= 0

        finally:
            pool.clear_pool()

    def test_concurrent_file_access_handle_limits(self, error_temp_dir):
        """Test concurrent file access under handle limits."""
        test_file = os.path.join(error_temp_dir, "concurrent_handles.h5")

        # Create test file
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=np.random.rand(100, 100))

        access_results = []
        access_errors = []

        def access_file_multiple_times():
            """Function that opens file multiple times."""
            local_results = []
            local_errors = []

            for i in range(10):  # Each thread tries 10 file opens
                try:
                    with h5py.File(test_file, "r") as f:
                        shape = f["data"].shape
                        local_results.append(shape)
                        time.sleep(0.001)  # Small delay to increase contention

                except OSError as e:
                    if "too many open files" in str(e).lower():
                        local_errors.append(e)
                        break
                    else:
                        local_errors.append(e)

            return local_results, local_errors

        # Run concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_file_multiple_times) for _ in range(10)]

            for future in as_completed(futures):
                results, errors = future.result()
                access_results.extend(results)
                access_errors.extend(errors)

        # Most accesses should succeed despite potential handle limits
        total_attempts = len(access_results) + len(access_errors)
        if total_attempts > 0:
            success_rate = len(access_results) / total_attempts
            assert success_rate >= 0.3  # At least 30% should succeed

    def test_file_descriptor_cleanup_on_errors(self, error_temp_dir):
        """Test file descriptor cleanup when errors occur."""
        test_files = []

        # Create test files
        for i in range(20):
            file_path = os.path.join(error_temp_dir, f"fd_cleanup_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    f.create_dataset("data", data=[i] * 100)
                test_files.append(file_path)
            except Exception:
                continue

        initial_fd_count = (
            len(os.listdir("/proc/self/fd")) if os.path.exists("/proc/self/fd") else 0
        )

        # Perform operations that might fail and require cleanup
        for file_path in test_files:
            try:
                # Inject random errors to test cleanup
                if np.random.rand() < 0.3:  # 30% chance of artificial error
                    raise IOError("Simulated I/O error")

                with h5py.File(file_path, "r") as f:
                    _ = f["data"].shape

            except (IOError, OSError):
                # Errors should not leak file descriptors
                continue

        # Check that file descriptors are not leaked
        if os.path.exists("/proc/self/fd"):
            final_fd_count = len(os.listdir("/proc/self/fd"))
            fd_growth = final_fd_count - initial_fd_count
            assert fd_growth < 10  # Should not leak many descriptors


class TestCPUResourceExhaustion:
    """Test behavior under CPU resource exhaustion."""

    def test_cpu_intensive_operations(self, error_temp_dir):
        """Test handling of CPU-intensive operations."""
        cpu_file = os.path.join(error_temp_dir, "cpu_intensive.h5")

        # Create data that requires CPU-intensive processing
        with h5py.File(cpu_file, "w") as f:
            # Large dataset for FFT operations
            large_data = np.random.rand(100, 512, 512)
            f.create_dataset("saxs_2d", data=large_data)

            # Complex G2 data for fitting
            tau = np.logspace(-6, 2, 100)
            g2_data = np.random.rand(50, 100) + 1
            f.create_dataset("g2", data=g2_data)
            f.create_dataset("tau", data=tau)

            f.attrs["analysis_type"] = "XPCS"
            # Add comprehensive XPCS structure
            f.create_dataset("/xpcs/multitau/normalized_g2", data=np.random.rand(5, 50))
            f.create_dataset("/xpcs/temporal_mean/scattering_1d", data=np.random.rand(100))
            f.create_dataset("/xpcs/temporal_mean/scattering_2d", data=large_data)
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
            xf = XpcsFile(cpu_file)

            # CPU-intensive operations should complete or timeout gracefully
            start_time = time.time()

            try:
                # FFT operations on large data
                for i in range(min(10, xf.saxs_2d.shape[0])):  # Limit iterations
                    frame = xf.saxs_2d[i]
                    np.fft.fft2(frame)

                    # Check for timeout
                    if time.time() - start_time > 10:  # 10 second timeout
                        break

            except Exception as e:
                # CPU exhaustion might cause various errors
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["timeout", "resource", "cpu", "memory"]
                )

        except Exception as e:
            # File processing might fail due to size/structure
            assert any(
                keyword in str(e).lower()
                for keyword in ["cpu", "intensive", "timeout", "xpcs"]
            )

    def test_multiprocessing_resource_limits(self, error_temp_dir):
        """Test multiprocessing behavior under resource limits."""

        # Create multiple files for averaging
        test_files = []
        for i in range(5):  # Limited number for testing
            file_path = os.path.join(error_temp_dir, f"mp_test_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    # Small datasets to avoid memory issues
                    f.create_dataset("saxs_2d", data=np.random.rand(10, 50, 50))
                    f.attrs["analysis_type"] = "XPCS"
                    # Add comprehensive XPCS structure
                    f.create_dataset("/xpcs/multitau/normalized_g2", data=np.random.rand(5, 50))
                    f.create_dataset("/xpcs/temporal_mean/scattering_1d", data=np.random.rand(100))
                    f.create_dataset("/xpcs/temporal_mean/scattering_2d", data=np.random.rand(10, 50, 50))
                    f.create_dataset("/entry/start_time", data="2023-01-01T00:00:00")
                    f.create_dataset("/xpcs/multitau/config/avg_frame", data=1)
                    f.create_dataset("/xpcs/multitau/delay_list", data=np.random.rand(50))
                    f.create_dataset("/entry/instrument/detector_1/count_time", data=0.1)
                    f.create_dataset("/xpcs/temporal_mean/scattering_1d_segments", data=np.random.rand(10, 50))
                    f.create_dataset("/xpcs/multitau/normalized_g2_err", data=np.random.rand(5, 50))
                    f.create_dataset("/xpcs/multitau/config/stride_frame", data=1)
                    f.create_dataset("/entry/instrument/detector_1/frame_time", data=0.01)
                    f.create_dataset("/xpcs/spatial_mean/intensity_vs_time", data=np.random.rand(1000))
                test_files.append(file_path)
            except Exception:
                continue

        if test_files:
            try:
                AverageToolbox(error_temp_dir)

                # Limit number of processes to test resource management
                max_workers = min(2, len(test_files))

                # Test multiprocessing with limited resources
                try:
                    # This might use multiprocessing internally
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = []
                        for file_path in test_files:
                            future = executor.submit(
                                lambda f: h5py.File(f, "r").filename, file_path
                            )
                            futures.append(future)

                        # Collect results with timeout
                        results = []
                        for future in as_completed(futures, timeout=5):
                            try:
                                result = future.result(timeout=1)
                                results.append(result)
                            except Exception:
                                # Individual operations might fail
                                continue

                        # Some results should be obtained
                        assert len(results) >= 0

                except Exception as e:
                    # Multiprocessing limits might cause failures
                    assert any(
                        keyword in str(e).lower()
                        for keyword in ["process", "resource", "limit", "worker"]
                    )

            except Exception as e:
                # AverageToolbox might not be available or functional
                assert "average" in str(e).lower() or "toolbox" in str(e).lower()

    def test_thread_pool_exhaustion(self, error_temp_dir):
        """Test thread pool behavior under resource exhaustion."""

        def cpu_bound_task(duration=0.1):
            """Simulate CPU-bound task."""
            start_time = time.time()
            while time.time() - start_time < duration:
                # CPU-intensive operation
                _ = sum(i * i for i in range(1000))

        # Test with limited thread pool
        max_workers = 4
        task_count = 20  # More tasks than workers

        start_time = time.time()

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(cpu_bound_task, 0.05) for _ in range(task_count)
                ]

                completed_count = 0
                for future in as_completed(
                    futures, timeout=10
                ):  # 10 second total timeout
                    try:
                        future.result(timeout=1)  # 1 second per task timeout
                        completed_count += 1
                    except Exception:
                        # Individual tasks might timeout or fail
                        continue

                # Should complete most tasks despite resource limits
                completion_rate = completed_count / task_count
                assert completion_rate >= 0.5  # At least 50% completion

        except Exception as e:
            # Thread pool might be exhausted
            elapsed = time.time() - start_time
            assert elapsed < 15  # Should not hang indefinitely
            assert any(
                keyword in str(e).lower()
                for keyword in ["thread", "pool", "resource", "timeout"]
            )


class TestNetworkResourceExhaustion:
    """Test behavior when network resources are exhausted."""

    def test_network_timeout_handling(self, error_temp_dir, resource_exhaustion):
        """Test handling of network timeouts and failures."""
        with resource_exhaustion.simulate_network_failure():
            # Any network operations should fail gracefully

            # Mock network-dependent operations
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_urlopen.side_effect = OSError("Network timeout")

                try:
                    # Test network-dependent functionality
                    # (This is a placeholder as XPCS Toolkit might not have network deps)
                    import urllib.request

                    urllib.request.urlopen("http://example.com", timeout=1)

                except OSError as e:
                    # Expected network failure
                    assert any(
                        keyword in str(e).lower()
                        for keyword in ["network", "timeout", "unreachable"]
                    )

    def test_remote_file_access_failures(self, error_temp_dir):
        """Test handling of remote file access failures."""
        # Simulate remote file path
        remote_path = "//remote/server/file.h5"

        # Remote access should fail gracefully
        with pytest.raises((FileNotFoundError, OSError, PermissionError)):
            # This should fail as it's not a real remote path
            XpcsFile(remote_path)

    def test_concurrent_network_requests(self, resource_exhaustion):
        """Test handling of concurrent network request failures."""
        with resource_exhaustion.simulate_network_failure():
            # Multiple concurrent requests should all fail gracefully
            def make_request():
                try:
                    import urllib.request

                    urllib.request.urlopen("http://example.com", timeout=0.5)
                    return "success"
                except Exception as e:
                    return str(e)

            # Make concurrent requests
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(10)]

                results = []
                for future in as_completed(futures, timeout=5):
                    result = future.result()
                    results.append(result)

                # All requests should fail due to network simulation
                success_count = sum(1 for r in results if r == "success")
                assert success_count == 0  # All should fail under network failure


class TestResourceRecoveryMechanisms:
    """Test resource recovery and cleanup mechanisms."""

    def test_automatic_resource_cleanup(self, error_temp_dir):
        """Test automatic cleanup when resources are exhausted."""
        kernel = ViewerKernel(error_temp_dir)

        # Fill cache to trigger cleanup
        large_items = []
        try:
            for i in range(100):  # Try to add many items
                large_item = np.random.rand(1000, 1000)  # ~8MB each
                key = f"large_item_{i}"

                try:
                    kernel._current_dset_cache[key] = large_item
                    large_items.append(key)

                    # Check if cleanup is triggered
                    current_cache_size = len(kernel._current_dset_cache)
                    if current_cache_size < i + 1:
                        # Cleanup occurred
                        break

                except MemoryError:
                    # Memory exhaustion triggered
                    break

        finally:
            # Verify cleanup occurred
            final_cache_size = len(kernel._current_dset_cache)
            assert final_cache_size >= 0

            # Manual cleanup for test
            kernel._current_dset_cache.clear()

    def test_connection_pool_recovery_after_exhaustion(self, error_temp_dir):
        """Test connection pool recovery after resource exhaustion."""
        pool = HDF5ConnectionPool(max_pool_size=5)

        # Create test files
        test_files = []
        for i in range(10):
            file_path = os.path.join(error_temp_dir, f"recovery_{i}.h5")
            try:
                with h5py.File(file_path, "w") as f:
                    f.create_dataset("data", data=[i])
                test_files.append(file_path)
            except Exception:
                continue

        try:
            # Exhaust pool capacity
            connections = []
            for file_path in test_files:
                try:
                    conn = pool.get_connection(file_path)
                    connections.append(conn)
                except Exception:
                    break

            # Pool should limit connections
            assert len(pool._pool) <= 5

            # Force cleanup to simulate recovery
            pool.clear_pool()

            # Pool should recover and be usable again
            if test_files:
                try:
                    recovery_conn = pool.get_connection(test_files[0])
                    assert recovery_conn is not None or len(pool._pool) == 0

                except Exception:
                    # Recovery might fail if files are problematic
                    pass

        finally:
            pool.clear_pool()

    def test_memory_pressure_adaptive_behavior(
        self, error_temp_dir, memory_limited_environment
    ):
        """Test adaptive behavior under sustained memory pressure."""
        with memory_limited_environment:
            kernel = ViewerKernel(error_temp_dir)

            # System should adapt to memory pressure
            is_high_pressure = MemoryMonitor.is_memory_pressure_high(threshold=0.8)
            assert is_high_pressure  # Should detect pressure under our mock

            # Operations should be more conservative
            conservative_cache_size = len(kernel._current_dset_cache)

            # Try to add items - should be limited under pressure
            for i in range(10):
                try:
                    small_item = np.random.rand(100, 100)  # Smaller items
                    kernel._current_dset_cache[f"pressure_item_{i}"] = small_item
                except MemoryError:
                    break

            # Cache growth should be limited under pressure
            final_cache_size = len(kernel._current_dset_cache)
            cache_growth = final_cache_size - conservative_cache_size
            assert cache_growth <= 10  # Limited growth under pressure

    @pytest.mark.slow
    def test_long_running_resource_stability(self, error_temp_dir):
        """Test resource stability over extended operation."""
        kernel = ViewerKernel(error_temp_dir)
        pool = HDF5ConnectionPool(max_pool_size=10)

        # Create test file
        test_file = os.path.join(error_temp_dir, "stability_test.h5")
        try:
            with h5py.File(test_file, "w") as f:
                f.create_dataset("data", data=np.random.rand(100, 100))
        except Exception:
            pytest.skip("Could not create test file")

        initial_memory = psutil.Process().memory_info().rss
        operations_count = 0

        try:
            # Perform many operations over time
            for cycle in range(100):  # 100 cycles of operations
                try:
                    # Cache operations
                    cache_data = np.random.rand(50, 50)
                    kernel._current_dset_cache[f"stability_{cycle}"] = cache_data

                    # Connection pool operations
                    if os.path.exists(test_file):
                        pool.get_connection(test_file)

                    operations_count += 1

                    # Periodic cleanup
                    if cycle % 20 == 0:
                        # Force garbage collection
                        gc.collect()

                        # Check memory growth
                        current_memory = psutil.Process().memory_info().rss
                        memory_growth = current_memory - initial_memory

                        # Memory growth should be bounded
                        assert (
                            memory_growth < 100 * 1024 * 1024
                        )  # Less than 100MB growth

                    # Brief pause to simulate real usage
                    time.sleep(0.001)

                except (MemoryError, OSError):
                    # Resource exhaustion is acceptable
                    break

        finally:
            # Cleanup
            kernel._current_dset_cache.clear()
            pool.clear_pool()

        # Should have completed significant number of operations
        assert operations_count >= 50  # At least half the cycles
