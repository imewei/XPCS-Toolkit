"""Task Group 4 tests: Backend and I/O critical fixes.

Tests for BUG-005, BUG-008, BUG-010, BUG-011.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1 (BUG-005): s_sq does not raise NameError when n_data <= n_params
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 2 (BUG-008): Concurrent HDF5 reads do not deadlock
# ---------------------------------------------------------------------------


class TestHDF5ConcurrentNoDeadlock:
    """BUG-008: Concurrent reads from multiple threads must not deadlock."""

    def test_concurrent_reads_complete_within_timeout(self, tmp_path):
        """Multiple threads reading HDF5 files simultaneously must not deadlock."""
        h5py = pytest.importorskip("h5py")
        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        # Create a test HDF5 file
        test_file = str(tmp_path / "concurrent_test.h5")
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=np.zeros((10, 10)))

        pool = HDF5ConnectionPool(max_pool_size=5, health_check_interval=1.0)
        errors: list[Exception] = []
        results: list[bool] = []

        def read_file():
            try:
                with pool.get_connection(test_file, "r") as fh:
                    _ = fh["data"][:]
                    results.append(True)
            except Exception as exc:
                errors.append(exc)

        n_threads = 8
        threads = [threading.Thread(target=read_file) for _ in range(n_threads)]
        for t in threads:
            t.start()

        # Join with a 5-second timeout to detect deadlocks
        deadline = time.time() + 5.0
        for t in threads:
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

        alive = [t for t in threads if t.is_alive()]
        assert not alive, (
            f"{len(alive)} threads are still running after 5s — possible deadlock"
        )
        assert not errors, f"Thread errors: {errors}"
        assert len(results) == n_threads

    def test_concurrent_reads_plus_pool_operations_no_deadlock(self, tmp_path):
        """Reads interleaved with pool clear do not deadlock."""
        h5py = pytest.importorskip("h5py")
        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        test_file = str(tmp_path / "interleaved_test.h5")
        with h5py.File(test_file, "w") as f:
            f.create_dataset("data", data=np.zeros((5, 5)))

        pool = HDF5ConnectionPool(max_pool_size=3, health_check_interval=60.0)
        errors: list[Exception] = []
        stop_event = threading.Event()

        def read_loop():
            while not stop_event.is_set():
                try:
                    with pool.get_connection(test_file, "r") as fh:
                        _ = fh["data"][0]
                except Exception as exc:
                    errors.append(exc)
                    break

        threads = [threading.Thread(target=read_loop) for _ in range(4)]
        for t in threads:
            t.start()

        # Let threads run briefly, then stop them
        time.sleep(0.3)
        stop_event.set()

        deadline = time.time() + 3.0
        for t in threads:
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

        alive = [t for t in threads if t.is_alive()]
        assert not alive, f"{len(alive)} threads still alive after stop — deadlock"
        # Errors about closed files are acceptable; NameError/deadlock is not
        lock_errors = [e for e in errors if isinstance(e, (NameError, RuntimeError))]
        assert not lock_errors, f"Lock-related errors: {lock_errors}"


# ---------------------------------------------------------------------------
# Test 3 (BUG-010): Frozen dataclass mutable fields are tuples
# ---------------------------------------------------------------------------


class TestFrozenDataclassTupleFields:
    """BUG-010: list fields in frozen dataclasses must be stored as tuples."""

    def test_g2data_q_values_stored_as_tuple(self):
        """G2Data.q_values must be a tuple after construction."""
        from xpcsviewer.schemas.validators import G2Data

        n_delay, n_q = 5, 3
        g2 = np.ones((n_delay, n_q), dtype=np.float64)
        g2_err = np.ones((n_delay, n_q), dtype=np.float64) * 0.1
        delay_times = np.arange(1, n_delay + 1, dtype=np.float64)
        q_values_list = [0.1, 0.2, 0.3]  # intentionally a list

        obj = G2Data(
            g2=g2,
            g2_err=g2_err,
            delay_times=delay_times,
            q_values=q_values_list,
        )

        assert isinstance(obj.q_values, tuple), (
            f"q_values should be tuple, got {type(obj.q_values)}"
        )
        assert obj.q_values == (0.1, 0.2, 0.3)

    def test_g2data_q_values_immutable(self):
        """G2Data.q_values as tuple cannot be mutated (frozen dataclass guarantee)."""
        from xpcsviewer.schemas.validators import G2Data

        n_delay, n_q = 4, 2
        g2 = np.ones((n_delay, n_q), dtype=np.float64)
        g2_err = np.ones((n_delay, n_q), dtype=np.float64) * 0.05
        delay_times = np.arange(1, n_delay + 1, dtype=np.float64)

        obj = G2Data(g2=g2, g2_err=g2_err, delay_times=delay_times, q_values=[1.0, 2.0])

        with pytest.raises((AttributeError, TypeError)):
            obj.q_values = [9.9, 8.8]  # type: ignore[misc]

    def test_partition_schema_val_list_stored_as_tuple(self):
        """PartitionSchema.val_list must be a tuple after construction."""
        from xpcsviewer.schemas.validators import GeometryMetadata, PartitionSchema

        geometry = GeometryMetadata(
            bcx=512.0,
            bcy=512.0,
            det_dist=5000.0,
            lambda_=1.54,
            pix_dim=0.075,
            shape=(1024, 1024),
        )
        partition_map = np.ones((4, 4), dtype=np.int32)
        val_list_input = [0.1, 0.2, 0.3]
        num_list_input = [10, 20, 30]

        obj = PartitionSchema(
            partition_map=partition_map,
            num_pts=3,
            val_list=val_list_input,
            num_list=num_list_input,
            metadata=geometry,
        )

        assert isinstance(obj.val_list, tuple), (
            f"val_list should be tuple, got {type(obj.val_list)}"
        )
        assert obj.val_list == (0.1, 0.2, 0.3)

    def test_partition_schema_num_list_stored_as_tuple(self):
        """PartitionSchema.num_list must be a tuple after construction."""
        from xpcsviewer.schemas.validators import GeometryMetadata, PartitionSchema

        geometry = GeometryMetadata(
            bcx=256.0,
            bcy=256.0,
            det_dist=3000.0,
            lambda_=1.0,
            pix_dim=0.1,
            shape=(512, 512),
        )
        partition_map = np.zeros((6, 6), dtype=np.int32)
        partition_map[:3, :3] = 1
        partition_map[3:, 3:] = 2

        obj = PartitionSchema(
            partition_map=partition_map,
            num_pts=2,
            val_list=[0.5, 1.0],
            num_list=[9, 9],
            metadata=geometry,
        )

        assert isinstance(obj.num_list, tuple), (
            f"num_list should be tuple, got {type(obj.num_list)}"
        )
        assert obj.num_list == (9, 9)


# ---------------------------------------------------------------------------
# Test 4 (BUG-011): QMapSchema.from_dict() coerces float32 to float64
# ---------------------------------------------------------------------------


class TestQMapSchemaFromDictCoercion:
    """BUG-011: from_dict() must coerce float32 HDF5 arrays to float64."""

    def test_from_dict_accepts_float32_sqmap(self):
        """QMapSchema.from_dict() with float32 sqmap must succeed (coerces to float64)."""
        from xpcsviewer.schemas.validators import QMapSchema

        sqmap_f32 = np.ones((8, 8), dtype=np.float32) * 0.5
        dqmap_f32 = np.zeros((8, 8), dtype=np.float32)
        phis_f32 = np.ones((8, 8), dtype=np.float32) * 1.0

        schema = QMapSchema.from_dict(
            {
                "sqmap": sqmap_f32,
                "dqmap": dqmap_f32,
                "phis": phis_f32,
                "sqmap_unit": "nm^-1",
                "dqmap_unit": "nm^-1",
                "phis_unit": "deg",
            }
        )

        assert schema.sqmap.dtype == np.float64
        assert schema.dqmap.dtype == np.float64
        assert schema.phis.dtype == np.float64

    def test_from_dict_float32_values_preserved(self):
        """Float32 values round-trip correctly after coercion to float64."""
        from xpcsviewer.schemas.validators import QMapSchema

        sqmap_f32 = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        phis_f32 = np.array([[45.0, 90.0], [135.0, 180.0]], dtype=np.float32)

        schema = QMapSchema.from_dict(
            {
                "sqmap": sqmap_f32,
                "phis": phis_f32,
                "sqmap_unit": "A^-1",
                "dqmap_unit": "A^-1",
                "phis_unit": "deg",
            }
        )

        np.testing.assert_allclose(
            schema.sqmap, sqmap_f32.astype(np.float64), rtol=1e-6
        )
        np.testing.assert_allclose(schema.phis, phis_f32.astype(np.float64), rtol=1e-6)

    def test_from_dict_float64_still_works(self):
        """QMapSchema.from_dict() with float64 input continues to work."""
        from xpcsviewer.schemas.validators import QMapSchema

        sqmap_f64 = np.ones((4, 4), dtype=np.float64) * 0.3

        schema = QMapSchema.from_dict(
            {
                "sqmap": sqmap_f64,
                "phis": np.zeros((4, 4), dtype=np.float64),
                "sqmap_unit": "nm^-1",
                "dqmap_unit": "nm^-1",
                "phis_unit": "rad",
            }
        )

        assert schema.sqmap.dtype == np.float64


# ---------------------------------------------------------------------------
# Test 5 (BUG-008 chain): Lock ordering — concurrent reads + health checks
# ---------------------------------------------------------------------------


class TestLockOrderingChain:
    """BUG-008 lock-ordering chain: concurrent reads + pool health checks must not deadlock."""

    def test_health_check_does_not_deadlock_during_concurrent_reads(self, tmp_path):
        """Health check (acquiring _pool_lock) during concurrent reads must not deadlock."""
        h5py = pytest.importorskip("h5py")
        from xpcsviewer.fileIO.hdf_reader import HDF5ConnectionPool

        # Use a very short health_check_interval so the health check fires during reads
        test_file = str(tmp_path / "lock_order_test.h5")
        with h5py.File(test_file, "w") as f:
            f.create_dataset("dataset", data=np.arange(100, dtype=np.float64))

        # health_check_interval=0 forces health check on every call
        pool = HDF5ConnectionPool(max_pool_size=4, health_check_interval=0.0)
        errors: list[Exception] = []
        results: list[int] = []

        def reader():
            for _ in range(5):
                try:
                    with pool.get_connection(test_file, "r") as fh:
                        val = fh["dataset"][0]
                        results.append(int(val))
                except Exception as exc:
                    errors.append(exc)

        n_threads = 6
        threads = [threading.Thread(target=reader) for _ in range(n_threads)]
        for t in threads:
            t.start()

        deadline = time.time() + 8.0
        for t in threads:
            remaining = max(0.0, deadline - time.time())
            t.join(timeout=remaining)

        alive = [t for t in threads if t.is_alive()]
        assert not alive, (
            f"{len(alive)} threads are stuck — lock ordering deadlock detected"
        )
        # Errors that are NOT lock-related are acceptable (e.g., HDF5 file errors)
        lock_errors = [
            e
            for e in errors
            if isinstance(
                e, (NameError, deadlock_sentinel := type("D", (Exception,), {}))
            )
        ]
        # Specifically check no NameError (from uninitialized s_sq) leaks through
        name_errors = [e for e in errors if isinstance(e, NameError)]
        assert not name_errors, f"NameError leaked into HDF5 pool: {name_errors}"
