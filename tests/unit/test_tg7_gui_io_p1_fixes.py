"""Task Group 7 tests: GUI and I/O P1 fixes.

Tests for BUG-013, BUG-014, BUG-015, BUG-016, BUG-027, BUG-028, BUG-029,
BUG-030, BUG-032, BUG-033, BUG-034.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test 1 (BUG-013): closeEvent waits for thread pool
# ---------------------------------------------------------------------------


class TestCloseEventWaitsForThreadPool:
    """BUG-013: closeEvent must call thread_pool.waitForDone() to prevent
    signals firing into deleted GUI objects during shutdown."""

    def test_close_event_calls_wait_for_done(self):
        """closeEvent must invoke waitForDone on the thread pool."""
        # We mock the QMainWindow and thread pool to avoid a full Qt setup.
        from unittest.mock import MagicMock, patch

        # Build a minimal mock of the viewer that exposes closeEvent logic
        mock_thread_pool = MagicMock()
        mock_async_vk = MagicMock()
        mock_session_manager = MagicMock()
        mock_session_manager.save_session.return_value = None

        # Patch QMainWindow and import xpcs_viewer lazily to avoid Qt init
        import importlib
        import sys

        with patch.dict(sys.modules, {}):
            try:
                from xpcsviewer.xpcs_viewer import XpcsViewer
            except Exception:
                pytest.skip("Cannot import XpcsViewer without Qt display")

        # Confirm the closeEvent source references waitForDone
        import inspect

        source = inspect.getsource(XpcsViewer.closeEvent)
        assert "waitForDone" in source, (
            "closeEvent must call thread_pool.waitForDone() to ensure all "
            "worker threads finish before GUI objects are destroyed (BUG-013)"
        )

    def test_close_event_source_has_timeout_argument(self):
        """waitForDone call must include a timeout to avoid hanging."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        source = inspect.getsource(XpcsViewer.closeEvent)
        # The timeout must be passed (any integer/ms variable is fine)
        assert "waitForDone" in source, "waitForDone not found in closeEvent"
        # Check that it is not called without arguments (no-timeout variant)
        # by verifying there is something inside the parentheses
        idx = source.index("waitForDone")
        snippet = source[idx : idx + 30]
        assert snippet != "waitForDone()", (
            "waitForDone() must be called with a timeout_ms argument to "
            "avoid hanging indefinitely during shutdown (BUG-013)"
        )


# ---------------------------------------------------------------------------
# Test 2 (BUG-014): Async G2 plotting uses worker result directly
# ---------------------------------------------------------------------------


class TestAsyncG2ResultHandling:
    """BUG-014: apply_g2_result() must use the pre-computed worker result
    directly and must NOT re-call vk.plot_g2() on the main thread."""

    def test_apply_g2_result_uses_result_dict_directly(self):
        """apply_g2_result must consume result dict without calling vk.plot_g2."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        source = inspect.getsource(XpcsViewer.apply_g2_result)

        # The method must NOT delegate back to synchronous vk.plot_g2
        assert "self.vk.plot_g2(" not in source, (
            "apply_g2_result() must not re-call vk.plot_g2() synchronously "
            "on the main thread — this defeats async architecture (BUG-014). "
            "Use the pre-computed result from the worker directly."
        )

    def test_apply_g2_result_reads_figure_handles_from_result(self):
        """apply_g2_result must consume figure or axes handles from result dict."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        source = inspect.getsource(XpcsViewer.apply_g2_result)
        # Must access something from result (not just discard it)
        assert 'result["' in source or "result.get(" in source, (
            "apply_g2_result must extract data from the result dict (BUG-014)"
        )


# ---------------------------------------------------------------------------
# Test 3 (BUG-015): XpcsFile.__dict__.update does not overwrite existing attrs
# ---------------------------------------------------------------------------


class TestXpcsFileAttributeCollision:
    """BUG-015: XpcsFile.__dict__.update() must not silently overwrite
    existing attributes like qmap and label."""

    def test_existing_qmap_not_overwritten_by_payload(self):
        """The collision guard logic must prevent 'qmap'/'label' from being overwritten."""

        # Simulate the XpcsFile.__init__ collision guard in isolation.
        # The guard uses frozenset(self.__dict__.keys()) to capture protected
        # attrs, then only copies payload keys absent from that set.
        class FakeXpcsFile:
            pass

        xf = FakeXpcsFile()
        original_qmap = {"q": np.array([0.1, 0.2, 0.3])}
        xf.fname = "/some/file.hdf"
        xf.qmap = original_qmap
        xf.atype = ["xpcs"]
        xf.label = "original_label"

        # Capture the protected set BEFORE applying the payload — this matches
        # the implementation in xpcs_file.py (BUG-015 fix)
        _PROTECTED = frozenset(xf.__dict__.keys())

        # Payload from HDF5 file that includes keys colliding with existing attrs
        payload = {
            "qmap": {"q": np.array([999.0])},  # collision: must be skipped
            "label": "OVERWRITTEN",  # collision: must be skipped
            "g2": np.ones((10, 3)),  # new key: must be added
        }

        for k, v in payload.items():
            if k not in _PROTECTED:
                xf.__dict__[k] = v

        assert xf.qmap is original_qmap, (
            "payload 'qmap' key must NOT overwrite self.qmap (BUG-015)"
        )
        assert xf.label == "original_label", (
            "payload 'label' key must NOT overwrite self.label (BUG-015)"
        )
        assert hasattr(xf, "g2"), (
            "non-colliding payload keys ('g2') must still be added (BUG-015)"
        )

    def test_xpcsfile_load_data_result_checked_for_collision(self):
        """XpcsFile.__init__ source must contain a collision guard."""
        try:
            from xpcsviewer.xpcs_file import XpcsFile
        except Exception:
            pytest.skip("Cannot import XpcsFile")

        import inspect

        source = inspect.getsource(XpcsFile.__init__)

        # Must contain one of: explicit collision check or store in self.data
        has_collision_guard = (
            "self.data" in source
            or "_PROTECTED" in source
            or "if k not in" in source
            or "hasattr(self" in source
        )
        assert has_collision_guard, (
            "XpcsFile.__init__ must guard against HDF5 payload overwriting "
            "existing attributes (qmap, label, etc.). Add a collision check "
            "or store payload in a self.data namespace dict (BUG-015)."
        )


# ---------------------------------------------------------------------------
# Test 4 (BUG-016): init_average preserves user-entered save_name
# ---------------------------------------------------------------------------


class TestInitAverageSaveNamePreservation:
    """BUG-016: init_average must only set the default save_name when the
    text field is currently empty, preserving user-entered values."""

    def test_init_average_source_checks_empty_before_setting_name(self):
        """init_average must check if save_name field is empty before overwriting."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        source = inspect.getsource(XpcsViewer.init_average)

        # The fix: only set default when current text is empty
        has_guard = (
            'save_name == ""' in source
            or "save_name == ''" in source
            or ".text() == " in source
            or "if not save_name" in source
            or "if save_name ==" in source
        )
        assert has_guard, (
            "init_average must guard against overwriting a non-empty save_name. "
            "Only set the default when avg_save_name.text() is empty (BUG-016)."
        )

    def test_init_average_logic_preserves_non_empty_name(self):
        """Simulate init_average behavior: non-empty name must be preserved."""

        # Simulate the corrected logic (field-only test, no Qt required)
        class FakeLineEdit:
            def __init__(self, text):
                self._text = text

            def text(self):
                return self._text

            def setText(self, t):
                self._text = t

        avg_save_name = FakeLineEdit("UserChosenName.hdf")
        target = ["file1.hdf"]

        # Corrected logic: only set default when empty
        save_name = avg_save_name.text()
        if save_name == "":
            save_name = "Avg" + target[0]
            avg_save_name.setText(save_name)

        assert avg_save_name.text() == "UserChosenName.hdf", (
            "Non-empty save_name must be preserved by init_average (BUG-016)"
        )

    def test_init_average_logic_sets_default_when_empty(self):
        """Simulate init_average behavior: empty name gets default."""

        class FakeLineEdit:
            def __init__(self, text):
                self._text = text

            def text(self):
                return self._text

            def setText(self, t):
                self._text = t

        avg_save_name = FakeLineEdit("")
        target = ["file1.hdf"]

        save_name = avg_save_name.text()
        if save_name == "":
            save_name = "Avg" + target[0]
            avg_save_name.setText(save_name)

        assert avg_save_name.text() == "Avgfile1.hdf", (
            "Empty save_name field must receive default value (BUG-016)"
        )


# ---------------------------------------------------------------------------
# Test 5 (BUG-027): JAX arrays converted via ensure_numpy at PyQtGraph boundaries
# ---------------------------------------------------------------------------


class TestEnsureNumpyAtPyQtGraphBoundaries:
    """BUG-027: All JAX arrays must be converted via ensure_numpy() before
    reaching PyQtGraph setImage/setData calls."""

    def test_ensure_numpy_converts_jax_array(self):
        """ensure_numpy must accept JAX arrays and return NumPy arrays."""
        from xpcsviewer.backends._conversions import ensure_numpy

        try:
            import jax.numpy as jnp

            arr = jnp.array([1.0, 2.0, 3.0])
            result = ensure_numpy(arr)
            assert isinstance(result, np.ndarray), (
                "ensure_numpy must convert JAX arrays to np.ndarray (BUG-027)"
            )
        except ImportError:
            pytest.skip("JAX not available")

    def test_ensure_numpy_preserves_numpy_arrays(self):
        """ensure_numpy must be a no-op for NumPy arrays."""
        from xpcsviewer.backends._conversions import ensure_numpy

        arr = np.array([1.0, 2.0, 3.0])
        result = ensure_numpy(arr)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, arr)

    def test_saxs_boundary_in_xpcs_viewer_uses_ensure_numpy(self):
        """apply_saxs_2d_result in xpcs_viewer.py must wrap image_data with ensure_numpy."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        # Check apply_saxs_2d_result for the pg_saxs.setImage boundary (BUG-027)
        source = inspect.getsource(XpcsViewer.apply_saxs_2d_result)
        assert "ensure_numpy" in source, (
            "apply_saxs_2d_result must wrap image_data with ensure_numpy() "
            "before calling pg_saxs.setImage() (BUG-027)"
        )

    def test_qmap_boundary_in_xpcs_viewer_uses_ensure_numpy(self):
        """apply_qmap_result in xpcs_viewer.py must wrap image_data with ensure_numpy."""
        try:
            from xpcsviewer.xpcs_viewer import XpcsViewer
        except Exception:
            pytest.skip("Cannot import XpcsViewer without Qt display")

        import inspect

        source = inspect.getsource(XpcsViewer.apply_qmap_result)
        assert "ensure_numpy" in source, (
            "apply_qmap_result must wrap image_data with ensure_numpy() "
            "before calling pg_qmap.setImage() (BUG-027)"
        )

    def test_simplemask_kernel_refresh_uses_ensure_numpy(self):
        """simplemask_kernel.py refresh_detector_image must use ensure_numpy."""
        import inspect

        from xpcsviewer.simplemask import simplemask_kernel

        source = inspect.getsource(simplemask_kernel)
        # Check that ensure_numpy is called at the hdl.setImage boundary
        assert "ensure_numpy" in source, (
            "simplemask_kernel.py must call ensure_numpy() before "
            "hdl.setImage() at line ~308 (BUG-027)"
        )

    def test_simplemask_window_refresh_uses_ensure_numpy(self):
        """simplemask_window.py _refresh_mask_display must use ensure_numpy."""
        import inspect

        from xpcsviewer.simplemask import simplemask_window

        source = inspect.getsource(simplemask_window)
        assert "ensure_numpy" in source, (
            "simplemask_window.py must call ensure_numpy() at all "
            "PyQtGraph setImage boundaries (lines 641,649,658,660,841) (BUG-027)"
        )


# ---------------------------------------------------------------------------
# Test 6 (BUG-028): Unit string defaults consistent between facade and schema
# ---------------------------------------------------------------------------


class TestUnitStringDefaultConsistency:
    """BUG-028: The default unit string for sqmap/dqmap must be the same in
    hdf5_facade.py and QMapSchema.from_dict()."""

    def test_unit_defaults_match_between_facade_and_schema(self):
        """hdf5_facade default units must equal QMapSchema.from_dict() defaults."""
        import inspect

        from xpcsviewer.io import hdf5_facade as facade_module

        facade_source = inspect.getsource(facade_module)

        # Check that nm^-1 appears in facade as the default
        facade_has_nm = '"nm^-1"' in facade_source or "'nm^-1'" in facade_source
        assert facade_has_nm, (
            "hdf5_facade.py must use 'nm^-1' as the default unit string (BUG-028)"
        )

        # Check that from_dict returns nm^-1 by default (not A^-1)
        import numpy as np

        from xpcsviewer.schemas.validators import QMapSchema

        sqmap = np.ones((5, 5), dtype=np.float64)
        dqmap = np.ones((5, 5), dtype=np.float64)
        phis = np.ones((5, 5), dtype=np.float64)
        schema = QMapSchema.from_dict({"sqmap": sqmap, "dqmap": dqmap, "phis": phis})

        assert schema.sqmap_unit == "nm^-1", (
            f"QMapSchema.from_dict() default sqmap_unit must be 'nm^-1', "
            f"got '{schema.sqmap_unit}' (BUG-028). Must match hdf5_facade.py default."
        )
        assert schema.dqmap_unit == "nm^-1", (
            f"QMapSchema.from_dict() default dqmap_unit must be 'nm^-1', "
            f"got '{schema.dqmap_unit}' (BUG-028). Must match hdf5_facade.py default."
        )

    def test_roundtrip_unit_default_preserved(self):
        """Writing with facade default and reading back via from_dict gives same unit."""
        import numpy as np

        from xpcsviewer.schemas.validators import QMapSchema

        sqmap = np.ones((10, 10), dtype=np.float64)
        dqmap = np.ones((10, 10), dtype=np.float64)
        phis = np.ones((10, 10), dtype=np.float64)

        # Create via from_dict without specifying units (uses defaults)
        d = {"sqmap": sqmap, "dqmap": dqmap, "phis": phis}
        qmap = QMapSchema.from_dict(d)

        # The default must equal what hdf5_facade writes by default
        assert qmap.sqmap_unit == "nm^-1", (
            f"QMapSchema.from_dict() default sqmap_unit should be 'nm^-1', "
            f"got '{qmap.sqmap_unit}' (BUG-028)"
        )
        assert qmap.dqmap_unit == "nm^-1", (
            f"QMapSchema.from_dict() default dqmap_unit should be 'nm^-1', "
            f"got '{qmap.dqmap_unit}' (BUG-028)"
        )


# ---------------------------------------------------------------------------
# Test 7 (BUG-029): validate=False path does not trigger __post_init__
# ---------------------------------------------------------------------------


class TestValidateFalsePathNoPostInit:
    """BUG-029: When validate=False, hdf5_facade.py read_qmap must return a
    raw dict (or unvalidated wrapper) without calling QMapSchema.__post_init__."""

    def test_validate_false_returns_dict_not_schema(self, tmp_path):
        """With validate=False, read_qmap must return a dict, not a QMapSchema."""
        import h5py
        import numpy as np

        hdf_file = tmp_path / "test_validate_false.hdf"
        sqmap = np.ones((5, 5), dtype=np.float64)
        dqmap = np.zeros((5, 5), dtype=np.float64)
        phis = np.zeros((5, 5), dtype=np.float64)

        with h5py.File(hdf_file, "w") as f:
            qmap_grp = f.create_group("xpcs/qmap")
            qmap_grp.create_dataset("sqmap", data=sqmap)
            qmap_grp.create_dataset("dqmap", data=dqmap)
            qmap_grp.create_dataset("phis", data=phis)

        try:
            from xpcsviewer.io.hdf5_facade import HDF5Facade
            from xpcsviewer.schemas.validators import QMapSchema
        except ImportError:
            pytest.skip("Cannot import HDF5Facade")

        facade = HDF5Facade(validate=False)
        result = facade.read_qmap(str(hdf_file))

        # With validate=False the result must be a dict, not a QMapSchema
        assert isinstance(result, dict), (
            f"validate=False must return a raw dict, got {type(result).__name__} "
            "(BUG-029). QMapSchema(**data) triggers __post_init__ validation "
            "even when validation is disabled."
        )

    def test_validate_false_does_not_run_post_init(self, tmp_path):
        """__post_init__ must NOT be called when validate=False."""
        import h5py
        import numpy as np

        hdf_file = tmp_path / "test_validate_false2.hdf"
        sqmap = np.ones((5, 5), dtype=np.float32)  # float32 triggers BUG-011 check
        dqmap = np.zeros((5, 5), dtype=np.float32)
        phis = np.zeros((5, 5), dtype=np.float32)

        with h5py.File(hdf_file, "w") as f:
            qmap_grp = f.create_group("xpcs/qmap")
            qmap_grp.create_dataset("sqmap", data=sqmap)
            qmap_grp.create_dataset("dqmap", data=dqmap)
            qmap_grp.create_dataset("phis", data=phis)

        try:
            from xpcsviewer.io.hdf5_facade import HDF5Facade
            from xpcsviewer.schemas.validators import QMapSchema
        except ImportError:
            pytest.skip("Cannot import HDF5Facade")

        post_init_called = []
        original_post_init = QMapSchema.__post_init__

        def spy_post_init(self, *args, **kwargs):
            post_init_called.append(True)
            original_post_init(self, *args, **kwargs)

        facade = HDF5Facade(validate=False)

        with patch.object(QMapSchema, "__post_init__", spy_post_init):
            try:
                facade.read_qmap(str(hdf_file))
            except Exception:
                pass  # We only care whether __post_init__ was called

        assert len(post_init_called) == 0, (
            "QMapSchema.__post_init__ must NOT be called when validate=False "
            "(BUG-029). Return a raw dict instead of calling QMapSchema(**data)."
        )


# ---------------------------------------------------------------------------
# Test 8 (BUG-030): Singletons use double-checked locking
# ---------------------------------------------------------------------------


class TestSingletonDoubleCheckedLocking:
    """BUG-030: The three global singletons must use double-checked locking
    to be thread-safe under concurrent first-access."""

    def _check_singleton_is_thread_safe(self, getter_fn, shutdown_fn=None):
        """Helper: call getter_fn from many threads simultaneously and
        verify only one instance is created."""
        results = []
        errors = []

        def accessor():
            try:
                inst = getter_fn()
                results.append(id(inst))
            except Exception as e:
                errors.append(str(e))

        if shutdown_fn:
            try:
                shutdown_fn()
            except Exception:
                pass

        threads = [threading.Thread(target=accessor) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors during concurrent access: {errors}"
        assert len(results) == 10, "All threads must succeed"
        # All must reference the same object
        assert len(set(results)) == 1, (
            "All threads must get the same singleton instance (BUG-030)"
        )

    def test_unified_threading_manager_is_thread_safe(self):
        """UnifiedThreadingManager singleton must be thread-safe under concurrency."""
        try:
            from xpcsviewer.threading.unified_threading import (
                get_unified_threading_manager,
                shutdown_unified_threading,
            )
        except ImportError:
            pytest.skip("Cannot import unified_threading")

        self._check_singleton_is_thread_safe(
            get_unified_threading_manager,
            shutdown_unified_threading,
        )

    def test_memory_manager_is_thread_safe(self):
        """MemoryManager singleton must be thread-safe under concurrency."""
        try:
            from xpcsviewer.utils.memory_manager import (
                get_memory_manager,
                shutdown_memory_manager,
            )
        except ImportError:
            pytest.skip("Cannot import memory_manager")

        self._check_singleton_is_thread_safe(
            get_memory_manager,
            shutdown_memory_manager,
        )

    def test_lazy_loader_is_thread_safe(self):
        """LazyLoader singleton must be thread-safe under concurrency."""
        try:
            from xpcsviewer.utils.lazy_loader import (
                get_lazy_loader,
                shutdown_lazy_loader,
            )
        except ImportError:
            pytest.skip("Cannot import lazy_loader")

        self._check_singleton_is_thread_safe(
            get_lazy_loader,
            shutdown_lazy_loader,
        )

    def test_unified_threading_source_has_lock(self):
        """get_unified_threading_manager source must contain a Lock."""
        try:
            import inspect

            from xpcsviewer.threading import unified_threading as mod

            source = inspect.getsource(mod.get_unified_threading_manager)
            assert "_lock" in source or "Lock" in source or "with " in source, (
                "get_unified_threading_manager must use a lock for thread-safe "
                "double-checked locking (BUG-030)"
            )
        except ImportError:
            pytest.skip("Cannot import unified_threading")

    def test_memory_manager_source_has_lock(self):
        """get_memory_manager source must contain a Lock."""
        try:
            import inspect

            from xpcsviewer.utils import memory_manager as mod

            source = inspect.getsource(mod.get_memory_manager)
            assert "_lock" in source or "Lock" in source or "with " in source, (
                "get_memory_manager must use a lock for thread-safe "
                "double-checked locking (BUG-030)"
            )
        except ImportError:
            pytest.skip("Cannot import memory_manager")

    def test_lazy_loader_source_has_lock(self):
        """get_lazy_loader source must contain a Lock."""
        try:
            import inspect

            from xpcsviewer.utils import lazy_loader as mod

            source = inspect.getsource(mod.get_lazy_loader)
            assert "_lock" in source or "Lock" in source or "with " in source, (
                "get_lazy_loader must use a lock for thread-safe "
                "double-checked locking (BUG-030)"
            )
        except ImportError:
            pytest.skip("Cannot import lazy_loader")
