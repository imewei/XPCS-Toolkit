"""Unit tests for ViewerKernel class.

This module provides comprehensive unit tests for the ViewerKernel class,
covering backend coordination, analysis orchestration, and file management.
"""

import weakref
from unittest.mock import Mock, patch

import pytest

from xpcs_toolkit.file_locator import FileLocator
from xpcs_toolkit.helper.listmodel import TableDataModel
from xpcs_toolkit.module.average_toolbox import AverageToolbox
from xpcs_toolkit.viewer_kernel import ViewerKernel


class TestViewerKernelInit:
    """Test suite for ViewerKernel initialization."""

    def test_init_basic(self, temp_dir):
        """Test basic ViewerKernel initialization."""
        kernel = ViewerKernel(temp_dir)

        # Check inheritance from FileLocator
        assert isinstance(kernel, FileLocator)

        # Check initialization attributes
        assert kernel.path == temp_dir
        assert kernel.statusbar is None
        assert kernel.meta is not None
        assert isinstance(kernel.avg_tb, AverageToolbox)
        assert isinstance(kernel.avg_worker, TableDataModel)
        assert kernel.avg_jid == 0
        assert isinstance(kernel.avg_worker_active, dict)

        # Check memory management attributes
        assert hasattr(kernel, "_current_dset_cache")
        assert isinstance(kernel._current_dset_cache, weakref.WeakValueDictionary)
        assert isinstance(kernel._plot_kwargs_record, dict)
        assert kernel._memory_cleanup_threshold == 0.8

    def test_init_with_statusbar(self, temp_dir):
        """Test ViewerKernel initialization with statusbar."""
        mock_statusbar = Mock()
        kernel = ViewerKernel(temp_dir, statusbar=mock_statusbar)

        assert kernel.statusbar is mock_statusbar
        assert kernel.path == temp_dir

    def test_init_calls_reset_meta(self, temp_dir):
        """Test that initialization calls reset_meta."""
        with patch.object(ViewerKernel, "reset_meta") as mock_reset:
            ViewerKernel(temp_dir)
            mock_reset.assert_called_once()


class TestViewerKernelMetaManagement:
    """Test suite for ViewerKernel metadata management."""

    def test_reset_meta_structure(self, temp_dir):
        """Test reset_meta creates correct metadata structure."""
        kernel = ViewerKernel(temp_dir)
        kernel.reset_meta()

        # Check essential meta keys exist
        expected_keys = [
            "saxs1d_bkg_fname",
            "saxs1d_bkg_xf",
        ]

        for key in expected_keys:
            assert key in kernel.meta

        # Check initial values
        assert kernel.meta["saxs1d_bkg_fname"] is None
        assert kernel.meta["saxs1d_bkg_xf"] is None

    def test_reset_meta_preserves_type(self, temp_dir):
        """Test that reset_meta maintains dictionary type."""
        kernel = ViewerKernel(temp_dir)

        # Modify meta
        kernel.meta["custom_key"] = "custom_value"

        # Reset should create new dict
        kernel.reset_meta()

        assert isinstance(kernel.meta, dict)
        assert "custom_key" not in kernel.meta
        assert kernel.meta["saxs1d_bkg_fname"] is None


class TestViewerKernelMemoryManagement:
    """Test suite for ViewerKernel memory management features."""

    def test_current_dset_cache_initialization(self, temp_dir):
        """Test that current_dset_cache is properly initialized."""
        kernel = ViewerKernel(temp_dir)

        assert isinstance(kernel._current_dset_cache, weakref.WeakValueDictionary)
        assert len(kernel._current_dset_cache) == 0

    def test_plot_kwargs_record_initialization(self, temp_dir):
        """Test that plot_kwargs_record is properly initialized."""
        kernel = ViewerKernel(temp_dir)

        assert isinstance(kernel._plot_kwargs_record, dict)
        assert len(kernel._plot_kwargs_record) == 0

    def test_memory_cleanup_threshold(self, temp_dir):
        """Test memory cleanup threshold setting."""
        kernel = ViewerKernel(temp_dir)

        assert kernel._memory_cleanup_threshold == 0.8

        # Test threshold can be modified
        kernel._memory_cleanup_threshold = 0.9
        assert kernel._memory_cleanup_threshold == 0.9


class TestViewerKernelAverageToolbox:
    """Test suite for ViewerKernel average toolbox integration."""

    @patch("xpcs_toolkit.viewer_kernel._get_module")
    def test_average_toolbox_initialization(self, mock_get_module, temp_dir):
        """Test AverageToolbox initialization."""
        mock_avg_tb_class = Mock()
        mock_avg_tb = Mock()
        mock_avg_tb_class.return_value = mock_avg_tb
        mock_get_module.return_value = mock_avg_tb_class

        kernel = ViewerKernel(temp_dir)

        mock_get_module.assert_called_with("average_toolbox")
        mock_avg_tb_class.assert_called_once_with(temp_dir)
        assert kernel.avg_tb is mock_avg_tb

    @patch("xpcs_toolkit.viewer_kernel.TableDataModel")
    def test_avg_worker_initialization(self, mock_table_model_class, temp_dir):
        """Test average worker model initialization."""
        mock_model = Mock()
        mock_table_model_class.return_value = mock_model

        kernel = ViewerKernel(temp_dir)

        mock_table_model_class.assert_called_once()
        assert kernel.avg_worker is mock_model
        assert kernel.avg_jid == 0
        assert isinstance(kernel.avg_worker_active, dict)


class TestViewerKernelInheritance:
    """Test suite for ViewerKernel inheritance from FileLocator."""

    def test_inherits_from_file_locator(self, temp_dir):
        """Test that ViewerKernel properly inherits from FileLocator."""
        kernel = ViewerKernel(temp_dir)

        assert isinstance(kernel, FileLocator)
        assert hasattr(kernel, "path")  # FileLocator attribute

    @patch("xpcs_toolkit.viewer_kernel.FileLocator.__init__")
    def test_calls_super_init(self, mock_super_init, temp_dir):
        """Test that ViewerKernel calls parent __init__."""
        ViewerKernel(temp_dir)

        mock_super_init.assert_called_once_with(temp_dir)


class TestViewerKernelProperties:
    """Test suite for ViewerKernel properties and attributes."""

    def test_path_property(self, temp_dir):
        """Test path property access."""
        kernel = ViewerKernel(temp_dir)

        assert kernel.path == temp_dir

        # Path should be modifiable
        new_path = "/new/test/path"
        kernel.path = new_path
        assert kernel.path == new_path

    def test_statusbar_property(self, temp_dir):
        """Test statusbar property access."""
        mock_statusbar = Mock()
        kernel = ViewerKernel(temp_dir, statusbar=mock_statusbar)

        assert kernel.statusbar is mock_statusbar

        # Statusbar should be modifiable
        new_statusbar = Mock()
        kernel.statusbar = new_statusbar
        assert kernel.statusbar is new_statusbar

    def test_avg_jid_counter(self, temp_dir):
        """Test average job ID counter."""
        kernel = ViewerKernel(temp_dir)

        assert kernel.avg_jid == 0

        # Should be modifiable for job tracking
        kernel.avg_jid = 5
        assert kernel.avg_jid == 5


class TestViewerKernelWeakReferences:
    """Test suite for ViewerKernel weak reference handling."""

    def test_weak_reference_cache_behavior(self, temp_dir):
        """Test that weak reference cache behaves correctly."""
        kernel = ViewerKernel(temp_dir)

        # Create a test object for caching
        class TestObject:
            def __init__(self, value):
                self.value = value

        # Add object to cache
        test_obj = TestObject(42)
        kernel._current_dset_cache["test_key"] = test_obj

        # Verify object is cached
        assert "test_key" in kernel._current_dset_cache
        assert kernel._current_dset_cache["test_key"].value == 42

        # Delete reference - object should be removed from weak cache
        del test_obj

        # Depending on garbage collection timing, the weak reference
        # may or may not be immediately cleared, but accessing it
        # should either work or raise KeyError
        try:
            kernel._current_dset_cache["test_key"]
        except KeyError:
            # Expected behavior when object is garbage collected
            assert "test_key" not in kernel._current_dset_cache


class TestViewerKernelStringRepresentation:
    """Test suite for ViewerKernel string representations."""

    def test_has_docstring(self, temp_dir):
        """Test that ViewerKernel class has proper docstring."""
        ViewerKernel(temp_dir)

        assert ViewerKernel.__doc__ is not None
        assert len(ViewerKernel.__doc__) > 100  # Substantial docstring
        assert "ViewerKernel" in ViewerKernel.__doc__
        assert "Parameters" in ViewerKernel.__doc__
        assert "Attributes" in ViewerKernel.__doc__


class TestViewerKernelPerformance:
    """Test suite for ViewerKernel performance characteristics."""

    def test_initialization_performance(self, temp_dir, performance_timer):
        """Test ViewerKernel initialization performance."""
        performance_timer.start()

        kernel = ViewerKernel(temp_dir)

        elapsed = performance_timer.stop()

        # Initialization should be fast (< 100ms)
        assert elapsed < 0.1
        assert kernel is not None

    def test_meta_reset_performance(self, temp_dir, performance_timer):
        """Test meta reset performance."""
        kernel = ViewerKernel(temp_dir)

        performance_timer.start()
        kernel.reset_meta()
        elapsed = performance_timer.stop()

        # Meta reset should be very fast (< 10ms)
        assert elapsed < 0.01
        assert kernel.meta is not None


class TestViewerKernelEdgeCases:
    """Test suite for ViewerKernel edge cases and error conditions."""

    def test_empty_path_handling(self):
        """Test handling of empty path."""
        with patch("xpcs_toolkit.viewer_kernel.FileLocator.__init__"):
            kernel = ViewerKernel("")
            assert kernel.path == ""

    def test_none_statusbar_handling(self, temp_dir):
        """Test handling of None statusbar."""
        kernel = ViewerKernel(temp_dir, statusbar=None)
        assert kernel.statusbar is None

    def test_nonexistent_path_handling(self):
        """Test handling of nonexistent path."""
        nonexistent_path = "/nonexistent/path/that/should/not/exist"

        # Should not raise exception during initialization
        with patch("xpcs_toolkit.viewer_kernel.FileLocator.__init__"):
            kernel = ViewerKernel(nonexistent_path)
            assert kernel.path == nonexistent_path


class TestViewerKernelMemoryIntegration:
    """Test suite for ViewerKernel memory management integration."""

    @patch("xpcs_toolkit.viewer_kernel.MemoryMonitor")
    def test_memory_monitor_integration(self, mock_memory_monitor, temp_dir):
        """Test integration with MemoryMonitor."""
        ViewerKernel(temp_dir)

        # Kernel should have access to MemoryMonitor through import
        # This tests that the import is successful and accessible
        from xpcs_toolkit.viewer_kernel import MemoryMonitor as ImportedMemoryMonitor

        assert ImportedMemoryMonitor is not None

    def test_memory_threshold_validation(self, temp_dir):
        """Test memory threshold validation."""
        kernel = ViewerKernel(temp_dir)

        # Default threshold should be valid
        assert 0.0 <= kernel._memory_cleanup_threshold <= 1.0

        # Test setting valid thresholds
        kernel._memory_cleanup_threshold = 0.5
        assert kernel._memory_cleanup_threshold == 0.5

        kernel._memory_cleanup_threshold = 0.95
        assert kernel._memory_cleanup_threshold == 0.95


@pytest.mark.parametrize(
    "path_input,expected_behavior",
    [
        ("/valid/path", "normal"),
        ("relative/path", "normal"),
        ("", "empty"),
        (".", "current_dir"),
        ("/nonexistent", "nonexistent"),
    ],
)
def test_path_variations(path_input, expected_behavior):
    """Test ViewerKernel with various path inputs."""
    with (
        patch("xpcs_toolkit.viewer_kernel.FileLocator.__init__") as mock_super,
        patch("xpcs_toolkit.module.average_toolbox.AverageToolbox"),
    ):
        kernel = ViewerKernel(path_input)

        mock_super.assert_called_once_with(path_input)
        assert kernel.path == path_input

        if expected_behavior == "empty":
            assert kernel.path == ""
        elif expected_behavior == "current_dir":
            assert kernel.path == "."
        else:
            assert len(kernel.path) > 0


@pytest.mark.parametrize(
    "statusbar_input",
    [
        None,
        Mock(),
        "invalid_statusbar",
        42,
    ],
)
def test_statusbar_variations(statusbar_input, temp_dir):
    """Test ViewerKernel with various statusbar inputs."""
    with patch("xpcs_toolkit.viewer_kernel.FileLocator.__init__"):
        kernel = ViewerKernel(temp_dir, statusbar=statusbar_input)

        assert kernel.statusbar is statusbar_input
