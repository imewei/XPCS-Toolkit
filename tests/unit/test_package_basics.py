"""Unit tests for basic package functionality.

This module provides unit tests for package-level functionality including
version access, CLI module imports, and basic component imports.
"""

import os

import pytest

from xpcs_toolkit import __version__
from xpcs_toolkit.cli import main


@pytest.mark.unit
def test_package_version():
    """Test that package version is accessible."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


@pytest.mark.unit
def test_cli_module_importable():
    """Test that CLI module can be imported."""
    assert callable(main)


@pytest.mark.unit
def test_basic_imports():
    """Test that basic modules can be imported."""
    try:
        # Test core module imports by importing and checking they exist
        import xpcs_toolkit.file_locator
        import xpcs_toolkit.viewer_kernel

        # Check that the modules have the expected classes
        assert hasattr(xpcs_toolkit.viewer_kernel, "ViewerKernel")
        assert hasattr(xpcs_toolkit.file_locator, "FileLocator")

    except ImportError as e:
        pytest.fail(f"Failed to import basic modules: {e}")


@pytest.mark.unit
def test_threading_imports():
    """Test that threading components can be imported without metaclass conflicts."""
    try:
        # Test that WorkerSignals can be instantiated
        from xpcs_toolkit.threading.async_workers import WorkerSignals

        os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Headless mode
        signals = WorkerSignals()
        assert signals is not None
    except Exception as e:
        pytest.fail(f"Failed to import threading modules: {e}")
