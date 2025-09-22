"""
Simplified GUI integration utilities for threading in XPCS Toolkit.

This module provides basic threading integration for GUI responsiveness.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from PySide6 import QtWidgets

from ..utils.logging_config import get_logger
from .async_workers import WorkerManager
from .progress_manager import ProgressManager

logger = get_logger(__name__)


class ThreadingIntegrator:
    """Simplified threading integration for GUI applications."""

    def __init__(self, main_window: QtWidgets.QMainWindow):
        """Initialize with basic threading components."""
        self.main_window = main_window

        # Initialize basic components
        self.worker_manager = WorkerManager()
        self.progress_manager = ProgressManager(main_window)

        # Set up status bar if available
        if hasattr(main_window, "statusbar") or hasattr(main_window, "statusBar"):
            statusbar = getattr(main_window, "statusbar", None) or getattr(
                main_window, "statusBar", None
            )
            if statusbar:
                self.progress_manager.set_statusbar(statusbar())

        logger.info("ThreadingIntegrator initialized")

    def get_worker_manager(self) -> WorkerManager:
        """Get the worker manager."""
        return self.worker_manager

    def get_progress_manager(self) -> ProgressManager:
        """Get the progress manager."""
        return self.progress_manager


def make_async(func: Callable) -> Callable:
    """Simple decorator to make a function asynchronous using WorkerManager."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # This is a simplified version - just call the function directly
        # In a real implementation, this would submit to a worker thread
        return func(*args, **kwargs)

    return wrapper


def setup_enhanced_threading(main_window: QtWidgets.QMainWindow) -> ThreadingIntegrator:
    """Set up simplified threading for a main window."""
    return ThreadingIntegrator(main_window)


# Simplified async functions for compatibility
def async_load_xpcs_files(file_paths, progress_callback=None):
    """Simplified async file loading."""
    logger.info(f"Loading {len(file_paths)} XPCS files")
    # This would normally be implemented with proper async loading
    return []


def async_generate_g2_plot(data, progress_callback=None):
    """Simplified async G2 plot generation."""
    logger.info("Generating G2 plot")
    # This would normally be implemented with proper async plotting


def async_generate_saxs_plot(data, progress_callback=None):
    """Simplified async SAXS plot generation."""
    logger.info("Generating SAXS plot")
    # This would normally be implemented with proper async plotting


def create_progress_callback(progress_manager, task_name):
    """Create a simple progress callback."""

    def callback(progress):
        logger.debug(f"{task_name}: {progress}%")

    return callback


def create_completion_callback(on_complete):
    """Create a simple completion callback."""

    def callback(result):
        if on_complete:
            on_complete(result)

    return callback


class AsyncMethodMixin:
    """Simplified mixin for async methods."""

    def make_async_method(self, method):
        """Make a method async (simplified)."""
        return make_async(method)
