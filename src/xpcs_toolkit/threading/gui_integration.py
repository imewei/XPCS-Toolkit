"""
GUI integration utilities for enhanced threading in XPCS Toolkit.

This module provides easy-to-use integration functions for upgrading
existing GUI code to use the enhanced threading capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional
from functools import wraps

from PySide6 import QtWidgets
from PySide6.QtCore import QObject

from .async_workers_enhanced import WorkerManager, EnhancedThreadPool, WorkerPriority
from .progress_manager import ProgressManager
from .main_thread_optimizer import MainThreadOptimizer


logger = logging.getLogger(__name__)


class ThreadingIntegrator(QObject):
    """
    Main integration class that sets up enhanced threading for an XPCS application.

    This class acts as a coordinator between the main window, worker manager,
    progress manager, and main thread optimizer.
    """

    def __init__(self, main_window: QtWidgets.QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        # Initialize threading components
        self.enhanced_thread_pool = EnhancedThreadPool(main_window)
        self.worker_manager = WorkerManager(self.enhanced_thread_pool)
        self.progress_manager = ProgressManager(main_window)
        self.main_thread_optimizer = MainThreadOptimizer(
            main_window, self.worker_manager, self.progress_manager
        )

        # Set up status bar if available
        if hasattr(main_window, "statusbar") or hasattr(main_window, "statusBar"):
            statusbar = getattr(main_window, "statusbar", None) or getattr(
                main_window, "statusBar", None
            )
            if statusbar:
                self.progress_manager.set_statusbar(statusbar())

        logger.info("ThreadingIntegrator initialized")

    def get_worker_manager(self) -> WorkerManager:
        """Get the enhanced worker manager."""
        return self.worker_manager

    def get_progress_manager(self) -> ProgressManager:
        """Get the progress manager."""
        return self.progress_manager

    def get_main_thread_optimizer(self) -> MainThreadOptimizer:
        """Get the main thread optimizer."""
        return self.main_thread_optimizer

    def async_file_load(
        self,
        file_paths: List[str],
        load_function: Callable,
        progress_callback: Optional[Callable] = None,
        completion_callback: Optional[Callable] = None,
        batch_size: int = 5,
    ) -> str:
        """Convenience method for async file loading."""
        return self.main_thread_optimizer.optimize_file_loading(
            file_paths,
            load_function,
            progress_callback,
            completion_callback,
            batch_size,
        )

    def async_plot_generate(
        self,
        plot_function: Callable,
        plot_args: tuple = (),
        plot_kwargs: dict = None,
        plot_type: str = "generic",
        priority: WorkerPriority = WorkerPriority.NORMAL,
    ) -> str:
        """Convenience method for async plot generation."""
        return self.main_thread_optimizer.optimize_plot_generation(
            plot_function, plot_args, plot_kwargs, plot_type, True, priority
        )

    def progressive_load(
        self,
        total_items: int,
        loader_function: Callable,
        ui_update_function: Callable,
        batch_size: int = 10,
    ) -> str:
        """Convenience method for progressive loading."""
        return self.main_thread_optimizer.implement_progressive_loading(
            total_items, loader_function, ui_update_function, batch_size
        )

    def validate_params_async(
        self,
        params: Dict[str, Any],
        validation_function: Callable,
        validation_callback: Optional[Callable] = None,
    ) -> str:
        """Convenience method for async parameter validation."""
        return self.main_thread_optimizer.validate_parameters_async(
            params, validation_function, validation_callback
        )

    def shutdown(self):
        """Shutdown all threading components."""
        self.main_thread_optimizer.shutdown()
        self.worker_manager.shutdown()
        self.progress_manager.hide_progress_dialog()


def make_async(
    plot_type: str = "generic",
    priority: WorkerPriority = WorkerPriority.NORMAL,
    cache_results: bool = True,
):
    """
    Decorator to make a plotting function asynchronous.

    Usage:
        @make_async("saxs_2d", WorkerPriority.HIGH)
        def plot_saxs_2d(self, handler, data):
            # Original plotting code
            return plot_result
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Check if the instance has threading integrator
            if not hasattr(self, "_threading_integrator"):
                # Fall back to synchronous execution
                logger.warning(
                    f"No threading integrator found for {func.__name__}, executing synchronously"
                )
                return func(self, *args, **kwargs)

            integrator: ThreadingIntegrator = self._threading_integrator

            # Extract handler from args if present
            handler = args[0] if args else kwargs.get("handler")

            # Create wrapper function for the async execution
            def async_plot_func():
                return func(self, *args, **kwargs)

            # Start async plot generation
            operation_id = integrator.async_plot_generate(
                async_plot_func, plot_type=plot_type, priority=priority
            )

            # Connect completion signal to update handler
            if handler:

                def on_plot_completed(op_id: str, result: Any):
                    if op_id == operation_id and result is not None:
                        # Apply result to handler if needed
                        if hasattr(handler, "setData") and hasattr(result, "data"):
                            handler.setData(result.data)
                        elif hasattr(handler, "setImage") and hasattr(result, "image"):
                            handler.setImage(result.image)

                integrator.main_thread_optimizer.operation_completed.connect(
                    on_plot_completed
                )

            return operation_id

        return wrapper

    return decorator


def setup_enhanced_threading(main_window: QtWidgets.QMainWindow) -> ThreadingIntegrator:
    """
    Set up enhanced threading for an XPCS main window.

    This function should be called during main window initialization
    to set up all threading components.

    Args:
        main_window: The main application window

    Returns:
        ThreadingIntegrator instance for managing threading
    """
    integrator = ThreadingIntegrator(main_window)

    # Store integrator in main window for easy access
    main_window._threading_integrator = integrator

    # Set up keyboard shortcuts for progress dialog
    from PySide6.QtGui import QShortcut, QKeySequence

    # Ctrl+P to show progress dialog
    progress_shortcut = QShortcut(QKeySequence("Ctrl+P"), main_window)
    progress_shortcut.activated.connect(
        integrator.progress_manager.show_progress_dialog
    )

    # Ctrl+Shift+P to hide progress dialog
    hide_progress_shortcut = QShortcut(QKeySequence("Ctrl+Shift+P"), main_window)
    hide_progress_shortcut.activated.connect(
        integrator.progress_manager.hide_progress_dialog
    )

    logger.info("Enhanced threading setup complete for main window")
    return integrator


class AsyncMethodMixin:
    """
    Mixin class that provides async method utilities for GUI classes.

    Classes that inherit from this mixin can easily convert their methods
    to async versions using the provided utilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_operations: Dict[str, str] = {}  # method_name -> operation_id

    def make_method_async(self, method_name: str, *args, **kwargs):
        """
        Make a method call asynchronous.

        Args:
            method_name: Name of the method to call asynchronously
            *args, **kwargs: Arguments to pass to the method

        Returns:
            Operation ID for tracking
        """
        if not hasattr(self, "_threading_integrator"):
            raise RuntimeError(
                "No threading integrator available. Call setup_enhanced_threading first."
            )

        if not hasattr(self, method_name):
            raise AttributeError(f"Method {method_name} not found")

        method = getattr(self, method_name)
        integrator: ThreadingIntegrator = self._threading_integrator

        # Create wrapper function
        def async_method():
            return method(*args, **kwargs)

        # Determine plot type from method name
        plot_type = method_name.replace("plot_", "").replace("_", " ")

        # Start async execution
        operation_id = integrator.async_plot_generate(
            async_method, plot_type=plot_type, priority=WorkerPriority.NORMAL
        )

        # Track the operation
        self._async_operations[method_name] = operation_id

        return operation_id

    def cancel_async_method(self, method_name: str) -> bool:
        """Cancel an async method execution."""
        if method_name in self._async_operations:
            if hasattr(self, "_threading_integrator"):
                integrator: ThreadingIntegrator = self._threading_integrator
                operation_id = self._async_operations[method_name]
                success = integrator.main_thread_optimizer.cancel_operation(
                    operation_id
                )
                if success:
                    del self._async_operations[method_name]
                return success
        return False

    def get_async_method_status(self, method_name: str) -> Dict[str, Any]:
        """Get status of an async method execution."""
        if method_name in self._async_operations and hasattr(
            self, "_threading_integrator"
        ):
            integrator: ThreadingIntegrator = self._threading_integrator
            operation_id = self._async_operations[method_name]
            return integrator.main_thread_optimizer.get_operation_status(operation_id)

        return {"exists": False}

    def wait_for_async_methods(self, timeout_ms: int = 5000):
        """Wait for all async methods to complete."""
        if not hasattr(self, "_threading_integrator"):
            return

        integrator: ThreadingIntegrator = self._threading_integrator
        integrator.worker_manager.thread_pool.waitForDone(timeout_ms)

    def cleanup_async_methods(self):
        """Clean up async method tracking."""
        # Cancel any remaining operations
        for method_name in list(self._async_operations.keys()):
            self.cancel_async_method(method_name)


def create_progress_callback(progress_manager: ProgressManager, operation_id: str):
    """
    Create a progress callback function for use with async operations.

    Args:
        progress_manager: The progress manager instance
        operation_id: Operation ID to update progress for

    Returns:
        Progress callback function
    """

    def progress_callback(current: int, total: int, message: str = ""):
        progress_manager.update_progress(operation_id, current, total, message)

    return progress_callback


def create_completion_callback(
    completion_handler: Callable, progress_manager: ProgressManager, operation_id: str
):
    """
    Create a completion callback function for async operations.

    Args:
        completion_handler: Function to call when operation completes
        progress_manager: The progress manager instance
        operation_id: Operation ID to complete

    Returns:
        Completion callback function
    """

    def completion_callback(result: Any):
        try:
            if result is not None:
                completion_handler(result)
            progress_manager.complete_operation(
                operation_id, success=result is not None
            )
        except Exception as e:
            logger.error(f"Error in completion callback for {operation_id}: {e}")
            progress_manager.complete_operation(
                operation_id, success=False, final_message=f"Callback error: {e}"
            )

    return completion_callback


# Example usage functions for common XPCS operations


def async_load_xpcs_files(
    integrator: ThreadingIntegrator,
    file_paths: List[str],
    ui_update_callback: Optional[Callable] = None,
) -> str:
    """
    Load XPCS files asynchronously with progress feedback.

    Args:
        integrator: Threading integrator instance
        file_paths: List of XPCS file paths to load
        ui_update_callback: Optional callback to update UI with loaded files

    Returns:
        Operation ID for tracking
    """

    def xpcs_loader(file_path: str):
        # Import here to avoid circular imports
        from ..xpcs_file import XpcsFile

        return XpcsFile(file_path)

    def progress_callback(partial_result: Dict[str, Any], is_final: bool):
        if ui_update_callback:
            ui_update_callback(partial_result, is_final)

    return integrator.async_file_load(
        file_paths, xpcs_loader, progress_callback, None, batch_size=3
    )


def async_generate_saxs_plot(
    integrator: ThreadingIntegrator, plot_handler, xf_list: List, **plot_kwargs
) -> str:
    """
    Generate SAXS plot asynchronously.

    Args:
        integrator: Threading integrator instance
        plot_handler: Plot handler for displaying results
        xf_list: List of XpcsFile objects
        **plot_kwargs: Additional plot arguments

    Returns:
        Operation ID for tracking
    """

    def saxs_plot_func():
        # Import here to avoid circular imports
        from ..module import saxs2d

        return saxs2d.pg_plot(plot_handler, xf_list, **plot_kwargs)

    return integrator.async_plot_generate(
        saxs_plot_func, plot_type="saxs_2d", priority=WorkerPriority.HIGH
    )


def async_generate_g2_plot(
    integrator: ThreadingIntegrator,
    plot_handler,
    xf_list: List,
    q_range: tuple,
    t_range: tuple,
    y_range: tuple,
    **plot_kwargs,
) -> str:
    """
    Generate G2 correlation plot asynchronously.

    Args:
        integrator: Threading integrator instance
        plot_handler: Plot handler for displaying results
        xf_list: List of XpcsFile objects
        q_range: Q range for analysis
        t_range: Time range for analysis
        y_range: Y axis range
        **plot_kwargs: Additional plot arguments

    Returns:
        Operation ID for tracking
    """

    def g2_plot_func():
        # Import here to avoid circular imports
        from ..module import g2mod

        return g2mod.pg_plot(
            plot_handler, xf_list, q_range, t_range, y_range, **plot_kwargs
        )

    return integrator.async_plot_generate(
        g2_plot_func, plot_type="g2_correlation", priority=WorkerPriority.HIGH
    )
