"""
Base plot worker class to reduce code duplication across plot workers.

This module provides a standardized base class for plot workers with common
initialization patterns, error handling, and progress reporting.
"""

from abc import abstractmethod
from typing import Any

from .async_workers import BaseAsyncWorker


class BasePlotWorker(BaseAsyncWorker):
    """
    Base class for plot workers with common initialization and patterns.

    This class standardizes the common patterns found across all plot workers:
    - Viewer kernel and plot handler initialization
    - Common plot kwargs handling
    - Standardized file list retrieval
    - Common error handling patterns
    - Progress reporting structure
    """

    def __init__(
        self,
        viewer_kernel,
        plot_handler,
        plot_kwargs: dict[str, Any],
        worker_id: str,
        worker_type: str = "plot",
    ):
        """
        Initialize base plot worker.

        Args:
            viewer_kernel: The viewer kernel instance
            plot_handler: The plot handler for display
            plot_kwargs: Dictionary of plotting parameters
            worker_id: Unique identifier for this worker
            worker_type: Type of worker for logging/debugging
        """
        super().__init__(worker_id)
        self.viewer_kernel = viewer_kernel
        self.plot_handler = plot_handler
        self.plot_kwargs = plot_kwargs
        self.worker_type = worker_type

    def get_file_list(self, rows: list[int] | None = None) -> list[Any]:
        """
        Get file list from viewer kernel with standardized error handling.

        Args:
            rows: Optional list of row indices to select

        Returns:
            List of XPCS file objects
        """
        if rows is None:
            rows = self.plot_kwargs.get("rows", [])

        return self.viewer_kernel.get_xf_list(rows)

    def validate_file_list(self, xf_list: list[Any], min_files: int = 1) -> bool:
        """
        Validate that we have sufficient files for plotting.

        Args:
            xf_list: List of XPCS file objects
            min_files: Minimum number of files required

        Returns:
            True if validation passes, False otherwise
        """
        if len(xf_list) < min_files:
            self.emit_status(
                f"Insufficient files: need {min_files}, got {len(xf_list)}"
            )
            return False
        return True

    def get_plot_parameter(self, key: str, default: Any = None) -> Any:
        """
        Safely get a parameter from plot_kwargs with default.

        Args:
            key: Parameter key to retrieve
            default: Default value if key not found

        Returns:
            Parameter value or default
        """
        return self.plot_kwargs.get(key, default)

    def emit_step_progress(
        self, current_step: int, total_steps: int, description: str
    ) -> None:
        """
        Emit progress with standardized step-based reporting.

        Args:
            current_step: Current step number (1-based)
            total_steps: Total number of steps
            description: Description of current step
        """
        self.emit_progress(current_step, total_steps, description)

    def create_error_result(self, error_message: str) -> dict[str, Any]:
        """
        Create standardized error result dictionary.

        Args:
            error_message: Error message to include

        Returns:
            Standardized error result dictionary
        """
        return {
            "success": False,
            "error": error_message,
            "worker_type": self.worker_type,
            "worker_id": self.worker_id,
        }

    def create_success_result(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Create standardized success result dictionary.

        Args:
            data: Result data to include

        Returns:
            Standardized success result dictionary
        """
        result = {
            "success": True,
            "worker_type": self.worker_type,
            "worker_id": self.worker_id,
        }
        result.update(data)
        return result

    def handle_common_exceptions(self, func_name: str = "plotting"):
        """
        Context manager for common exception handling.

        Args:
            func_name: Name of function for error messages
        """
        return CommonExceptionHandler(self, func_name)

    @abstractmethod
    def do_work(self) -> dict[str, Any]:
        """
        Execute the specific plotting work.

        This method must be implemented by subclasses to perform their
        specific plotting operations.

        Returns:
            Dictionary containing plot results
        """


class CommonExceptionHandler:
    """Context manager for standardized exception handling in plot workers."""

    def __init__(self, worker: BasePlotWorker, func_name: str):
        self.worker = worker
        self.func_name = func_name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"Error during {self.func_name}: {exc_val}"
            self.worker.emit_status(error_msg)
            # Return False to let the exception propagate
            return False
        return True


class StandardPlotProgressSteps:
    """Standardized progress step definitions for common plot types."""

    # Common step patterns
    BASIC_PLOT_STEPS = [
        "Loading data files",
        "Processing data",
        "Generating plot",
        "Finalizing display",
    ]

    FITTING_PLOT_STEPS = [
        "Loading data files",
        "Extracting data arrays",
        "Performing fitting",
        "Processing fit results",
        "Generating plot",
        "Finalizing display",
    ]

    MULTI_PANEL_STEPS = [
        "Loading data files",
        "Processing panel 1",
        "Processing panel 2",
        "Processing panel 3",
        "Combining panels",
        "Finalizing display",
    ]


def create_standard_plot_worker(
    worker_class: type,
    viewer_kernel,
    plot_handler,
    plot_kwargs: dict[str, Any],
    worker_id: str,
    worker_type: str = "plot",
) -> BasePlotWorker:
    """
    Factory function to create standardized plot workers.

    Args:
        worker_class: The specific worker class to instantiate
        viewer_kernel: The viewer kernel instance
        plot_handler: The plot handler for display
        plot_kwargs: Dictionary of plotting parameters
        worker_id: Unique identifier for this worker
        worker_type: Type of worker for logging/debugging

    Returns:
        Instantiated plot worker
    """
    return worker_class(
        viewer_kernel=viewer_kernel,
        plot_handler=plot_handler,
        plot_kwargs=plot_kwargs,
        worker_id=worker_id,
        worker_type=worker_type,
    )
