"""
GUI Initialization Stability Enhancement.

This module provides robust validation, error handling, and stability
improvements for the XPCS Toolkit GUI initialization process.
"""

import os
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PySide6 import QtCore, QtWidgets

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class InitializationStage(Enum):
    """Stages of GUI initialization for tracking progress."""

    PRE_QT = "pre_qt_setup"
    QT_APPLICATION = "qt_application_creation"
    MAIN_WINDOW = "main_window_creation"
    UI_SETUP = "ui_component_setup"
    SIGNAL_CONNECTIONS = "signal_slot_connections"
    ASYNC_COMPONENTS = "async_component_initialization"
    RESOURCE_VALIDATION = "resource_validation"
    FINAL_VALIDATION = "final_validation"
    STARTUP_COMPLETE = "startup_complete"


@dataclass
class InitializationResult:
    """Result of an initialization step."""

    stage: InitializationStage
    success: bool
    duration: float
    error: str | None = None
    warning: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class GuiInitializationValidator:
    """
    Validates and enhances GUI initialization stability.

    This class provides comprehensive validation, error handling,
    and progress tracking for the XPCS Toolkit GUI startup process.
    """

    def __init__(self):
        self.results: list[InitializationResult] = []
        self.current_stage = None
        self.start_time = None
        self.validation_errors = []
        self.validation_warnings = []

    def validate_initialization_step(
        self, stage: InitializationStage, operation: Callable, *args, **kwargs
    ) -> InitializationResult:
        """
        Validate and execute an initialization step with error handling.

        Args:
            stage: The initialization stage being executed
            operation: Function to execute for this stage
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation

        Returns:
            InitializationResult with success status and details
        """
        self.current_stage = stage
        start_time = time.time()

        logger.debug(f"Starting initialization stage: {stage.value}")

        try:
            # Execute the operation
            result = operation(*args, **kwargs)

            duration = time.time() - start_time
            init_result = InitializationResult(
                stage=stage, success=True, duration=duration, details={"result": result}
            )

            logger.info(f"✅ {stage.value} completed successfully in {duration:.3f}s")

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e!s}"

            init_result = InitializationResult(
                stage=stage,
                success=False,
                duration=duration,
                error=error_msg,
                details={"exception": e, "traceback": traceback.format_exc()},
            )

            logger.error(f"❌ {stage.value} failed after {duration:.3f}s: {error_msg}")
            self.validation_errors.append(f"{stage.value}: {error_msg}")

        self.results.append(init_result)
        return init_result

    def validate_qt_environment(self) -> bool:
        """
        Validate Qt environment prerequisites.

        Returns:
            True if Qt environment is valid
        """
        issues = []

        # Check Qt availability
        import importlib.util

        if importlib.util.find_spec("PySide6.QtCore") and importlib.util.find_spec(
            "PySide6.QtWidgets"
        ):
            logger.debug("✅ PySide6 modules available")
        else:
            issues.append("PySide6 modules not available")

        # Check display environment
        if os.name != "nt":  # Unix-like systems
            display = os.environ.get("DISPLAY")
            if not display and os.environ.get("QT_QPA_PLATFORM") not in [
                "offscreen",
                "minimal",
            ]:
                issues.append(
                    "No DISPLAY environment variable and no offscreen platform set"
                )

        # Check Qt platform plugin
        qt_platform = os.environ.get("QT_QPA_PLATFORM")
        if qt_platform:
            logger.debug(f"Qt platform plugin: {qt_platform}")

        # Memory check
        try:
            import psutil

            memory = psutil.virtual_memory()
            if memory.available < 500 * 1024 * 1024:  # 500MB
                issues.append("Low available memory for GUI initialization")
        except ImportError:
            logger.debug("psutil not available for memory check")

        if issues:
            for issue in issues:
                logger.warning(f"Qt environment issue: {issue}")
                self.validation_warnings.append(issue)
            return False

        logger.info("✅ Qt environment validation passed")
        return True

    def validate_gui_components(self, main_window) -> bool:
        """
        Validate that GUI components are properly initialized.

        Args:
            main_window: Main window instance to validate

        Returns:
            True if all components are valid
        """
        issues = []

        # Check main window
        if not isinstance(main_window, QtWidgets.QMainWindow):
            issues.append("Main window is not a QMainWindow instance")
            return False

        # Check window state
        if not main_window.isVisible():
            logger.debug("Main window is not visible (may be intentional)")

        # Check window size
        size = main_window.size()
        if size.width() < 100 or size.height() < 100:
            issues.append(f"Main window size too small: {size.width()}x{size.height()}")

        # Check if window has central widget or content
        central_widget = main_window.centralWidget()
        if central_widget is None:
            issues.append("Main window has no central widget")

        # Check for essential child widgets
        essential_widgets = []
        try:
            # Look for common essential widgets
            tab_widgets = main_window.findChildren(QtWidgets.QTabWidget)
            if tab_widgets:
                essential_widgets.extend(tab_widgets)

            plot_widgets = main_window.findChildren(QtWidgets.QWidget)
            plot_widgets = [w for w in plot_widgets if "plot" in w.objectName().lower()]
            essential_widgets.extend(plot_widgets)

            logger.debug(f"Found {len(essential_widgets)} essential widgets")

        except Exception as e:
            issues.append(f"Failed to validate child widgets: {e}")

        # Check signal connections
        try:
            # Verify that main window can process events
            QtWidgets.QApplication.processEvents()
            logger.debug("✅ Event processing functional")
        except Exception as e:
            issues.append(f"Event processing failed: {e}")

        if issues:
            for issue in issues:
                logger.warning(f"GUI component issue: {issue}")
                self.validation_warnings.append(issue)
            return False

        logger.info("✅ GUI components validation passed")
        return True

    def validate_async_components(self, main_window) -> bool:
        """
        Validate async components initialization.

        Args:
            main_window: Main window instance

        Returns:
            True if async components are properly initialized
        """
        issues = []

        try:
            # Check thread pool
            if hasattr(main_window, "thread_pool"):
                thread_pool = main_window.thread_pool
                if thread_pool.maxThreadCount() <= 0:
                    issues.append("Thread pool has invalid max thread count")
                else:
                    logger.debug(
                        f"✅ Thread pool initialized with {thread_pool.maxThreadCount()} threads"
                    )
            else:
                issues.append("Main window missing thread_pool attribute")

            # Check progress manager
            if hasattr(main_window, "progress_manager"):
                progress_manager = main_window.progress_manager
                if progress_manager is None:
                    issues.append("Progress manager is None")
                else:
                    logger.debug("✅ Progress manager initialized")
            else:
                issues.append("Main window missing progress_manager attribute")

            # Check async viewer kernel
            if hasattr(main_window, "async_vk"):
                if main_window.async_vk is None:
                    logger.debug(
                        "Async viewer kernel not yet initialized (may be normal)"
                    )
                else:
                    logger.debug("✅ Async viewer kernel initialized")

            # Check viewer kernel
            if hasattr(main_window, "vk"):
                if main_window.vk is None:
                    logger.debug("Viewer kernel not yet initialized (may be normal)")
                else:
                    logger.debug("✅ Viewer kernel initialized")

        except Exception as e:
            issues.append(f"Failed to validate async components: {e}")

        if issues:
            for issue in issues:
                logger.warning(f"Async component issue: {issue}")
                self.validation_warnings.append(issue)
            return False

        logger.info("✅ Async components validation passed")
        return True

    def get_initialization_summary(self) -> dict[str, Any]:
        """
        Get summary of initialization validation results.

        Returns:
            Dictionary with initialization summary
        """
        total_stages = len(self.results)
        successful_stages = sum(1 for r in self.results if r.success)
        total_duration = sum(r.duration for r in self.results)

        failed_stages = [r.stage.value for r in self.results if not r.success]

        return {
            "total_stages": total_stages,
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "total_duration": total_duration,
            "success_rate": (successful_stages / total_stages) * 100
            if total_stages > 0
            else 0,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "stage_details": [
                {
                    "stage": r.stage.value,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class RobustGuiStarter:
    """
    Robust GUI starter with enhanced error handling and validation.

    This class provides a more robust way to start the XPCS Toolkit GUI
    with comprehensive error handling, validation, and recovery mechanisms.
    """

    def __init__(self):
        self.validator = GuiInitializationValidator()
        self.app = None
        self.main_window = None
        self.startup_successful = False

    def setup_qt_environment(self) -> bool:
        """
        Set up Qt environment with proper configuration.

        Returns:
            True if setup successful
        """

        def setup_operation():
            # Configure Qt application attributes
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
            QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

            # Configure for better stability
            QtWidgets.QApplication.setAttribute(
                QtCore.Qt.AA_DontCreateNativeWidgetSiblings, True
            )

            logger.debug("Qt application attributes configured")
            return True

        result = self.validator.validate_initialization_step(
            InitializationStage.PRE_QT, setup_operation
        )
        return result.success

    def create_qt_application(self) -> bool:
        """
        Create Qt application with error handling.

        Returns:
            True if application created successfully
        """

        def create_app_operation():
            # Check if application already exists
            existing_app = QtWidgets.QApplication.instance()
            if existing_app is not None:
                logger.warning("QApplication already exists, using existing instance")
                self.app = existing_app
            else:
                self.app = QtWidgets.QApplication([])
                logger.info("Qt Application created")

            # Configure application properties
            self.app.setQuitOnLastWindowClosed(True)
            self.app.setOrganizationName("XPCS-Toolkit")
            self.app.setApplicationName("XPCS Toolkit")

            return self.app

        result = self.validator.validate_initialization_step(
            InitializationStage.QT_APPLICATION, create_app_operation
        )
        return result.success

    def create_main_window(self, path=None, label_style=None) -> bool:
        """
        Create main window with validation.

        Args:
            path: Optional path for initialization
            label_style: Optional label style

        Returns:
            True if main window created successfully
        """

        def create_window_operation():
            from ..xpcs_viewer import XpcsViewer

            # Create main window
            self.main_window = XpcsViewer(path=path, label_style=label_style)

            # Validate window creation
            if not self.main_window:
                raise RuntimeError("Failed to create XpcsViewer instance")

            logger.info("XpcsViewer window created successfully")
            return self.main_window

        result = self.validator.validate_initialization_step(
            InitializationStage.MAIN_WINDOW, create_window_operation
        )
        return result.success

    def validate_gui_initialization(self) -> bool:
        """
        Validate complete GUI initialization.

        Returns:
            True if validation passed
        """

        def validation_operation():
            if not self.main_window:
                raise RuntimeError("Main window not available for validation")

            # Validate Qt environment
            env_valid = self.validator.validate_qt_environment()

            # Validate GUI components
            components_valid = self.validator.validate_gui_components(self.main_window)

            # Validate async components
            async_valid = self.validator.validate_async_components(self.main_window)

            # Overall validation
            overall_valid = env_valid and components_valid and async_valid

            if not overall_valid:
                raise RuntimeError("GUI validation failed - see warnings for details")

            return overall_valid

        result = self.validator.validate_initialization_step(
            InitializationStage.FINAL_VALIDATION, validation_operation
        )
        return result.success

    def start_gui_application(self, path=None, label_style=None) -> bool:
        """
        Start GUI application with comprehensive error handling.

        Args:
            path: Optional path for initialization
            label_style: Optional label style

        Returns:
            True if startup successful
        """
        try:
            logger.info("Starting robust GUI initialization")

            # Step 1: Validate Qt environment
            if not self.validator.validate_qt_environment():
                logger.warning(
                    "Qt environment validation failed, proceeding with caution"
                )

            # Step 2: Set up Qt environment
            if not self.setup_qt_environment():
                logger.error("Failed to set up Qt environment")
                return False

            # Step 3: Create Qt application
            if not self.create_qt_application():
                logger.error("Failed to create Qt application")
                return False

            # Step 4: Create main window
            if not self.create_main_window(path, label_style):
                logger.error("Failed to create main window")
                return False

            # Step 5: Validate complete initialization
            if not self.validate_gui_initialization():
                logger.warning("GUI validation failed, but proceeding")

            # Mark startup as successful
            self.startup_successful = True

            # Final validation stage
            self.validator.validate_initialization_step(
                InitializationStage.STARTUP_COMPLETE, lambda: True
            )

            logger.info("🎉 Robust GUI initialization completed successfully")

            # Show summary
            summary = self.validator.get_initialization_summary()
            logger.info(
                f"Initialization summary: {summary['successful_stages']}/{summary['total_stages']} stages successful"
            )

            return True

        except Exception as e:
            logger.critical(
                f"Critical failure during GUI initialization: {e}", exc_info=True
            )
            self.startup_successful = False
            return False

    def run_application(self) -> int:
        """
        Run the Qt application event loop.

        Returns:
            Application exit code
        """
        if not self.startup_successful or not self.app:
            logger.error("Cannot run application - initialization failed")
            return 1

        try:
            logger.info("Starting Qt application event loop")
            exit_code = self.app.exec()
            logger.info(f"Application finished with exit code: {exit_code}")
            return exit_code

        except Exception as e:
            logger.critical(
                f"Critical error during application execution: {e}", exc_info=True
            )
            return 1

        finally:
            logger.info("Application shutting down")

    def get_validation_report(self) -> str:
        """
        Get detailed validation report.

        Returns:
            Formatted validation report
        """
        summary = self.validator.get_initialization_summary()

        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("GUI Initialization Validation Report")
        report_lines.append("=" * 60)

        report_lines.append(f"Overall Success Rate: {summary['success_rate']:.1f}%")
        report_lines.append(f"Total Stages: {summary['total_stages']}")
        report_lines.append(f"Successful Stages: {summary['successful_stages']}")
        report_lines.append(f"Total Duration: {summary['total_duration']:.3f}s")

        if summary["failed_stages"]:
            report_lines.append(
                f"\nFailed Stages: {', '.join(summary['failed_stages'])}"
            )

        if summary["validation_errors"]:
            report_lines.append("\nValidation Errors:")
            for error in summary["validation_errors"]:
                report_lines.append(f"  ❌ {error}")

        if summary["validation_warnings"]:
            report_lines.append("\nValidation Warnings:")
            for warning in summary["validation_warnings"]:
                report_lines.append(f"  ⚠️  {warning}")

        report_lines.append("\nStage Details:")
        for stage in summary["stage_details"]:
            status = "✅" if stage["success"] else "❌"
            report_lines.append(
                f"  {status} {stage['stage']}: {stage['duration']:.3f}s"
            )
            if stage["error"]:
                report_lines.append(f"    Error: {stage['error']}")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)


@contextmanager
def robust_gui_context():
    """
    Context manager for robust GUI operations.

    Usage:
        with robust_gui_context() as starter:
            if starter.start_gui_application():
                starter.run_application()
    """
    starter = RobustGuiStarter()
    try:
        yield starter
    except Exception as e:
        logger.critical(f"Critical error in GUI context: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        try:
            if starter.app and starter.app != QtWidgets.QApplication.instance():
                starter.app.quit()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")


# Enhanced main function for robust GUI startup
def robust_main_gui(path=None, label_style=None) -> int:
    """
    Enhanced main GUI function with robust initialization.

    Args:
        path: Optional path for initialization
        label_style: Optional label style

    Returns:
        Application exit code
    """
    logger.info("Starting XPCS Toolkit with robust initialization")

    with robust_gui_context() as starter:
        if starter.start_gui_application(path, label_style):
            # Show validation report if there were any issues
            summary = starter.validator.get_initialization_summary()
            if summary["validation_warnings"] or summary["validation_errors"]:
                logger.info("Validation report:\n" + starter.get_validation_report())

            return starter.run_application()
        logger.error("Failed to start GUI application")
        logger.error("Validation report:\n" + starter.get_validation_report())
        return 1
