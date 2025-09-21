"""Console script for xpcs-toolkit."""

import argparse
import atexit
import signal
import sys

from xpcs_toolkit.utils.logging_config import (
    get_logger,
    initialize_logging,
    set_log_level,
    setup_exception_logging,
)
from xpcs_toolkit.utils.exceptions import (
    XPCSConfigurationError,
    XPCSGUIError,
    convert_exception,
)

logger = get_logger(__name__)

# Global flag to track if we're shutting down
_shutting_down = False


def safe_shutdown():
    """Safely shutdown the application to prevent segfaults."""
    global _shutting_down
    if _shutting_down:
        return
    _shutting_down = True

    logger.info("Starting safe shutdown sequence...")

    # 1. Shutdown threading components first
    try:
        from xpcs_toolkit.threading.cleanup_optimized import (
            CleanupPriority,
            get_object_registry,
            schedule_type_cleanup,
            shutdown_optimized_cleanup,
        )

        # Use optimized registry lookup instead of expensive gc.get_objects()
        registry = get_object_registry()
        worker_managers = registry.get_objects_by_type("WorkerManager")
        for manager in worker_managers:
            manager.shutdown()

        logger.debug(f"Shutdown {len(worker_managers)} WorkerManager instances")
    except (ImportError, AttributeError, RuntimeError) as e:
        # Expected errors during shutdown - module not available or already cleaned up
        logger.debug(f"Worker manager shutdown issue (expected): {e}")
    except Exception as e:
        # Unexpected errors - convert and log but continue shutdown
        xpcs_error = convert_exception(e, "Unexpected error during worker manager shutdown")
        logger.warning(f"Unexpected shutdown error: {xpcs_error}")  # Log but continue shutdown

    # 2. Clear HDF5 connection pool
    try:
        from xpcs_toolkit.fileIO.hdf_reader import _connection_pool

        _connection_pool.clear_pool(from_destructor=True)
    except (ImportError, AttributeError) as e:
        # Expected - HDF5 module not available or already cleaned up
        logger.debug(f"HDF5 cleanup issue (expected): {e}")
    except Exception as e:
        # Unexpected HDF5 cleanup errors
        xpcs_error = convert_exception(e, "Unexpected error clearing HDF5 connection pool")
        logger.warning(f"HDF5 cleanup error: {xpcs_error}")  # Log but continue shutdown

    # 3. Schedule XpcsFile cache clearing in background (non-blocking)
    try:
        from xpcs_toolkit.threading.cleanup_optimized import (
            CleanupPriority,
            schedule_type_cleanup,
        )

        # Schedule high-priority cleanup for XpcsFile objects
        schedule_type_cleanup("XpcsFile", CleanupPriority.HIGH)

        logger.debug("Scheduled background cleanup for XpcsFile objects")
    except (ImportError, AttributeError, TypeError) as e:
        # Expected - cleanup module not available or configuration issues
        logger.debug(f"Cleanup scheduling issue (expected): {e}")
    except Exception as e:
        # Unexpected cleanup scheduling errors
        xpcs_error = convert_exception(e, "Unexpected error scheduling cleanup")
        logger.warning(f"Cleanup scheduling error: {xpcs_error}")  # Log but continue shutdown

    # 4. Smart garbage collection (non-blocking)
    try:
        from xpcs_toolkit.threading.cleanup_optimized import smart_gc_collect

        smart_gc_collect("shutdown")
        logger.info("Safe shutdown sequence completed")
    except (ImportError, AttributeError, MemoryError) as e:
        # Expected - GC module issues or memory pressure during shutdown
        logger.debug(f"Garbage collection issue (expected): {e}")
    except Exception as e:
        # Unexpected garbage collection errors
        xpcs_error = convert_exception(e, "Unexpected error during garbage collection")
        logger.warning(f"Garbage collection error: {xpcs_error}")  # Log but continue shutdown

    # 5. Shutdown cleanup system
    try:
        from xpcs_toolkit.threading.cleanup_optimized import shutdown_optimized_cleanup

        shutdown_optimized_cleanup()
    except (ImportError, AttributeError, RuntimeError) as e:
        # Expected - cleanup system not available or already shut down
        logger.debug(f"Cleanup system shutdown issue (expected): {e}")
    except Exception as e:
        # Unexpected cleanup system errors
        xpcs_error = convert_exception(e, "Unexpected error during cleanup system shutdown")
        logger.warning(f"Cleanup system error: {xpcs_error}")  # Log but continue shutdown


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    safe_shutdown()
    sys.exit(0)


def main():
    # Defer heavy imports until after argument parsing for faster startup
    def _get_version():
        from xpcs_toolkit import __version__
        return __version__

    def _start_gui(path, label_style):
        from xpcs_toolkit.xpcs_viewer import main_gui
        return main_gui(path, label_style)

    argparser = argparse.ArgumentParser(
        description="XPCS Toolkit: a GUI tool for XPCS data analysis"
    )

    argparser.add_argument(
        "--version", action="version", version=f"xpcs-toolkit: {_get_version()}"
    )

    argparser.add_argument(
        "--path", type=str, help="path to the result folder", default="./"
    )
    argparser.add_argument(
        "positional_path",
        nargs="?",
        default=None,
        help="positional path to the result folder",
    )
    # Determine the directory to monitor
    argparser.add_argument("--label_style", type=str, help="label style", default=None)

    argparser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level (default: INFO)",
    )

    args = argparser.parse_args()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(safe_shutdown)

    # Initialize logging system
    initialize_logging()

    # Set log level if specified
    if hasattr(args, "log_level") and args.log_level is not None:
        set_log_level(args.log_level)
        logger.info(f"Log level set to {args.log_level}")

    # Setup exception logging for uncaught exceptions
    setup_exception_logging()

    # Initialize optimized cleanup system
    try:
        from xpcs_toolkit.threading.cleanup_optimized import (
            initialize_optimized_cleanup,
        )

        initialize_optimized_cleanup()
        logger.info("Optimized cleanup system initialized")
    except (ImportError, ModuleNotFoundError) as e:
        # Expected - cleanup optimization module not available
        logger.info("Optimized cleanup system not available, using standard cleanup")
    except Exception as e:
        # Unexpected initialization errors
        xpcs_error = convert_exception(e, "Failed to initialize optimized cleanup system")
        logger.warning(f"Cleanup initialization error: {xpcs_error}")

    logger.info("XPCS Toolkit CLI started")
    logger.debug(f"Arguments: path='{args.path}', label_style='{args.label_style}'")

    if args.positional_path is not None:
        args.path = args.positional_path
        logger.debug(f"Using positional path: {args.path}")

    try:
        exit_code = _start_gui(args.path, args.label_style)
        logger.info(f"XPCS Toolkit GUI exited with code: {exit_code}")
        safe_shutdown()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        safe_shutdown()
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        # Missing dependencies - provide helpful error message
        logger.error(f"Missing required dependencies for GUI: {e}")
        logger.error("Please ensure all required packages are installed: pip install -e .")
        safe_shutdown()
        sys.exit(2)  # Different exit code for dependency issues
    except XPCSConfigurationError as e:
        # Configuration issues - user can fix
        logger.error(f"Configuration error: {e}")
        if e.recovery_suggestions:
            for suggestion in e.recovery_suggestions:
                logger.error(f"  - {suggestion}")
        safe_shutdown()
        sys.exit(3)  # Different exit code for config issues
    except XPCSGUIError as e:
        # GUI-specific errors
        logger.error(f"GUI initialization failed: {e}")
        safe_shutdown()
        sys.exit(4)  # Different exit code for GUI issues
    except Exception as e:
        # Unexpected critical errors - convert and provide context
        xpcs_error = convert_exception(e, "Critical error starting XPCS Toolkit")
        logger.error(f"Critical startup failure: {xpcs_error}", exc_info=True)
        safe_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
