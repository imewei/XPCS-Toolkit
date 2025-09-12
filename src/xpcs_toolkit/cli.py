"""Console script for xpcs-toolkit."""

import sys
import signal
import atexit
import argparse
from xpcs_toolkit.utils.logging_config import (
    get_logger,
    initialize_logging,
    set_log_level,
    setup_exception_logging,
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
            get_object_registry,
            schedule_type_cleanup,
            CleanupPriority,
            shutdown_optimized_cleanup,
        )

        # Use optimized registry lookup instead of expensive gc.get_objects()
        registry = get_object_registry()
        worker_managers = registry.get_objects_by_type("WorkerManager")
        for manager in worker_managers:
            manager.shutdown()

        logger.debug(f"Shutdown {len(worker_managers)} WorkerManager instances")
    except Exception as e:
        logger.debug(f"Error shutting down worker managers: {e}")  # Log but continue shutdown

    # 2. Clear HDF5 connection pool
    try:
        from xpcs_toolkit.fileIO.hdf_reader import _connection_pool

        _connection_pool.clear_pool(from_destructor=True)
    except Exception as e:
        logger.debug(f"Error clearing HDF5 connection pool: {e}")  # Log but continue shutdown

    # 3. Schedule XpcsFile cache clearing in background (non-blocking)
    try:
        from xpcs_toolkit.threading.cleanup_optimized import (
            schedule_type_cleanup,
            CleanupPriority,
        )

        # Schedule high-priority cleanup for XpcsFile objects
        schedule_type_cleanup("XpcsFile", CleanupPriority.HIGH)

        logger.debug("Scheduled background cleanup for XpcsFile objects")
    except Exception as e:
        logger.debug(f"Error scheduling cleanup: {e}")  # Log but continue shutdown

    # 4. Smart garbage collection (non-blocking)
    try:
        from xpcs_toolkit.threading.cleanup_optimized import smart_gc_collect

        smart_gc_collect("shutdown")
        logger.info("Safe shutdown sequence completed")
    except Exception as e:
        logger.debug(f"Error during smart garbage collection: {e}")  # Log but continue shutdown

    # 5. Shutdown cleanup system
    try:
        from xpcs_toolkit.threading.cleanup_optimized import shutdown_optimized_cleanup

        shutdown_optimized_cleanup()
    except Exception as e:
        logger.debug(f"Error during cleanup system shutdown: {e}")  # Log but continue shutdown


def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    safe_shutdown()
    sys.exit(0)


def main():
    from xpcs_toolkit import __version__
    from xpcs_toolkit.xpcs_viewer import main_gui

    argparser = argparse.ArgumentParser(
        description="XPCS Toolkit: a GUI tool for XPCS data analysis"
    )

    argparser.add_argument(
        "--version", action="version", version=f"xpcs-toolkit: {__version__}"
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
    except Exception as e:
        logger.warning(f"Failed to initialize optimized cleanup system: {e}")

    logger.info("XPCS Toolkit CLI started")
    logger.debug(f"Arguments: path='{args.path}', label_style='{args.label_style}'")

    if args.positional_path is not None:
        args.path = args.positional_path
        logger.debug(f"Using positional path: {args.path}")

    try:
        exit_code = main_gui(args.path, args.label_style)
        logger.info(f"XPCS Toolkit GUI exited with code: {exit_code}")
        safe_shutdown()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        safe_shutdown()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start GUI: {e}", exc_info=True)
        safe_shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
