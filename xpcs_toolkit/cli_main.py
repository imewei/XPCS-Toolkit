"""Console script for xpcs-toolkit."""

import argparse
import atexit
import signal
import sys

from xpcs_toolkit.utils.exceptions import convert_exception
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
        xpcs_error = convert_exception(
            e, "Unexpected error during worker manager shutdown"
        )
        logger.warning(
            f"Unexpected shutdown error: {xpcs_error}"
        )  # Log but continue shutdown

    # 2. Clear HDF5 connection pool
    try:
        from xpcs_toolkit.fileIO.hdf_reader import _connection_pool

        _connection_pool.clear_pool(from_destructor=True)
    except (ImportError, AttributeError) as e:
        # Expected - HDF5 module not available or already cleaned up
        logger.debug(f"HDF5 cleanup issue (expected): {e}")
    except Exception as e:
        # Unexpected HDF5 cleanup errors
        xpcs_error = convert_exception(
            e, "Unexpected error clearing HDF5 connection pool"
        )
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
        logger.warning(
            f"Cleanup scheduling error: {xpcs_error}"
        )  # Log but continue shutdown

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
        logger.warning(
            f"Garbage collection error: {xpcs_error}"
        )  # Log but continue shutdown

    # 5. Shutdown cleanup system
    try:
        from xpcs_toolkit.threading.cleanup_optimized import shutdown_optimized_cleanup

        shutdown_optimized_cleanup()
    except (ImportError, AttributeError, RuntimeError) as e:
        # Expected - cleanup system not available or already shut down
        logger.debug(f"Cleanup system shutdown issue (expected): {e}")
    except Exception as e:
        # Unexpected cleanup system errors
        xpcs_error = convert_exception(
            e, "Unexpected error during cleanup system shutdown"
        )
        logger.warning(
            f"Cleanup system error: {xpcs_error}"
        )  # Log but continue shutdown


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

    def _run_twotime_batch(args):
        from xpcs_toolkit.cli.twotime_batch import run_twotime_batch

        return run_twotime_batch(args)

    argparser = argparse.ArgumentParser(
        description="XPCS Toolkit: a GUI tool for XPCS data analysis"
    )

    argparser.add_argument(
        "--version", action="version", version=f"xpcs-toolkit: {_get_version()}"
    )

    # Add GUI arguments to main parser for backward compatibility
    argparser.add_argument(
        "--path", type=str, help="path to the result folder", default="./"
    )
    argparser.add_argument(
        "positional_path",
        nargs="?",
        default=None,
        help="positional path to the result folder (default GUI mode)",
    )
    argparser.add_argument("--label_style", type=str, help="label style", default=None)

    # Create subparsers for different commands (optional for backward compatibility)
    subparsers = argparser.add_subparsers(
        dest="command", help="Available commands", required=False
    )

    # GUI command (explicit)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI explicitly")
    gui_parser.add_argument(
        "--path", type=str, help="path to the result folder", default="./"
    )
    gui_parser.add_argument(
        "positional_path",
        nargs="?",
        default=None,
        help="positional path to the result folder",
    )
    gui_parser.add_argument("--label_style", type=str, help="label style", default=None)

    # Twotime batch processing command
    twotime_parser = subparsers.add_parser(
        "twotime", help="Batch process twotime correlation data"
    )
    twotime_parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="HDF file path or directory containing HDF files",
    )
    twotime_parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output directory for generated images",
    )

    # Mutually exclusive group for q/phi selection
    selection_group = twotime_parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument(
        "--q",
        type=float,
        help="Q-value to process (saves images at all phi angles for this q)",
    )
    selection_group.add_argument(
        "--phi",
        type=float,
        help="Phi-value to process (saves images at this phi angle for all q values)",
    )
    selection_group.add_argument(
        "--q-phi",
        type=str,
        help="Specific q-phi pair as 'q,phi' (saves single image for this exact combination)",
    )

    twotime_parser.add_argument(
        "--dpi", type=int, default=300, help="Image resolution in DPI (default: 300)"
    )
    twotime_parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg", "pdf", "svg"],
        help="Image format (default: png)",
    )

    # Global arguments
    argparser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set logging level (default: INFO)",
    )

    args = argparser.parse_args()

    # Handle case where no subcommand is provided (default to GUI)
    if args.command is None:
        args.command = "gui"

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
    except (ImportError, ModuleNotFoundError):
        # Expected - cleanup optimization module not available
        logger.info("Optimized cleanup system not available, using standard cleanup")
    except Exception as e:
        # Unexpected initialization errors
        xpcs_error = convert_exception(
            e, "Failed to initialize optimized cleanup system"
        )
        logger.warning(f"Cleanup initialization error: {xpcs_error}")

    logger.info("XPCS Toolkit CLI started")
    if args.command == "gui":
        logger.debug(
            f"GUI Arguments: path='{args.path}', label_style='{args.label_style}'"
        )
    elif args.command == "twotime":
        logger.debug(f"Twotime Arguments: input='{args.input}', output='{args.output}'")

    # Route to appropriate command handler
    if args.command == "gui":
        if hasattr(args, "positional_path") and args.positional_path is not None:
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
            logger.error(
                "Please ensure all required packages are installed: pip install -e ."
            )
            safe_shutdown()
            sys.exit(2)  # Different exit code for dependency issues
        except Exception as e:
            # Unexpected critical errors - convert and provide context
            xpcs_error = convert_exception(e, "Critical error starting XPCS Toolkit")
            logger.error(f"Critical startup failure: {xpcs_error}", exc_info=True)
            safe_shutdown()
            sys.exit(1)
    elif args.command == "twotime":
        try:
            exit_code = _run_twotime_batch(args)
            logger.info(
                f"XPCS Toolkit twotime batch processing completed with code: {exit_code}"
            )
            safe_shutdown()
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            safe_shutdown()
            sys.exit(0)
        except Exception as e:
            # Handle twotime-specific errors
            xpcs_error = convert_exception(e, "Error in twotime batch processing")
            logger.error(f"Twotime processing failed: {xpcs_error}", exc_info=True)
            safe_shutdown()
            sys.exit(1)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
