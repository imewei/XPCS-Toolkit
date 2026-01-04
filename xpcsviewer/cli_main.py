"""Command-line interface entry points for XPCS Viewer.

This module provides the main entry points for both the GUI and CLI:

    main_gui(): Launch the graphical interface (xpcsviewer-gui command)
    main(): CLI entry point with subcommands (xpcsviewer command)

The CLI supports batch processing operations like twotime correlation analysis,
while the GUI provides interactive visualization and analysis.

Example:
    Launch GUI::

        $ xpcsviewer-gui /path/to/data

    Run twotime batch processing::

        $ xpcsviewer twotime --input /data --output /results --q 0.05
"""

import argparse
import atexit
import signal
import sys

from xpcsviewer.utils.exceptions import convert_exception
from xpcsviewer.utils.logging_config import (
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
    global _shutting_down  # noqa: PLW0603 - intentional for graceful shutdown coordination
    if _shutting_down:
        return
    _shutting_down = True

    logger.info("Starting safe shutdown sequence...")

    # 1. Shutdown threading components first
    try:
        from xpcsviewer.threading.cleanup_optimized import (
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
        from xpcsviewer.fileIO.hdf_reader import _connection_pool

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
        from xpcsviewer.threading.cleanup_optimized import (
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
        from xpcsviewer.threading.cleanup_optimized import smart_gc_collect

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
        from xpcsviewer.threading.cleanup_optimized import shutdown_optimized_cleanup

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


def _get_version():
    """Get package version (deferred import for faster startup)."""
    from xpcsviewer import __version__

    return __version__


def _start_gui(path, label_style):
    """Start the GUI application (deferred import for faster startup)."""
    from xpcsviewer.xpcs_viewer import main_gui

    return main_gui(path, label_style)


def _run_twotime_batch(args):
    """Run twotime batch processing (deferred import for faster startup)."""
    from xpcsviewer.cli.twotime_batch import run_twotime_batch

    return run_twotime_batch(args)


def main_gui():
    """GUI entry point - launches the XPCS Viewer GUI directly."""
    argparser = argparse.ArgumentParser(
        description="XPCS Viewer GUI: Interactive XPCS data analysis"
    )

    argparser.add_argument(
        "--version", action="version", version=f"xpcsviewer: {_get_version()}"
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
    if args.log_level is not None:
        set_log_level(args.log_level)
        logger.info(f"Log level set to {args.log_level}")

    # Setup exception logging for uncaught exceptions
    setup_exception_logging()

    # Initialize optimized cleanup system
    try:
        from xpcsviewer.threading.cleanup_optimized import initialize_optimized_cleanup

        initialize_optimized_cleanup()
        logger.debug("Optimized cleanup system initialized")
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as e:
        xpcs_error = convert_exception(e, "Failed to initialize cleanup system")
        logger.warning(f"Cleanup initialization error: {xpcs_error}")

    # Resolve path
    path = args.positional_path if args.positional_path is not None else args.path
    logger.info(f"Starting XPCS Viewer GUI with path: {path}")

    try:
        exit_code = _start_gui(path, args.label_style)
        logger.info(f"XPCS Viewer GUI exited with code: {exit_code}")
        safe_shutdown()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        safe_shutdown()
        sys.exit(0)
    except (ImportError, ModuleNotFoundError) as e:
        logger.error(f"Missing required dependencies for GUI: {e}")
        logger.error(
            "Please ensure all required packages are installed: pip install -e ."
        )
        safe_shutdown()
        sys.exit(2)
    except Exception as e:
        xpcs_error = convert_exception(e, "Critical error starting XPCS Viewer")
        logger.error(f"Critical startup failure: {xpcs_error}", exc_info=True)
        safe_shutdown()
        sys.exit(1)


def main():
    """CLI entry point - provides command-line interface for XPCS Viewer."""
    argparser = argparse.ArgumentParser(
        description="XPCS Viewer CLI: Command-line tools for XPCS data analysis",
        epilog="Use 'xpcsviewer-gui' to launch the graphical interface.",
    )

    argparser.add_argument(
        "--version", action="version", version=f"xpcsviewer: {_get_version()}"
    )

    # Create subparsers for different commands
    subparsers = argparser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

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
        from xpcsviewer.threading.cleanup_optimized import initialize_optimized_cleanup

        initialize_optimized_cleanup()
        logger.debug("Optimized cleanup system initialized")
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as e:
        xpcs_error = convert_exception(e, "Failed to initialize cleanup system")
        logger.warning(f"Cleanup initialization error: {xpcs_error}")

    logger.info(f"XPCS Viewer CLI started: {args.command}")

    # Route to appropriate command handler
    if args.command == "twotime":
        logger.debug(f"Twotime Arguments: input='{args.input}', output='{args.output}'")
        try:
            exit_code = _run_twotime_batch(args)
            logger.info(f"Twotime batch processing completed with code: {exit_code}")
            safe_shutdown()
            sys.exit(exit_code)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            safe_shutdown()
            sys.exit(0)
        except Exception as e:
            xpcs_error = convert_exception(e, "Error in twotime batch processing")
            logger.error(f"Twotime processing failed: {xpcs_error}", exc_info=True)
            safe_shutdown()
            sys.exit(1)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
