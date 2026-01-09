"""
Utilities package for XPCS Viewer.

This package contains essential utility modules for logging infrastructure
and other core support functions.
"""

import os

# Import modules gracefully for documentation builds
if os.environ.get("BUILDING_DOCS"):
    # Provide placeholder modules for documentation
    class PlaceholderModule:
        """Placeholder module for documentation builds."""

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    visualization_optimizer = PlaceholderModule()

    # Import log formatters normally as they should work
    from .log_formatters import (
        ColoredConsoleFormatter,
        JSONFormatter,
        PerformanceFormatter,
        StructuredFileFormatter,
        create_formatter,
    )
else:
    from .log_formatters import (
        ColoredConsoleFormatter,
        JSONFormatter,
        PerformanceFormatter,
        StructuredFileFormatter,
        create_formatter,
    )

    try:
        from . import visualization_optimizer
    except ImportError:
        # Create placeholder if import fails
        class PlaceholderModule:
            """Placeholder module for documentation builds."""

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        visualization_optimizer = PlaceholderModule()


# Core logging utilities - lazy import to avoid initialization cascade
def get_logger(name=None):
    """Lazy import wrapper for get_logger to avoid circular imports."""
    from .logging_config import get_logger as _get_logger

    return _get_logger(name)


# New logging utilities (comprehensive logging system)
def LoggingContext(*args, **kwargs):
    """Lazy import wrapper for LoggingContext."""
    from .log_utils import LoggingContext as _LoggingContext

    return _LoggingContext(*args, **kwargs)


def RateLimitedLogger(logger, **kwargs):
    """Lazy import wrapper for RateLimitedLogger."""
    from .log_utils import RateLimitedLogger as _RateLimitedLogger

    return _RateLimitedLogger(logger, **kwargs)


def log_timing(**kwargs):
    """Lazy import wrapper for log_timing decorator."""
    from .log_utils import log_timing as _log_timing

    return _log_timing(**kwargs)


def sanitize_path(path, mode=None):
    """Lazy import wrapper for sanitize_path."""
    from .log_utils import sanitize_path as _sanitize_path

    return _sanitize_path(path, mode)


def set_log_level(level):
    """Lazy import wrapper for set_log_level."""
    from .logging_config import set_log_level as _set_log_level

    return _set_log_level(level)


def log_system_info():
    """Lazy import wrapper for log_system_info to avoid circular imports."""
    from .logging_config import log_system_info as _log_system_info

    return _log_system_info()


def setup_exception_logging():
    """Lazy import wrapper for setup_exception_logging to avoid circular imports."""
    from .logging_config import setup_exception_logging as _setup_exception_logging

    return _setup_exception_logging()


# Other logging utilities available via direct import from .logging_config
# Commented out to avoid circular import cascade:
# from .logging_config import (
#     get_log_file_path,
#     get_logging_config,
#     initialize_logging,
#     log_system_info,
#     set_log_level,
#     setup_exception_logging,
#     setup_logging,
# )

# Define essential exports
__all__ = [
    # Log formatters
    "ColoredConsoleFormatter",
    "JSONFormatter",
    "PerformanceFormatter",
    "StructuredFileFormatter",
    "create_formatter",
    # Core logging utilities (lazy import wrappers)
    "get_logger",
    "log_system_info",
    "setup_exception_logging",
    "set_log_level",
    # New comprehensive logging utilities
    "LoggingContext",
    "RateLimitedLogger",
    "log_timing",
    "sanitize_path",
    # Graphics utilities
    "visualization_optimizer",
]


# Convenience function for basic setup
def setup_basic_utilities(log_level="INFO"):
    """
    Setup basic XPCS Viewer utilities.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns
    -------
    dict
        Dictionary containing initialized utilities
    """
    # Import logging utilities on demand to avoid initialization cascade
    from .logging_config import get_logger, initialize_logging, set_log_level

    # Initialize logging
    initialize_logging()
    set_log_level(log_level)

    logger = get_logger(__name__)
    logger.info("XPCS Viewer utilities initialized")

    utilities = {
        "logger": logger,
    }

    return utilities


# Add to exports
__all__.append("setup_basic_utilities")
