"""
Utilities package for XPCS Toolkit.

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
    "ColoredConsoleFormatter",
    "JSONFormatter",
    "PerformanceFormatter",
    "StructuredFileFormatter",
    "create_formatter",
    "get_logger",  # Available as lazy import wrapper
    "log_system_info",  # Available as lazy import wrapper
    "setup_exception_logging",  # Available as lazy import wrapper
    "visualization_optimizer",  # Graphics optimization utilities
    # Other logging utilities available via direct import from .logging_config
    # "get_log_file_path",
    # "get_logging_config",
    # "initialize_logging",
    # "set_log_level",
    # "setup_logging",
]


# Convenience function for basic setup
def setup_basic_utilities(log_level="INFO"):
    """
    Setup basic XPCS Toolkit utilities.

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
    logger.info("XPCS Toolkit utilities initialized")

    utilities = {
        "logger": logger,
    }

    return utilities


# Add to exports
__all__.append("setup_basic_utilities")
