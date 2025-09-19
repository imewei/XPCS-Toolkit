"""
Utilities package for XPCS Toolkit.

This package contains essential utility modules for logging infrastructure
and other core support functions.
"""

from .log_formatters import (
    ColoredConsoleFormatter,
    JSONFormatter,
    PerformanceFormatter,
    StructuredFileFormatter,
    create_formatter,
)

# Core logging utilities - always imported
from .logging_config import (
    get_log_file_path,
    get_logger,
    get_logging_config,
    initialize_logging,
    log_system_info,
    set_log_level,
    setup_exception_logging,
    setup_logging,
)

# Define essential exports
__all__ = [
    "ColoredConsoleFormatter",
    "JSONFormatter",
    "PerformanceFormatter",
    "StructuredFileFormatter",
    "create_formatter",
    "get_log_file_path",
    # Always available logging utilities
    "get_logger",
    "get_logging_config",
    "initialize_logging",
    "log_system_info",
    "set_log_level",
    "setup_exception_logging",
    "setup_logging",
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