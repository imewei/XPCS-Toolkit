"""
Utilities package for XPCS Toolkit.

This package contains utility modules for performance profiling,
optimization helpers, logging infrastructure, and other support functions.
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

# Essential performance utilities
try:
    from .performance_profiler import (
        PerformanceProfiler,
        global_profiler,
        profile_algorithm,
        profile_block,
    )

    _profiler_available = True
except ImportError:
    _profiler_available = False

# Core memory management
try:
    from .memory_utils import (
        MemoryOptimizer,
        MemoryTracker,
        SystemMemoryMonitor,
    )

    _memory_available = True
except ImportError:
    _memory_available = False

# Essential caching functionality
try:
    from .advanced_cache import (
        CacheLevel,
        MultiLevelCache,
        cache_computation,
        get_global_cache,
    )

    _caching_available = True
except ImportError:
    _caching_available = False

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

# Add optional exports based on availability
if _profiler_available:
    __all__.extend(
        [
            "PerformanceProfiler",
            "global_profiler",
            "profile_algorithm",
            "profile_block",
        ]
    )

if _memory_available:
    __all__.extend(
        [
            "MemoryOptimizer",
            "MemoryTracker",
            "SystemMemoryMonitor",
        ]
    )

if _caching_available:
    __all__.extend(
        [
            "CacheLevel",
            "MultiLevelCache",
            "cache_computation",
            "get_global_cache",
        ]
    )


# Convenience function for basic setup
def setup_basic_utilities(log_level="INFO", enable_profiling=False):
    """
    Setup basic XPCS Toolkit utilities.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    enable_profiling : bool
        Whether to enable performance profiling

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
        "profiler_available": _profiler_available,
        "memory_available": _memory_available,
        "caching_available": _caching_available,
    }

    if enable_profiling and _profiler_available:
        utilities["profiler"] = global_profiler
        logger.info("Performance profiling enabled")

    return utilities


# Add to exports
__all__.append("setup_basic_utilities")
