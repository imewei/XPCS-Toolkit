"""Memory Testing Utilities.

Minimal implementation to support Qt error detection tests.
"""

import os

import psutil


class MemoryTestUtils:
    """Memory testing utilities for test framework."""

    @staticmethod
    def get_memory_usage():
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except Exception:
            # Fallback if psutil is not available
            return 0

    @staticmethod
    def get_memory_percent():
        """Get current memory usage as percentage."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_percent()
        except Exception:
            return 0.0
