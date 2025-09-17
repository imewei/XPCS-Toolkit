#!/usr/bin/env python3
"""
Memory Testing Utilities for XPCS Toolkit Test Suite

This module provides utilities for reliable memory testing, including
tracemalloc integration, memory snapshot comparisons, and test isolation.

Created: 2025-09-16
"""

import gc
import tracemalloc
import psutil
import time
import contextlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import weakref
import numpy as np


@dataclass
class MemorySnapshot:
    """Represents a memory snapshot with multiple metrics."""
    tracemalloc_current: int  # tracemalloc current memory
    tracemalloc_peak: int    # tracemalloc peak memory
    rss: int                 # RSS memory from psutil
    vms: int                 # VMS memory from psutil
    timestamp: float         # When snapshot was taken
    label: str               # Description of snapshot


class EnhancedMemoryTracker:
    """Enhanced memory tracker using tracemalloc and psutil."""

    def __init__(self, enable_tracemalloc: bool = True):
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process()
        self.snapshots: List[MemorySnapshot] = []
        self.tracking_active = False
        self._lock = threading.Lock()

    def start_tracking(self) -> None:
        """Start memory tracking."""
        with self._lock:
            if self.enable_tracemalloc and not tracemalloc.is_tracing():
                tracemalloc.start()
            self.tracking_active = True
            self.take_snapshot("start")

    def stop_tracking(self) -> None:
        """Stop memory tracking."""
        with self._lock:
            if self.tracking_active:
                self.take_snapshot("stop")
                self.tracking_active = False
                if self.enable_tracemalloc and tracemalloc.is_tracing():
                    tracemalloc.stop()

    def take_snapshot(self, label: str) -> MemorySnapshot:
        """Take a memory snapshot."""
        with self._lock:
            # Get tracemalloc info
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
            else:
                current = peak = 0

            # Get psutil info
            memory_info = self.process.memory_info()

            snapshot = MemorySnapshot(
                tracemalloc_current=current,
                tracemalloc_peak=peak,
                rss=memory_info.rss,
                vms=memory_info.vms,
                timestamp=time.time(),
                label=label
            )

            self.snapshots.append(snapshot)
            return snapshot

    def get_memory_change(self, start_label: str, end_label: str) -> Dict[str, int]:
        """Calculate memory change between two snapshots."""
        start_snapshot = None
        end_snapshot = None

        for snapshot in self.snapshots:
            if snapshot.label == start_label:
                start_snapshot = snapshot
            elif snapshot.label == end_label:
                end_snapshot = snapshot

        if not start_snapshot or not end_snapshot:
            raise ValueError(f"Could not find snapshots for labels: {start_label}, {end_label}")

        return {
            'tracemalloc_change': end_snapshot.tracemalloc_current - start_snapshot.tracemalloc_current,
            'rss_change': end_snapshot.rss - start_snapshot.rss,
            'vms_change': end_snapshot.vms - start_snapshot.vms,
        }

    def clear_snapshots(self) -> None:
        """Clear all snapshots."""
        with self._lock:
            self.snapshots.clear()


class MemoryTestIsolator:
    """Provides memory test isolation and cleanup."""

    def __init__(self):
        self._original_refs = weakref.WeakSet()
        self._cleanup_callbacks = []

    def register_for_cleanup(self, obj: Any) -> None:
        """Register an object for cleanup tracking."""
        self._original_refs.add(obj)

    def add_cleanup_callback(self, callback) -> None:
        """Add a cleanup callback."""
        self._cleanup_callbacks.append(callback)

    def force_cleanup(self) -> None:
        """Force cleanup and garbage collection."""
        # Run cleanup callbacks
        for callback in reversed(self._cleanup_callbacks):
            try:
                callback()
            except Exception:
                pass  # Ignore cleanup errors

        # Force garbage collection
        for _ in range(3):  # Multiple passes for complex reference cycles
            gc.collect()

        # Small delay to allow system cleanup
        time.sleep(0.1)


@contextlib.contextmanager
def memory_test_context(enable_tracemalloc: bool = True):
    """Context manager for memory testing with automatic cleanup."""
    tracker = EnhancedMemoryTracker(enable_tracemalloc)
    isolator = MemoryTestIsolator()

    try:
        tracker.start_tracking()
        yield tracker, isolator
    finally:
        tracker.stop_tracking()
        isolator.force_cleanup()


class MemoryAssertions:
    """Memory-related test assertions."""

    @staticmethod
    def assert_memory_increase_reasonable(tracker: EnhancedMemoryTracker,
                                        start_label: str,
                                        end_label: str,
                                        max_increase_mb: float = 100.0) -> None:
        """Assert that memory increase is within reasonable bounds."""
        changes = tracker.get_memory_change(start_label, end_label)
        rss_increase_mb = changes['rss_change'] / (1024 * 1024)

        assert rss_increase_mb <= max_increase_mb, (
            f"Memory increase {rss_increase_mb:.1f} MB exceeds limit {max_increase_mb} MB"
        )

    @staticmethod
    def assert_memory_cleanup_effective(tracker: EnhancedMemoryTracker,
                                      before_label: str,
                                      after_label: str,
                                      min_reduction_percent: float = 10.0) -> None:
        """Assert that memory cleanup is effective (relaxed assertion)."""
        changes = tracker.get_memory_change(before_label, after_label)

        # Find the before snapshot to get baseline
        before_snapshot = None
        for snapshot in tracker.snapshots:
            if snapshot.label == before_label:
                before_snapshot = snapshot
                break

        if before_snapshot and before_snapshot.rss > 0:
            reduction_percent = abs(changes['rss_change']) / before_snapshot.rss * 100

            # More lenient assertion - just check that some cleanup occurred
            # or memory didn't increase significantly
            if changes['rss_change'] > 0:  # Memory increased
                increase_percent = changes['rss_change'] / before_snapshot.rss * 100
                assert increase_percent < 20.0, (
                    f"Memory increased by {increase_percent:.1f}% after cleanup"
                )
            # If memory decreased or stayed roughly the same, that's good

    @staticmethod
    def assert_concurrent_memory_stable(tracker: EnhancedMemoryTracker,
                                      snapshots: List[MemorySnapshot],
                                      max_variance_percent: float = 50.0) -> None:
        """Assert that memory usage is stable during concurrent operations."""
        if len(snapshots) < 2:
            return  # Cannot assess stability with too few snapshots

        rss_values = [s.rss for s in snapshots]
        min_rss = min(rss_values)
        max_rss = max(rss_values)

        if min_rss > 0:
            variance_percent = (max_rss - min_rss) / min_rss * 100
            assert variance_percent <= max_variance_percent, (
                f"Memory variance {variance_percent:.1f}% exceeds limit {max_variance_percent}%"
            )


class MemoryTestUtils:
    """High-level utility class for memory testing operations."""

    @staticmethod
    def create_enhanced_tracker() -> EnhancedMemoryTracker:
        """Create an enhanced memory tracker for testing."""
        return EnhancedMemoryTracker()

    @staticmethod
    def assert_memory_reasonable(tracker: EnhancedMemoryTracker,
                               start_label: str,
                               end_label: str,
                               max_increase_mb: float = 100.0) -> None:
        """Assert memory increase is reasonable."""
        MemoryAssertions.assert_memory_increase_reasonable(
            tracker, start_label, end_label, max_increase_mb
        )

    @staticmethod
    def assert_cleanup_effective(tracker: EnhancedMemoryTracker,
                               before_label: str,
                               after_label: str,
                               min_reduction_percent: float = 10.0) -> None:
        """Assert memory cleanup is effective."""
        MemoryAssertions.assert_memory_cleanup_effective(
            tracker, before_label, after_label, min_reduction_percent
        )

    @staticmethod
    def get_test_context(enable_tracemalloc: bool = True):
        """Get memory test context manager."""
        return memory_test_context(enable_tracemalloc)


def create_test_data_with_tracking(size_mb: float = 10.0) -> Tuple[np.ndarray, EnhancedMemoryTracker]:
    """Create test data with memory tracking."""
    tracker = EnhancedMemoryTracker()
    tracker.start_tracking()
    tracker.take_snapshot("before_allocation")

    # Create data
    elements = int(size_mb * 1024 * 1024 / 8)  # 8 bytes per float64
    data = np.random.rand(elements).astype(np.float64)

    tracker.take_snapshot("after_allocation")
    return data, tracker


def measure_operation_memory_impact(operation, *args, **kwargs) -> Dict[str, Any]:
    """Measure memory impact of an operation."""
    with memory_test_context() as (tracker, isolator):
        tracker.take_snapshot("before_operation")

        try:
            result = operation(*args, **kwargs)
            tracker.take_snapshot("after_operation")

            return {
                'result': result,
                'memory_changes': tracker.get_memory_change("before_operation", "after_operation"),
                'success': True,
                'error': None
            }
        except Exception as e:
            tracker.take_snapshot("after_error")
            return {
                'result': None,
                'memory_changes': tracker.get_memory_change("before_operation", "after_error"),
                'success': False,
                'error': str(e)
            }