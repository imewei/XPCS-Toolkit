"""
Enhanced Qt signal optimization system for XPCS Toolkit threading performance.

This module provides optimizations for:
- Signal batching and deduplication
- Smart signal routing and connection pooling
- Attribute caching for hot paths
- Signal emission rate limiting
- Connection lifecycle management
"""

from __future__ import annotations

import threading
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import QMutex, QMutexLocker, QObject, QTimer, Signal

from xpcs_toolkit.utils.logging_config import get_logger

logger = get_logger(__name__)


class SignalPriority(Enum):
    """Priority levels for signal emission."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SignalBatch:
    """Container for batched signals."""

    signal_name: str
    signal_object: Signal
    batched_args: List[Tuple]
    priority: SignalPriority = SignalPriority.NORMAL
    timestamp: float = field(default_factory=time.perf_counter)
    batch_size_limit: int = 50
    time_limit: float = 0.05  # 50ms max batching delay

    def add_emission(self, args: Tuple) -> bool:
        """Add an emission to the batch. Returns True if batch should be flushed."""
        self.batched_args.append(args)
        current_time = time.perf_counter()

        # Flush if we hit size limit or time limit
        return (
            len(self.batched_args) >= self.batch_size_limit
            or current_time - self.timestamp >= self.time_limit
        )


@dataclass
class ConnectionInfo:
    """Information about a signal connection."""

    signal_object: Signal
    slot_func: Callable
    connection_type: Any  # Qt.ConnectionType
    weak_ref: Optional[weakref.ReferenceType] = None
    usage_count: int = 0
    last_used: float = field(default_factory=time.perf_counter)
    is_batched: bool = False


class SignalBatcher(QObject):
    """
    Batches frequent signals to reduce GUI update overhead.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._batches: Dict[str, SignalBatch] = {}
        self._batch_timer = QTimer()
        self._batch_timer.timeout.connect(self._flush_all_batches)
        self._batch_timer.start(16)  # ~60 FPS batch flushing

        self._mutex = QMutex()
        self._total_batched_signals = 0
        self._total_emissions = 0
        self._batching_enabled = True

        # Signal patterns that benefit from batching
        self._batchable_signals = {
            "progress",  # Progress updates
            "status",  # Status messages
            "resource_usage",  # Resource monitoring
            "state_changed",  # State transitions
            "partial_result",  # Partial results
        }

        # Signals that should never be batched (critical events)
        self._critical_signals = {
            "error",
            "finished",
            "cancelled",
            "started",
            "retry_attempt",
        }

    def should_batch_signal(self, signal_name: str) -> bool:
        """Determine if a signal should be batched."""
        if not self._batching_enabled:
            return False

        if signal_name in self._critical_signals:
            return False

        return signal_name in self._batchable_signals

    def batch_signal_emission(
        self,
        signal_name: str,
        signal_object: Signal,
        args: Tuple,
        priority: SignalPriority = SignalPriority.NORMAL,
    ):
        """Add a signal emission to the batch."""
        if not self.should_batch_signal(signal_name):
            # Emit immediately for non-batchable signals
            signal_object.emit(*args)
            self._total_emissions += 1
            return

        batch_key = f"{id(signal_object)}_{signal_name}"

        with QMutexLocker(self._mutex):
            if batch_key not in self._batches:
                self._batches[batch_key] = SignalBatch(
                    signal_name=signal_name,
                    signal_object=signal_object,
                    batched_args=[],
                    priority=priority,
                )

            batch = self._batches[batch_key]
            should_flush = batch.add_emission(args)
            self._total_batched_signals += 1

            if should_flush or priority == SignalPriority.CRITICAL:
                self._flush_batch(batch_key)

    def _flush_batch(self, batch_key: str):
        """Flush a specific batch."""
        if batch_key not in self._batches:
            return

        batch = self._batches[batch_key]
        if not batch.batched_args:
            return

        # For progress signals, only emit the latest value to reduce UI updates
        if batch.signal_name == "progress":
            # Emit only the most recent progress update
            latest_args = batch.batched_args[-1]
            batch.signal_object.emit(*latest_args)
            self._total_emissions += 1
        elif batch.signal_name == "resource_usage":
            # For resource usage, emit the peak values in the batch
            if batch.batched_args:
                # Assuming resource_usage has format (worker_id, cpu_percent, memory_mb)
                worker_id = batch.batched_args[0][0] if batch.batched_args[0] else ""
                max_cpu = max(
                    args[1] if len(args) > 1 else 0 for args in batch.batched_args
                )
                max_memory = max(
                    args[2] if len(args) > 2 else 0 for args in batch.batched_args
                )
                batch.signal_object.emit(worker_id, max_cpu, max_memory)
                self._total_emissions += 1
        else:
            # For other signals, emit all batched signals (but potentially at reduced rate)
            for args in batch.batched_args:
                batch.signal_object.emit(*args)
                self._total_emissions += 1

        # Clear the batch
        batch.batched_args.clear()
        del self._batches[batch_key]

    def _flush_all_batches(self):
        """Flush all pending batches."""
        with QMutexLocker(self._mutex):
            batch_keys = list(self._batches.keys())

        for batch_key in batch_keys:
            self._flush_batch(batch_key)

    def get_statistics(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return {
            "total_batched_signals": self._total_batched_signals,
            "total_emissions": self._total_emissions,
            "reduction_ratio": (self._total_batched_signals - self._total_emissions)
            / max(self._total_batched_signals, 1),
            "active_batches": len(self._batches),
            "batching_enabled": self._batching_enabled,
        }

    def set_batching_enabled(self, enabled: bool):
        """Enable or disable signal batching."""
        self._batching_enabled = enabled
        if not enabled:
            self._flush_all_batches()


class ConnectionPool(QObject):
    """
    Manages Qt signal connections with pooling and lifecycle optimization.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._connections: Dict[str, ConnectionInfo] = {}
        self._connection_cache: Dict[str, List[ConnectionInfo]] = defaultdict(list)
        self._cleanup_timer = QTimer()
        self._cleanup_timer.timeout.connect(self._cleanup_stale_connections)
        self._cleanup_timer.start(30000)  # Clean up every 30 seconds

        self._connection_count = 0
        self._reused_connections = 0
        self._mutex = QMutex()

    def get_connection_key(
        self, signal_obj: QObject, signal_name: str, slot_obj: QObject, slot_name: str
    ) -> str:
        """Generate a unique key for a connection."""
        return f"{id(signal_obj)}_{signal_name}_{id(slot_obj)}_{slot_name}"

    def create_optimized_connection(
        self,
        signal_obj: QObject,
        signal_name: str,
        slot_obj: QObject,
        slot_name: str,
        connection_type: Any = None,
    ) -> bool:
        """Create an optimized signal-slot connection with pooling."""
        connection_key = self.get_connection_key(
            signal_obj, signal_name, slot_obj, slot_name
        )

        with QMutexLocker(self._mutex):
            # Check if connection already exists
            if connection_key in self._connections:
                self._connections[connection_key].usage_count += 1
                self._connections[connection_key].last_used = time.perf_counter()
                self._reused_connections += 1
                return True

            # Create new connection
            signal = getattr(signal_obj, signal_name, None)
            slot = getattr(slot_obj, slot_name, None)

            if signal is None or slot is None:
                return False

            # Create the connection
            try:
                if connection_type is not None:
                    signal.connect(slot, connection_type)
                else:
                    signal.connect(slot)

                # Track the connection
                conn_info = ConnectionInfo(
                    signal_object=signal,
                    slot_func=slot,
                    connection_type=connection_type,
                    weak_ref=weakref.ref(slot_obj)
                    if hasattr(slot_obj, "__weakref__")
                    else None,
                    usage_count=1,
                )

                self._connections[connection_key] = conn_info
                self._connection_count += 1
                return True

            except Exception as e:
                logger.error(f"Failed to create connection {connection_key}: {e}")
                return False

    def disconnect_optimized(
        self, signal_obj: QObject, signal_name: str, slot_obj: QObject, slot_name: str
    ) -> bool:
        """Disconnect with connection pool management."""
        connection_key = self.get_connection_key(
            signal_obj, signal_name, slot_obj, slot_name
        )

        with QMutexLocker(self._mutex):
            if connection_key in self._connections:
                conn_info = self._connections[connection_key]

                try:
                    conn_info.signal_object.disconnect(conn_info.slot_func)
                    del self._connections[connection_key]
                    return True
                except Exception as e:
                    logger.error(f"Failed to disconnect {connection_key}: {e}")
                    return False

        return False

    def _cleanup_stale_connections(self):
        """Clean up stale or broken connections."""
        current_time = time.perf_counter()
        stale_connections = []

        with QMutexLocker(self._mutex):
            for key, conn_info in self._connections.items():
                # Check if the weak reference is still valid
                if conn_info.weak_ref and conn_info.weak_ref() is None:
                    stale_connections.append(key)
                # Check if connection hasn't been used in a long time
                elif current_time - conn_info.last_used > 300:  # 5 minutes
                    stale_connections.append(key)

            # Remove stale connections
            for key in stale_connections:
                if key in self._connections:
                    del self._connections[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "total_connections": self._connection_count,
            "active_connections": len(self._connections),
            "reused_connections": self._reused_connections,
            "reuse_ratio": self._reused_connections / max(self._connection_count, 1),
        }


class WorkerAttributeCache:
    """
    High-performance cache for frequently accessed worker attributes.
    """

    def __init__(self, cache_size_limit: int = 1000):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._cache_size_limit = cache_size_limit
        self._mutex = threading.Lock()

        # Statistics
        self._hit_count = 0
        self._miss_count = 0

    def get_cache_key(self, worker_obj: Any, attr_name: str) -> str:
        """Generate cache key for worker attribute."""
        return f"{id(worker_obj)}_{attr_name}"

    def get_cached_attribute(
        self, worker_obj: Any, attr_name: str, compute_func: Optional[Callable] = None
    ) -> Any:
        """Get cached attribute value or compute and cache it."""
        cache_key = self.get_cache_key(worker_obj, attr_name)

        with self._mutex:
            # Check if we have cached value
            if cache_key in self._cache:
                self._hit_count += 1
                self._access_times[cache_key] = time.perf_counter()
                self._access_counts[cache_key] += 1
                return self._cache[cache_key]

            # Cache miss - compute value
            self._miss_count += 1

            if compute_func:
                value = compute_func()
            else:
                value = getattr(worker_obj, attr_name, None)

            # Cache the value
            self._cache[cache_key] = value
            self._access_times[cache_key] = time.perf_counter()
            self._access_counts[cache_key] = 1

            # Evict if cache is too large
            if len(self._cache) > self._cache_size_limit:
                self._evict_least_used()

            return value

    def invalidate_cache(self, worker_obj: Any, attr_name: Optional[str] = None):
        """Invalidate cached attributes for a worker."""
        with self._mutex:
            if attr_name:
                cache_key = self.get_cache_key(worker_obj, attr_name)
                self._cache.pop(cache_key, None)
                self._access_times.pop(cache_key, None)
                self._access_counts.pop(cache_key, None)
            else:
                # Invalidate all attributes for this worker
                worker_id = id(worker_obj)
                keys_to_remove = [
                    key for key in self._cache.keys() if key.startswith(f"{worker_id}_")
                ]
                for key in keys_to_remove:
                    self._cache.pop(key, None)
                    self._access_times.pop(key, None)
                    self._access_counts.pop(key, None)

    def _evict_least_used(self):
        """Evict least recently used items."""
        if not self._access_times:
            return

        # Remove 10% of least recently used items
        sorted_by_time = sorted(self._access_times.items(), key=lambda x: x[1])
        num_to_evict = max(1, len(sorted_by_time) // 10)

        for i in range(num_to_evict):
            key_to_evict = sorted_by_time[i][0]
            self._cache.pop(key_to_evict, None)
            self._access_times.pop(key_to_evict, None)
            self._access_counts.pop(key_to_evict, None)

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._hit_count + self._miss_count
        return {
            "cache_size": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_ratio": self._hit_count / max(total_requests, 1),
            "total_requests": total_requests,
        }


class SignalOptimizer(QObject):
    """
    Main signal optimization coordinator that combines all optimization techniques.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signal_batcher = SignalBatcher(self)
        self.connection_pool = ConnectionPool(self)
        self.attribute_cache = WorkerAttributeCache()

        self._optimization_enabled = True
        self._start_time = time.perf_counter()

    def emit_optimized(
        self,
        signal_obj: Signal,
        signal_name: str,
        args: Tuple,
        priority: SignalPriority = SignalPriority.NORMAL,
    ):
        """Emit signal through optimization pipeline."""
        if not self._optimization_enabled:
            signal_obj.emit(*args)
            return

        self.signal_batcher.batch_signal_emission(
            signal_name, signal_obj, args, priority
        )

    def create_connection(
        self,
        signal_obj: QObject,
        signal_name: str,
        slot_obj: QObject,
        slot_name: str,
        connection_type: Any = None,
    ) -> bool:
        """Create optimized signal connection."""
        if not self._optimization_enabled:
            signal = getattr(signal_obj, signal_name, None)
            slot = getattr(slot_obj, slot_name, None)
            if signal and slot:
                signal.connect(
                    slot, connection_type
                ) if connection_type else signal.connect(slot)
                return True
            return False

        return self.connection_pool.create_optimized_connection(
            signal_obj, signal_name, slot_obj, slot_name, connection_type
        )

    def disconnect(
        self, signal_obj: QObject, signal_name: str, slot_obj: QObject, slot_name: str
    ) -> bool:
        """Disconnect optimized signal connection."""
        if not self._optimization_enabled:
            signal = getattr(signal_obj, signal_name, None)
            slot = getattr(slot_obj, slot_name, None)
            if signal and slot:
                signal.disconnect(slot)
                return True
            return False

        return self.connection_pool.disconnect_optimized(
            signal_obj, signal_name, slot_obj, slot_name
        )

    def get_cached_attribute(
        self, worker_obj: Any, attr_name: str, compute_func: Optional[Callable] = None
    ) -> Any:
        """Get cached worker attribute."""
        if not self._optimization_enabled:
            return getattr(worker_obj, attr_name, None)

        return self.attribute_cache.get_cached_attribute(
            worker_obj, attr_name, compute_func
        )

    def invalidate_worker_cache(self, worker_obj: Any, attr_name: Optional[str] = None):
        """Invalidate cached attributes for a worker."""
        self.attribute_cache.invalidate_cache(worker_obj, attr_name)

    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        uptime = time.perf_counter() - self._start_time

        return {
            "uptime_seconds": uptime,
            "optimization_enabled": self._optimization_enabled,
            "signal_batching": self.signal_batcher.get_statistics(),
            "connection_pool": self.connection_pool.get_statistics(),
            "attribute_cache": self.attribute_cache.get_statistics(),
        }

    def set_optimization_enabled(self, enabled: bool):
        """Enable or disable all optimizations."""
        self._optimization_enabled = enabled
        self.signal_batcher.set_batching_enabled(enabled)


# Global signal optimizer instance
_global_signal_optimizer: Optional[SignalOptimizer] = None


def get_signal_optimizer() -> SignalOptimizer:
    """Get the global signal optimizer instance."""
    global _global_signal_optimizer
    if _global_signal_optimizer is None:
        _global_signal_optimizer = SignalOptimizer()
    return _global_signal_optimizer


def initialize_signal_optimization(parent: Optional[QObject] = None) -> SignalOptimizer:
    """Initialize the global signal optimization system."""
    global _global_signal_optimizer
    if _global_signal_optimizer is None:
        _global_signal_optimizer = SignalOptimizer(parent)
    return _global_signal_optimizer


def shutdown_signal_optimization():
    """Shutdown the global signal optimization system."""
    global _global_signal_optimizer
    if _global_signal_optimizer:
        _global_signal_optimizer.set_optimization_enabled(False)
        _global_signal_optimizer = None
