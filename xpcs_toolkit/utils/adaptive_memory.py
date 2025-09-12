"""
Adaptive Memory Management System for XPCS Toolkit.

This module implements intelligent memory management with adaptive caching,
predictive prefetching, and dynamic resource allocation based on usage patterns.
"""

from __future__ import annotations

import time
import threading
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import numpy as np
from .advanced_cache import get_global_cache, CacheLevel
from .memory_utils import SystemMemoryMonitor
from .computation_cache import get_computation_cache
from .metadata_cache import get_metadata_cache
from .logging_config import get_logger

logger = get_logger(__name__)


class MemoryStrategy(Enum):
    """Memory management strategies."""

    CONSERVATIVE = "conservative"  # Minimize memory usage
    BALANCED = "balanced"  # Balance performance and memory
    AGGRESSIVE = "aggressive"  # Maximize performance


class UsagePattern(Enum):
    """Usage pattern types."""

    SEQUENTIAL = "sequential"  # Files accessed in sequence
    RANDOM = "random"  # Random file access
    BATCH = "batch"  # Multiple files processed together
    INTERACTIVE = "interactive"  # Single file intensive analysis


@dataclass
class AccessRecord:
    """Record of file/data access."""

    timestamp: float
    file_path: str
    data_type: str
    access_duration_ms: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class PredictionModel:
    """Simple prediction model for prefetching."""

    pattern_type: UsagePattern
    confidence: float  # 0.0 to 1.0
    next_files: List[str] = field(default_factory=list)
    next_data_types: List[str] = field(default_factory=list)
    predicted_at: float = field(default_factory=time.time)


class AdaptiveMemoryManager:
    """
    Intelligent memory management with pattern recognition and predictive caching.

    Features:
    - Usage pattern detection
    - Predictive prefetching
    - Dynamic cache sizing
    - Memory pressure response
    - Performance optimization
    """

    def __init__(
        self,
        strategy: MemoryStrategy = MemoryStrategy.BALANCED,
        learning_window_hours: float = 2.0,
        prediction_horizon_minutes: float = 5.0,
        max_prefetch_items: int = 10,
    ):
        self.strategy = strategy
        self.learning_window_hours = learning_window_hours
        self.prediction_horizon_minutes = prediction_horizon_minutes
        self.max_prefetch_items = max_prefetch_items

        # Access tracking
        self._access_history: deque[AccessRecord] = deque(maxlen=1000)
        self._file_sequences: Dict[str, List[str]] = {}  # file -> list of next files
        self._data_type_patterns: Dict[
            str, List[str]
        ] = {}  # file -> typical data types

        # Pattern recognition
        self._current_pattern: Optional[PredictionModel] = None
        self._pattern_confidence_threshold = 0.6

        # Cache references
        self._main_cache = get_global_cache()
        self._computation_cache = get_computation_cache()
        self._metadata_cache = get_metadata_cache()

        # Thread safety
        self._lock = threading.RLock()

        # Background processing
        self._analysis_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Memory thresholds based on strategy
        self._memory_thresholds = self._calculate_memory_thresholds()

        # Performance metrics
        self._prefetch_hits = 0
        self._prefetch_misses = 0
        self._total_memory_saved_mb = 0.0

        self._start_analysis_thread()

        logger.info(f"AdaptiveMemoryManager initialized with {strategy} strategy")

    def _calculate_memory_thresholds(self) -> Dict[str, float]:
        """Calculate memory thresholds based on strategy."""
        if self.strategy == MemoryStrategy.CONSERVATIVE:
            return {
                "cleanup_trigger": 0.75,  # Start cleanup at 75%
                "aggressive_cleanup": 0.85,  # Aggressive cleanup at 85%
                "cache_limit_factor": 0.3,  # Use 30% of available for caching
                "prefetch_limit": 0.80,  # Don't prefetch above 80%
            }
        elif self.strategy == MemoryStrategy.BALANCED:
            return {
                "cleanup_trigger": 0.80,
                "aggressive_cleanup": 0.90,
                "cache_limit_factor": 0.5,
                "prefetch_limit": 0.85,
            }
        else:  # AGGRESSIVE
            return {
                "cleanup_trigger": 0.85,
                "aggressive_cleanup": 0.95,
                "cache_limit_factor": 0.7,
                "prefetch_limit": 0.90,
            }

    def _start_analysis_thread(self):
        """Start background thread for pattern analysis."""
        self._analysis_thread = threading.Thread(
            target=self._analysis_worker, daemon=True
        )
        self._analysis_thread.start()

    def _analysis_worker(self):
        """Background worker for pattern analysis and optimization."""
        while not self._stop_event.wait(30.0):  # Analyze every 30 seconds
            try:
                self._analyze_usage_patterns()
                self._optimize_cache_sizes()
                self._cleanup_old_records()
                self._execute_predictive_prefetch()
            except Exception as e:
                logger.error(f"Error in adaptive memory analysis: {e}")

    def record_access(
        self,
        file_path: str,
        data_type: str,
        access_duration_ms: float = 0.0,
        memory_usage_mb: float = 0.0,
    ):
        """
        Record data access for pattern learning.

        Parameters
        ----------
        file_path : str
            Path to accessed file
        data_type : str
            Type of data accessed ('saxs_2d', 'g2', 'metadata', etc.)
        access_duration_ms : float
            Time taken to access data
        memory_usage_mb : float
            Memory used by the data
        """
        record = AccessRecord(
            timestamp=time.time(),
            file_path=file_path,
            data_type=data_type,
            access_duration_ms=access_duration_ms,
            memory_usage_mb=memory_usage_mb,
        )

        with self._lock:
            self._access_history.append(record)

            # Update sequence tracking
            if len(self._access_history) >= 2:
                prev_record = self._access_history[-2]
                prev_file = prev_record.file_path

                if prev_file != file_path:  # Different file accessed
                    if prev_file not in self._file_sequences:
                        self._file_sequences[prev_file] = []
                    self._file_sequences[prev_file].append(file_path)

                    # Limit sequence length
                    if len(self._file_sequences[prev_file]) > 20:
                        self._file_sequences[prev_file] = self._file_sequences[
                            prev_file
                        ][-20:]

            # Update data type patterns
            if file_path not in self._data_type_patterns:
                self._data_type_patterns[file_path] = []
            if data_type not in self._data_type_patterns[file_path]:
                self._data_type_patterns[file_path].append(data_type)

    def _analyze_usage_patterns(self):
        """Analyze recent access patterns to predict future needs."""
        if len(self._access_history) < 5:
            return

        current_time = time.time()
        cutoff_time = current_time - (self.learning_window_hours * 3600)

        # Get recent accesses
        recent_accesses = [
            record for record in self._access_history if record.timestamp > cutoff_time
        ]

        if len(recent_accesses) < 3:
            return

        # Detect pattern type
        pattern_type, confidence = self._detect_pattern_type(recent_accesses)

        if confidence > self._pattern_confidence_threshold:
            # Predict next files and data types
            next_files = self._predict_next_files(recent_accesses, pattern_type)
            next_data_types = self._predict_next_data_types(recent_accesses)

            with self._lock:
                self._current_pattern = PredictionModel(
                    pattern_type=pattern_type,
                    confidence=confidence,
                    next_files=next_files,
                    next_data_types=next_data_types,
                )

            logger.debug(
                f"Detected usage pattern: {pattern_type} (confidence: {confidence:.2f})"
            )

    def _detect_pattern_type(
        self, recent_accesses: List[AccessRecord]
    ) -> Tuple[UsagePattern, float]:
        """Detect the type of usage pattern from recent accesses."""
        if len(recent_accesses) < 3:
            return UsagePattern.RANDOM, 0.0

        # Extract file paths and timestamps
        files = [record.file_path for record in recent_accesses]
        times = [record.timestamp for record in recent_accesses]

        # Analyze file access patterns
        unique_files = list(set(files))
        file_count = len(unique_files)
        total_accesses = len(files)

        # Calculate time intervals
        intervals = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        avg_interval = np.mean(intervals) if intervals else 0
        interval_std = np.std(intervals) if len(intervals) > 1 else 0

        # Pattern detection logic
        if file_count == 1:
            # Single file intensive usage
            return UsagePattern.INTERACTIVE, 0.9

        elif file_count < total_accesses * 0.3:
            # Few unique files, many accesses - batch processing
            return UsagePattern.BATCH, 0.8

        elif self._is_sequential_pattern(files):
            # Sequential file access pattern
            return UsagePattern.SEQUENTIAL, 0.85

        else:
            # Random access pattern
            confidence = 0.7 if interval_std / avg_interval < 0.5 else 0.5
            return UsagePattern.RANDOM, confidence

    def _is_sequential_pattern(self, files: List[str]) -> bool:
        """Check if files follow a sequential naming pattern."""
        if len(files) < 3:
            return False

        try:
            # Look for numeric sequences in filenames
            import re

            # Extract numbers from filenames
            file_numbers = []
            for file_path in files:
                filename = file_path.split("/")[-1]  # Get basename
                numbers = re.findall(r"\d+", filename)
                if numbers:
                    # Use the longest number sequence
                    longest_num = max(numbers, key=len)
                    file_numbers.append(int(longest_num))

            if len(file_numbers) >= 3:
                # Check if numbers are mostly sequential
                diffs = [
                    file_numbers[i + 1] - file_numbers[i]
                    for i in range(len(file_numbers) - 1)
                ]
                consistent_diffs = sum(
                    1 for d in diffs if abs(d) <= 2
                )  # Allow small gaps
                return consistent_diffs / len(diffs) > 0.7  # 70% consistency

        except Exception:
            pass

        return False

    def _predict_next_files(
        self, recent_accesses: List[AccessRecord], pattern_type: UsagePattern
    ) -> List[str]:
        """Predict next files to be accessed based on pattern."""
        if not recent_accesses:
            return []

        current_file = recent_accesses[-1].file_path

        if pattern_type == UsagePattern.SEQUENTIAL:
            # Predict next files in sequence
            return self._predict_sequential_files(current_file, recent_accesses)

        elif pattern_type == UsagePattern.BATCH:
            # Predict files commonly accessed together
            return self._predict_batch_files(recent_accesses)

        elif pattern_type == UsagePattern.INTERACTIVE:
            # Same file likely to be accessed again with different data types
            return [current_file]

        else:  # RANDOM
            # Use frequency-based prediction
            return self._predict_frequent_files(recent_accesses)

    def _predict_sequential_files(
        self, current_file: str, recent_accesses: List[AccessRecord]
    ) -> List[str]:
        """Predict next files in a sequential pattern."""
        try:
            import re

            # Extract numeric part from current file
            filename = current_file.split("/")[-1]
            numbers = re.findall(r"\d+", filename)

            if not numbers:
                return []

            # Use the longest number for prediction
            longest_num = max(numbers, key=len)
            current_num = int(longest_num)

            # Generate next files by incrementing the number
            predicted_files = []
            for i in range(
                1, min(6, self.max_prefetch_items + 1)
            ):  # Predict next 5 files
                next_num = current_num + i
                next_filename = filename.replace(
                    longest_num, str(next_num).zfill(len(longest_num))
                )
                next_filepath = current_file.replace(filename, next_filename)
                predicted_files.append(next_filepath)

            return predicted_files

        except Exception as e:
            logger.debug(f"Failed to predict sequential files: {e}")
            return []

    def _predict_batch_files(self, recent_accesses: List[AccessRecord]) -> List[str]:
        """Predict files commonly accessed in batch."""
        file_cooccurrence = defaultdict(int)
        recent_files = [
            record.file_path for record in recent_accesses[-10:]
        ]  # Last 10 accesses

        # Count file co-occurrences
        unique_files = list(set(recent_files))
        for i, file1 in enumerate(unique_files):
            for j, file2 in enumerate(unique_files):
                if i != j:
                    file_cooccurrence[file2] += recent_files.count(
                        file1
                    ) * recent_files.count(file2)

        # Return most co-occurring files
        sorted_files = sorted(
            file_cooccurrence.items(), key=lambda x: x[1], reverse=True
        )
        return [file_path for file_path, _ in sorted_files[: self.max_prefetch_items]]

    def _predict_frequent_files(self, recent_accesses: List[AccessRecord]) -> List[str]:
        """Predict based on access frequency."""
        file_freq = defaultdict(int)
        for record in recent_accesses[-20:]:  # Last 20 accesses
            file_freq[record.file_path] += 1

        # Return most frequent files
        sorted_files = sorted(file_freq.items(), key=lambda x: x[1], reverse=True)
        return [file_path for file_path, _ in sorted_files[: self.max_prefetch_items]]

    def _predict_next_data_types(
        self, recent_accesses: List[AccessRecord]
    ) -> List[str]:
        """Predict next data types to be accessed."""
        if not recent_accesses:
            return []

        current_file = recent_accesses[-1].file_path

        # Get typical data types for this file
        typical_types = self._data_type_patterns.get(current_file, [])

        # Get recently accessed data types
        recent_types = [record.data_type for record in recent_accesses[-5:]]

        # Predict missing typical types and frequent recent types
        predicted_types = []

        # Add typical types not recently accessed
        for data_type in typical_types:
            if data_type not in recent_types:
                predicted_types.append(data_type)

        # Add frequently accessed types
        type_freq = defaultdict(int)
        for data_type in recent_types:
            type_freq[data_type] += 1

        for data_type, freq in sorted(
            type_freq.items(), key=lambda x: x[1], reverse=True
        ):
            if data_type not in predicted_types:
                predicted_types.append(data_type)

        return predicted_types[:5]  # Limit to 5 data types

    def _execute_predictive_prefetch(self):
        """Execute predictive prefetching based on current pattern."""
        if not self._current_pattern:
            return

        # Check memory pressure before prefetching
        memory_pressure = SystemMemoryMonitor.get_memory_info()[2] / 100.0
        if memory_pressure > self._memory_thresholds["prefetch_limit"]:
            logger.debug(
                f"Skipping prefetch due to memory pressure: {memory_pressure * 100:.1f}%"
            )
            return

        prediction_age = time.time() - self._current_pattern.predicted_at
        if prediction_age > (self.prediction_horizon_minutes * 60):
            return  # Prediction too old

        # Execute prefetch for predicted files
        prefetch_count = 0
        for file_path in self._current_pattern.next_files:
            if prefetch_count >= self.max_prefetch_items:
                break

            # Check if file exists before attempting prefetch
            try:
                import os

                if os.path.exists(file_path):
                    self._prefetch_file_data(
                        file_path, self._current_pattern.next_data_types
                    )
                    prefetch_count += 1
            except Exception as e:
                logger.debug(f"Failed to prefetch {file_path}: {e}")

        if prefetch_count > 0:
            logger.debug(f"Executed predictive prefetch for {prefetch_count} files")

    def _prefetch_file_data(self, file_path: str, data_types: List[str]):
        """Prefetch specific data types for a file."""
        # Use metadata cache for prefetching
        if "metadata" in data_types:
            self._metadata_cache.warm_cache([file_path], ["metadata"])

        # Could add more specific prefetch logic for other data types
        # This is a simplified implementation

    def _optimize_cache_sizes(self):
        """Dynamically optimize cache sizes based on usage patterns and memory pressure."""
        memory_pressure = SystemMemoryMonitor.get_memory_info()[2] / 100.0

        # Get current cache statistics
        cache_stats = self._main_cache.get_stats()
        hit_rates = cache_stats.get("hit_rates", {})

        # Adjust cache sizes based on performance and memory pressure
        if memory_pressure > self._memory_thresholds["aggressive_cleanup"]:
            # Aggressive cleanup mode
            self._perform_aggressive_cleanup()

        elif memory_pressure > self._memory_thresholds["cleanup_trigger"]:
            # Normal cleanup mode
            self._perform_normal_cleanup()

        # Log optimization decisions
        if memory_pressure > 0.8:
            logger.debug(
                f"Cache optimization: memory_pressure={memory_pressure * 100:.1f}%, "
                f"hit_rates={hit_rates}"
            )

    def _perform_aggressive_cleanup(self):
        """Perform aggressive cache cleanup to free memory."""
        # Clear L1 cache more aggressively
        self._main_cache.clear(CacheLevel.L1)

        # Force promotion from L2 to L3
        self._main_cache._promote_l2_to_l3(force=True, target_freed_mb=200.0)

        # Clean up computation caches
        self._computation_cache.cleanup_old_computations(max_age_hours=12.0)

        # Clean up metadata cache
        self._metadata_cache.cleanup_expired_metadata(max_age_hours=24.0)

        logger.info("Performed aggressive cache cleanup due to high memory pressure")

    def _perform_normal_cleanup(self):
        """Perform normal cache cleanup."""
        # Moderate L1 to L2 promotion
        self._main_cache._promote_l1_to_l2(force=False, target_freed_mb=100.0)

        # Clean up old computations
        self._computation_cache.cleanup_old_computations(max_age_hours=24.0)

        logger.debug("Performed normal cache cleanup")

    def _cleanup_old_records(self):
        """Clean up old access records to prevent memory growth."""
        cutoff_time = time.time() - (
            self.learning_window_hours * 2 * 3600
        )  # Keep 2x learning window

        with self._lock:
            # Clean access history (deque automatically limits size)

            # Clean file sequences
            for file_path in list(self._file_sequences.keys()):
                if not any(
                    record.file_path == file_path and record.timestamp > cutoff_time
                    for record in self._access_history
                ):
                    del self._file_sequences[file_path]

    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory management recommendations."""
        memory_info = SystemMemoryMonitor.get_memory_info()
        used_mb, available_mb, percent_used = memory_info

        cache_stats = self._main_cache.get_stats()
        hit_rates = cache_stats.get("hit_rates", {})

        recommendations = {
            "current_memory_pressure": percent_used,
            "recommended_strategy": self.strategy.value,
            "cache_performance": hit_rates,
            "memory_allocation": {
                "total_system_mb": used_mb + available_mb,
                "available_mb": available_mb,
                "used_percentage": percent_used,
                "recommended_cache_limit_mb": available_mb
                * self._memory_thresholds["cache_limit_factor"],
            },
        }

        # Add specific recommendations based on current state
        if percent_used > 90:
            recommendations["urgent_actions"] = [
                "Enable conservative memory strategy",
                "Clear L1 cache",
                "Reduce prefetch queue size",
                "Consider increasing system memory",
            ]
        elif percent_used > 80:
            recommendations["suggested_actions"] = [
                "Monitor memory usage closely",
                "Consider reducing cache sizes",
                "Enable more aggressive cleanup",
            ]
        else:
            recommendations["optimizations"] = [
                "Current memory usage is healthy",
                "Consider increasing cache sizes for better performance",
                "Enable more aggressive prefetching if pattern confidence is high",
            ]

        return recommendations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._lock:
            total_prefetch_attempts = self._prefetch_hits + self._prefetch_misses
            prefetch_hit_rate = (
                self._prefetch_hits / total_prefetch_attempts
                if total_prefetch_attempts > 0
                else 0.0
            )

            stats = {
                "pattern_recognition": {
                    "current_pattern": self._current_pattern.pattern_type.value
                    if self._current_pattern
                    else "none",
                    "pattern_confidence": self._current_pattern.confidence
                    if self._current_pattern
                    else 0.0,
                    "learning_window_hours": self.learning_window_hours,
                    "access_records": len(self._access_history),
                },
                "prefetch_performance": {
                    "hit_rate": prefetch_hit_rate,
                    "total_hits": self._prefetch_hits,
                    "total_misses": self._prefetch_misses,
                    "memory_saved_mb": self._total_memory_saved_mb,
                },
                "cache_integration": {
                    "main_cache_stats": self._main_cache.get_stats(),
                    "computation_cache_stats": self._computation_cache.get_computation_stats(),
                    "metadata_cache_stats": self._metadata_cache.get_cache_statistics(),
                },
            }

            return stats

    def shutdown(self):
        """Shutdown adaptive memory manager gracefully."""
        logger.info("Shutting down AdaptiveMemoryManager")

        # Stop analysis thread
        self._stop_event.set()
        if self._analysis_thread and self._analysis_thread.is_alive():
            self._analysis_thread.join(timeout=5.0)

        # Log final performance stats
        stats = self.get_performance_stats()
        prefetch_stats = stats["prefetch_performance"]
        logger.info(
            f"Final stats - Prefetch hit rate: {prefetch_stats['hit_rate']:.2f}, "
            f"Memory saved: {prefetch_stats['memory_saved_mb']:.1f}MB"
        )


# Global adaptive memory manager instance
_global_memory_manager: Optional[AdaptiveMemoryManager] = None


def get_adaptive_memory_manager(
    strategy: MemoryStrategy = MemoryStrategy.BALANCED, reset: bool = False
) -> AdaptiveMemoryManager:
    """
    Get or create global adaptive memory manager.

    Parameters
    ----------
    strategy : MemoryStrategy
        Memory management strategy
    reset : bool
        Whether to reset existing manager

    Returns
    -------
    AdaptiveMemoryManager
        Global memory manager instance
    """
    global _global_memory_manager

    if _global_memory_manager is None or reset:
        if _global_memory_manager is not None:
            _global_memory_manager.shutdown()

        _global_memory_manager = AdaptiveMemoryManager(strategy=strategy)

    return _global_memory_manager


def smart_cache_decorator(data_type: str, memory_cost_mb: float = 0.0):
    """
    Smart caching decorator that integrates with adaptive memory management.

    Parameters
    ----------
    data_type : str
        Type of data being cached
    memory_cost_mb : float
        Estimated memory cost of the operation

    Returns
    -------
    callable
        Decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = get_adaptive_memory_manager()

            # Record access attempt
            file_path = kwargs.get("file_path", args[0] if args else "unknown")
            start_time = time.time()

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Record successful access
                access_duration_ms = (time.time() - start_time) * 1000.0
                manager.record_access(
                    file_path=str(file_path),
                    data_type=data_type,
                    access_duration_ms=access_duration_ms,
                    memory_usage_mb=memory_cost_mb,
                )

                return result

            except Exception:
                # Record failed access
                access_duration_ms = (time.time() - start_time) * 1000.0
                manager.record_access(
                    file_path=str(file_path),
                    data_type=f"{data_type}_failed",
                    access_duration_ms=access_duration_ms,
                    memory_usage_mb=0.0,
                )
                raise

        return wrapper

    return decorator
