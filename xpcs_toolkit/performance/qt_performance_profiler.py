"""
Qt Performance Profiler and Analysis System.

This module provides comprehensive performance profiling capabilities for the Qt
compliance system, analyzing overhead, memory usage patterns, and optimization
opportunities.
"""

import cProfile
import gc
import pstats
import sys
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from PySide6.QtCore import QObject, QTimer, Signal, QThread, QMutex, QMutexLocker
from PySide6.QtWidgets import QApplication, QWidget

from ..monitoring import get_qt_error_detector, get_performance_metrics_collector
from ..threading import get_qt_compliant_thread_manager
from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ProfilerType(Enum):
    """Types of profilers available."""

    CPROFILE = "cprofile"
    MEMORY = "memory"
    QT_OVERHEAD = "qt_overhead"
    FILTERING = "filtering"
    CUSTOM = "custom"


class ProfilingScope(Enum):
    """Scope of profiling operation."""

    FUNCTION = "function"
    MODULE = "module"
    SYSTEM = "system"
    THREAD = "thread"


@dataclass
class ProfilerConfiguration:
    """Configuration for performance profiling."""

    profiler_type: ProfilerType = ProfilerType.CPROFILE
    scope: ProfilingScope = ProfilingScope.FUNCTION
    duration_seconds: float = 10.0
    sample_interval: float = 0.01
    enable_memory_tracking: bool = True
    enable_qt_tracking: bool = True
    output_directory: Optional[str] = None
    detailed_analysis: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class PerformanceProfile:
    """Performance profiling results."""

    timestamp: float
    profiler_type: ProfilerType
    scope: ProfilingScope
    duration_seconds: float

    # CPU Profiling Results
    total_calls: int = 0
    total_time: float = 0.0
    cumulative_time: float = 0.0
    function_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Memory Profiling Results
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    memory_allocations: int = 0
    gc_collections: int = 0

    # Qt-Specific Results
    qt_operations_count: int = 0
    qt_overhead_percent: float = 0.0
    qt_warning_processing_time: float = 0.0
    qt_signal_emissions: int = 0

    # Optimization Opportunities
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    optimization_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class FilteringPerformanceAnalysis:
    """Analysis of Qt warning filtering performance."""

    warning_volume: int
    processing_time_ms: float
    throughput_warnings_per_second: float
    memory_overhead_mb: float
    cpu_overhead_percent: float
    filter_efficiency: float
    bottlenecks: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class QtPerformanceProfiler(QObject):
    """
    Comprehensive Qt performance profiler and analyzer.

    Provides:
    - CPU profiling of Qt operations
    - Memory usage analysis
    - Qt-specific overhead measurement
    - Warning filtering performance analysis
    - Optimization opportunity identification
    """

    # Signals
    profiling_started = Signal(str)  # profiler_id
    profiling_completed = Signal(str, object)  # profiler_id, PerformanceProfile
    analysis_completed = Signal(object)  # Analysis results

    def __init__(self, parent: QObject = None):
        """Initialize Qt performance profiler."""
        super().__init__(parent)

        self._mutex = QMutex()
        self._active_profilers: Dict[str, Any] = {}
        self._profiling_results: Dict[str, PerformanceProfile] = {}

        # Performance tracking
        self._qt_operation_tracker = QtOperationTracker()
        self._memory_tracker = MemoryTracker()

        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}

        logger.info("Qt performance profiler initialized")

    def start_profiling(self, config: ProfilerConfiguration, profiler_id: str = None) -> str:
        """Start performance profiling with given configuration."""
        if profiler_id is None:
            profiler_id = f"profile_{int(time.perf_counter() * 1000)}"

        with QMutexLocker(self._mutex):
            if profiler_id in self._active_profilers:
                raise ValueError(f"Profiler {profiler_id} is already active")

            # Create profiler based on type
            profiler = self._create_profiler(config, profiler_id)
            self._active_profilers[profiler_id] = {
                "profiler": profiler,
                "config": config,
                "start_time": time.perf_counter()
            }

        # Start the profiler
        profiler.start()

        self.profiling_started.emit(profiler_id)
        logger.info(f"Started profiling: {profiler_id} ({config.profiler_type.value})")

        return profiler_id

    def stop_profiling(self, profiler_id: str) -> PerformanceProfile:
        """Stop profiling and return results."""
        with QMutexLocker(self._mutex):
            if profiler_id not in self._active_profilers:
                raise ValueError(f"Profiler {profiler_id} is not active")

            profiler_info = self._active_profilers[profiler_id]
            profiler = profiler_info["profiler"]
            config = profiler_info["config"]
            start_time = profiler_info["start_time"]

        # Stop the profiler
        results = profiler.stop()

        # Create performance profile
        duration = time.perf_counter() - start_time
        profile = self._create_performance_profile(results, config, duration)

        # Store results
        with QMutexLocker(self._mutex):
            del self._active_profilers[profiler_id]
            self._profiling_results[profiler_id] = profile

        self.profiling_completed.emit(profiler_id, profile)
        logger.info(f"Completed profiling: {profiler_id}")

        return profile

    def _create_profiler(self, config: ProfilerConfiguration, profiler_id: str) -> 'BaseProfiler':
        """Create profiler based on configuration."""
        if config.profiler_type == ProfilerType.CPROFILE:
            return CProfilerWrapper(config, profiler_id)
        elif config.profiler_type == ProfilerType.MEMORY:
            return MemoryProfiler(config, profiler_id)
        elif config.profiler_type == ProfilerType.QT_OVERHEAD:
            return QtOverheadProfiler(config, profiler_id)
        elif config.profiler_type == ProfilerType.FILTERING:
            return FilteringProfiler(config, profiler_id)
        else:
            raise ValueError(f"Unknown profiler type: {config.profiler_type}")

    def _create_performance_profile(self, results: Dict[str, Any],
                                  config: ProfilerConfiguration, duration: float) -> PerformanceProfile:
        """Create performance profile from profiler results."""
        profile = PerformanceProfile(
            timestamp=time.perf_counter(),
            profiler_type=config.profiler_type,
            scope=config.scope,
            duration_seconds=duration
        )

        # Populate profile with results
        if "cpu_stats" in results:
            cpu_stats = results["cpu_stats"]
            profile.total_calls = cpu_stats.get("total_calls", 0)
            profile.total_time = cpu_stats.get("total_time", 0.0)
            profile.cumulative_time = cpu_stats.get("cumulative_time", 0.0)
            profile.function_stats = cpu_stats.get("function_stats", {})

        if "memory_stats" in results:
            memory_stats = results["memory_stats"]
            profile.peak_memory_mb = memory_stats.get("peak_memory_mb", 0.0)
            profile.memory_growth_mb = memory_stats.get("memory_growth_mb", 0.0)
            profile.memory_allocations = memory_stats.get("allocations", 0)
            profile.gc_collections = memory_stats.get("gc_collections", 0)

        if "qt_stats" in results:
            qt_stats = results["qt_stats"]
            profile.qt_operations_count = qt_stats.get("operations_count", 0)
            profile.qt_overhead_percent = qt_stats.get("overhead_percent", 0.0)
            profile.qt_warning_processing_time = qt_stats.get("warning_processing_time", 0.0)
            profile.qt_signal_emissions = qt_stats.get("signal_emissions", 0)

        # Analyze for optimization opportunities
        profile.hotspots = self._identify_hotspots(profile)
        profile.optimization_suggestions = self._generate_optimization_suggestions(profile)

        return profile

    def _identify_hotspots(self, profile: PerformanceProfile) -> List[Dict[str, Any]]:
        """Identify performance hotspots."""
        hotspots = []

        # Analyze function statistics for hotspots
        for func_name, stats in profile.function_stats.items():
            cumulative_time = stats.get("cumulative_time", 0.0)
            calls = stats.get("calls", 0)

            # Identify functions taking significant time
            if cumulative_time > profile.total_time * 0.05:  # More than 5% of total time
                hotspots.append({
                    "type": "cpu_hotspot",
                    "function": func_name,
                    "cumulative_time": cumulative_time,
                    "calls": calls,
                    "avg_time_per_call": cumulative_time / calls if calls > 0 else 0,
                    "impact": "high" if cumulative_time > profile.total_time * 0.1 else "medium"
                })

        # Memory hotspots
        if profile.memory_growth_mb > 50:  # More than 50MB growth
            hotspots.append({
                "type": "memory_hotspot",
                "memory_growth_mb": profile.memory_growth_mb,
                "impact": "high" if profile.memory_growth_mb > 100 else "medium"
            })

        # Qt overhead hotspots
        if profile.qt_overhead_percent > 10:  # More than 10% overhead
            hotspots.append({
                "type": "qt_overhead_hotspot",
                "overhead_percent": profile.qt_overhead_percent,
                "impact": "high" if profile.qt_overhead_percent > 20 else "medium"
            })

        return hotspots

    def _generate_optimization_suggestions(self, profile: PerformanceProfile) -> List[str]:
        """Generate optimization suggestions based on profile."""
        suggestions = []

        # CPU optimization suggestions
        cpu_hotspots = [h for h in profile.hotspots if h["type"] == "cpu_hotspot"]
        if cpu_hotspots:
            suggestions.append("Consider optimizing CPU-intensive functions identified in hotspots")
            suggestions.append("Evaluate caching opportunities for frequently called functions")

        # Memory optimization suggestions
        if profile.memory_growth_mb > 50:
            suggestions.append("High memory growth detected - review object lifecycle management")
            suggestions.append("Consider implementing object pooling for frequently created objects")

        # Qt-specific suggestions
        if profile.qt_overhead_percent > 10:
            suggestions.append("Qt overhead is significant - review Qt operation efficiency")
            suggestions.append("Consider batching Qt operations to reduce overhead")

        if profile.qt_warning_processing_time > 1.0:
            suggestions.append("Qt warning processing is slow - optimize filtering algorithms")

        # General suggestions
        if profile.gc_collections > 100:
            suggestions.append("High garbage collection frequency - review object creation patterns")

        return suggestions

    def analyze_qt_overhead(self, test_duration: float = 10.0) -> Dict[str, Any]:
        """Analyze Qt compliance system overhead."""
        logger.info(f"Analyzing Qt overhead for {test_duration}s")

        # Get baseline measurements
        baseline_profile = self._measure_baseline_performance(test_duration / 2)

        # Get Qt compliance measurements
        qt_profile = self._measure_qt_compliance_performance(test_duration / 2)

        # Calculate overhead
        overhead_analysis = {
            "baseline_cpu_time": baseline_profile.get("cpu_time", 0.0),
            "qt_compliance_cpu_time": qt_profile.get("cpu_time", 0.0),
            "cpu_overhead_percent": 0.0,

            "baseline_memory_mb": baseline_profile.get("memory_mb", 0.0),
            "qt_compliance_memory_mb": qt_profile.get("memory_mb", 0.0),
            "memory_overhead_mb": 0.0,

            "baseline_operations_per_second": baseline_profile.get("operations_per_second", 0.0),
            "qt_compliance_operations_per_second": qt_profile.get("operations_per_second", 0.0),
            "throughput_impact_percent": 0.0
        }

        # Calculate overhead percentages
        if baseline_profile.get("cpu_time", 0) > 0:
            overhead_analysis["cpu_overhead_percent"] = (
                (qt_profile.get("cpu_time", 0) - baseline_profile.get("cpu_time", 0)) /
                baseline_profile.get("cpu_time", 0)
            ) * 100

        overhead_analysis["memory_overhead_mb"] = (
            qt_profile.get("memory_mb", 0) - baseline_profile.get("memory_mb", 0)
        )

        if baseline_profile.get("operations_per_second", 0) > 0:
            overhead_analysis["throughput_impact_percent"] = (
                (baseline_profile.get("operations_per_second", 0) - qt_profile.get("operations_per_second", 0)) /
                baseline_profile.get("operations_per_second", 0)
            ) * 100

        logger.info(f"Qt overhead analysis completed: CPU {overhead_analysis['cpu_overhead_percent']:.1f}%, "
                   f"Memory {overhead_analysis['memory_overhead_mb']:.1f}MB")

        return overhead_analysis

    def _measure_baseline_performance(self, duration: float) -> Dict[str, Any]:
        """Measure baseline performance without Qt compliance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        operations_count = 0

        # Perform baseline operations
        while time.perf_counter() - start_time < duration:
            # Simulate typical Qt operations without compliance overhead
            widget = QWidget()
            widget.setObjectName(f"baseline_widget_{operations_count}")
            widget.deleteLater()

            operations_count += 1

            if operations_count % 100 == 0:
                QApplication.processEvents()

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        return {
            "cpu_time": end_time - start_time,
            "memory_mb": end_memory,
            "operations_per_second": operations_count / (end_time - start_time)
        }

    def _measure_qt_compliance_performance(self, duration: float) -> Dict[str, Any]:
        """Measure performance with Qt compliance enabled."""
        # Initialize Qt compliance components
        qt_detector = get_qt_error_detector()
        thread_manager = get_qt_compliant_thread_manager()

        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        operations_count = 0

        # Perform operations with Qt compliance
        while time.perf_counter() - start_time < duration:
            # Simulate typical Qt operations with compliance overhead
            widget = QWidget()
            widget.setObjectName(f"compliance_widget_{operations_count}")

            # Trigger Qt compliance monitoring
            error_summary = qt_detector.get_error_summary()

            widget.deleteLater()
            operations_count += 1

            if operations_count % 100 == 0:
                QApplication.processEvents()

        end_time = time.perf_counter()
        end_memory = self._get_memory_usage()

        return {
            "cpu_time": end_time - start_time,
            "memory_mb": end_memory,
            "operations_per_second": operations_count / (end_time - start_time)
        }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    def analyze_filtering_performance(self, warning_volumes: List[int]) -> List[FilteringPerformanceAnalysis]:
        """Analyze Qt warning filtering performance at different volumes."""
        results = []

        qt_detector = get_qt_error_detector()

        for volume in warning_volumes:
            logger.info(f"Analyzing filtering performance at {volume} warnings")

            # Generate test warnings
            test_warnings = self._generate_test_warnings(volume)

            # Measure filtering performance
            start_time = time.perf_counter()
            start_memory = self._get_memory_usage()

            processed_warnings = 0
            for warning in test_warnings:
                # Simulate warning processing
                qt_detector._process_qt_message(warning["msg_type"], warning["message"])
                processed_warnings += 1

            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            processing_time = (end_time - start_time) * 1000  # Convert to ms
            throughput = processed_warnings / (end_time - start_time) if end_time > start_time else 0
            memory_overhead = end_memory - start_memory

            analysis = FilteringPerformanceAnalysis(
                warning_volume=volume,
                processing_time_ms=processing_time,
                throughput_warnings_per_second=throughput,
                memory_overhead_mb=memory_overhead,
                cpu_overhead_percent=0.0,  # Would be calculated in real implementation
                filter_efficiency=processed_warnings / volume if volume > 0 else 1.0
            )

            # Identify bottlenecks
            if processing_time > volume * 0.1:  # More than 0.1ms per warning
                analysis.bottlenecks.append("High per-warning processing time")

            if memory_overhead > volume * 0.001:  # More than 1KB per warning
                analysis.bottlenecks.append("High memory overhead per warning")

            results.append(analysis)

        return results

    def _generate_test_warnings(self, count: int) -> List[Dict[str, Any]]:
        """Generate test Qt warnings for performance testing."""
        from PySide6.QtCore import QtMsgType

        warnings = []
        warning_patterns = [
            "QObject::connect: Unique connection requires a pointer to member function",
            "QStyleHints::colorSchemeChanged() connection warning",
            "QTimer::start: Timers can only be used with threads started with QThread",
            "QWidget: Cannot create a QWidget when no GUI thread",
            "QPainter::begin: Paint device returned engine == 0"
        ]

        for i in range(count):
            warnings.append({
                "msg_type": QtMsgType.QtWarningMsg,
                "message": warning_patterns[i % len(warning_patterns)] + f" #{i}"
            })

        return warnings

    def benchmark_memory_patterns(self) -> Dict[str, Any]:
        """Benchmark memory usage patterns of Qt components."""
        logger.info("Benchmarking Qt memory patterns")

        # Test different Qt component creation patterns
        patterns = {
            "widget_creation": self._benchmark_widget_creation,
            "timer_creation": self._benchmark_timer_creation,
            "signal_connections": self._benchmark_signal_connections,
            "object_lifecycle": self._benchmark_object_lifecycle
        }

        results = {}

        for pattern_name, benchmark_func in patterns.items():
            logger.info(f"Benchmarking {pattern_name}")
            results[pattern_name] = benchmark_func()

        return results

    def _benchmark_widget_creation(self) -> Dict[str, Any]:
        """Benchmark widget creation memory patterns."""
        initial_memory = self._get_memory_usage()
        widgets = []

        # Create widgets
        for i in range(1000):
            widget = QWidget()
            widget.setObjectName(f"benchmark_widget_{i}")
            widgets.append(widget)

        peak_memory = self._get_memory_usage()

        # Cleanup widgets
        for widget in widgets:
            widget.deleteLater()

        widgets.clear()
        gc.collect()

        # Process deletion events
        for _ in range(20):
            QApplication.processEvents()
            time.sleep(0.001)

        final_memory = self._get_memory_usage()

        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": peak_memory - initial_memory,
            "memory_leaked_mb": final_memory - initial_memory,
            "cleanup_efficiency": (peak_memory - final_memory) / (peak_memory - initial_memory) if peak_memory > initial_memory else 1.0
        }

    def _benchmark_timer_creation(self) -> Dict[str, Any]:
        """Benchmark timer creation memory patterns."""
        initial_memory = self._get_memory_usage()
        timers = []

        # Create timers
        for i in range(500):
            timer = QTimer()
            timer.setObjectName(f"benchmark_timer_{i}")
            timers.append(timer)

        peak_memory = self._get_memory_usage()

        # Cleanup timers
        for timer in timers:
            timer.stop()
            timer.deleteLater()

        timers.clear()
        gc.collect()

        final_memory = self._get_memory_usage()

        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": peak_memory - initial_memory,
            "memory_leaked_mb": final_memory - initial_memory
        }

    def _benchmark_signal_connections(self) -> Dict[str, Any]:
        """Benchmark signal connection memory patterns."""
        initial_memory = self._get_memory_usage()

        # Create objects with signal connections
        objects = []
        for i in range(200):
            timer = QTimer()
            timer.timeout.connect(lambda: None)
            objects.append(timer)

        peak_memory = self._get_memory_usage()

        # Cleanup
        for obj in objects:
            obj.deleteLater()

        objects.clear()
        gc.collect()

        final_memory = self._get_memory_usage()

        return {
            "initial_memory_mb": initial_memory,
            "peak_memory_mb": peak_memory,
            "final_memory_mb": final_memory,
            "memory_growth_mb": peak_memory - initial_memory,
            "memory_leaked_mb": final_memory - initial_memory
        }

    def _benchmark_object_lifecycle(self) -> Dict[str, Any]:
        """Benchmark complete object lifecycle patterns."""
        results = []

        for cycle in range(5):
            initial_memory = self._get_memory_usage()

            # Create objects
            objects = []
            for i in range(100):
                widget = QWidget()
                timer = QTimer(widget)
                timer.timeout.connect(lambda: None)
                objects.append((widget, timer))

            peak_memory = self._get_memory_usage()

            # Use objects briefly
            for widget, timer in objects:
                timer.start(1000)
                timer.stop()

            # Cleanup
            for widget, timer in objects:
                widget.deleteLater()

            objects.clear()
            gc.collect()

            # Process events
            for _ in range(10):
                QApplication.processEvents()
                time.sleep(0.001)

            final_memory = self._get_memory_usage()

            results.append({
                "cycle": cycle,
                "memory_growth_mb": peak_memory - initial_memory,
                "memory_leaked_mb": final_memory - initial_memory
            })

        return {
            "cycles": results,
            "avg_memory_growth_mb": sum(r["memory_growth_mb"] for r in results) / len(results),
            "avg_memory_leaked_mb": sum(r["memory_leaked_mb"] for r in results) / len(results)
        }

    def generate_performance_report(self, profiler_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if profiler_ids is None:
            profiler_ids = list(self._profiling_results.keys())

        report = {
            "timestamp": time.perf_counter(),
            "profiles_analyzed": len(profiler_ids),
            "profiles": {},
            "summary": {},
            "recommendations": []
        }

        # Include profile data
        for profiler_id in profiler_ids:
            if profiler_id in self._profiling_results:
                profile = self._profiling_results[profiler_id]
                report["profiles"][profiler_id] = profile.to_dict()

        # Generate summary
        if self._profiling_results:
            profiles = [self._profiling_results[pid] for pid in profiler_ids if pid in self._profiling_results]

            report["summary"] = {
                "total_hotspots": sum(len(p.hotspots) for p in profiles),
                "avg_memory_growth_mb": sum(p.memory_growth_mb for p in profiles) / len(profiles),
                "avg_qt_overhead_percent": sum(p.qt_overhead_percent for p in profiles) / len(profiles),
                "critical_issues": sum(1 for p in profiles for h in p.hotspots if h.get("impact") == "high")
            }

            # Generate recommendations
            all_suggestions = []
            for profile in profiles:
                all_suggestions.extend(profile.optimization_suggestions)

            # Deduplicate and prioritize suggestions
            unique_suggestions = list(set(all_suggestions))
            report["recommendations"] = unique_suggestions

        return report


class BaseProfiler:
    """Base class for profilers."""

    def __init__(self, config: ProfilerConfiguration, profiler_id: str):
        self.config = config
        self.profiler_id = profiler_id

    def start(self):
        """Start profiling."""
        raise NotImplementedError

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return results."""
        raise NotImplementedError


class CProfilerWrapper(BaseProfiler):
    """Wrapper for Python's cProfile."""

    def __init__(self, config: ProfilerConfiguration, profiler_id: str):
        super().__init__(config, profiler_id)
        self.profiler = cProfile.Profile()

    def start(self):
        """Start cProfile profiling."""
        self.profiler.enable()

    def stop(self) -> Dict[str, Any]:
        """Stop profiling and return statistics."""
        self.profiler.disable()

        # Analyze statistics
        stats_stream = StringIO()
        stats = pstats.Stats(self.profiler, stream=stats_stream)
        stats.sort_stats('cumulative')

        # Extract function statistics
        function_stats = {}
        for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
            filename, line_number, function_name = func_info
            function_key = f"{filename}:{function_name}:{line_number}"

            function_stats[function_key] = {
                "calls": nc,
                "total_time": tt,
                "cumulative_time": ct,
                "filename": filename,
                "function_name": function_name,
                "line_number": line_number
            }

        return {
            "cpu_stats": {
                "total_calls": stats.total_calls,
                "total_time": sum(stat[2] for stat in stats.stats.values()),
                "cumulative_time": sum(stat[3] for stat in stats.stats.values()),
                "function_stats": function_stats
            }
        }


class MemoryProfiler(BaseProfiler):
    """Memory usage profiler."""

    def __init__(self, config: ProfilerConfiguration, profiler_id: str):
        super().__init__(config, profiler_id)
        self.initial_memory = 0.0
        self.peak_memory = 0.0
        self.gc_initial = 0

    def start(self):
        """Start memory profiling."""
        try:
            import psutil
            self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            self.initial_memory = 0.0

        self.gc_initial = len(gc.get_objects())

    def stop(self) -> Dict[str, Any]:
        """Stop memory profiling and return statistics."""
        try:
            import psutil
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        except ImportError:
            final_memory = 0.0

        gc_final = len(gc.get_objects())
        gc_collections = gc.collect()

        return {
            "memory_stats": {
                "initial_memory_mb": self.initial_memory,
                "final_memory_mb": final_memory,
                "peak_memory_mb": max(self.initial_memory, final_memory),
                "memory_growth_mb": final_memory - self.initial_memory,
                "allocations": gc_final - self.gc_initial,
                "gc_collections": gc_collections
            }
        }


class QtOverheadProfiler(BaseProfiler):
    """Qt-specific overhead profiler."""

    def __init__(self, config: ProfilerConfiguration, profiler_id: str):
        super().__init__(config, profiler_id)
        self.qt_operations = 0
        self.qt_start_time = 0.0

    def start(self):
        """Start Qt overhead profiling."""
        self.qt_start_time = time.perf_counter()

    def stop(self) -> Dict[str, Any]:
        """Stop Qt overhead profiling."""
        qt_total_time = time.perf_counter() - self.qt_start_time

        return {
            "qt_stats": {
                "operations_count": self.qt_operations,
                "total_time": qt_total_time,
                "overhead_percent": 0.0,  # Would be calculated based on comparison
                "warning_processing_time": 0.0,
                "signal_emissions": 0
            }
        }


class FilteringProfiler(BaseProfiler):
    """Qt warning filtering profiler."""

    def __init__(self, config: ProfilerConfiguration, profiler_id: str):
        super().__init__(config, profiler_id)
        self.warnings_processed = 0
        self.filtering_time = 0.0

    def start(self):
        """Start filtering profiling."""
        pass

    def stop(self) -> Dict[str, Any]:
        """Stop filtering profiling."""
        return {
            "filtering_stats": {
                "warnings_processed": self.warnings_processed,
                "filtering_time": self.filtering_time,
                "throughput": self.warnings_processed / self.filtering_time if self.filtering_time > 0 else 0
            }
        }


class QtOperationTracker:
    """Tracker for Qt operations."""

    def __init__(self):
        self.operations_count = 0
        self.signal_emissions = 0

    def track_operation(self, operation_type: str):
        """Track a Qt operation."""
        self.operations_count += 1

    def track_signal_emission(self):
        """Track a signal emission."""
        self.signal_emissions += 1


class MemoryTracker:
    """Memory usage tracker."""

    def __init__(self):
        self.snapshots = []

    def take_snapshot(self) -> float:
        """Take memory snapshot."""
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.snapshots.append(memory_mb)
            return memory_mb
        except ImportError:
            return 0.0


@contextmanager
def profile_qt_performance(config: ProfilerConfiguration = None):
    """Context manager for Qt performance profiling."""
    if config is None:
        config = ProfilerConfiguration()

    profiler = QtPerformanceProfiler()
    profiler_id = profiler.start_profiling(config)

    try:
        yield profiler
    finally:
        profile = profiler.stop_profiling(profiler_id)
        logger.info(f"Performance profiling completed: {len(profile.hotspots)} hotspots identified")


# Global instance
_qt_performance_profiler: Optional[QtPerformanceProfiler] = None


def get_qt_performance_profiler() -> QtPerformanceProfiler:
    """Get the global Qt performance profiler instance."""
    global _qt_performance_profiler

    if _qt_performance_profiler is None:
        _qt_performance_profiler = QtPerformanceProfiler()

    return _qt_performance_profiler


if __name__ == "__main__":
    # Example usage
    profiler = get_qt_performance_profiler()

    # Analyze Qt overhead
    overhead_analysis = profiler.analyze_qt_overhead(duration=5.0)
    print("Qt Overhead Analysis:")
    print(f"  CPU Overhead: {overhead_analysis['cpu_overhead_percent']:.1f}%")
    print(f"  Memory Overhead: {overhead_analysis['memory_overhead_mb']:.1f} MB")

    # Analyze filtering performance
    filtering_results = profiler.analyze_filtering_performance([100, 500, 1000])
    print("\nFiltering Performance Analysis:")
    for result in filtering_results:
        print(f"  {result.warning_volume} warnings: {result.throughput_warnings_per_second:.1f} warnings/sec")