"""
XPCS Toolkit Performance Package.

This package provides comprehensive performance optimization and analysis
capabilities for the Qt compliance system.
"""

from .qt_performance_profiler import (
    QtPerformanceProfiler,
    ProfilerConfiguration,
    ProfilerType,
    ProfilingScope,
    PerformanceProfile,
    FilteringPerformanceAnalysis,
    get_qt_performance_profiler,
    profile_qt_performance
)

from .qt_optimization_engine import (
    QtOptimizationEngine,
    OptimizationConfiguration,
    OptimizationType,
    OptimizationLevel,
    OptimizationResult,
    SmartCache,
    OptimizedQtMessageFilter,
    ObjectPool,
    MemoryOptimizer,
    ThreadingOptimizer,
    get_qt_optimization_engine,
    cached_qt_operation,
    optimized_qt_operations
)

from .qt_benchmark_suite import (
    QtComplianceBenchmarkSuite,
    BenchmarkConfiguration,
    BenchmarkType,
    BenchmarkScope,
    BenchmarkMetrics,
    BenchmarkResult,
    run_qt_compliance_benchmarks
)

from .qt_scalability_tester import (
    QtScalabilityTester,
    ScalabilityConfiguration,
    ScalabilityDimension,
    LoadPattern,
    ScalabilityMetrics,
    ScalabilityTestResult,
    get_qt_scalability_tester,
    run_qt_scalability_tests
)

__all__ = [
    # Performance profiling
    "QtPerformanceProfiler",
    "ProfilerConfiguration",
    "ProfilerType",
    "ProfilingScope",
    "PerformanceProfile",
    "FilteringPerformanceAnalysis",
    "get_qt_performance_profiler",
    "profile_qt_performance",

    # Optimization engine
    "QtOptimizationEngine",
    "OptimizationConfiguration",
    "OptimizationType",
    "OptimizationLevel",
    "OptimizationResult",
    "SmartCache",
    "OptimizedQtMessageFilter",
    "ObjectPool",
    "MemoryOptimizer",
    "ThreadingOptimizer",
    "get_qt_optimization_engine",
    "cached_qt_operation",
    "optimized_qt_operations",

    # Benchmarking
    "QtComplianceBenchmarkSuite",
    "BenchmarkConfiguration",
    "BenchmarkType",
    "BenchmarkScope",
    "BenchmarkMetrics",
    "BenchmarkResult",
    "run_qt_compliance_benchmarks",

    # Scalability testing
    "QtScalabilityTester",
    "ScalabilityConfiguration",
    "ScalabilityDimension",
    "LoadPattern",
    "ScalabilityMetrics",
    "ScalabilityTestResult",
    "get_qt_scalability_tester",
    "run_qt_scalability_tests"
]