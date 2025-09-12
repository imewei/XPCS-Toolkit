"""
Threading and concurrency utilities for XPCS Toolkit.

This package provides asynchronous workers and utilities to enhance
GUI responsiveness by moving heavy operations to background threads.
"""

from .async_workers import (
    BaseAsyncWorker,
    PlotWorker,
    DataLoadWorker,
    ComputationWorker,
    WorkerManager,
    WorkerSignals,
)

# Enhanced workers with advanced features
from .async_workers_enhanced import (
    BaseAsyncWorker as EnhancedBaseAsyncWorker,
    PlotWorker as EnhancedPlotWorker,
    DataLoadWorker as EnhancedDataLoadWorker,
    ComputationWorker as EnhancedComputationWorker,
    WorkerManager as EnhancedWorkerManager,
    EnhancedThreadPool,
    WorkerPriority,
    WorkerState,
    WorkerResult,
    WorkerStats,
    WorkerSignals as EnhancedWorkerSignals,
)

# Optimized threading system with performance enhancements
from .signal_optimization import (
    SignalOptimizer,
    SignalBatcher,
    ConnectionPool,
    WorkerAttributeCache,
    SignalPriority,
    get_signal_optimizer,
    initialize_signal_optimization,
    shutdown_signal_optimization,
)

from .enhanced_thread_pool import (
    SmartThreadPool,
    ThreadPoolManager,
    ThreadPoolMetrics,
    ThreadPoolHealth,
    LoadBalanceStrategy,
    get_thread_pool_manager,
    initialize_enhanced_threading,
    shutdown_enhanced_threading,
)

from .optimized_workers import (
    OptimizedBaseWorker,
    OptimizedPlotWorker,
    OptimizedDataLoadWorker,
    OptimizedComputationWorker,
    WorkerPerformanceMetrics,
    create_optimized_worker,
    submit_optimized_worker,
)

from .performance_monitor import (
    PerformanceMonitor,
    SystemSnapshot,
    PerformanceTrends,
    get_performance_monitor,
    initialize_performance_monitoring,
    shutdown_performance_monitoring,
)

from .plot_workers import (
    SaxsPlotWorker,
    G2PlotWorker,
    TwotimePlotWorker,
    IntensityPlotWorker,
    StabilityPlotWorker,
    QMapPlotWorker,
)

from .async_kernel import AsyncViewerKernel, AsyncDataPreloader

from .progress_manager import ProgressManager, ProgressDialog, ProgressIndicator

from .main_thread_optimizer import MainThreadOptimizer, OperationContext

from .gui_integration import (
    ThreadingIntegrator,
    setup_enhanced_threading,
    make_async,
    AsyncMethodMixin,
    create_progress_callback,
    create_completion_callback,
    async_load_xpcs_files,
    async_generate_saxs_plot,
    async_generate_g2_plot,
)

from .cleanup_optimized import (
    ObjectRegistry,
    BackgroundCleanupManager,
    SmartGarbageCollector,
    CleanupPriority,
    CleanupTask,
    get_object_registry,
    get_background_cleanup_manager,
    get_smart_gc,
    register_for_cleanup,
    schedule_type_cleanup,
    smart_gc_collect,
    initialize_optimized_cleanup,
    shutdown_optimized_cleanup,
)

__all__ = [
    # Original workers
    "BaseAsyncWorker",
    "PlotWorker",
    "DataLoadWorker",
    "ComputationWorker",
    "WorkerManager",
    "WorkerSignals",
    # Enhanced workers
    "EnhancedBaseAsyncWorker",
    "EnhancedPlotWorker",
    "EnhancedDataLoadWorker",
    "EnhancedComputationWorker",
    "EnhancedWorkerManager",
    "EnhancedThreadPool",
    "WorkerPriority",
    "WorkerState",
    "WorkerResult",
    "WorkerStats",
    "EnhancedWorkerSignals",
    # Optimized threading system
    "SignalOptimizer",
    "SignalBatcher",
    "ConnectionPool",
    "WorkerAttributeCache",
    "SignalPriority",
    "get_signal_optimizer",
    "initialize_signal_optimization",
    "shutdown_signal_optimization",
    # Enhanced thread pools
    "SmartThreadPool",
    "ThreadPoolManager",
    "ThreadPoolMetrics",
    "ThreadPoolHealth",
    "LoadBalanceStrategy",
    "get_thread_pool_manager",
    "initialize_enhanced_threading",
    "shutdown_enhanced_threading",
    # Optimized workers
    "OptimizedBaseWorker",
    "OptimizedPlotWorker",
    "OptimizedDataLoadWorker",
    "OptimizedComputationWorker",
    "WorkerPerformanceMetrics",
    "create_optimized_worker",
    "submit_optimized_worker",
    # Performance monitoring
    "PerformanceMonitor",
    "SystemSnapshot",
    "PerformanceTrends",
    "get_performance_monitor",
    "initialize_performance_monitoring",
    "shutdown_performance_monitoring",
    # Plot workers
    "SaxsPlotWorker",
    "G2PlotWorker",
    "TwotimePlotWorker",
    "IntensityPlotWorker",
    "StabilityPlotWorker",
    "QMapPlotWorker",
    # Async components
    "AsyncViewerKernel",
    "AsyncDataPreloader",
    # Progress management
    "ProgressManager",
    "ProgressDialog",
    "ProgressIndicator",
    # Main thread optimization
    "MainThreadOptimizer",
    "OperationContext",
    # GUI integration
    "ThreadingIntegrator",
    "setup_enhanced_threading",
    "make_async",
    "AsyncMethodMixin",
    "create_progress_callback",
    "create_completion_callback",
    "async_load_xpcs_files",
    "async_generate_saxs_plot",
    "async_generate_g2_plot",
    # Optimized cleanup system
    "ObjectRegistry",
    "BackgroundCleanupManager",
    "SmartGarbageCollector",
    "CleanupPriority",
    "CleanupTask",
    "get_object_registry",
    "get_background_cleanup_manager",
    "get_smart_gc",
    "register_for_cleanup",
    "schedule_type_cleanup",
    "smart_gc_collect",
    "initialize_optimized_cleanup",
    "shutdown_optimized_cleanup",
]
