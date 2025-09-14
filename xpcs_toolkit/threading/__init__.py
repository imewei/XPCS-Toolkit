"""
Threading and concurrency utilities for XPCS Toolkit.

This package provides asynchronous workers and utilities to enhance
GUI responsiveness by moving heavy operations to background threads.
"""

from .async_kernel import AsyncDataPreloader, AsyncViewerKernel
from .async_workers import (
    BaseAsyncWorker,
    ComputationWorker,
    DataLoadWorker,
    PlotWorker,
    WorkerManager,
    WorkerSignals,
)

# Enhanced workers with advanced features
from .async_workers_enhanced import BaseAsyncWorker as EnhancedBaseAsyncWorker
from .async_workers_enhanced import ComputationWorker as EnhancedComputationWorker
from .async_workers_enhanced import DataLoadWorker as EnhancedDataLoadWorker
from .async_workers_enhanced import (
    EnhancedThreadPool,
)
from .async_workers_enhanced import PlotWorker as EnhancedPlotWorker
from .async_workers_enhanced import WorkerManager as EnhancedWorkerManager
from .async_workers_enhanced import (
    WorkerPriority,
    WorkerResult,
)
from .async_workers_enhanced import WorkerSignals as EnhancedWorkerSignals
from .async_workers_enhanced import (
    WorkerState,
    WorkerStats,
)
from .cleanup_optimized import (
    BackgroundCleanupManager,
    CleanupPriority,
    CleanupTask,
    ObjectRegistry,
    SmartGarbageCollector,
    get_background_cleanup_manager,
    get_object_registry,
    get_smart_gc,
    initialize_optimized_cleanup,
    register_for_cleanup,
    schedule_type_cleanup,
    shutdown_optimized_cleanup,
    smart_gc_collect,
)
from .enhanced_thread_pool import (
    LoadBalanceStrategy,
    SmartThreadPool,
    ThreadPoolHealth,
    ThreadPoolManager,
    ThreadPoolMetrics,
    get_thread_pool_manager,
    initialize_enhanced_threading,
    shutdown_enhanced_threading,
)
from .gui_integration import (
    AsyncMethodMixin,
    ThreadingIntegrator,
    async_generate_g2_plot,
    async_generate_saxs_plot,
    async_load_xpcs_files,
    create_completion_callback,
    create_progress_callback,
    make_async,
    setup_enhanced_threading,
)
from .main_thread_optimizer import MainThreadOptimizer, OperationContext
from .optimized_workers import (
    OptimizedBaseWorker,
    OptimizedComputationWorker,
    OptimizedDataLoadWorker,
    OptimizedPlotWorker,
    WorkerPerformanceMetrics,
    create_optimized_worker,
    submit_optimized_worker,
)
from .performance_monitor import (
    PerformanceMonitor,
    PerformanceTrends,
    SystemSnapshot,
    get_performance_monitor,
    initialize_performance_monitoring,
    shutdown_performance_monitoring,
)
from .plot_workers import (
    G2PlotWorker,
    IntensityPlotWorker,
    QMapPlotWorker,
    SaxsPlotWorker,
    StabilityPlotWorker,
    TwotimePlotWorker,
)
from .progress_manager import ProgressDialog, ProgressIndicator, ProgressManager

# Optimized threading system with performance enhancements
from .signal_optimization import (
    ConnectionPool,
    SignalBatcher,
    SignalOptimizer,
    SignalPriority,
    WorkerAttributeCache,
    get_signal_optimizer,
    initialize_signal_optimization,
    shutdown_signal_optimization,
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
