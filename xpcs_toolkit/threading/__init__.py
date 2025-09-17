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
    WorkerPriority,
    WorkerResult,
    WorkerState,
    WorkerStats,
)
from .async_workers_enhanced import PlotWorker as EnhancedPlotWorker
from .async_workers_enhanced import WorkerManager as EnhancedWorkerManager
from .async_workers_enhanced import WorkerSignals as EnhancedWorkerSignals
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

# Qt-compliant thread management and safety
from .qt_compliant_thread_manager import (
    QtCompliantThreadManager,
    get_qt_compliant_thread_manager,
)
from .enhanced_worker_safety import SafeWorkerBase

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
from .thread_pool_integration_validator import (
    ThreadPoolIntegrationValidator,
    get_thread_pool_validator,
)

__all__ = [
    "AsyncDataPreloader",
    "AsyncMethodMixin",
    # Async components
    "AsyncViewerKernel",
    "BackgroundCleanupManager",
    # Original workers
    "BaseAsyncWorker",
    "CleanupPriority",
    "CleanupTask",
    "ComputationWorker",
    "ConnectionPool",
    "DataLoadWorker",
    # Enhanced workers
    "EnhancedBaseAsyncWorker",
    "EnhancedComputationWorker",
    "EnhancedDataLoadWorker",
    "EnhancedPlotWorker",
    "EnhancedThreadPool",
    "EnhancedWorkerManager",
    "EnhancedWorkerSignals",
    "G2PlotWorker",
    "IntensityPlotWorker",
    "LoadBalanceStrategy",
    # Main thread optimization
    "MainThreadOptimizer",
    # Optimized cleanup system
    "ObjectRegistry",
    "OperationContext",
    # Optimized workers
    "OptimizedBaseWorker",
    "OptimizedComputationWorker",
    "OptimizedDataLoadWorker",
    "OptimizedPlotWorker",
    # Performance monitoring
    "PerformanceMonitor",
    "PerformanceTrends",
    "PlotWorker",
    "ProgressDialog",
    "ProgressIndicator",
    # Progress management
    "ProgressManager",
    "QMapPlotWorker",
    # Qt-compliant thread management
    "QtCompliantThreadManager",
    # Plot workers
    "SafeWorkerBase",
    "SaxsPlotWorker",
    "SignalBatcher",
    # Optimized threading system
    "SignalOptimizer",
    "SignalPriority",
    "SmartGarbageCollector",
    # Enhanced thread pools
    "SmartThreadPool",
    "StabilityPlotWorker",
    "SystemSnapshot",
    "ThreadPoolHealth",
    "ThreadPoolIntegrationValidator",
    "ThreadPoolManager",
    "ThreadPoolMetrics",
    # GUI integration
    "ThreadingIntegrator",
    "TwotimePlotWorker",
    "WorkerAttributeCache",
    "WorkerManager",
    "WorkerPerformanceMetrics",
    "WorkerPriority",
    "WorkerResult",
    "WorkerSignals",
    "WorkerState",
    "WorkerStats",
    "async_generate_g2_plot",
    "async_generate_saxs_plot",
    "async_load_xpcs_files",
    "create_completion_callback",
    "create_optimized_worker",
    "create_progress_callback",
    "get_background_cleanup_manager",
    "get_object_registry",
    "get_performance_monitor",
    "get_qt_compliant_thread_manager",
    "get_signal_optimizer",
    "get_smart_gc",
    "get_thread_pool_manager",
    "get_thread_pool_validator",
    "initialize_enhanced_threading",
    "initialize_optimized_cleanup",
    "initialize_performance_monitoring",
    "initialize_signal_optimization",
    "make_async",
    "register_for_cleanup",
    "schedule_type_cleanup",
    "setup_enhanced_threading",
    "shutdown_enhanced_threading",
    "shutdown_optimized_cleanup",
    "shutdown_performance_monitoring",
    "shutdown_signal_optimization",
    "smart_gc_collect",
    "submit_optimized_worker",
]
