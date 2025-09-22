"""
Threading and concurrency utilities for XPCS Toolkit.

This package provides essential asynchronous workers and utilities to enhance
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
from .plot_workers import (
    G2PlotWorker,
    IntensityPlotWorker,
    QMapPlotWorker,
    SaxsPlotWorker,
    StabilityPlotWorker,
    TwotimePlotWorker,
)
from .progress_manager import ProgressDialog, ProgressIndicator, ProgressManager
from .unified_threading import (
    TaskPriority,
    TaskType,
    UnifiedTask,
    UnifiedThreadingManager,
    get_unified_threading_manager,
    shutdown_unified_threading,
)

__all__ = [
    # Async components
    "AsyncDataPreloader",
    # GUI integration
    "AsyncMethodMixin",
    "AsyncViewerKernel",
    # Basic workers
    "BaseAsyncWorker",
    "ComputationWorker",
    "DataLoadWorker",
    # Plot workers
    "G2PlotWorker",
    "IntensityPlotWorker",
    "PlotWorker",
    # Progress management
    "ProgressDialog",
    "ProgressIndicator",
    "ProgressManager",
    "QMapPlotWorker",
    "SaxsPlotWorker",
    "StabilityPlotWorker",
    # Unified threading
    "TaskPriority",
    "TaskType",
    "ThreadingIntegrator",
    "TwotimePlotWorker",
    "UnifiedTask",
    "UnifiedThreadingManager",
    "WorkerManager",
    "WorkerSignals",
    "async_generate_g2_plot",
    "async_generate_saxs_plot",
    "async_load_xpcs_files",
    "create_completion_callback",
    "create_progress_callback",
    "get_unified_threading_manager",
    "make_async",
    "setup_enhanced_threading",
    "shutdown_unified_threading",
]
